import torch
import torch.nn as nn

import ultralytics.nn.modules as umodules
import ultralytics.nn.tasks as utasks
import ultralytics.utils.loss as uloss
from ultralytics.utils.tal import bbox2dist


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (ICASSP 2023 / DESA-YOLO 2026).
    Uses horizontal + vertical coordinate pooling branches to aggregate pig facial features
    across vertical and horizontal pen-bar gaps.
    """
    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        groups = min(factor, channels)
        while groups > 1 and (channels % groups != 0):
            groups -= 1
        self.groups = groups
        c_g = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c_g, c_g)
        self.conv1x1 = nn.Conv2d(c_g, c_g, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c_g, c_g, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class SEAM(nn.Module):
    """
    Separated and Enhancement Attention Module (YOLO-FaceV2, Pattern Recognition / DESA-YOLO).
    Uses depthwise-separable residual convolutions + exponential channel weighting (exp(sigmoid))
    to amplify unoccluded facial cues and suppress fence-occluded background regions.
    """
    def __init__(self, c1: int, n: int = 1, reduction: int = 16):
        super().__init__()
        mid = max(4, c1 // reduction)
        self.DCovN = nn.Sequential(*[
            nn.Sequential(
                _Residual(nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1),
                    nn.GELU(),
                    nn.BatchNorm2d(c1),
                )),
                nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c1),
            )
            for _ in range(n)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


class EMA_SEAM(nn.Module):
    """
    Residual-gated EMA + SEAM occlusion-enhancement block for YOLO detection necks.
    Initialized with gate=0.25 so pretrained COCO backbone/neck features remain stable at epoch 0.
    """
    def __init__(self, c1: int, factor: int = 8, reduction: int = 16):
        super().__init__()
        self.ema = EMA(c1, factor=factor)
        self.seam = SEAM(c1, n=1, reduction=reduction)
        self.gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.seam(self.ema(x))
        return x + self.gate * (enhanced - x)


def shape_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    scale: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Shape-IoU (DGS-YOLO, MDPI 2026) bounding-box metric considering GT shape/scale weighting.
    Expects last dim = 4. Supports (x1, y1, x2, y2) when xywh=False or (x, y, w, h) when xywh=True.
    """
    if xywh:
        (x1, y1, w1, h1) = box1.chunk(4, -1)
        (x2, y2, w2, h2) = box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2.0, h1 / 2.0, w2 / 2.0, h2 / 2.0
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = (b1_x2 - b1_x1).clamp(min=eps)
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = (b2_x2 - b2_x1).clamp(min=eps)
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

    # Intersection & Union
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box diagonal^2
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # Center offsets
    xc1 = 0.5 * (b1_x1 + b1_x2)
    yc1 = 0.5 * (b1_y1 + b1_y2)
    xc2 = 0.5 * (b2_x1 + b2_x2)
    yc2 = 0.5 * (b2_y1 + b2_y2)

    # Shape-weighted aspect focus (using GT box dimensions w2, h2)
    w2_s = w2.detach().pow(scale)
    h2_s = h2.detach().pow(scale)
    denom_s = w2_s + h2_s + eps
    ww = 2.0 * w2_s / denom_s
    hh = 2.0 * h2_s / denom_s

    distance_shape = hh * (xc1 - xc2).pow(2) / c2 + ww * (yc1 - yc2).pow(2) / c2

    omiga_w = hh * torch.abs(w1 - w2) / (torch.maximum(w1, w2) + eps)
    omiga_h = ww * torch.abs(h1 - h2) / (torch.maximum(h1, h2) + eps)
    shape_cost = (1.0 - torch.exp(-omiga_w)).pow(4) + (1.0 - torch.exp(-omiga_h)).pow(4)

    return iou - distance_shape - 0.5 * shape_cost


def _shape_iou_bbox_loss_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
):
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = shape_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, scale=0.5)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0, device=pred_dist.device)

    return loss_iou, loss_dfl


def register_custom_modules():
    """Register EMA, SEAM, and EMA_SEAM into Ultralytics module namespaces at runtime."""
    for mod_cls in (EMA, SEAM, EMA_SEAM):
        setattr(utasks, mod_cls.__name__, mod_cls)
        setattr(umodules, mod_cls.__name__, mod_cls)


def patch_ultralytics_shape_iou():
    """Patch Ultralytics BboxLoss.forward at runtime to use Shape-IoU regression loss."""
    if not getattr(uloss.BboxLoss, "_is_shape_iou_patched", False):
        uloss.BboxLoss.forward = _shape_iou_bbox_loss_forward
        uloss.BboxLoss._is_shape_iou_patched = True
