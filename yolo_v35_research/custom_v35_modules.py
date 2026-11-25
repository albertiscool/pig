import torch  # Import first on Windows to avoid WinError 4551
import inspect
import torch.nn as nn
import ultralytics.nn.modules as umodules
import ultralytics.nn.tasks as utasks
import ultralytics.utils.loss as uloss
from ultralytics.utils.tal import bbox2dist


class BiFPN_Concat(nn.Module):
    """
    BiFPN Fast Normalized Weighted Feature Fusion (Tan et al., CVPR 2020 / YOLOv5-KCB 2024).
    Learns positive per-branch importance weights w_i / (sum(w) + eps) before concatenation,
    allowing occluded multi-scale features to adaptively emphasize cleaner scales.
    """

    def __init__(self, dimension=1, num_inputs=2, eps=1e-4):
        super().__init__()
        self.d = int(dimension)
        self.eps = float(eps)
        self.w = nn.Parameter(torch.ones(int(num_inputs), dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # x: list of tensors [t_0, t_1, ...]
        if len(x) != self.w.numel():
            # Fallback if dynamically called with different branch count
            return torch.cat(x, self.d)
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.eps)
        weighted = [weight[i] * x[i] for i in range(len(x))]
        return torch.cat(weighted, self.d)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (ICASSP 2023 / DESA-YOLO 2026)."""

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
    """Separated and Enhancement Attention Module (YOLO-FaceV2)."""

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
    """Residual-gated EMA + SEAM occlusion-enhancement block."""

    def __init__(self, c1: int, factor: int = 8, reduction: int = 16):
        super().__init__()
        self.ema = EMA(c1, factor=factor)
        self.seam = SEAM(c1, n=1, reduction=reduction)
        self.gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.seam(self.ema(x))
        return x + self.gate * (enhanced - x)


def shape_iou_components(box1: torch.Tensor, box2: torch.Tensor, scale: float = 0.5, eps: float = 1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    xc1 = 0.5 * (b1_x1 + b1_x2)
    yc1 = 0.5 * (b1_y1 + b1_y2)
    xc2 = 0.5 * (b2_x1 + b2_x2)
    yc2 = 0.5 * (b2_y1 + b2_y2)

    w2_s = w2.detach().pow(scale)
    h2_s = h2.detach().pow(scale)
    denom_s = w2_s + h2_s + eps
    ww = 2.0 * w2_s / denom_s
    hh = 2.0 * h2_s / denom_s

    distance_shape = hh * (xc1 - xc2).pow(2) / c2 + ww * (yc1 - yc2).pow(2) / c2
    omiga_w = hh * torch.abs(w1 - w2) / (torch.maximum(w1, w2) + eps)
    omiga_h = ww * torch.abs(h1 - h2) / (torch.maximum(h1, h2) + eps)
    shape_cost = (1.0 - torch.exp(-omiga_w)).pow(4) + (1.0 - torch.exp(-omiga_h)).pow(4)
    return iou, distance_shape, shape_cost


def _focaler_shape_wiou_v3_forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
):
    """
    Focaler-Shape-WIoUv3 Loss (Wise-IoU v3 + Focaler-IoU + Shape-IoU):
    - Focaler interval [d=0.0, u=0.95] concentrates gradient on hard partially occluded boxes.
    - Wise-IoU v3 dynamic non-monotonic focusing factor r = beta / (delta * alpha^(beta - delta))
      down-weights extreme outlier noise while boosting medium-hard occluded faces.
    - Shape-IoU weights center & aspect shape penalties by GT proportions.
    """
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou, distance_shape, shape_cost = shape_iou_components(pred_bboxes[fg_mask], target_bboxes[fg_mask], scale=0.5)

    # Focaler-IoU linear interval mapping
    d_f, u_f = 0.0, 0.95
    iou_focaler = ((iou - d_f) / (u_f - d_f)).clamp(0.0, 1.0)
    loss_focaler_shape = (1.0 - iou_focaler) + distance_shape + 0.5 * shape_cost

    # Wise-IoU v3 dynamic non-monotonic focusing
    if not hasattr(self, "_wiou_mean"):
        self._wiou_mean = torch.tensor(1.0, device=pred_dist.device, dtype=torch.float32)

    with torch.no_grad():
        l_iou_detach = (1.0 - iou).detach().float()
        if l_iou_detach.numel() > 0:
            batch_mean = l_iou_detach.mean().clamp(min=0.05, max=2.0)
            self._wiou_mean = 0.95 * self._wiou_mean.to(pred_dist.device) + 0.05 * batch_mean
        beta = (l_iou_detach / self._wiou_mean.clamp(min=0.05)).clamp(0.1, 6.0)
        alpha, delta = 1.9, 3.0
        r = (beta / (delta * torch.pow(alpha, beta - delta))).clamp(0.1, 3.0)

    loss_iou = (r.to(loss_focaler_shape.dtype) * loss_focaler_shape * weight).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0, device=pred_dist.device)

    return loss_iou, loss_dfl


def register_v35_modules():
    """Register BiFPN_Concat, EMA, SEAM, EMA_SEAM into Ultralytics and enable BiFPN_Concat in parse_model."""
    for mod_cls in (BiFPN_Concat, EMA, SEAM, EMA_SEAM):
        setattr(utasks, mod_cls.__name__, mod_cls)
        setattr(umodules, mod_cls.__name__, mod_cls)

    if not getattr(utasks.parse_model, "_is_v35_patched", False):
        src = inspect.getsource(utasks.parse_model)
        patched_src = src.replace(
            "elif m is Concat:",
            "elif m is Concat or m is BiFPN_Concat:",
        )
        ns = utasks.__dict__
        exec(compile(patched_src, "<v35_parse_model>", "exec"), ns)
        utasks.parse_model._is_v35_patched = True


def patch_ultralytics_focaler_shape_wiou():
    """Patch Ultralytics BboxLoss.forward at runtime with Focaler-Shape-WIoUv3."""
    if not getattr(uloss.BboxLoss, "_is_v35_wiou_patched", False):
        uloss.BboxLoss.forward = _focaler_shape_wiou_v3_forward
        uloss.BboxLoss._is_v35_wiou_patched = True
