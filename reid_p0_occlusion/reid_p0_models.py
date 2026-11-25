import torch  # Import first on Windows to avoid WinError 4551
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaselineECBModel(nn.Module):
    """Exact architecture of the existing v4 pig_ecb_best.pth checkpoint."""

    def __init__(self, num_classes=17):
        super().__init__()
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def forward(self, x):
        feat_cnn = self.cnn_model(x)
        return self.classifier_cnn(feat_cnn)

    def extract_features(self, x):
        feat_cnn = self.cnn_model(x)
        logits_cnn = self.classifier_cnn(feat_cnn)
        feat_norm = F.normalize(feat_cnn.float(), p=2, dim=1)
        return logits_cnn, feat_norm


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin (ArcFace) head.
    - When `labels` is provided (training): returns margin-penalized logits * scale.
    - When `labels` is None (eval/inference): returns cosine * scale (well-calibrated probs).
    """

    def __init__(self, in_features, out_features, s=16.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x.float(), p=2, dim=1), F.normalize(self.weight.float(), p=2, dim=1))
        if labels is None:
            return cosine * self.s

        with torch.amp.autocast("cuda", enabled=False):
            cosine_f = cosine.float().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            sine = torch.sqrt((1.0 - cosine_f.pow(2)).clamp(min=1e-6))
            phi = cosine_f * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine_f > self.th, phi, cosine_f - self.mm)

            one_hot = torch.zeros_like(cosine_f)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine_f)
            return output * self.s


class ECBHybridP0Model(nn.Module):
    """
    P0 Hybrid Re-ID Model (ViT-B/16 + ResNet-18):
    - `use_bnneck_arcface=True` (P0-Full): BNNeck (bias frozen at 0) + ArcFace head on both branches.
    - `use_bnneck_arcface=False` (P0-Ablation): standard Linear head (isolates AI-Fence aug gain).
    """

    def __init__(self, num_classes=17, use_bnneck_arcface=True, s=16.0, m=0.30):
        super().__init__()
        self.use_bnneck_arcface = use_bnneck_arcface
        self.scale = float(s)

        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()

        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()

        if use_bnneck_arcface:
            self.vit_bnneck = nn.BatchNorm1d(self.vit_head_dim)
            self.vit_bnneck.bias.requires_grad_(False)
            nn.init.constant_(self.vit_bnneck.bias, 0.0)
            self.classifier_vit = ArcMarginProduct(self.vit_head_dim, num_classes, s=s, m=m)

            self.cnn_bnneck = nn.BatchNorm1d(self.cnn_fc_dim)
            self.cnn_bnneck.bias.requires_grad_(False)
            nn.init.constant_(self.cnn_bnneck.bias, 0.0)
            self.classifier_cnn = ArcMarginProduct(self.cnn_fc_dim, num_classes, s=s, m=m)
        else:
            self.vit_bnneck = nn.Identity()
            self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)
            self.cnn_bnneck = nn.Identity()
            self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def load_backbone_from_baseline(self, baseline_ckpt_path):
        ckpt = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=False)
        src_sd = ckpt["model"]
        backbone_sd = {
            k: v for k, v in src_sd.items() if k.startswith("vit_model.") or k.startswith("cnn_model.")
        }
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        return len(backbone_sd), len(missing), len(unexpected)

    def forward(self, x, labels=None):
        raw_vit = self.vit_model(x)
        bn_vit = self.vit_bnneck(raw_vit)

        raw_cnn = self.cnn_model(x)
        bn_cnn = self.cnn_bnneck(raw_cnn)

        if self.use_bnneck_arcface:
            logits_vit_train = self.classifier_vit(bn_vit, labels)
            logits_cnn_train = self.classifier_cnn(bn_cnn, labels)
            cos_vit = self.classifier_vit(bn_vit, None)
            cos_cnn = self.classifier_cnn(bn_cnn, None)
        else:
            logits_vit_train = self.classifier_vit(bn_vit)
            logits_cnn_train = self.classifier_cnn(bn_cnn)
            cos_vit = logits_vit_train
            cos_cnn = logits_cnn_train

        emb_vit = F.normalize(bn_vit.float(), p=2, dim=1)
        emb_cnn = F.normalize(bn_cnn.float(), p=2, dim=1)
        return logits_vit_train, logits_cnn_train, cos_vit, cos_cnn, emb_vit, emb_cnn

    def extract_features(self, x):
        raw_cnn = self.cnn_model(x)
        bn_cnn = self.cnn_bnneck(raw_cnn)
        if self.use_bnneck_arcface:
            logits_cnn = self.classifier_cnn(bn_cnn, None)
        else:
            logits_cnn = self.classifier_cnn(bn_cnn)
        feat_norm = F.normalize(bn_cnn.float(), p=2, dim=1)
        return logits_cnn, feat_norm


class PigReIDCNNOnly(nn.Module):
    """Lightweight inference-only CNN branch (zero ViT weights, ~45 MB)."""

    def __init__(self, num_classes=17, use_bnneck_arcface=True, s=16.0):
        super().__init__()
        self.use_bnneck_arcface = use_bnneck_arcface
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()

        if use_bnneck_arcface:
            self.cnn_bnneck = nn.BatchNorm1d(self.cnn_fc_dim)
            self.cnn_bnneck.bias.requires_grad_(False)
            self.classifier_cnn = ArcMarginProduct(self.cnn_fc_dim, num_classes, s=s, m=0.0)
        else:
            self.cnn_bnneck = nn.Identity()
            self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def extract_features(self, x):
        raw_cnn = self.cnn_model(x)
        bn_cnn = self.cnn_bnneck(raw_cnn)
        if self.use_bnneck_arcface:
            logits_cnn = self.classifier_cnn(bn_cnn, None)
        else:
            logits_cnn = self.classifier_cnn(bn_cnn)
        feat_norm = F.normalize(bn_cnn.float(), p=2, dim=1)
        return logits_cnn, feat_norm


def export_cnn_only_checkpoint(full_model, classes, out_path, meta=None):
    """Extract only CNN branch weights into PigReIDCNNOnly checkpoint."""
    cnn_only = PigReIDCNNOnly(
        num_classes=len(classes),
        use_bnneck_arcface=full_model.use_bnneck_arcface,
        s=full_model.scale,
    )
    full_sd = full_model.state_dict()
    filtered_sd = {
        k: v.cpu()
        for k, v in full_sd.items()
        if k.startswith("cnn_model.") or k.startswith("cnn_bnneck.") or k.startswith("classifier_cnn.")
    }
    cnn_only.load_state_dict(filtered_sd, strict=True)
    torch.save(
        {
            "model": cnn_only.state_dict(),
            "classes": classes,
            "use_bnneck_arcface": full_model.use_bnneck_arcface,
            "scale": full_model.scale,
            "meta": meta or {},
        },
        out_path,
    )
    return cnn_only


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss on L2-normalized embeddings using cosine distance."""

    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings, labels):
        emb = embeddings.float()
        sim = torch.matmul(emb, emb.t()).clamp(-1.0, 1.0)
        dist = 1.0 - sim

        labels = labels.view(-1, 1)
        same_id = labels.eq(labels.t())
        diff_id = ~same_id

        eye = torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)
        pos_mask = same_id & (~eye)
        neg_mask = diff_id

        pos_dists = dist.masked_fill(~pos_mask, -1e4)
        hardest_pos, _ = pos_dists.max(dim=1)

        neg_dists = dist.masked_fill(~neg_mask, 1e4)
        hardest_neg, _ = neg_dists.min(dim=1)

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=emb.device, dtype=emb.dtype)

        losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return losses.mean()
