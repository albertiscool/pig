import torch  # Import first on Windows to avoid WinError 4551
import math
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeMTokenPool(nn.Module):
    """
    Generalized Mean (GeM) pooling over spatial / patch tokens (OpenAnimals 2024 / Radenovic et al.).
    Smoothly bridges average pooling (p=1) and max pooling (p->inf) so unoccluded facial cues
    dominate over fence-bar tokens.
    """

    def __init__(self, p=3.0, eps=1e-6, p_min=1.0, p_max=6.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

    def forward(self, tokens):
        # tokens: (B, N, C)
        p = self.p.clamp(min=self.p_min, max=self.p_max)
        # Shift by min per sample-channel so fractional power works cleanly on signed features
        t_min = tokens.amin(dim=1, keepdim=True)
        shifted = (tokens - t_min).clamp(min=self.eps)
        pooled = shifted.pow(p).mean(dim=1).pow(1.0 / p) + t_min.squeeze(1)
        return pooled


class SubCenterAdaArcFace(nn.Module):
    """
    Sub-center ArcFace (K=3, Deng et al. ECCV 2020 / PFOSR 2025) combined with
    AdaFace quality-adaptive margin (Kim et al. CVPR 2022).
    - K sub-centers per class accommodate clean, occluded, and multi-pose views.
    - Batch-normalized pre-L2 feature norm modulates angular margin so heavily occluded
      or blurry samples receive gentle margins without destabilizing the sphere.
    - At inference (labels=None): returns max_k cosine * scale (no margin).
    """

    def __init__(self, in_features, out_features, k=3, s=16.0, m=0.30, h=0.33, t_alpha=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = int(k)
        self.s = float(s)
        self.m = float(m)
        self.h = float(h)
        self.t_alpha = float(t_alpha)

        self.weight = nn.Parameter(torch.FloatTensor(out_features * self.k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 5.0)

    def forward(self, x, labels=None, raw_norm=None):
        x_norm = F.normalize(x.float(), p=2, dim=1)
        w_norm = F.normalize(self.weight.float(), p=2, dim=1)
        cos_all = F.linear(x_norm, w_norm).view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cos_all, dim=2)  # (B, C)

        if labels is None:
            return cosine * self.s

        with torch.amp.autocast("cuda", enabled=False):
            cosine_f = cosine.float().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            if raw_norm is None:
                raw_norm = torch.norm(x.float(), p=2, dim=1, keepdim=True)
            else:
                raw_norm = raw_norm.float().view(-1, 1)

            with torch.no_grad():
                safe_norm = raw_norm.clamp(min=0.001, max=100.0)
                cur_mean = safe_norm.mean()
                cur_std = safe_norm.std().clamp(min=1e-4)
                self.batch_mean.mul_(1.0 - self.t_alpha).add_(cur_mean * self.t_alpha)
                self.batch_std.mul_(1.0 - self.t_alpha).add_(cur_std * self.t_alpha)
                margin_scaler = ((safe_norm - self.batch_mean) / (self.batch_std + 1e-4)).clamp(-1.0, 1.0)
                # AdaFace adaptive angular & additive margin
                g_angle = -self.m * margin_scaler * self.h  # (B, 1)
                g_add = self.m + self.m * margin_scaler * self.h  # (B, 1)
                m_eff = (self.m + g_angle).clamp(0.05, 0.50)

            cos_m = torch.cos(m_eff)
            sin_m = torch.sin(m_eff)
            sine = torch.sqrt((1.0 - cosine_f.pow(2)).clamp(min=1e-6))
            phi = cosine_f * cos_m - sine * sin_m - g_add

            one_hot = torch.zeros_like(cosine_f)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine_f)
            return output * self.s


class FoundationPigReID(nn.Module):
    """
    Foundation-backbone Re-ID with GeM spatial/patch pooling, 512-D projection + BNNeck,
    and Sub-center AdaArcFace (K=3). Supports:
      - 'megadescriptor_t': hf-hub:BVRA/MegaDescriptor-T-224 (Swin-Tiny, wildlife ReID pretrained)
      - 'dinov2_s14': timm vit_small_patch14_dinov2.lvd142m @ 224x224
    """

    def __init__(self, backbone_type="dinov2_s14", num_classes=17, emb_dim=512, k=3, s=16.0, m=0.30, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)
        self.k = int(k)
        self.scale = float(s)

        if backbone_type == "dinov2_s14":
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                img_size=224,
                num_classes=0,
            )
            feat_dim = self.backbone.num_features  # 384
            # Combine CLS token + GeM-pooled patch tokens -> 2*384
            in_proj_dim = feat_dim * 2
        elif backbone_type == "megadescriptor_t":
            self.backbone = timm.create_model(
                "hf-hub:BVRA/MegaDescriptor-T-224",
                pretrained=pretrained,
                num_classes=0,
            )
            feat_dim = self.backbone.num_features  # 768
            in_proj_dim = feat_dim * 2
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.gem_pool = GeMTokenPool(p=3.0)
        self.proj = nn.Linear(in_proj_dim, emb_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(emb_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.head = SubCenterAdaArcFace(emb_dim, num_classes, k=k, s=s, m=m)

    def configure_partial_freeze(self):
        """Freeze early stages/blocks and unfreeze only the top semantic stage + pooling/head."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if self.backbone_type == "dinov2_s14":
            # Unfreeze blocks 9, 10, 11 + final norm
            for name, p in self.backbone.named_parameters():
                if any(name.startswith(prefix) for prefix in ("blocks.9.", "blocks.10.", "blocks.11.", "norm.")):
                    p.requires_grad_(True)
        elif self.backbone_type == "megadescriptor_t":
            # Unfreeze Swin final stage (layers.3) + final norm
            for name, p in self.backbone.named_parameters():
                if any(name.startswith(prefix) for prefix in ("layers.3.", "norm.")):
                    p.requires_grad_(True)

        for mod in (self.gem_pool, self.proj, self.bnneck, self.head):
            for p in mod.parameters():
                if p.requires_grad is not False:
                    p.requires_grad_(True)
        self.bnneck.bias.requires_grad_(False)

    def _extract_backbone_vec(self, x):
        feats = self.backbone.forward_features(x)
        if self.backbone_type == "dinov2_s14":
            # (B, 257, 384): token 0 is CLS, 1: are 16x16 patches
            cls_tok = feats[:, 0, :]
            gem_tok = self.gem_pool(feats[:, 1:, :])
            return torch.cat([cls_tok, gem_tok], dim=1)
        else:
            # Swin forward_features in timm returns (B, H, W, C) or (B, N, C)
            if feats.ndim == 4:
                b, h, w, c = feats.shape
                tokens = feats.reshape(b, h * w, c)
            else:
                tokens = feats
            avg_tok = tokens.mean(dim=1)
            gem_tok = self.gem_pool(tokens)
            return torch.cat([avg_tok, gem_tok], dim=1)

    def forward(self, x, labels=None):
        raw_vec = self._extract_backbone_vec(x)
        proj_vec = self.proj(raw_vec)
        raw_norm = torch.norm(proj_vec.float(), p=2, dim=1, keepdim=True)
        bn_vec = self.bnneck(proj_vec)

        logits_train = self.head(bn_vec, labels, raw_norm=raw_norm)
        cos_logits = self.head(bn_vec, None)
        emb = F.normalize(bn_vec.float(), p=2, dim=1)
        return logits_train, cos_logits, emb

    def extract_features(self, x):
        raw_vec = self._extract_backbone_vec(x)
        proj_vec = self.proj(raw_vec)
        bn_vec = self.bnneck(proj_vec)
        logits = self.head(bn_vec, None)
        emb = F.normalize(bn_vec.float(), p=2, dim=1)
        return logits, emb


class ZeroShotFoundationWrapper(nn.Module):
    """Wrapper for zero-shot evaluation of timm / MegaDescriptor backbones."""

    def __init__(self, model_name, num_classes=17, img_size=None):
        super().__init__()
        kwargs = {"pretrained": True, "num_classes": 0}
        if img_size is not None:
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.num_classes = num_classes

    def extract_features(self, x):
        feat = self.backbone(x)
        emb = F.normalize(feat.float(), p=2, dim=1)
        dummy_logits = torch.zeros((x.size(0), self.num_classes), device=x.device, dtype=torch.float32)
        return dummy_logits, emb
