import torch  # Import first on Windows to avoid WinError 4551
import os
import sys
import json
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
P0_CODE_DIR = os.path.join(BUNDLE_ROOT, "01_code", "reid_p0_occlusion")
for p in (SCRIPT_DIR, P0_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_fence_reid_aug import TRAIN_OVERLAY_NAMES, TEST_UNSEEN_OVERLAY_NAMES, load_overlays, RandomAIFenceOcclusion
from reid_p0_models import BatchHardTripletLoss
from train_and_eval_reid_p0 import (
    SHAOJIANG_ROOT,
    BASELINE_CKPT,
    ManifestImageDataset,
    EVAL_CACHED_TRANSFORM,
    build_chronological_split,
    build_deterministic_occluded_set,
    infer_dataset,
    compute_cosine_separation,
    list_images_sorted,
    natural_key,
)
from foundation_reid_models import GeMTokenPool, SubCenterAdaArcFace, ZeroShotFoundationWrapper
from run_reid_v2_experiments import evaluate_adapted_model, plot_v2_figures, OUT_ROOT, P0_RESULTS_JSON
import timm


class FrozenDINOv2AdapterReID(nn.Module):
    """
    Exp C: 100% Frozen DINOv2-ViT-S/14 backbone + Learnable Residual Token/Vector Adapter
    + GeM Pooling + BNNeck + Sub-center AdaArcFace (K=3).
    Preserves zero-shot open-set generalization while adapting to fence occlusions (~2.1M trainable params).
    """

    def __init__(self, num_classes=17, emb_dim=512, k=3, s=16.0, m=0.30, pretrained=True):
        super().__init__()
        self.backbone_type = "dinov2_s14_frozen_adapter"
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)
        self.k = int(k)
        self.scale = float(s)

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            img_size=224,
            num_classes=0,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        feat_dim = self.backbone.num_features  # 384
        self.gem_pool = GeMTokenPool(p=3.0)
        # Residual adapter over concatenated [CLS, GeM] (768-D)
        in_dim = feat_dim * 2
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 384),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(384, in_dim),
        )
        # Zero-init last linear so epoch 0 starts directly at pure foundation representation
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.adapter_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.proj = nn.Linear(in_dim, emb_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(emb_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.head = SubCenterAdaArcFace(emb_dim, num_classes, k=k, s=s, m=m)

    def _raw_vec(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        cls_tok = feats[:, 0, :]
        gem_tok = self.gem_pool(feats[:, 1:, :])
        cat_vec = torch.cat([cls_tok, gem_tok], dim=1)
        return cat_vec + self.adapter_gate * self.adapter(cat_vec)

    def forward(self, x, labels=None):
        adapted = self._raw_vec(x)
        proj_vec = self.proj(adapted)
        raw_norm = torch.norm(proj_vec.float(), p=2, dim=1, keepdim=True)
        bn_vec = self.bnneck(proj_vec)
        logits_tr = self.head(bn_vec, labels, raw_norm=raw_norm)
        cos_logits = self.head(bn_vec, None)
        emb = F.normalize(bn_vec.float(), p=2, dim=1)
        return logits_tr, cos_logits, emb

    def extract_features(self, x):
        adapted = self._raw_vec(x)
        proj_vec = self.proj(adapted)
        bn_vec = self.bnneck(proj_vec)
        logits = self.head(bn_vec, None)
        emb = F.normalize(bn_vec.float(), p=2, dim=1)
        return logits, emb


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    classes = list(base_ckpt["classes"])

    train_items, val_items, test_items, local_unknown_items, _ = build_chronological_split(classes)
    train_overlays = load_overlays(TRAIN_OVERLAY_NAMES)
    unseen_overlays = load_overlays(TEST_UNSEEN_OVERLAY_NAMES)
    val_occ_items, _ = build_deterministic_occluded_set(val_items, "val_occ", unseen_overlays, seed=202601)
    test_occ_items, _ = build_deterministic_occluded_set(test_items, "test_occ", unseen_overlays, seed=202602)
    sj_qry_dir = os.path.join(SHAOJIANG_ROOT, "clean_query")
    sj_qry_all = [(p, 0, pid) for pid in sorted(os.listdir(sj_qry_dir), key=natural_key) for p in list_images_sorted(os.path.join(sj_qry_dir, pid))]
    sj_occ_items, _ = build_deterministic_occluded_set(sj_qry_all, "sj_qry_occ", unseen_overlays, seed=202603)

    train_tfms = transforms.Compose([
        RandomAIFenceOcclusion(train_overlays, p=0.60, occ_range=(0.15, 0.40), seed=303),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    train_ds_cached = ManifestImageDataset(train_items, transform=train_tfms, cache_size=(256, 256))
    val_clean_ds = ManifestImageDataset(val_items, transform=EVAL_CACHED_TRANSFORM, cache_size=(224, 224))
    val_occ_ds = ManifestImageDataset(val_occ_items, transform=EVAL_CACHED_TRANSFORM, cache_size=(224, 224))
    train_loader = DataLoader(train_ds_cached, batch_size=48, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    torch.manual_seed(42)
    np.random.seed(42)
    model = FrozenDINOv2AdapterReID(num_classes=len(classes), pretrained=True).to(device)
    model.backbone.eval()

    teacher = ZeroShotFoundationWrapper("vit_small_patch14_dinov2.lvd142m", num_classes=len(classes), img_size=224).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Exp C Frozen DINOv2 + Adapter trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=2e-4)
    epochs = 18
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    triplet_fn = BatchHardTripletLoss(margin=0.35)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_c = os.path.join(OUT_ROOT, "pig_reid_v2_frozen_adapter_best.pth")
    best_score = -1.0
    hist_c = []
    for ep in range(1, epochs + 1):
        model.train()
        model.backbone.eval()
        ep_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, t_emb = teacher.extract_features(imgs)
                t_sim = torch.matmul(t_emb, t_emb.t())
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_tr, _, emb = model(imgs, lbls)
                loss = ce_fn(logits_tr, lbls) + 0.5 * triplet_fn(emb, lbls) + 0.6 * F.smooth_l1_loss(torch.matmul(emb, emb.t()), t_sim)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += float(loss.item())
        scheduler.step()

        lvc, fvc, yvc = infer_dataset(model, None, device, cached_ds=val_clean_ds)
        lvo, fvo, yvo = infer_dataset(model, None, device, cached_ds=val_occ_ds)
        acc_vc = float((torch.argmax(lvc, dim=1) == yvc).float().mean().item())
        acc_vo = float((torch.argmax(lvo, dim=1) == yvo).float().mean().item())
        _, _, mval = compute_cosine_separation(torch.cat([fvc, fvo], 0), torch.cat([yvc, yvo], 0))
        score = 0.40 * acc_vc + 0.40 * acc_vo + 0.20 * max(0.0, mval)
        hist_c.append({"epoch": ep, "val_clean_acc": acc_vc, "val_occ_acc": acc_vo, "val_cos_margin": mval, "val_score": score})
        mark = ""
        if score > best_score:
            best_score = score
            torch.save({"epoch": ep, "model": model.state_dict(), "classes": classes, "scale": 16.0, "k": 3, "val_score": score}, ckpt_c)
            mark = " *BEST*"
        print(f"  Ep {ep:02d}/{epochs} | val_clean={acc_vc*100:5.2f}% | val_occ={acc_vo*100:5.2f}% | cos_margin={mval:+.4f} | score={score:.4f}{mark}")

    ck = torch.load(ckpt_c, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    res_c = evaluate_adapted_model("v2 Frozen-DINOv2+Adapter", model, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device)

    # Export FP16 compact checkpoint
    fp16_c = os.path.join(OUT_ROOT, "pig_reid_v2_frozen_adapter_fp16.pth")
    half_sd = {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in ck["model"].items()}
    torch.save({"backbone_type": "dinov2_s14_frozen_adapter", "model": half_sd, "classes": classes, "scale": 16.0, "k": 3, "precision": "fp16"}, fp16_c)
    fp16_mb = os.path.getsize(fp16_c) / (1024 * 1024)

    m_fp16 = copy.deepcopy(model).half().eval()
    x1 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(20):
            _ = m_fp16.extract_features(x1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = m_fp16.extract_features(x1)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t1) * 1000.0 / 100.0

    # Update v2_reid_summary.json with Exp C
    sum_p = os.path.join(OUT_ROOT, "v2_reid_summary.json")
    with open(sum_p, "r", encoding="utf-8") as f:
        sdata = json.load(f)
    sdata["results"]["exp_c_frozen_dino_adapter"] = res_c
    sdata["histories"]["exp_c"] = hist_c
    sdata["fp16_export"]["frozen_adapter"] = {"fp16_mb": fp16_mb, "latency_fp16_ms": ms}
    with open(sum_p, "w", encoding="utf-8") as f:
        json.dump(sdata, f, indent=2)

    print(f"\nExp C Complete in {time.time()-t0:.1f}s | FP16={fp16_mb:.2f} MB ({ms:.2f} ms)")


if __name__ == "__main__":
    main()
