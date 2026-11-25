import torch  # Import first on Windows to avoid WinError 4551
import os
import re
import sys
import json
import time
import math
import shutil
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ai_fence_reid_aug import (
    TRAIN_OVERLAY_NAMES,
    TEST_UNSEEN_OVERLAY_NAMES,
    load_overlays,
    apply_ai_fence_occlusion,
    RandomAIFenceOcclusion,
)
from reid_p0_models import (
    BaselineECBModel,
    ECBHybridP0Model,
    PigReIDCNNOnly,
    BatchHardTripletLoss,
    export_cnn_only_checkpoint,
)

SPLIT2_ROOT = os.path.join(BUNDLE_ROOT, "03_datasets", "reid_ecb_dataset_split2")
SHAOJIANG_ROOT = os.path.join(BUNDLE_ROOT, "03_datasets", "external_test_datasets", "shaojiang_pig_faces")
HARD_SAMPLES_DIR = os.path.join(BUNDLE_ROOT, "03_datasets", "external_test_datasets", "generated_hard_samples")
BASELINE_CKPT = os.path.join(BUNDLE_ROOT, "02_models", "v4_parameters", "pig_ecb_best.pth")

OUT_ROOT = os.path.join(BUNDLE_ROOT, "runs_workstation", "reid_p0_occlusion")
CACHE_OCC_DIR = os.path.join(OUT_ROOT, "cached_eval_occ")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def natural_key(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def list_images_sorted(folder):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]


def union_dedup_sorted(folders):
    by_name = {}
    for fd in folders:
        for p in list_images_sorted(fd):
            bn = os.path.basename(p)
            if bn not in by_name:
                by_name[bn] = p
    keys = sorted(by_name.keys(), key=natural_key)
    return [by_name[k] for k in keys]


def build_chronological_split(baseline_classes):
    """
    Rebuild a strict zero-leakage chronological 17-class split from reid_ecb_dataset_split2.
    For each class sorted naturally by frame index (N frames):
      train: [0 : 0.70N], gap1: [0.70N : 0.75N], val: [0.75N : 0.85N],
      gap2: [0.85N : 0.90N], test_known: [0.90N : N].
    """
    class_sources = {
        "295-6": [os.path.join(SPLIT2_ROOT, "train", "295-6")],
        "297-6": [os.path.join(SPLIT2_ROOT, "train", "297-6")],
        "322-10": [os.path.join(SPLIT2_ROOT, "train", "322-10")],
        "322-2": [os.path.join(SPLIT2_ROOT, "train", "322-2")],
        "322-3": [os.path.join(SPLIT2_ROOT, "train", "322-3")],
        "323-6": [os.path.join(SPLIT2_ROOT, "train", "323-6")],
        "329-1": [os.path.join(SPLIT2_ROOT, "train", "329-1")],
        "329-7": [os.path.join(SPLIT2_ROOT, "train", "329-7")],
        "344-5": [os.path.join(SPLIT2_ROOT, "train", "344-5")],
        "376-3": [os.path.join(SPLIT2_ROOT, "train", "376-3")],
        "381-8": [os.path.join(SPLIT2_ROOT, "train", "381-8")],
        "382-5": [os.path.join(SPLIT2_ROOT, "train", "382-5")],
        "388-4_1": [os.path.join(SPLIT2_ROOT, "train", "388-4_1")],
        "472-8_2": [os.path.join(SPLIT2_ROOT, "val", "472-8_1")],
        "476-4_2": [
            os.path.join(SPLIT2_ROOT, "unknown", "feature", "476-4"),
            os.path.join(SPLIT2_ROOT, "unknown", "test", "476-4"),
        ],
        "478-2_2": [
            os.path.join(SPLIT2_ROOT, "unknown", "feature", "478-2"),
            os.path.join(SPLIT2_ROOT, "unknown", "test", "478-2"),
        ],
        "484-6_4": [os.path.join(SPLIT2_ROOT, "test_known", "484-6_4")],
    }

    assert list(baseline_classes) == list(class_sources.keys()), (
        f"Class order mismatch: {baseline_classes} vs {list(class_sources.keys())}"
    )

    train_items, val_items, test_items = [], [], []
    per_class_stats = {}

    for cls_idx, cls_name in enumerate(baseline_classes):
        paths = union_dedup_sorted(class_sources[cls_name])
        n = len(paths)
        assert n >= 100, f"Unexpectedly few images for {cls_name}: {n}"

        i_tr_end = int(0.70 * n)
        i_va_start = int(0.75 * n)
        i_va_end = int(0.85 * n)
        i_te_start = int(0.90 * n)

        tr_p = paths[:i_tr_end]
        va_p = paths[i_va_start:i_va_end]
        te_p = paths[i_te_start:]

        for p in tr_p:
            train_items.append((p, cls_idx, cls_name))
        for p in va_p:
            val_items.append((p, cls_idx, cls_name))
        for p in te_p:
            test_items.append((p, cls_idx, cls_name))

        per_class_stats[cls_name] = {
            "total": n,
            "train": len(tr_p),
            "gap1": i_va_start - i_tr_end,
            "val": len(va_p),
            "gap2": i_te_start - i_va_end,
            "test_known": len(te_p),
        }

    # Zero-leakage assertions
    tr_names = {os.path.basename(x[0]) for x in train_items}
    va_names = {os.path.basename(x[0]) for x in val_items}
    te_names = {os.path.basename(x[0]) for x in test_items}
    assert tr_names.isdisjoint(va_names), "Train/Val filename overlap!"
    assert tr_names.isdisjoint(te_names), "Train/Test filename overlap!"
    assert va_names.isdisjoint(te_names), "Val/Test filename overlap!"

    # Strictly unseen local unknown pigs: 472-10 + unknown_1779* (excludes 295-6, 472-8, 476-4, 478-2, 484-6)
    local_unknown_groups = {
        "472-10": union_dedup_sorted([
            os.path.join(SPLIT2_ROOT, "unknown", "feature", "472-10"),
            os.path.join(SPLIT2_ROOT, "unknown", "test", "472-10"),
        ]),
    }
    for sub in ("unknown_1779781577", "unknown_1779848860", "unknown_1779849593", "unknown_1779851258"):
        local_unknown_groups[sub] = union_dedup_sorted([
            os.path.join(SPLIT2_ROOT, "unknown", "feature", sub),
            os.path.join(SPLIT2_ROOT, "unknown", "test", sub),
        ])

    local_unknown_items = []
    for uid, plist in local_unknown_groups.items():
        for p in plist:
            local_unknown_items.append((p, -1, uid))

    manifest = {
        "classes": list(baseline_classes),
        "counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test_known": len(test_items),
            "local_unknown": len(local_unknown_items),
        },
        "per_class": per_class_stats,
        "local_unknown_per_pig": {k: len(v) for k, v in local_unknown_groups.items()},
    }
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return train_items, val_items, test_items, local_unknown_items, manifest


def build_deterministic_occluded_set(items, split_tag, unseen_overlays, seed, occ_range=(0.20, 0.42)):
    """Pre-generate deterministic occluded copies via PIL so Baseline/Ablation/Full see identical pixels."""
    out_dir = os.path.join(CACHE_OCC_DIR, split_tag)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached["items"]) == len(items):
            occ_items = [(os.path.join(out_dir, it["rel"]), it["label"], it["cls_name"]) for it in cached["items"]]
            if all(os.path.exists(x[0]) for x in occ_items[:5]):
                return occ_items, cached["mean_occ"]

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    occ_items = []
    meta_entries = []
    fracs = []
    for idx, (src_path, lbl, cls_name) in enumerate(items):
        rel_name = f"{idx:04d}_{cls_name}_{os.path.basename(src_path)}"
        if not rel_name.lower().endswith(".jpg"):
            rel_name = os.path.splitext(rel_name)[0] + ".jpg"
        dst_path = os.path.join(out_dir, rel_name)
        im = Image.open(src_path).convert("RGB")
        occ_im, frac, ov_name = apply_ai_fence_occlusion(im, unseen_overlays, rng, occ_range=occ_range)
        occ_im.save(dst_path, format="JPEG", quality=92)
        fracs.append(frac)
        occ_items.append((dst_path, lbl, cls_name))
        meta_entries.append({"rel": rel_name, "label": lbl, "cls_name": cls_name, "frac": round(frac, 4), "ov": ov_name})

    mean_occ = float(np.mean(fracs)) if fracs else 0.0
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"mean_occ": mean_occ, "items": meta_entries}, f, indent=2)
    return occ_items, mean_occ


class ManifestImageDataset(Dataset):
    def __init__(self, items, transform=None, cache_size=None):
        self.items = items
        self.transform = transform
        self.cached_imgs = None
        if cache_size is not None:
            t0 = time.time()
            self.cached_imgs = [
                Image.open(path).convert("RGB").resize(cache_size, Image.Resampling.BILINEAR)
                for path, _, _ in items
            ]
            print(f"  [RAM Cache] Preloaded {len(self.cached_imgs)} imgs @ {cache_size} in {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        _, label, _ = self.items[idx]
        if self.cached_imgs is not None:
            img = self.cached_imgs[idx]
        else:
            img = Image.open(self.items[idx][0]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_CACHED_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_sharpness_pil(img_path):
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_roc_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos_total = int(np.sum(labels == 1))
    neg_total = int(np.sum(labels == 0))
    if pos_total == 0 or neg_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, -1.0]), 0.0, 1.0, 0.5, 0.5

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    distinct_idx = np.where(np.diff(sorted_scores))[0]
    thresh_idx = np.r_[distinct_idx, sorted_labels.size - 1]

    tps = np.cumsum(sorted_labels)[thresh_idx]
    fps = (1 + thresh_idx) - tps

    tpr = np.r_[0.0, tps / pos_total]
    fpr = np.r_[0.0, fps / neg_total]
    thresholds = np.r_[sorted_scores[0] + 1e-6, sorted_scores[thresh_idx]]

    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[eer_idx] + fnr[eer_idx]))
    eer_thresh = float(thresholds[eer_idx])
    youden_idx = int(np.argmax(tpr - fpr))
    youden_thresh = float(thresholds[youden_idx])
    return fpr, tpr, thresholds, auc, eer, eer_thresh, youden_thresh


@torch.no_grad()
def infer_dataset(model, items, device, batch_size=64, use_amp=True, cached_ds=None):
    if cached_ds is not None:
        ds = cached_ds
    else:
        ds = ManifestImageDataset(items, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_logits, all_feats, all_labels = [], [], []
    model.eval()
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, feats = model.extract_features(imgs)
        all_logits.append(logits.float().cpu())
        all_feats.append(feats.float().cpu())
        all_labels.append(lbls)
    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_feats, dim=0),
        torch.cat(all_labels, dim=0),
    )


def compute_cosine_separation(feats, labels):
    """Compute mean intra-class and inter-class cosine similarity (excluding self-pairs)."""
    sim_mat = torch.matmul(feats, feats.t()).numpy()
    lbl_np = labels.numpy()
    same = (lbl_np[:, None] == lbl_np[None, :])
    eye = np.eye(len(lbl_np), dtype=bool)
    intra_vals = sim_mat[same & (~eye)]
    inter_vals = sim_mat[~same]
    intra_mean = float(np.mean(intra_vals)) if len(intra_vals) > 0 else 0.0
    inter_mean = float(np.mean(inter_vals)) if len(inter_vals) > 0 else 0.0
    return intra_mean, inter_mean, intra_mean - inter_mean


def train_variant(
    variant_name,
    use_bnneck_arcface,
    train_ds_cached,
    val_clean_ds,
    val_occ_ds,
    classes,
    device,
    out_ckpt_path,
    epochs=20,
    batch_size=48,
):
    print("\n" + "=" * 80)
    print(f"Training {variant_name} | BNNeck+ArcFace={use_bnneck_arcface} | epochs={epochs} | batch={batch_size}")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(train_ds_cached, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = ECBHybridP0Model(
        num_classes=len(classes),
        use_bnneck_arcface=use_bnneck_arcface,
        s=16.0,
        m=0.30,
    ).to(device)
    n_loaded, n_miss, n_unexp = model.load_backbone_from_baseline(BASELINE_CKPT)
    print(f"  Loaded backbone keys from baseline: {n_loaded} (missing head/bn={n_miss}, unexpected={n_unexp})")

    backbone_params = list(model.vit_model.parameters()) + list(model.cnn_model.parameters())
    head_params = (
        list(model.vit_bnneck.parameters())
        + list(model.classifier_vit.parameters())
        + list(model.cnn_bnneck.parameters())
        + list(model.classifier_cnn.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-4},
            {"params": head_params, "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 2), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss_fn = BatchHardTripletLoss(margin=0.35)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_score = -1.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                lv, lc, cos_v, cos_c, emb_v, emb_c = model(imgs, lbls)
                loss_vit = ce_loss_fn(lv, lbls)
                loss_cnn = ce_loss_fn(lc, lbls)
                # Temperature-scaled ViT -> CNN KD
                T = 2.0
                loss_kd = (
                    F.kl_div(
                        F.log_softmax(cos_c.float() / T, dim=1),
                        F.softmax(cos_v.detach().float() / T, dim=1),
                        reduction="batchmean",
                    )
                    * (T * T)
                )
                if use_bnneck_arcface:
                    loss_tri = triplet_loss_fn(emb_c, lbls)
                    loss = loss_vit + loss_cnn + 0.5 * loss_tri + 0.3 * loss_kd
                else:
                    loss = loss_vit + loss_cnn + 0.2 * loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += float(loss.item())

        scheduler.step()
        avg_loss = ep_loss / max(1, len(train_loader))

        # Validation on Clean + Unseen-Fence Occluded
        logits_vc, feats_vc, lbl_vc = infer_dataset(model, None, device, cached_ds=val_clean_ds)
        logits_vo, feats_vo, lbl_vo = infer_dataset(model, None, device, cached_ds=val_occ_ds)
        acc_vc = float((torch.argmax(logits_vc, dim=1) == lbl_vc).float().mean().item())
        acc_vo = float((torch.argmax(logits_vo, dim=1) == lbl_vo).float().mean().item())
        _, _, margin_val = compute_cosine_separation(
            torch.cat([feats_vc, feats_vo], dim=0),
            torch.cat([lbl_vc, lbl_vo], dim=0),
        )
        score = 0.45 * acc_vc + 0.45 * acc_vo + 0.10 * max(0.0, margin_val)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_clean_acc": acc_vc,
            "val_occ_acc": acc_vo,
            "val_cos_margin": margin_val,
            "val_score": score,
        })

        mark = ""
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "classes": classes,
                    "use_bnneck_arcface": use_bnneck_arcface,
                    "scale": 16.0,
                    "val_score": score,
                },
                out_ckpt_path,
            )
            mark = " *BEST*"
        print(
            f"  Ep {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val_clean={acc_vc*100:5.2f}% "
            f"| val_occ={acc_vo*100:5.2f}% | cos_margin={margin_val:+.4f} | score={score:.4f}{mark}"
        )

    elapsed = time.time() - t0
    print(f"Finished {variant_name} in {elapsed:.1f}s | Best Val Score={best_score:.4f}")
    return history, elapsed


def evaluate_shaojiang(model, device, shaojiang_occ_items):
    """Exact protocol of eval_clean_pig_faces.py on clean AND occluded query sets."""
    gal_dir = os.path.join(SHAOJIANG_ROOT, "clean_gallery")
    qry_dir = os.path.join(SHAOJIANG_ROOT, "clean_query")

    gal_pigs = sorted([d for d in os.listdir(gal_dir) if os.path.isdir(os.path.join(gal_dir, d))], key=natural_key)
    qry_pigs = sorted([d for d in os.listdir(qry_dir) if os.path.isdir(os.path.join(qry_dir, d))], key=natural_key)
    primary_pigs = sorted(list(set(gal_pigs) & set(qry_pigs)), key=natural_key)

    # All clean crops for unknown rejection (102 crops)
    all_clean_items = []
    for sub_dir in (gal_dir, qry_dir):
        for pid in sorted(os.listdir(sub_dir), key=natural_key):
            for p in list_images_sorted(os.path.join(sub_dir, pid)):
                all_clean_items.append((p, 0, pid))
    logits_all, _, _ = infer_dataset(model, all_clean_items, device)
    probs_all = F.softmax(logits_all, dim=1)
    max_probs = torch.max(probs_all, dim=1).values.numpy()

    # Primary gallery Top-5 sharpness centroids
    gal_items = []
    for pid in primary_pigs:
        for p in list_images_sorted(os.path.join(gal_dir, pid)):
            gal_items.append((p, primary_pigs.index(pid), pid))
    _, gal_feats, _ = infer_dataset(model, gal_items, device)

    gal_by_pig = {pid: [] for pid in primary_pigs}
    for idx, (p, _, pid) in enumerate(gal_items):
        gal_by_pig[pid].append((compute_sharpness_pil(p), gal_feats[idx]))

    centroids = []
    for pid in primary_pigs:
        shots = sorted(gal_by_pig[pid], key=lambda x: x[0], reverse=True)[:5]
        c_vec = F.normalize(torch.stack([s[1] for s in shots], dim=0).mean(dim=0, keepdim=True), p=2, dim=1)[0]
        centroids.append(c_vec)
    centroid_mat = torch.stack(centroids, dim=0)  # (11, 512)

    def _eval_query_set(q_items):
        _, q_feats, _ = infer_dataset(model, q_items, device)
        sim_mat = torch.matmul(q_feats, centroid_mat.t()).numpy()  # (Nq, 11)
        r1, r3 = 0, 0
        scores, labels = [], []
        intra_vals, inter_vals = [], []
        pca_feats, pca_pids = [], []
        for qi, (_, _, q_pid) in enumerate(q_items):
            if q_pid not in primary_pigs:
                continue
            true_idx = primary_pigs.index(q_pid)
            row_sims = sim_mat[qi]
            order = np.argsort(-row_sims)
            if order[0] == true_idx:
                r1 += 1
            if true_idx in order[:3]:
                r3 += 1
            for gi in range(len(primary_pigs)):
                s_val = float(row_sims[gi])
                scores.append(s_val)
                if gi == true_idx:
                    labels.append(1)
                    intra_vals.append(s_val)
                else:
                    labels.append(0)
                    inter_vals.append(s_val)
            pca_feats.append(q_feats[qi].numpy())
            pca_pids.append(q_pid)

        n_q = max(1, len(pca_pids))
        fpr, tpr, thresholds, auc, eer, eer_t, youden_t = compute_roc_metrics(scores, labels)
        scores_np = np.asarray(scores)
        labels_np = np.asarray(labels)
        fpr_082 = float(np.mean(scores_np[labels_np == 0] >= 0.82))
        fpr_090 = float(np.mean(scores_np[labels_np == 0] >= 0.90))
        tpr_youden = float(np.mean(scores_np[labels_np == 1] >= youden_t))
        fpr_youden = float(np.mean(scores_np[labels_np == 0] >= youden_t))
        intra_m = float(np.mean(intra_vals))
        inter_m = float(np.mean(inter_vals))
        return {
            "n_queries": len(pca_pids),
            "rank1": r1 / n_q,
            "rank3": r3 / n_q,
            "intra_mean": intra_m,
            "intra_std": float(np.std(intra_vals)),
            "inter_mean": inter_m,
            "inter_std": float(np.std(inter_vals)),
            "gap": intra_m - inter_m,
            "auc": auc,
            "eer": eer,
            "eer_thresh": eer_t,
            "youden_thresh": youden_t,
            "tpr_at_youden": tpr_youden,
            "fpr_at_youden": fpr_youden,
            "fpr_at_082": fpr_082,
            "fpr_at_090": fpr_090,
            "fpr_curve": fpr.tolist(),
            "tpr_curve": tpr.tolist(),
            "intra_vals": intra_vals,
            "inter_vals": inter_vals,
            "query_feats": np.asarray(pca_feats).tolist(),
            "query_pids": pca_pids,
        }

    # Clean queries
    clean_q_items = []
    for pid in primary_pigs:
        for p in list_images_sorted(os.path.join(qry_dir, pid)):
            clean_q_items.append((p, primary_pigs.index(pid), pid))
    occ_q_items = [x for x in shaojiang_occ_items if x[2] in primary_pigs]

    return {
        "primary_pigs": primary_pigs,
        "centroids": centroid_mat.numpy().tolist(),
        "unknown_rej": {
            "n_crops": len(max_probs),
            "mean_prob": float(np.mean(max_probs)),
            "max_prob": float(np.max(max_probs)),
            "trr_065": float(np.mean(max_probs < 0.65)),
            "far_065": float(np.mean(max_probs >= 0.65)),
            "trr_050": float(np.mean(max_probs < 0.50)),
            "far_050": float(np.mean(max_probs >= 0.50)),
        },
        "clean": _eval_query_set(clean_q_items),
        "occluded": _eval_query_set(occ_q_items),
    }


def evaluate_model_all(
    model_name,
    model,
    classes,
    test_clean_items,
    test_occ_items,
    local_unknown_items,
    shaojiang_occ_items,
    device,
):
    print(f"\n--- Evaluating [{model_name}] across 5 benchmarks ---")
    # Bench 1 & 2: Closed-Set Clean & Unseen-Fence Occluded
    logits_tc, feats_tc, lbl_tc = infer_dataset(model, test_clean_items, device)
    logits_to, feats_to, lbl_to = infer_dataset(model, test_occ_items, device)

    probs_tc = F.softmax(logits_tc, dim=1)
    probs_to = F.softmax(logits_to, dim=1)
    acc_tc = float((torch.argmax(probs_tc, dim=1) == lbl_tc).float().mean().item())
    acc_to = float((torch.argmax(probs_to, dim=1) == lbl_to).float().mean().item())
    true_conf_tc = float(probs_tc[torch.arange(len(lbl_tc)), lbl_tc].mean().item())
    true_conf_to = float(probs_to[torch.arange(len(lbl_to)), lbl_to].mean().item())
    intra_tc, inter_tc, gap_tc = compute_cosine_separation(feats_tc, lbl_tc)
    intra_to, inter_to, gap_to = compute_cosine_separation(feats_to, lbl_to)
    orig_vs_occ_cos = float(torch.sum(feats_tc * feats_to, dim=1).mean().item())

    # Bench 3: Local Unseen Unknown Rejection (472-10 + unknown_1779*)
    logits_unk, _, _ = infer_dataset(model, local_unknown_items, device)
    max_probs_unk = torch.max(F.softmax(logits_unk, dim=1), dim=1).values.numpy()
    local_unk_res = {
        "n": len(max_probs_unk),
        "mean_prob": float(np.mean(max_probs_unk)),
        "p90_prob": float(np.percentile(max_probs_unk, 90)),
        "max_prob": float(np.max(max_probs_unk)),
        "trr_065": float(np.mean(max_probs_unk < 0.65)),
        "far_065": float(np.mean(max_probs_unk >= 0.65)),
        "trr_050": float(np.mean(max_probs_unk < 0.50)),
        "far_050": float(np.mean(max_probs_unk >= 0.50)),
    }

    # Bench 4: External Shaojiang Clean & Unseen-Fence Occluded
    sj_res = evaluate_shaojiang(model, device, shaojiang_occ_items)

    # Bench 5: Generated Hard Samples Probe
    idx_295 = classes.index("295-6")
    centroid_295 = F.normalize(feats_tc[lbl_tc == idx_295].mean(dim=0, keepdim=True), p=2, dim=1)[0]
    p_gen_295 = os.path.join(HARD_SAMPLES_DIR, "01_reid_295-6_vertical_fence.png")
    p_gen_p16 = os.path.join(HARD_SAMPLES_DIR, "03_reid_pig16_angled_rails.png")

    logits_h1, feats_h1, _ = infer_dataset(model, [(p_gen_295, idx_295, "295-6")], device)
    probs_h1 = F.softmax(logits_h1, dim=1)[0]
    top1_h1 = int(torch.argmax(probs_h1).item())
    probe_295 = {
        "pred_id": classes[top1_h1],
        "top1_prob": float(probs_h1[top1_h1].item()),
        "prob_295_6": float(probs_h1[idx_295].item()),
        "cos_to_295_centroid": float(torch.dot(feats_h1[0], centroid_295).item()),
    }

    _, feats_h3, _ = infer_dataset(model, [(p_gen_p16, 0, "pig_16")], device)
    sj_centroids = torch.tensor(sj_res["centroids"], dtype=torch.float32)
    sims_p16 = torch.matmul(feats_h3[0], sj_centroids.t()).numpy()
    p_names = sj_res["primary_pigs"]
    order_p16 = np.argsort(-sims_p16)
    top3_p16 = [(p_names[i], float(sims_p16[i])) for i in order_p16[:3]]
    idx_p16 = p_names.index("pig_16")
    other_best = max(float(sims_p16[i]) for i in range(len(p_names)) if i != idx_p16)
    probe_p16 = {
        "top3": top3_p16,
        "pig16_sim": float(sims_p16[idx_p16]),
        "margin_over_next": float(sims_p16[idx_p16]) - other_best,
    }

    # Also store 2 representative test_known_occluded sample predictions for Fig 4
    sample_occ_probes = []
    for pick_cls in ("476-4_2", "484-6_4"):
        c_idx = classes.index(pick_cls)
        matches = [i for i, (_, l, _) in enumerate(test_occ_items) if l == c_idx]
        if matches:
            si = matches[len(matches) // 2]
            p_vec = probs_to[si]
            pred_i = int(torch.argmax(p_vec).item())
            sample_occ_probes.append({
                "path": test_occ_items[si][0],
                "true_cls": pick_cls,
                "pred_cls": classes[pred_i],
                "true_prob": float(p_vec[c_idx].item()),
                "top1_prob": float(p_vec[pred_i].item()),
            })

    summary = {
        "closed_clean": {
            "n": len(lbl_tc),
            "acc": acc_tc,
            "mean_true_conf": true_conf_tc,
            "intra_cos": intra_tc,
            "inter_cos": inter_tc,
            "gap": gap_tc,
        },
        "closed_occluded": {
            "n": len(lbl_to),
            "acc": acc_to,
            "mean_true_conf": true_conf_to,
            "intra_cos": intra_to,
            "inter_cos": inter_to,
            "gap": gap_to,
            "orig_vs_occ_cos": orig_vs_occ_cos,
        },
        "local_unknown": local_unk_res,
        "shaojiang": sj_res,
        "hard_probe": {
            "295_6_vertical_fence": probe_295,
            "pig16_angled_rails": probe_p16,
            "occ_samples": sample_occ_probes,
        },
    }

    print(
        f"  Closed Clean Acc={acc_tc*100:.2f}% (conf={true_conf_tc*100:.1f}%) | "
        f"Closed Occ Acc={acc_to*100:.2f}% (conf={true_conf_to*100:.1f}%)"
    )
    print(
        f"  Local Unk TRR@0.65={local_unk_res['trr_065']*100:.2f}% | "
        f"SJ Clean Rank1={sj_res['clean']['rank1']*100:.2f}% AUC={sj_res['clean']['auc']:.4f} EER={sj_res['clean']['eer']*100:.2f}% Gap={sj_res['clean']['gap']:+.4f}"
    )
    print(
        f"  SJ Occ Rank1={sj_res['occluded']['rank1']*100:.2f}% AUC={sj_res['occluded']['auc']:.4f} EER={sj_res['occluded']['eer']*100:.2f}% | "
        f"Hard 295-6 prob={probe_295['prob_295_6']*100:.1f}% | p16 margin={probe_p16['margin_over_next']:+.4f}"
    )
    return summary


def benchmark_cnn_only_deployment(full_model, classes, cnn_only_path, full_ckpt_path, device):
    cnn_only = export_cnn_only_checkpoint(
        full_model,
        classes,
        cnn_only_path,
        meta={"source": os.path.basename(full_ckpt_path)},
    ).to(device)
    cnn_only.eval()
    full_model.eval()

    # Parity check on random batch
    x_test = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        l_full, f_full = full_model.extract_features(x_test)
        l_cnn, f_cnn = cnn_only.extract_features(x_test)
    max_emb_diff = float((f_full - f_cnn).abs().max().item())
    max_log_diff = float((l_full - l_cnn).abs().max().item())

    # Speed & Peak VRAM benchmark (FP16 single-image 100 iters)
    def _bench(m):
        m_fp16 = m.half()
        x1 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(20):
                _ = m_fp16.extract_features(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = m_fp16.extract_features(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / 100.0
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
        return ms, mem_mb

    # Measure isolated load memory via model parameters size + runtime
    full_params_mb = sum(p.numel() * p.element_size() for p in full_model.parameters()) / (1024 * 1024)
    cnn_params_mb = sum(p.numel() * p.element_size() for p in cnn_only.parameters()) / (1024 * 1024)
    ms_full, _ = _bench(full_model)
    ms_cnn, _ = _bench(cnn_only)
    full_model.float()

    return {
        "full_ckpt_mb": os.path.getsize(full_ckpt_path) / (1024 * 1024),
        "cnn_only_ckpt_mb": os.path.getsize(cnn_only_path) / (1024 * 1024),
        "full_params_fp32_mb": full_params_mb,
        "cnn_params_fp32_mb": cnn_params_mb,
        "max_emb_diff": max_emb_diff,
        "max_logit_diff": max_log_diff,
        "latency_full_ms": ms_full,
        "latency_cnn_only_ms": ms_cnn,
    }


def generate_visual_figures(manifest, results_by_model, train_items, train_overlays, unseen_overlays, deploy_bench):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444", "axes.linewidth": 0.9})

    # ------------------------------------------------------------------
    # Fig 1: AI Fence Augmentation Samples + Chronological Zero-Leakage Split
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 8.5), dpi=180)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.14)

    sample_path = train_items[0][0]  # 295-6 clean sample
    orig_pil = Image.open(sample_path).convert("RGB").resize((256, 256))
    rng_vis = np.random.default_rng(777)

    tiles = [("Original (295-6)", orig_pil)]
    for ov_name, ov_img in [train_overlays[0], train_overlays[1]]:
        occ_im, frac, _ = apply_ai_fence_occlusion(orig_pil, [(ov_name, ov_img)], rng_vis, occ_range=(0.20, 0.38))
        short_n = ov_name.replace("ai_asset_", "Train ").replace("_rgba.png", "")
        tiles.append((f"{short_n}\n(occ={frac*100:.1f}%)", occ_im))
    for ov_name, ov_img in unseen_overlays:
        occ_im, frac, _ = apply_ai_fence_occlusion(orig_pil, [(ov_name, ov_img)], rng_vis, occ_range=(0.22, 0.38))
        short_n = ov_name.replace("ai_asset_", "Unseen Test ").replace("_rgba.png", "")
        tiles.append((f"{short_n}\n(occ={frac*100:.1f}%)", occ_im))

    for col, (title, pil_im) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pil_im)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    ax_bar = fig.add_subplot(gs[1, :])
    cls_names = manifest["classes"]
    pc = manifest["per_class"]
    y_pos = np.arange(len(cls_names))
    tr_v = np.array([pc[c]["train"] for c in cls_names])
    g1_v = np.array([pc[c]["gap1"] for c in cls_names])
    va_v = np.array([pc[c]["val"] for c in cls_names])
    g2_v = np.array([pc[c]["gap2"] for c in cls_names])
    te_v = np.array([pc[c]["test_known"] for c in cls_names])

    ax_bar.barh(y_pos, tr_v, color="#2b6cb0", label="Train (70%)")
    ax_bar.barh(y_pos, g1_v, left=tr_v, color="#cbd5e0", label="Excluded Temporal Buffer (5%+5%)")
    ax_bar.barh(y_pos, va_v, left=tr_v + g1_v, color="#38a169", label="Val (10%)")
    ax_bar.barh(y_pos, g2_v, left=tr_v + g1_v + va_v, color="#cbd5e0")
    ax_bar.barh(y_pos, te_v, left=tr_v + g1_v + va_v + g2_v, color="#dd6b20", label="Test Known (10%)")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(cls_names, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Chronological Frame Count per Pig Class (Zero Frame Leakage)", fontsize=10.5)
    ax_bar.set_title(
        f"Rebuilt 17-Class Chronological Split | Train={manifest['counts']['train']} | "
        f"Val={manifest['counts']['val']} | Test={manifest['counts']['test_known']} | "
        f"Unseen Local Unknown={manifest['counts']['local_unknown']}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_bar.legend(loc="lower right", ncol=4, fontsize=9, frameon=True)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
    fig.suptitle("P0 Data Pipeline: AI Fence Augmentation & Zero-Leakage Chronological Split", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(OUT_ROOT, "p0_fig1_augmentation_and_split.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: 3-Way Ablation Benchmark Comparison
    # ------------------------------------------------------------------
    m_keys = ["Baseline", "P0-Ablation", "P0-Full"]
    colors = ["#718096", "#3182ce", "#16a34a"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), dpi=180)
    # Subplot A: Closed-Set & Local Unknown Rejection
    labels_a = ["Closed\nClean Acc", "Closed\nOcc Acc", "Local Unk\nTRR@0.65", "SJ Unk\nTRR@0.65"]
    x_a = np.arange(len(labels_a))
    w = 0.25
    for i, mk in enumerate(m_keys):
        r = results_by_model[mk]
        vals = [
            r["closed_clean"]["acc"] * 100,
            r["closed_occluded"]["acc"] * 100,
            r["local_unknown"]["trr_065"] * 100,
            r["shaojiang"]["unknown_rej"]["trr_065"] * 100,
        ]
        bars = axes[0].bar(x_a + (i - 1) * w, vals, width=w, label=mk, color=colors[i])
        for b in bars:
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels(labels_a, fontsize=9)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("(a) Closed-Set Acc & Unknown Rejection", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Subplot B: External Shaojiang Open-Set Rank-1 & ROC-AUC
    labels_b = ["SJ Clean\nRank-1", "SJ Occ\nRank-1", "SJ Clean\nROC-AUC", "SJ Occ\nROC-AUC"]
    x_b = np.arange(len(labels_b))
    for i, mk in enumerate(m_keys):
        sj = results_by_model[mk]["shaojiang"]
        vals = [
            sj["clean"]["rank1"] * 100,
            sj["occluded"]["rank1"] * 100,
            sj["clean"]["auc"] * 100,
            sj["occluded"]["auc"] * 100,
        ]
        bars = axes[1].bar(x_b + (i - 1) * w, vals, width=w, label=mk, color=colors[i])
        for b in bars:
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(0, 112)
    axes[1].set_xticks(x_b)
    axes[1].set_xticklabels(labels_b, fontsize=9)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("(b) External Unseen Pigs (Shaojiang) Re-ID", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    # Subplot C: Open-Set Error & Margin (EER% lower is better, Separation Gap x100 higher is better)
    labels_c = ["Clean EER\n(Lower=Better)", "Occ EER\n(Lower=Better)", "FPR@0.82\n(Lower=Better)", "Cos Gap Δ*100\n(Higher=Better)"]
    x_c = np.arange(len(labels_c))
    for i, mk in enumerate(m_keys):
        sj = results_by_model[mk]["shaojiang"]
        vals = [
            sj["clean"]["eer"] * 100,
            sj["occluded"]["eer"] * 100,
            sj["clean"]["fpr_at_082"] * 100,
            sj["clean"]["gap"] * 100,
        ]
        bars = axes[2].bar(x_c + (i - 1) * w, vals, width=w, label=mk, color=colors[i])
        for b in bars:
            axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_ylim(0, 108)
    axes[2].set_xticks(x_c)
    axes[2].set_xticklabels(labels_c, fontsize=8.5)
    axes[2].set_ylabel("Value (%)")
    axes[2].set_title("(c) Open-Set EER, FPR@0.82 & Separation Gap", fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        f"P0 3-Way Ablation Benchmark (RTX 4090) | CNN-Only Export: {deploy_bench['full_ckpt_mb']:.1f}MB -> {deploy_bench['cnn_only_ckpt_mb']:.1f}MB",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "p0_fig2_benchmark_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: Cosine Distributions, ROC Curves & 2D PCA Separation
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=180)
    b_sj = results_by_model["Baseline"]["shaojiang"]
    f_sj = results_by_model["P0-Full"]["shaojiang"]

    # (0,0) Baseline Intra vs Inter histogram
    bins = np.linspace(0.0, 1.0, 32)
    axes[0, 0].hist(b_sj["clean"]["inter_vals"], bins=bins, density=True, alpha=0.55, color="#e53e3e", label=f"Diff-Pig ({b_sj['clean']['inter_mean']:.3f})")
    axes[0, 0].hist(b_sj["clean"]["intra_vals"], bins=bins, density=True, alpha=0.60, color="#2b6cb0", label=f"Same-Pig ({b_sj['clean']['intra_mean']:.3f})")
    axes[0, 0].axvline(0.82, color="black", linestyle="--", lw=1.3, label="Old Thresh 0.82")
    axes[0, 0].set_title(f"Baseline Cosine Distribution (Gap Δ = {b_sj['clean']['gap']:+.3f})", fontweight="bold")
    axes[0, 0].set_xlabel("Cosine Similarity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(alpha=0.3, linestyle="--")

    # (0,1) P0-Full Intra vs Inter histogram
    axes[0, 1].hist(f_sj["clean"]["inter_vals"], bins=bins, density=True, alpha=0.55, color="#e53e3e", label=f"Diff-Pig ({f_sj['clean']['inter_mean']:.3f})")
    axes[0, 1].hist(f_sj["clean"]["intra_vals"], bins=bins, density=True, alpha=0.60, color="#16a34a", label=f"Same-Pig ({f_sj['clean']['intra_mean']:.3f})")
    axes[0, 1].axvline(f_sj["clean"]["youden_thresh"], color="black", linestyle="--", lw=1.3, label=f"Youden Thresh {f_sj['clean']['youden_thresh']:.2f}")
    axes[0, 1].set_title(f"P0-Full Cosine Distribution (Gap Δ = {f_sj['clean']['gap']:+.3f})", fontweight="bold")
    axes[0, 1].set_xlabel("Cosine Similarity")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(alpha=0.3, linestyle="--")

    # (1,0) Overlaid ROC Curves (Clean & Occluded)
    roc_specs = [
        ("Baseline Clean", b_sj["clean"], "#718096", "-"),
        ("Baseline Occ", b_sj["occluded"], "#a0aec0", "--"),
        ("P0-Ablation Clean", results_by_model["P0-Ablation"]["shaojiang"]["clean"], "#3182ce", "-"),
        ("P0-Full Clean", f_sj["clean"], "#16a34a", "-"),
        ("P0-Full Occ", f_sj["occluded"], "#15803d", "--"),
    ]
    for lbl, d, col, ls in roc_specs:
        axes[1, 0].plot(d["fpr_curve"], d["tpr_curve"], color=col, linestyle=ls, lw=2.0, label=f"{lbl} (AUC={d['auc']:.3f}, EER={d['eer']*100:.1f}%)")
    axes[1, 0].plot([0, 1], [0, 1], "k:", alpha=0.4)
    axes[1, 0].set_xlabel("False Positive Rate (FPR)")
    axes[1, 0].set_ylabel("True Positive Rate (TPR)")
    axes[1, 0].set_title("Open-Set ROC Curves (Unseen Shaojiang Pigs)", fontweight="bold")
    axes[1, 0].legend(fontsize=8, loc="lower right")
    axes[1, 0].grid(alpha=0.3, linestyle="--")

    # (1,1) 2D PCA Embedding Scatter of P0-Full on 11 Unseen Pigs (with Silhouette compare)
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    b_X = np.array(b_sj["clean"]["query_feats"])
    f_X = np.array(f_sj["clean"]["query_feats"])
    pids = f_sj["clean"]["query_pids"]
    uniq = sorted(list(set(pids)), key=natural_key)
    pid_to_i = {p: i for i, p in enumerate(uniq)}
    y_ids = np.array([pid_to_i[p] for p in pids])

    sil_base = float(silhouette_score(b_X, y_ids, metric="cosine"))
    sil_full = float(silhouette_score(f_X, y_ids, metric="cosine"))
    proj = PCA(n_components=2, random_state=42).fit_transform(f_X)
    cmap = plt.get_cmap("tab20")
    for i, p in enumerate(uniq):
        mask = y_ids == i
        axes[1, 1].scatter(proj[mask, 0], proj[mask, 1], color=cmap(i), label=p, s=55, alpha=0.88, edgecolors="white", linewidth=0.5)
    axes[1, 1].set_title(
        f"P0-Full 2D PCA of 11 Unseen Pigs\n(Cosine Silhouette: Baseline={sil_base:+.3f} -> P0-Full={sil_full:+.3f})",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("PC 1")
    axes[1, 1].set_ylabel("PC 2")
    axes[1, 1].legend(ncol=3, fontsize=7.5, loc="best")
    axes[1, 1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("Embedding Space & Open-Set Separation Diagnostics (Baseline vs P0-Full)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "p0_fig3_cosine_roc_separation.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 4: Visual Hard Samples Probe (Baseline vs P0-Full)
    # ------------------------------------------------------------------
    b_hp = results_by_model["Baseline"]["hard_probe"]
    f_hp = results_by_model["P0-Full"]["hard_probe"]

    cards = [
        (
            os.path.join(HARD_SAMPLES_DIR, "01_reid_295-6_vertical_fence.png"),
            "AI Hard Sample: 295-6 + Vertical Fence",
            f"Baseline: {b_hp['295_6_vertical_fence']['pred_id']} ({b_hp['295_6_vertical_fence']['prob_295_6']*100:.1f}%)\n"
            f"P0-Full : {f_hp['295_6_vertical_fence']['pred_id']} ({f_hp['295_6_vertical_fence']['prob_295_6']*100:.1f}%)",
        ),
        (
            os.path.join(HARD_SAMPLES_DIR, "03_reid_pig16_angled_rails.png"),
            "Unseen Open-Set: pig_16 + Angled Rails",
            f"Baseline: {b_hp['pig16_angled_rails']['top3'][0][0]} (margin={b_hp['pig16_angled_rails']['margin_over_next']:+.3f})\n"
            f"P0-Full : {f_hp['pig16_angled_rails']['top3'][0][0]} (margin={f_hp['pig16_angled_rails']['margin_over_next']:+.3f})",
        ),
    ]
    for bi, fi in zip(b_hp["occ_samples"], f_hp["occ_samples"]):
        cards.append((
            fi["path"],
            f"Unseen Test Fence: True={fi['true_cls']}",
            f"Baseline: {bi['pred_cls']} ({bi['true_prob']*100:.1f}%)\n"
            f"P0-Full : {fi['pred_cls']} ({fi['true_prob']*100:.1f}%)",
        ))

    fig, axes = plt.subplots(1, len(cards), figsize=(16, 4.8), dpi=180)
    for ax, (im_p, title, sub) in zip(axes, cards):
        ax.imshow(Image.open(im_p).convert("RGB").resize((280, 280)))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(sub, fontsize=9, fontweight="bold", color="#1a202c", labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Before-vs-After Occluded Identity Probe (Baseline vs P0-Full)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "p0_fig4_hard_samples_probe.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    t_start = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running P0 Re-ID Pipeline on device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Baseline checkpoint & classes
    base_ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    classes = list(base_ckpt["classes"])
    print(f"Baseline checkpoint classes ({len(classes)}): {classes}")

    # 2. Rebuild chronological zero-leakage split
    train_items, val_items, test_items, local_unknown_items, manifest = build_chronological_split(classes)
    print(
        f"Chronological Split Built -> train={len(train_items)}, val={len(val_items)}, "
        f"test_known={len(test_items)}, local_unknown={len(local_unknown_items)}"
    )

    # 3. Load overlays & build deterministic unseen-fence evaluation sets
    train_overlays = load_overlays(TRAIN_OVERLAY_NAMES)
    unseen_overlays = load_overlays(TEST_UNSEEN_OVERLAY_NAMES)

    val_occ_items, val_occ_frac = build_deterministic_occluded_set(val_items, "val_occ", unseen_overlays, seed=202601)
    test_occ_items, test_occ_frac = build_deterministic_occluded_set(test_items, "test_occ", unseen_overlays, seed=202602)

    sj_qry_dir = os.path.join(SHAOJIANG_ROOT, "clean_query")
    sj_qry_all = []
    for pid in sorted(os.listdir(sj_qry_dir), key=natural_key):
        for p in list_images_sorted(os.path.join(sj_qry_dir, pid)):
            sj_qry_all.append((p, 0, pid))
    sj_occ_items, sj_occ_frac = build_deterministic_occluded_set(sj_qry_all, "sj_qry_occ", unseen_overlays, seed=202603)
    print(
        f"Deterministic Unseen-Fence Eval Sets Ready -> val_occ={val_occ_frac*100:.1f}%, "
        f"test_occ={test_occ_frac*100:.1f}%, sj_occ={sj_occ_frac*100:.1f}%"
    )

    # 4. Preload train & val images into RAM (256x256 / 224x224) for 10x faster RTX 4090 epochs
    train_tfms = transforms.Compose([
        RandomAIFenceOcclusion(train_overlays, p=0.60, occ_range=(0.15, 0.40), seed=101),
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

    # 5. Train P0-Ablation & P0-Full
    abl_ckpt_path = os.path.join(OUT_ROOT, "pig_reid_p0_ablation_best.pth")
    full_ckpt_path = os.path.join(OUT_ROOT, "pig_reid_p0_full_best.pth")
    cnn_only_path = os.path.join(OUT_ROOT, "pig_reid_p0_cnn_only.pth")

    hist_abl, time_abl = train_variant(
        "P0-Ablation (ECB + AI-Fence, Linear CE)",
        use_bnneck_arcface=False,
        train_ds_cached=train_ds_cached,
        val_clean_ds=val_clean_ds,
        val_occ_ds=val_occ_ds,
        classes=classes,
        device=device,
        out_ckpt_path=abl_ckpt_path,
        epochs=20,
        batch_size=48,
    )

    hist_full, time_full = train_variant(
        "P0-Full (ECB + AI-Fence + BNNeck + ArcFace + Triplet + KD)",
        use_bnneck_arcface=True,
        train_ds_cached=train_ds_cached,
        val_clean_ds=val_clean_ds,
        val_occ_ds=val_occ_ds,
        classes=classes,
        device=device,
        out_ckpt_path=full_ckpt_path,
        epochs=20,
        batch_size=48,
    )

    # 5. Load all 3 best models & run 5-benchmark evaluation
    m_base = BaselineECBModel(num_classes=len(classes)).to(device)
    m_base.load_state_dict(base_ckpt["model"])

    ck_abl = torch.load(abl_ckpt_path, map_location="cpu", weights_only=False)
    m_abl = ECBHybridP0Model(num_classes=len(classes), use_bnneck_arcface=False).to(device)
    m_abl.load_state_dict(ck_abl["model"])

    ck_full = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
    m_full = ECBHybridP0Model(num_classes=len(classes), use_bnneck_arcface=True, s=16.0, m=0.30).to(device)
    m_full.load_state_dict(ck_full["model"])

    results = {
        "Baseline": evaluate_model_all("Baseline", m_base, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device),
        "P0-Ablation": evaluate_model_all("P0-Ablation", m_abl, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device),
        "P0-Full": evaluate_model_all("P0-Full", m_full, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device),
    }

    # 6. Export CNN-only & benchmark deployment
    deploy_bench = benchmark_cnn_only_deployment(m_full, classes, cnn_only_path, full_ckpt_path, device)
    print("\n--- Lightweight CNN-Only Deployment Benchmark ---")
    print(json.dumps(deploy_bench, indent=2))

    # 7. Save JSON + Text Report + 4 Diagnostic Figures
    payload = {
        "elapsed_total_sec": time.time() - t_start,
        "train_times_sec": {"P0-Ablation": time_abl, "P0-Full": time_full},
        "manifest": manifest,
        "occlusion_eval_means": {"val_occ": val_occ_frac, "test_occ": test_occ_frac, "sj_occ": sj_occ_frac},
        "histories": {"P0-Ablation": hist_abl, "P0-Full": hist_full},
        "results": results,
        "deploy_bench": deploy_bench,
    }
    with open(os.path.join(OUT_ROOT, "p0_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    b, a, fu = results["Baseline"], results["P0-Ablation"], results["P0-Full"]
    lines = [
        "=" * 92,
        "P0 Occlusion-Robust Pig Re-ID Evaluation & 3-Way Ablation Report",
        "=" * 92,
        f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | Total Elapsed: {payload['elapsed_total_sec']:.1f}s",
        f"Split: Train={manifest['counts']['train']}, Val={manifest['counts']['val']}, Test={manifest['counts']['test_known']}, LocalUnk={manifest['counts']['local_unknown']}",
        f"Unseen Eval Fence Occlusion: val_occ={val_occ_frac*100:.1f}%, test_occ={test_occ_frac*100:.1f}%, sj_occ={sj_occ_frac*100:.1f}%",
        "",
        f"{'Metric':<42} | {'Baseline':>12} | {'P0-Ablation':>12} | {'P0-Full':>12}",
        "-" * 88,
        f"{'Closed-Set Clean Top-1 Acc':<42} | {b['closed_clean']['acc']*100:>11.2f}% | {a['closed_clean']['acc']*100:>11.2f}% | {fu['closed_clean']['acc']*100:>11.2f}%",
        f"{'Closed-Set Clean True-Class Conf':<42} | {b['closed_clean']['mean_true_conf']*100:>11.2f}% | {a['closed_clean']['mean_true_conf']*100:>11.2f}% | {fu['closed_clean']['mean_true_conf']*100:>11.2f}%",
        f"{'Closed-Set Unseen-Fence Top-1 Acc':<42} | {b['closed_occluded']['acc']*100:>11.2f}% | {a['closed_occluded']['acc']*100:>11.2f}% | {fu['closed_occluded']['acc']*100:>11.2f}%",
        f"{'Closed-Set Unseen-Fence True-Class Conf':<42} | {b['closed_occluded']['mean_true_conf']*100:>11.2f}% | {a['closed_occluded']['mean_true_conf']*100:>11.2f}% | {fu['closed_occluded']['mean_true_conf']*100:>11.2f}%",
        f"{'Closed-Set Orig-vs-Occ Cosine Sim':<42} | {b['closed_occluded']['orig_vs_occ_cos']:>12.4f} | {a['closed_occluded']['orig_vs_occ_cos']:>12.4f} | {fu['closed_occluded']['orig_vs_occ_cos']:>12.4f}",
        f"{'Local Unseen Unknown TRR@0.65 (n=387)':<42} | {b['local_unknown']['trr_065']*100:>11.2f}% | {a['local_unknown']['trr_065']*100:>11.2f}% | {fu['local_unknown']['trr_065']*100:>11.2f}%",
        f"{'SJ Unknown Rejection TRR@0.65 (n=102)':<42} | {b['shaojiang']['unknown_rej']['trr_065']*100:>11.2f}% | {a['shaojiang']['unknown_rej']['trr_065']*100:>11.2f}% | {fu['shaojiang']['unknown_rej']['trr_065']*100:>11.2f}%",
        f"{'SJ Clean Top-5 Centroid Rank-1':<42} | {b['shaojiang']['clean']['rank1']*100:>11.2f}% | {a['shaojiang']['clean']['rank1']*100:>11.2f}% | {fu['shaojiang']['clean']['rank1']*100:>11.2f}%",
        f"{'SJ Clean Top-5 Centroid Rank-3':<42} | {b['shaojiang']['clean']['rank3']*100:>11.2f}% | {a['shaojiang']['clean']['rank3']*100:>11.2f}% | {fu['shaojiang']['clean']['rank3']*100:>11.2f}%",
        f"{'SJ Clean Intra / Inter Cosine':<42} | {b['shaojiang']['clean']['intra_mean']:.3f}/{b['shaojiang']['clean']['inter_mean']:.3f} | {a['shaojiang']['clean']['intra_mean']:.3f}/{a['shaojiang']['clean']['inter_mean']:.3f} | {fu['shaojiang']['clean']['intra_mean']:.3f}/{fu['shaojiang']['clean']['inter_mean']:.3f}",
        f"{'SJ Clean Separation Gap (Delta)':<42} | {b['shaojiang']['clean']['gap']:>+12.4f} | {a['shaojiang']['clean']['gap']:>+12.4f} | {fu['shaojiang']['clean']['gap']:>+12.4f}",
        f"{'SJ Clean ROC-AUC':<42} | {b['shaojiang']['clean']['auc']:>12.4f} | {a['shaojiang']['clean']['auc']:>12.4f} | {fu['shaojiang']['clean']['auc']:>12.4f}",
        f"{'SJ Clean EER (lower=better)':<42} | {b['shaojiang']['clean']['eer']*100:>11.2f}% | {a['shaojiang']['clean']['eer']*100:>11.2f}% | {fu['shaojiang']['clean']['eer']*100:>11.2f}%",
        f"{'SJ Clean Youden Optimal Thresh':<42} | {b['shaojiang']['clean']['youden_thresh']:>12.4f} | {a['shaojiang']['clean']['youden_thresh']:>12.4f} | {fu['shaojiang']['clean']['youden_thresh']:>12.4f}",
        f"{'SJ Clean FPR@0.82 (lower=better)':<42} | {b['shaojiang']['clean']['fpr_at_082']*100:>11.2f}% | {a['shaojiang']['clean']['fpr_at_082']*100:>11.2f}% | {fu['shaojiang']['clean']['fpr_at_082']*100:>11.2f}%",
        f"{'SJ Occluded Rank-1 (Unseen Fence)':<42} | {b['shaojiang']['occluded']['rank1']*100:>11.2f}% | {a['shaojiang']['occluded']['rank1']*100:>11.2f}% | {fu['shaojiang']['occluded']['rank1']*100:>11.2f}%",
        f"{'SJ Occluded ROC-AUC':<42} | {b['shaojiang']['occluded']['auc']:>12.4f} | {a['shaojiang']['occluded']['auc']:>12.4f} | {fu['shaojiang']['occluded']['auc']:>12.4f}",
        f"{'SJ Occluded EER (lower=better)':<42} | {b['shaojiang']['occluded']['eer']*100:>11.2f}% | {a['shaojiang']['occluded']['eer']*100:>11.2f}% | {fu['shaojiang']['occluded']['eer']*100:>11.2f}%",
        f"{'Hard Probe 295-6 Fence Prob':<42} | {b['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>11.2f}% | {a['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>11.2f}% | {fu['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>11.2f}%",
        f"{'Hard Probe pig_16 Rails Margin':<42} | {b['hard_probe']['pig16_angled_rails']['margin_over_next']:>+12.4f} | {a['hard_probe']['pig16_angled_rails']['margin_over_next']:>+12.4f} | {fu['hard_probe']['pig16_angled_rails']['margin_over_next']:>+12.4f}",
        "",
        "--- Lightweight CNN-Only Export ---",
        f"Checkpoint Size: {deploy_bench['full_ckpt_mb']:.2f} MB -> {deploy_bench['cnn_only_ckpt_mb']:.2f} MB",
        f"Max Embedding Diff vs Full: {deploy_bench['max_emb_diff']:.2e} | Max Logit Diff: {deploy_bench['max_logit_diff']:.2e}",
        f"FP16 Single-Image Latency: Full={deploy_bench['latency_full_ms']:.3f} ms | CNN-Only={deploy_bench['latency_cnn_only_ms']:.3f} ms",
        "=" * 92,
    ]
    report_txt = "\n".join(lines)
    with open(os.path.join(OUT_ROOT, "p0_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")
    print("\n" + report_txt)

    generate_visual_figures(manifest, results, train_items, train_overlays, unseen_overlays, deploy_bench)
    print("\nAll 4 P0 diagnostic figures saved successfully under:", OUT_ROOT)


if __name__ == "__main__":
    main()
