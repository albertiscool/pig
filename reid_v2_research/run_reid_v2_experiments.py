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
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
P0_CODE_DIR = os.path.join(BUNDLE_ROOT, "01_code", "reid_p0_occlusion")
for p in (SCRIPT_DIR, P0_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_fence_reid_aug import (
    TRAIN_OVERLAY_NAMES,
    TEST_UNSEEN_OVERLAY_NAMES,
    load_overlays,
    RandomAIFenceOcclusion,
)
from reid_p0_models import BatchHardTripletLoss
from train_and_eval_reid_p0 import (
    SHAOJIANG_ROOT,
    HARD_SAMPLES_DIR,
    BASELINE_CKPT,
    ManifestImageDataset,
    EVAL_TRANSFORM,
    EVAL_CACHED_TRANSFORM,
    build_chronological_split,
    build_deterministic_occluded_set,
    infer_dataset,
    compute_cosine_separation,
    evaluate_shaojiang,
    list_images_sorted,
    natural_key,
)
from foundation_reid_models import FoundationPigReID, ZeroShotFoundationWrapper

P0_RESULTS_JSON = os.path.join(BUNDLE_ROOT, "runs_workstation", "reid_p0_occlusion", "p0_eval_summary.json")
OUT_ROOT = os.path.join(BUNDLE_ROOT, "runs_workstation", "reid_v2_research")


def eval_zeroshot_model(name, model, classes, train_items, test_clean_items, test_occ_items, shaojiang_occ_items, device):
    print(f"\n--- Zero-Shot Eval [{name}] ---")
    _, tr_feats, tr_lbls = infer_dataset(model, train_items, device, batch_size=64)
    _, tc_feats, tc_lbls = infer_dataset(model, test_clean_items, device, batch_size=64)
    _, to_feats, to_lbls = infer_dataset(model, test_occ_items, device, batch_size=64)

    # Nearest-centroid accuracy on closed-set using train centroids
    tr_cents = []
    for c in range(len(classes)):
        m = tr_lbls == c
        cent = F.normalize(tr_feats[m].mean(dim=0, keepdim=True), p=2, dim=1)[0]
        tr_cents.append(cent)
    cent_mat = torch.stack(tr_cents, dim=0)
    acc_tc = float((torch.argmax(torch.matmul(tc_feats, cent_mat.t()), dim=1) == tc_lbls).float().mean().item())
    acc_to = float((torch.argmax(torch.matmul(to_feats, cent_mat.t()), dim=1) == to_lbls).float().mean().item())
    intra_tc, inter_tc, gap_tc = compute_cosine_separation(tc_feats, tc_lbls)
    intra_to, inter_to, gap_to = compute_cosine_separation(to_feats, to_lbls)
    orig_vs_occ_cos = float(torch.sum(tc_feats * to_feats, dim=1).mean().item())

    sj_res = evaluate_shaojiang(model, device, shaojiang_occ_items)

    # Hard probe pig_16
    p_gen_p16 = os.path.join(HARD_SAMPLES_DIR, "03_reid_pig16_angled_rails.png")
    _, feats_h3, _ = infer_dataset(model, [(p_gen_p16, 0, "pig_16")], device)
    sj_cents = torch.tensor(sj_res["centroids"], dtype=torch.float32)
    sims_p16 = torch.matmul(feats_h3[0], sj_cents.t()).numpy()
    p_names = sj_res["primary_pigs"]
    order_p16 = np.argsort(-sims_p16)
    idx_p16 = p_names.index("pig_16")
    other_best = max(float(sims_p16[i]) for i in range(len(p_names)) if i != idx_p16)
    probe_p16 = {
        "top3": [(p_names[i], float(sims_p16[i])) for i in order_p16[:3]],
        "pig16_sim": float(sims_p16[idx_p16]),
        "margin_over_next": float(sims_p16[idx_p16]) - other_best,
    }

    print(
        f"  Train-Centroid Acc: Clean={acc_tc*100:.2f}% Occ={acc_to*100:.2f}% | "
        f"SJ Clean Rank1={sj_res['clean']['rank1']*100:.2f}% AUC={sj_res['clean']['auc']:.4f} EER={sj_res['clean']['eer']*100:.2f}% Gap={sj_res['clean']['gap']:+.4f} | "
        f"SJ Occ Rank1={sj_res['occluded']['rank1']*100:.2f}% AUC={sj_res['occluded']['auc']:.4f} EER={sj_res['occluded']['eer']*100:.2f}% | p16_margin={probe_p16['margin_over_next']:+.4f}"
    )
    return {
        "closed_clean": {"acc": acc_tc, "intra_cos": intra_tc, "inter_cos": inter_tc, "gap": gap_tc},
        "closed_occluded": {"acc": acc_to, "intra_cos": intra_to, "inter_cos": inter_to, "gap": gap_to, "orig_vs_occ_cos": orig_vs_occ_cos},
        "shaojiang": sj_res,
        "hard_probe": {"pig16_angled_rails": probe_p16},
    }


def train_foundation_reid(
    exp_name,
    backbone_type,
    train_ds_cached,
    val_clean_ds,
    val_occ_ds,
    classes,
    device,
    out_ckpt_path,
    epochs=18,
    batch_size=48,
):
    print("\n" + "=" * 84)
    print(f"Training {exp_name} | backbone={backbone_type} | epochs={epochs} | batch={batch_size}")
    print("=" * 84)
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(train_ds_cached, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    model = FoundationPigReID(
        backbone_type=backbone_type,
        num_classes=len(classes),
        emb_dim=512,
        k=3,
        s=16.0,
        m=0.30,
        pretrained=True,
    ).to(device)
    model.configure_partial_freeze()

    # Frozen foundation teacher for Anchor-KD (prevents forgetting open-set unseen visual structure)
    teacher = ZeroShotFoundationWrapper(
        "vit_small_patch14_dinov2.lvd142m" if backbone_type == "dinov2_s14" else "hf-hub:BVRA/MegaDescriptor-T-224",
        num_classes=len(classes),
        img_size=224 if backbone_type == "dinov2_s14" else None,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Projection from teacher raw dim to student 512-D is unnecessary if we preserve relational cosine geometry!
    # Relational Anchor-KD: match pairwise cosine similarity matrix within each batch to frozen foundation teacher!
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.gem_pool.parameters())
        + list(model.proj.parameters())
        + [p for p in model.bnneck.parameters() if p.requires_grad]
        + list(model.head.parameters())
    )
    trainable_m = sum(p.numel() for p in backbone_params + head_params) / 1e6
    print(f"  Trainable params: {trainable_m:.2f}M (backbone={sum(p.numel() for p in backbone_params)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 4e-4},
        ],
        weight_decay=1e-4,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 2), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    triplet_fn = BatchHardTripletLoss(margin=0.35)
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

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, t_emb = teacher.extract_features(imgs)
                t_sim = torch.matmul(t_emb, t_emb.t())

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_tr, _, emb = model(imgs, lbls)
                loss_ce = ce_fn(logits_tr, lbls)
                loss_tri = triplet_fn(emb, lbls)
                # Relational Anchor-KD preserves foundation open-set similarity geometry
                s_sim = torch.matmul(emb, emb.t())
                loss_anchor = F.smooth_l1_loss(s_sim, t_sim)
                loss = loss_ce + 0.5 * loss_tri + 0.4 * loss_anchor

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += float(loss.item())

        scheduler.step()
        avg_loss = ep_loss / max(1, len(train_loader))

        logits_vc, feats_vc, lbl_vc = infer_dataset(model, None, device, cached_ds=val_clean_ds)
        logits_vo, feats_vo, lbl_vo = infer_dataset(model, None, device, cached_ds=val_occ_ds)
        acc_vc = float((torch.argmax(logits_vc, dim=1) == lbl_vc).float().mean().item())
        acc_vo = float((torch.argmax(logits_vo, dim=1) == lbl_vo).float().mean().item())
        _, _, margin_val = compute_cosine_separation(
            torch.cat([feats_vc, feats_vo], dim=0),
            torch.cat([lbl_vc, lbl_vo], dim=0),
        )
        score = 0.40 * acc_vc + 0.40 * acc_vo + 0.20 * max(0.0, margin_val)
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
                    "backbone_type": backbone_type,
                    "model": model.state_dict(),
                    "classes": classes,
                    "scale": 16.0,
                    "k": 3,
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
    print(f"Finished {exp_name} in {elapsed:.1f}s | Best Val Score={best_score:.4f}")
    return history, elapsed


def evaluate_adapted_model(name, model, classes, test_clean_items, test_occ_items, local_unknown_items, shaojiang_occ_items, device):
    print(f"\n--- Full 5-Benchmark Eval [{name}] ---")
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

    # Unknown rejection via both Softmax & Max-Cosine (logit / 16.0)
    logits_lu, _, _ = infer_dataset(model, local_unknown_items, device)
    cos_tc = torch.max(logits_tc / 16.0, dim=1)[0].numpy()
    cos_lu = torch.max(logits_lu / 16.0, dim=1)[0].numpy()
    prob_lu = torch.max(F.softmax(logits_lu, dim=1), dim=1)[0].numpy()

    sj_res = evaluate_shaojiang(model, device, shaojiang_occ_items)
    # Also compute SJ max-cosine unknown rejection
    sj_all = []
    for sub in ("clean_gallery", "clean_query"):
        sd = os.path.join(SHAOJIANG_ROOT, sub)
        for pid in sorted(os.listdir(sd), key=natural_key):
            for p in list_images_sorted(os.path.join(sd, pid)):
                sj_all.append((p, 0, pid))
    l_sj_all, _, _ = infer_dataset(model, sj_all, device)
    cos_sj = torch.max(l_sj_all / 16.0, dim=1)[0].numpy()

    cos_sweep = []
    for th in [0.40, 0.45, 0.50, 0.55, 0.60]:
        cos_sweep.append({
            "cos_thresh": th,
            "known_accept": float((cos_tc >= th).mean()),
            "local_unk_trr": float((cos_lu < th).mean()),
            "sj_unk_trr": float((cos_sj < th).mean()),
        })

    # Hard probes
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
    sims_p16 = torch.matmul(feats_h3[0], sj_cents := sj_centroids.t()).numpy()
    p_names = sj_res["primary_pigs"]
    order_p16 = np.argsort(-sims_p16)
    idx_p16 = p_names.index("pig_16")
    other_best = max(float(sims_p16[i]) for i in range(len(p_names)) if i != idx_p16)
    probe_p16 = {
        "top3": [(p_names[i], float(sims_p16[i])) for i in order_p16[:3]],
        "pig16_sim": float(sims_p16[idx_p16]),
        "margin_over_next": float(sims_p16[idx_p16]) - other_best,
    }

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

    res = {
        "closed_clean": {"n": len(lbl_tc), "acc": acc_tc, "mean_true_conf": true_conf_tc, "intra_cos": intra_tc, "inter_cos": inter_tc, "gap": gap_tc},
        "closed_occluded": {"n": len(lbl_to), "acc": acc_to, "mean_true_conf": true_conf_to, "intra_cos": intra_to, "inter_cos": inter_to, "gap": gap_to, "orig_vs_occ_cos": orig_vs_occ_cos},
        "local_unknown": {"n": len(prob_lu), "mean_prob": float(np.mean(prob_lu)), "trr_065": float(np.mean(prob_lu < 0.65)), "cos_sweep": cos_sweep},
        "shaojiang": sj_res,
        "hard_probe": {"295_6_vertical_fence": probe_295, "pig16_angled_rails": probe_p16, "occ_samples": sample_occ_probes},
    }
    print(
        f"  Closed Clean={acc_tc*100:.2f}% | Closed Occ={acc_to*100:.2f}% | "
        f"SJ Clean Rank1={sj_res['clean']['rank1']*100:.2f}% AUC={sj_res['clean']['auc']:.4f} EER={sj_res['clean']['eer']*100:.2f}% Gap={sj_res['clean']['gap']:+.4f}"
    )
    print(
        f"  SJ Occ Rank1={sj_res['occluded']['rank1']*100:.2f}% AUC={sj_res['occluded']['auc']:.4f} EER={sj_res['occluded']['eer']*100:.2f}% | "
        f"Hard 295-6={probe_295['prob_295_6']*100:.1f}% | p16_margin={probe_p16['margin_over_next']:+.4f}"
    )
    return res


def export_fp16_compact(ckpt_path, out_fp16_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_fp32 = FoundationPigReID(
        backbone_type=ckpt["backbone_type"],
        num_classes=len(ckpt["classes"]),
        emb_dim=512,
        k=ckpt.get("k", 3),
        s=ckpt.get("scale", 16.0),
        pretrained=False,
    )
    m_fp32.load_state_dict(ckpt["model"])
    m_fp32.eval().to(device)

    half_sd = {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in ckpt["model"].items()}
    torch.save({
        "backbone_type": ckpt["backbone_type"],
        "model": half_sd,
        "classes": ckpt["classes"],
        "scale": ckpt.get("scale", 16.0),
        "k": ckpt.get("k", 3),
        "precision": "fp16",
    }, out_fp16_path)

    m_fp16 = copy.deepcopy(m_fp32).half()
    x = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        _, f32 = m_fp32.extract_features(x)
        _, f16 = m_fp16.extract_features(x.half())
    max_diff = float((f32 - f16.float()).abs().max().item())

    # Latency benchmark
    x1 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(20):
            _ = m_fp16.extract_features(x1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = m_fp16.extract_features(x1)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / 100.0

    return {
        "fp32_mb": os.path.getsize(ckpt_path) / (1024 * 1024),
        "fp16_mb": os.path.getsize(out_fp16_path) / (1024 * 1024),
        "max_emb_diff": max_diff,
        "latency_fp16_ms": ms,
    }


def plot_v2_figures(v0_res, v1_res, zs_mega, zs_dino, exp_a, exp_b, best_key, best_res):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444", "axes.linewidth": 0.9})

    # Fig 1: Method Comparison
    methods = [
        ("v0 ResNet18", v0_res, "#718096"),
        ("v1 P0-Full", v1_res, "#3182ce"),
        ("ZS Mega-T", zs_mega, "#a0aec0"),
        ("ZS DINOv2-S", zs_dino, "#805ad5"),
        ("v2 Mega-T*", exp_a, "#dd6b20"),
        ("v2 DINOv2*", exp_b, "#16a34a"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=180)
    x = np.arange(len(methods))

    # (a) Shaojiang Clean & Occ Rank-1
    r1_c = [m[1]["shaojiang"]["clean"]["rank1"] * 100 for m in methods]
    r1_o = [m[1]["shaojiang"]["occluded"]["rank1"] * 100 for m in methods]
    w = 0.36
    b1 = axes[0].bar(x - w / 2, r1_c, width=w, label="SJ Clean Rank-1", color="#2b6cb0")
    b2 = axes[0].bar(x + w / 2, r1_o, width=w, label="SJ Occ Rank-1", color="#38a169")
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in methods], rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Rank-1 Accuracy (%)")
    axes[0].set_title("(a) External Unseen Pigs Rank-1 (Clean & Occ)", fontweight="bold")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # (b) Shaojiang Clean & Occ ROC-AUC
    auc_c = [m[1]["shaojiang"]["clean"]["auc"] * 100 for m in methods]
    auc_o = [m[1]["shaojiang"]["occluded"]["auc"] * 100 for m in methods]
    b1 = axes[1].bar(x - w / 2, auc_c, width=w, label="SJ Clean AUC", color="#4c51bf")
    b2 = axes[1].bar(x + w / 2, auc_o, width=w, label="SJ Occ AUC", color="#dd6b20")
    for b in list(b1) + list(b2):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(40, 108)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m[0] for m in methods], rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("ROC-AUC (%)")
    axes[1].set_title("(b) External Open-Set Verification ROC-AUC", fontweight="bold")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    # (c) Closed-Set Occluded Acc & Clean/Occ EER
    trained = [("v0 ResNet18", v0_res), ("v1 P0-Full", v1_res), ("v2 Mega-T*", exp_a), ("v2 DINOv2*", exp_b)]
    xt = np.arange(len(trained))
    c_occ = [m[1]["closed_occluded"]["acc"] * 100 for m in trained]
    eer_c = [m[1]["shaojiang"]["clean"]["eer"] * 100 for m in trained]
    eer_o = [m[1]["shaojiang"]["occluded"]["eer"] * 100 for m in trained]
    w3 = 0.25
    b1 = axes[2].bar(xt - w3, c_occ, width=w3, label="Closed Occ Acc (High=Best)", color="#16a34a")
    b2 = axes[2].bar(xt, eer_c, width=w3, label="SJ Clean EER (Low=Best)", color="#e53e3e")
    b3 = axes[2].bar(xt + w3, eer_o, width=w3, label="SJ Occ EER (Low=Best)", color="#ed8936")
    for b in list(b1) + list(b2) + list(b3):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_ylim(0, 112)
    axes[2].set_xticks(xt)
    axes[2].set_xticklabels([m[0] for m in trained], rotation=15, ha="right", fontsize=8.5)
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_title("(c) Closed Occluded Acc vs Open-Set EER", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"Track 1 (v2 Re-ID): Foundation Backbones + GeM + SubCenter AdaArcFace + Anchor-KD", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig1_method_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Cosine Histograms & ROC Curves (v0 vs v1 vs Best v2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    bins = np.linspace(-0.1, 1.0, 34)
    for ax, title, d_sj in [
        (axes[0, 0], f"v1 P0-Full Clean Cosine (Gap={v1_res['shaojiang']['clean']['gap']:+.3f})", v1_res["shaojiang"]["clean"]),
        (axes[0, 1], f"{best_key} Clean Cosine (Gap={best_res['shaojiang']['clean']['gap']:+.3f})", best_res["shaojiang"]["clean"]),
    ]:
        ax.hist(d_sj["inter_vals"], bins=bins, density=True, alpha=0.55, color="#e53e3e", label=f"Diff-Pig ({d_sj['inter_mean']:.3f})")
        ax.hist(d_sj["intra_vals"], bins=bins, density=True, alpha=0.60, color="#16a34a", label=f"Same-Pig ({d_sj['intra_mean']:.3f})")
        ax.axvline(d_sj["youden_thresh"], color="black", linestyle="--", lw=1.3, label=f"Youden {d_sj['youden_thresh']:.2f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3, linestyle="--")

    for ax, split_k, stitle in [
        (axes[1, 0], "clean", "External Shaojiang Clean ROC Curves"),
        (axes[1, 1], "occluded", "External Shaojiang Unseen-Fence Occluded ROC Curves"),
    ]:
        for lbl, res_d, col, ls in [
            ("v0 Baseline", v0_res, "#718096", ":"),
            ("v1 P0-Full", v1_res, "#3182ce", "--"),
            ("ZS MegaDescriptor-T", zs_mega, "#a0aec0", "-."),
            ("v2 Mega-T Adapted", exp_a, "#dd6b20", "-"),
            ("v2 DINOv2-S Adapted", exp_b, "#16a34a", "-"),
        ]:
            d = res_d["shaojiang"][split_k]
            ax.plot(d["fpr_curve"], d["tpr_curve"], color=col, linestyle=ls, lw=2.0, label=f"{lbl} (AUC={d['auc']:.3f}, EER={d['eer']*100:.1f}%)")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
        ax.set_title(stitle, fontweight="bold")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"Cosine Separation & Open-Set ROC Diagnostics (Winner: {best_key})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig2_cosine_and_roc.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 3: 2D PCA Cluster Comparison (v0 vs v1 vs Best v2)
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=180)
    cmap = plt.get_cmap("tab20")
    for ax, (lbl, r_d) in zip(axes, [("v0 Baseline", v0_res), ("v1 P0-Full", v1_res), (best_key, best_res)]):
        X = np.array(r_d["shaojiang"]["clean"]["query_feats"])
        pids = r_d["shaojiang"]["clean"]["query_pids"]
        uniq = sorted(list(set(pids)), key=natural_key)
        y_ids = np.array([uniq.index(p) for p in pids])
        sil = float(silhouette_score(X, y_ids, metric="cosine"))
        proj = PCA(n_components=2, random_state=42).fit_transform(X)
        for i, p in enumerate(uniq):
            m = y_ids == i
            ax.scatter(proj[m, 0], proj[m, 1], color=cmap(i), label=p, s=48, alpha=0.88, edgecolors="white", linewidth=0.5)
        ax.set_title(f"{lbl}\nCosine Silhouette = {sil:+.3f} | Rank-1 = {r_d['shaojiang']['clean']['rank1']*100:.1f}%", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(alpha=0.3, linestyle="--")
    axes[2].legend(ncol=3, fontsize=7.5, loc="best")
    fig.suptitle("2D PCA of 11 Unseen External Pigs (Zero-Leakage Shaojiang Query Set)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig3_pca_clusters.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Hard Samples Probe (v0 vs v1 vs Best v2)
    b_hp, v1_hp, f_hp = v0_res["hard_probe"], v1_res["hard_probe"], best_res["hard_probe"]
    cards = [
        (
            os.path.join(HARD_SAMPLES_DIR, "01_reid_295-6_vertical_fence.png"),
            "AI Hard Sample: 295-6 + Vertical Fence",
            f"v0: {b_hp['295_6_vertical_fence']['pred_id']} ({b_hp['295_6_vertical_fence']['prob_295_6']*100:.1f}%)\n"
            f"v1: {v1_hp['295_6_vertical_fence']['pred_id']} ({v1_hp['295_6_vertical_fence']['prob_295_6']*100:.1f}%)\n"
            f"v2: {f_hp['295_6_vertical_fence']['pred_id']} ({f_hp['295_6_vertical_fence']['prob_295_6']*100:.1f}%)",
        ),
        (
            os.path.join(HARD_SAMPLES_DIR, "03_reid_pig16_angled_rails.png"),
            "Unseen Open-Set: pig_16 + Angled Rails",
            f"v0: {b_hp['pig16_angled_rails']['top3'][0][0]} (margin={b_hp['pig16_angled_rails']['margin_over_next']:+.3f})\n"
            f"v1: {v1_hp['pig16_angled_rails']['top3'][0][0]} (margin={v1_hp['pig16_angled_rails']['margin_over_next']:+.3f})\n"
            f"v2: {f_hp['pig16_angled_rails']['top3'][0][0]} (margin={f_hp['pig16_angled_rails']['margin_over_next']:+.3f})",
        ),
    ]
    for bi, v1i, fi in zip(b_hp["occ_samples"], v1_hp["occ_samples"], f_hp["occ_samples"]):
        cards.append((
            fi["path"],
            f"Unseen Test Fence: True={fi['true_cls']}",
            f"v0: {bi['pred_cls']} ({bi['true_prob']*100:.1f}%)\n"
            f"v1: {v1i['pred_cls']} ({v1i['true_prob']*100:.1f}%)\n"
            f"v2: {fi['pred_cls']} ({fi['true_prob']*100:.1f}%)",
        ))

    fig, axes = plt.subplots(1, len(cards), figsize=(16, 5.2), dpi=180)
    for ax, (im_p, title, sub) in zip(axes, cards):
        ax.imshow(Image.open(im_p).convert("RGB").resize((280, 280)))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(sub, fontsize=8.8, fontweight="bold", color="#1a202c", labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Before-vs-After Occluded Identity Probe (v0 vs v1 vs {best_key})", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig4_hard_samples.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    t_start = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Track 1 (v2 Foundation Re-ID) on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    base_ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    classes = list(base_ckpt["classes"])

    with open(P0_RESULTS_JSON, "r", encoding="utf-8") as f:
        p0_data = json.load(f)
    v0_res = p0_data["results"]["Baseline"]
    v1_res = p0_data["results"]["P0-Full"]

    train_items, val_items, test_items, local_unknown_items, manifest = build_chronological_split(classes)
    train_overlays = load_overlays(TRAIN_OVERLAY_NAMES)
    unseen_overlays = load_overlays(TEST_UNSEEN_OVERLAY_NAMES)

    val_occ_items, _ = build_deterministic_occluded_set(val_items, "val_occ", unseen_overlays, seed=202601)
    test_occ_items, _ = build_deterministic_occluded_set(test_items, "test_occ", unseen_overlays, seed=202602)
    sj_qry_dir = os.path.join(SHAOJIANG_ROOT, "clean_query")
    sj_qry_all = [(p, 0, pid) for pid in sorted(os.listdir(sj_qry_dir), key=natural_key) for p in list_images_sorted(os.path.join(sj_qry_dir, pid))]
    sj_occ_items, _ = build_deterministic_occluded_set(sj_qry_all, "sj_qry_occ", unseen_overlays, seed=202603)

    # Phase 1: Zero-Shot Foundation Benchmarks
    zs_mega_m = ZeroShotFoundationWrapper("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=len(classes)).to(device)
    zs_mega_res = eval_zeroshot_model("ZS MegaDescriptor-T-224", zs_mega_m, classes, train_items, test_items, test_occ_items, sj_occ_items, device)
    del zs_mega_m
    torch.cuda.empty_cache()

    zs_dino_m = ZeroShotFoundationWrapper("vit_small_patch14_dinov2.lvd142m", num_classes=len(classes), img_size=224).to(device)
    zs_dino_res = eval_zeroshot_model("ZS DINOv2-ViT-S/14", zs_dino_m, classes, train_items, test_items, test_occ_items, sj_occ_items, device)
    del zs_dino_m
    torch.cuda.empty_cache()

    # Phase 2: RAM Cache + Adapted Training (Exp A: MegaDescriptor-T, Exp B: DINOv2-S/14)
    train_tfms = transforms.Compose([
        RandomAIFenceOcclusion(train_overlays, p=0.60, occ_range=(0.15, 0.40), seed=202),
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

    ckpt_a = os.path.join(OUT_ROOT, "pig_reid_v2_mega_t_best.pth")
    ckpt_b = os.path.join(OUT_ROOT, "pig_reid_v2_dinov2_s_best.pth")

    hist_a, time_a = train_foundation_reid(
        "Exp A (MegaDescriptor-T + GeM + SubCenter-AdaArcFace + Triplet + AnchorKD)",
        "megadescriptor_t", train_ds_cached, val_clean_ds, val_occ_ds, classes, device, ckpt_a, epochs=18, batch_size=48
    )
    hist_b, time_b = train_foundation_reid(
        "Exp B (DINOv2-S/14 + GeM + SubCenter-AdaArcFace + Triplet + AnchorKD)",
        "dinov2_s14", train_ds_cached, val_clean_ds, val_occ_ds, classes, device, ckpt_b, epochs=18, batch_size=48
    )

    # Phase 3: Full Evaluation
    ck_a = torch.load(ckpt_a, map_location="cpu", weights_only=False)
    m_a = FoundationPigReID("megadescriptor_t", len(classes), pretrained=False).to(device)
    m_a.load_state_dict(ck_a["model"])
    res_a = evaluate_adapted_model("v2 MegaDescriptor-T*", m_a, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device)
    del m_a
    torch.cuda.empty_cache()

    ck_b = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    m_b = FoundationPigReID("dinov2_s14", len(classes), pretrained=False).to(device)
    m_b.load_state_dict(ck_b["model"])
    res_b = evaluate_adapted_model("v2 DINOv2-S*", m_b, classes, test_items, test_occ_items, local_unknown_items, sj_occ_items, device)
    del m_b
    torch.cuda.empty_cache()

    # Select winner by composite open-set + occlusion score
    def _rank_score(r):
        return (
            0.30 * r["closed_occluded"]["acc"]
            + 0.25 * r["shaojiang"]["clean"]["auc"]
            + 0.25 * r["shaojiang"]["occluded"]["auc"]
            + 0.20 * r["shaojiang"]["clean"]["rank1"]
        )

    if _rank_score(res_b) >= _rank_score(res_a):
        best_key, best_res, best_ckpt = "v2 DINOv2-S*", res_b, ckpt_b
    else:
        best_key, best_res, best_ckpt = "v2 Mega-T*", res_a, ckpt_a

    # Export FP16 compact checkpoints for both (and winner explicitly)
    fp16_dino_info = export_fp16_compact(ckpt_b, os.path.join(OUT_ROOT, "pig_reid_v2_dinov2_s_fp16.pth"), device)
    fp16_mega_info = export_fp16_compact(ckpt_a, os.path.join(OUT_ROOT, "pig_reid_v2_mega_t_fp16.pth"), device)
    best_fp16_info = fp16_dino_info if best_key == "v2 DINOv2-S*" else fp16_mega_info

    # Check Branch Gate for v2
    beats_v0_v1_open = (
        best_res["shaojiang"]["clean"]["auc"] > max(v0_res["shaojiang"]["clean"]["auc"], v1_res["shaojiang"]["clean"]["auc"])
        or best_res["shaojiang"]["occluded"]["auc"] > max(v0_res["shaojiang"]["occluded"]["auc"], v1_res["shaojiang"]["occluded"]["auc"])
    ) and best_res["shaojiang"]["clean"]["rank1"] > v1_res["shaojiang"]["clean"]["rank1"]
    gate_passed = beats_v0_v1_open and (best_res["closed_occluded"]["acc"] >= 0.94)

    summary = {
        "elapsed_sec": time.time() - t_start,
        "train_times_sec": {"Exp_A_MegaT": time_a, "Exp_B_DINOv2S": time_b},
        "best_model": best_key,
        "gate_passed": gate_passed,
        "fp16_export": {"dinov2_s": fp16_dino_info, "mega_t": fp16_mega_info, "winner": best_fp16_info},
        "results": {
            "v0_baseline": v0_res,
            "v1_p0_full": v1_res,
            "zs_megadescriptor_t": zs_mega_res,
            "zs_dinov2_s": zs_dino_res,
            "exp_a_mega_t": res_a,
            "exp_b_dinov2_s": res_b,
        },
        "histories": {"exp_a": hist_a, "exp_b": hist_b},
    }
    with open(os.path.join(OUT_ROOT, "v2_reid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "=" * 102,
        "Track 1 (v2) Foundation + GeM + SubCenter-AdaArcFace + Anchor-KD Re-ID Report",
        "=" * 102,
        f"Winner: {best_key} | Branch Gate Passed: {gate_passed} | Total Elapsed: {summary['elapsed_sec']:.1f}s",
        f"FP16 Export: DINOv2-S={fp16_dino_info['fp16_mb']:.2f}MB ({fp16_dino_info['latency_fp16_ms']:.2f}ms, diff={fp16_dino_info['max_emb_diff']:.2e}) | "
        f"Mega-T={fp16_mega_info['fp16_mb']:.2f}MB ({fp16_mega_info['latency_fp16_ms']:.2f}ms, diff={fp16_mega_info['max_emb_diff']:.2e})",
        "",
        f"{'Metric':<38} | {'v0 Base':>9} | {'v1 P0':>9} | {'ZS MegaT':>9} | {'ZS DINO':>9} | {'v2 MegaT':>9} | {'v2 DINO':>9}",
        "-" * 104,
        f"{'Closed Clean Acc':<38} | {v0_res['closed_clean']['acc']*100:>8.2f}% | {v1_res['closed_clean']['acc']*100:>8.2f}% | {zs_mega_res['closed_clean']['acc']*100:>8.2f}% | {zs_dino_res['closed_clean']['acc']*100:>8.2f}% | {res_a['closed_clean']['acc']*100:>8.2f}% | {res_b['closed_clean']['acc']*100:>8.2f}%",
        f"{'Closed Unseen-Fence Acc':<38} | {v0_res['closed_occluded']['acc']*100:>8.2f}% | {v1_res['closed_occluded']['acc']*100:>8.2f}% | {zs_mega_res['closed_occluded']['acc']*100:>8.2f}% | {zs_dino_res['closed_occluded']['acc']*100:>8.2f}% | {res_a['closed_occluded']['acc']*100:>8.2f}% | {res_b['closed_occluded']['acc']*100:>8.2f}%",
        f"{'SJ Clean Top-5 Rank-1':<38} | {v0_res['shaojiang']['clean']['rank1']*100:>8.2f}% | {v1_res['shaojiang']['clean']['rank1']*100:>8.2f}% | {zs_mega_res['shaojiang']['clean']['rank1']*100:>8.2f}% | {zs_dino_res['shaojiang']['clean']['rank1']*100:>8.2f}% | {res_a['shaojiang']['clean']['rank1']*100:>8.2f}% | {res_b['shaojiang']['clean']['rank1']*100:>8.2f}%",
        f"{'SJ Clean ROC-AUC':<38} | {v0_res['shaojiang']['clean']['auc']:>9.4f} | {v1_res['shaojiang']['clean']['auc']:>9.4f} | {zs_mega_res['shaojiang']['clean']['auc']:>9.4f} | {zs_dino_res['shaojiang']['clean']['auc']:>9.4f} | {res_a['shaojiang']['clean']['auc']:>9.4f} | {res_b['shaojiang']['clean']['auc']:>9.4f}",
        f"{'SJ Clean EER (lower=better)':<38} | {v0_res['shaojiang']['clean']['eer']*100:>8.2f}% | {v1_res['shaojiang']['clean']['eer']*100:>8.2f}% | {zs_mega_res['shaojiang']['clean']['eer']*100:>8.2f}% | {zs_dino_res['shaojiang']['clean']['eer']*100:>8.2f}% | {res_a['shaojiang']['clean']['eer']*100:>8.2f}% | {res_b['shaojiang']['clean']['eer']*100:>8.2f}%",
        f"{'SJ Clean Cos Gap Delta':<38} | {v0_res['shaojiang']['clean']['gap']:>+9.4f} | {v1_res['shaojiang']['clean']['gap']:>+9.4f} | {zs_mega_res['shaojiang']['clean']['gap']:>+9.4f} | {zs_dino_res['shaojiang']['clean']['gap']:>+9.4f} | {res_a['shaojiang']['clean']['gap']:>+9.4f} | {res_b['shaojiang']['clean']['gap']:>+9.4f}",
        f"{'SJ Occluded Top-5 Rank-1':<38} | {v0_res['shaojiang']['occluded']['rank1']*100:>8.2f}% | {v1_res['shaojiang']['occluded']['rank1']*100:>8.2f}% | {zs_mega_res['shaojiang']['occluded']['rank1']*100:>8.2f}% | {zs_dino_res['shaojiang']['occluded']['rank1']*100:>8.2f}% | {res_a['shaojiang']['occluded']['rank1']*100:>8.2f}% | {res_b['shaojiang']['occluded']['rank1']*100:>8.2f}%",
        f"{'SJ Occluded ROC-AUC':<38} | {v0_res['shaojiang']['occluded']['auc']:>9.4f} | {v1_res['shaojiang']['occluded']['auc']:>9.4f} | {zs_mega_res['shaojiang']['occluded']['auc']:>9.4f} | {zs_dino_res['shaojiang']['occluded']['auc']:>9.4f} | {res_a['shaojiang']['occluded']['auc']:>9.4f} | {res_b['shaojiang']['occluded']['auc']:>9.4f}",
        f"{'SJ Occluded EER (lower=better)':<38} | {v0_res['shaojiang']['occluded']['eer']*100:>8.2f}% | {v1_res['shaojiang']['occluded']['eer']*100:>8.2f}% | {zs_mega_res['shaojiang']['occluded']['eer']*100:>8.2f}% | {zs_dino_res['shaojiang']['occluded']['eer']*100:>8.2f}% | {res_a['shaojiang']['occluded']['eer']*100:>8.2f}% | {res_b['shaojiang']['occluded']['eer']*100:>8.2f}%",
        f"{'Hard Probe 295-6 Fence Prob':<38} | {v0_res['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>8.2f}% | {v1_res['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>8.2f}% |       n/a |       n/a | {res_a['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>8.2f}% | {res_b['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>8.2f}%",
        f"{'Hard Probe pig_16 Rails Margin':<38} | {v0_res['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f} | {v1_res['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f} | {zs_mega_res['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f} | {zs_dino_res['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f} | {res_a['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f} | {res_b['hard_probe']['pig16_angled_rails']['margin_over_next']:>+9.4f}",
        "",
        "Max-Cosine Unknown Rejection Sweep (Winner):",
    ]
    for sw in best_res["local_unknown"]["cos_sweep"]:
        lines.append(
            f"  cos >= {sw['cos_thresh']:.2f} -> Known Accept={sw['known_accept']*100:5.2f}% | "
            f"LocalUnk TRR={sw['local_unk_trr']*100:5.2f}% | SJ Unk TRR={sw['sj_unk_trr']*100:5.2f}%"
        )
    lines.append("=" * 102)
    rep_txt = "\n".join(lines)
    with open(os.path.join(OUT_ROOT, "v2_reid_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt + "\n")
    print("\n" + rep_txt)

    plot_v2_figures(v0_res, v1_res, zs_mega_res, zs_dino_res, res_a, res_b, best_key, best_res)
    print("All 4 v2 diagnostic figures saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
