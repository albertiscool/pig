import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import sys
import re
import json
import shutil
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


PROJECT_ROOT = r"C:\Users\User\Desktop\final_pig_v0"
V4_DIR = os.path.join(PROJECT_ROOT, "v4")
DET_V4_PATH = os.path.join(V4_DIR, "parameters", "nose_eyes_face_0817.pt")
DET_FACE_PT_PATH = os.path.join(PROJECT_ROOT, "pig_id_package", "models", "face.pt")
REID_MODEL_PATH = os.path.join(V4_DIR, "parameters", "pig_ecb_best.pth")

BASE_DIR = os.path.join(PROJECT_ROOT, "external_test_datasets", "shaojiang_pig_faces")
META_PATH = os.path.join(BASE_DIR, "crops_metadata.json")
CLEAN_GALLERY_DIR = os.path.join(BASE_DIR, "clean_gallery")
CLEAN_QUERY_DIR = os.path.join(BASE_DIR, "clean_query")
GRID_PATH = os.path.join(BASE_DIR, "clean_crops_grid.jpg")
PLOT_PATH = os.path.join(BASE_DIR, "clean_evaluation_plots.png")
REPORT_PATH = os.path.join(BASE_DIR, "clean_evaluation_report.txt")


def natural_key(path_str):
    parts = re.split(r"(\d+)", path_str)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def pad_gray(img, ratio=0.5):
    h, w = img.shape[:2]
    ph, pw = int(h * ratio), int(w * ratio)
    return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))


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


def build_clean_contact_sheet(clean_data, primary_pigs):
    tiles = []
    tile_w, tile_h = 180, 180
    bar_h = 32

    for pid in primary_pigs:
        g_sorted = sorted(clean_data["gallery"][pid], key=lambda x: x["sharpness"], reverse=True)[:2]
        q_sorted = sorted(clean_data["query"][pid], key=lambda x: x["sharpness"], reverse=True)[:2]
        row_items = [("G", it) for it in g_sorted] + [("Q", it) for it in q_sorted]
        for role_tag, it in row_items:
            img = cv2.imread(it["clean_path"])
            if img is None:
                continue
            resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((tile_h + bar_h, tile_w, 3), dtype=np.uint8)
            canvas[:tile_h, :] = resized
            canvas[tile_h:, :] = (28, 28, 28)
            txt = f"{pid} [{role_tag}] n={it['nose_conf']:.2f} f={it['face_pt_conf']:.2f}"
            cv2.putText(canvas, txt, (4, tile_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(canvas)

    if not tiles:
        return

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    cell_h, cell_w = tile_h + bar_h, tile_w
    pad = 6
    grid = np.full((rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3), 15, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = tile
    cv2.imwrite(GRID_PATH, grid)
    print(f"  Saved visual grid to {GRID_PATH} ({len(tiles)} tiles)")


def filter_crops(exclude_pigs, require_nose_only=False):
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shutil.rmtree(CLEAN_GALLERY_DIR, ignore_errors=True)
    shutil.rmtree(CLEAN_QUERY_DIR, ignore_errors=True)
    os.makedirs(CLEAN_GALLERY_DIR, exist_ok=True)
    os.makedirs(CLEAN_QUERY_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_v4 = YOLO(DET_V4_PATH)
    det_v4.to(device)
    det_face = YOLO(DET_FACE_PT_PATH)
    det_face.to(device)

    clean_data = {"gallery": {}, "query": {}}
    raw_count = 0
    kept_count = 0

    for role, target_root in [("gallery", CLEAN_GALLERY_DIR), ("query", CLEAN_QUERY_DIR)]:
        for pid in sorted(meta[role].keys(), key=natural_key):
            if pid in exclude_pigs:
                continue
            clean_data[role][pid] = []
            for it in meta[role][pid]:
                raw_count += 1
                src_path = it["path"]
                img = cv2.imread(src_path)
                if img is None:
                    continue
                pimg = pad_gray(img, 0.5)
                r1 = det_v4.predict(pimg, imgsz=320, verbose=False)[0]
                nose_c = max([float(b.conf) for b in r1.boxes if det_v4.names[int(b.cls)] == "nose"] + [0.0])
                eye_c = max([float(b.conf) for b in r1.boxes if det_v4.names[int(b.cls)] == "eye"] + [0.0])
                r2 = det_face.predict(pimg, imgsz=320, verbose=False)[0]
                fpt_c = max([float(b.conf) for b in r2.boxes if det_face.names[int(b.cls)] == "face"] + [0.0])

                if require_nose_only:
                    passed = nose_c >= 0.30
                else:
                    passed = (nose_c >= 0.30) or (eye_c >= 0.30 and fpt_c >= 0.45)

                if passed:
                    pig_out_dir = os.path.join(target_root, pid)
                    os.makedirs(pig_out_dir, exist_ok=True)
                    dst_path = os.path.join(pig_out_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst_path)
                    kept_count += 1
                    entry = dict(it)
                    entry["clean_path"] = dst_path
                    entry["nose_conf"] = round(nose_c, 4)
                    entry["eye_conf"] = round(eye_c, 4)
                    entry["face_pt_conf"] = round(fpt_c, 4)
                    clean_data[role][pid].append(entry)

    print(f"[1/4] Filtered {raw_count} raw crops -> {kept_count} landmark-verified clean crops (excluded: {sorted(exclude_pigs) if exclude_pigs else 'none'})")
    return clean_data, raw_count, kept_count


def evaluate_clean(clean_data, raw_count, kept_count, exclude_pigs, require_nose_only):
    sys.path.insert(0, V4_DIR)
    from pig_identity import ECBHybridModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = (device.type == "cuda")

    ckpt = torch.load(REID_MODEL_PATH, map_location=device, weights_only=False)
    classes = ckpt.get("classes", [])
    num_classes = len(classes) if len(classes) > 0 else 17

    model = ECBHybridModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if use_half:
        model.half()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def infer_crop(img_path):
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tfm(rgb).unsqueeze(0).to(device)
        if use_half:
            tensor = tensor.half()
        with torch.no_grad():
            logits, feat_norm = model.extract_features(tensor)
            probs = F.softmax(logits.float(), dim=1)[0]
            max_prob = float(torch.max(probs).item())
            top_idx = int(torch.argmax(probs).item())
            pred_cls = classes[top_idx] if top_idx < len(classes) else f"Class_{top_idx}"
        return max_prob, pred_cls, feat_norm[0].cpu()

    print("[2/4] Running ECBHybridModel (FP16) on landmark-verified clean crops...")
    all_max_probs = []
    gallery_feats = {}
    query_feats = {}

    for pid, items in clean_data["gallery"].items():
        gallery_feats[pid] = []
        for it in items:
            mp, pc, feat = infer_crop(it["clean_path"])
            all_max_probs.append(mp)
            gallery_feats[pid].append({
                "feat": feat,
                "sharpness": it["sharpness"],
                "path": it["clean_path"],
                "max_prob": mp,
            })

    for pid, items in clean_data["query"].items():
        query_feats[pid] = []
        for it in items:
            mp, pc, feat = infer_crop(it["clean_path"])
            all_max_probs.append(mp)
            query_feats[pid].append({
                "feat": feat,
                "sharpness": it["sharpness"],
                "path": it["clean_path"],
                "split": it["split"],
                "max_prob": mp,
            })

    primary_pigs = sorted(
        [
            pid for pid in clean_data["gallery"]
            if len(gallery_feats.get(pid, [])) >= 1 and len(query_feats.get(pid, [])) >= 1
        ],
        key=natural_key,
    )
    multishot_pigs = sorted(
        [
            pid for pid in primary_pigs
            if len(gallery_feats[pid]) >= 2 and len(query_feats[pid]) >= 2
        ],
        key=natural_key,
    )

    print(f"  Primary clean pigs (>=1 G & >=1 Q): {len(primary_pigs)} -> {primary_pigs}")
    print(f"  Multi-shot clean pigs (>=2 G & >=2 Q): {len(multishot_pigs)} -> {multishot_pigs}")
    if len(primary_pigs) < 5:
        raise RuntimeError(f"CLEAN_PIGS_TOO_FEW: {len(primary_pigs)} (< 5)")

    build_clean_contact_sheet(clean_data, primary_pigs)

    # --- Evaluation A: Closed-Set Rejection ---
    probs_arr = np.asarray(all_max_probs, dtype=np.float64)
    far_065 = float(np.mean(probs_arr >= 0.65))
    trr_065 = 1.0 - far_065
    far_050 = float(np.mean(probs_arr >= 0.50))
    trr_050 = 1.0 - far_050

    def evaluate_subset(pid_list):
        top5_centroids = {}
        all_centroids = {}
        all_gallery_shots = []

        for pid in pid_list:
            g_sorted = sorted(gallery_feats[pid], key=lambda x: x["sharpness"], reverse=True)
            top5_stack = torch.stack([x["feat"] for x in g_sorted[:5]], dim=0)
            top5_centroids[pid] = F.normalize(torch.mean(top5_stack, dim=0, keepdim=True), p=2, dim=1)[0].numpy()

            all_stack = torch.stack([x["feat"] for x in g_sorted], dim=0)
            all_centroids[pid] = F.normalize(torch.mean(all_stack, dim=0, keepdim=True), p=2, dim=1)[0].numpy()

            for x in g_sorted:
                all_gallery_shots.append((pid, x["feat"].numpy()))

        mat_top5 = np.stack([top5_centroids[p] for p in pid_list], axis=0)
        mat_all = np.stack([all_centroids[p] for p in pid_list], axis=0)
        mat_shots = np.stack([s[1] for s in all_gallery_shots], axis=0)
        shot_pids = [s[0] for s in all_gallery_shots]

        def eval_ranking(centroids_mat):
            top1_ok = 0
            top3_ok = 0
            total_q = 0
            per_pig = {}
            same_sims = []
            diff_sims = []
            pair_scores = []
            pair_labels = []

            for gt_idx, pid in enumerate(pid_list):
                q_items = query_feats[pid]
                p_ok1 = 0
                p_ok3 = 0
                for q in q_items:
                    qf = q["feat"].numpy()
                    sims = centroids_mat @ qf
                    ranked = np.argsort(-sims)
                    if ranked[0] == gt_idx:
                        top1_ok += 1
                        p_ok1 += 1
                    if gt_idx in ranked[:min(3, len(pid_list))]:
                        top3_ok += 1
                        p_ok3 += 1
                    total_q += 1

                    for c_idx in range(len(pid_list)):
                        s_val = float(sims[c_idx])
                        is_same = (c_idx == gt_idx)
                        pair_scores.append(s_val)
                        pair_labels.append(1 if is_same else 0)
                        if is_same:
                            same_sims.append(s_val)
                        else:
                            diff_sims.append(s_val)

                per_pig[pid] = {
                    "n_gallery": len(gallery_feats[pid]),
                    "n_query": len(q_items),
                    "top1": p_ok1,
                    "top3": p_ok3,
                    "acc1": p_ok1 / len(q_items),
                }
            return {
                "total_q": total_q,
                "top1_ok": top1_ok,
                "top3_ok": top3_ok,
                "top1_acc": top1_ok / total_q,
                "top3_acc": top3_ok / total_q,
                "per_pig": per_pig,
                "same_sims": np.asarray(same_sims),
                "diff_sims": np.asarray(diff_sims),
                "pair_scores": np.asarray(pair_scores),
                "pair_labels": np.asarray(pair_labels),
            }

        res_top5 = eval_ranking(mat_top5)
        res_all = eval_ranking(mat_all)

        nn_top1_ok = 0
        nn_top3_ok = 0
        nn_total_q = 0
        shot_same = []
        shot_diff = []
        shot_scores = []
        shot_labels = []

        for pid in pid_list:
            for q in query_feats[pid]:
                qf = q["feat"].numpy()
                sims = mat_shots @ qf
                max_per_pid = {}
                for s_idx, s_pid in enumerate(shot_pids):
                    val = float(sims[s_idx])
                    shot_scores.append(val)
                    if s_pid == pid:
                        shot_labels.append(1)
                        shot_same.append(val)
                    else:
                        shot_labels.append(0)
                        shot_diff.append(val)
                    if s_pid not in max_per_pid or val > max_per_pid[s_pid]:
                        max_per_pid[s_pid] = val
                ranked_pids = [k for k, _ in sorted(max_per_pid.items(), key=lambda kv: -kv[1])]
                if ranked_pids[0] == pid:
                    nn_top1_ok += 1
                if pid in ranked_pids[:min(3, len(pid_list))]:
                    nn_top3_ok += 1
                nn_total_q += 1

        roc_c = compute_roc_metrics(res_top5["pair_scores"], res_top5["pair_labels"])
        roc_s = compute_roc_metrics(shot_scores, shot_labels)

        return {
            "res_top5": res_top5,
            "res_all": res_all,
            "nn_top1_ok": nn_top1_ok,
            "nn_top3_ok": nn_top3_ok,
            "nn_total_q": nn_total_q,
            "roc_c": roc_c,
            "roc_s": roc_s,
            "shot_same": np.asarray(shot_same),
            "shot_diff": np.asarray(shot_diff),
        }

    prim = evaluate_subset(primary_pigs)
    multi = evaluate_subset(multishot_pigs) if len(multishot_pigs) >= 3 else None

    res_top5 = prim["res_top5"]
    res_all = prim["res_all"]
    fpr_c, tpr_c, _, auc_c, eer_c, eer_th_c, youden_th_c = prim["roc_c"]
    fpr_s, tpr_s, _, auc_s, eer_s, eer_th_s, _ = prim["roc_s"]

    same_c = res_top5["same_sims"]
    diff_c = res_top5["diff_sims"]
    tpr_082_c = float(np.mean(same_c >= 0.82))
    fpr_082_c = float(np.mean(diff_c >= 0.82))
    tpr_075_c = float(np.mean(same_c >= 0.75))
    fpr_075_c = float(np.mean(diff_c >= 0.75))

    same_s = prim["shot_same"]
    diff_s = prim["shot_diff"]
    tpr_082_s = float(np.mean(same_s >= 0.82))
    fpr_082_s = float(np.mean(diff_s >= 0.82))
    tpr_075_s = float(np.mean(same_s >= 0.75))
    fpr_075_s = float(np.mean(diff_s >= 0.75))

    # --- Visualization Plot ---
    print("[3/4] Generating clean evaluation plots and report...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)

    ax = axes[0]
    ax.hist(diff_c, bins=25, density=True, alpha=0.55, color="#d95f02", label=f"Different Pig (n={len(diff_c)})")
    ax.hist(same_c, bins=20, density=True, alpha=0.65, color="#1b9e77", label=f"Same Pig (n={len(same_c)})")
    ax.axvline(0.82, color="#e41a1c", linestyle="--", linewidth=1.8, label="Prod thresh = 0.82")
    ax.axvline(0.75, color="#377eb8", linestyle=":", linewidth=1.8, label="Pkg thresh = 0.75")
    ax.axvline(eer_th_c, color="#000000", linestyle="-.", linewidth=1.5, label=f"EER thresh = {eer_th_c:.3f}")
    ax.set_title("Clean Open-Set Re-ID Cosine Similarity\n(Query vs Top-5 Gallery Centroid)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(fpr_c, tpr_c, color="#1b9e77", linewidth=2.2, label=f"Centroid ROC (AUC={auc_c:.4f}, EER={eer_c*100:.2f}%)")
    ax.plot(fpr_s, tpr_s, color="#7570b3", linewidth=1.8, linestyle="--", label=f"Shot-to-Shot ROC (AUC={auc_s:.4f}, EER={eer_s*100:.2f}%)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0)
    ax.scatter([fpr_082_c], [tpr_082_c], color="#e41a1c", zorder=5, s=55, label=f"Centroid @0.82 (TPR={tpr_082_c*100:.1f}%, FPR={fpr_082_c*100:.1f}%)")
    ax.set_title("Clean Open-Set Verification ROC Curve")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.hist(probs_arr, bins=25, color="#377eb8", alpha=0.75, edgecolor="white")
    ax.axvline(0.65, color="#e41a1c", linestyle="--", linewidth=1.8, label=f"Prod thresh = 0.65 (FAR={far_065*100:.1f}%)")
    ax.axvline(0.50, color="#ff7f00", linestyle=":", linewidth=1.8, label=f"Pkg thresh = 0.50 (FAR={far_050*100:.1f}%)")
    ax.set_title(f"Closed-Set Rejection (17 Known Classes)\nClean Unknown External Faces (n={len(probs_arr)})")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Crop Count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    # --- Build Text Report ---
    tot_clean_g = sum(len(v) for v in clean_data["gallery"].values())
    tot_clean_q = sum(len(v) for v in clean_data["query"].values())
    prim_g = sum(len(gallery_feats[p]) for p in primary_pigs)
    prim_q = sum(len(query_feats[p]) for p in primary_pigs)
    prim_q_test = sum(1 for p in primary_pigs for q in query_feats[p] if q["split"] == "test_img")
    prim_q_val = sum(1 for p in primary_pigs for q in query_feats[p] if q["split"] == "val_img")

    rule_desc = "nose_conf>=0.30" if require_nose_only else "(nose_conf>=0.30 OR (eye_conf>=0.30 AND face.pt>=0.45)) on 50%-padded crop"

    lines = [
        "================================================================================",
        "Clean Landmark-Verified External Pig Face Recognition Evaluation Report",
        "================================================================================",
        f"Dataset Source     : Hugging Face `shaojiang/pig-images` (external unseen individuals)",
        f"Clean Filter Rule  : {rule_desc}",
        f"Excluded Pigs      : {sorted(exclude_pigs) if exclude_pigs else 'None'}",
        f"Re-ID Model        : {REID_MODEL_PATH} (ECBHybridModel ResNet18 inference branch, classes={num_classes})",
        f"Device / Precision : {device} ({'FP16' if use_half else 'FP32'})",
        "",
        "--- 1. Clean Extraction Summary ---",
        f"Raw YOLO face crops scanned : {raw_count}",
        f"All landmark-clean crops    : {kept_count} (Gallery={tot_clean_g}, Query={tot_clean_q})",
        f"Primary Re-ID test pigs     : {len(primary_pigs)} pigs (each >= 1 gallery & >= 1 query crop)",
        f"Primary crop counts         : Gallery={prim_g} (train_img), Query={prim_q} ({prim_q_test} from test_img, {prim_q_val} from val_img)",
        "",
        "--- 2. Evaluation A: Open-Set Unknown Rejection (17 Known Training Classes vs Clean Unseen Pigs) ---",
        f"Clean external face crops   : {len(probs_arr)}",
        f"Max Softmax Prob (Mean/Med) : {np.mean(probs_arr):.4f} / {np.median(probs_arr):.4f}",
        f"Max Softmax Prob (P90/Max)  : {np.percentile(probs_arr, 90):.4f} / {np.max(probs_arr):.4f}",
        f"At conf_threshold = 0.65    : False Accept Rate (FAR) = {far_065*100:.2f}% | True Rejection Rate (TRR) = {trr_065*100:.2f}%",
        f"At conf_threshold = 0.50    : False Accept Rate (FAR) = {far_050*100:.2f}% | True Rejection Rate (TRR) = {trr_050*100:.2f}%",
        "",
        "--- 3. Evaluation B: Open-Set Re-ID Generalization (Primary Set: >=1 Gallery & >=1 Query) ---",
        f"Unseen gallery IDs          : {len(primary_pigs)}",
        f"Total query crops evaluated : {res_top5['total_q']}",
        f"[Mode 1] Top-5 Sharpness Centroid (matches v4 target_shots=5):",
        f"  Rank-1 (Top-1) Accuracy   : {res_top5['top1_acc']*100:.2f}% ({res_top5['top1_ok']}/{res_top5['total_q']})",
        f"  Rank-3 (Top-3) Accuracy   : {res_top5['top3_acc']*100:.2f}% ({res_top5['top3_ok']}/{res_top5['total_q']})",
        f"[Mode 2] All-Gallery Centroid:",
        f"  Rank-1 (Top-1) Accuracy   : {res_all['top1_acc']*100:.2f}% ({res_all['top1_ok']}/{res_all['total_q']})",
        f"  Rank-3 (Top-3) Accuracy   : {res_all['top3_acc']*100:.2f}% ({res_all['top3_ok']}/{res_all['total_q']})",
        f"[Mode 3] 1-NN Individual Gallery Shot Matching:",
        f"  Rank-1 (Top-1) Accuracy   : {prim['nn_top1_ok']/prim['nn_total_q']*100:.2f}% ({prim['nn_top1_ok']}/{prim['nn_total_q']})",
        f"  Rank-3 (Top-3) Accuracy   : {prim['nn_top3_ok']/prim['nn_total_q']*100:.2f}% ({prim['nn_top3_ok']}/{prim['nn_total_q']})",
    ]

    if multi is not None:
        m_top5 = multi["res_top5"]
        m_roc = multi["roc_c"]
        lines.extend([
            f"[Multi-Shot Subset (>=2 Gallery & >=2 Query, {len(multishot_pigs)} pigs, {m_top5['total_q']} queries)]:",
            f"  Top-5 Centroid Rank-1 / Rank-3 : {m_top5['top1_acc']*100:.2f}% ({m_top5['top1_ok']}/{m_top5['total_q']}) | {m_top5['top3_acc']*100:.2f}% ({m_top5['top3_ok']}/{m_top5['total_q']}) | AUC={m_roc[3]:.4f} | EER={m_roc[4]*100:.2f}%",
        ])

    lines.extend([
        "",
        "--- 4. Cosine Similarity & Threshold Verification (Query vs Top-5 Centroid) ---",
        f"Same-Pig (Intra-class) Sim  : {np.mean(same_c):.4f} +- {np.std(same_c):.4f} (min={np.min(same_c):.4f}, max={np.max(same_c):.4f})",
        f"Diff-Pig (Inter-class) Sim  : {np.mean(diff_c):.4f} +- {np.std(diff_c):.4f} (min={np.min(diff_c):.4f}, max={np.max(diff_c):.4f})",
        f"ROC-AUC                     : {auc_c:.4f}",
        f"Equal Error Rate (EER)      : {eer_c*100:.2f}% (at threshold = {eer_th_c:.4f})",
        f"Youden Optimal Threshold    : {youden_th_c:.4f}",
        f"At sim_threshold = 0.82 (v4): TPR (Recall) = {tpr_082_c*100:.2f}% | FPR = {fpr_082_c*100:.2f}%",
        f"At sim_threshold = 0.75(pkg): TPR (Recall) = {tpr_075_c*100:.2f}% | FPR = {fpr_075_c*100:.2f}%",
        "",
        "--- 5. Pairwise Verification (Query vs Individual Gallery Shots) ---",
        f"Same-Pig Shot Sim           : {np.mean(same_s):.4f} +- {np.std(same_s):.4f}",
        f"Diff-Pig Shot Sim           : {np.mean(diff_s):.4f} +- {np.std(diff_s):.4f}",
        f"Shot-to-Shot ROC-AUC / EER  : {auc_s:.4f} / {eer_s*100:.2f}% (thresh={eer_th_s:.4f})",
        f"Shot-to-Shot @0.82          : TPR = {tpr_082_s*100:.2f}% | FPR = {fpr_082_s*100:.2f}%",
        f"Shot-to-Shot @0.75          : TPR = {tpr_075_s*100:.2f}% | FPR = {fpr_075_s*100:.2f}%",
        "",
        "--- 6. Per-Pig Rank-1 Breakdown (Top-5 Sharpness Centroid) ---",
        f"{'Pig ID':<10} | {'Gallery':>7} | {'Query':>5} | {'Top-1':>9} | {'Top-3':>9} | {'Rank-1 Acc':>10}",
        "-" * 65,
    ])
    for pid in primary_pigs:
        st = res_top5["per_pig"][pid]
        lines.append(
            f"{pid:<10} | {st['n_gallery']:7d} | {st['n_query']:5d} | "
            f"{st['top1']:4d}/{st['n_query']:<4d} | {st['top3']:4d}/{st['n_query']:<4d} | {st['acc1']*100:9.2f}%"
        )
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_pigs", type=str, default="", help="Comma-separated pig IDs to exclude (e.g. pig_4,pig_30)")
    parser.add_argument("--require_nose_only", action="store_true", help="Only keep crops with nose_conf >= 0.30")
    args = parser.parse_args()

    exclude_pigs = {s.strip() for s in args.exclude_pigs.split(",") if s.strip()}
    clean_data, raw_count, kept_count = filter_crops(exclude_pigs, require_nose_only=args.require_nose_only)
    evaluate_clean(clean_data, raw_count, kept_count, exclude_pigs, args.require_nose_only)
    print(f"\n[4/4] Clean evaluation complete! Files saved in {BASE_DIR}")


if __name__ == "__main__":
    main()
