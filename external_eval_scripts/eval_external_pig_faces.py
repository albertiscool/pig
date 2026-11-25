import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import sys
import re
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


PROJECT_ROOT = r"C:\Users\User\Desktop\final_pig_v0"
V4_DIR = os.path.join(PROJECT_ROOT, "v4")
DETECTOR_PATH = os.path.join(V4_DIR, "parameters", "nose_eyes_face_0817.pt")
REID_MODEL_PATH = os.path.join(V4_DIR, "parameters", "pig_ecb_best.pth")

OUT_DIR = os.path.join(PROJECT_ROOT, "external_test_datasets", "shaojiang_pig_faces")
GALLERY_DIR = os.path.join(OUT_DIR, "gallery")
QUERY_DIR = os.path.join(OUT_DIR, "query")
META_PATH = os.path.join(OUT_DIR, "crops_metadata.json")
GRID_PATH = os.path.join(OUT_DIR, "sample_crops_grid.jpg")
PLOT_PATH = os.path.join(OUT_DIR, "evaluation_plots.png")
REPORT_PATH = os.path.join(OUT_DIR, "evaluation_report.txt")

HF_DATASET_ID = "shaojiang/pig-images"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
HF_RESOLVE_PREFIX = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/"

FACE_CONF_THRESH = 0.45
MIN_CROP_SIZE = 60
MIN_SHARPNESS = 25.0
MAX_TRAIN_INITIAL = 35
MAX_TRAIN_EXTRA = 35
MAX_VAL_SUPP = 25
MIN_GALLERY_CROPS = 3
MIN_QUERY_CROPS = 3


def natural_key(path_str):
    parts = re.split(r"(\d+)", path_str)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def evenly_spaced(items, k):
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, num=k, dtype=int)
    seen = set()
    selected = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            selected.append(items[i])
    return selected


def fetch_hf_manifest():
    req = urllib.request.Request(HF_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    splits = {"train_img": {}, "val_img": {}, "test_img": {}}
    for s in data.get("siblings", []):
        fn = s.get("rfilename", "")
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = fn.split("/")
        if len(parts) != 3:
            continue
        sp, pig_id, _ = parts
        if sp in splits:
            splits[sp].setdefault(pig_id, []).append(fn)

    for sp in splits:
        for pig_id in splits[sp]:
            splits[sp][pig_id].sort(key=natural_key)
    return splits


def download_image_bytes(fn):
    url = HF_RESOLVE_PREFIX + fn
    if HAS_REQUESTS:
        if not hasattr(download_image_bytes, "_session"):
            import threading
            download_image_bytes._local = threading.local()
        sess = getattr(download_image_bytes._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": "Mozilla/5.0"})
            download_image_bytes._local.session = sess
        r = sess.get(url, timeout=20)
        r.raise_for_status()
        return r.content
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()


def download_batch(filenames, max_workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(download_image_bytes, fn): fn for fn in filenames}
        for fut in as_completed(fut_map):
            fn = fut_map[fut]
            try:
                raw = fut.result()
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    results.append((fn, img))
            except Exception as e:
                print(f"  [WARN] Failed to download {fn}: {e}")
    results.sort(key=lambda x: natural_key(x[0]))
    return results


def detect_and_crop_face(det_model, img, use_half=True):
    h_img, w_img = img.shape[:2]
    try:
        res = det_model.predict(img, imgsz=640, half=use_half, verbose=False)[0]
    except Exception:
        res = det_model.predict(img, imgsz=640, half=False, verbose=False)[0]

    best_box = None
    best_conf = -1.0
    for b in res.boxes:
        cls_name = det_model.names[int(b.cls)]
        conf = float(b.conf)
        if cls_name == "face" and conf >= FACE_CONF_THRESH and conf > best_conf:
            best_conf = conf
            best_box = [int(v) for v in b.xyxy[0].tolist()]

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))

    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        return None

    return crop, best_conf, sharpness, (x1, y1, x2, y2)


def prepare_dataset_and_crops(det_model, use_half=True):
    os.makedirs(GALLERY_DIR, exist_ok=True)
    os.makedirs(QUERY_DIR, exist_ok=True)

    print("[1/4] Fetching Hugging Face dataset manifest...")
    splits = fetch_hf_manifest()
    pig_ids = sorted(
        set(splits["train_img"].keys()) | set(splits["test_img"].keys()),
        key=natural_key,
    )
    print(f"  Found {len(pig_ids)} pigs in {HF_DATASET_ID}: {pig_ids}")

    metadata = {"gallery": {}, "query": {}, "stats": {"scanned_frames": 0, "valid_crops": 0}}
    t0 = time.time()

    for pid in pig_ids:
        label = f"pig_{pid}"
        g_dir = os.path.join(GALLERY_DIR, label)
        q_dir = os.path.join(QUERY_DIR, label)
        os.makedirs(g_dir, exist_ok=True)
        os.makedirs(q_dir, exist_ok=True)

        # --- Gallery from train_img ---
        train_files = splits["train_img"].get(pid, [])
        g_initial = evenly_spaced(train_files, MAX_TRAIN_INITIAL)
        g_initial_set = set(g_initial)
        g_crops = []

        downloaded_train = download_batch(g_initial, max_workers=8)
        metadata["stats"]["scanned_frames"] += len(downloaded_train)
        for fn, img in downloaded_train:
            det_res = detect_and_crop_face(det_model, img, use_half=use_half)
            if det_res is not None:
                crop, conf, sharp, bbox = det_res
                stem = os.path.splitext(os.path.basename(fn))[0]
                out_path = os.path.join(g_dir, f"train_{stem}.jpg")
                cv2.imwrite(out_path, crop)
                g_crops.append({
                    "path": out_path,
                    "source": fn,
                    "split": "train_img",
                    "conf": round(conf, 4),
                    "sharpness": round(sharp, 2),
                    "size": [int(crop.shape[1]), int(crop.shape[0])],
                })

        if len(g_crops) < MIN_GALLERY_CROPS:
            remaining_train = [f for f in train_files if f not in g_initial_set]
            g_extra = evenly_spaced(remaining_train, MAX_TRAIN_EXTRA)
            if g_extra:
                dl_extra = download_batch(g_extra, max_workers=8)
                metadata["stats"]["scanned_frames"] += len(dl_extra)
                for fn, img in dl_extra:
                    det_res = detect_and_crop_face(det_model, img, use_half=use_half)
                    if det_res is not None:
                        crop, conf, sharp, bbox = det_res
                        stem = os.path.splitext(os.path.basename(fn))[0]
                        out_path = os.path.join(g_dir, f"train_{stem}.jpg")
                        cv2.imwrite(out_path, crop)
                        g_crops.append({
                            "path": out_path,
                            "source": fn,
                            "split": "train_img",
                            "conf": round(conf, 4),
                            "sharpness": round(sharp, 2),
                            "size": [int(crop.shape[1]), int(crop.shape[0])],
                        })

        # --- Query from test_img (+ val_img fallback if needed) ---
        test_files = splits["test_img"].get(pid, [])
        q_crops = []
        dl_test = download_batch(test_files, max_workers=8)
        metadata["stats"]["scanned_frames"] += len(dl_test)
        for fn, img in dl_test:
            det_res = detect_and_crop_face(det_model, img, use_half=use_half)
            if det_res is not None:
                crop, conf, sharp, bbox = det_res
                stem = os.path.splitext(os.path.basename(fn))[0]
                out_path = os.path.join(q_dir, f"test_{stem}.jpg")
                cv2.imwrite(out_path, crop)
                q_crops.append({
                    "path": out_path,
                    "source": fn,
                    "split": "test_img",
                    "conf": round(conf, 4),
                    "sharpness": round(sharp, 2),
                    "size": [int(crop.shape[1]), int(crop.shape[0])],
                })

        if len(q_crops) < MIN_QUERY_CROPS:
            val_files = splits["val_img"].get(pid, [])
            v_pick = evenly_spaced(val_files, MAX_VAL_SUPP)
            if v_pick:
                dl_val = download_batch(v_pick, max_workers=8)
                metadata["stats"]["scanned_frames"] += len(dl_val)
                for fn, img in dl_val:
                    det_res = detect_and_crop_face(det_model, img, use_half=use_half)
                    if det_res is not None:
                        crop, conf, sharp, bbox = det_res
                        stem = os.path.splitext(os.path.basename(fn))[0]
                        out_path = os.path.join(q_dir, f"val_{stem}.jpg")
                        cv2.imwrite(out_path, crop)
                        q_crops.append({
                            "path": out_path,
                            "source": fn,
                            "split": "val_img",
                            "conf": round(conf, 4),
                            "sharpness": round(sharp, 2),
                            "size": [int(crop.shape[1]), int(crop.shape[0])],
                        })

        metadata["gallery"][label] = g_crops
        metadata["query"][label] = q_crops
        metadata["stats"]["valid_crops"] += len(g_crops) + len(q_crops)
        qual_mark = "OK" if (len(g_crops) >= MIN_GALLERY_CROPS and len(q_crops) >= MIN_QUERY_CROPS) else "LOW"
        print(f"  {label:>6s}: gallery={len(g_crops):2d} (train_img) | query={len(q_crops):2d} [{qual_mark}]")

    elapsed = time.time() - t0
    print(f"  Completed download & face extraction in {elapsed:.1f}s | total valid crops = {metadata['stats']['valid_crops']}")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def build_contact_sheet(metadata, qualified_pigs):
    tiles = []
    tile_w, tile_h = 180, 180
    bar_h = 34
    selected_pigs = qualified_pigs[:15]

    for label in selected_pigs:
        g_list = metadata["gallery"].get(label, [])
        q_list = metadata["query"].get(label, [])
        for role_tag, item in [("G", max(g_list, key=lambda x: x["sharpness"]) if g_list else None),
                               ("Q", max(q_list, key=lambda x: x["sharpness"]) if q_list else None)]:
            if item is None:
                continue
            img = cv2.imread(item["path"])
            if img is None:
                continue
            resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((tile_h + bar_h, tile_w, 3), dtype=np.uint8)
            canvas[:tile_h, :] = resized
            canvas[tile_h:, :] = (28, 28, 28)
            txt = f"{label} [{role_tag}] c={item['conf']:.2f}"
            cv2.putText(canvas, txt, (6, tile_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(canvas)

    if not tiles:
        return
    cols = 6
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
    print(f"  Saved contact sheet to {GRID_PATH} ({len(tiles)} tiles)")


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


def run_evaluation(metadata):
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

    print("[2/4] Extracting embeddings and Softmax predictions for all crops...")
    all_max_probs = []
    all_pred_classes = []
    gallery_feats = {}
    query_feats = {}

    for label, items in metadata["gallery"].items():
        gallery_feats[label] = []
        for it in items:
            mp, pc, feat = infer_crop(it["path"])
            all_max_probs.append(mp)
            all_pred_classes.append(pc)
            gallery_feats[label].append({
                "feat": feat,
                "sharpness": it["sharpness"],
                "path": it["path"],
                "max_prob": mp,
            })

    for label, items in metadata["query"].items():
        query_feats[label] = []
        for it in items:
            mp, pc, feat = infer_crop(it["path"])
            all_max_probs.append(mp)
            all_pred_classes.append(pc)
            query_feats[label].append({
                "feat": feat,
                "sharpness": it["sharpness"],
                "path": it["path"],
                "split": it["split"],
                "max_prob": mp,
            })

    qualified_pigs = sorted(
        [
            pid for pid in metadata["gallery"]
            if len(gallery_feats.get(pid, [])) >= MIN_GALLERY_CROPS
            and len(query_feats.get(pid, [])) >= MIN_QUERY_CROPS
        ],
        key=natural_key,
    )

    print(f"  Qualified pigs (>= {MIN_GALLERY_CROPS} gallery & >= {MIN_QUERY_CROPS} query): {len(qualified_pigs)}")
    if len(qualified_pigs) < 6:
        raise RuntimeError(f"QUALIFIED_PIGS_TOO_FEW: {len(qualified_pigs)} (< 6)")

    build_contact_sheet(metadata, qualified_pigs)

    # --- Evaluation A: Closed-Set Open-Set Rejection ---
    probs_arr = np.asarray(all_max_probs, dtype=np.float64)
    far_065 = float(np.mean(probs_arr >= 0.65))
    trr_065 = 1.0 - far_065
    far_050 = float(np.mean(probs_arr >= 0.50))
    trr_050 = 1.0 - far_050

    # --- Evaluation B: Open-Set Re-ID Generalization ---
    top5_centroids = {}
    all_centroids = {}
    all_gallery_shots = []  # (pid, feat_np)

    for pid in qualified_pigs:
        g_sorted = sorted(gallery_feats[pid], key=lambda x: x["sharpness"], reverse=True)
        top5_stack = torch.stack([x["feat"] for x in g_sorted[:5]], dim=0)
        c5 = F.normalize(torch.mean(top5_stack, dim=0, keepdim=True), p=2, dim=1)[0].numpy()
        top5_centroids[pid] = c5

        all_stack = torch.stack([x["feat"] for x in g_sorted], dim=0)
        ca = F.normalize(torch.mean(all_stack, dim=0, keepdim=True), p=2, dim=1)[0].numpy()
        all_centroids[pid] = ca

        for x in g_sorted:
            all_gallery_shots.append((pid, x["feat"].numpy()))

    pid_list = qualified_pigs
    mat_top5 = np.stack([top5_centroids[p] for p in pid_list], axis=0)
    mat_all = np.stack([all_centroids[p] for p in pid_list], axis=0)
    mat_shots = np.stack([s[1] for s in all_gallery_shots], axis=0)
    shot_pids = [s[0] for s in all_gallery_shots]

    def eval_ranking(centroids_mat, mode_name):
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
                sims = centroids_mat @ qf  # shape (N_pigs,)
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
            "mode": mode_name,
            "total_q": total_q,
            "top1_acc": top1_ok / total_q,
            "top3_acc": top3_ok / total_q,
            "top1_ok": top1_ok,
            "top3_ok": top3_ok,
            "per_pig": per_pig,
            "same_sims": np.asarray(same_sims),
            "diff_sims": np.asarray(diff_sims),
            "pair_scores": np.asarray(pair_scores),
            "pair_labels": np.asarray(pair_labels),
        }

    res_top5 = eval_ranking(mat_top5, "Top-5 Sharpness Gallery Centroid")
    res_all = eval_ranking(mat_all, "All-Gallery Centroid")

    # 1-NN against individual gallery shots + pairwise shot-to-shot stats
    nn_top1_ok = 0
    nn_top3_ok = 0
    nn_total_q = 0
    shot_same_sims = []
    shot_diff_sims = []
    shot_pair_scores = []
    shot_pair_labels = []

    for pid in pid_list:
        for q in query_feats[pid]:
            qf = q["feat"].numpy()
            sims = mat_shots @ qf
            # Max sim per pig ID for Top-k ranking
            max_per_pid = {}
            for s_idx, s_pid in enumerate(shot_pids):
                val = float(sims[s_idx])
                shot_pair_scores.append(val)
                if s_pid == pid:
                    shot_pair_labels.append(1)
                    shot_same_sims.append(val)
                else:
                    shot_pair_labels.append(0)
                    shot_diff_sims.append(val)
                if s_pid not in max_per_pid or val > max_per_pid[s_pid]:
                    max_per_pid[s_pid] = val
            ranked_pids = [k for k, _ in sorted(max_per_pid.items(), key=lambda kv: -kv[1])]
            if ranked_pids[0] == pid:
                nn_top1_ok += 1
            if pid in ranked_pids[:min(3, len(pid_list))]:
                nn_top3_ok += 1
            nn_total_q += 1

    # ROC & threshold metrics for Top-5 Centroid (primary open-set mode)
    fpr_c, tpr_c, _, auc_c, eer_c, eer_th_c, youden_th_c = compute_roc_metrics(
        res_top5["pair_scores"], res_top5["pair_labels"]
    )
    same_c = res_top5["same_sims"]
    diff_c = res_top5["diff_sims"]
    tpr_082_c = float(np.mean(same_c >= 0.82))
    fpr_082_c = float(np.mean(diff_c >= 0.82))
    tpr_075_c = float(np.mean(same_c >= 0.75))
    fpr_075_c = float(np.mean(diff_c >= 0.75))

    # Also compute ROC for Query-vs-Shot pairs
    fpr_s, tpr_s, _, auc_s, eer_s, eer_th_s, youden_th_s = compute_roc_metrics(
        shot_pair_scores, shot_pair_labels
    )
    same_s = np.asarray(shot_same_sims)
    diff_s = np.asarray(shot_diff_sims)
    tpr_082_s = float(np.mean(same_s >= 0.82))
    fpr_082_s = float(np.mean(diff_s >= 0.82))

    # --- Visualization Plot ---
    print("[3/4] Generating evaluation plots and text report...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)

    # Panel 1: Cosine similarity distribution (Query vs Top-5 Centroid)
    ax = axes[0]
    ax.hist(diff_c, bins=30, density=True, alpha=0.55, color="#d95f02", label=f"Different Pig (n={len(diff_c)})")
    ax.hist(same_c, bins=25, density=True, alpha=0.65, color="#1b9e77", label=f"Same Pig (n={len(same_c)})")
    ax.axvline(0.82, color="#e41a1c", linestyle="--", linewidth=1.8, label="Prod thresh = 0.82")
    ax.axvline(0.75, color="#377eb8", linestyle=":", linewidth=1.8, label="Pkg thresh = 0.75")
    ax.axvline(eer_th_c, color="#000000", linestyle="-.", linewidth=1.5, label=f"EER thresh = {eer_th_c:.3f}")
    ax.set_title("Open-Set Re-ID Cosine Similarity\n(Query vs Top-5 Gallery Centroid)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 2: ROC Curve
    ax = axes[1]
    ax.plot(fpr_c, tpr_c, color="#1b9e77", linewidth=2.2, label=f"Centroid ROC (AUC={auc_c:.4f}, EER={eer_c*100:.2f}%)")
    ax.plot(fpr_s, tpr_s, color="#7570b3", linewidth=1.8, linestyle="--", label=f"Shot-to-Shot ROC (AUC={auc_s:.4f}, EER={eer_s*100:.2f}%)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0)
    ax.scatter([fpr_082_c], [tpr_082_c], color="#e41a1c", zorder=5, s=55, label=f"Centroid @0.82 (TPR={tpr_082_c*100:.1f}%, FPR={fpr_082_c*100:.1f}%)")
    ax.set_title("Open-Set Verification ROC Curve")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel 3: Closed-Set Max Softmax Confidence for Unknown External Pigs
    ax = axes[2]
    ax.hist(probs_arr, bins=30, color="#377eb8", alpha=0.75, edgecolor="white")
    ax.axvline(0.65, color="#e41a1c", linestyle="--", linewidth=1.8, label=f"Prod thresh = 0.65 (FAR={far_065*100:.1f}%)")
    ax.axvline(0.50, color="#ff7f00", linestyle=":", linewidth=1.8, label=f"Pkg thresh = 0.50 (FAR={far_050*100:.1f}%)")
    ax.set_title(f"Closed-Set Rejection (17 Known Classes)\nMax Softmax on External Unknown Crops (n={len(probs_arr)})")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Crop Count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    # --- Build Report ---
    total_gallery_crops = sum(len(v) for v in metadata["gallery"].values())
    total_query_crops = sum(len(v) for v in metadata["query"].values())
    qual_gallery_crops = sum(len(gallery_feats[p]) for p in qualified_pigs)
    qual_query_crops = sum(len(query_feats[p]) for p in qualified_pigs)
    q_test_count = sum(1 for p in qualified_pigs for q in query_feats[p] if q["split"] == "test_img")
    q_val_count = sum(1 for p in qualified_pigs for q in query_feats[p] if q["split"] == "val_img")

    lines = [
        "================================================================================",
        "External Pig Face Recognition Evaluation Report",
        "================================================================================",
        f"Dataset Source     : Hugging Face `{HF_DATASET_ID}` (30 individuals, openrail license)",
        f"Face Detector      : {DETECTOR_PATH} (cls='face', conf>={FACE_CONF_THRESH}, min_size={MIN_CROP_SIZE}px, min_laplacian={MIN_SHARPNESS})",
        f"Re-ID Model        : {REID_MODEL_PATH} (ECBHybridModel ResNet18 inference branch, classes={num_classes})",
        f"Device / Precision : {device} ({'FP16' if use_half else 'FP32'})",
        "",
        "--- 1. Dataset Extraction Summary ---",
        f"Total frames scanned     : {metadata['stats']['scanned_frames']}",
        f"Total valid face crops   : {total_gallery_crops + total_query_crops} (Gallery={total_gallery_crops} from train_img, Query={total_query_crops} from test_img/val_img)",
        f"Qualified test pigs      : {len(qualified_pigs)} pigs (each >= {MIN_GALLERY_CROPS} gallery crops & >= {MIN_QUERY_CROPS} query crops)",
        f"Qualified crop counts    : Gallery={qual_gallery_crops} (all from train_img), Query={qual_query_crops} ({q_test_count} from test_img, {q_val_count} from val_img)",
        "",
        "--- 2. Evaluation A: Open-Set Unknown Rejection (17 Known Training Classes vs External Unseen Pigs) ---",
        f"All external face crops tested : {len(probs_arr)}",
        f"Max Softmax Prob (Mean / Med)  : {np.mean(probs_arr):.4f} / {np.median(probs_arr):.4f}",
        f"Max Softmax Prob (P90 / Max)   : {np.percentile(probs_arr, 90):.4f} / {np.max(probs_arr):.4f}",
        f"At conf_threshold = 0.65 (v4)  : False Accept Rate (FAR) = {far_065*100:.2f}% | True Rejection Rate (TRR) = {trr_065*100:.2f}%",
        f"At conf_threshold = 0.50 (pkg) : False Accept Rate (FAR) = {far_050*100:.2f}% | True Rejection Rate (TRR) = {trr_050*100:.2f}%",
        "",
        "--- 3. Evaluation B: Open-Set Re-ID Generalization (Unseen Pigs Registered -> Matched on Query) ---",
        f"Number of unseen gallery IDs   : {len(qualified_pigs)}",
        f"Total query crops evaluated    : {res_top5['total_q']}",
        f"[Mode 1] Top-5 Sharpness Centroid (matches v4 target_shots=5):",
        f"  Rank-1 (Top-1) Accuracy      : {res_top5['top1_acc']*100:.2f}% ({res_top5['top1_ok']}/{res_top5['total_q']})",
        f"  Rank-3 (Top-3) Accuracy      : {res_top5['top3_acc']*100:.2f}% ({res_top5['top3_ok']}/{res_top5['total_q']})",
        f"[Mode 2] All-Gallery Centroid:",
        f"  Rank-1 (Top-1) Accuracy      : {res_all['top1_acc']*100:.2f}% ({res_all['top1_ok']}/{res_all['total_q']})",
        f"  Rank-3 (Top-3) Accuracy      : {res_all['top3_acc']*100:.2f}% ({res_all['top3_ok']}/{res_all['total_q']})",
        f"[Mode 3] 1-NN Individual Gallery Shot Matching:",
        f"  Rank-1 (Top-1) Accuracy      : {nn_top1_ok/nn_total_q*100:.2f}% ({nn_top1_ok}/{nn_total_q})",
        f"  Rank-3 (Top-3) Accuracy      : {nn_top3_ok/nn_total_q*100:.2f}% ({nn_top3_ok}/{nn_total_q})",
        "",
        "--- 4. Cosine Similarity & Threshold Verification (Query vs Top-5 Centroid) ---",
        f"Same-Pig (Intra-class) Sim     : {np.mean(same_c):.4f} +- {np.std(same_c):.4f} (min={np.min(same_c):.4f}, max={np.max(same_c):.4f})",
        f"Diff-Pig (Inter-class) Sim     : {np.mean(diff_c):.4f} +- {np.std(diff_c):.4f} (min={np.min(diff_c):.4f}, max={np.max(diff_c):.4f})",
        f"ROC-AUC                        : {auc_c:.4f}",
        f"Equal Error Rate (EER)         : {eer_c*100:.2f}% (at threshold = {eer_th_c:.4f})",
        f"Youden Optimal Threshold       : {youden_th_c:.4f}",
        f"At sim_threshold = 0.82 (v4)   : TPR (Recall) = {tpr_082_c*100:.2f}% | FPR = {fpr_082_c*100:.2f}%",
        f"At sim_threshold = 0.75 (pkg)  : TPR (Recall) = {tpr_075_c*100:.2f}% | FPR = {fpr_075_c*100:.2f}%",
        "",
        "--- 5. Pairwise Verification (Query vs Individual Gallery Shots) ---",
        f"Same-Pig Shot Sim              : {np.mean(same_s):.4f} +- {np.std(same_s):.4f}",
        f"Diff-Pig Shot Sim              : {np.mean(diff_s):.4f} +- {np.std(diff_s):.4f}",
        f"Shot-to-Shot ROC-AUC / EER     : {auc_s:.4f} / {eer_s*100:.2f}% (thresh={eer_th_s:.4f})",
        f"Shot-to-Shot @0.82             : TPR = {tpr_082_s*100:.2f}% | FPR = {fpr_082_s*100:.2f}%",
        "",
        "--- 6. Per-Pig Rank-1 Breakdown (Top-5 Sharpness Centroid) ---",
        f"{'Pig ID':<10} | {'Gallery':>7} | {'Query':>5} | {'Top-1':>9} | {'Top-3':>9} | {'Rank-1 Acc':>10}",
        "-" * 65,
    ]
    for pid in qualified_pigs:
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
    print("[0/4] Loading YOLO face detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = (device == "cuda")
    det_model = YOLO(DETECTOR_PATH)
    det_model.to(device)
    metadata = prepare_dataset_and_crops(det_model, use_half=use_half)
    run_evaluation(metadata)
    print(f"\n[4/4] Done! Output saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
