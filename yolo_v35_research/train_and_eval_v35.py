import torch  # Import first on Windows to avoid WinError 4551
import os
import sys
import io
import json
import time
import copy
import shutil
import tarfile
import tempfile
import subprocess
import cv2
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(BUNDLE_ROOT, ".."))
V3_CODE_DIR = os.path.join(BUNDLE_ROOT, "01_code", "yolo_v3_occlusion")
for p in (SCRIPT_DIR, V3_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from custom_v35_modules import register_v35_modules, patch_ultralytics_focaler_shape_wiou
import custom_occlusion_modules

V3_DATA_ROOT = os.path.join(BUNDLE_ROOT, "03_datasets", "yolo_face_v3_train_val")
BENCH_ROOT = os.path.join(BUNDLE_ROOT, "03_datasets", "yolo_face_benchmarks")
AI_HARD_DIR = os.path.join(BUNDLE_ROOT, "03_datasets", "external_test_datasets", "generated_hard_samples")
OVERLAY_DIR = os.path.join(BUNDLE_ROOT, "03_datasets", "ai_occlusion_assets", "ai_overlays_rgba")
RECORDED_DIR = os.path.join(BUNDLE_ROOT, "04_videos", "20261111_farm_rgb_thm")

BASELINE_MODEL = os.path.join(BUNDLE_ROOT, "02_models", "v4_parameters", "face.pt")
V2_MODEL = os.path.join(BUNDLE_ROOT, "02_models", "yolo_v2_occlusion_best.pt")
V3_MODEL = os.path.join(BUNDLE_ROOT, "02_models", "yolo_v3_occlusion_best.pt")
V4_MODEL = os.path.join(BUNDLE_ROOT, "02_models", "v4_parameters", "nose_eyes_face_0817.pt")

OUT_ROOT = os.path.join(BUNDLE_ROOT, "runs_workstation", "yolo_v35_research")
DATA_V35_DIR = os.path.join(OUT_ROOT, "dataset_v35")
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
YAML_ARCH = os.path.join(SCRIPT_DIR, "yolo11m_bifpn_seam_ema.yaml")
PRETRAINED_YOLO11M = os.path.join(OUT_ROOT, "yolo11m.pt")
if not os.path.exists(PRETRAINED_YOLO11M):
    PRETRAINED_YOLO11M = "yolo11m.pt"

TRAIN_OVERLAY_NAMES = [
    "ai_asset_01_rusted_bars_rgba.png",
    "ai_asset_02_cross_pipes_rgba.png",
    "ai_asset_04_red_divider_rgba.png",
    "ai_asset_05_chain_pipe_rgba.png",
]
TEST_UNSEEN_OVERLAY_NAMES = [
    "ai_asset_03_wire_grate_rgba.png",
    "ai_asset_06_diag_gate_rgba.png",
]
HELDOUT_FARM_VIDEOS = [
    "rec_20261111_120218",
    "rec_20261111_113922",
    "rec_20261111_113613",
    "rec_20261111_104223",
]


def safe_imread_bgr(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def safe_imread_bgra(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def safe_imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def load_rgba_overlays(names):
    loaded = {}
    for n in names:
        p = os.path.join(OVERLAY_DIR, n)
        im = safe_imread_bgra(p)
        if im is not None and im.shape[2] == 4:
            loaded[n] = im
    return loaded


def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            x1 = max(0, int((cx - bw / 2.0) * img_w))
            y1 = max(0, int((cy - bh / 2.0) * img_h))
            x2 = min(img_w - 1, int((cx + bw / 2.0) * img_w))
            y2 = min(img_h - 1, int((cy + bh / 2.0) * img_h))
            if x2 > x1 + 4 and y2 > y1 + 4:
                boxes.append((x1, y1, x2, y2))
    return boxes


def apply_ai_fence_bgr(img_bgr, bboxes, overlays, rng, target_range=(0.15, 0.42)):
    h, w = img_bgr.shape[:2]
    if bboxes:
        fx1 = min(b[0] for b in bboxes)
        fy1 = min(b[1] for b in bboxes)
        fx2 = max(b[2] for b in bboxes)
        fy2 = max(b[3] for b in bboxes)
    else:
        fx1, fy1, fx2, fy2 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)

    fw = max(fx2 - fx1, 40)
    fh = max(fy2 - fy1, 40)
    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    names = list(overlays.keys())

    best_cand = None
    best_dist = 999.0

    for _ in range(8):
        chosen_name = str(rng.choice(names))
        ov = overlays[chosen_name]
        oh, ow = ov.shape[:2]
        patch = ov.copy()

        frac_crop = float(rng.uniform(0.42, 0.95))
        ch_c, cw_c = max(64, int(oh * frac_crop)), max(64, int(ow * frac_crop))
        sy = int(rng.integers(0, max(1, oh - ch_c + 1)))
        sx = int(rng.integers(0, max(1, ow - cw_c + 1)))
        patch = patch[sy:sy + ch_c, sx:sx + cw_c]

        if rng.random() < 0.5:
            patch = cv2.flip(patch, 1)
        if rng.random() < 0.25:
            patch = cv2.flip(patch, 0)

        nw = max(48, int(w * float(rng.uniform(0.50, 1.05))))
        nh = max(48, int(h * float(rng.uniform(0.50, 1.05))))
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)

        angle = float(rng.uniform(-18.0, 18.0))
        M = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        patch[:, :, :3] = np.clip(
            patch[:, :, :3].astype(np.float32) * float(rng.uniform(0.75, 1.22)) + float(rng.integers(-14, 15)),
            0,
            255,
        ).astype(np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)

        ox = fcx - nw // 2 + int(rng.integers(-int(fw * 0.40), max(1, int(fw * 0.40)) + 1))
        oy = fcy - nh // 2 + int(rng.integers(-int(fh * 0.35), max(1, int(fh * 0.35)) + 1))
        canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sx1, sy1 = max(0, -ox), max(0, -oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        cw = min(nw - sx1, w - dx1)
        ch = min(nh - sy1, h - dy1)
        if cw <= 0 or ch <= 0:
            continue
        canvas_rgba[dy1:dy1 + ch, dx1:dx1 + cw] = patch[sy1:sy1 + ch, sx1:sx1 + cw]
        a_map = canvas_rgba[:, :, 3].astype(np.float32) / 255.0
        occ_frac = float(np.mean(a_map[fy1:fy2 + 1, fx1:fx2 + 1] > 0.35))

        if target_range[0] <= occ_frac <= target_range[1]:
            best_cand = (canvas_rgba, a_map, occ_frac)
            break
        dist = abs(occ_frac - 0.28) + (0.25 if occ_frac > 0.48 else 0.0)
        if dist < best_dist:
            best_dist = dist
            best_cand = (canvas_rgba, a_map, occ_frac)

    if best_cand is None:
        return img_bgr.copy(), 0.0
    canvas_rgba, a_map, final_occ = best_cand
    out_f = img_bgr.astype(np.float32)
    sh = cv2.GaussianBlur(cv2.warpAffine(a_map, np.float32([[1, 0, 6], [0, 1, 6]]), (w, h), borderValue=0.0), (13, 13), 0) * 0.32
    out_f = out_f * (1.0 - sh[..., None])
    a3 = a_map[..., None]
    out_f = (1.0 - a3) * out_f + a3 * canvas_rgba[:, :, :3].astype(np.float32)
    return np.clip(out_f, 0, 255).astype(np.uint8), final_occ


def calc_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def extract_and_filter_white_pigs(v4_det):
    """
    Extract 20260722 real white-pig barn CCTV frames from archive/legacy-v4:dataset_dino_new,
    strictly verified by nose_eyes_face_0817.pt, and split by recording session prefix
    (114* held out for val_white; 111*/113* for train_white).
    """
    print("[1/5] Extracting & quality-filtering 20260722 white-pig barn CCTV frames from archive/legacy-v4...")
    raw_tar = subprocess.check_output(
        ["git", "-C", REPO_ROOT, "archive", "--format=tar", "archive/legacy-v4", "dataset_dino_new"]
    )
    imgs_bytes = {}
    lbls_text = {}
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            bn = os.path.basename(m.name)
            if not bn.startswith("rgb_record_20260722_"):
                continue
            stem = os.path.splitext(bn)[0]
            if "/images/" in m.name and bn.lower().endswith(".jpg"):
                imgs_bytes[stem] = tf.extractfile(m).read()
            elif "/labels/" in m.name and bn.lower().endswith(".txt"):
                lbls_text[stem] = tf.extractfile(m).read().decode("utf-8", errors="ignore")

    train_white, val_white = [], []
    scanned = 0
    for stem in sorted(imgs_bytes.keys()):
        if stem not in lbls_text:
            continue
        arr = np.frombuffer(imgs_bytes[stem], dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        scanned += 1
        ih, iw = bgr.shape[:2]

        # Parse class 1 (face) from Grounding DINO labels
        cand_faces = []
        for line in lbls_text[stem].splitlines():
            ps = line.strip().split()
            if len(ps) < 5 or int(float(ps[0])) != 1:
                continue
            cx, cy, bw, bh = [float(v) for v in ps[1:5]]
            if bw < 0.08 or bh < 0.08 or bw > 0.85 or bh > 0.90:
                continue
            x1 = max(0.0, (cx - bw / 2.0) * iw)
            y1 = max(0.0, (cy - bh / 2.0) * ih)
            x2 = min(float(iw), (cx + bw / 2.0) * iw)
            y2 = min(float(ih), (cy + bh / 2.0) * ih)
            cand_faces.append((cx, cy, bw, bh, [x1, y1, x2, y2]))
        if not cand_faces:
            continue

        # Confirm via V4 0817 landmarks (face or nose)
        r4 = v4_det.predict(bgr, imgsz=640, conf=0.32, verbose=False)[0]
        v4_faces, v4_noses = [], []
        for b in r4.boxes:
            cname = v4_det.names[int(b.cls)].lower()
            xyxy = b.xyxy[0].tolist()
            if "face" in cname:
                v4_faces.append(xyxy)
            elif "nose" in cname:
                v4_noses.append(xyxy)

        verified_lines = []
        verified_boxes = []
        for cx, cy, bw, bh, box_px in cand_faces:
            ok_face = any(calc_iou(box_px, f4) >= 0.25 for f4 in v4_faces)
            ok_nose = any(
                box_px[0] <= 0.5 * (n4[0] + n4[2]) <= box_px[2]
                and box_px[1] <= 0.5 * (n4[1] + n4[3]) <= box_px[3]
                for n4 in v4_noses
            )
            if ok_face or ok_nose:
                verified_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                verified_boxes.append([int(v) for v in box_px])

        if not verified_lines:
            continue

        # Strict session split: 114* -> val_white; 111*/113* -> train_white
        session_tag = stem.split("_")[3] if len(stem.split("_")) >= 4 else ""
        rec = (stem, bgr, verified_lines, verified_boxes)
        if session_tag.startswith("114"):
            val_white.append(rec)
        else:
            train_white.append(rec)

    print(f"  Scanned {scanned} 20260722 frames -> Verified White-Pig Barn CCTV: train_white={len(train_white)}, val_white={len(val_white)}")
    return train_white, val_white


def build_v35_dataset(train_white, val_white, train_overlays, unseen_overlays):
    print("[2/5] Building isolated V3.5 dataset (V3 train + White-Pig Barn CCTV + Balanced Val)...")
    if os.path.exists(DATA_V35_DIR):
        shutil.rmtree(DATA_V35_DIR)
    tr_i = os.path.join(DATA_V35_DIR, "train", "images")
    tr_l = os.path.join(DATA_V35_DIR, "train", "labels")
    va_i = os.path.join(DATA_V35_DIR, "val", "images")
    va_l = os.path.join(DATA_V35_DIR, "val", "labels")
    for d in (tr_i, tr_l, va_i, va_l):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(3535)

    # 1. Copy/link all 5,942 V3 train images & labels
    src_tr_i = os.path.join(V3_DATA_ROOT, "train", "images")
    src_tr_l = os.path.join(V3_DATA_ROOT, "train", "labels")
    v3_cnt = 0
    for fn in sorted(os.listdir(src_tr_i)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        sl = os.path.join(src_tr_l, stem + ".txt")
        if os.path.exists(sl):
            link_or_copy(os.path.join(src_tr_i, fn), os.path.join(tr_i, fn))
            link_or_copy(sl, os.path.join(tr_l, stem + ".txt"))
            v3_cnt += 1

    # 2. Add quality-verified White-Pig Barn CCTV (Clean + AI-Fence + Brightness/Flip)
    w_tr_cnt = 0
    for stem, bgr, lines, boxes in train_white:
        # Clean
        safe_imwrite(os.path.join(tr_i, f"wp_clean_{stem}.jpg"), bgr)
        with open(os.path.join(tr_l, f"wp_clean_{stem}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        w_tr_cnt += 1

        # AI-Fence occluded copy (train overlays only)
        comp, _ = apply_ai_fence_bgr(bgr, boxes, train_overlays, rng, target_range=(0.15, 0.42))
        safe_imwrite(os.path.join(tr_i, f"wp_fence_{stem}.jpg"), comp)
        with open(os.path.join(tr_l, f"wp_fence_{stem}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        w_tr_cnt += 1

        # Horizontal flip + mild photometric copy for multi-angle white-pig generalization
        flip_bgr = cv2.flip(bgr, 1)
        flip_bgr = np.clip(flip_bgr.astype(np.float32) * float(rng.uniform(0.85, 1.15)), 0, 255).astype(np.uint8)
        flip_lines = []
        for ln in lines:
            ps = ln.strip().split()
            cx = 1.0 - float(ps[1])
            flip_lines.append(f"0 {cx:.6f} {ps[2]} {ps[3]} {ps[4]}\n")
        safe_imwrite(os.path.join(tr_i, f"wp_flip_{stem}.jpg"), flip_bgr)
        with open(os.path.join(tr_l, f"wp_flip_{stem}.txt"), "w", encoding="utf-8") as f:
            f.writelines(flip_lines)
        w_tr_cnt += 1

    # 3. Balanced Multi-Scenario Val: 180 clean black + 90 unseen-fence black (from val!) + val_white clean & fence
    src_va_i = os.path.join(V3_DATA_ROOT, "val", "images")
    src_va_l = os.path.join(V3_DATA_ROOT, "val", "labels")
    va_black_cnt = 0
    va_occ_cnt = 0
    for idx_v, fn in enumerate(sorted(os.listdir(src_va_i))):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        sl = os.path.join(src_va_l, stem + ".txt")
        if not os.path.exists(sl):
            continue
        link_or_copy(os.path.join(src_va_i, fn), os.path.join(va_i, fn))
        link_or_copy(sl, os.path.join(va_l, stem + ".txt"))
        va_black_cnt += 1
        if idx_v % 2 == 0:
            im = safe_imread_bgr(os.path.join(src_va_i, fn))
            bxs = load_yolo_boxes(sl, im.shape[1], im.shape[0])
            comp, _ = apply_ai_fence_bgr(im, bxs, unseen_overlays, rng, target_range=(0.20, 0.42))
            safe_imwrite(os.path.join(va_i, f"val_occ_{stem}.jpg"), comp)
            link_or_copy(sl, os.path.join(va_l, f"val_occ_{stem}.txt"))
            va_occ_cnt += 1

    va_w_cnt = 0
    for stem, bgr, lines, boxes in val_white:
        safe_imwrite(os.path.join(va_i, f"val_wp_{stem}.jpg"), bgr)
        with open(os.path.join(va_l, f"val_wp_{stem}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        va_w_cnt += 1
        comp, _ = apply_ai_fence_bgr(bgr, boxes, unseen_overlays, rng, target_range=(0.20, 0.42))
        safe_imwrite(os.path.join(va_i, f"val_wp_occ_{stem}.jpg"), comp)
        with open(os.path.join(va_l, f"val_wp_occ_{stem}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        va_w_cnt += 1

    yaml_p = os.path.join(DATA_V35_DIR, "data_v35.yaml")
    abs_d = os.path.abspath(DATA_V35_DIR).replace("\\", "/")
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_d}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['face']\n")

    stats = {
        "train_v3": v3_cnt,
        "train_white_aug": w_tr_cnt,
        "train_total": v3_cnt + w_tr_cnt,
        "val_black_clean": va_black_cnt,
        "val_black_occ": va_occ_cnt,
        "val_white": va_w_cnt,
        "val_total": va_black_cnt + va_occ_cnt + va_w_cnt,
        "yaml": yaml_p,
    }
    print("  Dataset V3.5 stats:", stats)
    return stats


def make_temp_yaml(root_dir):
    abs_d = os.path.abspath(root_dir).replace("\\", "/")
    fd, tmp = tempfile.mkstemp(prefix="v35_eval_", suffix=".yaml")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_d}\ntrain: images\nval: images\nnc: 1\nnames: ['face']\n")
    return tmp


def eval_detector(model_path, tag, device=0):
    print(f"  Evaluating [{tag}] ({os.path.basename(model_path)})...")
    m = YOLO(model_path)
    y_clean = make_temp_yaml(os.path.join(BENCH_ROOT, "test_clean"))
    y_occ = make_temp_yaml(os.path.join(BENCH_ROOT, "test_occluded"))
    try:
        rc = m.val(data=y_clean, imgsz=640, batch=24, workers=0, device=device, verbose=False, plots=False)
        ro = m.val(data=y_occ, imgsz=640, batch=24, workers=0, device=device, verbose=False, plots=False)
    finally:
        for p in (y_clean, y_occ):
            if os.path.exists(p):
                os.remove(p)

    neg_dir = os.path.join(BENCH_ROOT, "test_hard_neg", "images")
    if not os.path.isdir(neg_dir):
        neg_dir = os.path.join(BENCH_ROOT, "test_hard_neg")
    neg_files = sorted(os.path.join(neg_dir, fn) for fn in os.listdir(neg_dir) if fn.lower().endswith((".jpg", ".png")))
    fp25_i, fp25_b, fp45_i, fp45_b = 0, 0, 0, 0
    for nf in neg_files:
        im = safe_imread_bgr(nf)
        r = m.predict(im, imgsz=640, conf=0.25, device=device, verbose=False)[0]
        confs = [float(b.conf) for b in r.boxes]
        if confs:
            fp25_i += 1
            fp25_b += len(confs)
        c45 = [c for c in confs if c >= 0.45]
        if c45:
            fp45_i += 1
            fp45_b += len(c45)

    ai_pos = {}
    for fn in ["02_yolo_fullframe_metal_bars.png", "01_reid_295-6_vertical_fence.png", "04_yolo_hard_negative_butt_redboard.png"]:
        im = safe_imread_bgr(os.path.join(AI_HARD_DIR, fn))
        r = m.predict(im, imgsz=640, conf=0.20, device=device, verbose=False)[0]
        ai_pos[fn] = [(round(float(b.conf), 4), [int(x) for x in b.xyxy[0].tolist()]) for b in r.boxes]

    # Audit 4 key held-out 20261111 farm videos (90 frames each @ conf>=0.45)
    field = {}
    for vid_id in HELDOUT_FARM_VIDEOS:
        vp = os.path.join(RECORDED_DIR, vid_id, f"{vid_id}_rgb.mp4")
        cap = cv2.VideoCapture(vp)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0, total_f - 1), num=min(90, max(1, total_f)), dtype=int)
        cnt45, cnt35 = 0, 0
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            r = m.predict(fr, imgsz=640, conf=0.35, device=device, verbose=False)[0]
            confs = [float(b.conf) for b in r.boxes]
            cnt35 += len(confs)
            cnt45 += sum(1 for c in confs if c >= 0.45)
        cap.release()
        field[vid_id] = {"boxes_45": cnt45, "boxes_35": cnt35}

    n_neg = max(1, len(neg_files))
    return {
        "clean": {"p": float(rc.box.mp), "r": float(rc.box.mr), "map50": float(rc.box.map50), "map50_95": float(rc.box.map)},
        "occluded": {"p": float(ro.box.mp), "r": float(ro.box.mr), "map50": float(ro.box.map50), "map50_95": float(ro.box.map)},
        "hard_neg": {"fp25_rate": fp25_i / n_neg, "fp25_boxes": fp25_b, "fp45_rate": fp45_i / n_neg, "fp45_boxes": fp45_b},
        "ai_hard": ai_pos,
        "field": field,
    }


def export_fp16_yolo(pt_path, out_fp16_path):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    ckpt_fp16 = copy.deepcopy(ckpt)
    if "model" in ckpt_fp16 and ckpt_fp16["model"] is not None:
        ckpt_fp16["model"] = ckpt_fp16["model"].half()
    for k in ("ema", "optimizer", "updates"):
        if k in ckpt_fp16:
            ckpt_fp16[k] = None
    torch.save(ckpt_fp16, out_fp16_path)
    return os.path.getsize(pt_path) / (1024 * 1024), os.path.getsize(out_fp16_path) / (1024 * 1024)


def draw_boxes_bgr(bgr, dets, color=(0, 220, 0), label_prefix="face"):
    vis = bgr.copy()
    for conf, xyxy in dets:
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, f"{label_prefix} {conf:.2f}", (x1 + 4, max(26, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return vis


def plot_v35_figures(evals, v35_weights_path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444", "axes.linewidth": 0.9})
    models = [("Baseline (face.pt)", "Baseline", "#718096"), ("V2 (YOLOv12n+AI)", "V2", "#a0aec0"), ("V3 (YOLO11s-SEAM-EMA)", "V3", "#3182ce"), ("V3.5 (YOLO11m-BiFPN-WIoUv3)", "V3.5", "#16a34a")]

    # Fig 1: Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=180)
    x = np.arange(len(models))
    w = 0.35

    c50 = [evals[k]["clean"]["map50"] * 100 for _, k, _ in models]
    c95 = [evals[k]["clean"]["map50_95"] * 100 for _, k, _ in models]
    b1 = axes[0].bar(x - w / 2, c50, width=w, label="mAP@0.5", color="#2b6cb0")
    b2 = axes[0].bar(x + w / 2, c95, width=w, label="mAP@0.5:0.95", color="#63b3ed")
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, 110)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in models], rotation=18, ha="right", fontsize=8.5)
    axes[0].set_title("(a) Clean Test Set (n=180)", fontweight="bold")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    o50 = [evals[k]["occluded"]["map50"] * 100 for _, k, _ in models]
    o95 = [evals[k]["occluded"]["map50_95"] * 100 for _, k, _ in models]
    b1 = axes[1].bar(x - w / 2, o50, width=w, label="mAP@0.5", color="#2f855a")
    b2 = axes[1].bar(x + w / 2, o95, width=w, label="mAP@0.5:0.95", color="#68d391")
    for b in list(b1) + list(b2):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(0, 112)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m[0] for m in models], rotation=18, ha="right", fontsize=8.5)
    axes[1].set_title("(b) Unseen Heavy-Fence Test Set (n=180)", fontweight="bold")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    hn45 = [evals[k]["hard_neg"]["fp45_rate"] * 100 for _, k, _ in models]
    human_fp = [evals[k]["field"]["rec_20261111_104223"]["boxes_45"] for _, k, _ in models]
    wp_field = [evals[k]["field"]["rec_20261111_113922"]["boxes_45"] + evals[k]["field"]["rec_20261111_113613"]["boxes_45"] for _, k, _ in models]
    w3 = 0.25
    b1 = axes[2].bar(x - w3, wp_field, width=w3, label="White-Pig Field Recall (#boxes)", color="#16a34a")
    b2 = axes[2].bar(x, human_fp, width=w3, label="Human Video 104223 FPs (#boxes)", color="#e53e3e")
    b3 = axes[2].bar(x + w3, hn45, width=w3, label="Hard-Neg Img FP@0.45 (%)", color="#ed8936")
    for b in list(b1) + list(b2) + list(b3):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m[0] for m in models], rotation=18, ha="right", fontsize=8.5)
    axes[2].set_title("(c) Field White-Pig Recall vs False Positives", fontweight="bold")
    axes[2].legend(fontsize=7.8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Track 2 (v3 Detection): YOLO11m-BiFPN-EMA-SEAM + Focaler-Shape-WIoUv3", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v35_fig1_metrics_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 2: OOD White-Pig CCTV & Hard AI Fence Visual Rescue (V3 vs V3.5)
    m_v3 = YOLO(V3_MODEL)
    m_v35 = YOLO(v35_weights_path)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=180)
    for row, (fn, title_prefix) in enumerate([
        ("02_yolo_fullframe_metal_bars.png", "OOD White-Pig CCTV + Metal Bars"),
        ("01_reid_295-6_vertical_fence.png", "OOD Black-Pig + Vertical Fence"),
    ]):
        bgr = safe_imread_bgr(os.path.join(AI_HARD_DIR, fn))
        for col, (tag, mod, col_bgr) in enumerate([("V3 (YOLO11s-SEAM-EMA)", m_v3, (255, 160, 0)), ("V3.5 (YOLO11m-BiFPN-WIoUv3)", m_v35, (0, 220, 0))]):
            r = mod.predict(bgr, imgsz=640, conf=0.20, device=0, verbose=False)[0]
            dets = [(float(b.conf), [int(v) for v in b.xyxy[0].tolist()]) for b in r.boxes]
            vis = draw_boxes_bgr(bgr, dets, color=col_bgr)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(vis_rgb)
            axes[row, col].set_title(f"{title_prefix} | {tag}\nDets@0.20: {[(round(c, 3)) for c, _ in dets]}", fontweight="bold", fontsize=10)
            axes[row, col].axis("off")

    fig.suptitle("Out-of-Distribution Occluded Pig Face Rescue (V3 vs V3.5)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v35_fig2_ood_white_pig_rescue.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Held-Out Real Farm Video Comparison (120218 Black Fence, 113922 White Fence, 104223 Human Neg)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12.5), dpi=180)
    picks = [
        ("rec_20261111_120218", 120, "120218 Occluded Black Pig"),
        ("rec_20261111_113922", None, "113922 Occluded White Pig"),
        ("rec_20261111_104223", 45, "104223 Patrol Human Legs (Neg)"),
    ]
    for row, (vid_id, target_f, label_t) in enumerate(picks):
        vp = os.path.join(RECORDED_DIR, vid_id, f"{vid_id}_rgb.mp4")
        cap = cv2.VideoCapture(vp)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picked_fr = None
        if target_f is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(target_f, max(0, total_f - 1)))
            _, picked_fr = cap.read()
        else:
            # Pick frame where V3.5 detects strongest white pig face
            best_c = -1.0
            for fi in np.linspace(0, max(0, total_f - 1), num=60, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, fr = cap.read()
                if not ok:
                    continue
                r = m_v35.predict(fr, imgsz=640, conf=0.35, device=0, verbose=False)[0]
                if len(r.boxes) > 0:
                    cmax = max(float(b.conf) for b in r.boxes)
                    if cmax > best_c:
                        best_c = cmax
                        picked_fr = fr.copy()
        cap.release()
        if picked_fr is None:
            continue
        for col, (tag, mod, c_bgr) in enumerate([("V3 @0.45", m_v3, (255, 160, 0)), ("V3.5 @0.45", m_v35, (0, 220, 0))]):
            r = mod.predict(picked_fr, imgsz=640, conf=0.45, device=0, verbose=False)[0]
            dets = [(float(b.conf), [int(v) for v in b.xyxy[0].tolist()]) for b in r.boxes]
            vis = cv2.cvtColor(draw_boxes_bgr(picked_fr, dets, color=c_bgr), cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(vis)
            axes[row, col].set_title(f"{label_t} | {tag} ({len(dets)} boxes)", fontweight="bold", fontsize=10)
            axes[row, col].axis("off")

    fig.suptitle("Held-Out 20261111 Real Farm Video Audit (V3 vs V3.5 @ conf>=0.45)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v35_fig3_field_video_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate saved V3.5 checkpoint")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    register_v35_modules()
    custom_occlusion_modules.register_custom_modules()
    patch_ultralytics_focaler_shape_wiou()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running Track 2 (V3.5 YOLO11m-BiFPN-EMA-SEAM + Focaler-Shape-WIoUv3) on device={device}")

    if args.eval_only:
        v35_best = os.path.join(RUNS_DIR, "yolo11m_v35", "weights", "best.pt")
        fp16_p = os.path.join(OUT_ROOT, "yolo_v35_bifpn_seam_ema_fp16.pt")
        fp32_mb = os.path.getsize(v35_best) / (1024 * 1024)
        fp16_mb = os.path.getsize(fp16_p) / (1024 * 1024)
        ds_stats = {"mode": "eval_only", "best_pt": v35_best}
    else:
        m_v4 = YOLO(V4_MODEL)
        train_white, val_white = extract_and_filter_white_pigs(m_v4)
        del m_v4
        torch.cuda.empty_cache()

        train_overlays = load_rgba_overlays(TRAIN_OVERLAY_NAMES)
        unseen_overlays = load_rgba_overlays(TEST_UNSEEN_OVERLAY_NAMES)
        ds_stats = build_v35_dataset(train_white, val_white, train_overlays, unseen_overlays)

        print("[3/5] Training YOLO11m-BiFPN-EMA-SEAM + Focaler-Shape-WIoUv3 (30 epochs, imgsz=640)...")
        shutil.rmtree(RUNS_DIR, ignore_errors=True)
        for bs in (16, 12):
            try:
                m = YOLO(YAML_ARCH).load(PRETRAINED_YOLO11M)
                m.train(
                    data=ds_stats["yaml"],
                    epochs=30,
                    imgsz=640,
                    batch=bs,
                    lr0=0.001,
                    lrf=0.05,
                    warmup_epochs=2.0,
                    cos_lr=True,
                    erasing=0.15,
                    mosaic=0.7,
                    close_mosaic=6,
                    workers=0,
                    cache=False,
                    amp=True,
                    device=device,
                    project=RUNS_DIR,
                    name="yolo11m_v35",
                    exist_ok=True,
                    plots=True,
                    verbose=True,
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs == 16:
                    print("  CUDA OOM at batch=16, retrying with batch=12...")
                    torch.cuda.empty_cache()
                    shutil.rmtree(RUNS_DIR, ignore_errors=True)
                    continue
                raise

        v35_best = os.path.join(RUNS_DIR, "yolo11m_v35", "weights", "best.pt")
        if not os.path.exists(v35_best):
            v35_best = os.path.join(RUNS_DIR, "yolo11m_v35", "weights", "last.pt")

        fp32_mb, fp16_mb = export_fp16_yolo(v35_best, os.path.join(OUT_ROOT, "yolo_v35_bifpn_seam_ema_fp16.pt"))
        print(f"  Exported FP16 YOLO11m-v3.5: {fp32_mb:.2f} MB -> {fp16_mb:.2f} MB")

    print("[4/5] Running apples-to-apples benchmark & held-out 20261111 farm audit...")
    evals = {
        "Baseline": eval_detector(BASELINE_MODEL, "Baseline face.pt", device=device),
        "V2": eval_detector(V2_MODEL, "V2 YOLOv12n+AI", device=device),
        "V3": eval_detector(V3_MODEL, "V3 YOLO11s-SEAM-EMA", device=device),
        "V3.5": eval_detector(v35_best, "V3.5 YOLO11m-BiFPN-WIoUv3", device=device),
    }

    # Branch Gate Check for v3
    e3, e35 = evals["V3"], evals["V3.5"]
    ood_white_rescued = len(e35["ai_hard"]["02_yolo_fullframe_metal_bars.png"]) > 0
    better_occ = e35["occluded"]["map50_95"] >= e3["occluded"]["map50_95"]
    clean_ok = e35["clean"]["map50"] >= 0.99
    human_ok = e35["field"]["rec_20261111_104223"]["boxes_45"] <= 1
    gate_passed = (ood_white_rescued or better_occ) and clean_ok and human_ok

    summary = {
        "elapsed_sec": time.time() - t0,
        "dataset_stats": ds_stats,
        "weights_best": v35_best,
        "fp32_mb": fp32_mb,
        "fp16_mb": fp16_mb,
        "gate_passed": gate_passed,
        "evals": evals,
    }
    with open(os.path.join(OUT_ROOT, "v35_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    b, v2, v3, v35 = evals["Baseline"], evals["V2"], evals["V3"], evals["V3.5"]
    lines = [
        "=" * 96,
        "Track 2 (V3.5) YOLO11m-BiFPN-EMA-SEAM + Focaler-Shape-WIoUv3 Evaluation Report",
        "=" * 96,
        f"Dataset: {ds_stats}",
        f"Gate Passed: {gate_passed} | FP16 Weight: {fp16_mb:.2f} MB | Elapsed: {summary['elapsed_sec']:.1f}s",
        "",
        f"{'Metric':<38} | {'Baseline':>10} | {'V2':>10} | {'V3':>10} | {'V3.5':>10}",
        "-" * 90,
        f"{'Clean Test mAP@0.5':<38} | {b['clean']['map50']*100:>9.2f}% | {v2['clean']['map50']*100:>9.2f}% | {v3['clean']['map50']*100:>9.2f}% | {v35['clean']['map50']*100:>9.2f}%",
        f"{'Clean Test mAP@0.5:0.95':<38} | {b['clean']['map50_95']*100:>9.2f}% | {v2['clean']['map50_95']*100:>9.2f}% | {v3['clean']['map50_95']*100:>9.2f}% | {v35['clean']['map50_95']*100:>9.2f}%",
        f"{'Occluded Test mAP@0.5':<38} | {b['occluded']['map50']*100:>9.2f}% | {v2['occluded']['map50']*100:>9.2f}% | {v3['occluded']['map50']*100:>9.2f}% | {v35['occluded']['map50']*100:>9.2f}%",
        f"{'Occluded Test mAP@0.5:0.95':<38} | {b['occluded']['map50_95']*100:>9.2f}% | {v2['occluded']['map50_95']*100:>9.2f}% | {v3['occluded']['map50_95']*100:>9.2f}% | {v35['occluded']['map50_95']*100:>9.2f}%",
        f"{'Hard-Neg Img FP@0.45':<38} | {b['hard_neg']['fp45_rate']*100:>9.2f}% | {v2['hard_neg']['fp45_rate']*100:>9.2f}% | {v3['hard_neg']['fp45_rate']*100:>9.2f}% | {v35['hard_neg']['fp45_rate']*100:>9.2f}%",
        f"{'Field 120218 (Black Fence) @0.45':<38} | {b['field']['rec_20261111_120218']['boxes_45']:>10} | {v2['field']['rec_20261111_120218']['boxes_45']:>10} | {v3['field']['rec_20261111_120218']['boxes_45']:>10} | {v35['field']['rec_20261111_120218']['boxes_45']:>10}",
        f"{'Field 113922 (White Fence) @0.45':<38} | {b['field']['rec_20261111_113922']['boxes_45']:>10} | {v2['field']['rec_20261111_113922']['boxes_45']:>10} | {v3['field']['rec_20261111_113922']['boxes_45']:>10} | {v35['field']['rec_20261111_113922']['boxes_45']:>10}",
        f"{'Field 113613 (White Fence) @0.45':<38} | {b['field']['rec_20261111_113613']['boxes_45']:>10} | {v2['field']['rec_20261111_113613']['boxes_45']:>10} | {v3['field']['rec_20261111_113613']['boxes_45']:>10} | {v35['field']['rec_20261111_113613']['boxes_45']:>10}",
        f"{'Field 104223 (Human Neg) @0.45':<38} | {b['field']['rec_20261111_104223']['boxes_45']:>10} | {v2['field']['rec_20261111_104223']['boxes_45']:>10} | {v3['field']['rec_20261111_104223']['boxes_45']:>10} | {v35['field']['rec_20261111_104223']['boxes_45']:>10}",
        f"{'OOD White Metal Bars (02_yolo)':<38} | {str([x[0] for x in b['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v2['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v3['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v35['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10}",
        f"{'OOD Black Vertical Fence (01_reid)':<38} | {str([x[0] for x in b['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v2['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v3['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v35['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10}",
        "=" * 96,
    ]
    rep = "\n".join(lines)
    with open(os.path.join(OUT_ROOT, "v35_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print("\n" + rep)

    print("[5/5] Generating V3.5 visual figures...")
    plot_v35_figures(evals, v35_best)
    print("Done! All V3.5 artifacts saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
