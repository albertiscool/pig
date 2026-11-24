import torch

import os
import sys
import ast
import csv
import glob
import time
import shutil
import tarfile
import urllib.request

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

EXP_V3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_V3_DIR)
from custom_occlusion_modules import register_custom_modules, patch_ultralytics_shape_iou


PROJECT_ROOT = os.path.abspath(os.path.join(EXP_V3_DIR, "..", ".."))
BRAIN_DIR = os.path.join(PROJECT_ROOT, "scratch", "brain_assets")

BASELINE_MODEL = os.path.join(PROJECT_ROOT, "v4", "parameters", "face.pt")
V2_MODEL = os.path.join(PROJECT_ROOT, "experiments", "yolo_fence_occlusion_v2", "runs", "face_ai_fence_v2", "weights", "best.pt")
V4_MODEL = os.path.join(PROJECT_ROOT, "v4", "parameters", "nose_eyes_face_0817.pt")
YOLO11S_PRETRAINED = os.path.join(PROJECT_ROOT, "v2", "yolo11s.pt")
if not os.path.exists(YOLO11S_PRETRAINED):
    YOLO11S_PRETRAINED = "yolo11s.pt"

V1_EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "yolo_fence_occlusion_exp")
V2_RGBA_DIR = os.path.join(PROJECT_ROOT, "experiments", "yolo_fence_occlusion_v2", "ai_overlays_rgba")
FACE_TRAIN_ROOT = os.path.join(PROJECT_ROOT, "v3", "data", "face_train")
SHAOJIANG_ROOT = os.path.join(PROJECT_ROOT, "external_test_datasets", "shaojiang_pig_faces")
AI_HARD_DIR = os.path.join(PROJECT_ROOT, "external_test_datasets", "generated_hard_samples")
RECORDED_DIR = os.path.join(PROJECT_ROOT, "recorded_videos")

YAML_ARCH_PATH = os.path.join(EXP_V3_DIR, "yolo11s_seam_ema.yaml")
AI_NEW_ASSETS_DIR = os.path.join(EXP_V3_DIR, "ai_new_assets")
RGBA_DIR = os.path.join(EXP_V3_DIR, "ai_overlays_rgba")
HF_UNIQUEDATA_DIR = os.path.join(EXP_V3_DIR, "external_hf_uniquedata")
DATA_DIR = os.path.join(EXP_V3_DIR, "data")
RUNS_DIR = os.path.join(EXP_V3_DIR, "runs")

DATASET_PREVIEW_PATH = os.path.join(EXP_V3_DIR, "v3_dataset_preview_grid.jpg")
SUSP_GRID_PATH = os.path.join(EXP_V3_DIR, "v3_field_suspicious_grid.jpg")
FIELD_COMP_GRID_PATH = os.path.join(EXP_V3_DIR, "v3_vs_v2_vs_v4_field_grid.jpg")
VIDEO_OUT_PATH = os.path.join(EXP_V3_DIR, "rec_20261111_120218_v4_vs_v3_check.mp4")
CHART_PATH = os.path.join(EXP_V3_DIR, "v3_metrics_chart.png")
REPORT_PATH = os.path.join(EXP_V3_DIR, "v3_full_report.txt")

# Strictly split recorded videos: train negatives from 20261110 + 103456; held-out test from 8x 20261111 videos
TRAIN_NEG_VIDEOS = [
    "rec_20261110_175801",
    "rec_20261110_155533",
    "rec_20261110_155732",
    "rec_20261110_161639",
    "rec_20261110_181030",
    "rec_20261110_180408",
    "rec_20261110_175312",
    "rec_20261110_162254",
    "rec_20261110_160356",
    "rec_20261111_103456",
]

HELDOUT_FARM_VIDEOS = [
    "rec_20261111_120218",
    "rec_20261111_120047",
    "rec_20261111_115925",
    "rec_20261111_114820",
    "rec_20261111_113922",
    "rec_20261111_113613",
    "rec_20261111_110932",
    "rec_20261111_104223",
]


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


def chroma_key_extract(src_img_path, out_rgba_path):
    bgr = cv2.imread(src_img_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    pure_green = (h >= 35) & (h <= 90) & (s >= 60) & (v >= 50)
    b_ch = bgr[:, :, 0].astype(np.float32)
    g_ch = bgr[:, :, 1].astype(np.float32)
    r_ch = bgr[:, :, 2].astype(np.float32)
    max_rb = np.maximum(r_ch, b_ch)
    green_excess = g_ch - max_rb

    alpha = np.where(pure_green, 0.0, 1.0).astype(np.float32)
    edge_green = (~pure_green) & (green_excess > 18.0) & (h >= 32) & (h <= 95)
    alpha[edge_green] = np.clip(1.0 - (green_excess[edge_green] - 18.0) / 45.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha[pure_green] = 0.0

    despill_mask = (alpha < 0.99) | (green_excess > 15.0)
    bgr_clean = bgr.copy()
    bgr_clean[:, :, 1] = np.clip(np.where(despill_mask, np.minimum(g_ch, max_rb + 4.0), g_ch), 0, 255).astype(np.uint8)

    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([bgr_clean, alpha_u8])
    ys, xs = np.where(alpha_u8 > 25)
    if len(ys) > 0:
        rgba = rgba[max(0, int(ys.min()) - 2):min(rgba.shape[0], int(ys.max()) + 3),
                    max(0, int(xs.min()) - 2):min(rgba.shape[1], int(xs.max()) + 3)]
    cv2.imwrite(out_rgba_path, rgba)
    return rgba


def prepare_all_ai_assets():
    print("[1/6] Loading V2 RGBA overlays + extracting new AI chain/pipe/gate assets...")
    os.makedirs(AI_NEW_ASSETS_DIR, exist_ok=True)
    os.makedirs(RGBA_DIR, exist_ok=True)

    overlays = {}
    for p in sorted(glob.glob(os.path.join(V2_RGBA_DIR, "*_rgba.png"))):
        name = os.path.basename(p).replace("_rgba.png", "")
        dst = os.path.join(RGBA_DIR, os.path.basename(p))
        shutil.copy2(p, dst)
        im = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
        if im is not None and im.shape[2] == 4:
            overlays[name] = im

    new_chroma = [
        ("ai_asset_05_chain_pipe", "pig_pen_chain_and_nipple_pipe_1795275401220.jpg"),
        ("ai_asset_06_diag_gate", "rusty_farm_gate_bars_chroma_green_1795275402909.jpg"),
    ]
    for name, fname in new_chroma:
        src = os.path.join(BRAIN_DIR, fname)
        if os.path.exists(src):
            raw_dst = os.path.join(AI_NEW_ASSETS_DIR, f"{name}.png")
            Image.open(src).save(raw_dst, format="PNG")
            rgba = chroma_key_extract(raw_dst, os.path.join(RGBA_DIR, f"{name}_rgba.png"))
            overlays[name] = rgba

    ai_hn_paths = []
    for out_name, fname in [
        ("ai_hn_v3_01_pig_rears.png", "pig_barn_cctv_rear_view_1795275405097.jpg"),
        ("ai_hn_v3_02_barn_boots_bars.png", "pig_farm_cctv_corridor_v2_1795275423431.jpg"),
    ]:
        src = os.path.join(BRAIN_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(AI_NEW_ASSETS_DIR, out_name)
            Image.open(src).save(dst, format="PNG")
            ai_hn_paths.append(dst)

    print(f"  Ready {len(overlays)} AI chroma-key overlays: {sorted(overlays.keys())}")
    print(f"  Ready {len(ai_hn_paths)} AI Hard-Negative CCTV scenes")
    return overlays, ai_hn_paths


def _prep_patch(name, rgba, target_w, target_h, rng):
    oh, ow = rgba.shape[:2]
    patch = rgba.copy()

    if name == "ai_asset_01_rusted_bars":
        cw = max(32, int(ow * float(rng.uniform(0.28, 0.90))))
        sx = int(rng.integers(0, max(1, ow - cw + 1)))
        patch = patch[:, sx:sx + cw]
        if rng.random() < 0.30:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif name == "ai_asset_05_chain_pipe":
        mode = rng.choice(["chain", "pipe", "both"], p=[0.4, 0.4, 0.2])
        if mode == "chain":
            patch = patch[:, :max(32, int(ow * 0.48))]
        elif mode == "pipe":
            patch = patch[:, max(0, int(ow * 0.42)):]
        if rng.random() < 0.15:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif name in ("ai_asset_03_wire_grate", "ai_asset_06_diag_gate"):
        frac = float(rng.uniform(0.38, 0.90))
        ch, cw = max(64, int(oh * frac)), max(64, int(ow * frac))
        sy = int(rng.integers(0, max(1, oh - ch + 1)))
        sx = int(rng.integers(0, max(1, ow - cw + 1)))
        patch = patch[sy:sy + ch, sx:sx + cw]
    elif name == "ai_asset_02_cross_pipes":
        if rng.random() < 0.25:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)

    if rng.random() < 0.5:
        patch = cv2.flip(patch, 1)
    if rng.random() < 0.35:
        patch = cv2.flip(patch, 0)

    new_w = max(32, int(target_w * float(rng.uniform(0.55, 1.15))))
    new_h = max(32, int(target_h * float(rng.uniform(0.55, 1.15))))
    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    angle = float(rng.uniform(-20.0, 20.0))
    ch, cw = patch.shape[:2]
    M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
    patch = cv2.warpAffine(patch, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    rgb = np.clip(patch[:, :, :3].astype(np.float32) * float(rng.uniform(0.72, 1.22)) + float(rng.integers(-16, 17)), 0, 255).astype(np.uint8)
    patch[:, :, :3] = rgb
    k = 3 if rng.random() < 0.7 else 5
    patch = cv2.GaussianBlur(patch, (k, k), 0)
    return patch


def apply_ai_fence_clamped(img_bgr, bboxes, overlays, rng, target_range=(0.15, 0.45)):
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
    all_names = list(overlays.keys())

    best_cand = None
    best_dist = 999.0

    for _ in range(6):
        n_layers = min(len(all_names), 2 if rng.random() < 0.20 else 1)
        chosen = rng.choice(all_names, size=n_layers, replace=False)

        layers_placed = []
        cand_alpha = np.zeros((h, w), dtype=np.float32)
        for name in chosen:
            patch = _prep_patch(name, overlays[name], w, h, rng)
            ph, pw = patch.shape[:2]
            canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            if name == "ai_asset_04_red_divider":
                corner = int(rng.integers(0, 4))
                ox = int(fx1 - pw * float(rng.uniform(0.42, 0.70))) if corner % 2 == 0 else int(fx2 - pw * float(rng.uniform(0.30, 0.58)))
                oy = int(fy2 - ph * float(rng.uniform(0.32, 0.60))) if corner < 2 else int(fy1 - ph * float(rng.uniform(0.45, 0.72)))
            else:
                ox = fcx - pw // 2 + int(rng.integers(-int(fw * 0.35), max(1, int(fw * 0.35)) + 1))
                oy = fcy - ph // 2 + int(rng.integers(-int(fh * 0.32), max(1, int(fh * 0.32)) + 1))

            sx1, sy1 = max(0, -ox), max(0, -oy)
            dx1, dy1 = max(0, ox), max(0, oy)
            cw = min(pw - sx1, w - dx1)
            ch = min(ph - sy1, h - dy1)
            if cw <= 0 or ch <= 0:
                continue
            canvas_rgba[dy1:dy1 + ch, dx1:dx1 + cw] = patch[sy1:sy1 + ch, sx1:sx1 + cw]
            a_map = canvas_rgba[:, :, 3].astype(np.float32) / 255.0
            np.maximum(cand_alpha, a_map, out=cand_alpha)
            layers_placed.append((canvas_rgba, a_map))

        if not bboxes:
            best_cand = (layers_placed, 0.0)
            break

        occ_frac = float(np.mean(cand_alpha[fy1:fy2 + 1, fx1:fx2 + 1] > 0.35))
        if target_range[0] <= occ_frac <= target_range[1]:
            best_cand = (layers_placed, occ_frac)
            break
        dist = abs(occ_frac - 0.30) + (0.2 if occ_frac > 0.48 else 0.0)
        if dist < best_dist:
            best_dist = dist
            best_cand = (layers_placed, occ_frac)

    out_float = img_bgr.astype(np.float32)
    layers_placed, final_occ = best_cand if best_cand is not None else ([], 0.0)
    for canvas_rgba, a_map in layers_placed:
        sh_dx, sh_dy = int(rng.integers(4, 11)), int(rng.integers(4, 11))
        M_sh = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
        sh = cv2.warpAffine(a_map, M_sh, (w, h), borderValue=0.0)
        sh = cv2.GaussianBlur(sh, (15, 15), 0) * float(rng.uniform(0.26, 0.42))
        out_float = out_float * (1.0 - sh[..., None])
        a3 = a_map[..., None]
        out_float = (1.0 - a3) * out_float + a3 * canvas_rgba[:, :, :3].astype(np.float32)

    return np.clip(out_float, 0, 255).astype(np.uint8), final_occ


def download_uniquedata_hf():
    print("[2/6] Downloading HuggingFace UniqueData/pigs-detection-dataset...")
    os.makedirs(HF_UNIQUEDATA_DIR, exist_ok=True)
    items = []
    try:
        img_tar = os.path.join(HF_UNIQUEDATA_DIR, "images.tar.gz")
        csv_path = os.path.join(HF_UNIQUEDATA_DIR, "pigs-detection-dataset.csv")
        base_url = "https://huggingface.co/datasets/UniqueData/pigs-detection-dataset/resolve/main/data/"
        for fn, dst in [("images.tar.gz", img_tar), ("pigs-detection-dataset.csv", csv_path)]:
            if not os.path.exists(dst):
                req = urllib.request.Request(base_url + fn, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                    f.write(r.read())

        extract_dir = os.path.join(HF_UNIQUEDATA_DIR, "extracted")
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(img_tar, "r:gz") as tf:
                tf.extractall(extract_dir)

        with open(csv_path, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                rel = row["image_name"].strip()
                # Search under extracted dir
                cand = glob.glob(os.path.join(extract_dir, "**", os.path.basename(rel)), recursive=True)
                if not cand:
                    continue
                img_p = cand[0]
                im = cv2.imread(img_p)
                if im is None:
                    continue
                ih, iw = im.shape[:2]
                anns = ast.literal_eval(row["annotations"])
                yolo_lines = []
                boxes = []
                for a in anns:
                    if a.get("label") != "pig_face":
                        continue
                    pts = a.get("points", [])
                    if len(pts) != 2:
                        continue
                    xs = [float(pts[0][0]), float(pts[1][0])]
                    ys = [float(pts[0][1]), float(pts[1][1])]
                    x1 = max(0.0, min(xs))
                    x2 = min(float(iw), max(xs))
                    y1 = max(0.0, min(ys))
                    y2 = min(float(ih), max(ys))
                    bw, bh = x2 - x1, y2 - y1
                    if bw >= 12.0 and bh >= 12.0:
                        cx = (x1 + x2) / (2.0 * iw)
                        cy = (y1 + y2) / (2.0 * ih)
                        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw/iw:.6f} {bh/ih:.6f}\n")
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
                if yolo_lines:
                    items.append((img_p, yolo_lines, boxes))
        print(f"  Loaded {len(items)} valid multi-breed pig_face images from UniqueData HF")
    except Exception as e:
        print(f"  [WARN] UniqueData HF download/parse skipped: {e}")
    return items


def letterbox_or_crop_640(img_bgr, rng, mode="letterbox"):
    h, w = img_bgr.shape[:2]
    if mode == "letterbox":
        scale = min(640.0 / w, 640.0 / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        res = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((640, 640, 3), int(rng.integers(38, 125)), dtype=np.uint8)
        y0, x0 = (640 - nh) // 2, (640 - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = res
        return canvas
    elif mode == "crop":
        if min(h, w) >= 320:
            ch = int(rng.uniform(0.45, 0.85) * h)
            cw = int(rng.uniform(0.45, 0.85) * w)
            y0 = int(rng.integers(0, max(1, h - ch + 1)))
            x0 = int(rng.integers(0, max(1, w - cw + 1)))
            patch = img_bgr[y0:y0 + ch, x0:x0 + cw]
            return cv2.resize(patch, (640, 640), interpolation=cv2.INTER_AREA)
    return cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)


def mine_real_zero_pig_negatives(v4_det):
    print("[3/6] Mining verified zero-pig real frames from 20261110 + 103456 videos...")
    frames_pool = []
    for vid_id in TRAIN_NEG_VIDEOS:
        vp = os.path.join(RECORDED_DIR, vid_id, f"{vid_id}_rgb.mp4")
        if not os.path.exists(vp):
            continue
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0, total_f - 1), num=min(40, max(1, total_f)), dtype=int)
        kept = 0
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            r4 = v4_det.predict(frame, imgsz=640, conf=0.20, verbose=False)[0]
            if len(r4.boxes) == 0:
                frames_pool.append((vid_id, frame))
                kept += 1
        cap.release()
        print(f"  {vid_id}: kept {kept}/{len(idxs)} zero-pig frames")
    print(f"  Total mined real zero-pig frames: {len(frames_pool)}")
    return frames_pool


def make_concrete_barn_bg(rng):
    base = int(rng.integers(58, 125))
    bg = np.full((640, 640, 3), base, dtype=np.float32)
    split_y = int(rng.integers(180, 420))
    bg[:split_y, :] += np.array([rng.integers(-12, 12), rng.integers(-10, 10), rng.integers(-8, 15)], dtype=np.float32)
    bg[split_y:, :] += np.array([rng.integers(-8, 15), rng.integers(-8, 12), rng.integers(-10, 10)], dtype=np.float32)
    coarse = rng.random((16, 16), dtype=np.float32)
    coarse_up = cv2.resize(coarse, (640, 640), interpolation=cv2.INTER_CUBIC)
    bg = bg * (0.72 + 0.52 * coarse_up[..., None])
    noise = rng.normal(0, 6.0, bg.shape).astype(np.float32)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def paste_feathered_face(bg_img, crop_bgr, rng):
    bh, bw = bg_img.shape[:2]
    ch, cw = crop_bgr.shape[:2]
    if rng.random() < 0.5:
        crop_bgr = cv2.flip(crop_bgr, 1)
    crop_bgr = np.clip(crop_bgr.astype(np.float32) * float(rng.uniform(0.80, 1.20)) + float(rng.integers(-15, 16)), 0, 255).astype(np.uint8)
    scale = (float(rng.uniform(0.28, 0.72)) * min(bh, bw)) / max(ch, cw)
    nw = max(48, min(bw - 16, int(cw * scale)))
    nh = max(48, min(bh - 16, int(ch * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x1 = int(rng.integers(8, max(9, bw - nw - 7)))
    y1 = int(rng.integers(8, max(9, bh - nh - 7)))
    x2, y2 = x1 + nw, y1 + nh

    mask = np.ones((nh, nw), dtype=np.float32)
    pad = max(4, min(10, nw // 16, nh // 16))
    mask[:pad, :] = np.linspace(0.0, 1.0, pad)[:, None]
    mask[-pad:, :] = np.minimum(mask[-pad:, :], np.linspace(1.0, 0.0, pad)[:, None])
    mask[:, :pad] = np.minimum(mask[:, :pad], np.linspace(0.0, 1.0, pad)[None, :])
    mask[:, -pad:] = np.minimum(mask[:, -pad:], np.linspace(1.0, 0.0, pad)[None, :])

    out = bg_img.astype(np.float32)
    m3 = mask[..., None]
    out[y1:y2, x1:x2] = (1.0 - m3) * out[y1:y2, x1:x2] + m3 * resized.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), (x1, y1, x2, y2), f"0 {(x1+x2)/(2.0*bw):.6f} {(y1+y2)/(2.0*bh):.6f} {nw/float(bw):.6f} {nh/float(bh):.6f}\n"


def prepare_v3_dataset(overlays, ai_hn_paths, hf_items, v4_det, base_det):
    print("[4/6] Assembling V3 Training Set (~6,000 imgs, zero leakage)...")
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    dst_tr_i = os.path.join(DATA_DIR, "train", "images")
    dst_tr_l = os.path.join(DATA_DIR, "train", "labels")
    dst_va_i = os.path.join(DATA_DIR, "val", "images")
    dst_va_l = os.path.join(DATA_DIR, "val", "labels")
    for d in (dst_tr_i, dst_tr_l, dst_va_i, dst_va_l):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(2028)
    preview_samples = []

    # 1. All 4,092 real clean black pig images
    tr_img_src = os.path.join(FACE_TRAIN_ROOT, "train", "images")
    tr_lbl_src = os.path.join(FACE_TRAIN_ROOT, "train", "labels")
    by_base = {}
    clean_cnt = 0
    for f in sorted(os.listdir(tr_img_src)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        sl = os.path.join(tr_lbl_src, stem + ".txt")
        if os.path.exists(sl):
            shutil.copy2(os.path.join(tr_img_src, f), os.path.join(dst_tr_i, f))
            shutil.copy2(sl, os.path.join(dst_tr_l, stem + ".txt"))
            clean_cnt += 1
            if os.path.getsize(sl) > 0:
                by_base.setdefault(f.split("_jpg.rf.")[0], []).append(f)

    # 2. +1,000 AI-fence occluded black pig images (clamped 15-45%)
    picked_base = [sorted(by_base[b])[0] for b in sorted(by_base.keys())]
    if len(picked_base) > 1000:
        idx = rng.choice(len(picked_base), size=1000, replace=False)
        picked_base = [picked_base[i] for i in sorted(idx)]

    occ_fracs = []
    for fname in picked_base:
        stem = os.path.splitext(fname)[0]
        im = cv2.imread(os.path.join(dst_tr_i, fname))
        lp = os.path.join(dst_tr_l, stem + ".txt")
        boxes = load_yolo_boxes(lp, im.shape[1], im.shape[0])
        comp, frac = apply_ai_fence_clamped(im, boxes, overlays, rng, target_range=(0.15, 0.45))
        occ_fracs.append(frac)
        cv2.imwrite(os.path.join(dst_tr_i, f"ai6_fence_{stem}.jpg"), comp)
        shutil.copy2(lp, os.path.join(dst_tr_l, f"ai6_fence_{stem}.txt"))
        if len(preview_samples) < 4:
            vis = comp.copy()
            for b in boxes:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
            preview_samples.append((vis, f"AI Fence Black ({frac*100:.0f}%)"))

    # 3. Multi-breed white/pink pig positives (~250 imgs: UniqueData real + AI fenced + shaojiang clean_gallery)
    wb_cnt = 0
    for idx_hf, (img_p, yolo_lines, boxes) in enumerate(hf_items):
        im = cv2.imread(img_p)
        if im is None:
            continue
        out_name = f"hf_uniq_clean_{idx_hf:03d}.jpg"
        cv2.imwrite(os.path.join(dst_tr_i, out_name), im)
        with open(os.path.join(dst_tr_l, f"hf_uniq_clean_{idx_hf:03d}.txt"), "w") as lf:
            lf.writelines(yolo_lines)
        wb_cnt += 1
        if idx_hf == 0:
            vis = im.copy()
            for b in boxes:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
            preview_samples.append((vis, f"HF UniqueData Clean ({len(boxes)}f)"))

        comp, frac = apply_ai_fence_clamped(im, boxes, overlays, rng, target_range=(0.15, 0.45))
        out_name_f = f"hf_uniq_fence_{idx_hf:03d}.jpg"
        cv2.imwrite(os.path.join(dst_tr_i, out_name_f), comp)
        with open(os.path.join(dst_tr_l, f"hf_uniq_fence_{idx_hf:03d}.txt"), "w") as lf:
            lf.writelines(yolo_lines)
        wb_cnt += 1
        if idx_hf == 1:
            vis = comp.copy()
            for b in boxes:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
            preview_samples.append((vis, f"HF UniqueData+Fence ({frac*100:.0f}%)"))

    # Collect clean shaojiang crops & non-face barn backgrounds
    white_crops = sorted(glob.glob(os.path.join(SHAOJIANG_ROOT, "clean_gallery", "pig_*", "*.jpg")))
    need_wb = max(120, 250 - wb_cnt)
    for i in range(need_wb):
        cp = white_crops[i % len(white_crops)]
        cim = cv2.imread(cp)
        if cim is None:
            continue
        bg = make_concrete_barn_bg(rng)
        comp, bbox, yline = paste_feathered_face(bg, cim, rng)
        fenced = False
        if rng.random() < 0.55:
            comp, _ = apply_ai_fence_clamped(comp, [bbox], overlays, rng, target_range=(0.15, 0.45))
            fenced = True
        cv2.imwrite(os.path.join(dst_tr_i, f"shaojiang_wb_{i:04d}.jpg"), comp)
        with open(os.path.join(dst_tr_l, f"shaojiang_wb_{i:04d}.txt"), "w") as lf:
            lf.write(yline)
        wb_cnt += 1
        if len(preview_samples) < 8:
            vis = comp.copy()
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 0), 2)
            preview_samples.append((vis, "WhitePig+AI Fence" if fenced else "WhitePig Clean"))

    # 4. +600 Diverse Hard Negatives (empty .txt)
    hn_cnt = 0
    # 4a. ~350 Real Lab / Chair / Human / Corridor zero-pig negatives from 20261110 + 103456
    real_neg_frames = mine_real_zero_pig_negatives(v4_det)
    for i in range(350):
        vid_id, fr = real_neg_frames[i % len(real_neg_frames)]
        mode = ["letterbox", "crop", "stretch"][i % 3]
        out_im = letterbox_or_crop_640(fr.copy(), rng, mode=mode)
        if rng.random() < 0.5:
            out_im = cv2.flip(out_im, 1)
        out_im = np.clip(out_im.astype(np.float32) * float(rng.uniform(0.8, 1.2)) + float(rng.integers(-15, 16)), 0, 255).astype(np.uint8)
        if rng.random() < 0.25:
            out_im, _ = apply_ai_fence_clamped(out_im, [], overlays, rng)
        cv2.imwrite(os.path.join(dst_tr_i, f"real_lab_hn_{i:04d}.jpg"), out_im)
        open(os.path.join(dst_tr_l, f"real_lab_hn_{i:04d}.txt"), "w").close()
        hn_cnt += 1
        if len(preview_samples) < 11 and vid_id.endswith(("175801", "155533", "103456")):
            preview_samples.append((out_im.copy(), f"Real ZeroPig ({vid_id[-6:]})"))

    # 4b. 100 Augmented views from 2 new AI Hard-Negative CCTV scenes (pig rears + barn boots/bars)
    ai_hn_imgs = [cv2.imread(p) for p in ai_hn_paths if cv2.imread(p) is not None]
    for i in range(100):
        fr = ai_hn_imgs[i % len(ai_hn_imgs)]
        mode = ["letterbox", "crop", "stretch"][i % 3]
        out_im = letterbox_or_crop_640(fr.copy(), rng, mode=mode)
        if rng.random() < 0.5:
            out_im = cv2.flip(out_im, 1)
        out_im = np.clip(out_im.astype(np.float32) * float(rng.uniform(0.78, 1.22)) + float(rng.integers(-16, 17)), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_tr_i, f"ai_cctv_hn_{i:04d}.jpg"), out_im)
        open(os.path.join(dst_tr_l, f"ai_cctv_hn_{i:04d}.txt"), "w").close()
        hn_cnt += 1
        if len(preview_samples) < 14 and i < 3:
            preview_samples.append((out_im.copy(), f"AI HardNeg CCTV ({mode})"))

    # 4c. 150 shaojiang real non-face crops (pig_6, 11, 14, 26 verified 0 face) + AI divider empty scenes
    ext_negs = []
    for pid in ("pig_6", "pig_11", "pig_14", "pig_26"):
        for role, crole in [("gallery", "clean_gallery"), ("query", "clean_query")]:
            rd = os.path.join(SHAOJIANG_ROOT, role, pid)
            cd = os.path.join(SHAOJIANG_ROOT, crole, pid)
            cset = set(os.listdir(cd)) if os.path.isdir(cd) else set()
            if os.path.isdir(rd):
                for fn in sorted(os.listdir(rd)):
                    if fn not in cset:
                        im = cv2.imread(os.path.join(rd, fn))
                        if im is not None:
                            pad = cv2.copyMakeBorder(im, im.shape[0]//2, im.shape[0]//2, im.shape[1]//2, im.shape[1]//2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                            if len(base_det.predict(pad, imgsz=320, conf=0.35, verbose=False)[0].boxes) == 0 and len(v4_det.predict(pad, imgsz=320, conf=0.25, verbose=False)[0].boxes) == 0:
                                ext_negs.append(im)
    for i in range(150):
        if i < 80 and len(ext_negs) > 0:
            out_im = letterbox_or_crop_640(ext_negs[i % len(ext_negs)], rng, mode="letterbox" if i % 2 == 0 else "stretch")
            if rng.random() < 0.4:
                out_im, _ = apply_ai_fence_clamped(out_im, [], overlays, rng)
        else:
            bg = make_concrete_barn_bg(rng)
            out_im, _ = apply_ai_fence_clamped(bg, [], overlays, rng)
        cv2.imwrite(os.path.join(dst_tr_i, f"ext_ai_hn_{i:04d}.jpg"), out_im)
        open(os.path.join(dst_tr_l, f"ext_ai_hn_{i:04d}.txt"), "w").close()
        hn_cnt += 1
        if len(preview_samples) < 16 and i >= 80:
            preview_samples.append((out_im.copy(), "AI Empty Pen Scene"))

    # Save 4x4 Dataset Preview Grid
    tw, th, bar = 240, 220, 28
    tiles = []
    for vis, lbl in preview_samples[:16]:
        res = cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA)
        t = np.zeros((th + bar, tw, 3), dtype=np.uint8)
        t[:th, :] = res
        t[th:, :] = (28, 28, 28)
        cv2.putText(t, lbl, (6, th + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 4
    rows_c = int(np.ceil(len(tiles) / cols))
    pad = 6
    pgrid = np.full((rows_c * (th + bar + pad) + pad, cols * (tw + pad) + pad, 3), 12, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r, c = idx // cols, idx % cols
        pgrid[pad + r * (th + bar + pad):pad + r * (th + bar + pad) + th + bar, pad + c * (tw + pad):pad + c * (tw + pad) + tw] = t
    cv2.imwrite(DATASET_PREVIEW_PATH, pgrid)

    # Copy valid split (180 real imgs)
    va_src_i = os.path.join(FACE_TRAIN_ROOT, "valid", "images")
    va_src_l = os.path.join(FACE_TRAIN_ROOT, "valid", "labels")
    val_cnt = 0
    for f in sorted(os.listdir(va_src_i)):
        stem = os.path.splitext(f)[0]
        sl = os.path.join(va_src_l, stem + ".txt")
        if os.path.exists(sl):
            shutil.copy2(os.path.join(va_src_i, f), os.path.join(dst_va_i, f))
            shutil.copy2(sl, os.path.join(dst_va_l, stem + ".txt"))
            val_cnt += 1

    yaml_v3 = os.path.join(EXP_V3_DIR, "data_train_v3.yaml")
    with open(yaml_v3, "w", encoding="utf-8") as f:
        f.write(f"path: {DATA_DIR.replace(chr(92), '/')}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['face']\n")

    info = {
        "train_clean": clean_cnt,
        "train_ai_fence": len(picked_base),
        "train_white_breed": wb_cnt,
        "train_hard_neg": hn_cnt,
        "train_total": clean_cnt + len(picked_base) + wb_cnt + hn_cnt,
        "mean_ai_occ": float(np.mean(occ_fracs)),
        "val_clean": val_cnt,
        "yaml_train": yaml_v3,
        "yaml_clean": os.path.join(V1_EXP_DIR, "data_test_clean.yaml"),
        "yaml_occ": os.path.join(V1_EXP_DIR, "data_test_occluded.yaml"),
        "neg_test_dir": os.path.join(V1_EXP_DIR, "data", "test_hard_neg", "images"),
        "test_occ_dir": os.path.join(V1_EXP_DIR, "data", "test_occluded", "images"),
        "test_occ_lbl_dir": os.path.join(V1_EXP_DIR, "data", "test_occluded", "labels"),
    }
    print(f"  V3 Train Total={info['train_total']} ({clean_cnt} real + {len(picked_base)} AI fence @{info['mean_ai_occ']*100:.1f}% + {wb_cnt} white-pig + {hn_cnt} hard-neg) | Val={val_cnt}")
    return info


def calc_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def landmark_inside_box(box, lm_boxes, expand=0.10):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ex1, ey1, ex2, ey2 = x1 - bw * expand, y1 - bh * expand, x2 + bw * expand, y2 + bh * expand
    for lx1, ly1, lx2, ly2 in lm_boxes:
        cx, cy = 0.5 * (lx1 + lx2), 0.5 * (ly1 + ly2)
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            return True
    return False


def make_tile(img_bgr, top_text, bot_text, tw=430, th=250):
    bar_h = 46
    ih, iw = img_bgr.shape[:2]
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    res = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th + bar_h, tw, 3), 20, dtype=np.uint8)
    canvas[(th - nh) // 2:(th - nh) // 2 + nh, (tw - nw) // 2:(tw - nw) // 2 + nw] = res
    canvas[th:, :] = (32, 32, 32)
    cv2.putText(canvas, top_text, (8, th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, bot_text, (8, th + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 230, 180), 1, cv2.LINE_AA)
    return canvas


def draw_boxes(img_bgr, boxes, color=(0, 220, 0), label_prefix="face"):
    vis = img_bgr.copy()
    thick = max(2, int(round(min(vis.shape[:2]) / 250.0)))
    fscale = max(0.55, min(vis.shape[:2]) / 780.0)
    for item in boxes:
        if len(item) == 2:
            conf, box = item
            lbl = f"{label_prefix} {conf:.2f}"
        else:
            cls_n, conf, box = item
            lbl = f"{cls_n} {conf:.2f}"
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, lbl, (x1 + 4, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick, cv2.LINE_AA)
    return vis


def draw_v4_boxes(img_bgr, v4_items):
    vis = img_bgr.copy()
    cmap = {"face": (0, 220, 0), "nose": (0, 145, 255), "eye": (255, 220, 0)}
    thick = max(2, int(round(min(vis.shape[:2]) / 250.0)))
    fscale = max(0.55, min(vis.shape[:2]) / 780.0)
    for cls_n, conf, box in v4_items:
        c = cmap.get(cls_n, (255, 0, 255))
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, thick)
        cv2.putText(vis, f"{cls_n} {conf:.2f}", (x1 + 4, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, fscale, c, thick, cv2.LINE_AA)
    return vis


def eval_std_suite(model_path, info, tag):
    print(f"  Std Eval [{tag}]...")
    m = YOLO(model_path)
    res_clean = m.val(data=info["yaml_clean"], imgsz=640, batch=24, workers=0, verbose=False, plots=False)
    res_occ = m.val(data=info["yaml_occ"], imgsz=640, batch=24, workers=0, verbose=False, plots=False)

    neg_files = sorted([os.path.join(info["neg_test_dir"], f) for f in os.listdir(info["neg_test_dir"])])
    fp25_b, fp25_i, fp45_b, fp45_i = 0, 0, 0, 0
    neg_ex = {}
    for nf in neg_files:
        r = m.predict(nf, imgsz=640, conf=0.25, verbose=False)[0]
        confs = [float(b.conf) for b in r.boxes]
        if confs:
            fp25_i += 1
            fp25_b += len(confs)
            neg_ex[os.path.basename(nf)] = [round(c, 3) for c in confs]
        c45 = [c for c in confs if c >= 0.45]
        if c45:
            fp45_i += 1
            fp45_b += len(c45)

    ai_pos = {}
    for fn in ["02_yolo_fullframe_metal_bars.png", "01_reid_295-6_vertical_fence.png"]:
        p = os.path.join(AI_HARD_DIR, fn)
        r = m.predict(p, imgsz=640, conf=0.20, verbose=False)[0]
        ai_pos[fn] = [(round(float(b.conf), 4), [int(x) for x in b.xyxy[0].tolist()]) for b in r.boxes]

    return {
        "clean": {"precision": float(res_clean.box.mp), "recall": float(res_clean.box.mr), "map50": float(res_clean.box.map50), "map50_95": float(res_clean.box.map)},
        "occluded": {"precision": float(res_occ.box.mp), "recall": float(res_occ.box.mr), "map50": float(res_occ.box.map50), "map50_95": float(res_occ.box.map)},
        "hard_neg": {"n_images": len(neg_files), "fp25_rate": fp25_i / max(1, len(neg_files)), "fp25_boxes": fp25_b, "fp45_rate": fp45_i / max(1, len(neg_files)), "fp45_boxes": fp45_b, "examples": neg_ex},
        "ai_fence": ai_pos,
    }


def audit_heldout_farm_videos(m_base, m_v2, m_v3, m_v4):
    print(f"  Field Audit across {len(HELDOUT_FARM_VIDEOS)} held-out 20261111 farm videos (90f/video)...")
    per_vid = []
    tot = {k: 0 for k in ("frames", "v3_45", "v2_45", "base_45", "v4_45", "v3_susp45", "v2_susp45", "v3_35", "v3_susp35", "v3_agree45")}
    susp_events = []
    comp_frames = {}

    for vid_id in HELDOUT_FARM_VIDEOS:
        vp = os.path.join(RECORDED_DIR, vid_id, f"{vid_id}_rgb.mp4")
        cap = cv2.VideoCapture(vp)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idxs = np.linspace(0, max(0, total_f - 1), num=min(90, max(1, total_f)), dtype=int)

        vs = {k: 0 for k in ("scanned", "v3_45", "v2_45", "base_45", "v4_45", "v3_susp45", "v2_susp45", "v3_35", "v3_susp35")}
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sec = fidx / fps
            vs["scanned"] += 1
            tot["frames"] += 1

            r_b = m_base.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
            r_2 = m_v2.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
            r_3 = m_v3.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
            r_4 = m_v4.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]

            bb = [(float(b.conf), b.xyxy[0].tolist()) for b in r_b.boxes]
            b2 = [(float(b.conf), b.xyxy[0].tolist()) for b in r_2.boxes]
            b3 = [(float(b.conf), b.xyxy[0].tolist()) for b in r_3.boxes]
            b4_all = [(m_v4.names[int(b.cls)], float(b.conf), b.xyxy[0].tolist()) for b in r_4.boxes]
            b4f = [(c, b) for cn, c, b in b4_all if cn == "face"]
            lms = [b for cn, _, b in b4_all if cn in ("nose", "eye")]

            bb45 = [x for x in bb if x[0] >= 0.45]
            b245 = [x for x in b2 if x[0] >= 0.45]
            b345 = [x for x in b3 if x[0] >= 0.45]
            b4f45 = [x for x in b4f if x[0] >= 0.45]

            vs["base_45"] += len(bb45)
            vs["v2_45"] += len(b245)
            vs["v3_45"] += len(b345)
            vs["v4_45"] += len(b4f45)
            vs["v3_35"] += len(b3)
            tot["base_45"] += len(bb45)
            tot["v2_45"] += len(b245)
            tot["v3_45"] += len(b345)
            tot["v4_45"] += len(b4f45)
            tot["v3_35"] += len(b3)

            # Check suspicious V2 boxes @0.45
            for _, box2 in b245:
                if max([calc_iou(box2, x[1]) for x in bb45] + [0.0]) < 0.25 and max([calc_iou(box2, x[1]) for x in b4f45] + [0.0]) < 0.25 and not landmark_inside_box(box2, lms):
                    vs["v2_susp45"] += 1
                    tot["v2_susp45"] += 1

            # Check suspicious V3 boxes @0.45 and @0.35
            s_idxs45 = set()
            for idx_b, (c3, box3) in enumerate(b345):
                iou_b = max([calc_iou(box3, x[1]) for x in bb45] + [0.0])
                iou_4 = max([calc_iou(box3, x[1]) for x in b4f45] + [0.0])
                has_lm = landmark_inside_box(box3, lms)
                if iou_b < 0.25 and iou_4 < 0.25 and not has_lm:
                    vs["v3_susp45"] += 1
                    tot["v3_susp45"] += 1
                    s_idxs45.add(idx_b)
                else:
                    tot["v3_agree45"] += 1

            s_idxs35 = set()
            for idx_b, (c3, box3) in enumerate(b3):
                if max([calc_iou(box3, x[1]) for x in bb] + [0.0]) < 0.25 and max([calc_iou(box3, x[1]) for x in b4f] + [0.0]) < 0.25 and not landmark_inside_box(box3, lms):
                    vs["v3_susp35"] += 1
                    tot["v3_susp35"] += 1
                    s_idxs35.add(idx_b)

            if s_idxs35:
                top_c = max(b3[i][0] for i in s_idxs35)
                susp_events.append((top_c, vid_id, sec, b3, s_idxs35, bb, b4_all, frame.copy()))

            # Collect representative frames for 3-way field grid (V4 vs V2 vs V3)
            if vid_id == "rec_20261111_104223" and len(b245) > 0 and len(b4f45) == 0:
                sc_1042 = max(x[0] for x in b245) + (10.0 if len(b345) == 0 else 0.0)
                if "104223_fp_fixed" not in comp_frames or sc_1042 > comp_frames["104223_fp_fixed"][0]:
                    comp_frames["104223_fp_fixed"] = (sc_1042, vid_id, sec, b4_all, b245, b345, frame.copy())
            if vid_id == "rec_20261111_113922" and len(b245) > 0 and len(b4f45) == 0:
                sc_1139 = max(x[0] for x in b245) + (10.0 if len(b345) == 0 else 0.0)
                if "113922_bars_fixed" not in comp_frames or sc_1139 > comp_frames["113922_bars_fixed"][0]:
                    comp_frames["113922_bars_fixed"] = (sc_1139, vid_id, sec, b4_all, b245, b345, frame.copy())
            if vid_id in ("rec_20261111_120047", "rec_20261111_120218", "rec_20261111_115925") and len(b345) > 0:
                score = (max(x[0] for x in b345)) + (1.0 if len(b4f45) == 0 else 0.0)
                prev = comp_frames.get(vid_id)
                if prev is None or score > prev[0]:
                    comp_frames[vid_id] = (score, vid_id, sec, b4_all, b245, b345, frame.copy())
            if vid_id == "rec_20261111_110932" and len(b345) > 0 and vid_id not in comp_frames:
                comp_frames[vid_id] = (max(x[0] for x in b345), vid_id, sec, b4_all, b245, b345, frame.copy())

        cap.release()
        per_vid.append((vid_id, vs))

    # Build v3_field_suspicious_grid.jpg (up to 24 tiles, >=2s apart per video)
    susp_events.sort(key=lambda x: -x[0])
    picked_s, seen_ts = [], {}
    for ev in susp_events:
        conf, vid_id, sec, b3, s_idxs, bb, b4_all, frame = ev
        if any(abs(sec - ps) < 2.0 for ps in seen_ts.get(vid_id, [])):
            continue
        seen_ts.setdefault(vid_id, []).append(sec)
        picked_s.append(ev)
        if len(picked_s) >= 24:
            break

    if picked_s:
        tiles = []
        for conf, vid_id, sec, b3, s_idxs, bb, b4_all, frame in picked_s:
            vis = frame.copy()
            thick = max(2, int(round(min(vis.shape[:2]) / 240.0)))
            fscale = max(0.65, min(vis.shape[:2]) / 700.0)
            for i_b, (c3, bx3) in enumerate(b3):
                x1, y1, x2, y2 = [int(v) for v in bx3]
                col = (0, 0, 255) if i_b in s_idxs else (0, 220, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick + 1)
                cv2.putText(vis, f"V3 {c3:.2f}", (x1 + 4, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, fscale, col, thick, cv2.LINE_AA)
            v4_fc = sum(1 for x in b4_all if x[0] == "face")
            tiles.append(make_tile(vis, f"{vid_id.replace('rec_20261111_', '')} @{sec:.1f}s | V3-only", f"V3={conf:.2f} | Base={len(bb)} V4_f={v4_fc}"))
        cols = 4
        rc = int(np.ceil(len(tiles) / cols))
        tw_c, th_c = tiles[0].shape[1], tiles[0].shape[0]
        pad = 6
        sgrid = np.full((rc * (th_c + pad) + pad, cols * (tw_c + pad) + pad, 3), 12, dtype=np.uint8)
        for idx, t in enumerate(tiles):
            r, c = idx // cols, idx % cols
            sgrid[pad + r * (th_c + pad):pad + r * (th_c + pad) + th_c, pad + c * (tw_c + pad):pad + c * (tw_c + pad) + tw_c] = t
        cv2.imwrite(SUSP_GRID_PATH, sgrid)
    else:
        blank = np.full((220, 920, 3), 25, dtype=np.uint8)
        cv2.putText(blank, "ZERO suspicious unconfirmed V3 boxes across all 720 held-out frames!", (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 140), 2, cv2.LINE_AA)
        cv2.imwrite(SUSP_GRID_PATH, blank)

    # Build v3_vs_v2_vs_v4_field_grid.jpg (Left=V4 Live, Middle=V2, Right=V3)
    crows = []
    for key in ["104223_fp_fixed", "113922_bars_fixed", "rec_20261111_120047", "rec_20261111_120218", "rec_20261111_115925", "rec_20261111_110932"]:
        if key not in comp_frames:
            continue
        _, vid_id, sec, b4_all, b245, b345, frame = comp_frames[key]
        short_v = vid_id.replace("rec_20261111_", "")
        v4_vis = draw_v4_boxes(frame, [x for x in b4_all if x[1] >= 0.45])
        v2_vis = draw_boxes(frame, b245, color=(0, 180, 255), label_prefix="V2")
        v3_vis = draw_boxes(frame, b345, color=(0, 230, 0), label_prefix="V3")
        v4_fc = [round(x[1], 2) for x in b4_all if x[0] == "face" and x[1] >= 0.45]
        t0 = make_tile(v4_vis, f"V4 Live | {short_v} @{sec:.1f}s", f"face: {v4_fc}", tw=420, th=245)
        t1 = make_tile(v2_vis, f"V2 (YOLOv12n+AI) | {short_v} @{sec:.1f}s", f"face: {[round(x[0], 2) for x in b245]}", tw=420, th=245)
        t2 = make_tile(v3_vis, f"V3 (YOLO11s-SEAM-EMA) | {short_v} @{sec:.1f}s", f"face: {[round(x[0], 2) for x in b345]}", tw=420, th=245)
        sv = np.full((t0.shape[0], 8, 3), 12, dtype=np.uint8)
        crows.append(np.hstack([t0, sv, t1, sv, t2]))
    if crows:
        sh = np.full((8, crows[0].shape[1], 3), 12, dtype=np.uint8)
        fg = crows[0]
        for r in crows[1:]:
            fg = np.vstack([fg, sh, r])
        cv2.imwrite(FIELD_COMP_GRID_PATH, fg)

    # Render side-by-side MP4 on rec_20261111_120218_rgb.mp4 (V4 Live vs V3 YOLO11s-SEAM-EMA)
    src_vid = os.path.join(RECORDED_DIR, "rec_20261111_120218", "rec_20261111_120218_rgb.mp4")
    cap = cv2.VideoCapture(src_vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    for fc in ("mp4v", "avc1", "XVID"):
        w = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*fc), fps, (1920, 540))
        if w.isOpened():
            writer = w
            break
    v4_vid_sum, v3_vid_sum, nf = 0, 0, 0
    if writer is not None:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            r4 = m_v4.predict(fr, imgsz=640, conf=0.45, verbose=False)[0]
            r3 = m_v3.predict(fr, imgsz=640, conf=0.45, verbose=False)[0]
            v4i = [(m_v4.names[int(b.cls)], float(b.conf), b.xyxy[0].tolist()) for b in r4.boxes]
            v3i = [(float(b.conf), b.xyxy[0].tolist()) for b in r3.boxes]
            v4f = [x for x in v4i if x[0] == "face"]
            v4_vid_sum += len(v4f)
            v3_vid_sum += len(v3i)
            left = cv2.resize(draw_v4_boxes(fr, v4i), (960, 540), interpolation=cv2.INTER_AREA)
            right = cv2.resize(draw_boxes(fr, v3i, color=(0, 230, 0), label_prefix="face"), (960, 540), interpolation=cv2.INTER_AREA)
            cv2.rectangle(left, (0, 0), (960, 38), (20, 20, 20), -1)
            cv2.rectangle(right, (0, 0), (960, 38), (20, 20, 20), -1)
            cv2.putText(left, f"V4 Live (nose_eyes_face) | face={len(v4f)} {[round(x[1],2) for x in v4f]}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(right, f"V3 (YOLO11s-SEAM-EMA) | face={len(v3i)} {[round(x[0],2) for x in v3i]}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 140), 2, cv2.LINE_AA)
            writer.write(np.hstack([left, right]))
            nf += 1
        writer.release()
    cap.release()

    return {
        "totals": tot,
        "per_vid": per_vid,
        "susp_events_shown": picked_s,
        "vid_avg_v4": v4_vid_sum / max(1, nf),
        "vid_avg_v3": v3_vid_sum / max(1, nf),
    }


def build_chart(e_base, e_v2, e_v3):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=140)
    labels = ["Clean\nmAP@0.5", "Clean\nmAP50-95", "Occluded\nmAP@0.5", "Occluded\nRecall", "Occluded\nmAP50-95"]
    bv = [e_base["clean"]["map50"]*100, e_base["clean"]["map50_95"]*100, e_base["occluded"]["map50"]*100, e_base["occluded"]["recall"]*100, e_base["occluded"]["map50_95"]*100]
    v2v = [e_v2["clean"]["map50"]*100, e_v2["clean"]["map50_95"]*100, e_v2["occluded"]["map50"]*100, e_v2["occluded"]["recall"]*100, e_v2["occluded"]["map50_95"]*100]
    v3v = [e_v3["clean"]["map50"]*100, e_v3["clean"]["map50_95"]*100, e_v3["occluded"]["map50"]*100, e_v3["occluded"]["recall"]*100, e_v3["occluded"]["map50_95"]*100]

    x = np.arange(len(labels))
    w_bar = 0.25
    ax = axes[0]
    for off, vals, lbl, col in [(-w_bar, bv, "Baseline (face.pt)", "#7f7f7f"), (0, v2v, "V2 (YOLOv12n+AI)", "#e6ab02"), (w_bar, v3v, "V3 (YOLO11s-SEAM-EMA+ShapeIoU)", "#1b9e77")]:
        rs = ax.bar(x + off, vals, w_bar, label=lbl, color=col)
        for rect in rs:
            val = rect.get_height()
            ax.annotate(f"{val:.1f}", xy=(rect.get_x() + rect.get_width()/2, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 114)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Pig Face Detection Benchmark (3-Way)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    nl = ["Hard-Neg Img FP\n(conf >= 0.25)", "Hard-Neg Img FP\n(conf >= 0.45)"]
    nb = [e_base["hard_neg"]["fp25_rate"]*100, e_base["hard_neg"]["fp45_rate"]*100]
    n2 = [e_v2["hard_neg"]["fp25_rate"]*100, e_v2["hard_neg"]["fp45_rate"]*100]
    n3 = [e_v3["hard_neg"]["fp25_rate"]*100, e_v3["hard_neg"]["fp45_rate"]*100]
    x2 = np.arange(len(nl))
    for off, vals, lbl, col in [(-w_bar, nb, "Baseline", "#7f7f7f"), (0, n2, "V2", "#e6ab02"), (w_bar, n3, "V3 (YOLO11s-SEAM-EMA)", "#377eb8")]:
        rs = ax2.bar(x2 + off, vals, w_bar, label=lbl, color=col)
        for rect in rs:
            val = rect.get_height()
            ax2.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, max(28.0, max(nb + n2 + n3) + 8.0))
    ax2.set_xticks(x2)
    ax2.set_xticklabels(nl)
    ax2.set_ylabel("False Positive Rate (%)")
    ax2.set_title(f"Held-Out External Hard Negatives (Lower = Better, n={e_base['hard_neg']['n_images']})")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    register_custom_modules()
    patch_ultralytics_shape_iou()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_v4 = YOLO(V4_MODEL)
    m_v4.to(device)
    m_base = YOLO(BASELINE_MODEL)
    m_base.to(device)

    overlays, ai_hn_paths = prepare_all_ai_assets()
    hf_items = download_uniquedata_hf()
    info = prepare_v3_dataset(overlays, ai_hn_paths, hf_items, m_v4, m_base)

    print("[5/6] Training V3 (YOLO11s-SEAM-EMA + Shape-IoU, 20 epochs, COCO-transferred)...")
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
    for bs in (24, 16):
        try:
            m_v3 = YOLO(YAML_ARCH_PATH).load(YOLO11S_PRETRAINED)
            m_v3.train(
                data=info["yaml_train"],
                epochs=20,
                imgsz=640,
                batch=bs,
                lr0=0.0015,
                lrf=0.05,
                warmup_epochs=2.0,
                cos_lr=True,
                erasing=0.15,
                mosaic=0.7,
                close_mosaic=5,
                workers=0,
                cache=False,
                amp=True,
                device=0 if torch.cuda.is_available() else "cpu",
                project=RUNS_DIR,
                name="yolo11s_seam_ema_v3",
                exist_ok=True,
                plots=True,
                verbose=True,
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs == 24:
                print("  [WARN] CUDA OOM at batch=24, retrying with batch=16...")
                torch.cuda.empty_cache()
                shutil.rmtree(RUNS_DIR, ignore_errors=True)
                continue
            raise

    v3_weights = os.path.join(RUNS_DIR, "yolo11s_seam_ema_v3", "weights", "best.pt")
    if not os.path.exists(v3_weights):
        v3_weights = os.path.join(RUNS_DIR, "yolo11s_seam_ema_v3", "weights", "last.pt")

    print("[6/6] Running complete benchmark + 8-video held-out 20261111 farm audit...")
    e_base = eval_std_suite(BASELINE_MODEL, info, tag="Baseline face.pt")
    e_v2 = eval_std_suite(V2_MODEL, info, tag="V2 YOLOv12n+AI")
    e_v3 = eval_std_suite(v3_weights, info, tag="V3 YOLO11s-SEAM-EMA")
    build_chart(e_base, e_v2, e_v3)

    m_v2 = YOLO(V2_MODEL)
    m_v2.to(device)
    m_v3_best = YOLO(v3_weights)
    m_v3_best.to(device)
    field_audit = audit_heldout_farm_videos(m_base, m_v2, m_v3_best, m_v4)

    lines = [
        "================================================================================",
        "YOLO11s-SEAM-EMA + Shape-IoU (V3) Full Evaluation & Field Video Audit",
        "================================================================================",
        f"Architecture     : YOLO11s (COCO pretrained) + EMA + SEAM (YOLO-FaceV2) + Shape-IoU (DGS-YOLO)",
        f"V3 Weights       : {v3_weights}",
        f"GPU Device       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"Total Elapsed    : {time.time() - t_start:.1f}s",
        "",
        "--- 1. V3 Training Dataset Summary (Zero-Leakage) ---",
        f"Train Total      : {info['train_total']} imgs ({info['train_clean']} real black + {info['train_ai_fence']} AI fence @{info['mean_ai_occ']*100:.1f}% + {info['train_white_breed']} white/multi-breed + {info['train_hard_neg']} hard_neg)",
        f"HF UniqueData    : {len(hf_items)} real multi-breed images (+ {len(hf_items)} AI-fence copies)",
        f"Real Neg Source  : 20261110 lab/office/human clips + 103456 zero-pig frames (8x 20261111 farm videos strictly held out!)",
        "",
        "--- 2. Benchmark Comparison (Baseline face.pt vs V2 vs V3 YOLO11s-SEAM-EMA) ---",
        f"{'Benchmark & Metric':<36} | {'Baseline':>10} | {'V2':>10} | {'V3 (Ours)':>10} | {'V3-Base':>9}",
        "-" * 88,
    ]

    def row(label, b, v2, v3, is_pct=True):
        if is_pct:
            lines.append(f"{label:<36} | {b*100:9.2f}% | {v2*100:9.2f}% | {v3*100:9.2f}% | {(v3-b)*100:+8.2f}%")
        else:
            lines.append(f"{label:<36} | {b:10d} | {v2:10d} | {v3:10d} | {v3-b:+9d}")

    row("Clean Test - Precision", e_base["clean"]["precision"], e_v2["clean"]["precision"], e_v3["clean"]["precision"])
    row("Clean Test - Recall", e_base["clean"]["recall"], e_v2["clean"]["recall"], e_v3["clean"]["recall"])
    row("Clean Test - mAP@0.5", e_base["clean"]["map50"], e_v2["clean"]["map50"], e_v3["clean"]["map50"])
    row("Clean Test - mAP@0.5:0.95", e_base["clean"]["map50_95"], e_v2["clean"]["map50_95"], e_v3["clean"]["map50_95"])
    lines.append("-" * 88)
    row("Occluded Test - Precision", e_base["occluded"]["precision"], e_v2["occluded"]["precision"], e_v3["occluded"]["precision"])
    row("Occluded Test - Recall", e_base["occluded"]["recall"], e_v2["occluded"]["recall"], e_v3["occluded"]["recall"])
    row("Occluded Test - mAP@0.5", e_base["occluded"]["map50"], e_v2["occluded"]["map50"], e_v3["occluded"]["map50"])
    row("Occluded Test - mAP@0.5:0.95", e_base["occluded"]["map50_95"], e_v2["occluded"]["map50_95"], e_v3["occluded"]["map50_95"])
    lines.append("-" * 88)
    row("Hard-Neg Img FP Rate (conf>=0.25)", e_base["hard_neg"]["fp25_rate"], e_v2["hard_neg"]["fp25_rate"], e_v3["hard_neg"]["fp25_rate"])
    row("Hard-Neg Box FPs (conf>=0.25)", e_base["hard_neg"]["fp25_boxes"], e_v2["hard_neg"]["fp25_boxes"], e_v3["hard_neg"]["fp25_boxes"], is_pct=False)
    row("Hard-Neg Img FP Rate (conf>=0.45)", e_base["hard_neg"]["fp45_rate"], e_v2["hard_neg"]["fp45_rate"], e_v3["hard_neg"]["fp45_rate"])
    row("Hard-Neg Box FPs (conf>=0.45)", e_base["hard_neg"]["fp45_boxes"], e_v2["hard_neg"]["fp45_boxes"], e_v3["hard_neg"]["fp45_boxes"], is_pct=False)
    lines.append("")
    lines.append("--- 3. Unseen Out-of-Distribution AI Fence Photos (face boxes @ conf>=0.20) ---")
    for fn in ["02_yolo_fullframe_metal_bars.png", "01_reid_295-6_vertical_fence.png"]:
        lines.append(f"  {fn}: Base={e_base['ai_fence'].get(fn, [])} | V2={e_v2['ai_fence'].get(fn, [])} | V3={e_v3['ai_fence'].get(fn, [])}")
    lines.append(f"--- Hard Negative FP Details (@0.25) ---")
    lines.append(f"  V3: {e_v3['hard_neg']['examples']}")
    lines.append("")
    ft = field_audit["totals"]
    lines.extend([
        f"--- 4. Held-Out Real Farm Video Audit (8 unseen 20261111 videos, {ft['frames']} frames @ conf>=0.45) ---",
        f"Totals @0.45 : V3_boxes={ft['v3_45']} (V3_susp={ft['v3_susp45']}, agree={(ft['v3_agree45']/max(1,ft['v3_45']))*100:.1f}%) | V2_boxes={ft['v2_45']} (V2_susp={ft['v2_susp45']}) | Base={ft['base_45']} | V4_Live={ft['v4_45']}",
        f"Totals @0.35 : V3_boxes={ft['v3_35']} (V3_susp={ft['v3_susp35']})",
        f"120218 Full Video (570f @0.45) avg face/frame : V4_Live={field_audit['vid_avg_v4']:.2f} | V3={field_audit['vid_avg_v3']:.2f}",
        "",
        f"{'Held-Out Video':<22} | {'Fr':>3} | {'Base':>4} {'V4':>4} {'V2(Susp)':>10} {'V3(Susp@45)':>13} {'V3@35(Susp)':>13}",
        "-" * 78,
    ])
    for vid_id, vs in field_audit["per_vid"]:
        lines.append(
            f"{vid_id:<22} | {vs['scanned']:3d} | {vs['base_45']:4d} {vs['v4_45']:4d} "
            f"{vs['v2_45']:4d}({vs['v2_susp45']:2d})  {vs['v3_45']:5d}({vs['v3_susp45']:2d})   {vs['v3_35']:5d}({vs['v3_susp35']:2d})"
        )
    lines.append("")
    lines.append(f"Top Suspicious V3 Boxes on Held-Out 20261111 Videos ({len(field_audit['susp_events_shown'])} shown):")
    for idx, ev in enumerate(field_audit["susp_events_shown"], 1):
        conf, vid_id, sec, b3, s_idxs, bb, b4_all, _ = ev
        lines.append(f"  [{idx:02d}] {vid_id} @{sec:5.1f}s | V3_susp_conf={conf:.3f} | Base={len(bb)} | V4_f={sum(1 for x in b4_all if x[0]=='face')}")
    lines.append("=" * 80)

    rep = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print("\n" + rep)
    print(f"\nDone! All V3 deliverables saved in {EXP_V3_DIR}")


if __name__ == "__main__":
    main()
