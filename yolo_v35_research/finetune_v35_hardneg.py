import torch  # Import first on Windows to avoid WinError 4551
import os
import sys
import json
import time
import shutil
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V3_CODE_DIR = os.path.join(BUNDLE_ROOT, "01_code", "yolo_v3_occlusion")
for p in (SCRIPT_DIR, V3_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from custom_v35_modules import register_v35_modules, patch_ultralytics_focaler_shape_wiou
import custom_occlusion_modules
import train_and_eval_v35 as v35_base

LAB_NEG_DIR = os.path.join(BUNDLE_ROOT, "04_videos", "20261110_lab_negatives_rgb_thm")
DATA_V35_DIR = v35_base.DATA_V35_DIR
DATA_V35_HN_DIR = os.path.join(v35_base.OUT_ROOT, "dataset_v35_hn")
RUNS_DIR = v35_base.RUNS_DIR
OUT_ROOT = v35_base.OUT_ROOT


def apply_ai_fence_on_negative(img_bgr, overlays_bgra, rng, zoom_closeup=False):
    """Composite training fence overlays (and optional blurred close-up zoom) on a zero-pig frame."""
    h, w = img_bgr.shape[:2]
    base = img_bgr.copy()
    if zoom_closeup:
        # Random close-up crop to simulate near-camera blurred bars / sweatpants without pigs
        scale = float(rng.uniform(0.35, 0.65))
        cw, ch = max(64, int(w * scale)), max(64, int(h * scale))
        x0 = int(rng.integers(0, max(1, w - cw)))
        y0 = int(rng.integers(0, max(1, h - ch)))
        base = cv2.resize(base[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    name = list(overlays_bgra.keys())[int(rng.integers(0, len(overlays_bgra)))]
    ov = overlays_bgra[name]
    ys, xs = np.where(ov[:, :, 3] > 20)
    if len(xs) == 0:
        return base
    patch = ov[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    s_min, s_max = (0.85, 1.35) if zoom_closeup else (0.55, 1.25)
    nw = max(48, int(w * float(rng.uniform(s_min, s_max))))
    nh = max(48, int(h * float(rng.uniform(s_min, s_max))))
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)

    angle = float(rng.uniform(-18.0, 18.0))
    M = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
    patch = cv2.warpAffine(
        patch, M, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    patch[:, :, :3] = np.clip(
        patch[:, :, :3].astype(np.float32) * float(rng.uniform(0.70, 1.25)) + float(rng.integers(-18, 19)),
        0, 255,
    ).astype(np.uint8)
    k = 5 if (zoom_closeup and float(rng.random()) < 0.65) else 3
    patch = cv2.GaussianBlur(patch, (k, k), 0)

    ox = int(rng.integers(-nw // 4, max(1, w - (3 * nw) // 4)))
    oy = int(rng.integers(-nh // 4, max(1, h - (3 * nh) // 4)))
    canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    sx1, sy1 = max(0, -ox), max(0, -oy)
    dx1, dy1 = max(0, ox), max(0, oy)
    cw = min(nw - sx1, w - dx1)
    ch = min(nh - sy1, h - dy1)
    if cw <= 0 or ch <= 0:
        return base
    canvas_rgba[dy1:dy1 + ch, dx1:dx1 + cw] = patch[sy1:sy1 + ch, sx1:sx1 + cw]
    a_map = canvas_rgba[:, :, 3].astype(np.float32) / 255.0
    out_f = base.astype(np.float32)
    sh = cv2.GaussianBlur(
        cv2.warpAffine(a_map, np.float32([[1, 0, 6], [0, 1, 6]]), (w, h), borderValue=0.0),
        (13, 13), 0,
    ) * 0.32
    out_f = out_f * (1.0 - sh[..., None])
    a3 = a_map[..., None]
    out_f = (1.0 - a3) * out_f + a3 * canvas_rgba[:, :, :3].astype(np.float32)
    return np.clip(out_f, 0, 255).astype(np.uint8)


def write_empty_label(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def build_hardneg_dataset(pre_hn_weights, train_overlays, device=0):
    print("[1/4] Building strict zero-leakage V3.5-HN dataset (20261110 negatives + fence-only breakers)...")
    if os.path.exists(DATA_V35_HN_DIR):
        shutil.rmtree(DATA_V35_HN_DIR)
    tr_i = os.path.join(DATA_V35_HN_DIR, "train", "images")
    tr_l = os.path.join(DATA_V35_HN_DIR, "train", "labels")
    va_i = os.path.join(DATA_V35_HN_DIR, "val", "images")
    va_l = os.path.join(DATA_V35_HN_DIR, "val", "labels")
    for d in (tr_i, tr_l, va_i, va_l):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(35359)

    # 1. Hard-link existing dataset_v35 train & collect V3 empty negatives
    src_tr_i = os.path.join(DATA_V35_DIR, "train", "images")
    src_tr_l = os.path.join(DATA_V35_DIR, "train", "labels")
    base_tr_cnt = 0
    v3_neg_paths = []
    for fn in sorted(os.listdir(src_tr_i)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        sp_i = os.path.join(src_tr_i, fn)
        sp_l = os.path.join(src_tr_l, stem + ".txt")
        if os.path.exists(sp_l):
            v35_base.link_or_copy(sp_i, os.path.join(tr_i, fn))
            v35_base.link_or_copy(sp_l, os.path.join(tr_l, stem + ".txt"))
            base_tr_cnt += 1
            if os.path.getsize(sp_l) == 0:
                v3_neg_paths.append(sp_i)

    # 2. Hard-link existing dataset_v35 val
    src_va_i = os.path.join(DATA_V35_DIR, "val", "images")
    src_va_l = os.path.join(DATA_V35_DIR, "val", "labels")
    base_va_cnt = 0
    for fn in sorted(os.listdir(src_va_i)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        sp_l = os.path.join(src_va_l, stem + ".txt")
        if os.path.exists(sp_l):
            v35_base.link_or_copy(os.path.join(src_va_i, fn), os.path.join(va_i, fn))
            v35_base.link_or_copy(sp_l, os.path.join(va_l, stem + ".txt"))
            base_va_cnt += 1

    # 3. Split 20261110_lab_negatives_rgb_thm (strictly zero 20261111 videos!): first 20 train, last 3 val
    vid_dirs = sorted(d for d in os.listdir(LAB_NEG_DIR) if os.path.isdir(os.path.join(LAB_NEG_DIR, d)))
    assert len(vid_dirs) >= 23, f"Expected >=23 lab negative videos, found {len(vid_dirs)}"
    train_vids = vid_dirs[:20]
    val_vids = vid_dirs[-3:]

    lab_tr_frames = []
    mined_candidates = []
    m_probe = YOLO(pre_hn_weights)

    for vid in train_vids:
        vp = os.path.join(LAB_NEG_DIR, vid, f"{vid}_rgb.mp4")
        cap = cv2.VideoCapture(vp)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot <= 0:
            cap.release()
            continue
        uniform_idxs = set(int(x) for x in np.linspace(0, max(0, tot - 1), num=15, dtype=int))
        mine_idxs = [int(x) for x in np.linspace(0, max(0, tot - 1), num=40, dtype=int)]
        seen_idx = set()
        for fidx in mine_idxs:
            if fidx in seen_idx:
                continue
            seen_idx.add(fidx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if fidx in uniform_idxs:
                stem = f"hn10_u_{vid}_{fidx:05d}"
                out_img = os.path.join(tr_i, stem + ".jpg")
                v35_base.safe_imwrite(out_img, fr)
                write_empty_label(os.path.join(tr_l, stem + ".txt"))
                lab_tr_frames.append(out_img)
            r = m_probe.predict(fr, imgsz=640, conf=0.15, device=device, verbose=False)[0]
            if len(r.boxes) > 0:
                max_c = max(float(b.conf) for b in r.boxes)
                mined_candidates.append((max_c, vid, fidx, fr.copy()))
        cap.release()

    del m_probe
    torch.cuda.empty_cache()

    # Keep top 150 highest-confidence FP mined frames
    mined_candidates.sort(key=lambda x: x[0], reverse=True)
    mined_cnt = 0
    for max_c, vid, fidx, fr in mined_candidates[:150]:
        stem = f"hn10_fp_{vid}_{fidx:05d}"
        out_img = os.path.join(tr_i, stem + ".jpg")
        if not os.path.exists(out_img):
            v35_base.safe_imwrite(out_img, fr)
            write_empty_label(os.path.join(tr_l, stem + ".txt"))
            lab_tr_frames.append(out_img)
        mined_cnt += 1

    # 4. Fence-only shortcut-breaker negatives (250 full + 60 close-up zoom) using TRAIN_OVERLAY_NAMES only
    neg_pool = v3_neg_paths + lab_tr_frames
    fence_neg_cnt = 0
    for idx in range(250):
        src_p = neg_pool[int(rng.integers(0, len(neg_pool)))]
        im = v35_base.safe_imread_bgr(src_p)
        if im is None:
            continue
        comp = apply_ai_fence_on_negative(im, train_overlays, rng, zoom_closeup=False)
        stem = f"hn_fence_neg_{idx:04d}"
        v35_base.safe_imwrite(os.path.join(tr_i, stem + ".jpg"), comp)
        write_empty_label(os.path.join(tr_l, stem + ".txt"))
        fence_neg_cnt += 1

    closeup_neg_cnt = 0
    for idx in range(60):
        src_p = neg_pool[int(rng.integers(0, len(neg_pool)))]
        im = v35_base.safe_imread_bgr(src_p)
        if im is None:
            continue
        comp = apply_ai_fence_on_negative(im, train_overlays, rng, zoom_closeup=True)
        stem = f"hn_fence_zoom_{idx:04d}"
        v35_base.safe_imwrite(os.path.join(tr_i, stem + ".jpg"), comp)
        write_empty_label(os.path.join(tr_l, stem + ".txt"))
        closeup_neg_cnt += 1

    # 5. Build 60 zero-pig val negatives from last 3 20261110 videos (30 clean + 30 TRAIN_OVERLAY_NAMES)
    val_raw = []
    for vid in val_vids:
        vp = os.path.join(LAB_NEG_DIR, vid, f"{vid}_rgb.mp4")
        cap = cv2.VideoCapture(vp)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot > 0:
            for fidx in np.linspace(0, max(0, tot - 1), num=12, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ok, fr = cap.read()
                if ok and fr is not None:
                    val_raw.append((f"{vid}_{int(fidx):05d}", fr))
        cap.release()
    rng.shuffle(val_raw)
    val_clean_cnt, val_fence_cnt = 0, 0
    for idx, (tag, fr) in enumerate(val_raw[:30]):
        stem = f"val_hn10_clean_{tag}_{idx:02d}"
        v35_base.safe_imwrite(os.path.join(va_i, stem + ".jpg"), fr)
        write_empty_label(os.path.join(va_l, stem + ".txt"))
        val_clean_cnt += 1
    for idx, (tag, fr) in enumerate(val_raw[:30]):
        comp = apply_ai_fence_on_negative(fr, train_overlays, rng, zoom_closeup=(idx % 3 == 0))
        stem = f"val_hn10_fence_{tag}_{idx:02d}"
        v35_base.safe_imwrite(os.path.join(va_i, stem + ".jpg"), comp)
        write_empty_label(os.path.join(va_l, stem + ".txt"))
        val_fence_cnt += 1

    yaml_p = os.path.join(DATA_V35_HN_DIR, "data_v35_hn.yaml")
    abs_d = os.path.abspath(DATA_V35_HN_DIR).replace("\\", "/")
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_d}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['face']\n")

    uniform_added = len([p for p in lab_tr_frames if "hn10_u_" in os.path.basename(p)])
    stats = {
        "base_train": base_tr_cnt,
        "lab10_uniform_neg": uniform_added,
        "lab10_mined_fp_neg": mined_cnt,
        "lab10_mined_candidates_total": len(mined_candidates),
        "fence_only_neg": fence_neg_cnt,
        "fence_zoom_neg": closeup_neg_cnt,
        "train_total": base_tr_cnt + uniform_added + mined_cnt + fence_neg_cnt + closeup_neg_cnt,
        "base_val": base_va_cnt,
        "val_neg_clean": val_clean_cnt,
        "val_neg_fence": val_fence_cnt,
        "val_total": base_va_cnt + val_clean_cnt + val_fence_cnt,
        "train_vids": train_vids,
        "val_vids": val_vids,
        "yaml": yaml_p,
    }
    print("  Dataset V3.5-HN stats:", json.dumps(stats, indent=2))
    return stats


def format_report(evals, ds_stats, gate_passed, fp16_mb, elapsed_sec):
    b, v2, v3 = evals["Baseline"], evals["V2"], evals["V3"]
    v35_pre = evals["V3.5_preHN"]
    v35_hn = evals["V3.5_HN"]
    lines = [
        "=" * 112,
        "Track 2 (V3.5-HN) YOLO11m-BiFPN-EMA-SEAM + Focaler-Shape-WIoUv3 (Hard-Neg Hardened) Report",
        "=" * 112,
        f"Train Total: {ds_stats['train_total']} | Mined FP Neg: {ds_stats['lab10_mined_fp_neg']} | Fence Neg: {ds_stats['fence_only_neg']}+{ds_stats['fence_zoom_neg']} | Val Neg: {ds_stats['val_neg_clean']+ds_stats['val_neg_fence']}",
        f"Gate Passed: {gate_passed} | FP16 Weight: {fp16_mb:.2f} MB | Elapsed: {elapsed_sec:.1f}s",
        "",
        f"{'Metric':<36} | {'Baseline':>10} | {'V2':>10} | {'V3':>10} | {'V3.5-pre':>12} | {'V3.5-HN':>12}",
        "-" * 108,
        f"{'Clean Test mAP@0.5':<36} | {b['clean']['map50']*100:>9.2f}% | {v2['clean']['map50']*100:>9.2f}% | {v3['clean']['map50']*100:>9.2f}% | {v35_pre['clean']['map50']*100:>11.2f}% | {v35_hn['clean']['map50']*100:>11.2f}%",
        f"{'Clean Test mAP@0.5:0.95':<36} | {b['clean']['map50_95']*100:>9.2f}% | {v2['clean']['map50_95']*100:>9.2f}% | {v3['clean']['map50_95']*100:>9.2f}% | {v35_pre['clean']['map50_95']*100:>11.2f}% | {v35_hn['clean']['map50_95']*100:>11.2f}%",
        f"{'Occluded Test mAP@0.5':<36} | {b['occluded']['map50']*100:>9.2f}% | {v2['occluded']['map50']*100:>9.2f}% | {v3['occluded']['map50']*100:>9.2f}% | {v35_pre['occluded']['map50']*100:>11.2f}% | {v35_hn['occluded']['map50']*100:>11.2f}%",
        f"{'Occluded Test mAP@0.5:0.95':<36} | {b['occluded']['map50_95']*100:>9.2f}% | {v2['occluded']['map50_95']*100:>9.2f}% | {v3['occluded']['map50_95']*100:>9.2f}% | {v35_pre['occluded']['map50_95']*100:>11.2f}% | {v35_hn['occluded']['map50_95']*100:>11.2f}%",
        f"{'Hard-Neg Img FP@0.45':<36} | {b['hard_neg']['fp45_rate']*100:>9.2f}% | {v2['hard_neg']['fp45_rate']*100:>9.2f}% | {v3['hard_neg']['fp45_rate']*100:>9.2f}% | {v35_pre['hard_neg']['fp45_rate']*100:>11.2f}% | {v35_hn['hard_neg']['fp45_rate']*100:>11.2f}%",
        f"{'Field 120218 (Black Fence) @0.45':<36} | {b['field']['rec_20261111_120218']['boxes_45']:>10} | {v2['field']['rec_20261111_120218']['boxes_45']:>10} | {v3['field']['rec_20261111_120218']['boxes_45']:>10} | {v35_pre['field']['rec_20261111_120218']['boxes_45']:>12} | {v35_hn['field']['rec_20261111_120218']['boxes_45']:>12}",
        f"{'Field 113922 (White Fence) @0.45':<36} | {b['field']['rec_20261111_113922']['boxes_45']:>10} | {v2['field']['rec_20261111_113922']['boxes_45']:>10} | {v3['field']['rec_20261111_113922']['boxes_45']:>10} | {v35_pre['field']['rec_20261111_113922']['boxes_45']:>12} | {v35_hn['field']['rec_20261111_113922']['boxes_45']:>12}",
        f"{'Field 113613 (White Fence) @0.45':<36} | {b['field']['rec_20261111_113613']['boxes_45']:>10} | {v2['field']['rec_20261111_113613']['boxes_45']:>10} | {v3['field']['rec_20261111_113613']['boxes_45']:>10} | {v35_pre['field']['rec_20261111_113613']['boxes_45']:>12} | {v35_hn['field']['rec_20261111_113613']['boxes_45']:>12}",
        f"{'Field 104223 (Human Neg) @0.45':<36} | {b['field']['rec_20261111_104223']['boxes_45']:>10} | {v2['field']['rec_20261111_104223']['boxes_45']:>10} | {v3['field']['rec_20261111_104223']['boxes_45']:>10} | {v35_pre['field']['rec_20261111_104223']['boxes_45']:>12} | {v35_hn['field']['rec_20261111_104223']['boxes_45']:>12}",
        f"{'OOD White Metal Bars (02_yolo)':<36} | {str([x[0] for x in b['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v2['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v3['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>10} | {str([x[0] for x in v35_pre['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>12} | {str([x[0] for x in v35_hn['ai_hard']['02_yolo_fullframe_metal_bars.png']]):>12}",
        f"{'OOD Black Vertical Fence (01_reid)':<36} | {str([x[0] for x in b['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v2['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v3['ai_hard']['01_reid_295-6_vertical_fence.png']]):>10} | {str([x[0] for x in v35_pre['ai_hard']['01_reid_295-6_vertical_fence.png']]):>12} | {str([x[0] for x in v35_hn['ai_hard']['01_reid_295-6_vertical_fence.png']]):>12}",
        "=" * 112,
    ]
    return "\n".join(lines)


def main():
    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    register_v35_modules()
    custom_occlusion_modules.register_custom_modules()
    patch_ultralytics_focaler_shape_wiou()

    device = 0 if torch.cuda.is_available() else "cpu"
    pre_hn_best = os.path.join(RUNS_DIR, "yolo11m_v35", "weights", "best.pt")
    assert os.path.exists(pre_hn_best), f"Missing pre-HN best weights: {pre_hn_best}"

    train_overlays = v35_base.load_rgba_overlays(v35_base.TRAIN_OVERLAY_NAMES)
    ds_stats = build_hardneg_dataset(pre_hn_best, train_overlays, device=device)

    print("[2/4] Fine-tuning YOLO11m-BiFPN-EMA-SEAM on V3.5-HN (12 epochs, lr0=0.0003)...")
    hn_run_dir = os.path.join(RUNS_DIR, "yolo11m_v35_hn")
    shutil.rmtree(hn_run_dir, ignore_errors=True)
    for bs in (16, 12):
        try:
            m = YOLO(pre_hn_best)
            m.train(
                data=ds_stats["yaml"],
                epochs=12,
                imgsz=640,
                batch=bs,
                lr0=0.0003,
                lrf=0.05,
                warmup_epochs=1.0,
                cos_lr=True,
                erasing=0.15,
                mosaic=0.5,
                close_mosaic=4,
                workers=0,
                cache=False,
                amp=True,
                device=device,
                project=RUNS_DIR,
                name="yolo11m_v35_hn",
                exist_ok=True,
                plots=True,
                verbose=True,
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs == 16:
                print("  CUDA OOM at batch=16, retrying with batch=12...")
                torch.cuda.empty_cache()
                shutil.rmtree(hn_run_dir, ignore_errors=True)
                continue
            raise

    hn_best = os.path.join(hn_run_dir, "weights", "best.pt")
    if not os.path.exists(hn_best):
        hn_best = os.path.join(hn_run_dir, "weights", "last.pt")

    print("[3/4] Running apples-to-apples evaluation & held-out 20261111 audit for V3.5-HN...")
    cached_p = os.path.join(OUT_ROOT, "v35_eval_summary.json")
    if os.path.exists(cached_p):
        with open(cached_p, "r", encoding="utf-8") as f:
            prev = json.load(f)
        pe = prev["evals"]
        evals = {
            "Baseline": pe["Baseline"],
            "V2": pe["V2"],
            "V3": pe["V3"],
            "V3.5_preHN": pe.get("V3.5_preHN", pe["V3.5"]),
        }
    else:
        evals = {
            "Baseline": v35_base.eval_detector(v35_base.BASELINE_MODEL, "Baseline face.pt", device=device),
            "V2": v35_base.eval_detector(v35_base.V2_MODEL, "V2 YOLOv12n+AI", device=device),
            "V3": v35_base.eval_detector(v35_base.V3_MODEL, "V3 YOLO11s-SEAM-EMA", device=device),
            "V3.5_preHN": v35_base.eval_detector(pre_hn_best, "V3.5-preHN", device=device),
        }

    evals["V3.5_HN"] = v35_base.eval_detector(hn_best, "V3.5-HN YOLO11m-BiFPN-WIoUv3", device=device)

    e3, ehn = evals["V3"], evals["V3.5_HN"]
    clean_ok = ehn["clean"]["map50"] >= 0.990
    human_ok = ehn["field"]["rec_20261111_104223"]["boxes_45"] <= 2
    gain_ok = (ehn["occluded"]["map50_95"] >= 0.6834) or (
        len(ehn["ai_hard"]["02_yolo_fullframe_metal_bars.png"]) > 0 and ehn["occluded"]["map50_95"] >= 0.675
    )
    gate_passed = bool(clean_ok and human_ok and gain_ok)

    tmp_fp16 = os.path.join(OUT_ROOT, "yolo_v35_hn_fp16.pt")
    fp32_mb, fp16_mb = v35_base.export_fp16_yolo(hn_best, tmp_fp16)

    elapsed = time.time() - t0
    rep = format_report(evals, ds_stats, gate_passed, fp16_mb, elapsed)
    with open(os.path.join(OUT_ROOT, "v35_hn_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")

    evals_compat = {
        "Baseline": evals["Baseline"],
        "V2": evals["V2"],
        "V3": evals["V3"],
        "V3.5": evals["V3.5_HN"],
        "V3.5_preHN": evals["V3.5_preHN"],
        "V3.5_HN": evals["V3.5_HN"],
    }
    hn_summary = {
        "elapsed_sec": elapsed,
        "dataset_stats": ds_stats,
        "weights_best": hn_best,
        "fp32_mb": fp32_mb,
        "fp16_mb": fp16_mb,
        "gate_passed": gate_passed,
        "gate_details": {"clean_ok": clean_ok, "human_ok": human_ok, "gain_ok": gain_ok},
        "evals": evals_compat,
    }
    with open(os.path.join(OUT_ROOT, "v35_hn_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(hn_summary, f, indent=2)

    if gate_passed and fp16_mb < 50.0:
        print("[4/4] Gate PASSED! Promoting V3.5-HN -> canonical V3.5 release & regenerating figures...")
        shutil.copy2(tmp_fp16, os.path.join(OUT_ROOT, "yolo_v35_bifpn_seam_ema_fp16.pt"))
        with open(os.path.join(OUT_ROOT, "v35_eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(hn_summary, f, indent=2)
        with open(os.path.join(OUT_ROOT, "v35_eval_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep + "\n")
        v35_base.plot_v35_figures(evals_compat, hn_best)
    else:
        print(f"[4/4] Gate status: gate_passed={gate_passed}, fp16_mb={fp16_mb:.2f} MB. Keeping v35_hn_* separate.")

    print("\n" + rep)


if __name__ == "__main__":
    main()
