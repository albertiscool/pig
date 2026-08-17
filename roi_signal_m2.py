# -*- coding: utf-8 -*-
"""
(B2) roi_signal_m2.py — 冷區分割型（改良版：G通道 + 最大連通區塊）
流程：邊界縮減 → (可選)預模糊 → CLAHE → 分割(預設 Otsu) → 形態學 → 取最大塊 → 面積/亮度 → 時間處理
"""

import os, cv2, csv, json, math, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 修正處：強制使用 Agg 後端以避免 Qt 插件錯誤 ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ----------------------------------------------

from scipy.signal import butter, filtfilt, savgol_filter

# -----------------------------
# 基本工具
# -----------------------------
def video_info(cap):
    if not cap or not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return dict(fps=fps, nframes=nframes, width=w, height=h)

def frame_time(idx, fps):
    return float(idx) / float(fps) if fps > 0 else 0.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

def clamp_bbox(x1, y1, x2, y2, W, H):
    x1c = int(clamp(math.floor(x1), 0, W-1))
    y1c = int(clamp(math.floor(y1), 0, H-1))
    x2c = int(clamp(math.ceil (x2), 0, W-1))
    y2c = int(clamp(math.ceil (y2), 0, H-1))
    return x1c, y1c, x2c, y2c

def load_roi_csv(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                k = int(r["frame_th"])
            except Exception:
                continue
            mapping[k] = r  # 這裡會保留所有欄位，包含 pig_id
    return mapping

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        high = min(0.999, low + 1e-3)
    b, a = butter(order, [low, high], btype='band')
    return b, a

# -----------------------------
# B2 分割與後處理
# -----------------------------
def make_clahe(clip, tile_w, tile_h):
    tile_w = max(1, int(tile_w)); tile_h = max(1, int(tile_h))
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile_w, tile_h))

def apply_threshold(roi_gray_8u, mode="otsu", win=21, C=3):
    win = int(win)
    if win % 2 == 0: win += 1
    if mode == "adaptive_mean":
        return cv2.adaptiveThreshold(roi_gray_8u, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, win, int(C))
    elif mode == "adaptive_gaussian":
        return cv2.adaptiveThreshold(roi_gray_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, win, int(C))
    else:
        _, th = cv2.threshold(roi_gray_8u, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th

def morph_cleanup(mask, open_px=3, close_px=5):
    m = mask.copy()
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def largest_component(mask):
    if mask is None or mask.size == 0:
        return None, 0.0, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_id = 1 + np.argmax(areas)
    largest = (labels == max_id).astype(np.uint8) * 255
    area_ratio = float(areas.max()) / float(mask.shape[0] * mask.shape[1])
    comps = int(num_labels - 1)
    return largest, area_ratio, comps

# -----------------------------
# 主流程
# -----------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    roi_map = load_roi_csv(args.roi_csv)
    cap = cv2.VideoCapture(args.thermal)
    meta = video_info(cap)
    if not meta or meta["nframes"] == 0 or meta["fps"] <= 0:
        raise RuntimeError("無法開啟熱像影片或 FPS/幀數異常。")

    fps, W, H, nframes = meta["fps"], meta["width"], meta["height"], meta["nframes"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_out_path = os.path.join(args.out_dir, "thermal_with_roi_m2.mp4")
    writer = cv2.VideoWriter(vid_out_path, fourcc, fps, (W, H))

    frame_idx = np.arange(nframes, dtype=np.int32)
    time_sec = frame_idx / fps
    valid = np.zeros(nframes, dtype=np.int32)
    clip_reason = np.array([""] * nframes, dtype=object)
    cold_area_ratio = np.full(nframes, np.nan, dtype=np.float64)
    cold_mean = np.full(nframes, np.nan, dtype=np.float64)

    clahe = make_clahe(args.clahe_clip, args.clahe_tile[0], args.clahe_tile[1])
    inner = args.inner_shrink

    with tqdm(total=nframes, desc="(B2) ROI sampling (cold mask)", unit="frame") as pbar:
        for i in range(nframes):
            ok, frame = cap.read()
            if not ok: break

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = frame[:, :, 1] if args.gray_mode in ["green", "auto"] else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
            else:
                gray = frame.copy()
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            r = roi_map.get(i, None)
            if r is None:
                clip_reason[i] = "NoCSV"
                writer.write(vis); pbar.update(1); continue

            if str(r.get("valid","0")).lower() not in ("1","true"):
                clip_reason[i] = r.get("clip_reason","Invalid")
                writer.write(vis); pbar.update(1); continue

            try:
                x1, y1 = int(float(r["x1_th"])), int(float(r["y1_th"]))
                x2, y2 = int(float(r["x2_th"])), int(float(r["y2_th"]))
            except:
                clip_reason[i] = "CSVCoordErr"
                writer.write(vis); pbar.update(1); continue

            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2, lineType=cv2.LINE_AA)
            xi1, yi1, xi2, yi2 = clamp_bbox(x1+inner, y1+inner, x2-inner, y2-inner, W, H)
            
            if (xi2 - xi1) < 1 or (yi2 - yi1) < 1:
                clip_reason[i] = "ShrinkTooSmall"; writer.write(vis); pbar.update(1); continue

            roi = gray[yi1:yi2, xi1:xi2]
            if roi.size == 0: clip_reason[i] = "ZeroROI"; writer.write(vis); pbar.update(1); continue

            if args.pre_blur > 0:
                k = int(args.pre_blur); k = k + 1 if k % 2 == 0 else k
                roi = cv2.GaussianBlur(roi, (k, k), 0)

            roi_clahe = clahe.apply(roi)
            mask = apply_threshold(roi_clahe, mode=args.th_mode, win=args.th_win, C=args.th_C)
            mask = morph_cleanup(mask, open_px=args.open_px, close_px=args.close_px)
            largest, area_ratio, comps = largest_component(mask) if args.largest_only else (mask, float(cv2.countNonZero(mask))/mask.size, 1)

            if largest is None or area_ratio < args.min_area or area_ratio > args.max_area:
                clip_reason[i] = "MaskAreaInvalid"; writer.write(vis); pbar.update(1); continue

            cmean = float(np.mean(roi[largest > 0])) if np.any(largest > 0) else np.nan
            cold_area_ratio[i], cold_mean[i], valid[i], clip_reason[i] = area_ratio, cmean, 1, ""
            writer.write(vis); pbar.update(1)

    # 數據正規化與訊號生成
    vmask = (valid == 1) & ~np.isnan(cold_area_ratio) & ~np.isnan(cold_mean)
    if np.any(vmask):
        A5, A95 = np.percentile(cold_area_ratio[vmask], [5, 95])
        area_norm = np.clip((cold_area_ratio - A5) / (A95 - A5 + 1e-12), 0.0, 1.0)
        M5, M95 = np.percentile(cold_mean[vmask], [5, 95])
        mean_norm = np.clip((cold_mean - M5) / (M95 - M5 + 1e-12), 0.0, 1.0)
        signal_raw = area_norm * (1.0 - mean_norm)
    else:
        signal_raw = np.full(nframes, np.nan)

    # 時間序列處理 (去趨勢/帶通/SG)
    def moving_average_nanaware(x, win):
        if win <= 1: return x.copy()
        x = np.asarray(x, dtype=np.float64)
        mask = ~np.isnan(x)
        y, kernel = np.nan_to_num(x, nan=0.0), np.ones(win, dtype=np.float64)
        num, den = np.convolve(y, kernel, mode="same"), np.convolve(mask.astype(np.float64), kernel, mode="same")
        return np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))

    win_det = max(1, int(round(args.detrend_win_sec * fps)))
    signal_detrend = signal_raw - moving_average_nanaware(signal_raw, win_det)
    sig_filled = pd.Series(signal_detrend).interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float64)
    
    b, a = butter_bandpass(args.bp_low, args.bp_high, fs=fps, order=args.butterworth_order)
    try: sig_bp = filtfilt(b, a, sig_filled, method="gust")
    except: sig_bp = filtfilt(b, a, sig_filled)

    sg_win = max(3, int(round(args.savgol_win_sec * fps)))
    sg_win = sg_win + 1 if sg_win % 2 == 0 else sg_win
    signal_final = savgol_filter(sig_bp, window_length=sg_win, polyorder=min(args.savgol_poly, sg_win-1), mode="interp")
    
    signal_final[~vmask] = np.nan
    if np.any(vmask):
        signal_final = (signal_final - np.mean(signal_final[vmask])) / (np.std(signal_final[vmask]) + 1e-12)

    # --- 提取 pig_id 序列 ---
    pig_ids = [roi_map.get(i, {}).get("pig_id", "Unknown") for i in range(nframes)]

    # 輸出 CSV 與 預覽圖
    csv_out = os.path.join(args.out_dir, "signal.csv")
    pd.DataFrame({
        "frame_th": frame_idx, 
        "time_sec": time_sec, 
        "valid": valid, 
        "method": "m2",
        "signal_raw": signal_raw, 
        "signal_detrend": signal_detrend, 
        "signal_bp": sig_bp, 
        "signal_final": signal_final, 
        "clip_reason": clip_reason,
        "pig_id": pig_ids  # 新增這一欄
    }).to_csv(csv_out, index=False)

    png_out = os.path.join(args.out_dir, "preview_plot_m2.png")
    plt.figure(figsize=(12, 5))
    plt.plot(time_sec, signal_raw, label="raw"); plt.plot(time_sec, signal_final, label="final")
    plt.title("Signals - B2 (m2)"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(png_out, dpi=150); plt.close()

    writer.release(); cap.release()
    print(f"[OK] signal.csv → {csv_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="(B2) roi_signal_m2 — 冷區分割型")
    p.add_argument("--thermal", type=str, required=True)
    p.add_argument("--roi-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="out_m2")
    p.add_argument("--inner-shrink", type=int, default=3)
    p.add_argument("--gray-mode", type=str, default="auto", choices=["auto","bgr2gray","green"])
    p.add_argument("--pre-blur", type=int, default=3)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-tile", type=int, nargs=2, default=[8,8])
    p.add_argument("--th-mode", type=str, default="otsu", choices=["adaptive_gaussian","adaptive_mean","otsu"])
    p.add_argument("--th-win", type=int, default=21)
    p.add_argument("--th-C", type=int, default=3)
    p.add_argument("--open-px", type=int, default=3)
    p.add_argument("--close-px", type=int, default=5)
    p.add_argument("--largest-only", action="store_true", default=True)
    p.add_argument("--min-area", type=float, default=0.02)
    p.add_argument("--max-area", type=float, default=0.60)
    p.add_argument("--detrend-win-sec", type=float, default=15.0)
    p.add_argument("--bp-low", type=float, default=0.20)
    p.add_argument("--bp-high", type=float, default=2.50)
    p.add_argument("--savgol-win-sec", type=float, default=0.5)
    p.add_argument("--savgol-poly", type=int, default=3)
    p.add_argument("--butterworth-order", type=int, default=6)
    main(p.parse_args())