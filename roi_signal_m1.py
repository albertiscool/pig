# -*- coding: utf-8 -*-
"""
(B1) roi_signal_m1.py  —  穩定亮度型
邊界縮減 → Bilateral → 參考環帶扣除 → trimmed mean → 去趨勢 → 帶通 → SavGol
"""

import os
import cv2
import csv
import json
import math
import argparse
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
# 工具函式
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

def trimmed_mean(arr, trim_ratio=0.1):
    a = arr.astype(np.float32).ravel()
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return np.nan
    a.sort()
    k = int(n * trim_ratio)
    k2 = int(n * (1.0 - trim_ratio))
    k = max(0, min(k, n-1))
    k2 = max(k, min(k2, n))
    sl = a[k:k2]
    if sl.size == 0:
        return np.nan
    return float(np.mean(sl))

def mean_ring(img, inner, thickness, W, H):
    x1, y1, x2, y2 = inner
    outer = (x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness)
    ox1, oy1, ox2, oy2 = clamp_bbox(*outer, W, H)

    if ox2 <= ox1 or oy2 <= oy1:
        return np.nan, 0

    outer_roi = img[oy1:oy2, ox1:ox2]
    outer_sum = np.sum(outer_roi, dtype=np.float64)
    outer_cnt = outer_roi.size

    ix1 = clamp(x1, ox1, ox2)
    iy1 = clamp(y1, oy1, oy2)
    ix2 = clamp(x2, ox1, ox2)
    iy2 = clamp(y2, oy1, oy2)
    ix1, iy1, ix2, iy2 = int(ix1), int(iy1), int(ix2), int(iy2)

    inner_cnt = 0
    inner_sum = 0.0
    if ix2 > ix1 and iy2 > iy1:
        inner_roi = img[iy1:iy2, ix1:ix2]
        inner_sum = float(np.sum(inner_roi, dtype=np.float64))
        inner_cnt = inner_roi.size

    ring_cnt = outer_cnt - inner_cnt
    if ring_cnt <= 0:
        return np.nan, 0
    ring_sum = outer_sum - inner_sum
    ring_mean = float(ring_sum / ring_cnt)
    return ring_mean, ring_cnt

def moving_average_nanaware(x, win):
    if win <= 1:
        return x.copy()
    x = np.asarray(x, dtype=np.float64)
    mask = ~np.isnan(x)
    y = np.nan_to_num(x, nan=0.0)
    kernel = np.ones(win, dtype=np.float64)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(mask.astype(np.float64), kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))
    return out

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        high = min(0.999, low + 1e-3)
    b, a = butter(order, [low, high], btype='band')
    return b, a


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

    fps = meta["fps"]
    W, H = meta["width"], meta["height"]
    nframes = meta["nframes"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_out_path = os.path.join(args.out_dir, "thermal_with_roi_m1.mp4")
    writer = cv2.VideoWriter(vid_out_path, fourcc, fps, (W, H))

    frame_idx = np.arange(nframes, dtype=np.int32)
    time_sec = frame_idx / fps
    valid = np.zeros(nframes, dtype=np.int32)
    clip_reason = np.array([""] * nframes, dtype=object)
    signal_raw = np.full(nframes, np.nan, dtype=np.float64)

    inner = args.inner_shrink
    ring_t = args.ring_thickness
    bilateral_d = args.bilateral_d
    bilateral_sigma = args.bilateral_sigma

    with tqdm(total=nframes, desc="(B1) ROI sampling", unit="frame") as pbar:
        for i in range(nframes):
            ok, frame = cap.read()
            if not ok: break

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
            else:
                gray = frame.copy()
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            r = roi_map.get(i, None)
            if r is None:
                clip_reason[i] = "NoCSV"
                cv2.putText(vis, "NoCSV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(vis); pbar.update(1); continue

            is_valid_A = str(r.get("valid", "0")) in ("1", "True", "true")
            if not is_valid_A:
                clip_reason[i] = r.get("clip_reason", "Invalid")
                cv2.putText(vis, f"{clip_reason[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(vis); pbar.update(1); continue

            try:
                x1 = int(float(r["x1_th"])); y1 = int(float(r["y1_th"]))
                x2 = int(float(r["x2_th"])); y2 = int(float(r["y2_th"]))
            except Exception:
                clip_reason[i] = "CSVCoordErr"
                cv2.putText(vis, "CSVCoordErr", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(vis); pbar.update(1); continue

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2, lineType=cv2.LINE_AA)
            xi1, yi1 = x1 + inner, y1 + inner
            xi2, yi2 = x2 - inner, y2 - inner
            xi1, yi1, xi2, yi2 = clamp_bbox(xi1, yi1, xi2, yi2, W, H)

            if (xi2 - xi1) < 1 or (yi2 - yi1) < 1:
                clip_reason[i] = "ShrinkTooSmall"
                writer.write(vis); pbar.update(1); continue

            roi = gray[yi1:yi2, xi1:xi2]
            roi_f = cv2.bilateralFilter(roi, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma) if bilateral_d > 0 else roi
            roi_tm = trimmed_mean(roi_f, trim_ratio=args.trim_ratio)

            ring_mean, ring_cnt = mean_ring(gray, (xi1, yi1, xi2, yi2), ring_t, W, H)
            if np.isnan(ring_mean):
                clip_reason[i] = "RingOOB"
                writer.write(vis); pbar.update(1); continue

            delta_I = float(roi_tm - ring_mean)
            signal_raw[i] = delta_I
            valid[i] = 1
            clip_reason[i] = ""

            cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), (255,255,0), 2, lineType=cv2.LINE_AA)
            writer.write(vis); pbar.update(1)

    # 時間處理
    win_det = max(1, int(round(args.detrend_win_sec * fps)))
    trend = moving_average_nanaware(signal_raw, win_det)
    signal_detrend = signal_raw - trend
    sig_detrend_filled = pd.Series(signal_detrend).interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float64)

    b, a = butter_bandpass(args.bp_low, args.bp_high, fs=fps, order=args.butterworth_order)
    try: sig_bp = filtfilt(b, a, sig_detrend_filled, method="gust")
    except: sig_bp = filtfilt(b, a, sig_detrend_filled)

    sg_win = max(3, int(round(args.savgol_win_sec * fps)))
    if sg_win % 2 == 0: sg_win += 1
    sg_poly = min(args.savgol_poly, sg_win - 1)
    sig_final = savgol_filter(sig_bp, window_length=sg_win, polyorder=sg_poly, mode="interp")

    sig_bp[np.isnan(signal_raw)] = np.nan
    sig_final[np.isnan(signal_raw)] = np.nan

    vmask = (valid == 1) & ~np.isnan(sig_final)
    if np.any(vmask):
        mu, sd = float(np.mean(sig_final[vmask])), float(np.std(sig_final[vmask]) + 1e-12)
        sig_final = (sig_final - mu) / sd

    # 存檔與繪圖
    csv_out = os.path.join(args.out_dir, "signal.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        # 修改標頭，加入 pig_id
        writer_csv.writerow(["frame_th", "time_sec", "valid", "method", "signal_raw", 
                             "signal_detrend", "signal_bp", "signal_final", "clip_reason", "pig_id"])
        for i in range(nframes):
            # 取得對應影格的 pig_id，若無則標註 Unknown
            pid = roi_map.get(i, {}).get("pig_id", "Unknown")
            writer_csv.writerow([
                int(frame_idx[i]), 
                float(time_sec[i]), 
                int(valid[i]), 
                "m1", 
                signal_raw[i], 
                signal_detrend[i], 
                sig_bp[i], 
                sig_final[i], 
                clip_reason[i],
                pid # 新增寫入 ID
            ])

    png_out = os.path.join(args.out_dir, "preview_plot_m1.png")
    plt.figure(figsize=(12, 5))
    plt.plot(time_sec, signal_raw, label="signal_raw", lw=1.0)
    plt.plot(time_sec, sig_bp, label="signal_bp", lw=1.0)
    plt.plot(time_sec, sig_final, label="signal_final", lw=1.2)
    plt.xlabel("Time (s)"); plt.ylabel("Relative intensity"); plt.title("Signals - B1 (m1)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(png_out, dpi=150); plt.close()

    writer.release(); cap.release()
    print(f"[OK] signal.csv → {csv_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(B1) roi_signal_m1 — 穩定亮度型")
    parser.add_argument("--thermal", type=str, required=True)
    parser.add_argument("--roi-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_m1")
    parser.add_argument("--inner-shrink", type=int, default=3)
    parser.add_argument("--ring-thickness", type=int, default=4)
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--bilateral-d", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=20.0)
    parser.add_argument("--detrend-win-sec", type=float, default=15.0)
    parser.add_argument("--bp-low", type=float, default=0.20)
    parser.add_argument("--bp-high", type=float, default=2.50)
    parser.add_argument("--savgol-win-sec", type=float, default=0.5)
    parser.add_argument("--savgol-poly", type=int, default=3)
    parser.add_argument("--butterworth-order", type=int, default=6)
    main(parser.parse_args())