# -*- coding: utf-8 -*-
"""
(B0 + C) roi_signal_baseline_rr.py
單檔完成：ROI 平均 → 單一頻帶(0.2–2.5 Hz) → 滑窗 FFT 主峰 → RR(BPM)

輸入：
  --thermal   熱像影片
  --roi-csv   rgb2thermal_roi.py 產生的 roi_boxes.csv（以「熱像時間」為主）
輸出（--out-dir）：
  signal.csv             （與 m1/m2/m3 相同 schema, method="baseline"）
  preview_plot_baseline.png
  thermal_with_roi_baseline.mp4（方便肉眼檢查）
  rr_series.csv / rr_table.csv / rr_trend.png / window_fft_overlays.png
"""
import os, cv2, csv, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt, savgol_filter

# ---------- 小工具 ----------
def video_info(cap):
    if not cap or not cap.isOpened(): return None
    return dict(
        fps=cap.get(cv2.CAP_PROP_FPS) or 0.0,
        nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )

def clamp(v, lo, hi): return max(lo, min(hi, v))
def clamp_bbox(x1,y1,x2,y2,W,H):
    x1c=int(clamp(math.floor(x1),0,W-1)); y1c=int(clamp(math.floor(y1),0,H-1))
    x2c=int(clamp(math.ceil (x2),0,W-1)); y2c=int(clamp(math.ceil (y2),0,H-1))
    return x1c,y1c,x2c,y2c

def load_roi_csv(path):
    m={}
    with open(path,"r",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: k=int(r["frame_th"])
            except: continue
            m[k]=r
    return m

def moving_average_nanaware(x, win):
    if win<=1: return x.copy()
    x = np.asarray(x, np.float64)
    mask = np.isfinite(x)
    y = np.nan_to_num(x, nan=0.0)
    k = np.ones(win, np.float64)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(mask.astype(np.float64), k, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=(den>0))
    return out

def butter_bandpass(lowcut, highcut, fs, order):
    nyq=0.5*fs
    low=max(1e-6, lowcut/nyq); high=min(0.999, highcut/nyq)
    if high<=low: high=min(0.999, low+1e-3)
    from scipy.signal import butter
    b,a = butter(order, [low,high], btype='band')
    return b,a

def hann(N):
    n=np.arange(N, dtype=np.float64)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))

def resample_equal_time(tw, sw, t0, t1, fs_target):
    if fs_target<=0 or len(tw)<2: return np.array([]), np.array([])
    step = 1.0/float(fs_target)
    tg = np.arange(t0, t1+1e-9, step, dtype=np.float64)
    sg = np.interp(tg, tw, sw, left=np.nan, right=np.nan)
    m = np.isfinite(sg)
    return tg[m], sg[m]

def fft_peak_and_snr(x, fs, f_lo, f_hi, peak_half_bw=0.05, zero_pad=4):
    N=len(x)
    if N<8 or fs<=0: return np.nan, np.nan, np.nan, None, None
    padN = int(2**np.ceil(np.log2(N*zero_pad)))
    X = np.fft.rfft(x, n=padN)
    P = (np.abs(X)**2) / (N**2)
    f = np.fft.rfftfreq(padN, d=1.0/fs)
    mask = (f>=f_lo) & (f<=f_hi)
    if not np.any(mask): return np.nan, np.nan, np.nan, f, P
    fb, Pb = f[mask], P[mask]
    if fb.size<3: return np.nan, np.nan, np.nan, f, P
    k = int(np.argmax(Pb)); fpk = float(fb[k]); pamp=float(Pb[k])
    df = fb[1]-fb[0]; bw_bins = max(1, int(round(peak_half_bw/df)))
    k0 = max(0, k-bw_bins); k1 = min(Pb.size-1, k+bw_bins)
    main = Pb[k0:k1+1]
    noise = np.concatenate([Pb[:k0], Pb[k1+1:]]) if (k0>0 or k1<Pb.size-1) else np.array([])
    snr = np.nan if noise.size==0 else 10.0*np.log10( (np.mean(main)+1e-12)/(np.median(noise)+1e-12) )
    return fpk, pamp, float(snr), f, P

# ---------- B0：baseline 訊號抽取 ----------
def extract_signal_baseline(args):
    os.makedirs(args.out_dir, exist_ok=True)
    roi_map = load_roi_csv(args.roi_csv)
    cap = cv2.VideoCapture(args.thermal)
    meta = video_info(cap)
    if not meta or meta["fps"]<=0 or meta["nframes"]==0:
        raise RuntimeError("無法開啟熱像影片或 FPS/幀數異常。")
    fps,W,H,nframes = meta["fps"], meta["width"], meta["height"], meta["nframes"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vis_path = os.path.join(args.out_dir, "thermal_with_roi_baseline.mp4")
    writer = cv2.VideoWriter(vis_path, fourcc, fps, (W,H))

    idx = np.arange(nframes, dtype=np.int32)
    tsec = idx / fps
    valid = np.zeros(nframes, np.int32)
    clip_reason = np.array([""]*nframes, dtype=object)
    signal_raw = np.full(nframes, np.nan, np.float64)

    with tqdm(total=nframes, desc="(B0) ROI mean", unit="frame") as pbar:
        for i in range(nframes):
            ok, frame = cap.read()
            if not ok: break
            if len(frame.shape)==3 and frame.shape[2]==3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
            else:
                gray = frame.copy()
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            r = roi_map.get(i, None)
            if r is None:
                clip_reason[i]="NoCSV"; cv2.putText(vis,"NoCSV",(10,30),0,1.0,(0,0,255),2); writer.write(vis); pbar.update(1); continue
            if str(r.get("valid","0")).lower() not in ("1","true"):
                clip_reason[i]=r.get("clip_reason","Invalid"); cv2.putText(vis,clip_reason[i],(10,30),0,1.0,(0,0,255),2); writer.write(vis); pbar.update(1); continue

            try:
                x1=int(float(r["x1_th"])); y1=int(float(r["y1_th"]))
                x2=int(float(r["x2_th"])); y2=int(float(r["y2_th"]))
            except:
                clip_reason[i]="CSVCoordErr"; cv2.putText(vis,"CSVCoordErr",(10,30),0,1.0,(0,0,255),2); writer.write(vis); pbar.update(1); continue

            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)

            # 內縮，避免邊緣（可選）
            xi1,yi1=x1+args.inner_shrink, y1+args.inner_shrink
            xi2,yi2=x2-args.inner_shrink, y2-args.inner_shrink
            xi1,yi1,xi2,yi2 = clamp_bbox(xi1,yi1,xi2,yi2,W,H)
            if (xi2-xi1)<1 or (yi2-yi1)<1:
                clip_reason[i]="ShrinkTooSmall"; cv2.putText(vis,"ShrinkTooSmall",(10,30),0,1.0,(0,0,255),2); writer.write(vis); pbar.update(1); continue

            roi = gray[yi1:yi2, xi1:xi2]
            if roi.size==0:
                clip_reason[i]="ZeroROI"; cv2.putText(vis,"ZeroROI",(10,30),0,1.0,(0,0,255),2); writer.write(vis); pbar.update(1); continue

            # === 核心：ROI 直接平均 ===
            signal_raw[i] = float(np.mean(roi))
            valid[i] = 1; clip_reason[i] = ""

            # 可視化
            cv2.rectangle(vis,(xi1,yi1),(xi2,yi2),(255,255,0),2,cv2.LINE_AA)
            cv2.putText(vis, f"mean={signal_raw[i]:.2f}", (10,30), 0, 0.8, (255,255,255),2)
            writer.write(vis); pbar.update(1)

    cap.release(); writer.release()

    # 時間處理：detrend → bandpass(0.2–2.5) → SG → z-score
    win_det = max(1, int(round(args.detrend_win_sec * fps)))
    trend = moving_average_nanaware(signal_raw, win_det)
    detrend = signal_raw - trend
    series = pd.Series(detrend).interpolate(method="linear", limit_direction="both")
    sig = series.to_numpy(np.float64)

    b,a = butter_bandpass(args.bp_low, args.bp_high, fs=fps, order=args.butterworth_order)
    try: sig_bp = filtfilt(b,a,sig,method="gust")
    except: sig_bp = filtfilt(b,a,sig)

    sg_win = max(3, int(round(args.savgol_win_sec * fps)))
    if sg_win % 2 == 0: sg_win += 1
    sg_poly = max(1, int(args.savgol_poly)); sg_poly = min(sg_poly, sg_win-1)
    sig_final = savgol_filter(sig_bp, window_length=sg_win, polyorder=sg_poly, mode="interp")

    # 還原 invalid → NaN；z-score（僅 valid）
    sig_bp[np.isnan(signal_raw)] = np.nan
    sig_final[np.isnan(signal_raw)] = np.nan
    vmask = (valid==1) & np.isfinite(sig_final)
    if np.any(vmask):
        mu=float(np.mean(sig_final[vmask])); sd=float(np.std(sig_final[vmask])+1e-12)
        sig_final = (sig_final - mu)/sd

    # 寫 signal.csv（schema 對齊 m1/m2/m3）
    csv_out = os.path.join(args.out_dir, "signal.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_th","time_sec","valid","method",
                    "signal_raw","signal_detrend","signal_bp","signal_final","clip_reason"])
        for i in range(nframes):
            w.writerow([
                int(idx[i]), float(tsec[i]), int(valid[i]), "baseline",
                ("" if np.isnan(signal_raw[i]) else float(signal_raw[i])),
                ("" if np.isnan(detrend[i]) else float(detrend[i])),
                ("" if np.isnan(sig_bp[i]) else float(sig_bp[i])),
                ("" if np.isnan(sig_final[i]) else float(sig_final[i])),
                clip_reason[i]
            ])

    # 預覽圖
    png_out = os.path.join(args.out_dir, "preview_plot_baseline.png")
    plt.figure(figsize=(12,5))
    plt.plot(tsec, signal_raw,  label="signal_raw",  lw=1.0)
    plt.plot(tsec, sig_bp,     label="signal_bp",   lw=1.0)
    plt.plot(tsec, sig_final,  label="signal_final",lw=1.2)
    plt.xlabel("Time (s)"); plt.ylabel("Relative intensity (arb. unit)")
    plt.title("Baseline (ROI mean) — raw / band-pass / final")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(png_out, dpi=150); plt.close()

    # 回傳供下一段用
    return dict(fps=fps, time=tsec, valid=valid, signal_final=sig_final, meta=meta)

# ---------- C(單頻帶簡化版)：滑窗 FFT → RR ----------
def analyze_rr_singleband(args, series_dict):
    os.makedirs(args.out_dir, exist_ok=True)
    t = series_dict["time"]; s = series_dict["signal_final"]; v = series_dict["valid"]
    fps = series_dict["meta"]["fps"]

    # 視窗中心
    t_min = float(t[np.isfinite(t)].min())
    t_max = float(t[np.isfinite(t)].max())
    centers=[]; cur=t_min
    while cur + args.win_sec <= t_max + 1e-9:
        centers.append(cur + args.win_sec/2.0)
        cur += args.hop_sec
    centers = np.array(centers, np.float64)
    if centers.size==0:
        raise RuntimeError("有效重疊時間太短，無法產生任何滑窗。")

    # 逐窗找主峰
    detail_rows=[]; ref_fft=[]
    for c in tqdm(centers, desc="(C) sliding FFT", unit="win"):
        t0 = c - args.win_sec/2.0; t1 = c + args.win_sec/2.0
        mask = (t>=t0)&(t<=t1)
        if not np.any(mask):
            detail_rows.append([c,np.nan,np.nan,np.nan,0,"NoSamples",t0,t1])
            ref_fft.append(None); continue

        tw = t[mask]; sw = s[mask]; vw = v[mask]
        n_total=len(tw); n_valid=int(np.sum(vw==1)); cover=n_valid/float(n_total)
        if cover < args.min_cover or n_valid < 8:
            detail_rows.append([c,np.nan,np.nan,np.nan,0,"TooFewValid",t0,t1])
            ref_fft.append(None); continue

        tw = tw[vw==1]; sw = sw[vw==1]
        tg, sg = resample_equal_time(tw, sw, t0, t1, args.resample_hz)
        if len(sg) < 8:
            detail_rows.append([c,np.nan,np.nan,np.nan,0,"TooFewResampled",t0,t1])
            ref_fft.append(None); continue

        # 去均值 + Hann
        sg = sg - float(np.mean(sg))
        sgw = sg * hann(len(sg))
        fpk, pamp, snr, f_all, P_all = fft_peak_and_snr(sgw, args.resample_hz,
                                                        args.bp_low, args.bp_high,
                                                        peak_half_bw=args.peak_half_bw, zero_pad=4)
        ref_fft.append((f_all,P_all))
        if not np.isfinite(fpk):
            detail_rows.append([c,np.nan,np.nan,np.nan,0,"NoPeak",t0,t1]); continue
        rr = 60.0*float(fpk)
        detail_rows.append([c, float(fpk), float(rr),
                            (np.nan if not np.isfinite(snr) else float(snr)), 1, "", t0, t1])

    df_det = pd.DataFrame(detail_rows, columns=[
        "t_center_sec","f_peak_hz","rr_bpm","snr_db","valid","reason","t_start_sec","t_end_sec"
    ])

    # 中值平滑 + 跳變限幅
    rr = df_det["rr_bpm"].to_numpy(np.float64)
    # 中值窗
    k=max(1, int(args.median_win));  k = (k if k%2==1 else k+1)
    rr_smooth = rr.copy()
    if k>1:
        half=k//2
        for i in range(len(rr)):
            s=max(0,i-half); e=min(len(rr), i+half+1)
            seg=rr[s:e]; seg = seg[np.isfinite(seg)]
            if seg.size>0: rr_smooth[i]=np.median(seg)
    # 限幅
    for i in range(1, len(rr_smooth)):
        if np.isfinite(rr_smooth[i-1]) and np.isfinite(rr_smooth[i]):
            if abs(rr_smooth[i]-rr_smooth[i-1]) > args.jump_limit_bpm:
                rr_smooth[i] = np.nan
    df_det["rr_bpm"] = rr_smooth

    # 輸出 CSV
    outdir=args.out_dir
    df_det.to_csv(os.path.join(outdir,"rr_series.csv"), index=False)
    df_tbl = df_det[["t_start_sec","t_end_sec","f_peak_hz","rr_bpm","valid","reason"]].copy()
    df_tbl.insert(0,"window_idx", np.arange(1,len(df_tbl)+1))
    df_tbl = df_tbl.rename(columns={"f_peak_hz":"dominant_hz"})
    df_tbl.to_csv(os.path.join(outdir,"rr_table.csv"), index=False)

    # 視覺化
    # 1) RR 趨勢 + 平均線
    fig = plt.figure(figsize=(12,5))
    tc = df_det["t_center_sec"].to_numpy(np.float64)
    rb = df_det["rr_bpm"].to_numpy(np.float64)
    m = np.isfinite(rb)
    plt.plot(tc[m], rb[m], "-o", ms=3)
    if np.any(m):
        mean_rr = float(np.nanmean(rb[m]))
        plt.axhline(mean_rr, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
        ax=plt.gca()
        ax.text(0.98, 0.95, f"Mean RR = {mean_rr:.2f} BPM", transform=ax.transAxes,
                ha="right", va="top", color="gray", fontsize=10)
    plt.xlabel("Time (s)"); plt.ylabel("Respiration Rate (BPM)")
    plt.title("Respiration Rate Over Time (baseline)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"rr_trend.png"), dpi=150); plt.close(fig)

    # 2) 各窗 FFT 疊圖（參考單通道）
    fig = plt.figure(figsize=(12,5)); drawn=0
    for w_idx, c in enumerate(centers):
        pair = ref_fft[w_idx] if w_idx<len(ref_fft) else None
        if pair is None: continue
        f_all,P_all = pair
        if f_all is None or P_all is None: continue
        mask = (f_all>=args.bp_low)&(f_all<=args.bp_high)
        if np.any(mask):
            t0=float(df_det.iloc[w_idx]["t_start_sec"]); t1=float(df_det.iloc[w_idx]["t_end_sec"])
            label=f"W{w_idx+1} ({t0:.0f}–{t1:.0f}s)"
            plt.plot(f_all[mask], P_all[mask], lw=1.0, alpha=0.85, label=label); drawn+=1
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (arb.)")
    plt.title("FFT Analysis for All Windows (baseline)")
    if drawn>0: plt.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    plt.grid(alpha=0.3); plt.tight_layout()
    if drawn>0: plt.savefig(os.path.join(outdir,"window_fft_overlays.png"), dpi=150)
    plt.close(fig)

    # meta
    with open(os.path.join(outdir,"rr_meta.json"),"w",encoding="utf-8") as f:
        json.dump({
            "mode":"paper_single",
            "input_signal_csv": os.path.abspath(os.path.join(outdir,"signal.csv")),
            "params":{
                "win_sec":args.win_sec,"hop_sec":args.hop_sec,
                "band":[args.bp_low, args.bp_high],
                "resample_hz":args.resample_hz,"min_cover":args.min_cover,
                "peak_half_bw":args.peak_half_bw,"median_win":args.median_win,
                "jump_limit_bpm":args.jump_limit_bpm
            }
        }, f, ensure_ascii=False, indent=2)

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Baseline ROI-mean → single-band RR (all-in-one)")
    # I/O
    p.add_argument("--thermal", type=str, required=True)
    p.add_argument("--roi-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="out_baseline")

    # Baseline 抽訊號
    p.add_argument("--inner-shrink", type=int, default=3, help="ROI 內縮像素（避免邊界污染）")
    p.add_argument("--detrend-win-sec", type=float, default=15.0)
    p.add_argument("--bp-low", type=float, default=0.20)
    p.add_argument("--bp-high", type=float, default=2.50)
    p.add_argument("--butterworth-order", type=int, default=6)
    p.add_argument("--savgol-win-sec", type=float, default=0.5)
    p.add_argument("--savgol-poly", type=int, default=3)

    # RR（單頻帶）
    p.add_argument("--win-sec", type=float, default=30.0)
    p.add_argument("--hop-sec", type=float, default=10.0)
    p.add_argument("--resample-hz", type=float, default=10.0)
    p.add_argument("--min-cover", type=float, default=0.70)
    p.add_argument("--peak-half-bw", type=float, default=0.05)
    p.add_argument("--median-win", type=int, default=3)
    p.add_argument("--jump-limit-bpm", type=float, default=20.0)

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    series = extract_signal_baseline(args)
    analyze_rr_singleband(args, series)
    print(f"[OK] 完成：{os.path.abspath(args.out_dir)}")
    print(" - signal.csv / preview_plot_baseline.png / thermal_with_roi_baseline.mp4")
    print(" - rr_series.csv / rr_table.csv / rr_trend.png / window_fft_overlays.png")

if __name__=="__main__":
    main()

# python roi_signal_baseline_rr.py ^
#   --thermal "D:\videos\sample_thm.mp4" ^
#   --roi-csv D:\exp\out_roi\roi_boxes.csv ^
#   --out-dir D:\exp\out_baseline ^
#   --bp-low 0.20 --bp-high 2.50 ^
#   --win-sec 30 --hop-sec 10 --resample-hz 10