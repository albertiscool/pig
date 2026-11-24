# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------- 工具 -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def load_signal_csv(folder, default_name):
    p = os.path.join(folder, "signal.csv")
    if not os.path.exists(p): raise FileNotFoundError(f"找不到 {p}")
    df = pd.read_csv(p)
    for c in ["time_sec", "signal_final", "valid"]:
        if c not in df.columns: raise ValueError(f"{p} 缺少必要欄位: {c}")
    
    ch = str(df["method"].iloc[0]).strip() if "method" in df.columns and df["method"].notna().any() else default_name
    pid = str(df["pig_id"].mode()[0]) if "pig_id" in df.columns else "Unknown"
    return ch, df[["time_sec","signal_final","valid"]].copy(), pid

def hann(N):
    n = np.arange(N, dtype=np.float64)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))

def window_times(t_min, t_max, win_sec, hop_sec):
    t, centers = t_min, []
    while t + win_sec <= t_max + 1e-9:
        centers.append(t + win_sec/2.0)
        t += hop_sec
    return np.array(centers, dtype=np.float64)

def resample_equal_time(tw, sw, t0, t1, fs_target):
    if fs_target <= 0 or len(tw) < 2: return np.array([]), np.array([])
    tg = np.arange(t0, t1 + 1e-9, 1.0/fs_target, dtype=np.float64)
    sg = np.interp(tg, tw, sw, left=np.nan, right=np.nan)
    return tg[np.isfinite(sg)], sg[np.isfinite(sg)]

def fft_peak_and_snr(x, fs, band, peak_half_bw=0.05, zero_pad=8):
    N = len(x)
    if N < 8 or not np.isfinite(fs) or fs<=0: return np.nan, np.nan, np.nan, None, None
    padN = int(2**np.ceil(np.log2(N*zero_pad)))
    X = np.fft.rfft(x, n=padN); P = (np.abs(X)**2) / (N**2)
    f = np.fft.rfftfreq(padN, d=1.0/fs)
    mask = (f>=band[0]) & (f<=band[1])
    if not np.any(mask): return np.nan, np.nan, np.nan, f, P
    fb, Pb = f[mask], P[mask]
    k_peak = int(np.argmax(Pb))
    f_pk, p_amp = float(fb[k_peak]), float(Pb[k_peak])
    df = fb[1]-fb[0]
    bw_bins = max(1, int(round(peak_half_bw/df)))
    k0, k1 = max(0, k_peak - bw_bins), min(Pb.size-1, k_peak + bw_bins)
    noise = np.concatenate([Pb[:k0], Pb[k1+1:]])
    snr = 10.0*np.log10((np.mean(Pb[k0:k1+1])+1e-12)/(np.median(noise)+1e-12)) if noise.size>0 else np.nan
    return f_pk, p_amp, snr, f, P

def cluster_by_proximity(freqs, weights, df_thresh):
    if len(freqs)==0: return np.nan, []
    ord = np.argsort(freqs); f, w = freqs[ord], weights[ord]
    clusters, cur = [], [0]
    for i in range(1, len(f)):
        if abs(f[i]-f[cur[-1]]) <= df_thresh: cur.append(i)
        else: clusters.append(cur); cur=[i]
    clusters.append(cur)
    best = max(clusters, key=lambda c: float(np.sum(w[c])))
    return float(np.average(f[best], weights=w[best])), ord[best].tolist()

def median_filter_1d(x, k=3):
    if k<=1 or len(x)==0: return x.copy()
    n, y, half = len(x), x.copy(), k//2
    for i in range(n):
        seg = y[max(0,i-half):min(n,i+half+1)]; segf = seg[np.isfinite(seg)]
        if segf.size>0: y[i] = np.median(segf)
    return y

# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="rr_out")
    ap.add_argument("--win-sec", type=float, default=30.0)
    ap.add_argument("--hop-sec", type=float, default=10.0)
    ap.add_argument("--dual-band", action="store_true")
    ap.add_argument("--snr-min", type=float, default=4.0)
    ap.add_argument("--agree-df", type=float, default=0.17)
    ap.add_argument("--median-win", type=int, default=3)
    # --- 補回 auto_rr_pipeline 會傳入但之前漏掉的參數 ---
    ap.add_argument("--mode", type=str, default="multi_vote")
    ap.add_argument("--jump-limit-bpm", type=float, default=20.0)
    ap.add_argument("--per-channel-plots", action="store_true")
    
    args = ap.parse_args(); outdir = ensure_dir(args.out_dir)
    
    channels, final_pid = [], "Unknown"
    for i, folder in enumerate(args.inputs):
        name, df, pid = load_signal_csv(folder, f"m{i+1}")
        if pid != "Unknown": final_pid = pid
        channels.append((name, df))

    t_min = max([df["time_sec"].min() for _, df in channels])
    t_max = min([df["time_sec"].max() for _, df in channels])
    centers = window_times(t_min, t_max, args.win_sec, args.hop_sec)
    
    detail_rows, prev_rr_hz = [], {ch[0]: np.nan for ch in channels}
    for ch_name, df in channels:
        t, s, v = df["time_sec"].values, df["signal_final"].values, df["valid"].values
        for c in tqdm(centers, desc=f"Channel {ch_name}"):
            t0, t1 = c-args.win_sec/2, c+args.win_sec/2
            mask = (t>=t0) & (t<=t1)
            if np.sum(v[mask]==1)/len(t[mask]) < 0.7:
                detail_rows.append([c, ch_name, np.nan, np.nan, 0, np.nan, 0]); continue
            tw, sw = resample_equal_time(t[mask][v[mask]==1], s[mask][v[mask]==1], t0, t1, 10.0)
            if len(sw) < 8: continue
            sw = (sw - np.mean(sw)) * hann(len(sw))
            
            if args.dual_band:
                f_l, _, snr_l, _, _ = fft_peak_and_snr(sw, 10.0, (0.2, 1.4))
                f_h, _, snr_h, _, _ = fft_peak_and_snr(sw, 10.0, (0.8, 2.5))
                if abs(snr_h - snr_l) > 5.0: f_pk, snr = (f_h, snr_h) if snr_h > snr_l else (f_l, snr_l)
                elif np.isfinite(prev_rr_hz[ch_name]):
                    f_pk, snr = (f_h, snr_h) if abs(f_h-prev_rr_hz[ch_name]) < abs(f_l-prev_rr_hz[ch_name]) else (f_l, snr_l)
                else: f_pk, snr = f_l, snr_l
                if abs(f_h - 2*f_l) < 0.06 and (snr_h - snr_l) < 3.0: f_pk, snr = f_l, snr_l
            else:
                f_pk, _, snr, _, _ = fft_peak_and_snr(sw, 10.0, (0.2, 2.5))
            
            if np.isfinite(f_pk): prev_rr_hz[ch_name] = f_pk
            detail_rows.append([c, ch_name, f_pk, f_pk*60, 0, snr, 1 if np.isfinite(f_pk) else 0])

    df_det = pd.DataFrame(detail_rows, columns=["t_center_sec","channel","f_peak_hz","rr_bpm","peak_amp","snr_db","valid"])
    series_rows = []
    for c in centers:
        win = df_det[(df_det["t_center_sec"]==c) & (df_det["valid"]==1) & (df_det["snr_db"]>=args.snr_min)]
        if not win.empty:
            f_fin, mem = cluster_by_proximity(win["f_peak_hz"].values, win["snr_db"].values - args.snr_min + 1.0, args.agree_df)
            if len(mem) >= 2: series_rows.append([c, f_fin, f_fin*60, 1])
            else: series_rows.append([c, np.nan, np.nan, 0])
        else: series_rows.append([c, np.nan, np.nan, 0])

    df_series = pd.DataFrame(series_rows, columns=["t_center_sec","rr_hz","rr_bpm","valid"])
    df_series["rr_bpm"] = median_filter_1d(df_series["rr_bpm"].values, k=args.median_win)
    df_series.to_csv(os.path.join(outdir, "rr_series.csv"), index=False)

    ch_summary = {}
    for ch_name, _ in channels:
        bpm_vals = df_det[(df_det["channel"] == ch_name) & (df_det["valid"] == 1)]["rr_bpm"]
        ch_summary[ch_name] = bpm_vals.mean() if not bpm_vals.empty else 0.0

    m_rr_fusion = df_series["rr_bpm"].mean() if df_series["rr_bpm"].notna().any() else 0.0

    plt.figure(figsize=(10, 5))
    plt.plot(df_series["t_center_sec"], df_series["rr_bpm"], "-o", ms=4, label="Fusion RR")
    if m_rr_fusion > 0:
        plt.axhline(m_rr_fusion, color="gray", ls="--")
        plt.text(0.95, 0.95, f"ID: {final_pid}\nMean: {m_rr_fusion:.2f} BPM", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title(f"Breathing Trend - {final_pid}"); plt.xlabel("Time (s)"); plt.ylabel("BPM"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rr_trend.png"), dpi=150); plt.close()

    print("\n" + "="*40)
    print(f"📊 呼吸率分析報告 - 豬隻 ID: [{final_pid}]")
    print("-" * 40)
    for name, val in ch_summary.items():
        print(f"🔹 方法 {name:10}: {val:6.2f} BPM")
    print("-" * 40)
    print(f"⭐ 多通道融合 (Fusion): {m_rr_fusion:6.2f} BPM")
    print("="*40)

if __name__ == "__main__":
    main()