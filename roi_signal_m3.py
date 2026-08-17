# -*- coding: utf-8 -*-
"""
(B3) roi_signal_m3.py — 雙通道同心圓溫差法
高斯權重均值 + 雙通道同心圓溫差法 → 各自帶通 → 一致性投票
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

# ----------- 工具 -----------
def video_info(cap):
    if not cap or not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return dict(fps=fps, nframes=nframes, width=w, height=h)

def frame_time(idx, fps): return float(idx)/float(fps) if fps>0 else 0.0
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
            m[k]=r # 這裡會保留所有欄位，包含 pig_id
    return m

def butter_bandpass(lowcut, highcut, fs, order):
    nyq=0.5*fs; low=max(1e-6, lowcut/nyq); high=min(0.999, highcut/nyq)
    if high<=low: high=min(0.999, low+1e-3)
    b,a= butter(order,[low,high],btype="band")
    return b,a

# ----------- B3：鼻通道 & 暖帶 -----------
def find_local_dark_anchor(roi_8u, k=9):
    k=int(max(3,k));  k += (k%2==0)
    blur=cv2.GaussianBlur(roi_8u,(3,3),0)
    box=cv2.blur(blur,(k,k))
    minv,_,loc,_=cv2.minMaxLoc(box)
    return float(loc[0]), float(loc[1])

def gaussian_weight(h,w,cx,cy,sigma_ratio=0.25):
    s=max(1.0, sigma_ratio*min(w,h)); yy,xx=np.mgrid[0:h,0:w]
    g=np.exp(-(((xx-cx)**2)/(2*s*s)+((yy-cy)**2)/(2*s*s))).astype(np.float32)
    g/= (float(g.sum())+1e-12)
    return g

def warm_band_inner(gray_roi, inner_ratio, outer_ratio):
    h,w=gray_roi.shape[:2]
    r_base=min(w,h)/2.0
    r_in=max(0.0, inner_ratio*r_base)
    r_out=min(r_base, outer_ratio*r_base)
    cy=(h-1)/2.0; cx=(w-1)/2.0
    yy,xx=np.mgrid[0:h,0:w]; dx=xx-cx; dy=yy-cy
    rr=np.sqrt(dx*dx+dy*dy)
    mask=((rr>=r_in) & (rr<r_out)).astype(np.uint8)
    vals=gray_roi[mask>0].ravel()
    cover=float(vals.size)/float(gray_roi.size+1e-12)
    return vals, cover, (int(cx),int(cy),int(r_in),int(r_out))

# ----------- 主流程 -----------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    roi_map=load_roi_csv(args.roi_csv)
    cap=cv2.VideoCapture(args.thermal); meta=video_info(cap)
    if not meta or meta["fps"]<=0 or meta["nframes"]==0:
        raise RuntimeError("無法開啟熱像影片或 FPS/幀數異常。")
    fps,W,H,nframes=meta["fps"],meta["width"],meta["height"],meta["nframes"]

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(os.path.join(args.out_dir,"thermal_with_roi_m3.mp4"), fourcc, fps, (W,H))

    idx=np.arange(nframes,dtype=np.int32); time_sec=idx/fps
    valid=np.zeros(nframes,np.int32); clip_reason=np.array([""]*nframes, dtype=object)
    nose_raw=np.full(nframes,np.nan); cheek_raw=np.full(nframes,np.nan)
    signal_raw=np.full(nframes,np.nan); signal_detrend=np.full(nframes,np.nan)
    signal_bp=np.full(nframes,np.nan); signal_final=np.full(nframes,np.nan)

    with tqdm(total=nframes, desc="(B3) ROI sampling (nose vs cheek)", unit="frame") as pbar:
        for i in range(nframes):
            ok,frame=cap.read()
            if not ok: break

            if len(frame.shape)==3 and frame.shape[2]==3:
                gray=frame[:,:,1] if args.gray_mode in ["green", "auto"] else cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                vis=frame.copy()
            else:
                gray=frame.copy(); vis=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

            r=roi_map.get(i,None)
            if r is None or str(r.get("valid","0")).lower() not in ("1","true"):
                clip_reason[i]="InvalidOrNoCSV"; writer.write(vis); pbar.update(1); continue

            try:
                x1, y1 = int(float(r["x1_th"])), int(float(r["y1_th"]))
                x2, y2 = int(float(r["x2_th"])), int(float(r["y2_th"]))
            except:
                clip_reason[i]="CSVCoordErr"; writer.write(vis); pbar.update(1); continue

            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)
            xi1, yi1, xi2, yi2 = clamp_bbox(x1+args.inner_shrink, y1+args.inner_shrink, x2-args.inner_shrink, y2-args.inner_shrink, W, H)
            
            if (xi2-xi1)<3 or (yi2-yi1)<3:
                clip_reason[i]="ShrinkTooSmall"; writer.write(vis); pbar.update(1); continue

            roi=gray[yi1:yi2, xi1:xi2]
            if roi.size==0:
                clip_reason[i]="ZeroROI"; writer.write(vis); pbar.update(1); continue

            # ---- 圖像預處理功能（自 m3_o 加回） ----
            roi_proc = roi.copy()
            if args.pre_blur > 0:
                k = int(args.pre_blur)
                k += (k % 2 == 0)
                roi_proc = cv2.GaussianBlur(roi_proc, (k, k), 0)
            if args.proc_mode == "bilateral" and args.bilateral_d > 0:
                roi_proc = cv2.bilateralFilter(roi_proc, d=args.bilateral_d, sigmaColor=args.bilateral_sigma, sigmaSpace=args.bilateral_sigma)
            elif args.proc_mode == "clahe":
                clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(args.clahe_tile[0], args.clahe_tile[1]))
                roi_proc = clahe.apply(roi_proc)
            # ----------------------------------------

            # 鼻通道：高斯權重 (套用預處理後的 roi_proc)
            if args.anchor == "roi_center":
                cx, cy = (roi_proc.shape[1] - 1) / 2.0, (roi_proc.shape[0] - 1) / 2.0
            else:
                cx, cy = find_local_dark_anchor(roi_proc, k=max(7, int(0.15 * min(roi_proc.shape[:2]))))
            g=gaussian_weight(roi_proc.shape[0], roi_proc.shape[1], cx, cy, args.sigma_ratio)
            nose_val=float(np.sum(roi_proc.astype(np.float32)*g))
            nose_raw[i]=nose_val

            # 暖帶：內環帶 (依據 m3_o 邏輯，暖帶使用原始 roi)
            ring_vals, cover, _ = warm_band_inner(roi, args.inner_band_ratio[0], args.inner_band_ratio[1])
            if ring_vals.size == 0 or cover <= 0:
                clip_reason[i] = "InnerRingEmpty"; writer.write(vis); pbar.update(1); continue
        
            p_lo,p_hi=sorted(args.warm_pct)
            sel=ring_vals[(ring_vals>=np.percentile(ring_vals,p_lo))&(ring_vals<=np.percentile(ring_vals,p_hi))]
            cheek_val=float(np.mean(sel)) if sel.size>0 else float(np.mean(ring_vals))
            cheek_raw[i]=cheek_val

            delta=cheek_val-nose_val
            signal_raw[i]=delta; valid[i]=1; clip_reason[i]=""
            writer.write(vis); pbar.update(1)

    # ---- 時間處理 ----
    def moving_average_nanaware(x,win):
        if win<=1: return x.copy()
        x=np.asarray(x,np.float64); mask=~np.isnan(x); y=np.nan_to_num(x, nan=0.0)
        kernel=np.ones(win,np.float64)
        num=np.convolve(y,kernel,mode="same"); den=np.convolve(mask.astype(np.float64),kernel,mode="same")
        return np.divide(num,den,out=np.full_like(num,np.nan),where=(den>0))

    win_det=max(1,int(round(args.detrend_win_sec*fps)))
    signal_detrend=signal_raw - moving_average_nanaware(signal_raw, win_det)
    sig_filled=pd.Series(signal_detrend).interpolate(method="linear", limit_direction="both").to_numpy(np.float64)
    
    b,a=butter_bandpass(args.bp_low, args.bp_high, fs=fps, order=args.butterworth_order)
    try: sig_bp=filtfilt(b,a,sig_filled,method="gust")
    except: sig_bp=filtfilt(b,a,sig_filled)
    
    sg_win=max(3,int(round(args.savgol_win_sec*fps))); sg_win += (sg_win%2==0)
    signal_final=savgol_filter(sig_bp, window_length=sg_win, polyorder=min(args.savgol_poly, sg_win-1), mode="interp")
    
    vmask=(valid==1)&~np.isnan(signal_final)
    if np.any(vmask):
        signal_final=(signal_final-np.mean(signal_final[vmask]))/(np.std(signal_final[vmask])+1e-12)

    # --- 提取 pig_id 序列 ---
    pig_ids = [roi_map.get(i, {}).get("pig_id", "Unknown") for i in range(nframes)]

    # ---- 輸出 ----
    csv_out=os.path.join(args.out_dir,"signal.csv")
    pd.DataFrame({
        "frame_th": idx, 
        "time_sec": time_sec, 
        "valid": valid, 
        "method": "m3",
        "signal_raw": signal_raw, 
        "signal_detrend": signal_detrend, 
        "signal_bp": sig_bp, 
        "signal_final": signal_final, 
        "clip_reason": clip_reason,
        "nose_raw": nose_raw,
        "cheek_raw": cheek_raw,
        "pig_id": pig_ids
    }).to_csv(csv_out, index=False)

    png_out=os.path.join(args.out_dir,"preview_plot_m3.png")
    plt.figure(figsize=(12,5))
    plt.plot(time_sec, signal_raw, label="raw"); plt.plot(time_sec, signal_final, label="final")
    plt.xlabel("Time (s)"); plt.ylabel("Relative intensity"); plt.title("B3 (m3) Signals")
    plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(png_out, dpi=150); plt.close()

    writer.release(); cap.release()
    print(f"[OK] signal.csv → {csv_out}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description="(B3) roi_signal_m3 — 同心圓溫差法")
    p.add_argument("--thermal", type=str, required=True)
    p.add_argument("--roi-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="out_m3")
    p.add_argument("--inner-shrink", type=int, default=3)
    p.add_argument("--gray-mode", type=str, default="auto", choices=["auto","bgr2gray","green"])
    p.add_argument("--pre-blur", type=int, default=3)
    
    # 圖像預處理參數
    p.add_argument("--proc-mode", type=str, default="bilateral", choices=["bilateral","clahe"])
    p.add_argument("--bilateral-d", type=int, default=5)
    p.add_argument("--bilateral-sigma", type=float, default=20.0)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-tile", type=int, nargs=2, default=[8,8], metavar=("W","H"))

    p.add_argument("--anchor", type=str, default="roi_center", choices=["local_dark","roi_center"])
    p.add_argument("--sigma-ratio", type=float, default=0.25)
    p.add_argument("--warm-pct", type=int, nargs=2, default=[80,100])
    p.add_argument("--inner-band-ratio", type=float, nargs=2, default=[0.6,1.0])
    p.add_argument("--detrend-win-sec", type=float, default=15.0)
    p.add_argument("--bp-low", type=float, default=0.20)
    p.add_argument("--bp-high", type=float, default=2.50)
    p.add_argument("--savgol-win-sec", type=float, default=0.5)
    p.add_argument("--savgol-poly", type=int, default=3)
    p.add_argument("--butterworth-order", type=int, default=6)
    main(p.parse_args())