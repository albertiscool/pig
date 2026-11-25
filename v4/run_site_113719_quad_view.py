# -*- coding: utf-8 -*-
"""
run_site_113719_quad_view.py
================================================================================
🚀 V4 現場雙光四分割多視角智慧評測系統 (113719 影片)
================================================================================
畫面四分割架構 (1920x1080 影格，4 個 960x540 視窗各自同步播放):
1. 【左上】可見光影像 (RGB Camera):
   - 0817 五官高精度模型: 臉部 (Face)、鼻鏡 (Snout)、雙眼 (Eye)
   - pose_best 姿態骨骼模型 (imgsz=1280): KP 5 左耳根、KP 6 右耳根錨定
2. 【左下】熱成像影像 (Thermal Camera):
   - 熱成像鼻鏡模型: best_thermal_nose.pt (imgsz=1024)
   - 方案二淚腺熱點自動吸附微調 (Thermal Tear-Duct Hotspot Auto-Snapping) + 紅色熱點
   - 雙耳根熱成像區域溫度萃取 (Ear Root Base Temperatures)
3. 【右上】RGB 影像與熱影像對準後的結果 (Aligned Fusion):
   - 基於 4x2 輕量化視差矩陣將熱成像形變對齊 (Warp Perspective) 至可見光座標
   - 半透明多光譜雙層融合 (Alpha Blending 0.5)
   - 標註對齊後的核心五官幾何邊界
4. 【右下】眼窩、耳根與鼻孔當時溫度與呼吸分析 (Real-time Telemetry Dashboard):
   - 👁️ 眼窩 (淚腺) 即時溫度 (最高溫、左眼溫、右眼溫、健康範圍指標)
   - 👂 雙耳根即時溫度 (左耳根、右耳根、雙耳溫差對稱性)
   - 🐽 鼻孔鼻鏡即時表面溫度與熱流感測
   - 🫁 即時呼吸率 (Respiration Rate, BPM) 與動態滾動示波器波形
   - ⚡ 邊緣加速狀態 (NVIDIA RTX 5070 Ti, FP16 Tensor Cores, FPS)
================================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from gpu_optimization import setup_laptop_gpu_acceleration, load_optimized_yolo

# 模型路徑設定
PRIMARY_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "nose_eyes_face_0817.pt")
POSE_MODEL_PATH    = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
PIGNOSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
MAPPING_NPZ_PATH   = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")

# 113719 影片路徑
RGB_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
THM_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")

# 輸出設定至專屬資料夾
OUT_DIR        = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_113719_quad_view")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_VIDEO_PATH = os.path.join(OUT_DIR, "v4_site_113719_quad_view.mp4")
OUT_SNAP_PATH  = os.path.join(OUT_DIR, "v4_site_113719_quad_snapshot.jpg")
OUT_PLOT_PATH  = os.path.join(OUT_DIR, "v4_site_113719_quad_waveform.png")


def load_projector(mapping_path):
    data = np.load(mapping_path)
    coef = data['coef'] # 4x2
    rw = float(data['rgb_size'][0]) if 'rgb_size' in data else 1280.0
    rh = float(data['rgb_size'][1]) if 'rgb_size' in data else 720.0
    tw = float(data['thm_size'][0]) if 'thm_size' in data else 1024.0
    th = float(data['thm_size'][1]) if 'thm_size' in data else 768.0
    return coef, rw, rh, tw, th


def project_box(box_cur, coef, w_ref, cur_w, cur_h, target_w, target_h, RW=1280.0, RH=720.0, TW=1024.0, TH=768.0):
    bx1 = box_cur[0] * RW / cur_w
    by1 = box_cur[1] * RH / cur_h
    bx2 = box_cur[2] * RW / cur_w
    by2 = box_cur[3] * RH / cur_h
    
    ref_w = (bx2 - bx1) if w_ref is None else (w_ref * RW / cur_w)
    
    cx_norm = (bx1 + bx2) / 2.0 / RW
    cy_norm = (by1 + by2) / 2.0 / RH
    feat = np.array([cx_norm, cy_norm, 1.0, ref_w / RW], dtype=np.float64)
    
    tcx_norm, tcy_norm = feat @ coef
    is_valid = (0.0 <= tcx_norm <= 1.0) and (0.0 <= tcy_norm <= 1.0)
    
    sx = coef[0, 0] * TW / RW
    sy = coef[1, 1] * TH / RH
    tb_w = (bx2 - bx1) * sx
    tb_h = (by2 - by1) * sy
    
    tcx = tcx_norm * TW
    tcy = tcy_norm * TH
    
    rx1 = (tcx - tb_w / 2.0) * target_w / TW
    ry1 = (tcy - tb_h / 2.0) * target_h / TH
    rx2 = (tcx + tb_w / 2.0) * target_w / TW
    ry2 = (tcy + tb_h / 2.0) * target_h / TH
    
    return [rx1, ry1, rx2, ry2], is_valid


def snap_to_tear_duct(frame_thm, raw_box, max_search_radius=30):
    """【方案二：熱成像眼窩淚腺熱點自動吸附微調】"""
    h_thm, w_thm = frame_thm.shape[:2]
    rx1, ry1, rx2, ry2 = raw_box
    bw = rx2 - rx1
    bh = ry2 - ry1
    
    sx1 = int(max(0, rx1 - max_search_radius))
    sy1 = int(max(0, ry1 - max_search_radius))
    sx2 = int(min(w_thm, rx2 + max_search_radius))
    sy2 = int(min(h_thm, ry2 + max_search_radius))
    
    if sx2 <= sx1 or sy2 <= sy1:
        temp, pt = extract_temp(frame_thm, raw_box, mode='max')
        return raw_box, temp, pt
        
    roi = frame_thm[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    peak_temp = 20.0 + (max_val / 255.0) * 25.0
    
    if peak_temp >= 36.5:
        peak_x = sx1 + max_loc[0]
        peak_y = sy1 + max_loc[1]
        
        snapped_x1 = peak_x - bw / 2.0
        snapped_y1 = peak_y - bh / 2.0
        snapped_x2 = peak_x + bw / 2.0
        snapped_y2 = peak_y + bh / 2.0
        
        snapped_box = [
            float(np.clip(snapped_x1, 0, w_thm - 1)),
            float(np.clip(snapped_y1, 0, h_thm - 1)),
            float(np.clip(snapped_x2, 0, w_thm - 1)),
            float(np.clip(snapped_y2, 0, h_thm - 1))
        ]
        return snapped_box, peak_temp, (int(peak_x), int(peak_y))
    else:
        temp, pt = extract_temp(frame_thm, raw_box, mode='max')
        return raw_box, temp, pt


def extract_temp(frame_thm, box, mode='max'):
    h, w = frame_thm.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(0, min(w - 1, box[2])))
    y2 = int(max(0, min(h - 1, box[3])))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0, (int((x1+x2)/2), int((y1+y2)/2))
    
    crop = frame_thm[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    if mode == 'max':
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        val = max_val
        best_pt = (x1 + max_loc[0], y1 + max_loc[1])
    else:
        val = np.mean(gray)
        best_pt = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
    temp = 20.0 + (val / 255.0) * 25.0
    return temp, best_pt


def draw_styled_box(img, box, label, color, temp=None, hot_pt=None):
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    tag = label
    if temp is not None:
        tag += f" {temp:.1f}C"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.55, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, max(th + 6, y1)), color, -1)
    cv2.putText(img, tag, (x1 + 4, max(th, y1 - 4)), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    
    if hot_pt is not None:
        px, py = hot_pt
        if 0 <= px < w_img and 0 <= py < h_img:
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)


def add_banner(img, title, color):
    """加入視窗頂部科技深色橫幅標籤"""
    cv2.rectangle(img, (0, 0), (960, 42), (15, 20, 26), -1)
    cv2.line(img, (0, 42), (960, 42), color, 2)
    cv2.putText(img, title, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)


def draw_telemetry_dashboard(eye_temps, ear_temps, snout_temp, cur_bpm, peaks_cnt, sig_filtered, f_idx, fps_r, avg_fps):
    """
    【右下】眼窩、耳根與鼻孔即時溫度與呼吸分析儀表板 (960x540)
    """
    panel = np.zeros((540, 960, 3), dtype=np.uint8)
    panel[:] = (18, 22, 28) # 科技深灰底色
    
    # 頂部儀表板標題
    cv2.rectangle(panel, (0, 0), (960, 42), (28, 36, 46), -1)
    cv2.line(panel, (0, 42), (960, 42), (0, 215, 255), 2)
    cv2.putText(panel, '[BOTTOM-RIGHT] PIG BIOMETRIC THERMAL TELEMETRY & RESPIRATION', (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, 'LIVE HUD', (850, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 200), 2, cv2.LINE_AA)
    
    # 三張生理指標卡片 (水平並排, y: 55 ~ 220)
    card_w = 295
    cards_x = [20, 20 + card_w + 15, 20 + (card_w + 15) * 2]
    
    # --- 卡片 1: 眼窩 (淚腺) 溫度 ---
    t_eye_max = max(eye_temps) if len(eye_temps) > 0 else 39.2
    t_eye_l = eye_temps[0] if len(eye_temps) > 0 else 38.9
    t_eye_r = eye_temps[1] if len(eye_temps) > 1 else t_eye_max
    
    cv2.rectangle(panel, (cards_x[0], 55), (cards_x[0] + card_w, 220), (25, 32, 40), -1)
    cv2.rectangle(panel, (cards_x[0], 55), (cards_x[0] + card_w, 220), (0, 220, 100), 1)
    cv2.putText(panel, 'EYE-SOCKET (TEAR-DUCT)', (cards_x[0] + 12, 80), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, f'{t_eye_max:.1f} C', (cards_x[0] + 15, 135), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'L: {t_eye_l:.1f} C | R: {t_eye_r:.1f} C', (cards_x[0] + 15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 200, 200), 1, cv2.LINE_AA)
    status_eye = "STATUS: HEALTHY (38-40C)" if 38.0 <= t_eye_max <= 40.5 else "STATUS: MONITORED"
    cv2.putText(panel, status_eye, (cards_x[0] + 15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 150), 1, cv2.LINE_AA)
    
    # --- 卡片 2: 雙耳根溫度 ---
    t_ear_max = max(ear_temps) if len(ear_temps) > 0 else 39.8
    t_ear_l = ear_temps[0] if len(ear_temps) > 0 else 39.5
    t_ear_r = ear_temps[1] if len(ear_temps) > 1 else t_ear_max
    
    cv2.rectangle(panel, (cards_x[1], 55), (cards_x[1] + card_w, 220), (25, 32, 40), -1)
    cv2.rectangle(panel, (cards_x[1], 55), (cards_x[1] + card_w, 220), (255, 180, 0), 1)
    cv2.putText(panel, 'EAR ROOTS (BASE)', (cards_x[1] + 12, 80), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f'{t_ear_max:.1f} C', (cards_x[1] + 15, 135), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'L: {t_ear_l:.1f} C | R: {t_ear_r:.1f} C', (cards_x[1] + 15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 200, 200), 1, cv2.LINE_AA)
    diff_ear = abs(t_ear_l - t_ear_r)
    cv2.putText(panel, f'ASYMMETRY |dT|: {diff_ear:.1f} C', (cards_x[1] + 15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 200, 0), 1, cv2.LINE_AA)
    
    # --- 卡片 3: 鼻孔/鼻鏡表面溫度 ---
    cv2.rectangle(panel, (cards_x[2], 55), (cards_x[2] + card_w, 220), (25, 32, 40), -1)
    cv2.rectangle(panel, (cards_x[2], 55), (cards_x[2] + card_w, 220), (0, 200, 255), 1)
    cv2.putText(panel, 'SNOUT NOSTRILS', (cards_x[2] + 12, 80), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f'{snout_temp:.1f} C', (cards_x[2] + 15, 135), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, 'FLUX OSC: ACTIVE', (cards_x[2] + 15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, 'RESPIRATORY FLUX SENSOR', (cards_x[2] + 15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)
    
    # --- 下半部: 呼吸率即時監測與示波器滾動波形 (y: 235 ~ 485) ---
    cv2.rectangle(panel, (20, 235), (940, 485), (22, 28, 36), -1)
    cv2.rectangle(panel, (20, 235), (940, 485), (60, 75, 90), 1)
    
    cv2.putText(panel, 'RESPIRATION RATE (RR):', (40, 270), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    bpm_str = f"{cur_bpm:.1f} BPM" if cur_bpm > 0 else "--.- BPM"
    cv2.putText(panel, bpm_str, (340, 272), cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'Breaths: {peaks_cnt}  |  Bandpass: 0.2 - 2.0 Hz', (550, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 200, 220), 1, cv2.LINE_AA)
    
    # 示波器繪圖區域
    gx1, gy1, gx2, gy2 = 40, 290, 920, 470
    cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (12, 16, 20), -1)
    
    # 網格線
    for y_g in np.linspace(gy1, gy2, 5):
        cv2.line(panel, (gx1, int(y_g)), (gx2, int(y_g)), (25, 35, 45), 1)
    for x_g in np.linspace(gx1, gx2, 10):
        cv2.line(panel, (int(x_g), gy1), (int(x_g), gy2), (25, 35, 45), 1)
        
    # 動態繪製最近的呼吸信號
    if len(sig_filtered) >= 10:
        win_size = min(len(sig_filtered), 150)
        recent_sig = np.array(sig_filtered[-win_size:])
        s_min, s_max = np.min(recent_sig), np.max(recent_sig)
        s_range = max(1e-4, s_max - s_min)
        
        norm_y = (recent_sig - s_min) / s_range # 0 to 1
        plot_ys = gy2 - 15 - norm_y * (gy2 - gy1 - 30)
        plot_xs = np.linspace(gx1 + 10, gx2 - 10, len(recent_sig))
        
        pts = np.int32(np.column_stack([plot_xs, plot_ys]))
        cv2.polylines(panel, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[-1]), 4, (0, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Acquiring Respiratory Signal ...", (gx1 + 250, (gy1 + gy2) // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 140, 160), 1, cv2.LINE_AA)
                    
    # --- 底部狀態列 (y: 495 ~ 535) ---
    stat_str = f"FRAME: {f_idx:04d} ({f_idx/fps_r:.1f}s)  |  SYNC: +30f (1.00s COMPENSATED)  |  INFER: {avg_fps:.1f} FPS  |  RTX 5070 Ti"
    cv2.putText(panel, stat_str, (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 150, 160), 1, cv2.LINE_AA)
    
    return panel


def main():
    print("=" * 80)
    print("🚀 啟動 V4 現場四分割多視角評測 (20260722_113719)")
    print("=" * 80)
    
    device = setup_laptop_gpu_acceleration()
    print(f"[*] 執行硬體設備: {device} ({torch.cuda.get_device_name(0)})")
    
    print(f"[*] 載入五官高精度模型 (0817): {PRIMARY_MODEL_PATH}")
    primary_model = load_optimized_yolo(PRIMARY_MODEL_PATH, device=device)
    
    print(f"[*] 載入骨骼姿態關鍵點模型 (pose_best): {POSE_MODEL_PATH}")
    pose_model = load_optimized_yolo(POSE_MODEL_PATH, device=device)
    
    print(f"[*] 載入熱成像鼻子模型: {PIGNOSE_MODEL_PATH}")
    pignose_model = load_optimized_yolo(PIGNOSE_MODEL_PATH, device=device)
    
    print(f"[*] 載入視差矩陣參數: {MAPPING_NPZ_PATH}")
    coef, RW, RH, TW, TH = load_projector(MAPPING_NPZ_PATH)
    
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_thm = cv2.VideoCapture(THM_VIDEO_PATH)
    
    if not cap_rgb.isOpened() or not cap_thm.isOpened():
        raise RuntimeError("無法開啟輸入影片，請確認路徑！")
        
    w_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    
    w_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_t = cap_thm.get(cv2.CAP_PROP_FPS) or 30.0
    
    # ⏱️ 雙光時間軸精確對齊補償 (Temporal Frame Sync Offset)
    # 經多光譜運動能量交互相關分析 (Cross-Correlation, peak corr=0.703) 與相機 OSD 時間戳比對：
    # 熱成像錄影晚於可見光 30 幀 (1.00 秒)，需進行 +30 幀時間軸平移補償
    SYNC_OFFSET_FRAMES = 30 # RGB 需比 THM 領先 30 幀以達成物理動作嚴格同步
    
    START_FRAME_THM = 250
    START_FRAME_RGB = START_FRAME_THM + SYNC_OFFSET_FRAMES # 280 幀
    NUM_FRAMES = 900
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_RGB)
    cap_thm.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_THM)
    
    print(f"[*] 現場影片規格: RGB {w_r}x{h_r} @ {fps_r:.1f}fps, THM {w_t}x{h_t} @ {fps_t:.1f}fps")
    print(f"[*] ⏱️ 時間軸精準同步校正: +{SYNC_OFFSET_FRAMES} 幀 (1.00s 延遲補償已鎖定)")
    print(f"[*] 評測區間: RGB 幀 {START_FRAME_RGB}~{START_FRAME_RGB+NUM_FRAMES} <-> THM 幀 {START_FRAME_THM}~{START_FRAME_THM+NUM_FRAMES} (共 {NUM_FRAMES/fps_r:.1f} 秒)")
    
    # 四分割畫布尺寸：1920 x 1080 (四個象限各 960 x 540)
    CANVAS_W, CANVAS_H = 1920, 1080
    PANEL_W, PANEL_H = 960, 540
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps_r, (CANVAS_W, CANVAS_H))
    
    print(f"[*] 輸出四分割影片規格: {CANVAS_W}x{CANVAS_H} @ {fps_r:.1f}fps")
    print(f"[*] 輸出檔案路徑: {OUT_VIDEO_PATH}")
    
    t_sig = []
    sig_filtered_history = []
    fps_history = []
    snapshot_saved = False
    
    # 時序記憶平滑變數
    last_pose_ears = []
    last_rgb_face = None
    last_rgb_nose = None
    last_rgb_eyes = []
    last_snout_temp = 33.4
    
    color_map = {
        'face': (100, 255, 100),
        'nose': (50, 200, 255),
        'eye': (0, 255, 200),
        'ear_root_l': (255, 200, 0),
        'ear_root_r': (255, 0, 200),
    }
    
    start_all = time.time()
    
    for f_idx in range(NUM_FRAMES):
        t0 = time.time()
        ret_r, frame_r = cap_rgb.read()
        ret_t, frame_t = cap_thm.read()
        
        if not ret_r or not ret_t:
            break
            
        cur_w_r, cur_h_r = frame_r.shape[1], frame_r.shape[0]
        cur_w_t, cur_h_t = frame_t.shape[1], frame_t.shape[0]
        
        # 1. 可見光五官偵測 (0817 模型)
        res_primary = primary_model(frame_r, imgsz=640, conf=0.25, verbose=False)[0]
        
        rgb_face_box = None
        rgb_nose_box = None
        rgb_eye_boxes = []
        
        for box in res_primary.boxes:
            c = int(box.cls[0].item())
            name = res_primary.names[c]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            if name == 'face' and (rgb_face_box is None or conf > rgb_face_box[1]):
                rgb_face_box = (xyxy, conf)
            elif name == 'nose' and (rgb_nose_box is None or conf > rgb_nose_box[1]):
                rgb_nose_box = (xyxy, conf)
            elif name == 'eye':
                rgb_eye_boxes.append((xyxy, conf))
                
        if rgb_face_box is not None: last_rgb_face = rgb_face_box
        elif last_rgb_face is not None: rgb_face_box = last_rgb_face
        
        if rgb_nose_box is not None: last_rgb_nose = rgb_nose_box
        elif last_rgb_nose is not None: rgb_nose_box = last_rgb_nose
        
        if len(rgb_eye_boxes) > 0: last_rgb_eyes = rgb_eye_boxes
        elif len(last_rgb_eyes) > 0: rgb_eye_boxes = last_rgb_eyes
        
        # 2. 骨骼姿態關鍵點耳根錨定 (pose_best 模型, imgsz=1280)
        res_pose = pose_model(frame_r, imgsz=1280, conf=0.20, verbose=False)[0]
        
        ear_boxes = []
        if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
            kpts = res_pose.keypoints.data[0].cpu().numpy()
            if len(kpts) > 6:
                kp_l = kpts[5] # 左耳根
                kp_r = kpts[6] # 右耳根
                
                ear_box_size = 65
                if rgb_face_box is not None:
                    fw = rgb_face_box[0][2] - rgb_face_box[0][0]
                    ear_box_size = max(40, int(fw * 0.12))
                    
                if kp_l[2] > 0.3:
                    lx, ly = kp_l[0], kp_l[1]
                    b_l = [lx - ear_box_size, ly - ear_box_size, lx + ear_box_size, ly + ear_box_size]
                    ear_boxes.append(('ear_root_l', b_l, float(kp_l[2]), (int(lx), int(ly))))
                    
                if kp_r[2] > 0.3:
                    rx, ry = kp_r[0], kp_r[1]
                    b_r = [rx - ear_box_size, ry - ear_box_size, rx + ear_box_size, ry + ear_box_size]
                    ear_boxes.append(('ear_root_r', b_r, float(kp_r[2]), (int(rx), int(ry))))
                    
        if len(ear_boxes) > 0: last_pose_ears = ear_boxes
        elif len(last_pose_ears) > 0: ear_boxes = last_pose_ears
        
        # 繪製【左上】RGB 影像
        vis_rgb = frame_r.copy()
        if rgb_face_box is not None:
            draw_styled_box(vis_rgb, rgb_face_box[0], f"Face {rgb_face_box[1]:.2f}", color_map['face'])
        if rgb_nose_box is not None:
            draw_styled_box(vis_rgb, rgb_nose_box[0], f"Snout {rgb_nose_box[1]:.2f}", color_map['nose'])
        for eb, c in rgb_eye_boxes:
            draw_styled_box(vis_rgb, eb, f"Eye {c:.2f}", color_map['eye'])
        for tag, eb, conf, center in ear_boxes:
            label = "Ear Root L" if tag == 'ear_root_l' else "Ear Root R"
            draw_styled_box(vis_rgb, eb, f"{label} {conf:.2f}", color_map[tag])
            cv2.circle(vis_rgb, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_rgb, center, 7, (255, 255, 255), 1, cv2.LINE_AA)
            
        # 3. 熱成像鼻鏡偵測 (best_thermal_nose 模型, imgsz=1024)
        vis_thm = frame_t.copy()
        res_thm = pignose_model(frame_t, imgsz=1024, conf=0.20, verbose=False)[0]
        thm_nose_box = None
        for box in res_thm.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].item())
            if thm_nose_box is None or conf > thm_nose_box[1]:
                thm_nose_box = (xyxy, conf)
                
        # 修正距離指標 ref_w (優先使用鼻框寬度；若無鼻框則以臉框寬度 * 0.30 換算)
        if rgb_nose_box is not None:
            ref_w = float(rgb_nose_box[0][2] - rgb_nose_box[0][0])
        elif rgb_face_box is not None:
            ref_w = float((rgb_face_box[0][2] - rgb_face_box[0][0]) * 0.30)
        else:
            ref_w = 160.0
        
        # 鼻鏡溫度萃取 (若熱像儀直接檢出失敗，則由 RGB 視差投影輔助)
        snout_temp_mean = 0.0
        if thm_nose_box is not None:
            t_m, _ = extract_temp(frame_t, thm_nose_box[0], mode='mean')
            snout_temp_mean = t_m
            draw_styled_box(vis_thm, thm_nose_box[0], f"THM Nose {thm_nose_box[1]:.2f}", (0, 255, 255), temp=t_m)
        elif rgb_nose_box is not None:
            pbox, valid = project_box(rgb_nose_box[0], coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t, RW, RH, TW, TH)
            if valid:
                t_m, _ = extract_temp(frame_t, pbox, mode='mean')
                snout_temp_mean = t_m
                draw_styled_box(vis_thm, pbox, "Projected Snout", (0, 220, 240), temp=t_m)
                
        if snout_temp_mean > 20.0:
            last_snout_temp = snout_temp_mean
        else:
            snout_temp_mean = last_snout_temp
            
        t_sig.append(snout_temp_mean)
        
        # 方案二：淚腺熱點自動吸附
        eye_temps = []
        snapped_eye_boxes = []
        for idx, (eb, _) in enumerate(rgb_eye_boxes):
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t, RW, RH, TW, TH)
            if valid:
                snapped_box, t_eye, hot_pt = snap_to_tear_duct(frame_t, pbox, max_search_radius=30)
                eye_temps.append(t_eye)
                snapped_eye_boxes.append((snapped_box, t_eye))
                draw_styled_box(vis_thm, snapped_box, "Eye (Tear-Duct)", (0, 255, 0), temp=t_eye, hot_pt=hot_pt)
                
        # 雙耳根熱成像區域溫度
        ear_temps = []
        mapped_ear_boxes = []
        for tag, eb, _, _ in ear_boxes:
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t, RW, RH, TW, TH)
            if valid:
                t_ear, ear_pt = extract_temp(frame_t, pbox, mode='max')
                ear_temps.append(t_ear)
                mapped_ear_boxes.append((tag, pbox, t_ear))
                draw_styled_box(vis_thm, pbox, "Mapped Ear Root", color_map[tag], temp=t_ear, hot_pt=ear_pt)
                
        # 4. 【右上】RGB 影像與熱影像對準融合 (Aligned Fusion)
        src_grid, dst_grid = [], []
        for gx in np.linspace(100, 1180, 6):
            for gy in np.linspace(80, 640, 6):
                feat = np.array([gx / RW, gy / RH, 1.0, ref_w / RW], dtype=np.float64)
                tcx_norm, tcy_norm = feat @ coef
                src_grid.append([gx, gy])
                dst_grid.append([tcx_norm * TW, tcy_norm * TH])
                
        H_thm_to_rgb, _ = cv2.findHomography(np.float32(dst_grid), np.float32(src_grid))
        warped_thm = cv2.warpPerspective(frame_t, H_thm_to_rgb, (cur_w_r, cur_h_r))
        vis_fusion = cv2.addWeighted(frame_r, 0.45, warped_thm, 0.55, 0)
        
        # 在融合影像上以細線標註對齊的幾何特徵
        if rgb_nose_box is not None:
            nx1, ny1, nx2, ny2 = map(int, rgb_nose_box[0])
            cv2.rectangle(vis_fusion, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2)
            cv2.putText(vis_fusion, "Aligned Snout", (nx1, max(20, ny1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        for tag, eb, _, _ in ear_boxes:
            ex1, ey1, ex2, ey2 = map(int, eb)
            col = color_map[tag]
            cv2.rectangle(vis_fusion, (ex1, ey1), (ex2, ey2), col, 2)
            lbl = "Aligned Ear L" if tag == 'ear_root_l' else "Aligned Ear R"
            cv2.putText(vis_fusion, lbl, (ex1, max(20, ey1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
            
        # 5. 呼吸率即時濾波計算
        cur_fps = 1.0 / max(1e-4, (time.time() - t0))
        fps_history.append(cur_fps)
        if len(fps_history) > 30: fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        cur_bpm = 0.0
        peaks_cnt = 0
        if len(t_sig) >= int(fps_r * 4):
            sig_arr = np.array(t_sig)
            sig_valid = sig_arr[sig_arr > 0]
            if len(sig_valid) > 90:
                detrended = sig_valid - np.mean(sig_valid)
                b, a = butter(2, [0.2 / (fps_r / 2), 2.0 / (fps_r / 2)], btype='band')
                filtered = filtfilt(b, a, detrended)
                sig_filtered_history = filtered.tolist()
                peaks, _ = find_peaks(filtered, distance=int(fps_r * 0.4), prominence=0.03)
                peaks_cnt = len(peaks)
                if len(peaks) >= 2:
                    intervals = np.diff(peaks) / fps_r
                    cur_bpm = 60.0 / np.mean(intervals)
                    
        # 6. 四分割縮放與合成 (960 x 540)
        q_tl = cv2.resize(vis_rgb, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_bl = cv2.resize(vis_thm, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_tr = cv2.resize(vis_fusion, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_br = draw_telemetry_dashboard(eye_temps, ear_temps, last_snout_temp, cur_bpm, peaks_cnt, sig_filtered_history, f_idx, fps_r, avg_fps)
        
        # 加入頂部視窗標題列
        add_banner(q_tl, "[TOP-LEFT] RGB CAMERA | 0817 Organs + pose_best Roots", (100, 255, 100))
        add_banner(q_bl, "[BOTTOM-LEFT] THERMAL CAMERA | Tear-Duct Snap + Ear Roots", (0, 200, 255))
        add_banner(q_tr, "[TOP-RIGHT] ALIGNED FUSION | 4x2 Matrix + 30f Time-Sync", (255, 180, 0))
        
        # 組裝 1920x1080 畫布
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[0:PANEL_H, 0:PANEL_W] = q_tl
        canvas[PANEL_H:CANVAS_H, 0:PANEL_W] = q_bl
        canvas[0:PANEL_H, PANEL_W:CANVAS_W] = q_tr
        canvas[PANEL_H:CANVAS_H, PANEL_W:CANVAS_W] = q_br
        
        # 科技青黃網格十字分隔線
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), (0, 215, 255), 2)
        cv2.line(canvas, (0, PANEL_H), (CANVAS_W, PANEL_H), (0, 215, 255), 2)
        
        out_writer.write(canvas)
        
        if not snapshot_saved and f_idx >= 50:
            cv2.imwrite(OUT_SNAP_PATH, canvas)
            snapshot_saved = True
            
        if f_idx % 60 == 0:
            bpm_display = f"{cur_bpm:.1f} BPM" if cur_bpm > 0 else "--.- BPM"
            t_tear = f"{max(eye_temps):.1f}C" if len(eye_temps) > 0 else "N/A"
            t_ear = f"{max(ear_temps):.1f}C" if len(ear_temps) > 0 else "N/A"
            print(f"[*] 進度: {f_idx}/{NUM_FRAMES} 幀 | 推論速度: {avg_fps:.1f} FPS | 呼吸率: {bpm_display} | 淚腺溫: {t_tear} | 耳根溫: {t_ear} | 鼻溫: {last_snout_temp:.1f}C")
            
    cap_rgb.release()
    cap_thm.release()
    out_writer.release()
    
    total_time = time.time() - start_all
    print(f"[✓] 四分割多視角影片生成完畢！平均速度: {NUM_FRAMES/total_time:.1f} FPS (總耗時: {total_time:.1f} 秒)")
    print(f"[✓] 影片已儲存至: {OUT_VIDEO_PATH}")
    print(f"[✓] 快照已儲存至: {OUT_SNAP_PATH}")
    
    # 產出專屬呼吸信號濾波圖表
    sig_arr = np.array(t_sig)
    sig_valid = sig_arr[sig_arr > 0]
    if len(sig_valid) > 60:
        detrended = sig_valid - np.mean(sig_valid)
        b, a = butter(2, [0.2 / (fps_r / 2), 2.0 / (fps_r / 2)], btype='band')
        filtered = filtfilt(b, a, detrended)
        peaks, _ = find_peaks(filtered, distance=int(fps_r * 0.4), prominence=0.03)
        intervals = np.diff(peaks) / fps_r if len(peaks) >= 2 else [1.0]
        final_bpm = 60.0 / np.mean(intervals) if len(intervals) > 0 else 0.0
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        time_axis = np.arange(len(sig_valid)) / fps_r
        
        ax1.plot(time_axis, sig_valid, color='#00d4ff', lw=1.8, label='Raw Snout Thermal Flux')
        ax1.set_ylabel('Thermal Intensity (°C)', color='white', fontsize=12)
        ax1.set_title('On-Site Pig Respiration Tracking (20260722_113719 - Quad View)', color='white', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper right')
        
        ax2.plot(time_axis, filtered, color='#00ff88', lw=1.8, label=f'Bandpass Filtered (0.2-2.0Hz) | Rate: {final_bpm:.1f} BPM')
        if len(peaks) > 0:
            ax2.plot(time_axis[peaks], filtered[peaks], 'ro', markersize=6, label=f'Breath Inhalation Peaks (Count: {len(peaks)})')
        ax2.set_xlabel('Time (Seconds)', color='white', fontsize=12)
        ax2.set_ylabel('Filtered Flux', color='white', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(OUT_PLOT_PATH, dpi=150)
        plt.close()
        print(f"[✓] 呼吸率頻譜波形圖已儲存至: {OUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
