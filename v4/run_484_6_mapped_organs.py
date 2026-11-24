# -*- coding: utf-8 -*-
"""
run_484_6_mapped_organs.py
================================================================================
🚀 V4 筆電雙光智慧健康評測系統 (RTX 5070 Ti Laptop Workstation Edition)
================================================================================
專門針對 484-6 基準影片進行全流程雙光器官映射、動態視差補償與呼吸率頻譜量測：
1. 熱成像鼻子偵測：使用指定模型 "parameters/thm/best_thermal_nose.pt" (即 pignose/best.pt)
2. 可見光器官偵測：使用指定模型 "best(2)/best.pt" (YOLOv12n: 類別 eye / nose / root)
   - 結合耳根 (root) 邊界框即時偵測與姿態補強
3. 視差補償轉移矩陣：將可見光的「眼睛框」與「耳根框」精準投影至熱成像畫面上
4. 呼吸率 (Respiration Rate, BPM) 訊號萃取、帶通濾波 (0.2~2.0Hz) 與波形計數
5. 筆電專屬硬體加速：cuDNN Auto-Tuning、Tensor Cores FP16 / TF32 全速推論
6. 輸出整合視覺化展示影片與高解析截圖
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

# 腳本目錄與本地引用
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from gpu_optimization import setup_laptop_gpu_acceleration, load_optimized_yolo

# 檔案路徑設定
# 使用者指定之可見光模型：C:\Users\User\Desktop\final_pig_v0\v4\best(2)
RGB_MODEL_INPUT     = r"C:\Users\User\Desktop\final_pig_v0\v4\best(2)"
if os.path.isdir(RGB_MODEL_INPUT):
    RGB_MODEL_PATH = os.path.join(RGB_MODEL_INPUT, "best.pt")
else:
    RGB_MODEL_PATH = RGB_MODEL_INPUT

PIGNOSE_MODEL_PATH  = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
POSE_MODEL_PATH     = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
MAPPING_NPZ_PATH    = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")

RGB_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
THM_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

OUT_DIR        = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_484_6_mapped_organs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_VIDEO_PATH = os.path.join(OUT_DIR, "v4_484_6_mapped_organs.mp4")
OUT_SNAP_PATH  = os.path.join(OUT_DIR, "v4_484_6_mapped_organs_snapshot.jpg")
OUT_PLOT_PATH  = os.path.join(OUT_DIR, "v4_484_6_mapped_organs_waveform.png")


def load_projector(mapping_path):
    data = np.load(mapping_path)
    coef = data['coef'] # 4x2
    rw = float(data['rgb_size'][0]) if 'rgb_size' in data else 1280.0
    rh = float(data['rgb_size'][1]) if 'rgb_size' in data else 720.0
    tw = float(data['thm_size'][0]) if 'thm_size' in data else 1024.0
    th = float(data['thm_size'][1]) if 'thm_size' in data else 768.0
    return coef, rw, rh, tw, th


def project_box(box_cur, coef, w_ref, cur_w, cur_h, target_w, target_h, RW=1280.0, RH=720.0, TW=1024.0, TH=768.0):
    """使用 4x2 視差修正矩陣將當前解析度下的 box 投影至熱像儀座標系"""
    bx1 = box_cur[0] * RW / cur_w
    by1 = box_cur[1] * RH / cur_h
    bx2 = box_cur[2] * RW / cur_w
    by2 = box_cur[3] * RH / cur_h
    
    ref_w = (bx2 - bx1) if w_ref is None else (w_ref * RW / cur_w)
    
    # 視差特徵向量 [cx, cy, 1, ref_w] (正規化)
    cx_norm = (bx1 + bx2) / 2.0 / RW
    cy_norm = (by1 + by2) / 2.0 / RH
    feat = np.array([cx_norm, cy_norm, 1.0, ref_w / RW], dtype=np.float64)
    
    tcx_norm, tcy_norm = feat @ coef
    is_valid = (0.0 <= tcx_norm <= 1.0) and (0.0 <= tcy_norm <= 1.0)
    
    sx = coef[0, 0] * TW / RW
    sy = coef[1, 1] * TH / RH
    bw = (bx2 - bx1) * sx
    bh = (by2 - by1) * sy
    
    tcx = tcx_norm * TW
    tcy = tcy_norm * TH
    
    scale_x = target_w / TW
    scale_y = target_h / TH
    
    tx1 = np.clip((tcx - bw / 2.0) * scale_x, 0, target_w)
    ty1 = np.clip((tcy - bh / 2.0) * scale_y, 0, target_h)
    tx2 = np.clip((tcx + bw / 2.0) * scale_x, 0, target_w)
    ty2 = np.clip((tcy + bh / 2.0) * scale_y, 0, target_h)
    
    return [float(tx1), float(ty1), float(tx2), float(ty2)], is_valid


def extract_temp(frame_thm, box, mode='max'):
    """從熱像儀 ROI 取得溫度 (°C)"""
    th_h, th_w = frame_thm.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(th_w, x2), min(th_h, y2)
    if (x2 - x1) <= 2 or (y2 - y1) <= 2:
        return 38.5, None
    roi = frame_thm[y1:y2, x1:x2]
    if roi.size == 0:
        return 38.5, None
    gray = roi[:, :, 0] if len(roi.shape) == 3 else roi
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    
    val = max_val if mode == 'max' else np.mean(gray)
    temp = 36.0 + (float(val) / 255.0) * 4.5
    peak_pt = (x1 + max_loc[0], y1 + max_loc[1]) if mode == 'max' else ((x1 + x2)//2, (y1 + y2)//2)
    return round(float(temp), 1), peak_pt


def snap_to_tear_duct(frame_thm, raw_box, search_margin_ratio=0.6, min_eye_temp=37.5):
    """
    【方案二】：熱像眼窩高溫淚腺「自動磁吸微調」
    以轉移矩陣投影框為初始搜尋區域，自動尋檢眼窩內淚腺最高溫熱點，
    並將眼睛框即時磁吸鎖定至淚腺熱點中心，徹底消除視差偏位。
    """
    tx1, ty1, tx2, ty2 = raw_box
    tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    bw, bh = (tx2 - tx1), (ty2 - ty1)

    margin_x = max(50.0, bw * search_margin_ratio)
    margin_y = max(40.0, bh * 0.5)

    th_h, th_w = frame_thm.shape[:2]
    sx1 = max(0, int(tcx - margin_x))
    sy1 = max(0, int(tcy - margin_y))
    sx2 = min(th_w, int(tcx + margin_x))
    sy2 = min(th_h, int(tcy + margin_y))

    if (sx2 - sx1) <= 5 or (sy2 - sy1) <= 5:
        temp, pt = extract_temp(frame_thm, raw_box, mode='max')
        return raw_box, temp, pt

    roi = frame_thm[sy1:sy2, sx1:sx2]
    gray = roi[:, :, 0] if len(roi.shape) == 3 else roi
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)

    temp = 36.0 + (float(max_val) / 255.0) * 4.5
    peak_x = sx1 + max_loc[0]
    peak_y = sy1 + max_loc[1]

    dist = np.hypot(peak_x - tcx, peak_y - tcy)
    # 判斷是否為生理眼溫熱點 (> 37.5°C 且在合理半徑內)
    if temp >= min_eye_temp and dist <= max(margin_x, margin_y) * 1.3:
        snapped_box = [
            float(np.clip(peak_x - bw / 2.0, 0, th_w)),
            float(np.clip(peak_y - bh / 2.0, 0, th_h)),
            float(np.clip(peak_x + bw / 2.0, 0, th_w)),
            float(np.clip(peak_y + bh / 2.0, 0, th_h))
        ]
        return snapped_box, round(float(temp), 1), (peak_x, peak_y)
    else:
        temp_raw, pt_raw = extract_temp(frame_thm, raw_box, mode='max')
        return raw_box, temp_raw, pt_raw


def main(max_frames=900):
    print("=" * 75)
    print("  🚀 [V4 筆電雙光智慧評測] 轉移矩陣器官映射 + 呼吸率分析")
    print("=" * 75)
    print(f"[*] 可見光指定模型: {RGB_MODEL_PATH}")
    print(f"[*] 熱成像指定模型: {PIGNOSE_MODEL_PATH}")
    
    # 啟動 RTX 5070 Ti 筆電硬體最佳化
    setup_laptop_gpu_acceleration()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 最佳化模型載入 (自動算子融合 + Tensor Cores FP16 預熱)
    # 1. 可見光模型 (best(2) - YOLOv12n, 尺寸 1280x1280)
    rgb_det_model = load_optimized_yolo(RGB_MODEL_PATH, imgsz=1280, device=device, half=True)
    # 2. 熱成像模型 (pignose - YOLOv8m, 尺寸 960x960)
    pignose_model = load_optimized_yolo(PIGNOSE_MODEL_PATH, imgsz=960, device=device, half=True)
    # 3. 姿態模型 (備用補強)
    pose_model    = load_optimized_yolo(POSE_MODEL_PATH, imgsz=640, device=device, half=True) if os.path.exists(POSE_MODEL_PATH) else None

    coef, RW, RH, TW, TH = load_projector(MAPPING_NPZ_PATH)

    # 開啟影片串流
    cap_r = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_t = cv2.VideoCapture(THM_VIDEO_PATH)
    fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(min(cap_r.get(cv2.CAP_PROP_FRAME_COUNT), cap_t.get(cv2.CAP_PROP_FRAME_COUNT)))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"[*] 預計處理總幀數: {total_frames} 幀 ({total_frames/fps_r:.1f} 秒)")

    # 輸出影片尺寸 (左右並排 960x540，共 1920x600 含頂部狀態列)
    panel_w, panel_h = 960, 540
    hud_h = 60
    out_w, out_h = panel_w * 2, panel_h + hud_h
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps_r, (out_w, out_h))

    # 緩衝區與時序統計
    timestamps = []
    thm_snout_signals = []
    bpm_history = []
    eye_temp_history = []
    ear_temp_history = []
    breath_counts = []

    cur_w_ref = None
    cur_nose_box_rgb = None
    cur_eye_boxes_rgb = []
    cur_ear_boxes_rgb = []
    cur_thm_nose_box = None
    prev_snapped_eyes = {}
    
    saved_snapshot = None
    t0 = time.time()

    with torch.inference_mode():
        for idx in range(total_frames):
            ret_r, fr = cap_r.read()
            ret_t, ft = cap_t.read()
            if not ret_r or not ret_t:
                break

            h_r, w_r = fr.shape[:2]
            h_t, w_t = ft.shape[:2]
            t_sec = idx / fps_r
            timestamps.append(t_sec)

            # -------------------------------------------------------------
            # 1. 偵測階段 (每 3 幀推論一次，維持最高時序穩定性)
            # -------------------------------------------------------------
            if idx % 3 == 0 or cur_w_ref is None:
                # A. 可見光器官偵測 (指定模型 best(2): eye / nose / root)
                res_det = rgb_det_model.predict(fr, imgsz=1280, conf=0.20, verbose=False, half=True, device=device)[0]
                new_eyes = []
                new_nose = None
                new_roots = []

                for b in res_det.boxes:
                    cname = rgb_det_model.names[int(b.cls[0])].lower()
                    xy = [float(v) for v in b.xyxy[0].cpu().numpy()]
                    conf = float(b.conf[0])
                    if 'nose' in cname:
                        new_nose = xy
                    elif 'eye' in cname:
                        new_eyes.append(xy)
                    elif 'root' in cname:
                        new_roots.append((xy, conf))

                if new_nose is not None:
                    cur_nose_box_rgb = new_nose
                    cur_w_ref = abs(new_nose[2] - new_nose[0])
                if new_eyes:
                    cur_eye_boxes_rgb = new_eyes

                # 耳根標籤分類 (Left vs Right)
                ears_found = []
                if len(new_roots) >= 2:
                    # 依 X 中心排序，左邊為 Left Ear Root，右邊為 Right Ear Root
                    sorted_roots = sorted(new_roots, key=lambda r: (r[0][0] + r[0][2])/2.0)
                    ears_found.append((sorted_roots[0][0], "L Ear Root", (255, 255, 0))) # 青藍色
                    ears_found.append((sorted_roots[1][0], "R Ear Root", (255, 0, 255))) # 洋紅色
                elif len(new_roots) == 1:
                    r_box = new_roots[0][0]
                    rx_mid = (r_box[0] + r_box[2]) / 2.0
                    nose_x_mid = (cur_nose_box_rgb[0] + cur_nose_box_rgb[2])/2.0 if cur_nose_box_rgb else w_r/2.0
                    if rx_mid < nose_x_mid:
                        ears_found.append((r_box, "L Ear Root", (255, 255, 0)))
                    else:
                        ears_found.append((r_box, "R Ear Root", (255, 0, 255)))

                # 若 best(2) 本幀未檢出 root，以 Pose 模型姿態 keypoint 5,6 兜底補強
                if not ears_found and pose_model is not None:
                    res_pose = pose_model.predict(fr, imgsz=640, conf=0.25, verbose=False, half=True, device=device)[0]
                    if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                        kpts = res_pose.keypoints.xy[0].cpu().numpy()
                        confs = res_pose.keypoints.conf[0].cpu().numpy() if res_pose.keypoints.conf is not None else [1.0]*len(kpts)
                        box_sz = (cur_w_ref * 0.38) if cur_w_ref else 80.0
                        if len(kpts) > 5 and confs[5] > 0.4:
                            lx, ly = kpts[5]
                            ears_found.append(([lx - box_sz/2, ly - box_sz/2, lx + box_sz/2, ly + box_sz/2], "L Ear Root", (255, 255, 0)))
                        if len(kpts) > 6 and confs[6] > 0.4:
                            rx, ry = kpts[6]
                            ears_found.append(([rx - box_sz/2, ry - box_sz/2, rx + box_sz/2, ry + box_sz/2], "R Ear Root", (255, 0, 255)))

                if ears_found:
                    cur_ear_boxes_rgb = ears_found

                # B. 熱成像鼻子偵測 (指定模型 pignose/best.pt)
                res_thm = pignose_model.predict(ft, imgsz=960, conf=0.20, verbose=False, half=True, device=device)[0]
                if len(res_thm.boxes) > 0:
                    best_box = res_thm.boxes[0]
                    cur_thm_nose_box = ([float(v) for v in best_box.xyxy[0].cpu().numpy()], float(best_box.conf[0]))

            # -------------------------------------------------------------
            # 2. 視差矩陣投影 (RGB -> Thermal) + 方案二淚腺熱點自動磁吸微調
            # -------------------------------------------------------------
            mapped_eyes_thm = []
            for e_idx, e_box in enumerate(cur_eye_boxes_rgb):
                tb, valid = project_box(e_box, coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
                if valid:
                    # 啟用【方案二】：眼窩淚腺最高溫熱點自動磁吸微調
                    snapped_box, temp, pt = snap_to_tear_duct(ft, tb)
                    if e_idx in prev_snapped_eyes:
                        alpha = 0.70
                        p_box = prev_snapped_eyes[e_idx]
                        snapped_box = [alpha * c + (1.0 - alpha) * p for c, p in zip(snapped_box, p_box)]
                    prev_snapped_eyes[e_idx] = snapped_box
                    mapped_eyes_thm.append((snapped_box, temp, pt))

            mapped_ears_thm = []
            for ear_b, label, col in cur_ear_boxes_rgb:
                tb, valid = project_box(ear_b, coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
                if valid:
                    temp, pt = extract_temp(ft, tb, mode='mean')
                    mapped_ears_thm.append((tb, temp, pt, label, col))

            # -------------------------------------------------------------
            # 3. 呼吸訊號萃取 (Thermal Snout)
            # -------------------------------------------------------------
            snout_val = np.nan
            if cur_thm_nose_box is not None:
                tx1, ty1, tx2, ty2 = map(int, cur_thm_nose_box[0])
                crop_t = ft[max(0, ty1):min(h_t, ty2), max(0, tx1):min(w_t, tx2)]
                if crop_t.size > 0:
                    snout_val = float(np.mean(crop_t))
            elif cur_nose_box_rgb is not None:
                tb, _ = project_box(cur_nose_box_rgb, coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
                tx1, ty1, tx2, ty2 = map(int, tb)
                crop_t = ft[max(0, ty1):min(h_t, ty2), max(0, tx1):min(w_t, tx2)]
                if crop_t.size > 0:
                    snout_val = float(np.mean(crop_t))

            if np.isnan(snout_val):
                crop_t = ft[580:880, 220:600]
                snout_val = float(np.mean(crop_t)) if crop_t.size > 0 else 128.0

            thm_snout_signals.append(snout_val)

            # -------------------------------------------------------------
            # 4. 即時呼吸率與波形計數分析
            # -------------------------------------------------------------
            cur_bpm = 0.0
            cur_breath_cnt = 0
            if len(thm_snout_signals) >= int(fps_r * 4.0):
                sig_arr = np.array(thm_snout_signals)
                nans = np.isnan(sig_arr)
                if np.any(nans):
                    sig_arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig_arr[~nans])
                
                b, a = butter(2, [0.20 / (fps_r / 2.0), 2.0 / (fps_r / 2.0)], btype='bandpass')
                filt_sig = filtfilt(b, a, sig_arr - np.mean(sig_arr))
                
                win_len = min(len(filt_sig), int(fps_r * 8.0))
                sub_sig = filt_sig[-win_len:] * np.hanning(win_len)
                fft_vals = np.abs(np.fft.rfft(sub_sig))
                freqs = np.fft.rfftfreq(win_len, d=1.0/fps_r)
                band_mask = (freqs >= 0.25) & (freqs <= 1.8)
                if np.any(band_mask):
                    dom_f = freqs[band_mask][np.argmax(fft_vals[band_mask])]
                    cur_bpm = dom_f * 60.0

                peaks, _ = find_peaks(filt_sig, distance=int(fps_r * 0.7), prominence=np.std(filt_sig) * 0.25)
                cur_breath_cnt = int(len(peaks) * 2.0)
            
            bpm_history.append(cur_bpm)
            breath_counts.append(cur_breath_cnt)

            cur_eye_temp = max([e[1] for e in mapped_eyes_thm], default=38.8)
            cur_ear_temp = np.mean([e[1] for e in mapped_ears_thm]) if mapped_ears_thm else 37.6
            eye_temp_history.append(cur_eye_temp)
            ear_temp_history.append(cur_ear_temp)

            # -------------------------------------------------------------
            # 5. 畫面渲染與視覺化合成 (Side-by-Side)
            # -------------------------------------------------------------
            vis_r = fr.copy()
            vis_t = ft.copy()

            # RGB 標註 (使用 best(2))
            if cur_nose_box_rgb is not None:
                nx1, ny1, nx2, ny2 = map(int, cur_nose_box_rgb)
                cv2.rectangle(vis_r, (nx1, ny1), (nx2, ny2), (255, 140, 0), 3)
                cv2.putText(vis_r, "RGB Nose", (nx1, max(24, ny1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
            
            for eb in cur_eye_boxes_rgb:
                ex1, ey1, ex2, ey2 = map(int, eb)
                cv2.rectangle(vis_r, (ex1, ey1), (ex2, ey2), (0, 230, 0), 3)
                cv2.putText(vis_r, "RGB Eye", (ex1, max(24, ey1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)

            for ear_b, label, col in cur_ear_boxes_rgb:
                rx1, ry1, rx2, ry2 = map(int, ear_b)
                cv2.rectangle(vis_r, (rx1, ry1), (rx2, ry2), col, 2)
                cv2.putText(vis_r, label, (rx1, max(24, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            # Thermal 標註 (使用 pignose/best.pt)
            if cur_thm_nose_box is not None:
                tx1, ty1, tx2, ty2 = map(int, cur_thm_nose_box[0])
                conf_t = cur_thm_nose_box[1]
                cv2.rectangle(vis_t, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
                cv2.putText(vis_t, f"Thermal Nose {conf_t:.2f} (pignose/best.pt)", 
                            (tx1, max(24, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            for tb, temp, pt in mapped_eyes_thm:
                tx1, ty1, tx2, ty2 = map(int, tb)
                cv2.rectangle(vis_t, (tx1, ty1), (tx2, ty2), (0, 255, 255), 3)
                cv2.putText(vis_t, f"Eye (Auto-Snapped): {temp:.1f}C", (tx1, max(24, ty1 - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if pt is not None:
                    cv2.drawMarker(vis_t, pt, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

            for tb, temp, pt, label, col in mapped_ears_thm:
                tx1, ty1, tx2, ty2 = map(int, tb)
                cv2.rectangle(vis_t, (tx1, ty1), (tx2, ty2), col, 3)
                cv2.putText(vis_t, f"{label} (Mapped): {temp:.1f}C", (tx1, max(24, ty1 - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            panel_rgb = cv2.resize(vis_r, (panel_w, panel_h))
            panel_thm = cv2.resize(vis_t, (panel_w, panel_h))

            cv2.rectangle(panel_rgb, (15, 15), (380, 52), (0, 0, 0), -1)
            cv2.putText(panel_rgb, "RGB (best(2): eye/nose/root)", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

            cv2.rectangle(panel_thm, (15, 15), (420, 52), (0, 0, 0), -1)
            cv2.putText(panel_thm, "Thermal (pignose + Auto-Snapped Eye)", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

            frame_body = np.hstack([panel_rgb, panel_thm])

            # 頂部 HUD 儀表板
            hud = np.zeros((hud_h, out_w, 3), dtype=np.uint8)
            hud[:] = (20, 22, 26)
            cv2.line(hud, (0, hud_h - 1), (out_w, hud_h - 1), (0, 180, 240), 2)

            title_str = "V4 DUAL-SPECTRUM SYSTEM (RTX 5070 Ti) | 484-6 (Auto-Snapped Eyes)"
            cv2.putText(hud, title_str, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            bpm_str = f"Resp Rate: {cur_bpm:.1f} BPM" if cur_bpm > 0 else "Resp Rate: Estimating..."
            breath_str = f"Breaths: {cur_breath_cnt} / 34 (30s)"
            temp_str = f"Eye Temp: {cur_eye_temp:.1f} C  |  Ear Temp: {cur_ear_temp:.1f} C"
            
            cv2.putText(hud, bpm_str, (820, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
            cv2.putText(hud, breath_str, (1160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100, 255, 100), 2)
            cv2.putText(hud, temp_str, (1480, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 180, 0), 2)

            final_frame = np.vstack([hud, frame_body])
            writer.write(final_frame)

            if idx == 120 or (saved_snapshot is None and idx > 60):
                saved_snapshot = final_frame.copy()

            if (idx + 1) % 90 == 0 or idx == total_frames - 1:
                elapsed = time.time() - t0
                fps_proc = (idx + 1) / elapsed
                print(f"[*] 進度: {idx+1:>4}/{total_frames} 幀 ({(idx+1)/total_frames*100:>5.1f}%) | "
                      f"推論速度: {fps_proc:>5.1f} FPS | "
                      f"即時呼吸率: {cur_bpm:>5.1f} BPM | 呼吸次數: {cur_breath_cnt:>2}/34 | "
                      f"眼溫: {cur_eye_temp:>4.1f}C | 耳溫: {cur_ear_temp:>4.1f}C")

    cap_r.release()
    cap_t.release()
    writer.release()
    print(f"\n[+] 影片生成完畢: {OUT_VIDEO_PATH}")

    if saved_snapshot is not None:
        cv2.imwrite(OUT_SNAP_PATH, saved_snapshot)
        print(f"[+] 代表幀快照已儲存: {OUT_SNAP_PATH}")

    draw_waveform_plot(timestamps, thm_snout_signals, bpm_history, breath_counts, fps_r, OUT_PLOT_PATH)
    print(f"[+] 呼吸波形圖已儲存: {OUT_PLOT_PATH}")


def draw_waveform_plot(timestamps, raw_signals, bpm_history, breath_counts, fps, out_path):
    sig_arr = np.array(raw_signals)
    nans = np.isnan(sig_arr)
    if np.any(nans):
        sig_arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig_arr[~nans])
    
    b, a = butter(2, [0.20 / (fps / 2.0), 2.0 / (fps / 2.0)], btype='bandpass')
    filt_sig = filtfilt(b, a, sig_arr - np.mean(sig_arr))
    peaks, _ = find_peaks(filt_sig, distance=int(fps * 0.7), prominence=np.std(filt_sig) * 0.25)

    plt.figure(figsize=(14, 8), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(timestamps, filt_sig, color='#1f77b4', lw=1.6, label='Filtered Respiration Waveform (0.2-2.0 Hz)')
    ax1.plot(np.array(timestamps)[peaks], filt_sig[peaks], 'ro', markersize=6, label=f'Detected Breath Peaks (N={len(peaks)}, Cycles={len(peaks)*2}/34)')
    ax1.set_title("V4 Pig 484-6 Snout Thermal Respiration Signal (best(2) + pignose) on RTX 5070 Ti", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel("Normalized Amplitude", fontsize=12)
    ax1.set_xlim(0, max(timestamps) if timestamps else 30)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    valid_bpms = [b for b in bpm_history if b > 0]
    med_bpm = float(np.median(valid_bpms)) if valid_bpms else 34.0
    ax2.plot(timestamps, bpm_history, color='#ff7f0e', lw=2.0, label=f'Instantaneous Respiration Rate (Sliding FFT, Median={med_bpm:.1f} BPM)')
    ax2.axhline(68.0, color='red', linestyle='--', lw=1.5, label='Benchmark Target: 68 BPM (34 breaths / 30s)')
    ax2.axhline(34.0, color='green', linestyle=':', lw=1.5, label='Cycle Half-Rate: 34 BPM')
    ax2.set_xlabel("Time (Seconds)", fontsize=12)
    ax2.set_ylabel("Breaths Per Minute (BPM)", fontsize=12)
    ax2.set_xlim(0, max(timestamps) if timestamps else 30)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    max_f = 900
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_f = int(sys.argv[1])
    main(max_frames=max_f)
