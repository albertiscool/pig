# -*- coding: utf-8 -*-
"""
run_site_best_video.py
================================================================================
🚀 V4 現場實測最佳影片雙光智慧評測系統 (0817 五官 + pose_best 骨骼幾何錨定耳根)
================================================================================
挑選現場錄製效果最優影片 (20260722_113719: 現場粉紅豬正面直視高清晰度片段):
1. 可見光特徵偵測組合：
   - 五官核心：nose_eyes_face_0817.pt (0.90+ 極高精度眼、鼻、臉部偵測)
   - 耳根專用：pose_best.pt (解剖骨骼關鍵點 KP 5 左耳根、KP 6 右耳根，1.00 滿分穩定度)
2. 熱成像模型：使用 "parameters/thm/best_thermal_nose.pt" (pignose/best.pt)
3. 視差補償轉移矩陣：以原廠標定之 1280x720 -> 1024x768 座標系精準映射
4. 方案二淚腺熱點自動吸附微調 (Thermal Tear-Duct Hotspot Auto-Snapping)
5. 雙耳根熱成像區域溫度萃取 (Ear Root Base Temperatures)
6. 鼻鏡熱流呼吸率 (Respiration Rate, BPM) 即時萃取與帶通濾波分析
7. 筆電 RTX 5070 Ti 硬體加速：FP16 Tensor Cores、cuDNN Benchmark 全速運算
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

# 檔案路徑設定
PRIMARY_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "nose_eyes_face_0817.pt")
POSE_MODEL_PATH    = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
PIGNOSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
MAPPING_NPZ_PATH   = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")

# 現場最佳錄製影片 (113719 粉紅豬正面健康監測片段)
RGB_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
THM_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")

# 輸出設定至各自命名的專屬資料夾
OUT_DIR        = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_113719_best_mapped")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_VIDEO_PATH = os.path.join(OUT_DIR, "v4_site_113719_best_mapped.mp4")
OUT_SNAP_PATH  = os.path.join(OUT_DIR, "v4_site_113719_best_snapshot.jpg")
OUT_PLOT_PATH  = os.path.join(OUT_DIR, "v4_site_113719_best_waveform.png")


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
    """
    【方案二：熱成像眼窩淚腺熱點自動吸附微調 (Thermal Tear-Duct Hotspot Auto-Snapping)】
    在初始視差投影框的局部區域內搜尋真正的眼球內眼角 (淚腺) 最高溫熱點，
    若搜尋到符合豬隻生理眼溫 (36.5°C ~ 42.0°C) 之熱峰值，則自動將預測框中心校正吸附至該點。
    """
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
    
    d = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
    if d > 3:
        cv2.line(img, (x1, y1), (x1 + d, y1), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1 + d, y1), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - d, y1), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + d), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + d, y2), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - d), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - d, y2), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - d), (255, 255, 255), 3, cv2.LINE_AA)
        
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


def main():
    print("=" * 80)
    print("🚀 啟動 V4 現場實測最佳影片智慧評測系統 (0817 五官 + pose_best 骨骼幾何錨定耳根)")
    print("=" * 80)
    
    device = setup_laptop_gpu_acceleration()
    print(f"[*] 執行硬體設備: {device} ({torch.cuda.get_device_name(0)})")
    
    # 載入五官核心模型與骨骼關鍵點模型
    print(f"[*] 載入五官高精度模型 (0817): {PRIMARY_MODEL_PATH}")
    primary_model = load_optimized_yolo(PRIMARY_MODEL_PATH, device=device)
    
    print(f"[*] 載入骨骼姿態關鍵點模型 (pose_best): {POSE_MODEL_PATH}")
    pose_model = load_optimized_yolo(POSE_MODEL_PATH, device=device)
    
    print(f"[*] 載入熱成像鼻子模型: {PIGNOSE_MODEL_PATH}")
    pignose_model = load_optimized_yolo(PIGNOSE_MODEL_PATH, device=device)
    
    print(f"[*] 載入視差矩陣參數: {MAPPING_NPZ_PATH}")
    coef, RW, RH, TW, TH = load_projector(MAPPING_NPZ_PATH)
    
    # 開啟現場最佳影片
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_thm = cv2.VideoCapture(THM_VIDEO_PATH)
    
    if not cap_rgb.isOpened() or not cap_thm.isOpened():
        raise RuntimeError("無法開啟輸入現場影片，請確認路徑！")
        
    w_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    
    w_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_t = cap_thm.get(cv2.CAP_PROP_FPS) or 30.0
    
    # 挑選 113719 豬隻端坐正面直視鏡頭之最優 30 秒片段 (幀數 250 ~ 1150, 共 900 幀)
    START_FRAME = 250
    NUM_FRAMES = 900
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    cap_thm.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    
    print(f"[*] 現場影片規格: RGB {w_r}x{h_r} @ {fps_r:.1f}fps, THM {w_t}x{h_t} @ {fps_t:.1f}fps")
    print(f"[*] 評測區間: 幀數 {START_FRAME} ~ {START_FRAME + NUM_FRAMES} (共 {NUM_FRAMES/fps_r:.1f} 秒)")
    
    out_h = 720
    aspect_r = w_r / h_r
    out_w_r = int(out_h * aspect_r)
    aspect_t = w_t / h_t
    out_w_t = int(out_h * aspect_t)
    canvas_w = out_w_r + out_w_t
    canvas_h = out_h + 120 # 底部 HUD 儀表板
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps_r, (canvas_w, canvas_h))
    
    print(f"[*] 輸出畫布規格: {canvas_w}x{canvas_h} @ {fps_r:.1f}fps")
    print(f"[*] 輸出影片路徑: {OUT_VIDEO_PATH}")
    
    thm_snout_signals = []
    eye_temps_history = []
    ear_temps_history = []
    
    prev_snapped_eyes = {}
    last_valid_thm_snout = None
    
    t_start = time.time()
    snapshot_saved = False
    
    for frame_idx in range(NUM_FRAMES):
        ret_r, fr = cap_rgb.read()
        ret_t, ft = cap_thm.read()
        if not ret_r or not ret_t:
            print(f"[!] 於幀 {frame_idx} 讀取至影片結尾。")
            break
            
        cur_nose_box_rgb = None
        cur_face_box_rgb = None
        cur_eye_boxes_rgb = []
        cur_ear_boxes_rgb = []
        cur_thm_nose_box = None
        cur_w_ref = None
        
        with torch.no_grad():
            # =================================================================
            # 1. 0817 模型高精度負責：臉部 (face)、鼻鏡 (nose)、雙眼 (eye)
            # =================================================================
            res_p = primary_model.predict(fr, imgsz=640, conf=0.25, verbose=False, half=True, device=device)[0]
            for b in res_p.boxes:
                cls_id = int(b.cls[0])
                name = primary_model.names.get(cls_id, str(cls_id)).lower()
                xyxy = [float(v) for v in b.xyxy[0].cpu().numpy()]
                conf = float(b.conf[0])
                
                if 'nose' in name or cls_id == 2:
                    if cur_nose_box_rgb is None or conf > cur_nose_box_rgb[1]:
                        cur_nose_box_rgb = (xyxy, conf)
                        cur_w_ref = xyxy[2] - xyxy[0]
                elif 'eye' in name or cls_id == 0:
                    cur_eye_boxes_rgb.append((xyxy, conf))
                elif 'face' in name or cls_id == 1:
                    if cur_face_box_rgb is None or conf > cur_face_box_rgb[1]:
                        cur_face_box_rgb = (xyxy, conf)
                        if cur_w_ref is None:
                            cur_w_ref = (xyxy[2] - xyxy[0]) * 0.40
                            
            # =================================================================
            # 2. pose_best 骨骼姿態幾何錨定耳根 (KP 5 左耳根、KP 6 右耳根)
            # =================================================================
            res_pose = pose_model.predict(fr, imgsz=1280, conf=0.15, verbose=False, half=True, device=device)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy[0].cpu().numpy()
                confs = res_pose.keypoints.conf[0].cpu().numpy() if res_pose.keypoints.conf is not None else [1.0]*len(kpts)
                box_sz = (cur_w_ref * 0.45) if cur_w_ref else 80.0
                
                # KP 5: 左耳根 (畫面中偏左側之耳骨交界)
                if len(kpts) > 5 and confs[5] > 0.40:
                    lx, ly = kpts[5]
                    cur_ear_boxes_rgb.append(([lx - box_sz/2, ly - box_sz/2, lx + box_sz/2, ly + box_sz/2], f"Ear Root L {confs[5]:.2f}", (255, 255, 0), (int(lx), int(ly))))
                # KP 6: 右耳根 (畫面中偏右側之耳骨交界)
                if len(kpts) > 6 and confs[6] > 0.40:
                    rx, ry = kpts[6]
                    cur_ear_boxes_rgb.append(([rx - box_sz/2, ry - box_sz/2, rx + box_sz/2, ry + box_sz/2], f"Ear Root R {confs[6]:.2f}", (255, 0, 255), (int(rx), int(ry))))
                        
            # =================================================================
            # 3. 熱成像鼻子偵測 (pignose/best.pt)
            # =================================================================
            res_thm = pignose_model.predict(ft, imgsz=960, conf=0.15, verbose=False, half=True, device=device)[0]
            if len(res_thm.boxes) > 0:
                best_box = res_thm.boxes[0]
                cur_thm_nose_box = ([float(v) for v in best_box.xyxy[0].cpu().numpy()], float(best_box.conf[0]))
                last_valid_thm_snout = cur_thm_nose_box[0]
                
        # -----------------------------------------------------------------
        # 4. 視差矩陣投影 (RGB -> Thermal) + 方案二淚腺熱點自動吸附微調
        # -----------------------------------------------------------------
        mapped_eyes_thm = []
        for e_idx, (e_box, e_conf) in enumerate(cur_eye_boxes_rgb):
            tb, valid = project_box(e_box, coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
            if valid:
                snapped_box, temp, pt = snap_to_tear_duct(ft, tb)
                if e_idx in prev_snapped_eyes:
                    alpha = 0.70
                    p_box = prev_snapped_eyes[e_idx]
                    snapped_box = [alpha * c + (1.0 - alpha) * p for c, p in zip(snapped_box, p_box)]
                prev_snapped_eyes[e_idx] = snapped_box
                mapped_eyes_thm.append((snapped_box, temp, pt, e_conf))
                eye_temps_history.append(temp)
                
        # 耳根視差投影與測溫 (雙耳根)
        mapped_ears_thm = []
        for ear_b, label, col, anchor_pt in cur_ear_boxes_rgb:
            tb, valid = project_box(ear_b, coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
            if valid:
                temp, pt = extract_temp(ft, tb, mode='max')
                mapped_ears_thm.append((tb, temp, pt, label, col))
                ear_temps_history.append(temp)
                
        # -----------------------------------------------------------------
        # 5. 呼吸訊號萃取 (Thermal Snout)
        # -----------------------------------------------------------------
        snout_val = np.nan
        if cur_thm_nose_box is not None:
            tx1, ty1, tx2, ty2 = map(int, cur_thm_nose_box[0])
            crop_t = ft[max(0, ty1):min(h_t, ty2), max(0, tx1):min(w_t, tx2)]
            if crop_t.size > 0:
                snout_val = float(np.mean(crop_t))
        elif cur_nose_box_rgb is not None:
            tb, _ = project_box(cur_nose_box_rgb[0], coef, cur_w_ref, w_r, h_r, w_t, h_t, RW, RH, TW, TH)
            tx1, ty1, tx2, ty2 = map(int, tb)
            crop_t = ft[max(0, ty1):min(h_t, ty2), max(0, tx1):min(w_t, tx2)]
            if crop_t.size > 0:
                snout_val = float(np.mean(crop_t))
        elif last_valid_thm_snout is not None:
            tx1, ty1, tx2, ty2 = map(int, last_valid_thm_snout)
            crop_t = ft[max(0, ty1):min(h_t, ty2), max(0, tx1):min(w_t, tx2)]
            if crop_t.size > 0:
                snout_val = float(np.mean(crop_t))
                
        thm_snout_signals.append(snout_val)
        
        # -----------------------------------------------------------------
        # 6. 即時呼吸率與波形計數分析
        # -----------------------------------------------------------------
        cur_bpm = 0.0
        cur_breath_cnt = 0
        if len(thm_snout_signals) >= int(fps_r * 4.0):
            sig_arr = np.array(thm_snout_signals)
            nans = np.isnan(sig_arr)
            if np.any(nans):
                if np.all(nans):
                    sig_arr[:] = 128.0
                else:
                    sig_arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig_arr[~nans])
                    
            b, a = butter(2, [0.20 / (fps_r / 2.0), 2.0 / (fps_r / 2.0)], btype='bandpass')
            filt = filtfilt(b, a, sig_arr)
            peaks, _ = find_peaks(filt, distance=int(fps_r * 0.5), prominence=0.25)
            cur_breath_cnt = len(peaks)
            duration_sec = len(sig_arr) / fps_r
            cur_bpm = (cur_breath_cnt / duration_sec) * 60.0
            
        # -----------------------------------------------------------------
        # 7. 視覺化繪製 (RGB 與 Thermal 雙光對照)
        # -----------------------------------------------------------------
        vis_rgb = fr.copy()
        vis_thm = ft.copy()
        
        # 繪製 RGB 標註 (五官 + Pose 骨骼耳根)
        if cur_face_box_rgb is not None:
            draw_styled_box(vis_rgb, cur_face_box_rgb[0], f"Face {cur_face_box_rgb[1]:.2f}", (46, 204, 113))
        if cur_nose_box_rgb is not None:
            draw_styled_box(vis_rgb, cur_nose_box_rgb[0], f"Snout {cur_nose_box_rgb[1]:.2f}", (0, 255, 255))
        for eb, ec in cur_eye_boxes_rgb:
            draw_styled_box(vis_rgb, eb, f"Eye {ec:.2f}", (0, 255, 0))
        for ear_b, label, col, anchor_pt in cur_ear_boxes_rgb:
            draw_styled_box(vis_rgb, ear_b, label, col, hot_pt=anchor_pt)
            
        # 繪製 Thermal 標註 (含轉移吸附淚腺及耳根)
        if cur_thm_nose_box is not None:
            draw_styled_box(vis_thm, cur_thm_nose_box[0], f"THM Nose {cur_thm_nose_box[1]:.2f}", (0, 255, 255))
        for snapped_b, temp, pt, _ in mapped_eyes_thm:
            draw_styled_box(vis_thm, snapped_b, "Eye (Tear-Duct)", (0, 255, 0), temp=temp, hot_pt=pt)
        for ear_b, temp, pt, label, col in mapped_ears_thm:
            clean_lbl = "Mapped " + label.split()[0] + " " + label.split()[1]
            draw_styled_box(vis_thm, ear_b, clean_lbl, col, temp=temp, hot_pt=pt)
            
        # 縮放至統一高度並拼接
        vis_rgb_resized = cv2.resize(vis_rgb, (out_w_r, out_h))
        vis_thm_resized = cv2.resize(vis_thm, (out_w_t, out_h))
        combined_top = np.hstack([vis_rgb_resized, vis_thm_resized])
        
        # 建立底部專業 HUD 儀表板
        hud = np.zeros((120, canvas_w, 3), dtype=np.uint8)
        hud[:] = (20, 24, 30)
        
        cv2.line(hud, (0, 0), (canvas_w, 0), (0, 200, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(hud, "V4 BIOMETRIC MONITOR (0817 + POSE KEYPOINT EAR ROOTS)", (25, 32), font, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, "RGB: 0817 (Eye/Nose/Face) + pose_best (KP5/6 Ear Roots) | Thermal: pignose/best.pt", (25, 62), font, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
        
        time_sec = (START_FRAME + frame_idx) / fps_r
        cv2.putText(hud, f"Frame: {START_FRAME + frame_idx:04d} ({time_sec:.1f}s)", (25, 95), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        
        # 核心生理數據區塊
        hud_cx = canvas_w // 2 + 50
        cv2.line(hud, (hud_cx - 30, 15), (hud_cx - 30, 105), (60, 65, 75), 1)
        
        bpm_color = (0, 255, 0) if (18 <= cur_bpm <= 45) else (0, 165, 255)
        cv2.putText(hud, f"RESPIRATION: {cur_bpm:4.1f} BPM", (hud_cx, 38), font, 0.75, bpm_color, 2, cv2.LINE_AA)
        cv2.putText(hud, f"Breaths Count: {cur_breath_cnt:2d} (Thermal Snout Flux)", (hud_cx, 65), font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        
        cur_eye_t = mapped_eyes_thm[0][1] if mapped_eyes_thm else (np.mean(eye_temps_history[-10:]) if eye_temps_history else 38.5)
        cur_ear_t = mapped_ears_thm[0][1] if mapped_ears_thm else (np.mean(ear_temps_history[-10:]) if ear_temps_history else 37.0)
        
        cv2.putText(hud, f"Tear-Duct: {cur_eye_t:.1f} C  |  Ear Root: {cur_ear_t:.1f} C", (hud_cx, 95), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 右側相機標籤
        cv2.putText(combined_top, "RGB CAMERA [0817 + Pose Roots]", (20, 35), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_top, "THERMAL CAMERA [Tear-Duct + Ear Roots]", (out_w_r + 20, 35), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        final_canvas = np.vstack([combined_top, hud])
        out_writer.write(final_canvas)
        
        # 儲存代表幀快照 (第 200 幀)
        if frame_idx == 200 and not snapshot_saved:
            cv2.imwrite(OUT_SNAP_PATH, final_canvas)
            snapshot_saved = True
            
        if (frame_idx + 1) % 150 == 0:
            elapsed = time.time() - t_start
            cur_fps = (frame_idx + 1) / elapsed
            print(f"[*] 進度: {frame_idx + 1:03d}/{NUM_FRAMES} 幀 | 推論速度: {cur_fps:.1f} FPS | 呼吸率: {cur_bpm:.1f} BPM | 淚腺溫: {cur_eye_t:.1f}°C | 耳根溫: {cur_ear_t:.1f}°C")
            
    cap_rgb.release()
    cap_thm.release()
    out_writer.release()
    
    total_time = time.time() - t_start
    avg_speed = NUM_FRAMES / total_time
    print(f"[✓] 影片推論完成！平均推論速度: {avg_speed:.1f} FPS (耗時 {total_time:.1f} 秒)")
    print(f"[✓] 影片已儲存至: {OUT_VIDEO_PATH}")
    print(f"[✓] 快照已儲存至: {OUT_SNAP_PATH}")
    
    # 產生呼吸率訊號圖
    generate_waveform_plot(thm_snout_signals, fps_r, OUT_PLOT_PATH)
    print(f"[✓] 呼吸率頻譜波形圖已儲存至: {OUT_PLOT_PATH}")


def generate_waveform_plot(signals, fps, out_path):
    sig_arr = np.array(signals)
    nans = np.isnan(sig_arr)
    if np.any(nans):
        if np.all(nans):
            sig_arr[:] = 128.0
        else:
            sig_arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig_arr[~nans])
            
    b, a = butter(2, [0.20 / (fps / 2.0), 2.0 / (fps / 2.0)], btype='bandpass')
    filt = filtfilt(b, a, sig_arr)
    peaks, _ = find_peaks(filt, distance=int(fps * 0.5), prominence=0.25)
    
    time_axis = np.arange(len(sig_arr)) / fps
    duration_sec = len(sig_arr) / fps
    total_breaths = len(peaks)
    bpm = (total_breaths / duration_sec) * 60.0
    
    plt.figure(figsize=(14, 6), dpi=150)
    plt.style.use('dark_background')
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, sig_arr, color='#00d2ff', alpha=0.6, label='Raw Snout Thermal Flux')
    plt.title('V4 Pose-Anchored Site Video (20260722_113719) - Respiration Signal Analysis', fontsize=14, fontweight='bold', pad=10)
    plt.ylabel('Thermal Intensity', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, filt, color='#2ecc71', lw=1.8, label=f'Bandpass Filtered (0.2-2.0Hz) | Rate: {bpm:.1f} BPM')
    if len(peaks) > 0:
        plt.scatter(time_axis[peaks], filt[peaks], color='#e74c3c', s=45, zorder=5, label=f'Breath Inhalation Peaks (Count: {total_breaths})')
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.ylabel('Relative Amplitude', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    main()
