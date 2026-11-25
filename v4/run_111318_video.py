# -*- coding: utf-8 -*-
"""
run_111318_video.py
================================================================================
🚀 V4 現場側臥/近距實測影片智慧評測 (20260722_111318: 0817 五官 + pose_best 骨骼錨定)
================================================================================
評測現場錄製 20260722_111318 影片 (1280x720 RGB + 1024x768 THM):
1. 可見光特徵偵測組合：
   - 五官核心：nose_eyes_face_0817.pt (0.90+ 極高精度眼、鼻、臉部偵測)
   - 耳根專用：pose_best.pt (解剖骨骼關鍵點 KP 5 左耳根、KP 6 右耳根，imgsz=1280)
2. 熱成像模型：使用 "parameters/thm/best_thermal_nose.pt" (pignose/best.pt)
3. 視差補償轉移矩陣：4x2 矩陣座標系映射
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

# 現場錄製影片 (111318 側臥近距片段)
RGB_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_111318.mp4")
THM_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_111318.mp4")

# 輸出設定至各自命名的專屬資料夾
OUT_DIR        = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_111318_mapped")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_VIDEO_PATH = os.path.join(OUT_DIR, "v4_site_111318_mapped.mp4")
OUT_SNAP_PATH  = os.path.join(OUT_DIR, "v4_site_111318_snapshot.jpg")
OUT_PLOT_PATH  = os.path.join(OUT_DIR, "v4_site_111318_waveform.png")


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
    
    d = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
    if d > 3:
        cv2.line(img, (x1, y1), (x1 + d, y1), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + d), (255, 255, 255), 3, cv2.LINE_AA)
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
    print("🚀 啟動 V4 現場實測影片評測 (20260722_111318: 0817 五官 + pose_best 骨骼錨定)")
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
        raise RuntimeError("無法開啟輸入現場影片，請確認路徑！")
        
    w_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    
    w_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_t = cap_thm.get(cv2.CAP_PROP_FPS) or 30.0
    
    n_frames_total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    START_FRAME = 0
    NUM_FRAMES = n_frames_total
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
    
    t_sig = []
    fps_history = []
    snapshot_saved = False
    
    color_map = {
        'face': (100, 255, 100),
        'nose': (50, 200, 255),
        'eye': (0, 255, 200),
        'ear_root_l': (255, 200, 0),
        'ear_root_r': (255, 0, 200),
    }
    
    start_all = time.time()
    
    last_pose_ears = []
    last_rgb_face = None
    last_rgb_nose = None
    last_rgb_eyes = []
    
    for f_idx in range(NUM_FRAMES):
        t0 = time.time()
        ret_r, frame_r = cap_rgb.read()
        ret_t, frame_t = cap_thm.read()
        
        if not ret_r or not ret_t:
            break
            
        cur_w_r, cur_h_r = frame_r.shape[1], frame_r.shape[0]
        cur_w_t, cur_h_t = frame_t.shape[1], frame_t.shape[0]
        
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
        
        res_pose = pose_model(frame_r, imgsz=1280, conf=0.15, verbose=False)[0]
        
        ear_boxes = []
        if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
            kpts = res_pose.keypoints.data[0].cpu().numpy()
            if len(kpts) > 6:
                kp_l = kpts[5] # Left ear root
                kp_r = kpts[6] # Right ear root
                
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
                    
        if len(ear_boxes) > 0:
            last_pose_ears = ear_boxes
        elif len(last_pose_ears) > 0:
            ear_boxes = last_pose_ears
            
        vis_r = frame_r.copy()
        
        if rgb_face_box is not None:
            draw_styled_box(vis_r, rgb_face_box[0], f"Face {rgb_face_box[1]:.2f}", color_map['face'])
        if rgb_nose_box is not None:
            draw_styled_box(vis_r, rgb_nose_box[0], f"Snout {rgb_nose_box[1]:.2f}", color_map['nose'])
        for eb, c in rgb_eye_boxes:
            draw_styled_box(vis_r, eb, f"Eye {c:.2f}", color_map['eye'])
            
        for tag, eb, conf, center in ear_boxes:
            label = "Ear Root L" if tag == 'ear_root_l' else "Ear Root R"
            draw_styled_box(vis_r, eb, f"{label} {conf:.2f}", color_map[tag])
            cv2.circle(vis_r, center, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_r, center, 8, (255, 255, 255), 1, cv2.LINE_AA)
            
        vis_t = frame_t.copy()
        
        res_thm = pignose_model(frame_t, imgsz=640, conf=0.25, verbose=False)[0]
        thm_nose_box = None
        for box in res_thm.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].item())
            if thm_nose_box is None or conf > thm_nose_box[1]:
                thm_nose_box = (xyxy, conf)
                
        snout_temp_mean = 0.0
        if thm_nose_box is not None:
            t_m, _ = extract_temp(frame_t, thm_nose_box[0], mode='mean')
            snout_temp_mean = t_m
            draw_styled_box(vis_t, thm_nose_box[0], f"THM Nose {thm_nose_box[1]:.2f}", (0, 255, 255), temp=t_m)
            
        t_sig.append(snout_temp_mean)
        
        if rgb_nose_box is not None:
            ref_w = float(rgb_nose_box[0][2] - rgb_nose_box[0][0])
        elif rgb_face_box is not None:
            ref_w = float((rgb_face_box[0][2] - rgb_face_box[0][0]) * 0.30)
        else:
            ref_w = 160.0
        
        eye_temps = []
        for idx, (eb, _) in enumerate(rgb_eye_boxes):
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t, RW, RH, TW, TH)
            if valid:
                snapped_box, t_eye, hot_pt = snap_to_tear_duct(frame_t, pbox, max_search_radius=30)
                eye_temps.append(t_eye)
                draw_styled_box(vis_t, snapped_box, f"Eye (Tear-Duct)", (0, 255, 0), temp=t_eye, hot_pt=hot_pt)
                
        ear_temps = []
        for tag, eb, _, _ in ear_boxes:
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t, RW, RH, TW, TH)
            if valid:
                t_ear, ear_pt = extract_temp(frame_t, pbox, mode='max')
                ear_temps.append(t_ear)
                draw_styled_box(vis_t, pbox, "Mapped Ear Root", color_map[tag], temp=t_ear, hot_pt=ear_pt)
                
        cur_fps = 1.0 / max(1e-4, (time.time() - t0))
        fps_history.append(cur_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
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
                peaks, _ = find_peaks(filtered, distance=int(fps_r * 0.4), prominence=0.03)
                peaks_cnt = len(peaks)
                if len(peaks) >= 2:
                    intervals = np.diff(peaks) / fps_r
                    cur_bpm = 60.0 / np.mean(intervals)
                    
        vis_r_resized = cv2.resize(vis_r, (out_w_r, out_h), interpolation=cv2.INTER_LINEAR)
        vis_t_resized = cv2.resize(vis_t, (out_w_t, out_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:out_h, :out_w_r] = vis_r_resized
        canvas[:out_h, out_w_r:] = vis_t_resized
        
        cv2.putText(canvas, "RGB CAMERA [0817 + Pose Roots]", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "THERMAL CAMERA [Tear-Duct + Ear Roots]", (out_w_r + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        
        hud = canvas[out_h:, :]
        hud[:] = (20, 20, 20)
        cv2.line(canvas, (0, out_h), (canvas_w, out_h), (255, 180, 0), 2)
        
        bpm_str = f"{cur_bpm:.1f} BPM" if cur_bpm > 0 else "--.- BPM"
        cv2.putText(hud, f"RESPIRATION: {bpm_str}", (out_w_r + 30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"Breaths Count: {peaks_cnt} (Thermal Snout Flux)", (out_w_r + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        
        t_tear_str = f"{max(eye_temps):.1f} C" if len(eye_temps) > 0 else "N/A"
        t_ear_str = f"{max(ear_temps):.1f} C" if len(ear_temps) > 0 else "N/A"
        cv2.putText(hud, f"Tear-Duct: {t_tear_str}  |  Ear Root: {t_ear_str}", (out_w_r + 30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(hud, "V4 ON-SITE MONITOR (20260722_111318)", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, "RGB: 0817 (Eye/Nose/Face) + pose_best (KP5/6 Ear Roots) | Thermal: pignose/best.pt", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Frame: {f_idx:04d} ({f_idx/fps_r:.1f}s) | Infer Speed: {avg_fps:.1f} FPS", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)
        
        out_writer.write(canvas)
        
        if not snapshot_saved and (f_idx >= 55 or f_idx == NUM_FRAMES // 2):
            cv2.imwrite(OUT_SNAP_PATH, canvas)
            snapshot_saved = True
            
        if f_idx % 60 == 0:
            print(f"[*] 進度: {f_idx}/{NUM_FRAMES} 幀 | 推論速度: {avg_fps:.1f} FPS | 呼吸率: {bpm_str} | 淚腺溫: {t_tear_str} | 耳根溫: {t_ear_str}")
            
    cap_rgb.release()
    cap_thm.release()
    out_writer.release()
    
    total_time = time.time() - start_all
    print(f"[✓] 影片推論完成！平均推論速度: {NUM_FRAMES/total_time:.1f} FPS (耗時 {total_time:.1f} 秒)")
    print(f"[✓] 影片已儲存至: {OUT_VIDEO_PATH}")
    print(f"[✓] 快照已儲存至: {OUT_SNAP_PATH}")
    
    # 產出呼吸信號與濾波分析波形圖
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
        ax1.set_title('On-Site Pig Respiration Tracking (20260722_111318)', color='white', fontsize=14, fontweight='bold')
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
