# -*- coding: utf-8 -*-
"""
run_quad_view_unified.py
================================================================================
🚀 V4 雙光通用四分割多視角智慧評測系統
================================================================================
支援任意現場雙光影片評測，自動適配解析度 (1280x720 / 1920x1080 / 1280x960):
1. 【左上】可見光影像 (0817 五官 + pose_best KP 5/6 幾何骨骼耳根)
2. 【左下】熱成像影像 (best_thermal_nose + 方案二淚腺熱點自動吸附 + 雙耳根區域溫)
3. 【右上】雙光空間對準融合 (4x2 動態視差形變矩陣 + 修正鼻寬指標 + 時間軸同步)
4. 【右下】眼窩、耳根與鼻孔即時生理體溫看板 + 示波器滾動呼吸波形
================================================================================
"""

import os
import sys
import time
import argparse
import csv
import json
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
from pig_identity import PigIdentifier, DEFAULT_ID_MODEL_PATH

# 預設模型路徑
PRIMARY_MODEL_PATH   = os.path.join(SCRIPT_DIR, "parameters", "nose_eyes_face_0817.pt")
POSE_MODEL_PATH      = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
YOLO_ROOT_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "best_v12_eye_nose_root.pt")
PIGNOSE_MODEL_PATH   = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
MAPPING_NPZ_PATH     = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")
ID_MODEL_PATH        = DEFAULT_ID_MODEL_PATH


def load_projector(mapping_path):
    data = np.load(mapping_path)
    coef = data['coef'] # 4x2
    rw = float(data['rgb_size'][0]) if 'rgb_size' in data else 1280.0
    rh = float(data['rgb_size'][1]) if 'rgb_size' in data else 720.0
    tw = float(data['thm_size'][0]) if 'thm_size' in data else 1024.0
    th = float(data['thm_size'][1]) if 'thm_size' in data else 768.0
    return coef, rw, rh, tw, th


def project_box(box_cur, coef, w_ref, cur_w, cur_h, target_w, target_h):
    bx1, by1, bx2, by2 = box_cur
    ref_w = (bx2 - bx1) if w_ref is None else w_ref
    
    cx_norm = (bx1 + bx2) / 2.0 / cur_w
    cy_norm = (by1 + by2) / 2.0 / cur_h
    feat = np.array([cx_norm, cy_norm, 1.0, ref_w / cur_w], dtype=np.float64)
    
    tcx_norm, tcy_norm = feat @ coef
    is_valid = (0.0 <= tcx_norm <= 1.0) and (0.0 <= tcy_norm <= 1.0)
    
    sx = coef[0, 0] * (target_w / cur_w)
    sy = coef[1, 1] * (target_h / cur_h)
    tb_w = (bx2 - bx1) * sx
    tb_h = (by2 - by1) * sy
    
    tcx = tcx_norm * target_w
    tcy = tcy_norm * target_h
    
    rx1 = tcx - tb_w / 2.0
    ry1 = tcy - tb_h / 2.0
    rx2 = tcx + tb_w / 2.0
    ry2 = tcy + tb_h / 2.0
    
    return [rx1, ry1, rx2, ry2], is_valid


def extract_temp_top10(frame_thm, box):
    """擷取辨識框內最高溫的 10 個 pixel 數據進行平均"""
    h, w = frame_thm.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(0, min(w - 1, box[2])))
    y2 = int(max(0, min(h - 1, box[3])))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0, (int((x1+x2)/2), int((y1+y2)/2))
    
    crop = frame_thm[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()
    if len(flat) == 0:
        return 0.0, (int((x1+x2)/2), int((y1+y2)/2))
        
    k = min(10, len(flat))
    top10_vals = np.partition(flat, -k)[-k:]
    val = float(np.mean(top10_vals))
    
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    best_pt = (x1 + max_loc[0], y1 + max_loc[1])
    
    temp = 20.0 + (val / 255.0) * 25.0
    return temp, best_pt


def snap_to_tear_duct(frame_thm, raw_box, max_search_radius=30):
    """【方案二：熱成像眼窩淚腺熱點自動吸附微調 + 框內最高溫 10 個 pixel 平均】"""
    h_thm, w_thm = frame_thm.shape[:2]
    rx1, ry1, rx2, ry2 = raw_box
    bw = rx2 - rx1
    bh = ry2 - ry1
    
    sx1 = int(max(0, rx1 - max_search_radius))
    sy1 = int(max(0, ry1 - max_search_radius))
    sx2 = int(min(w_thm, rx2 + max_search_radius))
    sy2 = int(min(h_thm, ry2 + max_search_radius))
    
    if sx2 <= sx1 or sy2 <= sy1:
        temp, pt = extract_temp_top10(frame_thm, raw_box)
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
        temp_top10, pt_top = extract_temp_top10(frame_thm, snapped_box)
        return snapped_box, temp_top10, pt_top
    else:
        temp, pt = extract_temp_top10(frame_thm, raw_box)
        return raw_box, temp, pt


def extract_temp(frame_thm, box, mode='top10'):
    if mode == 'top10' or mode == 'max':
        return extract_temp_top10(frame_thm, box)
    h, w = frame_thm.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(0, min(w - 1, box[2])))
    y2 = int(max(0, min(h - 1, box[3])))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0, (int((x1+x2)/2), int((y1+y2)/2))
    
    crop = frame_thm[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
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
    if y1 < 100:
        # 靠近頂部橫幅時，將標籤畫在框內底部 (下巴處)，避免被橫幅遮擋
        cv2.rectangle(img, (x1, max(0, y2 - th - 8)), (x1 + tw + 8, y2), color, -1)
        cv2.putText(img, tag, (x1 + 4, y2 - 4), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, tag, (x1 + 4, y1 - 4), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    
    if hot_pt is not None:
        px, py = hot_pt
        if 0 <= px < w_img and 0 <= py < h_img:
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)


def add_banner(img, title, color):
    cv2.rectangle(img, (0, 0), (960, 42), (15, 20, 26), -1)
    cv2.line(img, (0, 42), (960, 42), color, 2)
    cv2.putText(img, title, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)


def draw_telemetry_dashboard(eye_temps, ear_temps, snout_temp, cur_bpm, peaks_cnt, sig_filtered, f_idx, fps_r, avg_fps, video_tag, pig_id="Detecting...", id_conf=0.0, ear_mode="pose", period_avgs=None):
    panel = np.zeros((540, 960, 3), dtype=np.uint8)
    panel[:] = (18, 22, 28)
    
    # 頂部儀表板標題
    cv2.rectangle(panel, (0, 0), (960, 42), (28, 36, 46), -1)
    cv2.line(panel, (0, 42), (960, 42), (0, 215, 255), 2)
    
    if pig_id == "Unknown":
        title_text = '[BOTTOM-RIGHT] TELEMETRY DASHBOARD'
    elif pig_id != "Detecting...":
        title_text = '[BOTTOM-RIGHT] TELEMETRY DASHBOARD'
    else:
        title_text = f'[BOTTOM-RIGHT] TELEMETRY | {video_tag}'
        
    cv2.putText(panel, title_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 右上角顯示身分徽章 (Pig ID Badge)
    if pig_id != "Detecting..." and pig_id != "Unknown":
        badge_col = (50, 255, 120) if not str(pig_id).startswith("Pig_Reg_") else (0, 220, 255)
        bg_col = (25, 60, 38) if not str(pig_id).startswith("Pig_Reg_") else (35, 50, 20)
        cv2.rectangle(panel, (640, 7), (945, 36), bg_col, -1)
        cv2.rectangle(panel, (640, 7), (945, 36), badge_col, 1)
        cv2.putText(panel, f'ID: {pig_id}', (650, 27), cv2.FONT_HERSHEY_DUPLEX, 0.56, badge_col, 2, cv2.LINE_AA)
        tag_sub = "NEW REG" if str(pig_id).startswith("Pig_Reg_") else f"{id_conf*100:.0f}%"
        cv2.putText(panel, tag_sub, (860, 27), cv2.FONT_HERSHEY_DUPLEX, 0.50, (220, 255, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, 'REGISTERING...', (780, 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 200, 255), 2, cv2.LINE_AA)
    
    # 四個獨立溫度區塊: 左眼窩、右眼窩、左耳根、右耳根 (不顯示左右平均與鼻子溫度)
    card_w = 218
    gap = 16
    cards_x = [20 + i * (card_w + gap) for i in range(4)]
    
    def _fmt(v):
        return f'{v:.1f}' if (v is not None and v > 0) else '--.-'
        
    if isinstance(eye_temps, dict):
        t_eye_l = eye_temps.get('eye_l')
        t_eye_r = eye_temps.get('eye_r')
    else:
        t_eye_l = eye_temps[0] if len(eye_temps) > 0 else None
        t_eye_r = eye_temps[1] if len(eye_temps) > 1 else None
        
    if isinstance(ear_temps, dict):
        t_ear_l = ear_temps.get('ear_root_l')
        t_ear_r = ear_temps.get('ear_root_r')
    else:
        t_ear_l = ear_temps[0] if len(ear_temps) > 0 else None
        t_ear_r = ear_temps[1] if len(ear_temps) > 1 else None
        
    period_avgs = period_avgs or {}
    avg_eye_l = period_avgs.get('eye_l')
    avg_eye_r = period_avgs.get('eye_r')
    avg_ear_l = period_avgs.get('ear_l')
    avg_ear_r = period_avgs.get('ear_r')
    
    blocks = [
        ("LEFT EYE (L)",  t_eye_l, avg_eye_l, (0, 230, 118)),
        ("RIGHT EYE (R)", t_eye_r, avg_eye_r, (0, 210, 180)),
        ("LEFT EAR (L)",  t_ear_l, avg_ear_l, (255, 200, 0)),
        ("RIGHT EAR (R)", t_ear_r, avg_ear_r, (255, 100, 220)),
    ]
    
    for idx, (b_title, cur_val, avg_val, b_col) in enumerate(blocks):
        cx = cards_x[idx]
        cv2.rectangle(panel, (cx, 55), (cx + card_w, 220), (25, 32, 40), -1)
        cv2.rectangle(panel, (cx, 55), (cx + card_w, 220), b_col, 1)
        cv2.putText(panel, b_title, (cx + 12, 80), cv2.FONT_HERSHEY_DUPLEX, 0.52, b_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f'{_fmt(cur_val)} C', (cx + 14, 132), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Period Avg: {_fmt(avg_val)} C', (cx + 14, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, 'Top-10 Pixels Mean', (cx + 14, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_col, 1, cv2.LINE_AA)
    
    # 下半部: 呼吸率即時監測與示波器滾動波形
    cv2.rectangle(panel, (20, 235), (940, 485), (22, 28, 36), -1)
    cv2.rectangle(panel, (20, 235), (940, 485), (60, 75, 90), 1)
    
    cv2.putText(panel, 'RESPIRATION RATE (RR):', (40, 270), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    bpm_str = f"{cur_bpm:.1f} BPM" if cur_bpm > 0 else "--.- BPM"
    cv2.putText(panel, bpm_str, (340, 272), cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'Breaths: {peaks_cnt}  |  Bandpass: 0.2 - 2.0 Hz', (550, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 200, 220), 1, cv2.LINE_AA)
    
    gx1, gy1, gx2, gy2 = 40, 290, 920, 470
    cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (12, 16, 20), -1)
    
    for y_g in np.linspace(gy1, gy2, 5):
        cv2.line(panel, (gx1, int(y_g)), (gx2, int(y_g)), (25, 35, 45), 1)
    for x_g in np.linspace(gx1, gx2, 10):
        cv2.line(panel, (int(x_g), gy1), (int(x_g), gy2), (25, 35, 45), 1)
        
    if len(sig_filtered) >= 10:
        win_size = min(len(sig_filtered), 150)
        recent_sig = np.array(sig_filtered[-win_size:])
        s_min, s_max = np.min(recent_sig), np.max(recent_sig)
        s_range = max(1e-4, s_max - s_min)
        
        norm_y = (recent_sig - s_min) / s_range
        plot_ys = gy2 - 15 - norm_y * (gy2 - gy1 - 30)
        plot_xs = np.linspace(gx1 + 10, gx2 - 10, len(recent_sig))
        
        pts = np.int32(np.column_stack([plot_xs, plot_ys]))
        cv2.polylines(panel, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[-1]), 4, (0, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Acquiring Respiratory Signal ...", (gx1 + 250, (gy1 + gy2) // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 140, 160), 1, cv2.LINE_AA)
                    
    id_tag_str = f"  |  ID: {pig_id} ({id_conf*100:.0f}%)" if (pig_id != "Detecting...") else ""
    stat_str = f"FRAME: {f_idx:04d} ({f_idx/fps_r:.1f}s)  |  INFER: {avg_fps:.1f} FPS{id_tag_str}  |  GPU: RTX 5070 Ti (FP16)"
    cv2.putText(panel, stat_str, (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 150, 160), 1, cv2.LINE_AA)
    
    return panel


def save_vitals_report(out_dir, video_tag, pig_id, id_conf, final_bpm, breath_cnt, duration_sec, hist_eye_l, hist_eye_r, hist_ear_l, hist_ear_r):
    """將量測期間之呼吸率、左右眼窩平均體溫、左右耳根平均體溫與豬隻身分存成檔案 (CSV + JSON + TXT)"""
    import json
    import csv
    os.makedirs(out_dir, exist_ok=True)
    
    avg_eye_l = round(float(np.mean(hist_eye_l)), 2) if len(hist_eye_l) > 0 else None
    avg_eye_r = round(float(np.mean(hist_eye_r)), 2) if len(hist_eye_r) > 0 else None
    avg_ear_l = round(float(np.mean(hist_ear_l)), 2) if len(hist_ear_l) > 0 else None
    avg_ear_r = round(float(np.mean(hist_ear_r)), 2) if len(hist_ear_r) > 0 else None
    final_bpm_int = int(float(final_bpm) + 0.5) if final_bpm is not None else 0
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    record = {
        "timestamp": ts_str,
        "video_tag": video_tag,
        "pig_id": pig_id,
        "id_confidence": round(float(id_conf), 3),
        "respiration_rate_bpm": final_bpm_int,
        "breath_peaks_count": int(breath_cnt),
        "measurement_duration_sec": round(float(duration_sec), 1),
        "left_eye_temp_avg_c": avg_eye_l,
        "right_eye_temp_avg_c": avg_eye_r,
        "left_ear_root_temp_avg_c": avg_ear_l,
        "right_ear_root_temp_avg_c": avg_ear_r,
        "valid_sample_frames": {
            "left_eye": len(hist_eye_l),
            "right_eye": len(hist_eye_r),
            "left_ear_root": len(hist_ear_l),
            "right_ear_root": len(hist_ear_r)
        },
        "temperature_method": "Top-10 highest pixels mean in ROI"
    }
    
    # 1. 儲存 JSON 檔案
    json_path = os.path.join(out_dir, f"{video_tag}_vitals_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)
        
    # 2. 儲存易讀 TXT 報告
    txt_path = os.path.join(out_dir, f"{video_tag}_vitals_summary.txt")
    def _s(val): return f"{val:.2f} °C" if val is not None else "N/A (未檢出)"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("====================================================================\n")
        f.write(f"🐷 豬隻生理體溫與呼吸率量測結算報告 ({video_tag})\n")
        f.write("====================================================================\n")
        f.write(f"量測時間         : {ts_str}\n")
        f.write(f"豬隻身分 (Pig ID) : {pig_id} (信心度: {id_conf*100:.1f}%)\n")
        f.write(f"量測時長         : {duration_sec:.1f} 秒\n")
        f.write(f"每分鐘呼吸次數   : {final_bpm_int} BPM (共偵測到 {breath_cnt} 次吸氣波峰)\n")
        f.write("--------------------------------------------------------------------\n")
        f.write("【各部位全時段平均體溫 (採框內最高溫 Top-10 Pixels 平均)】\n")
        f.write(f"  - 左眼窩平均溫度 : {_s(avg_eye_l)} (有效取樣: {len(hist_eye_l)} 幀)\n")
        f.write(f"  - 右眼窩平均溫度 : {_s(avg_eye_r)} (有效取樣: {len(hist_eye_r)} 幀)\n")
        f.write(f"  - 左耳根平均溫度 : {_s(avg_ear_l)} (有效取樣: {len(hist_ear_l)} 幀)\n")
        f.write(f"  - 右耳根平均溫度 : {_s(avg_ear_r)} (有效取樣: {len(hist_ear_r)} 幀)\n")
        f.write("====================================================================\n")
        
    # 3. 儲存/追加至總表 CSV 檔案
    csv_path = os.path.join(out_dir, "pig_vitals_summary_records.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "量測時間", "影片標籤", "豬隻身分(Pig_ID)", "每分鐘呼吸次數(BPM)",
                "左眼窩平均溫度(C)", "右眼窩平均溫度(C)", "左耳根平均溫度(C)", "右耳根平均溫度(C)",
                "左眼有效幀數", "右眼有效幀數", "左耳有效幀數", "右耳有效幀數"
            ])
        writer.writerow([
            ts_str, video_tag, pig_id, f"{final_bpm_int}",
            f"{avg_eye_l:.2f}" if avg_eye_l is not None else "N/A",
            f"{avg_eye_r:.2f}" if avg_eye_r is not None else "N/A",
            f"{avg_ear_l:.2f}" if avg_ear_l is not None else "N/A",
            f"{avg_ear_r:.2f}" if avg_ear_r is not None else "N/A",
            len(hist_eye_l), len(hist_eye_r), len(hist_ear_l), len(hist_ear_r)
        ])
        
    print(f"[✓] 生理量測結算檔案已儲存:\n    - TXT:  {txt_path}\n    - JSON: {json_path}\n    - CSV:  {csv_path}")
    return record


def _analyze_respiration(sig_valid, fps):
    """以帶通濾波抓波峰，並直接依總波峰數 / 有效秒數 * 60 換算 BPM。"""
    detrended = sig_valid - np.mean(sig_valid)
    b, a = butter(2, [0.2 / (fps / 2), min(0.95, 2.0 / (fps / 2))], btype='band')
    filtered = filtfilt(b, a, detrended)
    peaks, _ = find_peaks(filtered, distance=max(1, int(fps * 0.4)), prominence=0.03)
    valid_dur_sec = len(sig_valid) / max(1.0, float(fps))
    bpm = (len(peaks) / valid_dur_sec) * 60.0 if len(peaks) >= 2 else 0.0
    return bpm, peaks, filtered


def run_pipeline(rgb_path, thm_path, out_dir, video_tag, start_frame_rgb=0, start_frame_thm=0, num_frames=900, fast_mode=True, ear_mode="pose", auto_register=True, sig_window=None):
    if ear_mode == 'best2_all':
        mode_title = "RGB 全器官單一模型: best(2).pt (眼、鼻、耳根 + 幾何合成臉)"
    elif ear_mode == 'yolo_bbox':
        mode_title = "耳根: YOLO 物件框 (best_v12_eye_nose_root)"
    else:
        mode_title = "耳根: Pose 幾何姿態 (pose_best)"
    print("=" * 80)
    print(f"🚀 啟動 V4 通用四分割智慧評測: {video_tag} | {mode_title} | 模式: {'現場高速/電池省電 (Fast/Eco)' if fast_mode else '全幀推論 (Full)'}")
    print("=" * 80)
    
    os.makedirs(out_dir, exist_ok=True)
    if ear_mode == 'best2_all':
        out_video_path = os.path.join(out_dir, f"{video_tag}_quad_view_best2_all.mp4")
        out_snap_path  = os.path.join(out_dir, f"{video_tag}_quad_snapshot_best2_all.jpg")
        out_plot_path  = os.path.join(out_dir, f"{video_tag}_quad_waveform_best2_all.png")
    elif ear_mode == 'yolo_bbox':
        out_video_path = os.path.join(out_dir, f"{video_tag}_quad_view_yolo_root.mp4")
        out_snap_path  = os.path.join(out_dir, f"{video_tag}_quad_snapshot_yolo_root.jpg")
        out_plot_path  = os.path.join(out_dir, f"{video_tag}_quad_waveform_yolo_root.png")
    else:
        out_video_path = os.path.join(out_dir, f"{video_tag}_quad_view.mp4")
        out_snap_path  = os.path.join(out_dir, f"{video_tag}_quad_snapshot.jpg")
        out_plot_path  = os.path.join(out_dir, f"{video_tag}_quad_waveform.png")
    
    device = setup_laptop_gpu_acceleration()
    if ear_mode == 'best2_all':
        primary_model = load_optimized_yolo(YOLO_ROOT_MODEL_PATH, imgsz=1280, device=device, warmup=False)
        yolo_root_model = None
        pose_model = None
    elif ear_mode == 'yolo_bbox':
        primary_model = load_optimized_yolo(PRIMARY_MODEL_PATH, imgsz=640, device=device, warmup=False)
        yolo_root_model = load_optimized_yolo(YOLO_ROOT_MODEL_PATH, imgsz=1280, device=device, warmup=False)
        pose_model = None
    else:
        primary_model = load_optimized_yolo(PRIMARY_MODEL_PATH, imgsz=640, device=device, warmup=False)
        pose_model = load_optimized_yolo(POSE_MODEL_PATH, imgsz=448, device=device, warmup=False)
        yolo_root_model = None
    pignose_model = load_optimized_yolo(PIGNOSE_MODEL_PATH, imgsz=512, device=device, warmup=False)
    pig_identifier = PigIdentifier(model_path=ID_MODEL_PATH, device=device, half=True, auto_register=auto_register)
    coef, RW, RH, TW, TH = load_projector(MAPPING_NPZ_PATH)
    
    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_thm = cv2.VideoCapture(thm_path)
    
    if not cap_rgb.isOpened() or not cap_thm.isOpened():
        raise RuntimeError(f"無法開啟輸入影片: {rgb_path} 或 {thm_path}")
        
    w_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    
    w_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_t = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_t = cap_thm.get(cv2.CAP_PROP_FPS) or 30.0
    
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, start_frame_rgb)
    cap_thm.set(cv2.CAP_PROP_POS_FRAMES, start_frame_thm)
    
    print(f"[*] 影片規格: RGB {w_r}x{h_r} @ {fps_r:.1f}fps, THM {w_t}x{h_t} @ {fps_t:.1f}fps")
    print(f"[*] 評測區間: RGB {start_frame_rgb}~{start_frame_rgb+num_frames} <-> THM {start_frame_thm}~{start_frame_thm+num_frames}")
    
    CANVAS_W, CANVAS_H = 1920, 1080
    PANEL_W, PANEL_H = 960, 540
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps_r, (CANVAS_W, CANVAS_H))
    
    per_frame_csv_path = os.path.join(out_dir, f"{video_tag}_vitals_per_frame.csv")
    per_frame_f = open(per_frame_csv_path, mode="w", newline="", encoding="utf-8-sig")
    per_frame_writer = csv.writer(per_frame_f)
    per_frame_writer.writerow([
        "Frame_Idx", "Pig_ID", "ID_Confidence",
        "Left_Eye_Temp_C", "Right_Eye_Temp_C", "Left_Ear_Temp_C", "Right_Ear_Temp_C",
        "Snout_Temp_C", "Respiration_BPM", "Breath_Peaks"
    ])
    
    t_sig = []
    sig_filtered_history = []
    fps_history = []
    snapshot_saved = False
    best_snap_score = -1
    
    # 全時段四部位溫度歷史累積陣列 (左眼窩、右眼窩、左耳根、右耳根)
    hist_eye_l = []
    hist_eye_r = []
    hist_ear_l = []
    hist_ear_r = []
    
    last_pose_ears = []
    last_rgb_face = None
    last_rgb_nose = None
    last_rgb_eyes = []
    last_snout_temp = 0.0
    last_thm_nose_box = None
    current_pig_id = "Detecting..."
    current_id_conf = 0.0
    
    color_map = {
        'face': (100, 255, 100),
        'nose': (50, 200, 255),
        'eye': (0, 255, 200),
        'ear_root_l': (255, 200, 0),
        'ear_root_r': (255, 0, 200),
    }
    
    start_all = time.time()
    
    for f_idx in range(num_frames):
        t0 = time.time()
        ret_r, frame_r = cap_rgb.read()
        ret_t, frame_t = cap_thm.read()
        
        if not ret_r or not ret_t:
            break
            
        cur_w_r, cur_h_r = frame_r.shape[1], frame_r.shape[0]
        cur_w_t, cur_h_t = frame_t.shape[1], frame_t.shape[0]
        
        # 1. 可見光五官偵測 (0817 模型 或 best(2) 全器官模型)
        run_primary = (not fast_mode) or (f_idx % 2 == 0) or (last_rgb_face is None)
        if run_primary:
            if ear_mode == 'best2_all':
                res_primary = primary_model(frame_r, imgsz=1280, conf=0.15, verbose=False)[0]
                rgb_face_box, rgb_nose_box, rgb_eye_boxes = None, None, []
                raw_roots = []
                for box in res_primary.boxes:
                    c = int(box.cls[0].item())
                    name = res_primary.names[c]
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    if name == 'nose' and (rgb_nose_box is None or conf > rgb_nose_box[1]):
                        rgb_nose_box = (xyxy, conf)
                    elif name == 'eye':
                        rgb_eye_boxes.append((xyxy, conf))
                    elif name == 'root':
                        raw_roots.append((xyxy, conf))
                
                # 幾何合成臉部框 (Synthetic Face Box)
                if len(rgb_eye_boxes) > 0 and rgb_nose_box is not None:
                    all_x = [e[0][0] for e in rgb_eye_boxes] + [e[0][2] for e in rgb_eye_boxes] + [rgb_nose_box[0][0], rgb_nose_box[0][2]]
                    all_y = [e[0][1] for e in rgb_eye_boxes] + [e[0][3] for e in rgb_eye_boxes] + [rgb_nose_box[0][1], rgb_nose_box[0][3]]
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    pw = int((max_x - min_x) * 0.25)
                    ph = int((max_y - min_y) * 0.20)
                    synth_xyxy = [max(0, min_x - pw), max(0, min_y - ph), min(cur_w_r, max_x + pw), min(cur_h_r, max_y + ph)]
                    avg_c = round((sum(e[1] for e in rgb_eye_boxes)/len(rgb_eye_boxes) + rgb_nose_box[1])/2, 2)
                    rgb_face_box = (synth_xyxy, avg_c)
                elif rgb_nose_box is not None:
                    nx1, ny1, nx2, ny2 = rgb_nose_box[0]
                    nw, nh = nx2 - nx1, ny2 - ny1
                    synth_xyxy = [max(0, nx1 - nw), max(0, ny1 - int(nh * 2.2)), min(cur_w_r, nx2 + nw), min(cur_h_r, ny2 + int(nh * 0.5))]
                    rgb_face_box = (synth_xyxy, round(rgb_nose_box[1] * 0.8, 2))
                elif len(rgb_eye_boxes) > 0:
                    all_x = [e[0][0] for e in rgb_eye_boxes] + [e[0][2] for e in rgb_eye_boxes]
                    all_y = [e[0][1] for e in rgb_eye_boxes] + [e[0][3] for e in rgb_eye_boxes]
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    span_w = max_x - min_x
                    synth_xyxy = [max(0, min_x - int(span_w * 0.3)), max(0, min_y - int(span_w * 0.4)), min(cur_w_r, max_x + int(span_w * 0.3)), min(cur_h_r, max_y + int(span_w * 1.5))]
                    rgb_face_box = (synth_xyxy, round(sum(e[1] for e in rgb_eye_boxes)/len(rgb_eye_boxes) * 0.8, 2))
                
                # 耳根 NMS 與左右指派
                if len(raw_roots) > 1:
                    import torchvision.ops as ops
                    b_t = torch.tensor([r[0] for r in raw_roots], device=device, dtype=torch.float32)
                    c_t = torch.tensor([r[1] for r in raw_roots], device=device, dtype=torch.float32)
                    keep = ops.nms(b_t, c_t, iou_threshold=0.45)
                    raw_roots = [raw_roots[k] for k in keep.tolist()]
                
                cur_ears = []
                if len(raw_roots) >= 2:
                    sorted_roots = sorted(raw_roots[:2], key=lambda x: (x[0][0] + x[0][2]) / 2.0)
                    b_l, conf_l = sorted_roots[0]
                    b_r, conf_r = sorted_roots[1]
                    cur_ears.append(('ear_root_l', b_l, conf_l, (int((b_l[0]+b_l[2])/2), int((b_l[1]+b_l[3])/2))))
                    cur_ears.append(('ear_root_r', b_r, conf_r, (int((b_r[0]+b_r[2])/2), int((b_r[1]+b_r[3])/2))))
                elif len(raw_roots) == 1:
                    b_one, conf_one = raw_roots[0]
                    rcx = (b_one[0] + b_one[2]) / 2.0
                    nose_x = (rgb_nose_box[0][0] + rgb_nose_box[0][2]) / 2.0 if rgb_nose_box else ((rgb_face_box[0][0] + rgb_face_box[0][2]) / 2.0 if rgb_face_box else cur_w_r / 2.0)
                    center = (int((b_one[0] + b_one[2]) / 2), int((b_one[1] + b_one[3]) / 2))
                    if rcx < nose_x:
                        cur_ears.append(('ear_root_l', b_one, conf_one, center))
                    else:
                        cur_ears.append(('ear_root_r', b_one, conf_one, center))
                
                last_rgb_face = rgb_face_box
                last_rgb_nose = rgb_nose_box
                last_rgb_eyes = rgb_eye_boxes
                last_pose_ears = cur_ears
            else:
                res_primary = primary_model(frame_r, imgsz=640, conf=0.25, verbose=False)[0]
                rgb_face_box, rgb_nose_box, rgb_eye_boxes = None, None, []
                for box in res_primary.boxes:
                    c = int(box.cls[0].item())
                    name = res_primary.names[c]
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    if name == 'face' and (rgb_face_box is None or conf > rgb_face_box[1]): rgb_face_box = (xyxy, conf)
                    elif name == 'nose' and (rgb_nose_box is None or conf > rgb_nose_box[1]): rgb_nose_box = (xyxy, conf)
                    elif name == 'eye': rgb_eye_boxes.append((xyxy, conf))
                
                # 未偵測到即清除 (不沿用舊框，避免框卡在空中)
                last_rgb_face = rgb_face_box
                last_rgb_nose = rgb_nose_box
                last_rgb_eyes = rgb_eye_boxes
        else:
            # fast_mode 跳幀：僅沿用「上一幀」的推論結果 (最多延遲 1 幀)
            rgb_face_box = last_rgb_face
            rgb_nose_box = last_rgb_nose
            rgb_eye_boxes = last_rgb_eyes
            
        # 1.5 豬隻個體身分辨識 (Pig ID Recognition - 每 3 幀更新一次，單次僅 ~1.8ms)
        target_face_id = rgb_face_box or last_rgb_face
        if target_face_id is not None and (f_idx % 3 == 0 or current_pig_id == "Detecting..."):
            fx1, fy1, fx2, fy2 = map(int, target_face_id[0])
            crop_face = frame_r[max(0, fy1):min(cur_h_r, fy2), max(0, fx1):min(cur_w_r, fx2)]
            current_pig_id, current_id_conf = pig_identifier.update(crop_face)
        
        # 2. 耳根錨定 (依 ear_mode 決定：best2_all, yolo_bbox 邊界框模型 或 pose 骨骼姿態關鍵點模型)
        if ear_mode == 'best2_all':
            ear_boxes = list(last_pose_ears)
        else:
            ear_boxes = []
            target_face = rgb_face_box or last_rgb_face
            run_ear = (not fast_mode) or (f_idx % 2 == 1) or (len(last_pose_ears) == 0)
            
            if run_ear:
                if ear_mode == 'yolo_bbox':
                    res_root = yolo_root_model(frame_r, imgsz=1280, conf=0.10, verbose=False)[0]
                    raw_roots = []
                    for box in res_root.boxes:
                        c = int(box.cls[0].item())
                        cname = res_root.names[c]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        if cname == 'root' or c == 2:
                            # 空間濾除離豬臉過遠的背景雜訊 (如欄杆頂部)
                            if target_face is not None:
                                fx1, fy1, fx2, fy2 = target_face[0]
                                fw = fx2 - fx1
                                fcx = (fx1 + fx2) / 2.0
                                fcy = (fy1 + fy2) / 2.0
                                rcx = (xyxy[0] + xyxy[2]) / 2.0
                                rcy = (xyxy[1] + xyxy[3]) / 2.0
                                if abs(rcx - fcx) > fw * 1.5 or (fcy - rcy) > fw * 1.5:
                                    continue
                            raw_roots.append((xyxy, conf))
                        elif (cname == 'nose' or c == 1) and conf >= 0.22 and rgb_nose_box is None:
                            rgb_nose_box = (xyxy, conf)
                            last_rgb_nose = rgb_nose_box
                        elif (cname == 'eye' or c == 0) and conf >= 0.22 and len(rgb_eye_boxes) < 2:
                            rgb_eye_boxes.append((xyxy, conf))
                            last_rgb_eyes = rgb_eye_boxes
                    
                    # 若黑豬暗部未檢出完整 face 框，但有檢出鼻/眼，合成頭部框供身分辨識與自動註冊
                    if target_face_id is None and (rgb_nose_box is not None or len(rgb_eye_boxes) > 0):
                        ref_boxes = ([rgb_nose_box[0]] if rgb_nose_box else []) + [eb[0] for eb in rgb_eye_boxes]
                        rx1 = max(0, int(min(b[0] for b in ref_boxes) - 180))
                        ry1 = max(0, int(min(b[1] for b in ref_boxes) - 180))
                        rx2 = min(cur_w_r, int(max(b[2] for b in ref_boxes) + 180))
                        ry2 = min(cur_h_r, int(max(b[3] for b in ref_boxes) + 180))
                        rgb_face_box = ([rx1, ry1, rx2, ry2], 0.50)
                        last_rgb_face = rgb_face_box
                        if f_idx % 3 == 0 or current_pig_id in ("Detecting...", "Unknown"):
                            crop_face = frame_r[ry1:ry2, rx1:rx2]
                            current_pig_id, current_id_conf = pig_identifier.update(crop_face)
                    
                    # NMS 去除重複重疊框
                    if len(raw_roots) > 1:
                        import torchvision.ops as ops
                        b_t = torch.tensor([r[0] for r in raw_roots], device=device, dtype=torch.float32)
                        c_t = torch.tensor([r[1] for r in raw_roots], device=device, dtype=torch.float32)
                        keep = ops.nms(b_t, c_t, iou_threshold=0.45)
                        raw_roots = [raw_roots[k] for k in keep.tolist()]
                    
                    if len(raw_roots) >= 2:
                        sorted_roots = sorted(raw_roots[:2], key=lambda x: (x[0][0] + x[0][2]) / 2.0)
                        b_l, conf_l = sorted_roots[0]
                        b_r, conf_r = sorted_roots[1]
                        ear_boxes.append(('ear_root_l', b_l, conf_l, (int((b_l[0]+b_l[2])/2), int((b_l[1]+b_l[3])/2))))
                        ear_boxes.append(('ear_root_r', b_r, conf_r, (int((b_r[0]+b_r[2])/2), int((b_r[1]+b_r[3])/2))))
                    elif len(raw_roots) == 1:
                        b_one, conf_one = raw_roots[0]
                        rcx = (b_one[0] + b_one[2]) / 2.0
                        nose_x = (rgb_nose_box[0][0] + rgb_nose_box[0][2]) / 2.0 if rgb_nose_box else ((target_face[0][0] + target_face[0][2]) / 2.0 if target_face else cur_w_r / 2.0)
                        center = (int((b_one[0] + b_one[2]) / 2), int((b_one[1] + b_one[3]) / 2))
                        if rcx < nose_x:
                            ear_boxes.append(('ear_root_l', b_one, conf_one, center))
                        else:
                            ear_boxes.append(('ear_root_r', b_one, conf_one, center))
                else:
                    if target_face is not None:
                        fx1, fy1, fx2, fy2 = map(int, target_face[0])
                        fw, fh = fx2 - fx1, fy2 - fy1
                        cx1 = max(0, fx1 - int(fw * 0.30))
                        cy1 = max(0, fy1 - int(fh * 0.40))
                        cx2 = min(cur_w_r, fx2 + int(fw * 0.30))
                        cy2 = min(cur_h_r, fy2 + int(fh * 0.15))
                        
                        crop_head = frame_r[cy1:cy2, cx1:cx2]
                        if crop_head.size > 0:
                            res_pose = pose_model(crop_head, imgsz=448, conf=0.20, verbose=False)[0]
                            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                                kpts = res_pose.keypoints.data[0].cpu().numpy()
                                if len(kpts) > 6:
                                    kp_l, kp_r = kpts[5], kpts[6]
                                    ear_box_size = max(40, int(fw * 0.12))
                                    if kp_l[2] > 0.3:
                                        lx, ly = cx1 + kp_l[0], cy1 + kp_l[1]
                                        b_l = [lx - ear_box_size, ly - ear_box_size, lx + ear_box_size, ly + ear_box_size]
                                        ear_boxes.append(('ear_root_l', b_l, float(kp_l[2]), (int(lx), int(ly))))
                                    if kp_r[2] > 0.3:
                                        rx, ry = cx1 + kp_r[0], cy1 + kp_r[1]
                                        b_r = [rx - ear_box_size, ry - ear_box_size, rx + ear_box_size, ry + ear_box_size]
                                        ear_boxes.append(('ear_root_r', b_r, float(kp_r[2]), (int(rx), int(ry))))
            elif not run_ear:
                # fast_mode 跳幀：僅沿用上一幀的推論結果
                ear_boxes = list(last_pose_ears)
            
            # 推論幀若未偵測到耳根 → 清空，不保留舊框 (避免框卡在空中)
            if run_ear:
                last_pose_ears = ear_boxes
        
        # 繪製【左上】RGB 影像
        vis_rgb = frame_r.copy()
        if rgb_face_box is not None:
            if current_pig_id == "Unknown":
                face_lbl = f"Face {rgb_face_box[1]:.2f} | ID: Registering..."
                face_col = (0, 165, 255)
            elif current_pig_id != "Detecting...":
                face_lbl = f"Pig ID: {current_pig_id} ({current_id_conf*100:.0f}%)"
                face_col = (50, 255, 120) if not str(current_pig_id).startswith("Pig_Reg_") else (0, 220, 255)
            else:
                face_lbl = f"Face {rgb_face_box[1]:.2f}"
                face_col = color_map['face']
            draw_styled_box(vis_rgb, rgb_face_box[0], face_lbl, face_col)
        if rgb_nose_box is not None: draw_styled_box(vis_rgb, rgb_nose_box[0], f"Snout {rgb_nose_box[1]:.2f}", color_map['nose'])
        for eb, c in rgb_eye_boxes: draw_styled_box(vis_rgb, eb, f"Eye {c:.2f}", color_map['eye'])
        for tag, eb, conf, center in ear_boxes:
            prefix = "best(2)" if ear_mode == 'best2_all' else ("YOLO Ear" if ear_mode == 'yolo_bbox' else "Ear")
            lbl = f"{prefix} Root L" if tag == 'ear_root_l' else f"{prefix} Root R"
            draw_styled_box(vis_rgb, eb, f"{lbl} {conf:.2f}", color_map[tag])
            cv2.circle(vis_rgb, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_rgb, center, 7, (255, 255, 255), 1, cv2.LINE_AA)
            
        # 3. 熱成像鼻鏡偵測 (best_thermal_nose, 優化 imgsz=512，fast_mode 每 3 幀推論一次)
        vis_thm = frame_t.copy()
        run_thm = (not fast_mode) or (f_idx % 3 == 0) or (last_thm_nose_box is None)
        if run_thm:
            res_thm = pignose_model(frame_t, imgsz=512, conf=0.20, verbose=False)[0]
            thm_nose_box = None
            for box in res_thm.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                if thm_nose_box is None or conf > thm_nose_box[1]: thm_nose_box = (xyxy, conf)
            # 未偵測到即清除，不沿用舊框
            last_thm_nose_box = thm_nose_box
        else:
            thm_nose_box = last_thm_nose_box
            
        # 修正距離視差指標 ref_w (優先使用鼻框寬度)
        if rgb_nose_box is not None:
            ref_w = float(rgb_nose_box[0][2] - rgb_nose_box[0][0])
        elif rgb_face_box is not None:
            ref_w = float((rgb_face_box[0][2] - rgb_face_box[0][0]) * 0.30)
        else:
            ref_w = float(cur_w_r * 0.12)
            
        snout_temp_mean = 0.0
        if thm_nose_box is not None:
            t_m, _ = extract_temp(frame_t, thm_nose_box[0], mode='mean')
            snout_temp_mean = t_m
            draw_styled_box(vis_thm, thm_nose_box[0], f"THM Nose {thm_nose_box[1]:.2f}", (0, 255, 255), temp=t_m)
        elif rgb_nose_box is not None:
            pbox, valid = project_box(rgb_nose_box[0], coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t)
            if valid:
                t_m, _ = extract_temp(frame_t, pbox, mode='mean')
                snout_temp_mean = t_m
                draw_styled_box(vis_thm, pbox, "Projected Snout", (0, 220, 240), temp=t_m)
                
        if snout_temp_mean > 20.0: last_snout_temp = snout_temp_mean
        else: snout_temp_mean = last_snout_temp
        t_sig.append(snout_temp_mean)
        
        # 方案二：淚腺熱點自動吸附 + 左右眼窩區分 + Top-10 像素平均
        eye_boxes_tagged = []
        if len(rgb_eye_boxes) >= 2:
            sorted_by_conf = sorted(rgb_eye_boxes, key=lambda x: x[1], reverse=True)
            distinct_eyes = [sorted_by_conf[0]]
            for cand in sorted_by_conf[1:]:
                b0, bc = distinct_eyes[0][0], cand[0]
                cx0, cxc = (b0[0] + b0[2]) / 2.0, (bc[0] + bc[2]) / 2.0
                w0 = max(10.0, b0[2] - b0[0])
                if abs(cx0 - cxc) > w0 * 0.6:
                    distinct_eyes.append(cand)
                    break
            if len(distinct_eyes) >= 2:
                sorted_lr = sorted(distinct_eyes[:2], key=lambda x: (x[0][0] + x[0][2]) / 2.0)
                eye_boxes_tagged.append(('eye_l', sorted_lr[0][0], sorted_lr[0][1]))
                eye_boxes_tagged.append(('eye_r', sorted_lr[1][0], sorted_lr[1][1]))
            else:
                b_one, conf_one = distinct_eyes[0]
                ecx = (b_one[0] + b_one[2]) / 2.0
                ref_x = (rgb_nose_box[0][0] + rgb_nose_box[0][2]) / 2.0 if rgb_nose_box else ((rgb_face_box[0][0] + rgb_face_box[0][2]) / 2.0 if rgb_face_box else cur_w_r / 2.0)
                tag_e = 'eye_l' if ecx < ref_x else 'eye_r'
                eye_boxes_tagged.append((tag_e, b_one, conf_one))
        elif len(rgb_eye_boxes) == 1:
            b_one, conf_one = rgb_eye_boxes[0]
            ecx = (b_one[0] + b_one[2]) / 2.0
            ref_x = (rgb_nose_box[0][0] + rgb_nose_box[0][2]) / 2.0 if rgb_nose_box else ((rgb_face_box[0][0] + rgb_face_box[0][2]) / 2.0 if rgb_face_box else cur_w_r / 2.0)
            tag_e = 'eye_l' if ecx < ref_x else 'eye_r'
            eye_boxes_tagged.append((tag_e, b_one, conf_one))

        eye_temps = {}
        for tag_e, eb, _ in eye_boxes_tagged:
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t)
            if valid:
                snapped_box, t_eye, hot_pt = snap_to_tear_duct(frame_t, pbox, max_search_radius=30)
                eye_temps[tag_e] = t_eye
                if t_eye > 20.0:
                    if tag_e == 'eye_l': hist_eye_l.append(t_eye)
                    else: hist_eye_r.append(t_eye)
                lbl_e = "Eye L (Top10)" if tag_e == 'eye_l' else "Eye R (Top10)"
                draw_styled_box(vis_thm, snapped_box, lbl_e, (0, 255, 0), temp=t_eye, hot_pt=hot_pt)
                
        # 雙耳根熱成像區域溫度 (以 L/R 標籤存放，採框內最高溫 Top-10 pixels 平均)
        ear_temps = {}
        for tag, eb, _, _ in ear_boxes:
            pbox, valid = project_box(eb, coef, ref_w, cur_w_r, cur_h_r, cur_w_t, cur_h_t)
            if valid:
                t_ear, ear_pt = extract_temp_top10(frame_t, pbox)
                ear_temps[tag] = t_ear
                if t_ear > 20.0:
                    if tag == 'ear_root_l': hist_ear_l.append(t_ear)
                    else: hist_ear_r.append(t_ear)
                lbl_ear = "Ear L (Top10)" if tag == 'ear_root_l' else "Ear R (Top10)"
                draw_styled_box(vis_thm, pbox, lbl_ear, color_map[tag], temp=t_ear, hot_pt=ear_pt)
                
        # 4. 【右上】RGB 影像與熱影像對準融合 (Aligned Fusion)
        src_grid, dst_grid = [], []
        ref_w_norm = ref_w / cur_w_r
        for gx in np.linspace(cur_w_r * 0.1, cur_w_r * 0.9, 6):
            for gy in np.linspace(cur_h_r * 0.1, cur_h_r * 0.9, 6):
                feat = np.array([gx / cur_w_r, gy / cur_h_r, 1.0, ref_w_norm], dtype=np.float64)
                tcx_norm, tcy_norm = feat @ coef
                src_grid.append([gx, gy])
                dst_grid.append([tcx_norm * cur_w_t, tcy_norm * cur_h_t])
                
        H_thm_to_rgb, _ = cv2.findHomography(np.float32(dst_grid), np.float32(src_grid))
        warped_thm = cv2.warpPerspective(frame_t, H_thm_to_rgb, (cur_w_r, cur_h_r))
        vis_fusion = cv2.addWeighted(frame_r, 0.45, warped_thm, 0.55, 0)
        
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
        sig_win = t_sig[-sig_window:] if (sig_window and len(t_sig) > sig_window) else t_sig
        if len(sig_win) >= int(fps_r * 4):
            sig_arr = np.array(sig_win)
            sig_valid = sig_arr[sig_arr > 0]
            if len(sig_valid) > 90:
                cur_bpm, peaks, filtered = _analyze_respiration(sig_valid, fps_r)
                sig_filtered_history = filtered.tolist()
                peaks_cnt = len(peaks)
                    
        # 計算當前累計平均
        period_avgs = {
            'eye_l': float(np.mean(hist_eye_l)) if len(hist_eye_l) > 0 else None,
            'eye_r': float(np.mean(hist_eye_r)) if len(hist_eye_r) > 0 else None,
            'ear_l': float(np.mean(hist_ear_l)) if len(hist_ear_l) > 0 else None,
            'ear_r': float(np.mean(hist_ear_r)) if len(hist_ear_r) > 0 else None,
        }
                    
        # 6. 四分割縮放與合成 (960 x 540)
        q_tl = cv2.resize(vis_rgb, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_bl = cv2.resize(vis_thm, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_tr = cv2.resize(vis_fusion, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        q_br = draw_telemetry_dashboard(eye_temps, ear_temps, last_snout_temp, cur_bpm, peaks_cnt, sig_filtered_history, f_idx, fps_r, avg_fps, video_tag, pig_id=current_pig_id, id_conf=current_id_conf, ear_mode=ear_mode, period_avgs=period_avgs)
        
        ear_mode_tag = " | Single YOLO: best(2)" if ear_mode == 'best2_all' else (" | Ear: YOLO BBox" if ear_mode == 'yolo_bbox' else " | Ear: Pose KP")
        if current_pig_id == "Unknown":
            top_left_banner = f"[TOP-LEFT] RGB CAMERA | Pig ID: Registering...{ear_mode_tag}"
            top_left_col = (0, 165, 255)
        elif current_pig_id != "Detecting...":
            top_left_banner = f"[TOP-LEFT] RGB CAMERA | Pig ID: {current_pig_id} ({current_id_conf*100:.0f}%){ear_mode_tag}"
            top_left_col = (50, 255, 120)
        else:
            top_left_banner = f"[TOP-LEFT] RGB CAMERA | {video_tag}{ear_mode_tag}"
            top_left_col = (100, 255, 100)
        add_banner(q_tl, top_left_banner, top_left_col)
        add_banner(q_bl, f"[BOTTOM-LEFT] THERMAL CAMERA | Top-10 Pixels Mean + {('best(2) Roots' if ear_mode == 'best2_all' else ('YOLO Roots' if ear_mode == 'yolo_bbox' else 'Ear Roots'))}", (0, 200, 255))
        add_banner(q_tr, f"[TOP-RIGHT] ALIGNED FUSION | 4x2 Matrix {('(best(2) Roots)' if ear_mode == 'best2_all' else ('(YOLO Roots)' if ear_mode == 'yolo_bbox' else ''))}", (255, 180, 0))
        
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[0:PANEL_H, 0:PANEL_W] = q_tl
        canvas[PANEL_H:CANVAS_H, 0:PANEL_W] = q_bl
        canvas[0:PANEL_H, PANEL_W:CANVAS_W] = q_tr
        canvas[PANEL_H:CANVAS_H, PANEL_W:CANVAS_W] = q_br
        
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), (0, 215, 255), 2)
        cv2.line(canvas, (0, PANEL_H), (CANVAS_W, PANEL_H), (0, 215, 255), 2)
        
        out_writer.write(canvas)
        
        # 寫入逐幀生理量測 CSV
        t_el_val = eye_temps.get('eye_l')
        t_er_val = eye_temps.get('eye_r')
        t_al_val = ear_temps.get('ear_root_l')
        t_ar_val = ear_temps.get('ear_root_r')
        per_frame_writer.writerow([
            f_idx, current_pig_id, f"{current_id_conf:.2f}",
            f"{t_el_val:.2f}" if (t_el_val is not None and t_el_val > 0) else "",
            f"{t_er_val:.2f}" if (t_er_val is not None and t_er_val > 0) else "",
            f"{t_al_val:.2f}" if (t_al_val is not None and t_al_val > 0) else "",
            f"{t_ar_val:.2f}" if (t_ar_val is not None and t_ar_val > 0) else "",
            f"{snout_temp_mean:.2f}" if snout_temp_mean > 0 else "",
            f"{int(float(cur_bpm) + 0.5)}" if cur_bpm > 0 else "",
            peaks_cnt
        ])
        
        if f_idx >= min(120, max(10, num_frames - 10)):
            snap_score = (10 if rgb_face_box else 0) + 8 * len(eye_temps) + 8 * len(ear_temps) + (5 if cur_bpm > 0 else 0)
            if not snapshot_saved or snap_score > best_snap_score:
                cv2.imwrite(out_snap_path, canvas)
                snapshot_saved = True
                best_snap_score = snap_score
            
        if f_idx % 60 == 0:
            bpm_display = f"{cur_bpm:.1f} BPM" if cur_bpm > 0 else "--.- BPM"
            t_el = f"{eye_temps['eye_l']:.1f}C" if 'eye_l' in eye_temps else "N/A"
            t_er = f"{eye_temps['eye_r']:.1f}C" if 'eye_r' in eye_temps else "N/A"
            t_earl = f"{ear_temps['ear_root_l']:.1f}C" if 'ear_root_l' in ear_temps else "N/A"
            t_earr = f"{ear_temps['ear_root_r']:.1f}C" if 'ear_root_r' in ear_temps else "N/A"
            id_str = f"{current_pig_id} ({current_id_conf*100:.0f}%)"
            print(f"[*] [{video_tag}] 進度: {f_idx}/{num_frames} 幀 | 豬隻ID: {id_str} | 速度: {avg_fps:.1f} FPS | 呼吸率: {bpm_display} | 左/右眼溫: {t_el}/{t_er} | 左/右耳溫: {t_earl}/{t_earr}")
            
    cap_rgb.release()
    cap_thm.release()
    out_writer.release()
    per_frame_f.close()
    
    total_time = time.time() - start_all
    print(f"[✓] [{video_tag}] 四分割影片生成完畢！平均速度: {num_frames/max(1e-3, total_time):.1f} FPS (耗時: {total_time:.1f} 秒)")
    print(f"[✓] 影片: {out_video_path}")
    print(f"[✓] 快照: {out_snap_path}")
    print(f"[✓] 逐幀生理數據 CSV: {per_frame_csv_path}")
    
    # 產出呼吸信號濾波圖表並結算最終呼吸率與平均體溫
    final_bpm = cur_bpm
    final_peaks_cnt = peaks_cnt
    sig_arr = np.array(t_sig)
    sig_valid = sig_arr[sig_arr > 0]
    if len(sig_valid) > 60:
        final_bpm, peaks, filtered = _analyze_respiration(sig_valid, fps_r)
        final_peaks_cnt = len(peaks)
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        time_axis = np.arange(len(sig_valid)) / fps_r
        
        ax1.plot(time_axis, sig_valid, color='#00d4ff', lw=1.8, label='Raw Snout Thermal Flux')
        ax1.set_ylabel('Thermal Intensity (°C)', color='white', fontsize=12)
        ax1.set_title(f'On-Site Pig Respiration Tracking ({video_tag} | ID: {current_pig_id})', color='white', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper right')
        
        ax2.plot(time_axis, filtered, color='#00ff88', lw=1.8, label=f'Bandpass Filtered (0.2-2.0Hz) | Rate: {int(float(final_bpm) + 0.5)} BPM')
        if len(peaks) > 0:
            ax2.plot(time_axis[peaks], filtered[peaks], 'ro', markersize=6, label=f'Breath Inhalation Peaks (Count: {len(peaks)})')
        ax2.set_xlabel('Time (Seconds)', color='white', fontsize=12)
        ax2.set_ylabel('Filtered Flux', color='white', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=150)
        plt.close()
        print(f"[✓] 呼吸率頻譜波形圖已儲存至: {out_plot_path}")
        
    # 儲存最終結算之呼吸率、左右眼窩與左右耳根平均體溫及豬隻身分檔案
    duration_sec = len(t_sig) / max(1.0, fps_r)
    save_vitals_report(
        out_dir=out_dir,
        video_tag=video_tag,
        pig_id=current_pig_id,
        id_conf=current_id_conf,
        final_bpm=final_bpm,
        breath_cnt=final_peaks_cnt,
        duration_sec=duration_sec,
        hist_eye_l=hist_eye_l,
        hist_eye_r=hist_eye_r,
        hist_ear_l=hist_ear_l,
        hist_ear_r=hist_ear_r
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="295_6", choices=["295_6", "484_6", "484_6_51", "478_2", "356_2", "472_10", "111506", "111318", "113528", "113719", "114220"])
    parser.add_argument("--frames", type=int, default=600, help="處理影格數")
    parser.add_argument("--full", action="store_true", help="啟用全幀推論 (預設為高速交替跳幀模式)")
    parser.add_argument("--ear_mode", type=str, default="pose", choices=["pose", "yolo_bbox", "best2_all"], help="耳根/RGB偵測模式: pose (幾何姿態關鍵點 pose_best)、yolo_bbox (YOLO 雙模型) 或 best2_all (best(2) 單一模型全包: 臉/鼻/眼/耳根)")
    args = parser.parse_args()
    
    fast_mode = not args.full
    
    if args.video == "295_6":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "15棟_295-6_11407031549_可見光_正面.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "15棟_295-6_11407031549_熱成像_正面.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_295_6_quad_view")
        run_pipeline(rgb, thm, out, "Pig_295-6", start_frame_rgb=60, start_frame_thm=60, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "484_6":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_484_6_quad_view")
        run_pipeline(rgb, thm, out, "Pig_484-6", start_frame_rgb=0, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "484_6_51":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181547_可見光_正面(30秒呼吸51次).dav")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181547_熱成像_正面(30秒呼吸51次).dav")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_484_6_51_quad_view")
        run_pipeline(rgb, thm, out, "Pig_484-6_51", start_frame_rgb=0, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "478_2":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "478-2_11410181559_可見光_正面(30秒呼吸46次).mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "478-2_11410181559_熱成像_正面(30秒呼吸46次).mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_478_2_quad_view")
        run_pipeline(rgb, thm, out, "Pig_478-2", start_frame_rgb=0, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "356_2":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "356-2_11409261610_可見光_正面(20秒呼吸30次).mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "356-2_11409261610_熱成像_正面(20秒呼吸30次).mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_356_2_quad_view")
        run_pipeline(rgb, thm, out, "Pig_356-2", start_frame_rgb=0, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "472_10":
        rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "472-10_11410181618_可見光_正面(30秒呼吸56次).mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "472-10_11410181559_熱成像_正面(30秒呼吸56次).mp4" if not os.path.exists(os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "472-10_11410181618_熱成像_正面(30秒呼吸56次).mp4")) else os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "472-10_11410181618_熱成像_正面(30秒呼吸56次).mp4"))
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_472_10_quad_view")
        run_pipeline(rgb, thm, out, "Pig_472-10", start_frame_rgb=0, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
        
    elif args.video == "111506":
        rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_111506.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_111506.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_111506_quad_view")
        run_pipeline(rgb, thm, out, "Site_111506", start_frame_rgb=230, start_frame_thm=200, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)

    elif args.video == "111318":
        rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_111318.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_111318.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_111318_quad_view")
        run_pipeline(rgb, thm, out, "Site_111318", start_frame_rgb=150, start_frame_thm=120, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)

    elif args.video == "113528":
        rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113528.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113528.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_113528_quad_view")
        run_pipeline(rgb, thm, out, "v4_site_113528", start_frame_rgb=30, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)

    elif args.video == "113719":
        rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_113719_quad_view")
        run_pipeline(rgb, thm, out, "v4_site_113719", start_frame_rgb=280, start_frame_thm=250, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)

    elif args.video == "114220":
        rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_114220.mp4")
        thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_114220.mp4")
        out = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_site_114220_quad_view")
        run_pipeline(rgb, thm, out, "v4_site_114220", start_frame_rgb=30, start_frame_thm=0, num_frames=args.frames, fast_mode=fast_mode, ear_mode=args.ear_mode)
