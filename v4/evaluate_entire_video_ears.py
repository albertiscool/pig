# -*- coding: utf-8 -*-
"""
evaluate_entire_video_ears.py
評測 113719 整部影片 (1527 影格) 的耳根辨識率:
1. YOLO 物件邊界框模型: best(2)\best.pt (root 類別)
2. 姿態幾何骨骼關鍵點模型: pose_best.pt (KP 5/6 耳根)
同時統計: 雙耳檢出率、單耳檢出率、完全漏檢率、平均信心度、時間軸動態分佈。
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

RGB_PATH   = r"C:\Users\User\Desktop\final_pig_v0\v3\rgb_record_20260722_113719.mp4"
BEST2_PATH = r"C:\Users\User\Desktop\final_pig_v0\v4\best(2)\best.pt"
M0817_PATH = r"C:\Users\User\Desktop\final_pig_v0\v4\parameters\nose_eyes_face_0817.pt"
POSE_PATH  = r"C:\Users\User\Desktop\final_pig_v0\v4\parameters\pose_best.pt"
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\c44ac0e4-dd1c-4cd9-9334-ef9209e06714"

print("=" * 80)
print("🚀 啟動 113719 整部影片耳根辨識率全面評測 (Frame 280 ~ End)")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] 運算裝置: {device}")

model_best2 = YOLO(BEST2_PATH)
model_0817  = YOLO(M0817_PATH)
model_pose  = YOLO(POSE_PATH)

cap = cv2.VideoCapture(RGB_PATH)
start_frame = 280
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame

# 統計變數
yolo_results = {
    'both': 0,
    'single': 0,
    'none': 0,
    'conf_list': [],
    'count_per_frame': [],
    'timeline_status': [] # 0: none, 1: single, 2: both
}

pose_results = {
    'both': 0,
    'single': 0,
    'none': 0,
    'conf_list': [],
    'count_per_frame': [],
    'timeline_status': []
}

face_results = {
    'detected': 0,
    'conf_list': []
}

t0 = time.time()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cur_h, cur_w = frame.shape[:2]
    
    # ----------------------------------------------------
    # 1. 0817 臉部偵測 (用於 Pose 裁切與對照基準)
    # ----------------------------------------------------
    res_0817 = model_0817(frame, imgsz=640, conf=0.25, verbose=False)[0]
    face_box = None
    for b in res_0817.boxes:
        c = int(b.cls[0].item())
        if model_0817.names[c] == 'face':
            conf = float(b.conf[0].item())
            if face_box is None or conf > face_box[1]:
                face_box = (b.xyxy[0].cpu().numpy().tolist(), conf)
                
    if face_box is not None:
        face_results['detected'] += 1
        face_results['conf_list'].append(face_box[1])

    # ----------------------------------------------------
    # 2. YOLO best(2) 耳根偵測 (全圖推論, 類別 2: root)
    # ----------------------------------------------------
    res_b2 = model_best2(frame, imgsz=1280, conf=0.10, verbose=False)[0]
    yolo_roots = []
    for b in res_b2.boxes:
        c = int(b.cls[0].item())
        cname = res_b2.names[c]
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        if cname == 'root' or c == 2:
            # 空間合理性過濾 (避免誤框天花板欄杆)
            if face_box is not None:
                fx1, fy1, fx2, fy2 = face_box[0]
                fw = fx2 - fx1
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                rcx, rcy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
                if abs(rcx - fcx) > fw * 1.5 or (fcy - rcy) > fw * 1.5:
                    continue
            yolo_roots.append((xyxy, conf))
            
    # NMS 去重
    if len(yolo_roots) > 1:
        import torchvision.ops as ops
        b_t = torch.tensor([r[0] for r in yolo_roots], device=device, dtype=torch.float32)
        c_t = torch.tensor([r[1] for r in yolo_roots], device=device, dtype=torch.float32)
        keep = ops.nms(b_t, c_t, iou_threshold=0.45)
        yolo_roots = [yolo_roots[k] for k in keep.tolist()]
        
    cnt_yolo = len(yolo_roots)
    yolo_results['count_per_frame'].append(cnt_yolo)
    for r in yolo_roots:
        yolo_results['conf_list'].append(r[1])
        
    if cnt_yolo >= 2:
        yolo_results['both'] += 1
        yolo_results['timeline_status'].append(2)
    elif cnt_yolo == 1:
        yolo_results['single'] += 1
        yolo_results['timeline_status'].append(1)
    else:
        yolo_results['none'] += 1
        yolo_results['timeline_status'].append(0)

    # ----------------------------------------------------
    # 3. Pose 姿態幾何關鍵點耳根 (KP 5 & 6)
    # ----------------------------------------------------
    pose_ears_cnt = 0
    if face_box is not None:
        fx1, fy1, fx2, fy2 = map(int, face_box[0])
        fw, fh = fx2 - fx1, fy2 - fy1
        cx1 = max(0, fx1 - int(fw * 0.30))
        cy1 = max(0, fy1 - int(fh * 0.40))
        cx2 = min(cur_w, fx2 + int(fw * 0.30))
        cy2 = min(cur_h, fy2 + int(fh * 0.15))
        
        crop_head = frame[cy1:cy2, cx1:cx2]
        if crop_head.size > 0:
            res_p = model_pose(crop_head, imgsz=448, conf=0.20, verbose=False)[0]
            if res_p.keypoints is not None and len(res_p.keypoints) > 0:
                kpts = res_p.keypoints.data[0].cpu().numpy()
                if len(kpts) > 6:
                    kp_l, kp_r = kpts[5], kpts[6]
                    if kp_l[2] > 0.3:
                        pose_ears_cnt += 1
                        pose_results['conf_list'].append(float(kp_l[2]))
                    if kp_r[2] > 0.3:
                        pose_ears_cnt += 1
                        pose_results['conf_list'].append(float(kp_r[2]))
                        
    pose_results['count_per_frame'].append(pose_ears_cnt)
    if pose_ears_cnt >= 2:
        pose_results['both'] += 1
        pose_results['timeline_status'].append(2)
    elif pose_ears_cnt == 1:
        pose_results['single'] += 1
        pose_results['timeline_status'].append(1)
    else:
        pose_results['none'] += 1
        pose_results['timeline_status'].append(0)
        
    frame_idx += 1
    if frame_idx % 200 == 0 or frame_idx == total_frames:
        elapsed = time.time() - t0
        fps = frame_idx / elapsed
        print(f"[*] 進度: {frame_idx}/{total_frames} 幀 ({frame_idx/total_frames*100:.1f}%) | 速度: {fps:.1f} FPS")

cap.release()
total_eval = frame_idx
print(f"[✓] 評測完成！總評測影格數: {total_eval} 幀，耗時: {time.time()-t0:.2f} 秒")

# ----------------------------------------------------
# 4. 指標計算
# ----------------------------------------------------
y_both_pct = yolo_results['both'] / total_eval * 100
y_single_pct = yolo_results['single'] / total_eval * 100
y_none_pct = yolo_results['none'] / total_eval * 100
y_any_pct = (yolo_results['both'] + yolo_results['single']) / total_eval * 100
y_mean_conf = np.mean(yolo_results['conf_list']) if yolo_results['conf_list'] else 0.0

p_both_pct = pose_results['both'] / total_eval * 100
p_single_pct = pose_results['single'] / total_eval * 100
p_none_pct = pose_results['none'] / total_eval * 100
p_any_pct = (pose_results['both'] + pose_results['single']) / total_eval * 100
p_mean_conf = np.mean(pose_results['conf_list']) if pose_results['conf_list'] else 0.0

print("\n" + "=" * 80)
print(f"📊 113719 整部影片耳根辨識率結果統計 (總計 {total_eval} 幀，約 {total_eval/30:.1f} 秒)")
print("=" * 80)
print(f"{'指標項目':<28} | {'YOLO best(2) 邊界框':<22} | {'Pose 幾何關鍵點':<22}")
print("-" * 80)
print(f"{'雙耳同時檢出率 (Both Ears)':<25} | {y_both_pct:>6.2f}% ({yolo_results['both']:>4}/{total_eval})   | {p_both_pct:>6.2f}% ({pose_results['both']:>4}/{total_eval})")
print(f"{'單耳檢出率 (Single Ear)':<27} | {y_single_pct:>6.2f}% ({yolo_results['single']:>4}/{total_eval})   | {p_single_pct:>6.2f}% ({pose_results['single']:>4}/{total_eval})")
print(f"{'完全漏檢率 (No Ears)':<29} | {y_none_pct:>6.2f}% ({yolo_results['none']:>4}/{total_eval})   | {p_none_pct:>6.2f}% ({pose_results['none']:>4}/{total_eval})")
print(f"{'有效耳根檢出率 (>= 1 Ear)':<25} | {y_any_pct:>6.2f}% ({yolo_results['both']+yolo_results['single']:>4}/{total_eval})   | {p_any_pct:>6.2f}% ({pose_results['both']+pose_results['single']:>4}/{total_eval})")
print(f"{'平均信心度 (Mean Conf)':<27} | {y_mean_conf:>6.3f}                  | {p_mean_conf:>6.3f}")
print("=" * 80)

# ----------------------------------------------------
# 5. 繪製圖表 (時間軸動態圖 + 比例長條圖 + 信心度分佈圖)
# ----------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='#181c24')
plt.subplots_adjust(hspace=0.45)

time_axis = np.arange(total_eval) / 30.0

# 圖 1: 時間軸逐幀耳根辨識狀態對比
ax1 = axes[0]
ax1.set_facecolor('#202632')
ax1.plot(time_axis, pose_results['timeline_status'], label='Pose Keypoint (KP 5/6)', color='#00ff88', lw=1.5, alpha=0.9)
ax1.plot(time_axis, yolo_results['timeline_status'], label='YOLO best(2) BBox (root)', color='#ff9900', lw=1.2, alpha=0.85, linestyle='--')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['0: None', '1: Single Ear', '2: Both Ears'], color='white', fontsize=10)
ax1.set_xlabel('Time (Seconds)', color='white', fontsize=11)
ax1.set_title(f'113719 Ear Root Tracking Timeline ({total_eval} Frames / {total_eval/30:.1f}s)', color='white', fontsize=13, fontweight='bold')
ax1.tick_params(colors='white')
ax1.grid(True, linestyle=':', alpha=0.3, color='gray')
ax1.legend(loc='upper right', facecolor='#2a3242', edgecolor='none', labelcolor='white')

# 圖 2: 移動平均平滑檢出率 (1 秒視窗 = 30 幀)
window = 30
y_roll = np.convolve(np.array(yolo_results['count_per_frame']) > 0, np.ones(window)/window, mode='same') * 100
p_roll = np.convolve(np.array(pose_results['count_per_frame']) > 0, np.ones(window)/window, mode='same') * 100

ax2 = axes[1]
ax2.set_facecolor('#202632')
ax2.plot(time_axis, p_roll, label=f'Pose Moving Detection Rate (Avg: {p_any_pct:.1f}%)', color='#00e5ff', lw=2.0)
ax2.plot(time_axis, y_roll, label=f'YOLO best(2) Moving Detection Rate (Avg: {y_any_pct:.1f}%)', color='#ff3366', lw=2.0)
ax2.set_ylim(-5, 105)
ax2.set_ylabel('Detection Rate (%)', color='white', fontsize=11)
ax2.set_xlabel('Time (Seconds)', color='white', fontsize=11)
ax2.set_title('1-Second Rolling Detection Success Rate (Pose vs YOLO)', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
ax2.grid(True, linestyle=':', alpha=0.3, color='gray')
ax2.legend(loc='lower right', facecolor='#2a3242', edgecolor='none', labelcolor='white')

# 圖 3: 統計直方長條對比圖
ax3 = axes[2]
ax3.set_facecolor('#202632')
labels = ['Both Ears (雙耳齊全)', 'Single Ear (單耳可見)', 'Missed (完全漏檢)', 'Total Active (有效檢出)']
pose_vals = [p_both_pct, p_single_pct, p_none_pct, p_any_pct]
yolo_vals = [y_both_pct, y_single_pct, y_none_pct, y_any_pct]
x = np.arange(len(labels))
width = 0.35

rects1 = ax3.bar(x - width/2, pose_vals, width, label='Pose Keypoint', color='#00ff88', alpha=0.85)
rects2 = ax3.bar(x + width/2, yolo_vals, width, label='YOLO best(2) BBox', color='#ff9900', alpha=0.85)

for rect in rects1:
    h = rect.get_height()
    ax3.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', color='#00ff88', fontweight='bold', fontsize=10)
for rect in rects2:
    h = rect.get_height()
    ax3.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', color='#ff9900', fontweight='bold', fontsize=10)

ax3.set_xticks(x)
ax3.set_xticklabels(labels, color='white', fontsize=11)
ax3.set_ylabel('Percentage (%)', color='white', fontsize=11)
ax3.set_ylim(0, 115)
ax3.set_title('Overall Recognition Ratio Comparison across 1,527 Frames', color='white', fontsize=13, fontweight='bold')
ax3.tick_params(colors='white')
ax3.grid(True, linestyle=':', alpha=0.3, color='gray', axis='y')
ax3.legend(loc='upper right', facecolor='#2a3242', edgecolor='none', labelcolor='white')

out_plot_path = os.path.join(ARTIFACT_DIR, "ear_detection_full_video_113719.png")
plt.tight_layout()
plt.savefig(out_plot_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[✓] 完整評測圖表已儲存至: {out_plot_path}")
