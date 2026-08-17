# -*- coding: utf-8 -*-
"""
video_pipeline2.py  —  即時呼吸率核心訊號處理管線
整合了 YOLO + P2P 自動追蹤，並實作基於 Saatchi 2019 論文的 ThermalRRCalculator
包含：去趨勢(Detrend)、Butterworth帶通濾波、FFT、SNR篩選與 SMA 平滑化。
"""

import os
import argparse
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, detrend
from ultralytics import YOLO
from torchvision.models import swin_t, Swin_T_Weights

# -------------------------------------------------------------
# --- 目錄與常數設定 ---
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_VID_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "test_vid"))
ROOT_DIR = SCRIPT_DIR
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_KEYPOINT_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "best_yang.pt")

IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2] 
CONF_THRESHOLD = 0.5 
YOLO_EYE_ID = 0  
YOLO_NOSE_ID = 2 

# -------------------------------------------------------------
# --- 1. 核心訊號處理 (ThermalRRCalculator) ---
# -------------------------------------------------------------
class ThermalRRCalculator:
    """
    即時多通道呼吸率計算器 (M1/M2/M3 Triple-Channel + 即時投票融合)
    負責時間序列的維護、預處理、頻譜分析與異常值過濾。
    """
    def __init__(self, fps=30.0, window_sec=15, min_bpm=15, max_bpm=100):
        self.fps = fps
        self.window_sec = window_sec
        self.max_frames = int(fps * window_sec)
        
        # 三通道緩衝區
        self.timestamps = deque(maxlen=self.max_frames)
        self.m1_buffer = deque(maxlen=self.max_frames)
        self.m2_buffer = deque(maxlen=self.max_frames)
        self.m3_buffer = deque(maxlen=self.max_frames)
        
        # 頻率與生理限制
        self.min_freq = min_bpm / 60.0  # 預設 15 BPM = 0.25 Hz
        self.max_freq = max_bpm / 60.0  # 預設 100 BPM = 1.66 Hz
        
        # 平滑化用
        self.last_valid_bpm = None
        self.alpha = 0.3  # Exponential Moving Average 權重
        
    def butter_bandpass(self, order=6):
        """設計 Butterworth Bandpass Filter (帶通濾波器)"""
        nyq = 0.5 * self.fps
        low = self.min_freq / nyq
        high = self.max_freq / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def get_m1(self, gray_crop):
        return float(np.mean(gray_crop))
        
    def get_m2(self, gray_crop):
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi_clahe = clahe.apply(gray_crop)
            blur = cv2.GaussianBlur(roi_clahe, (3,3), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            m = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_id = 1 + np.argmax(areas)
                area_ratio = float(areas.max()) / float(m.size)
                cmean = float(np.mean(gray_crop[labels == max_id]))
                return area_ratio * (255.0 - cmean)
            return float(np.mean(gray_crop))
        except: return float(np.mean(gray_crop))

    def get_m3(self, gray_crop):
        try:
            h, w = gray_crop.shape[:2]
            cy, cx = (h-1)/2.0, (w-1)/2.0
            s = 0.25 * min(w,h)
            if s < 1.0: return float(np.mean(gray_crop))
            
            yy, xx = np.mgrid[0:h, 0:w]
            g = np.exp(-(((xx-cx)**2)/(2*s*s) + ((yy-cy)**2)/(2*s*s)))
            g /= (g.sum() + 1e-12)
            nose_val = float(np.sum(gray_crop.astype(np.float32) * g))
            
            r_base = min(w,h)/2.0
            r_in, r_out = 0.6 * r_base, 1.0 * r_base
            rr = np.sqrt((xx-cx)**2 + (yy-cy)**2)
            mask = ((rr >= r_in) & (rr < r_out))
            cheek_vals = gray_crop[mask]
            
            if cheek_vals.size > 0:
                p80, p100 = np.percentile(cheek_vals, [80, 100])
                sel = cheek_vals[(cheek_vals >= p80) & (cheek_vals <= p100)]
                cheek_val = float(np.mean(sel)) if sel.size > 0 else float(np.mean(cheek_vals))
            else:
                cheek_val = nose_val
            return cheek_val - nose_val
        except: return float(np.mean(gray_crop))

    def add_frame_data(self, roi_crop, current_time):
        """計算三個特徵推入緩衝區"""
        if roi_crop is not None and roi_crop.size > 0:
            val_m1 = self.get_m1(roi_crop)
            val_m2 = self.get_m2(roi_crop)
            val_m3 = self.get_m3(roi_crop)
        else:
            val_m1 = self.m1_buffer[-1] if len(self.m1_buffer) > 0 else 0.0
            val_m2 = self.m2_buffer[-1] if len(self.m2_buffer) > 0 else 0.0
            val_m3 = self.m3_buffer[-1] if len(self.m3_buffer) > 0 else 0.0
            
        self.timestamps.append(current_time)
        self.m1_buffer.append(val_m1)
        self.m2_buffer.append(val_m2)
        self.m3_buffer.append(val_m3)

    def process_channel(self, signal_buffer):
        """對單一通道進行 DSP 處理，回傳最佳頻率與 SNR"""
        signal = np.array(signal_buffer, dtype=np.float64)
        if np.std(signal) < 0.01:
            return np.nan, 0.0

        signal_detrend = detrend(signal, type='linear')
        b, a = self.butter_bandpass(order=6)
        try:
            signal_filtered = filtfilt(b, a, signal_detrend)
        except Exception:
            signal_filtered = signal_detrend

        N = len(signal_filtered)
        window = np.hanning(N)
        signal_win = signal_filtered * window
        
        pad_N = int(2 ** np.ceil(np.log2(N * 4)))
        X = np.fft.rfft(signal_win, n=pad_N)
        magnitude_spectrum = np.abs(X) ** 2
        freqs = np.fft.rfftfreq(pad_N, d=1.0/self.fps)
        
        valid_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(valid_mask):
            return np.nan, 0.0
            
        valid_freqs = freqs[valid_mask]
        valid_mags = magnitude_spectrum[valid_mask]
        
        peak_idx = np.argmax(valid_mags)
        best_freq = valid_freqs[peak_idx]
        
        mean_mag = np.mean(valid_mags)
        peak_mag = valid_mags[peak_idx]
        snr = peak_mag / (mean_mag + 1e-9)
        
        return best_freq, max(0.0, snr)

    def cluster_by_proximity(self, freqs, weights, df_thresh=0.17):
        """將頻率相近的選票群聚，選擇權重最大的群集，並加權平均"""
        if len(freqs) == 0: return np.nan
        ord_idx = np.argsort(freqs)
        f, w = freqs[ord_idx], weights[ord_idx]
        clusters, cur = [], [0]
        for i in range(1, len(f)):
            if abs(f[i]-f[cur[-1]]) <= df_thresh: cur.append(i)
            else: clusters.append(cur); cur=[i]
        clusters.append(cur)
        best = max(clusters, key=lambda c: float(np.sum(w[c])))
        return float(np.average(f[best], weights=w[best]))

    def compute_rr(self):
        """分析三通道，並進行 SNR 投票融合"""
        if len(self.m1_buffer) < int(self.fps * 5):
            return "計算中..."

        f1, snr1 = self.process_channel(self.m1_buffer)
        f2, snr2 = self.process_channel(self.m2_buffer)
        f3, snr3 = self.process_channel(self.m3_buffer)
        
        freqs, snrs = [], []
        for f_val, s_val in [(f1, snr1), (f2, snr2), (f3, snr3)]:
            # 只有當 SNR 大於 2.0 (代表峰值清晰) 才能成為有效票
            if np.isfinite(f_val) and s_val > 2.0:
                freqs.append(f_val)
                snrs.append(s_val)
                
        if len(freqs) == 0:
            return "訊號微弱"
            
        # 進行投票聚類融合 (+1.0 避免權重為 0)
        final_freq = self.cluster_by_proximity(np.array(freqs), np.array(snrs) + 1.0, df_thresh=0.17)
        
        if not np.isfinite(final_freq) or final_freq <= 0:
            return "無法偵測"

        raw_bpm = final_freq * 60.0
        
        # 平滑化 (EMA Smoothing)
        if self.last_valid_bpm is None:
            self.last_valid_bpm = raw_bpm
        else:
            diff = abs(raw_bpm - self.last_valid_bpm)
            alpha = self.alpha if diff < 20 else self.alpha * 0.1
            self.last_valid_bpm = (alpha * raw_bpm) + ((1 - alpha) * self.last_valid_bpm)
            
        return f"{self.last_valid_bpm:.1f} BPM (V:{len(freqs)})"


# -------------------------------------------------------------
# --- 2. Swin Transformer 與 P2P 追蹤模型 ---
# -------------------------------------------------------------
class SwinBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.features = swin.features 
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, feat_dim))
    def forward(self, x):
        x = self.features(x); x = self.proj(x)
        return x.permute(0, 3, 1, 2)

class SwinTransformerP2P(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=6):
        super().__init__()
        self.feat_dim, self.img_size = feat_dim, 640
        self.backbone = SwinBackbone(feat_dim=feat_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos_embed = nn.Linear(2, feat_dim)
        self.regression_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 2))
    def bilinear_sample(self, fmap, pts):
        x = pts[..., 0] / self.img_size * 2 - 1
        y = pts[..., 1] / self.img_size * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2) 
        feat = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear')
        return feat.squeeze(3).permute(0, 2, 1)
    def forward(self, rgb, thm, rgb_pts):
        feat_rgb, feat_thm = self.backbone(rgb), self.backbone(thm)
        memory = feat_thm.flatten(2).permute(0, 2, 1)
        query = self.bilinear_sample(feat_rgb, rgb_pts) + self.query_pos_embed(rgb_pts / self.img_size)
        return self.regression_head(self.decoder(query, memory))

def process_yolo_detection_and_sort_points(yolo_results):
    eye_pts, nose_pts = [], []
    if hasattr(yolo_results, 'boxes'):
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD: continue
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_pt = np.array([x_center, y_center], dtype=np.float32)
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            if cls_id == YOLO_EYE_ID: eye_pts.append({'pt': center_pt, 'bbox': bbox, 'x': x_center})
            elif cls_id == YOLO_NOSE_ID: nose_pts.append({'pt': center_pt, 'bbox': bbox})
            
    final_pts = {}
    if nose_pts: final_pts[2] = (nose_pts[0]['pt'], nose_pts[0]['bbox']) 
    if len(eye_pts) >= 2:
        eye_pts.sort(key=lambda item: item['x'])
        final_pts[0], final_pts[1] = (eye_pts[0]['pt'], eye_pts[0]['bbox']), (eye_pts[1]['pt'], eye_pts[1]['bbox'])
    elif len(eye_pts) == 1 and 2 in final_pts:
        single, nose = eye_pts[0], final_pts[2][0]
        final_pts[1 if single['x'] > nose[0] else 0] = (single['pt'], single['bbox'])
    elif len(eye_pts) == 1:
        final_pts[0] = (eye_pts[0]['pt'], eye_pts[0]['bbox']) 
    return final_pts

def get_model_warp_matrix_inference(rgb_img, thm_img, detect_map, p2p_model, device):
    h_orig, w_orig = rgb_img.shape[:2]
    thm_h, thm_w = thm_img.shape[:2]
    
    pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
    if not pts_found: return None
    rgb_pts_in = [pts_found[cid] if cid in pts_found else [0.0,0.0] for cid in P2P_CLS_IDS]
    model_in = [[p[0]*IMAGE_SIZE/w_orig, p[1]*IMAGE_SIZE/h_orig] for p in rgb_pts_in]

    t_rgb = torch.from_numpy(rgb_img).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
    t_thm = torch.from_numpy(thm_img).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
    t_rgb = F.interpolate(t_rgb.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    t_thm = F.interpolate(t_thm.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    t_pts = torch.tensor([model_in], dtype=torch.float32, device=device).half()

    with torch.inference_mode():
        pred = p2p_model(t_rgb.unsqueeze(0), t_thm.unsqueeze(0), t_pts).cpu().float().numpy()[0]

    dst_pts = [[pred[i][0]*thm_w/IMAGE_SIZE, pred[i][1]*thm_h/IMAGE_SIZE] for i in range(3) if i in pts_found]
    src_np = np.float32([pts_found[i] for i in range(3) if i in pts_found])
    dst_np = np.float32(dst_pts)

    M = None
    if len(src_np) >= 3:
        M = np.vstack([cv2.getAffineTransform(src_np[:3], dst_np[:3]), [0,0,1]])
    elif len(src_np) == 2:
        M_p, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
        if M_p is not None: M = np.vstack([M_p, [0,0,1]])
    elif len(src_np) == 1:
        dx, dy = dst_np[0] - src_np[0]
        M = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    return M

def get_warped_box(box, matrix):
    pts_rgb = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
    pts_poly = cv2.perspectiveTransform(pts_rgb, matrix).reshape(-1, 2)
    x1, y1 = np.min(pts_poly[:,0]), np.min(pts_poly[:,1])
    x2, y2 = np.max(pts_poly[:,0]), np.max(pts_poly[:,1])
    return int(x1), int(y1), int(x2), int(y2)


# -------------------------------------------------------------
# --- 3. 影片選取與管線執行 (與 video_pipeline 相同結構) ---
# -------------------------------------------------------------
def list_available_videos():
    if not os.path.isdir(TEST_VID_DIR): return {}
    all_files = sorted(os.listdir(TEST_VID_DIR), key=lambda x: (not x.endswith('.mp4'), x))
    pairs = {}
    for fname in all_files:
        if not fname.lower().endswith(('.mp4', '.avi')): continue
        parts = fname.split('_')
        pig_id = parts[1] if '棟' in parts[0] and len(parts) > 1 else parts[0]
        if pig_id not in pairs:
            pairs[pig_id] = {'rgb': None, 'thm': None}
        full_path = os.path.join(TEST_VID_DIR, fname)
        if '可見光' in fname or 'rgb' in fname.lower(): pairs[pig_id]['rgb'] = full_path
        elif '熱成像' in fname or 'thm' in fname.lower() or 'thermal' in fname.lower(): pairs[pig_id]['thm'] = full_path
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=str, default=None)
    parser.add_argument('--thm', type=str, default=None)
    parser.add_argument('--pig_id', type=str, default=None)
    args = parser.parse_args()

    rgb_path, thm_path = args.rgb, args.thm
    selected_id = args.pig_id if args.pig_id else "Unknown"
    
    if not rgb_path or not thm_path:
        pairs = list_available_videos()
        if not pairs:
            print("[錯誤] 找不到測試影片目錄")
            return
            
        print("\n" + "=" * 60)
        print("  核心訊號處理實測 - 可用的測試影片組合")
        valid_ids = []
        for i, (pid, info) in enumerate(sorted(pairs.items())):
            status = "可用" if info['rgb'] and info['thm'] else "缺少"
            print(f"  [{i+1}] {pid:>10s} | {status}")
            valid_ids.append(pid)
            
        choice = input("\n>> 請選擇影片編號 (或直接輸入 pig_id): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(valid_ids):
                selected_id = valid_ids[idx]
            else:
                print("[錯誤] 無效的編號。")
                return
        else:
            selected_id = choice
            
        if selected_id in pairs:
            rgb_path = pairs[selected_id]['rgb']
            thm_path = pairs[selected_id]['thm']
        else:
            print(f"[錯誤] 找不到 {selected_id} 的影片組合")
            return

    # --- 載入追蹤模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] 載入模型中 (Device: {device})...")
    detect_model = YOLO(YOLOV8_KEYPOINT_PATH)
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
    if device.type == 'cuda': p2p_model.half()
    p2p_model.eval()
    
    # --- 初始化影片與 RR 計算器 ---
    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_thm = cv2.VideoCapture(thm_path)
    
    fps = cap_thm.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    
    # 初始化 ThermalRRCalculator，限制 15 - 100 BPM
    rr_calc = ThermalRRCalculator(fps=fps, window_sec=15, min_bpm=15, max_bpm=100)
    
    print("[*] 開始逐幀分析... (按 'q' 離開)")
    frame_idx = 0
    import pandas as pd
    roi_records = []
    
    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_thm, frame_thm = cap_thm.read()
        if not ret_rgb or not ret_thm:
            print(">> 影片結束！")
            break
            
        thm_h, thm_w = frame_thm.shape[:2]
        
        # 1. 執行 YOLO 偵測
        yolo_res = detect_model.predict(frame_rgb, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
        detect_map = process_yolo_detection_and_sort_points(yolo_res)
        
        # 2. 執行 P2P 配準
        M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, p2p_model, device)
        
        nx1, ny1, nx2, ny2 = 0, 0, 0, 0
        lx1, ly1, lx2, ly2 = 0, 0, 0, 0
        rx1, ry1, rx2, ry2 = 0, 0, 0, 0
        has_nose = False
        
        if M is not None:
            if 2 in detect_map:
                _, box = detect_map[2]
                nx1, ny1, nx2, ny2 = get_warped_box(box, M)
                has_nose = True
            if 0 in detect_map:
                _, box = detect_map[0]
                lx1, ly1, lx2, ly2 = get_warped_box(box, M)
            if 1 in detect_map:
                _, box = detect_map[1]
                rx1, ry1, rx2, ry2 = get_warped_box(box, M)
                
        if has_nose:
            roi_records.append([frame_idx, 1, nx1, ny1, nx2, ny2, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2, selected_id])

        roi_crop = None
        vis_thm = frame_thm.copy()
        
        # 3. 投射邊界框並擷取 ROI
        if has_nose:
            # 邊界保護
            nx1, ny1 = max(0, nx1), max(0, ny1)
            nx2, ny2 = min(thm_w, nx2), min(thm_h, ny2)
            
            if (nx2 - nx1) > 2 and (ny2 - ny1) > 2:
                # 縮小框 (避免邊緣毛髮干擾)
                inner = 3
                cx1, cy1 = max(0, nx1 + inner), max(0, ny1 + inner)
                cx2, cy2 = min(thm_w, nx2 - inner), min(thm_h, ny2 - inner)
                
                roi_crop = frame_thm[cy1:cy2, cx1:cx2]
                if len(roi_crop.shape) == 3:
                    roi_crop = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                    
                # 畫出追蹤到的 ROI
                cv2.rectangle(vis_thm, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
                
        # 4. 送入 RR 計算器
        current_time = frame_idx / fps
        rr_calc.add_frame_data(roi_crop, current_time)
        
        # 5. 計算即時呼吸率 (每幀計算一次，滑動視窗更新)
        current_bpm_str = rr_calc.compute_rr()
        
        # 6. 視覺化顯示
        cv2.putText(vis_thm, f"Live BPM: {current_bpm_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_thm, f"Frame: {frame_idx}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.imshow("Thermal Pipeline (Live BPM)", vis_thm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1

    cap_rgb.release()
    cap_thm.release()
    cv2.destroyAllWindows()

    # ==========================================
    # 執行與 video_pipeline 相同的離線重算機制
    # ==========================================
    print("\n" + "=" * 60)
    print("  即時追蹤結束！準備啟動離線 auto_rr_pipeline 重算完整呼吸率...")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    auto_rr_path = os.path.join(script_dir, "auto_rr_pipeline.py")
    out_root = "data/test_output"
    os.makedirs(out_root, exist_ok=True)
    
    csv_path = os.path.join(out_root, "roi_boxes.csv")
    df = pd.DataFrame(roi_records, columns=[
        'frame_th', 'valid', 
        'x1_th', 'y1_th', 'x2_th', 'y2_th',       
        'leye_x1', 'leye_y1', 'leye_x2', 'leye_y2', 
        'reye_x1', 'reye_y1', 'reye_x2', 'reye_y2', 
        'pig_id'
    ])
    df.to_csv(csv_path, index=False)
    print(f"\n>> 錄影期間已同步產生 ROI 座標，儲存至: {csv_path}")
    
    cmd = [
        sys.executable, auto_rr_path,
        "--rgb", rgb_path,
        "--thm", thm_path,
        "--out-root", out_root,
        "--skip-step1"
    ]
    
    print(f"執行指令: {' '.join(cmd)}\n")
    try:
        import subprocess
        subprocess.run(cmd, check=True)
        print("\n>> 處理完成！所有完整分析報告已儲存至:", out_root)
    except subprocess.CalledProcessError as e:
        print(f"\n[錯誤] auto_rr_pipeline 執行失敗: {e}")

if __name__ == "__main__":
    main()
