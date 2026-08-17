# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import pandas as pd
import argparse
import time
import queue
import multiprocessing as mp
from threading import Thread, Event
from ultralytics import YOLO
from torchvision import models
from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
# -------------------------------------------------------------
# --- THREADING & MULTIPROCESSING FRAMEWORK ---
# -------------------------------------------------------------

def data_acquisition_worker(rgb_path, thm_path, input_queue, stop_event):
    """
    Producer (Thread 1): 
    Continuously reads frames from RGB and Thermal video sources, 
    synchronizes them by timestamp, and puts them into the input_queue.
    """
    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_thm = cv2.VideoCapture(thm_path)
    
    while not stop_event.is_set():
        if not input_queue.full():
            ret_rgb, rgb_img = cap_rgb.read()
            ret_thm, thm_img = cap_thm.read()
            
            if not ret_rgb or not ret_thm:
                # Video ended
                try:
                    input_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                break
                
            frame_data = {
                'rgb': rgb_img,
                'thm': thm_img,
                'timestamp': time.time()  # Mock sync timestamp
            }
            input_queue.put(frame_data)
        else:
            time.sleep(0.002)
            
    cap_rgb.release()
    cap_thm.release()

def respiratory_algorithm_worker(result_queue, stop_event):
    """
    Consumer (Process 1): 
    100% Offline-Grade Real-Time Triple-Channel (M1, M2, M3) Respiratory Algorithm.
    Applies CLAHE+Morphology, Savgol Filter, Dual-Band Harmonic FFT, and SNR Multi-Vote.
    """
    print("✅ [Consumer Process] Real-time Multi-Channel Voting Algorithm Started.", flush=True)
    np.random.seed(os.getpid())
    
    buffers_t = []
    buffers_m1, buffers_m2, buffers_m3 = [], [], []
    
    import cv2
    from scipy.signal import butter, filtfilt, detrend, savgol_filter
    
    def butter_bandpass(lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return b, a
        
    def resample_equal_time(tw, sw, fs_target=10.0):
        if len(tw) < 2: return np.array([])
        t0, t1 = tw[0], tw[-1]
        tg = np.arange(t0, t1, 1.0/fs_target)
        sg = np.interp(tg, tw, sw)
        return sg

    def cluster_by_proximity(freqs, weights, df_thresh=0.17):
        if len(freqs) == 0: return np.nan, []
        ord_idx = np.argsort(freqs)
        f, w = freqs[ord_idx], weights[ord_idx]
        clusters, cur = [], [0]
        for i in range(1, len(f)):
            if abs(f[i]-f[cur[-1]]) <= df_thresh: cur.append(i)
            else: clusters.append(cur); cur=[i]
        clusters.append(cur)
        best = max(clusters, key=lambda c: float(np.sum(w[c])))
        return float(np.average(f[best], weights=w[best])), ord_idx[best].tolist()
        
    def get_m1(gray_crop):
        return float(np.mean(gray_crop))
        
    def get_m2(gray_crop):
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

    def get_m3(gray_crop):
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

    def process_channel(time_array, signal_array, fs_in=30.0, fs_target=10.0):
        if len(signal_array) < 30: return np.nan, 0.0
        
        sw = np.array(signal_array, dtype=np.float64)
        tw = np.array(time_array, dtype=np.float64) / fs_in
        
        # 1. 為了避免 Live Window 首尾的邊緣效應，先安全降採樣到 10Hz
        sg = resample_equal_time(tw, sw, fs_target=fs_target)
        N = len(sg)
        if N < int(fs_target * 5): return np.nan, 0.0
        
        # 2. 安全去趨勢 (線性) ，避免卷積邊緣造成低頻突波 (這是剛才變 15 BPM 的主因)
        sg = detrend(sg, type='linear')
        
        # 3. Butterworth Bandpass (在 10Hz 環境下過濾 0.2~2.5Hz)
        b, a = butter_bandpass(0.20, 2.50, fs_target, order=6)
        try: sg = filtfilt(b, a, sg)
        except: pass
        
        # 4. Savgol 平滑濾波
        try: sg = savgol_filter(sg, window_length=11, polyorder=3, mode="interp")
        except: pass
        
        # 5. Hanning 波形成形與均值歸零
        sg_win_h = (sg - np.mean(sg)) * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))
        
        # 6. Zero-pad 擴充至 8 倍提升頻率密度
        padN = int(2**np.ceil(np.log2(N * 8)))
        X = np.fft.rfft(sg_win_h, n=padN)
        P = (np.abs(X)**2) / (N**2)
        f = np.fft.rfftfreq(padN, d=1.0/fs_target)
        
        def peak_snr(band):
            mask = (f >= band[0]) & (f <= band[1])
            if not np.any(mask): return np.nan, 0.0
            fb, Pb = f[mask], P[mask]
            k_peak = int(np.argmax(Pb))
            f_pk = float(fb[k_peak])
            df = fb[1] - fb[0]
            bw = max(1, int(round(0.05/df)))
            k0, k1 = max(0, k_peak-bw), min(len(Pb)-1, k_peak+bw)
            noise = np.concatenate([Pb[:k0], Pb[k1+1:]])
            snr = 10.0 * np.log10((np.mean(Pb[k0:k1+1])+1e-12)/(np.median(noise)+1e-12)) if len(noise)>0 else 0.0
            return f_pk, max(0.0, snr)

        # Dual-Band Harmonic Defence
        fl, snrl = peak_snr((0.2, 1.4))
        fh, snrh = peak_snr((0.8, 2.5))
        
        if not np.isfinite(fl): return np.nan, 0.0
        if abs(snrh - snrl) > 5.0: return (fh, snrh) if snrh > snrl else (fl, snrl)
        elif abs(fh - 2*fl) < 0.06 and (snrh - snrl) < 3.0: return fl, snrl
        return fl, snrl

    bpm_display = 0.0
    history_bpm = []
    final_pig_id = "Unknown"

    while not stop_event.is_set():
        try:
            data = result_queue.get(timeout=1.0)
            if data is None:
                break
                
            frame_idx = data['frame_idx']
            pig_id = data['pig_id']
            gray_crop = data.get('gray_crop', None)
            
            # --- 真實的呼吸率訊號累積與 SNR 投票運算 ---
            if gray_crop is not None:
                val_m1 = get_m1(gray_crop)
                val_m2 = get_m2(gray_crop)
                val_m3 = get_m3(gray_crop)
                
                buffers_t.append(frame_idx)
                buffers_m1.append(val_m1)
                buffers_m2.append(val_m2)
                buffers_m3.append(val_m3)
                
                # 容量管控 (限制在最多30秒, 以30FPS計約900幀)
                if len(buffers_t) > 900:
                    buffers_t.pop(0)
                    buffers_m1.pop(0)
                    buffers_m2.pop(0)
                    buffers_m3.pop(0)
                
                # 累積滿約 5 秒才開始投票
                if len(buffers_t) >= 150:
                    try:
                        f1, snr1 = process_channel(buffers_t, buffers_m1, fs_in=30.0, fs_target=10.0)
                        f2, snr2 = process_channel(buffers_t, buffers_m2, fs_in=30.0, fs_target=10.0)
                        f3, snr3 = process_channel(buffers_t, buffers_m3, fs_in=30.0, fs_target=10.0)
                        
                        freqs, snrs = [], []
                        for f_val, s_val in [(f1, snr1), (f2, snr2), (f3, snr3)]:
                            if np.isfinite(f_val):
                                freqs.append(f_val)
                                snrs.append(s_val)
                                
                        if len(freqs) > 0:
                            f_fin, _ = cluster_by_proximity(np.array(freqs), np.array(snrs) + 1.0, 0.17)
                            if np.isfinite(f_fin) and f_fin > 0:
                                bpm_display = f_fin * 60.0
                                history_bpm.append(bpm_display)
                                final_pig_id = pig_id
                    except Exception as e:
                        pass
            
            if frame_idx % 30 == 0:
                print(f"🫁 [Consumer] Frame: {frame_idx:4d} | Pig ID: {pig_id} | Live BPM: {bpm_display:.1f} (Multi-Vote)", flush=True)
                
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ [Consumer Process] Error: {e}", flush=True)
            break
            
    # -----------------------------
    # ✅ 在結束時輸出最終融合總表
    # -----------------------------
    if len(history_bpm) > 0:
        # 使用 IQR (四分位距) 剃除離群值，算出最純淨的平均
        if len(history_bpm) > 3:
            Q1, Q3 = np.percentile(history_bpm, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            valid_bpm = [b for b in history_bpm if lower_bound <= b <= upper_bound]
            final_bpm = np.mean(valid_bpm) if len(valid_bpm) > 0 else np.median(history_bpm)
            outlier_count = len(history_bpm) - len(valid_bpm)
        else:
            valid_bpm = history_bpm
            final_bpm = np.mean(valid_bpm)
            outlier_count = 0

        print("\n" + "="*45, flush=True)
        print(f"📊 Live 即時監測分析報告 - 豬隻 ID: [{final_pig_id}]", flush=True)
        print("-" * 45, flush=True)
        print(f"⭐ 最終即時多通道融合 (Fusion): {final_bpm:6.2f} BPM", flush=True)
        print(f"   (共有 {len(history_bpm)} 筆選票，過濾 {outlier_count} 筆離群值後取平均)")
        print("="*45 + "\n", flush=True)
    else:
        print("\n⚠️ 監測時間過短或未捕獲連續訊號，無法產出最終 BPM 總表。\n", flush=True)
        
    print("🛑 [Consumer Process] Process Ended.", flush=True)

# -------------------------------------------------------------
# --- THREADED VIDEO WRITER ---
# -------------------------------------------------------------
class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, size, queue_size=256):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while not self.stopped or not self.q.empty():
            if not self.q.empty():
                frame = self.q.get()
                self.writer.write(frame)
            else:
                time.sleep(0.002)
                
    def write(self, frame):
        self.q.put(frame) 
        
    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.writer.release()

# -------------------------------------------------------------
# --- MODEL DEFINITION 1: Swin Transformer P2P ---
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
        B, C, H, W = fmap.shape
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

# -------------------------------------------------------------
# --- MODEL DEFINITION 2: ECBHybridModel (Pig ID) ---
# -------------------------------------------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def extract_features(self, x):
        feat_cnn = self.cnn_model(x) 
        return F.normalize(feat_cnn, dim=1)

def get_pig_id(name: str):
    return name.split('_')[0]

# --------------------
# --- CONFIGURATION ---
# --------------------
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_KEYPOINT_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")
YOLOV8_FACE_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")  

IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2] 
FUSION_ALPHA = 0.6 
ID_NAMES = ["Left_Eye", "Right_Eye", "Nose"] 
CONF_THRESHOLD = 0.5 
YOLO_EYE_ID = 0  
YOLO_NOSE_ID = 2 

ID_CHECK_INTERVAL = 15
SIM_THRESHOLD = 0.80

# -------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# -------------------------------------------------------------
def add_frame_text(img, frame, title, transform_type="", fps=0.0, pig_id="Unknown"):
    h_out, w_out, _ = img.shape
    cv2.rectangle(img, (0, 0), (w_out, 110), (0, 0, 0), -1) 
    cv2.putText(img, f"{title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Frame: {frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Process FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"ID: {pig_id}", (w_out - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if transform_type:
        cv2.putText(img, f"Transform: {transform_type}", (w_out - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

def process_yolo_detection_and_sort_points(yolo_results):
    eye_pts = [] 
    nose_pts = [] 
    if hasattr(yolo_results, 'boxes'):
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD: continue
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_pt = np.array([x_center, y_center], dtype=np.float32)
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            if cls_id == YOLO_EYE_ID: 
                eye_pts.append({'pt': center_pt, 'bbox': bbox, 'x': x_center})
            elif cls_id == YOLO_NOSE_ID: 
                nose_pts.append({'pt': center_pt, 'bbox': bbox})
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
    if not pts_found: return None, None, {}, "None"

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

    M, t_type = None, "None"
    if len(src_np) >= 3:
        M = np.vstack([cv2.getAffineTransform(src_np[:3], dst_np[:3]), [0,0,1]])
        t_type = "Affine"
    elif len(src_np) == 2:
        M_p, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
        if M_p is not None: M = np.vstack([M_p, [0,0,1]])
        t_type = "Similarity"
    elif len(src_np) == 1:
        dx, dy = dst_np[0] - src_np[0]
        M = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
        t_type = "Translation"
        
    return M, dst_np, pts_found, t_type

# -------------------------------------------------------------
# --- MAIN EXECUTION (CORE PROCESSOR) ---
# -------------------------------------------------------------

def run_pipeline_step(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Step 1: Producer-Consumer Threaded Mode] Processing: {os.path.basename(args.rgb)}")
    print(f"   Using Device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        detect_model = YOLO(YOLOV8_KEYPOINT_PATH) 
        p2p_model = SwinTransformerP2P().to(device)
        p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
        if device.type == 'cuda': p2p_model.half() 
        p2p_model.eval()

        # 不再單獨載入 face_model，直接共用 detect_model (bestv12) 來節省大量顯示卡記憶體
        face_model_loaded = True

        id_model = None
        if face_model_loaded and os.path.exists(args.id_model) and os.path.exists(args.id_feat):
            id_checkpoint = torch.load(args.id_model, map_location=device)
            num_classes = len(id_checkpoint["classes"]) if "classes" in id_checkpoint else id_checkpoint['model']['classifier_cnn.weight'].shape[0]
            
            id_model = ECBHybridModel(num_classes=num_classes).to(device)
            id_model.load_state_dict(id_checkpoint["model"])
            if device.type == 'cuda': id_model.half() 
            id_model.eval()

            id_features = np.load(args.id_feat)
            id_labels = np.load(args.id_label)
            id_features_torch = torch.tensor(id_features, dtype=torch.float16 if device.type=='cuda' else torch.float32, device=device)
            
            mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            if device.type == 'cuda':
                mean_tensor, std_tensor = mean_tensor.half(), std_tensor.half()
            
            print("✅ 載入身分特徵比對模型成功 (已啟用 GPU 影像加速)！")

    except Exception as e:
        print(f"❌ 模型載入發生錯誤: {e}"); return

    # Queue configuration
    input_queue = queue.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    
    stop_event_th = Event()
    stop_event_mp = mp.Event()

    # Start Producer Thread
    producer_thread = Thread(target=data_acquisition_worker, args=(args.rgb, args.thm, input_queue, stop_event_th))
    producer_thread.daemon = True
    producer_thread.start()

    # Start Consumer Process conditionally
    consumer_process = None
    if args.live_bpm:
        consumer_process = mp.Process(target=respiratory_algorithm_worker, args=(result_queue, stop_event_mp))
        consumer_process.daemon = True
        consumer_process.start()
    else:
        print("⚡ [Mode] Fast Offline Parallel Tracker (Live BPM Disabled)")

    roi_records = []
    frame_count = 0
    current_pig_id = "Unknown"
    collected_faces = []

    writer_thm = None

    if not args.no_show: 
        cv2.namedWindow("Thermal + ROI", cv2.WINDOW_NORMAL)
    
    smoothed_fps = 0.0
    prev_time = time.time()
    start_total_time = time.time()

    def get_warped_box(box, matrix):
        pts_rgb = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
        pts_poly = cv2.perspectiveTransform(pts_rgb, matrix).reshape(-1, 2)
        x1, y1 = np.min(pts_poly[:,0]), np.min(pts_poly[:,1])
        x2, y2 = np.max(pts_poly[:,0]), np.max(pts_poly[:,1])
        return pts_poly, int(x1), int(y1), int(x2), int(y2)

    while True:
        try:
            frame_data = input_queue.get(timeout=1.0)
        except queue.Empty:
            if not producer_thread.is_alive():
                break
            continue
            
        if frame_data is None: # EOF
            if args.live_bpm and consumer_process is not None:
                try: result_queue.put(None, timeout=1.0)
                except: pass
            break
            
        rgb_img = frame_data['rgb']
        thm_img = frame_data['thm']
        
        current_frame = frame_count 
        frame_count += 1
        rgb_h, rgb_w = rgb_img.shape[:2]

        if writer_thm is None:
            thm_h, thm_w = thm_img.shape[:2]
            fps = 30.0 # Standard fallback
            video_out_path = os.path.join(args.out_dir, "out_thm_preview.mp4")
            writer_thm = ThreadedVideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (thm_w, thm_h))

        yolo_res = detect_model.predict(rgb_img, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
        detect_map = process_yolo_detection_and_sort_points(yolo_res)
        
        if 2 in detect_map:
            nose_y = detect_map[2][0][1]
            for eid in [0,1]:
                if eid in detect_map and detect_map[eid][0][1] > nose_y:
                    detect_map[eid], detect_map[2] = detect_map[2], detect_map[eid]
        if 0 in detect_map and 1 in detect_map and detect_map[0][0][0] > detect_map[1][0][0]:
             detect_map[0], detect_map[1] = detect_map[1], detect_map[0]

        M, dst_pts, raw_pts, t_type = get_model_warp_matrix_inference(rgb_img, thm_img, detect_map, p2p_model, device)
        
        # 決定是否需要執行臉部偵測：每 15 幀辨識一次，或當前是 Unknown 且開啟蒐集時每 3 幀一次
        need_face_detect = False
        if id_model and face_model_loaded:
            if current_frame % ID_CHECK_INTERVAL == 0:
                need_face_detect = True
            elif args.collect_faces and current_pig_id.startswith("Unknown") and current_frame % 3 == 0:
                need_face_detect = True

        if need_face_detect:
            # 由於已經使用了三合一模型 (bestv12)，直接從剛剛抓好的 yolo_res 提取臉部框 (class 1)
            face_boxes = [box for box in yolo_res.boxes if int(box.cls[0].cpu().numpy()) == 1]
            if len(face_boxes) > 0:
                # 預設取信心度最高的那個臉部框
                best_box = sorted(face_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0].xyxy[0].cpu().numpy()
                x1_f, y1_f, x2_f, y2_f = map(int, best_box)
                
                pad_x, pad_y = int((x2_f - x1_f) * 0.1), int((y2_f - y1_f) * 0.1)
                x1_f, y1_f = max(0, x1_f - pad_x), max(0, y1_f - pad_y)
                x2_f, y2_f = min(rgb_w, x2_f + pad_x), min(rgb_h, y2_f + pad_y)
                
                face_crop = rgb_img[y1_f:y2_f, x1_f:x2_f]
                
                if face_crop.size > 0:
                    if current_frame % ID_CHECK_INTERVAL == 0:
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        t_face = torch.from_numpy(face_rgb).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
                        t_face = t_face.unsqueeze(0)
                        t_face = F.interpolate(t_face, size=(224, 224), mode='bilinear', align_corners=False)
                        img_tensor_id = (t_face - mean_tensor) / std_tensor
                        
                        with torch.inference_mode():
                            embedding_id = id_model.extract_features(img_tensor_id)
                            sim_id = F.cosine_similarity(embedding_id, id_features_torch)
                            max_sim = torch.max(sim_id).item()
                            if max_sim >= SIM_THRESHOLD:
                                best_idx = torch.argmax(sim_id).item()
                                current_pig_id = f"{get_pig_id(str(id_labels[best_idx]))} ({max_sim:.2f})"
                            else:
                                current_pig_id = f"Unknown ({max_sim:.2f})"

                    if args.collect_faces and current_pig_id.startswith("Unknown") and current_frame % 3 == 0:
                        collected_faces.append(face_crop.copy())

        vis_thm = thm_img
        current_roi_box = {}
        
        if M is not None:
            nx1, ny1, nx2, ny2 = 0, 0, 0, 0
            lx1, ly1, lx2, ly2 = 0, 0, 0, 0
            rx1, ry1, rx2, ry2 = 0, 0, 0, 0
            has_nose = False
            
            if 2 in detect_map:
                _, box = detect_map[2]
                pts_poly, nx1, ny1, nx2, ny2 = get_warped_box(box, M)
                cv2.polylines(vis_thm, [np.int32(pts_poly)], True, (0, 0, 255), 2, cv2.LINE_AA) 
                cv2.rectangle(vis_thm, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2) 
                has_nose = True
                current_roi_box['nose'] = (nx1, ny1, nx2, ny2)

            if 0 in detect_map:
                _, box = detect_map[0]
                pts_poly, lx1, ly1, lx2, ly2 = get_warped_box(box, M)
                cv2.polylines(vis_thm, [np.int32(pts_poly)], True, (255, 0, 0), 2, cv2.LINE_AA) 
                cv2.rectangle(vis_thm, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2) 

            if 1 in detect_map:
                _, box = detect_map[1]
                pts_poly, rx1, ry1, rx2, ry2 = get_warped_box(box, M)
                cv2.polylines(vis_thm, [np.int32(pts_poly)], True, (255, 0, 0), 2, cv2.LINE_AA) 
                cv2.rectangle(vis_thm, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2) 

            if has_nose:
                roi_records.append([current_frame, 1, nx1, ny1, nx2, ny2, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2, current_pig_id])
                
            cv2.putText(vis_thm, "Nose ROI (Green)", (10, thm_h-40), 0, 0.6, (0,255,0), 2)
            cv2.putText(vis_thm, "Eyes ROI (Cyan)", (10, thm_h-15), 0, 0.6, (255,255,0), 2)

        # -----------------------------
        # ✅ 把超輕量灰階 Nose ROI 傳送給 Consumer (僅限 Live 模式)
        # -----------------------------
        if args.live_bpm and consumer_process is not None:
            current_gray_crop = None
            if 'nose' in current_roi_box:
                try:
                    cx1, cy1, cx2, cy2 = current_roi_box['nose']
                    inner = 3
                    cx1_s, cy1_s = max(0, cx1 + inner), max(0, cy1 + inner)
                    cx2_s, cy2_s = min(thm_w, cx2 - inner), min(thm_h, cy2 - inner)
                    
                    if (cx2_s - cx1_s) > 2 and (cy2_s - cy1_s) > 2:
                        crop = thm_img[cy1_s:cy2_s, cx1_s:cx2_s]
                        if crop.size > 0:
                            current_gray_crop = crop[:, :, 1] if len(crop.shape) == 3 else crop # 取 G channel
                except Exception:
                    pass
                    
            # Send info to Consumer Process
            detection_data = {
               'frame_idx': current_frame,
               'pig_id': current_pig_id,
               'roi_box': current_roi_box,
               'gray_crop': current_gray_crop
            }
            try:
                result_queue.put(detection_data, block=False)
            except queue.Full:
                pass

        curr_time = time.time()
        process_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * process_fps if smoothed_fps > 0 else process_fps
        prev_time = curr_time

        add_frame_text(vis_thm, current_frame, "Thermal Output", t_type, smoothed_fps, pig_id=current_pig_id)
        
        writer_thm.write(vis_thm)
        
        if not args.no_show:
            if current_frame % 3 == 0:
                cv2.imshow("Thermal + ROI", vis_thm)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Clean UP
    stop_event_th.set()
    producer_thread.join(timeout=1.0)
    
    if args.live_bpm and consumer_process is not None:
        stop_event_mp.set()
        consumer_process.join(timeout=2.0)
        if consumer_process.is_alive():
            consumer_process.terminate()

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    csv_path = os.path.join(args.out_dir, "roi_boxes.csv")
    df = pd.DataFrame(roi_records, columns=[
        'frame_th', 'valid', 
        'x1_th', 'y1_th', 'x2_th', 'y2_th',       
        'leye_x1', 'leye_y1', 'leye_x2', 'leye_y2', 
        'reye_x1', 'reye_y1', 'reye_x2', 'reye_y2', 
        'pig_id'
    ])
    df.to_csv(csv_path, index=False)
    
    if args.collect_faces and len(collected_faces) > 0:
        default_id = get_pig_id(os.path.basename(args.rgb))
        print(f"\n⚠️ 系統在此次分析中蒐集到了 {len(collected_faces)} 張「未知」的豬臉影像！")
        user_input = input(f">> 請輸入這隻豬的正確編號 (直接按 Enter 預設使用 '{default_id}'): ").strip()
        
        final_pig_id = user_input if user_input else default_id
        
        faces_dir = os.path.join(args.out_dir, "collected_faces", final_pig_id)
        os.makedirs(faces_dir, exist_ok=True)
        for i, face_img in enumerate(collected_faces):
            cv2.imwrite(os.path.join(faces_dir, f"{final_pig_id}_{i:04d}.jpg"), face_img)
        print(f"📸 [Data Collection] 已將 {len(collected_faces)} 張臉部影像存入資料夾: {faces_dir}")

    if writer_thm:
        writer_thm.release()
        
    if not args.no_show: 
        cv2.destroyAllWindows()
    
    print(f"\n✅ [Step 1 Done] Producer-Consumer Architected Mode Execution Completed！")
    print(f"📍 ROI CSV Saved To: {csv_path}")
    print(f"⏱️ Total Cost Time: {total_time:.2f} s | Frames: {frame_count}")
    print(f"🚀 Avg FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    mp.freeze_support() # Support for Windows multiprocessing
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, help="Path to RGB video")
    parser.add_argument("--thm", type=str, help="Path to Thermal video")
    parser.add_argument("--out-dir", type=str, default="data/test_output", help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Disable visual windows")
    parser.add_argument("--live-bpm", action="store_true", help="開啟即時呼吸率監測 Consumer 背景進程 (啟用後會在終端機即時輸出多通道投票 BPM)")
    parser.add_argument("--collect-faces", action="store_true", help="開啟未知識別豬隻臉部蒐集")
    
    parser.add_argument("--id-model", type=str, default=os.path.join(ROOT_DIR, "pig_ecb_best.pth"))
    parser.add_argument("--id-feat", type=str, default=os.path.join(ROOT_DIR, "pig_features_ecb.npy"))
    parser.add_argument("--id-label", type=str, default=os.path.join(ROOT_DIR, "pig_labels_ecb.npy"))
    
    args = parser.parse_args()

    if args.rgb is None:
        args.rgb = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
        args.thm = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

    run_pipeline_step(args)