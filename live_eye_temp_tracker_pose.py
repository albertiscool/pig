import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import re
import csv
import argparse
from datetime import datetime
import easyocr
from ultralytics import YOLO
from scipy.signal import butter, filtfilt, detrend, savgol_filter
import glob
from threading import Thread, Event

# ---------------------------------------------------------
# 雙攝影機同步器（自動偵測模式）：
#   RTSP 即時串流 → 時間戳 buffer 比對，消除網路延遲差
#   影片檔        → 簡單逐幀同步讀取（RGB 第N幀配 THM 第N幀）
# ---------------------------------------------------------
import collections

class SyncedCapture:
    def __init__(self, src_rgb, src_thm, max_buf=60):
        self.cap_rgb = cv2.VideoCapture(src_rgb)
        self.cap_thm = cv2.VideoCapture(src_thm)
        self.cap_rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_thm.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 自動偵測是 RTSP 還是影片檔
        self._is_rtsp = (str(src_rgb).startswith("rtsp") or
                         str(src_thm).startswith("rtsp"))

        if self._is_rtsp:
            # RTSP 模式：各自獨立執行緒 + 時間戳配對
            self.buf_rgb = collections.deque(maxlen=max_buf)
            self.buf_thm = collections.deque(maxlen=max_buf)
            self._eof_rgb = False
            self._eof_thm = False
            self._stop = Event()
            self._t_rgb = Thread(target=self._reader,
                                 args=(self.cap_rgb, self.buf_rgb, '_eof_rgb'),
                                 daemon=True)
            self._t_thm = Thread(target=self._reader,
                                 args=(self.cap_thm, self.buf_thm, '_eof_thm'),
                                 daemon=True)
            self._t_rgb.start()
            self._t_thm.start()
        # 影片模式：直接在 read() 裡逐幀同步讀取，不開執行緒

    def _reader(self, cap, buf, eof_attr):
        """RTSP 專用：背景執行緒持續抓幀並打時間戳"""
        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret:
                buf.append((time.time(), frame))
            else:
                setattr(self, eof_attr, True)
                break

    def read(self):
        if self._is_rtsp:
            # RTSP：從 buffer 找時間差最小的配對
            if not self.buf_rgb or not self.buf_thm:
                return False, None, False, None
            ts_rgb, frame_rgb = self.buf_rgb[-1]
            ts_thm, frame_thm = min(self.buf_thm,
                                    key=lambda x: abs(x[0] - ts_rgb))
            eof = self._eof_rgb or self._eof_thm
            return (not eof), frame_rgb, (not eof), frame_thm
        else:
            # 影片檔：逐幀同步讀取，第N幀配第N幀
            ret_rgb, frame_rgb = self.cap_rgb.read()
            ret_thm, frame_thm = self.cap_thm.read()
            return ret_rgb, frame_rgb, ret_thm, frame_thm

    def isOpened(self):
        return self.cap_rgb.isOpened() and self.cap_thm.isOpened()

    def release(self):
        if self._is_rtsp:
            self._stop.set()
            self._t_rgb.join(timeout=2.0)
            self._t_thm.join(timeout=2.0)
        self.cap_rgb.release()
        self.cap_thm.release()


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
VID_DIR  = os.path.join(ROOT_DIR, "..", "data", "test_vid")

RTSP_RGB_DEFAULT = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
RTSP_THM_DEFAULT = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

# ---------------------------------------------------------
# 從現有的主程式直接匯入 YOLO 與 P2P 映射模型
# (這樣就不需要重複寫上百行的模型架構代碼)
# ---------------------------------------------------------
from video_pipeline2 import (
    SwinTransformerP2P, 
    get_model_warp_matrix_inference, 
    P2P_MODEL_PATH,
    IMAGE_SIZE,
    CONF_THRESHOLD
)

# 預設參數設定區
CSV_SAVE_PATH = "field_eye_temp_records.csv"
ID_CHECK_INTERVAL = 15   # 每幾幀做一次身分辨識
SIM_THRESHOLD     = 0.80  # 相似度門標値

# ---------------------------------------------------------
# 豬隻身分辨識模型 (ECBHybridModel)
# ---------------------------------------------------------
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

def extract_temperature(text):
    match = re.search(r'(\d+\.\d+|\d+)', text)
    if match:
        return float(match.group(1))
    return None

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

def process_channel(time_array, signal_array, fs_in, fs_target=10.0):
    if len(signal_array) < 30: return np.nan, 0.0
    sw = np.array(signal_array, dtype=np.float64)
    tw = np.array(time_array, dtype=np.float64)
    sg = resample_equal_time(tw, sw, fs_target=fs_target)
    N = len(sg)
    if N < int(fs_target * 5): return np.nan, 0.0
    sg = detrend(sg, type='linear')
    b, a = butter_bandpass(0.20, 2.50, fs_target, order=6)
    try: sg = filtfilt(b, a, sg)
    except: pass
    try: sg = savgol_filter(sg, window_length=11, polyorder=3, mode="interp")
    except: pass
    sg_win_h = (sg - np.mean(sg)) * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))
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

    fl, snrl = peak_snr((0.2, 1.4))
    fh, snrh = peak_snr((0.8, 2.5))
    if not np.isfinite(fl): return np.nan, 0.0
    if abs(snrh - snrl) > 5.0: return (fh, snrh) if snrh > snrl else (fl, snrl)
    elif abs(fh - 2*fl) < 0.06 and (snrh - snrl) < 3.0: return fl, snrl
    return fl, snrl

def filter_3sigma(data):
    if not data or len(data) == 0: return None
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr)
    filtered = arr[(arr >= mean - 3 * std) & (arr <= mean + 3 * std)]
    if len(filtered) == 0: return mean
    return np.mean(filtered)

def get_m1(gray_crop):
    """M1: 簡單像素均值"""
    return float(np.mean(gray_crop))

def get_m2(gray_crop):
    """M2: CLAHE + 形態學 + 連通元件面積比"""
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
    """M3: 高斯中心湫度 - 環形周圍差値"""
    try:
        h, w = gray_crop.shape[:2]
        cy, cx = (h-1)/2.0, (w-1)/2.0
        s = 0.25 * min(w, h)
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

def cluster_by_proximity(freqs, weights, df_thresh=0.17):
    """SNR 加權要求群計，決定最佳频率。"""
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

def compute_bpm_triple_channel(buffer_time, buf_m1, buf_m2, buf_m3, elapsed):
    """M1+M2+M3 三通道 SNR 投票，輸出最佳 BPM。"""
    n = len(buf_m1)
    if n < 30 or len(buffer_time) < 2: return 0.0, 0.0
    fs_in = n / elapsed if elapsed > 0 else 30.0
    results = []
    for buf in [buf_m1, buf_m2, buf_m3]:
        try:
            f_val, snr = process_channel(buffer_time, buf, fs_in=fs_in, fs_target=10.0)
            if np.isfinite(f_val):
                results.append((f_val, snr))
        except: pass
    if not results: return 0.0, 0.0
    freqs = np.array([r[0] for r in results])
    snrs  = np.array([r[1] for r in results])
    f_fin, _ = cluster_by_proximity(freqs, snrs + 1.0, df_thresh=0.17)
    if np.isfinite(f_fin) and f_fin > 0:
        best_snr = float(np.max(snrs))
        return f_fin * 60.0, best_snr
    return 0.0, 0.0

def find_video_pair(keyword):
    """根據關鍵字在 data/test_vid 資料夾中尋找可見光和熱成像影片對。"""
    vid_dir = os.path.normpath(VID_DIR)
    all_files = glob.glob(os.path.join(vid_dir, f"*{keyword}*"))
    if not all_files:
        return None, None
    rgb_path = next((f for f in all_files if "可見光" in f), None)
    thm_path = next((f for f in all_files if "熱成像" in f), None)
    return rgb_path, thm_path

def list_available_videos():
    """列出 data/test_vid 資料夾中所有可用的影片組。"""
    vid_dir = os.path.normpath(VID_DIR)
    if not os.path.isdir(vid_dir):
        print(f"[警告] 找不到影片資料夾: {vid_dir}")
        return
    rgb_files = sorted(glob.glob(os.path.join(vid_dir, "*可見光*")))
    print(f"\n[可用影片] 資料夾: {vid_dir}")
    print("-" * 60)
    for f in rgb_files:
        name = os.path.basename(f)
        print(f"  {name}")
    print("-" * 60)
    print("請輸入影片關鍵字（例如 '484-6' 或 '295-6'），")
    print("或直接按 Enter 使用 RTSP 攝影機。")

def main():
    parser = argparse.ArgumentParser(description="豬隻眼部體溫追蹤系統 (Pose版)")
    parser.add_argument("--rgb", type=str, default=None, help="可見光影片路徑或 RTSP URL")
    parser.add_argument("--thm", type=str, default=None, help="熱成像影片路徑或 RTSP URL")
    parser.add_argument("--vid", type=str, default=None, help="影片關鍵字，自動從 data/test_vid 找對應的可見光與熱成像")
    args = parser.parse_args()

    print("=" * 60)
    print("  豬隻體溫追蹤系統 (Live Tracker - Pose版)")
    print("=" * 60)

    # --- 決定影片來源 ---
    rgb_source = None
    thm_source = None

    if args.rgb and args.thm:
        # 直接指定完整路徑
        rgb_source = args.rgb
        thm_source = args.thm
    elif args.vid:
        # 指定關鍵字自動配對
        rgb_source, thm_source = find_video_pair(args.vid)
        if not rgb_source or not thm_source:
            print(f"[錯誤] 找不到含有關鍵字 '{args.vid}' 的可見光與熱成像影片組！")
            list_available_videos()
            return
    else:
        # 互動模式：列出影片清單，讓使用者輸入
        list_available_videos()
        keyword = input("\n請輸入影片關鍵字: ").strip()
        if keyword == "":
            print("[*] 使用 RTSP 攝影機模式")
            rgb_source = RTSP_RGB_DEFAULT
            thm_source = RTSP_THM_DEFAULT
        else:
            rgb_source, thm_source = find_video_pair(keyword)
            if not rgb_source or not thm_source:
                print(f"[錯誤] 找不到含有關鍵字 '{keyword}' 的影片組！請確認關鍵字是否正確。")
                return

    print(f"\n[*] 可見光影片: {os.path.basename(str(rgb_source))}")
    print(f"[*] 熱成像影片: {os.path.basename(str(thm_source))}")

    # 1. 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在載入 YOLO Pose 與 Swin P2P 模型 (Device: {device})...")
    model_path = os.path.join(ROOT_DIR, "0529_1519trainingresultmore", "train_result", "weights", "best.pt")
    detect_model = YOLO(model_path)

    # 身分辨識用的可見光偵測模型 (bestv12 可偵測臉部)
    id_detect_model = YOLO(os.path.join(ROOT_DIR, "..", "parameters", "bestv12.pt"))
    
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model_path = os.path.join(ROOT_DIR, "..", "parameters", "thm", "swin_p2p_best_1873.pth")
    p2p_model.load_state_dict(torch.load(p2p_model_path, map_location=device, weights_only=True))
    if device.type == 'cuda': p2p_model.half()
    p2p_model.eval()
    print(f"[*] P2P 映射模型: swin_p2p_best_1873.pth")

    # 載入 ECBHybridModel 身分辨識模型
    id_model = None
    id_features_torch = None
    id_labels = None
    mean_tensor = std_tensor = None
    id_model_path = os.path.join(ROOT_DIR, "..", "parameters", "pig_ecb_best.pth")
    id_feat_path  = os.path.join(ROOT_DIR, "..", "parameters", "pig_features_ecb.npy")
    id_label_path = os.path.join(ROOT_DIR, "..", "parameters", "pig_labels_ecb.npy")
    if os.path.exists(id_model_path) and os.path.exists(id_feat_path):
        try:
            id_ckpt = torch.load(id_model_path, map_location=device)
            num_classes = len(id_ckpt["classes"]) if "classes" in id_ckpt else id_ckpt['model']['classifier_cnn.weight'].shape[0]
            id_model = ECBHybridModel(num_classes=num_classes).to(device)
            id_model.load_state_dict(id_ckpt["model"])
            if device.type == 'cuda': id_model.half()
            id_model.eval()
            id_features = np.load(id_feat_path)
            id_labels    = np.load(id_label_path)
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            id_features_torch = torch.tensor(id_features, dtype=dtype, device=device)
            mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            std_tensor  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
            if device.type == 'cuda':
                mean_tensor, std_tensor = mean_tensor.half(), std_tensor.half()
            print("[*] 身分辨識模型載入成功！")
        except Exception as e:
            print(f"[警告] 身分辨識模型載入失敗: {e}")
            id_model = None
    else:
        print("[提示] 找不到身分辨識模型檔案，將以 Unknown 顯示豬隻身分。")

    print("[*] 正在載入 EasyOCR 動態溫標辨識模型...")
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    # 2. 開啟影像串流
    print(f"\n[*] 正在開啟影片...")
    # 用底面同步執行緒同時抓取兩台攝影機，消除 1 秒同步差
    cap = SyncedCapture(rgb_source, thm_source)

    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片，請確認路徑是否正確！")
        return

    # 3. 初始化 CSV 記錄檔
    if not os.path.exists(CSV_SAVE_PATH):
        with open(CSV_SAVE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Pig_ID', 'Left_Eye', 'Right_Eye', 'Left_Nostril', 'Right_Nostril', 'Nose_Center', 'Left_Ear', 'Right_Ear', 'Max_Scale', 'Min_Scale', 'BPM'])

    # 變數初始化
    current_max_temp = 35.0
    current_min_temp = 20.0
    last_ocr_time = 0
    
    # 60秒錄製系統狀態
    is_recording = False
    record_start_time = 0
    buffer_temps = {}
    # M1/M2/M3 三通道訊號緩衝區
    buffer_m1, buffer_m2, buffer_m3 = [], [], []
    buffer_time = []
    
    # 最終穩定數值
    final_bpm = 0.0
    final_temps = {'Left Eye': None, 'Right Eye': None, 'Left Nostril': None, 'Right Nostril': None, 'Nose Center': None, 'Left Ear': None, 'Right Ear': None}
    
    # 豬隻身分辨識狀態
    current_pig_id = "Unknown"
    current_frame  = 0
    
    print("\n[系統準備就緒] 視窗即將跳出。")
    print("操作說明:")
    print("  - 按 'r' 鍵：開始完整錄製並計算穩定呼吸率與 3-Sigma 溫度 (影片模式下自動從頭錄製)。")
    print("  - 按 's' 鍵：手動立即將目前的溫度寫入 CSV 記錄檔。")
    print("  - 按 'q' 鍵：離開程式。")

    # 若是影片檔，自動開始錄製（不需要手動按 r）
    is_video_file = not str(rgb_source).startswith("rtsp")
    if is_video_file:
        is_recording = True
        record_start_time = time.time()
        buffer_temps = {k: [] for k in ['Left Eye', 'Right Eye', 'Left Nostril', 'Right Nostril', 'Nose Center', 'Left Ear', 'Right Ear']}
        buffer_m1, buffer_m2, buffer_m3 = [], [], []
        buffer_time = []
        print("[自動錄製] 偵測到影片檔，自動從第一幀開始錄製，影片結束時自動結算！")

    while True:
        ret_rgb, frame_rgb, ret_thm, frame_thm = cap.read()

        # 如果幀還沒到（執行緒初始化中），等一下
        if frame_rgb is None or frame_thm is None:
            time.sleep(0.01)
            continue
        
        if not ret_rgb or not ret_thm:
            # 影片播放結束
            print("\n[影片結束] 自動觸發結算...")
            if is_recording and len(buffer_m1) >= 30:
                # 自動結算（不管有沒有滿 60 秒）
                is_recording = False
                elapsed = time.time() - record_start_time
                print(f"[結算] 共錄製 {elapsed:.1f} 秒，{len(buffer_m1)} 筆鼻部訊號")

                for name in final_temps.keys():
                    final_temps[name] = filter_3sigma(buffer_temps.get(name, []))

                final_bpm, best_snr = compute_bpm_triple_channel(buffer_time, buffer_m1, buffer_m2, buffer_m3, elapsed)

                print(f"-> 呼吸率: {final_bpm:.1f} BPM (Best SNR: {best_snr:.1f}, M1+M2+M3 投票)")
                print("-> 濾波後溫度:", {k: round(v, 2) if v else 'N/A' for k, v in final_temps.items()})

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"-> Pig ID: {current_pig_id}")
                vals = [timestamp, current_pig_id] + [round(final_temps[k], 2) if final_temps[k] else "" for k in ['Left Eye', 'Right Eye', 'Left Nostril', 'Right Nostril', 'Nose Center', 'Left Ear', 'Right Ear']]
                vals += [current_max_temp, current_min_temp, round(final_bpm, 1)]
                with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(vals)
                print("-> 數據已自動存入 CSV！")
                print("\n[完成] 按任意鍵關閉視窗...")
                # 顯示結果畫面，等待使用者按鍵關閉
                result_bar = np.zeros((120, 1600, 3), dtype=np.uint8)
                cv2.putText(result_bar, f"DONE  BPM: {final_bpm:.1f}  Pig: {current_pig_id}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
                cv2.putText(result_bar,
                            f"Eye L/R: {round(final_temps['Left Eye'],1) if final_temps['Left Eye'] else 'N/A'} / "
                            f"{round(final_temps['Right Eye'],1) if final_temps['Right Eye'] else 'N/A'}  "
                            f"Ear L/R: {round(final_temps['Left Ear'],1) if final_temps['Left Ear'] else 'N/A'} / "
                            f"{round(final_temps['Right Ear'],1) if final_temps['Right Ear'] else 'N/A'}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result_bar, "Press any key to close",
                            (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow("Pose-Based Temp Tracker", result_bar)
                while True:
                    if cv2.waitKey(100) != -1:
                        break
            else:
                print("[提示] 未在錄製狀態，數據未儲存。請下次執行時按 'r' 鍵開始錄製。")
                print("\n[完成] 按任意鍵關閉視窗...")
                while True:
                    if cv2.waitKey(100) != -1:
                        break
            break
            
        rgb_h, rgb_w = frame_rgb.shape[:2]
        thm_h, thm_w = frame_thm.shape[:2]

        # ---------------------------------------------------------
        # Step A: 執行 OCR 更新溫標 (每秒執行一次)
        # ---------------------------------------------------------
        current_time = time.time()
        if current_time - last_ocr_time > 1.0:
            roi_top_right = frame_thm[0:int(thm_h*0.25), int(thm_w*0.8):thm_w]
            roi_bot_right = frame_thm[int(thm_h*0.75):thm_h, int(thm_w*0.8):thm_w]
            
            top_results = ocr_reader.readtext(roi_top_right, detail=0)
            bot_results = ocr_reader.readtext(roi_bot_right, detail=0)
            
            for t in top_results:
                val = extract_temperature(t)
                if val is not None and 10 < val < 100:
                    current_max_temp = val
                    break
                    
            for t in bot_results:
                val = extract_temperature(t)
                if val is not None and -20 < val < 50:
                    current_min_temp = val
                    break
                    
            last_ocr_time = current_time

        # ---------------------------------------------------------
        # Step B1: 豬隻身分辨識 (每 15 幀執行一次)
        # ---------------------------------------------------------
        current_frame += 1
        if id_model is not None and current_frame % ID_CHECK_INTERVAL == 0:
            id_res = id_detect_model.predict(frame_rgb, imgsz=IMAGE_SIZE, conf=0.5, verbose=False, device=device)[0]
            face_boxes = [box for box in id_res.boxes if int(box.cls[0].cpu().numpy()) == 1]
            if len(face_boxes) > 0:
                best_box = sorted(face_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0].xyxy[0].cpu().numpy()
                x1f, y1f, x2f, y2f = map(int, best_box)
                pad_x = int((x2f - x1f) * 0.1); pad_y = int((y2f - y1f) * 0.1)
                x1f, y1f = max(0, x1f-pad_x), max(0, y1f-pad_y)
                x2f, y2f = min(rgb_w, x2f+pad_x), min(rgb_h, y2f+pad_y)
                face_crop = frame_rgb[y1f:y2f, x1f:x2f]
                if face_crop.size > 0:
                    face_rgb_t = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    t_face = torch.from_numpy(face_rgb_t).permute(2,0,1).to(device, non_blocking=True)
                    t_face = t_face.half() / 255.0 if device.type == 'cuda' else t_face.float() / 255.0
                    t_face = t_face.unsqueeze(0)
                    t_face = F.interpolate(t_face, size=(224, 224), mode='bilinear', align_corners=False)
                    img_id = (t_face - mean_tensor) / std_tensor
                    with torch.inference_mode():
                        emb = id_model.extract_features(img_id)
                        sim = F.cosine_similarity(emb, id_features_torch)
                        max_sim = torch.max(sim).item()
                    if max_sim >= SIM_THRESHOLD:
                        best_idx = torch.argmax(sim).item()
                        current_pig_id = f"{get_pig_id(str(id_labels[best_idx]))} ({max_sim:.2f})"
                    else:
                        current_pig_id = f"Unknown ({max_sim:.2f})"

        # ---------------------------------------------------------
        # Step B2: YOLO Pose 偵測 & Swin P2P 空間映射
        # ---------------------------------------------------------
        yolo_res = detect_model.predict(frame_rgb, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
        
        detect_map = {}
        kpts = []
        if yolo_res.keypoints is not None and len(yolo_res.keypoints) > 0:
            kpts = yolo_res.keypoints.xy[0].cpu().numpy()
            if len(kpts) >= 7:
                # 為了讓 P2P 模型可以順利算出變形矩陣 M，我們建構一個假的外框給它
                # P2P 模型需要：0=左眼, 1=右眼, 2=鼻子
                dummy_box = np.array([0, 0, 0, 0], dtype=np.int32)
                detect_map[0] = (kpts[0], dummy_box) # 左眼窩
                detect_map[1] = (kpts[1], dummy_box) # 右眼窩
                detect_map[2] = (kpts[4], dummy_box) # 鼻子中間
        
        # 將 RGB 偵測到的座標，算出一個變形矩陣 M 來映射到 THM
        M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, p2p_model, device)
        
        gray_thm = cv2.cvtColor(frame_thm, cv2.COLOR_BGR2GRAY)
        
        temps_data = {'Left Eye': None, 'Right Eye': None, 'Left Nostril': None, 'Right Nostril': None, 'Nose Center': None, 'Left Ear': None, 'Right Ear': None}

        # ---------------------------------------------------------
        # Step C: 擷取映射過去的關鍵點 ROI (40x40 正方形平均)，並換算溫度
        # ---------------------------------------------------------
        if M is not None and len(kpts) >= 7:
            # 定義所有要抓取的部位：(Keypoint Index, 名稱, 畫框顏色 BGR)
            parts = [
                (0, 'Left Eye', (255, 0, 0)),       # 藍色
                (1, 'Right Eye', (255, 0, 0)),
                (2, 'Left Nostril', (0, 255, 0)),   # 綠色
                (3, 'Right Nostril', (0, 255, 0)),
                (4, 'Nose Center', (0, 255, 0)),
                (5, 'Left Ear', (0, 0, 255)),       # 紅色
                (6, 'Right Ear', (0, 0, 255))
            ]
            
            # 用於收集三個鼻部關鍵點映射後的座標，組成完整鼻部 ROI
            nose_mapped_pts = []

            for idx, name, color in parts:
                pt = kpts[idx]
                if pt[0] > 0 and pt[1] > 0:
                    # 在可見光畫面畫出十字準心
                    cv2.drawMarker(frame_rgb, (int(pt[0]), int(pt[1])), color, cv2.MARKER_CROSS, 10, 2)
                    cv2.putText(frame_rgb, name, (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 取得映射到熱影像上的座標
                    pt_mapped = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]]), M)[0][0]
                    cx, cy = int(pt_mapped[0]), int(pt_mapped[1])

                    # 收集鼻部三點 (idx 2,3,4) 的映射座標
                    if idx in [2, 3, 4]:
                        nose_mapped_pts.append((cx, cy))

                    # 取長寬各 40 像素的正方形 (半徑 = 20)
                    radius = 20
                    cx1, cy1 = max(0, cx - radius), max(0, cy - radius)
                    cx2, cy2 = min(thm_w, cx + radius), min(thm_h, cy + radius)

                    roi_crop = gray_thm[cy1:cy2, cx1:cx2]
                    if roi_crop.size > 0:
                        # 取整個正方形區域的平均值
                        pixel_intensity = roi_crop.mean()

                        # 帶入 OCR 動態溫標公式
                        temp = current_min_temp + (pixel_intensity / 255.0) * (current_max_temp - current_min_temp)
                        temps_data[name] = temp

                        if is_recording:
                            buffer_temps[name].append(temp)

                        # 在熱影像上畫出 40x40 映射框與溫度
                        cv2.rectangle(frame_thm, (cx1, cy1), (cx2, cy2), color, 2)
                        cv2.putText(frame_thm, f"{temp:.1f}C", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # -------------------------------------------------------
            # 建立「三點合圍鼻部 ROI」作為呼吸率訊號 (品質優於單點)
            # 取左鼻孔、右鼻孔、鼻子中間三點的 bounding box + 20px 緩衝
            # -------------------------------------------------------
            if len(nose_mapped_pts) >= 2:
                xs = [p[0] for p in nose_mapped_pts]
                ys = [p[1] for p in nose_mapped_pts]
                pad = 20
                nose_height = max(ys) - min(ys)  # 鼻子中間 → 鼻孔的高度
                nx1 = max(0, min(xs) - pad)
                ny1 = max(0, min(ys) - pad)       # 上緣：鼻子中間往上 pad
                nx2 = min(thm_w, max(xs) + pad)
                # 下緣：鼻孔位置再往下延伸相同的鼻子高度 (框住呼吸氣流出口)
                ny2 = min(thm_h, max(ys) + nose_height + pad)
                nose_roi = gray_thm[ny1:ny2, nx1:nx2]
                if nose_roi.size > 0:
                    # 在熱影像上畫出整個鼻部 ROI 框 (黃色粗框)
                    cv2.rectangle(frame_thm, (nx1, ny1), (nx2, ny2), (0, 255, 255), 3)
                    cv2.putText(frame_thm, "Nose ROI (BPM)", (nx1, ny1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    if is_recording:
                        m1 = get_m1(nose_roi)
                        m2 = get_m2(nose_roi)
                        m3 = get_m3(nose_roi)
                        buffer_m1.append(m1)
                        buffer_m2.append(m2)
                        buffer_m3.append(m3)
                        buffer_time.append(time.time() - record_start_time)
                        
        if is_recording:
            # 60 秒計時器：只在 RTSP 即時攝影機模式下使用
            # 影片模式下直接等 EOF 自動結算，避免處理速度慢造成提早觸發
            elapsed = time.time() - record_start_time
            if elapsed >= 60.0 and not is_video_file:
                is_recording = False
                print("\n[錄製完成] 正在計算 3-Sigma 溫度與 BPM...")
                
                # 計算 3-Sigma 濾波後的各部位溫度
                for name in final_temps.keys():
                    filtered_temp = filter_3sigma(buffer_temps.get(name, []))
                    final_temps[name] = filtered_temp
                
                # 計算 BPM
                final_bpm, best_snr = compute_bpm_triple_channel(buffer_time, buffer_m1, buffer_m2, buffer_m3, elapsed)
                
                print(f"-> 呼吸率: {final_bpm:.1f} BPM (Best SNR: {best_snr:.1f}, M1+M2+M3 投票)")
                print("-> 濾波後溫度:", {k: round(v, 2) if v else 'N/A' for k, v in final_temps.items()})
                
                # 自動存檔
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"-> Pig ID: {current_pig_id}")
                vals = [timestamp, current_pig_id] + [round(final_temps[k], 2) if final_temps[k] else "" for k in ['Left Eye', 'Right Eye', 'Left Nostril', 'Right Nostril', 'Nose Center', 'Left Ear', 'Right Ear']]
                vals += [current_max_temp, current_min_temp, round(final_bpm, 1)]
                with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(vals)
                print("-> 數據已自動存入 CSV！")

        # ---------------------------------------------------------
        # Step D: 繪製 UI 面板與互動按鍵
        # ---------------------------------------------------------
        # 顯示 OCR 資訊
        cv2.putText(frame_thm, f"OCR Max: {current_max_temp}", (thm_w - 200, 40), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_thm, f"OCR Min: {current_min_temp}", (thm_w - 200, thm_h - 40), 0, 0.7, (255, 0, 0), 2)
        
        # 為了能在同一個視窗並排顯示，將影像統一縮放為 800x600
        vis_rgb = cv2.resize(frame_rgb, (800, 600))
        vis_thm = cv2.resize(frame_thm, (800, 600))
        combined_view = np.hstack((vis_rgb, vis_thm))
        
        # 添加下方狀態列
        status_bar = np.zeros((80, 1600, 3), dtype=np.uint8)
        
        # 準備顯示用的字串
        def get_t(name):
            val = temps_data[name] if not is_recording else final_temps.get(name, temps_data[name])
            return f"{val:.1f}C" if val else "N/A"
            
        if is_recording:
            elapsed = time.time() - record_start_time
            cv2.putText(status_bar, f"RECORDING: {elapsed:.1f} / 60.0s", (1250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(status_bar, f"Last BPM: {final_bpm:.1f}", (1250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        cv2.putText(status_bar, f"Pig: {current_pig_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2)
        cv2.putText(status_bar, f"Eye: {get_t('Left Eye')} / {get_t('Right Eye')} | Ear: {get_t('Left Ear')} / {get_t('Right Ear')} | Nose: {get_t('Nose Center')}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(status_bar, f"Nostril: {get_t('Left Nostril')} / {get_t('Right Nostril')}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.putText(status_bar, "[R]: Record 60s | [S]: Save CSV | [Q]: Quit", (1100, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        final_ui = np.vstack((combined_view, status_bar))
        
        cv2.imshow("Pose-Based Temp Tracker", final_ui)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not is_recording:
            is_recording = True
            record_start_time = time.time()
            buffer_temps = {k: [] for k in temps_data.keys()}
            buffer_m1, buffer_m2, buffer_m3 = [], [], []
            buffer_time = []
            print("\n[開始錄製] 60秒鐘數據收集與分析啟動（M1+M2+M3 三通道）...")
        elif key == ord('s') and not is_recording:
            # 按下 's' 將數據存入 CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vals = [
                timestamp,
                round(temps_data['Left Eye'], 2) if temps_data['Left Eye'] else "",
                round(temps_data['Right Eye'], 2) if temps_data['Right Eye'] else "",
                round(temps_data['Left Nostril'], 2) if temps_data['Left Nostril'] else "",
                round(temps_data['Right Nostril'], 2) if temps_data['Right Nostril'] else "",
                round(temps_data['Nose Center'], 2) if temps_data['Nose Center'] else "",
                round(temps_data['Left Ear'], 2) if temps_data['Left Ear'] else "",
                round(temps_data['Right Ear'], 2) if temps_data['Right Ear'] else "",
                current_max_temp,
                current_min_temp,
                "" # 手動存檔無 BPM
            ]
            with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(vals)
            print(f"[{timestamp}] 手動儲存成功！")

    cap.release()
    cv2.waitKey(1)          # 清空 GUI 事件佇列，Windows 必須
    cv2.destroyAllWindows()
    cv2.waitKey(1)          # 再次清空，確保視窗被乷除

if __name__ == "__main__":
    main()
