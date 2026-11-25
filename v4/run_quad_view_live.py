# -*- coding: utf-8 -*-
"""
run_quad_view_live.py
================================================================================
🚀 V4 雙光現場即時智慧監控系統 (Live Real-Time Quad-View AI Monitor)
================================================================================
全面整合 Xavier 雙路 RTSP 串流與即時生理辨識：
1. 【零延遲雙路 RTSP 取流 (ZeroLatencyRTSP)】:
   - 預設對接 Xavier 雙光 RTSP:
     * RGB: rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0 (失敗時自動降級 subtype=1)
     * THM: rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0 (失敗時自動降級 subtype=1)
   - TCP 傳輸 + nobuffer + low_delay + 斷線自動重連，延遲壓至 50~100ms
2. 【四分割智慧看板即時顯示 (1920x1080)】:
   - 左上: RGB 五官 (0817) 與 耳根物件框 (best_v12) / 姿態骨骼 (pose_best) + 豬隻身分 ID
   - 左下: 熱成像淚腺熱點自動吸附 + 雙耳根區域溫 + 鼻鏡即時溫
   - 右上: 雙光空間對齊融合 (4x2 動態視差形變矩陣)
   - 右下: 即時生理體溫看板 + 滾動呼吸率示波器 (0.2~2.0Hz 帶通濾波)
3. 【Xavier 雲台/追蹤座標即時 UDP 回傳】:
   - 偵測到豬臉目標時，自動將中心點 (cx, cy) 透過 UDP 回傳至 192.168.50.184:5007
4. 【現場即時鍵盤交互】:
   - [Q / ESC]: 安全退出並關閉串流
   - [S]: 即時截圖儲存 (Snapshot)
   - [SPACE]: 暫停 / 繼續
   - [R]: 重設呼吸波形基準線
================================================================================
"""

import os
# ==================== 【零緩衝與低延遲 FFmpeg 參數】 ====================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

import sys
import time
import json
import csv
import socket
import argparse
import threading
import queue
import cv2
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from gpu_optimization import setup_laptop_gpu_acceleration, load_optimized_yolo
from pig_identity import PigIdentifier, DEFAULT_ID_MODEL_PATH

# 預設 Xavier 網路設定與 RTSP 串流網址 (與 windows_gpu_tracker.py / ubantu_to _windows.py 一致)
DEFAULT_RGB_URL   = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
DEFAULT_THM_URL   = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

def detect_xavier_ip(default="192.168.50.236"):
    """自動偵測 Xavier Wi-Fi IP：連上小車熱點 (10.42.0.x) 自動回傳 10.42.0.1，否則從路由表對接"""
    try:
        import subprocess
        out_gw = subprocess.check_output("route print 0.0.0.0", shell=True).decode("cp950", errors="ignore")
        for line in out_gw.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                gw = parts[2]
                if gw.startswith("10.42.0."):
                    return "10.42.0.1"
        out = subprocess.check_output("route print 192.168.1.*", shell=True).decode("cp950", errors="ignore")
        for line in out.splitlines():
            if "192.168.1.108" in line and "255.255.255.255" in line:
                parts = line.split()
                if len(parts) >= 3 and (parts[2].startswith("192.168.50.") or parts[2].startswith("10.42.0.")):
                    return parts[2]
    except Exception:
        pass
    return default

DEFAULT_XAVIER_IP = detect_xavier_ip()
DEFAULT_UDP_PORT  = 5007

# ==================== 【自動網卡雙工智能路由橋接器】 ====================
_LOCAL_BRIDGE_SERVER = None

def ensure_camera_bridge(target_ip="192.168.1.108", target_port=554, bind_ip="192.168.1.200", local_port=8554):
    """
    自動檢測攝影機連線路徑：
    若預設路由可直通，直接使用原網址；
    若因 Windows 殘留路由導致預設走 Wi-Fi，則自動啟動本機極速 TCP 橋接器 (綁定 192.168.1.200 有線網卡)，
    將串流透明映射至 127.0.0.1:8554，達成 100% 免手動設定、開機即連！
    """
    global _LOCAL_BRIDGE_SERVER
    # 1. 測試預設路由是否已通
    s = socket.socket()
    s.settimeout(1.2)
    r = s.connect_ex((target_ip, target_port))
    s.close()
    if r == 0:
        return f"{target_ip}:{target_port}"
        
    # 2. 測試透過有線網卡 192.168.1.200 是否通
    s_eth = socket.socket()
    try:
        s_eth.bind((bind_ip, 0))
        s_eth.settimeout(0.6)
        r_eth = s_eth.connect_ex((target_ip, target_port))
    except Exception:
        r_eth = -1
    finally:
        s_eth.close()
        
    if r_eth == 0:
        if _LOCAL_BRIDGE_SERVER is None:
            def _forward(s1, s2):
                try:
                    while True:
                        buf = s1.recv(65536)
                        if not buf: break
                        s2.sendall(buf)
                except Exception: pass
                finally:
                    try: s1.close()
                    except Exception: pass
                    try: s2.close()
                    except Exception: pass

            def _bridge_worker():
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(("127.0.0.1", local_port))
                srv.listen(10)
                while True:
                    client, _ = srv.accept()
                    cam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        cam.bind((bind_ip, 0))
                        cam.connect((target_ip, target_port))
                        threading.Thread(target=_forward, args=(client, cam), daemon=True).start()
                        threading.Thread(target=_forward, args=(cam, client), daemon=True).start()
                    except Exception:
                        client.close()

            t = threading.Thread(target=_bridge_worker, daemon=True)
            t.start()
            _LOCAL_BRIDGE_SERVER = t
            time.sleep(0.3)
            print(f"[*] [智能路由] 偵測到有線網卡直連，已啟動零延遲本機橋接器 (127.0.0.1:{local_port} -> {target_ip}:{target_port})")
            
        return f"127.0.0.1:{local_port}"
        
    return f"{target_ip}:{target_port}"


# 預設模型路徑
PRIMARY_MODEL_PATH   = os.path.join(SCRIPT_DIR, "parameters", "nose_eyes_face_0817.pt")
POSE_MODEL_PATH      = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
YOLO_ROOT_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "best_v12_eye_nose_root.pt")
PIGNOSE_MODEL_PATH   = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
MAPPING_NPZ_PATH     = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")
ID_MODEL_PATH        = DEFAULT_ID_MODEL_PATH


class ZeroLatencyRTSP:
    """
    零延遲串流取流器 (支援 RTSP、USB 攝影機、本地影片檔):
    背景執行緒持續擷取最新影格並清空 Buffer，保證推論永遠取得最新鮮畫面；
    遇網路斷線自動發起重連，確保 Wi-Fi 環境極致穩定。
    """
    def __init__(self, src, name="Camera", target_size=None):
        self.src = int(src) if str(src).isdigit() else str(src)
        self.name = name
        self.target_size = target_size
        self.is_local_file = isinstance(self.src, str) and not self.src.startswith("rtsp://") and os.path.exists(self.src)
        self.cap = None
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.is_connected = False
        self.tcp_reachable = False
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if isinstance(self.src, str) and self.src.startswith("rtsp://"):
                    # 快速 TCP 探針 (1.0 秒逾時)，避免在鏡頭離線時 FFmpeg 阻塞 30 秒卡死程序
                    try:
                        parts = self.src.split("://")[-1].split("/")[0]
                        if "@" in parts:
                            parts = parts.split("@")[-1]
                        probe_host = parts.split(":")[0]
                        probe_port = int(parts.split(":")[1]) if ":" in parts else 554
                        _probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        _probe_sock.settimeout(1.5)
                        _probe_res = _probe_sock.connect_ex((probe_host, probe_port))
                        _probe_sock.close()
                        self.tcp_reachable = (_probe_res == 0)
                        if _probe_res != 0:
                            with self.lock:
                                self.is_connected = False
                            time.sleep(0.5)
                            continue
                    except Exception:
                        pass

                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                else:
                    self.cap = cv2.VideoCapture(self.src)
                
                if self.cap is not None and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.is_connected = True
                    print(f"[*] [{self.name}] 成功連接串流: {self.src}")
                else:
                    self.is_connected = False
                    time.sleep(0.5)
                    continue

            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.target_size and (frame.shape[1], frame.shape[0]) != self.target_size:
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.latest_frame = frame
                if self.is_local_file:
                    time.sleep(0.032)
            else:
                if self.is_local_file and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self.lock:
                    self.is_connected = False
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
                time.sleep(0.3)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def is_opened(self):
        with self.lock:
            return self.is_connected and (self.latest_frame is not None)

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class AsyncTriVideoRecorder:
    """
    非同步三路同步錄影器 (Quad View + 原始可見光 RGB + 原始熱成像 THM)：
    - 背景執行緒寫入磁碟，確保不拖慢即時 AI 推論 FPS
    - 以恆定牆上時間節奏 (wall-clock pacing) 寫入三路 MP4，保證回放為 1.0x 原速且三路影片幀數完全對齊
    - 支援分段章節索引 (pig_segments.csv) 與安全 flush 關檔
    """
    def __init__(self, video_root_dir, session_tag=None, fps=25.0, queue_size=60):
        self.fps = float(fps)
        base_tag = session_tag or time.strftime("rec_%Y%m%d_%H%M%S")
        candidate_tag = base_tag
        candidate_dir = os.path.join(video_root_dir, candidate_tag)
        suffix = 2
        while os.path.exists(candidate_dir):
            candidate_tag = f"{base_tag}_{suffix}"
            candidate_dir = os.path.join(video_root_dir, candidate_tag)
            suffix += 1
        self.session_tag = candidate_tag
        self.session_dir = candidate_dir
        os.makedirs(self.session_dir, exist_ok=True)

        self.quad_path = os.path.join(self.session_dir, f"{self.session_tag}_quad.mp4")
        self.rgb_path  = os.path.join(self.session_dir, f"{self.session_tag}_rgb.mp4")
        self.thm_path  = os.path.join(self.session_dir, f"{self.session_tag}_thm.mp4")
        self.segments_csv = os.path.join(self.session_dir, f"{self.session_tag}_pig_segments.csv")

        self._q = queue.Queue(maxsize=queue_size)
        self._running = True
        self._w_quad = None
        self._w_rgb = None
        self._w_thm = None
        self._quad_size = None
        self._rgb_size = None
        self._thm_size = None

        self.written_frames = 0
        self.dropped_frames = 0
        self.start_wall_t = None
        self._acc_time = 0.0
        self._last_enqueue_t = None

        self._thread = threading.Thread(target=self._worker, name="TriVideoWriter", daemon=True)
        self._thread.start()

    def elapsed_sec(self):
        return self.written_frames / max(1.0, self.fps)

    def write(self, quad_frame, rgb_frame, thm_frame):
        if not self._running or quad_frame is None or rgb_frame is None or thm_frame is None:
            return
        now = time.perf_counter()
        if self._last_enqueue_t is None:
            self._last_enqueue_t = now
            self.start_wall_t = time.time()
            repeat = 1
        else:
            dt = now - self._last_enqueue_t
            self._last_enqueue_t = now
            if dt > 1.0:
                # 若剛從暫停恢復，避免插入長時間靜止幀
                dt = 1.0 / self.fps
            self._acc_time += dt
            desired_frames = int(self._acc_time * self.fps) + 1
            repeat = desired_frames - self.written_frames
            if repeat <= 0:
                return
            if repeat > 10:
                repeat = 10

        try:
            self._q.put_nowait((quad_frame.copy(), rgb_frame.copy(), thm_frame.copy(), repeat))
            self.written_frames += repeat
        except queue.Full:
            self.dropped_frames += 1

    def record_pig_segment(self, pig_id, rec_mode, start_str, end_str, start_vid_sec, end_vid_sec,
                           eye_l, eye_r, ear_l, ear_r, bpm):
        try:
            exists = os.path.exists(self.segments_csv)
            with open(self.segments_csv, "a", newline="", encoding="utf-8-sig") as f:
                cw = csv.writer(f)
                if not exists:
                    cw.writerow([
                        "Pig_ID", "Record_Mode", "Start_Time", "End_Time",
                        "Video_Start_Sec", "Video_End_Sec", "Duration_Sec",
                        "Left_Eye_C", "Right_Eye_C", "Left_Ear_C", "Right_Ear_C",
                        "Respiration_BPM", "Quad_Video", "RGB_Video", "THM_Video"
                    ])
                cw.writerow([
                    pig_id, rec_mode, start_str, end_str,
                    f"{start_vid_sec:.2f}", f"{end_vid_sec:.2f}", f"{max(0.0, end_vid_sec - start_vid_sec):.2f}",
                    eye_l, eye_r, ear_l, ear_r, bpm,
                    os.path.basename(self.quad_path),
                    os.path.basename(self.rgb_path),
                    os.path.basename(self.thm_path),
                ])
        except Exception as e:
            print(f"[!] [三路錄影分段索引寫入失敗] {e}")

    def _init_writers(self, quad_f, rgb_f, thm_f):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        qh, qw = quad_f.shape[:2]
        rh, rw = rgb_f.shape[:2]
        th, tw = thm_f.shape[:2]
        self._quad_size = (qw, qh)
        self._rgb_size = (rw, rh)
        self._thm_size = (tw, th)
        self._w_quad = cv2.VideoWriter(self.quad_path, fourcc, self.fps, self._quad_size)
        self._w_rgb  = cv2.VideoWriter(self.rgb_path,  fourcc, self.fps, self._rgb_size)
        self._w_thm  = cv2.VideoWriter(self.thm_path,  fourcc, self.fps, self._thm_size)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            quad_f, rgb_f, thm_f, repeat = item
            try:
                if self._w_quad is None:
                    self._init_writers(quad_f, rgb_f, thm_f)

                if (quad_f.shape[1], quad_f.shape[0]) != self._quad_size:
                    quad_f = cv2.resize(quad_f, self._quad_size)
                if (rgb_f.shape[1], rgb_f.shape[0]) != self._rgb_size:
                    rgb_f = cv2.resize(rgb_f, self._rgb_size)
                if (thm_f.shape[1], thm_f.shape[0]) != self._thm_size:
                    thm_f = cv2.resize(thm_f, self._thm_size)

                for _ in range(repeat):
                    if self._w_quad is not None and self._w_quad.isOpened():
                        self._w_quad.write(quad_f)
                    if self._w_rgb is not None and self._w_rgb.isOpened():
                        self._w_rgb.write(rgb_f)
                    if self._w_thm is not None and self._w_thm.isOpened():
                        self._w_thm.write(thm_f)
            except Exception as e:
                print(f"[!] [三路錄影寫入異常] {e}")
            finally:
                self._q.task_done()

        for w in (self._w_quad, self._w_rgb, self._w_thm):
            if w is not None:
                try:
                    w.release()
                except Exception:
                    pass

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            self._q.put(None, timeout=2.0)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=15.0)


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


def snap_to_tear_duct(frame_thm, raw_box, max_search_radius=20):
    """熱成像眼窩淚腺熱點自動吸附微調 + 框內最高溫 10 個 pixel 平均"""
    h_thm, w_thm = frame_thm.shape[:2]
    rx1, ry1, rx2, ry2 = raw_box
    bw = rx2 - rx1
    bh = ry2 - ry1
    
    sr = min(max_search_radius, int(min(w_thm, h_thm) * 0.10))
    sx1 = int(max(0, rx1 - sr))
    sy1 = int(max(0, ry1 - sr))
    sx2 = int(min(w_thm, rx2 + sr))
    sy2 = int(min(h_thm, ry2 + sr))
    
    if sx2 <= sx1 or sy2 <= sy1:
        temp, pt = extract_temp_top10(frame_thm, raw_box)
        return raw_box, temp, pt
        
    roi = frame_thm[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    peak_temp = 20.0 + (max_val / 255.0) * 25.0
    
    if peak_temp >= 36.5:
        peak_x = sx1 + max_loc[0]
        peak_y = sy1 + max_loc[1]
        
        snapped_box = [
            float(np.clip(peak_x - bw / 2.0, 0, w_thm - 1)),
            float(np.clip(peak_y - bh / 2.0, 0, h_thm - 1)),
            float(np.clip(peak_x + bw / 2.0, 0, w_thm - 1)),
            float(np.clip(peak_y + bh / 2.0, 0, h_thm - 1))
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
    if y1 < 60:
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


def draw_telemetry_dashboard(eye_temps, ear_temps, snout_temp, cur_bpm, peaks_cnt, sig_filtered, f_idx, avg_fps, is_live=True, pig_id="Detecting...", id_conf=0.0, period_avgs=None, is_recording=False, rec_duration_sec=0.0, rec_count=0, auto_mode=True, is_auto_session=False, is_video_rec=False):
    panel = np.zeros((540, 960, 3), dtype=np.uint8)
    panel[:] = (18, 22, 28)
    
    # 頂部儀表板標題
    cv2.rectangle(panel, (0, 0), (960, 42), (28, 36, 46), -1)
    cv2.line(panel, (0, 42), (960, 42), (0, 215, 255), 2)
    
    status_tag = "XAVIER LIVE RTSP" if is_live else "PLAYBACK"
    title_text = f'[BOTTOM-RIGHT] TELEMETRY | {status_tag}'
    cv2.putText(panel, title_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # 錄影/生理數據記錄指示徽章
    if is_recording:
        m, s = divmod(int(rec_duration_sec), 60)
        mode_prefix = "AUTO REC" if is_auto_session else "MANUAL REC"
        rec_tag_text = f"{mode_prefix} [{m:02d}:{s:02d}] ({rec_count} pts)"
        cv2.rectangle(panel, (320, 8), (620, 34), (20, 20, 150), -1)
        cv2.rectangle(panel, (320, 8), (620, 34), (50, 50, 255), 1)
        dot_col = (0, 0, 255) if int(rec_duration_sec * 2) % 2 == 0 else (120, 120, 255)
        cv2.circle(panel, (336, 21), 5, dot_col, -1, cv2.LINE_AA)
        cv2.putText(panel, rec_tag_text, (348, 26), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    elif auto_mode:
        # 自動感應伺服器待命標籤
        cv2.rectangle(panel, (350, 8), (620, 34), (20, 45, 30), -1)
        cv2.rectangle(panel, (350, 8), (620, 34), (40, 200, 100), 1)
        cv2.circle(panel, (366, 21), 4, (40, 255, 120), -1, cv2.LINE_AA)
        cv2.putText(panel, "AUTO SENSING STANDBY", (378, 26), cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 255, 210), 1, cv2.LINE_AA)
    
    # 右上角顯示已辨識身分徽章 (Pig ID Badge)
    if pig_id != "Detecting..." and pig_id != "Unknown":
        badge_col = (50, 255, 120) if not str(pig_id).startswith("Pig_Reg_") else (0, 220, 255)
        bg_col = (25, 60, 38) if not str(pig_id).startswith("Pig_Reg_") else (35, 50, 20)
        cv2.rectangle(panel, (640, 7), (945, 36), bg_col, -1)
        cv2.rectangle(panel, (640, 7), (945, 36), badge_col, 1)
        cv2.putText(panel, f'ID: {pig_id}', (650, 27), cv2.FONT_HERSHEY_DUPLEX, 0.56, badge_col, 2, cv2.LINE_AA)
        tag_sub = "NEW REG" if str(pig_id).startswith("Pig_Reg_") else f"{id_conf*100:.0f}%"
        cv2.putText(panel, tag_sub, (860, 27), cv2.FONT_HERSHEY_DUPLEX, 0.50, (220, 255, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, 'ONLINE HUD', (810, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 200), 2, cv2.LINE_AA)
    
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
                    
    id_tag_str = f" | ID:{pig_id} ({id_conf*100:.0f}%)" if (pig_id != "Detecting...") else ""
    vid_tag = "[C]:STOP VID" if is_video_rec else "[C]:3CamRec"
    auto_tag = "[A]:Auto(ON)" if auto_mode else "[A]:Auto(OFF)"
    rec_tag = "[T]:STOP" if is_recording else "[T]:Vitals"
    stat_str = f"F:{f_idx:05d} | {avg_fps:.1f}FPS{id_tag_str} | {vid_tag} | {auto_tag} | {rec_tag} | [S]:Snap [Q]:Exit"
    cv2.putText(panel, stat_str, (15, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 150, 160), 1, cv2.LINE_AA)
    
    return panel


def start_live_monitor(src_rgb=DEFAULT_RGB_URL, src_thm=DEFAULT_THM_URL, 
                       xavier_ip=DEFAULT_XAVIER_IP, udp_port=DEFAULT_UDP_PORT,
                       save_video=False, out_dir=None, fast_mode=True, ear_mode="pose",
                       max_frames=0, headless=False, auto_record=True,
                       frame_callback=None, command_queue=None, fallback_demo=True,
                       video_dir=None):
    print("=" * 80)
    print("🚀 啟動 V4 現場雙光即時智慧監控系統 (Xavier Live AI Monitor)")
    print(f"[*] 運作模式:   {'現場高速/電池省電 (Fast/Eco) - 30+ FPS' if fast_mode else '全幀推論 (Full Inference)'}")
    print(f"[*] 耳根架構:   {ear_mode.upper()} ({'PPT 指定純 YOLO 雙模型' if ear_mode == 'yolo_bbox' else 'Pose 幾何骨骼姿態'})")
    print(f"[*] RGB 視訊源:  {src_rgb}")
    print(f"[*] THM 視訊源:  {src_thm}")
    print(f"[*] Xavier UDP: {xavier_ip}:{udp_port}")
    print("=" * 80)

    out_dir = out_dir or os.path.join(SCRIPT_DIR, "live_monitor_outputs")
    os.makedirs(out_dir, exist_ok=True)
    video_dir = video_dir or os.path.join(PROJECT_ROOT, "recorded_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # 載入硬體加速與模型
    device = setup_laptop_gpu_acceleration()
    primary_model = load_optimized_yolo(PRIMARY_MODEL_PATH, imgsz=640, device=device, warmup=False)
    pignose_model = load_optimized_yolo(PIGNOSE_MODEL_PATH, imgsz=512, device=device, warmup=False)
    
    if ear_mode == 'yolo_bbox':
        yolo_root_model = load_optimized_yolo(YOLO_ROOT_MODEL_PATH, imgsz=640, device=device, warmup=False)
        pose_model = None
    else:
        pose_model = load_optimized_yolo(POSE_MODEL_PATH, imgsz=448, device=device, warmup=False)
        yolo_root_model = None
        
    pig_identifier = PigIdentifier(model_path=ID_MODEL_PATH, device=device, half=True)
    coef, RW, RH, TW, TH = load_projector(MAPPING_NPZ_PATH)
    
    # 建立 Xavier 座標回傳 UDP Socket
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 自動檢測並橋接攝影機網路路徑 (防止 Wi-Fi/有線網卡路由混淆)
    if "192.168.1.108:554" in src_rgb or "192.168.1.108:554" in src_thm:
        bridge_target = ensure_camera_bridge()
        if "127.0.0.1" in bridge_target:
            src_rgb = src_rgb.replace("192.168.1.108:554", bridge_target)
            src_thm = src_thm.replace("192.168.1.108:554", bridge_target)

    # 建立多線程零延遲非同步取流 (ZeroLatencyRTSP)
    print("[*] 正在連線雙路 RTSP 串流...")
    cam_rgb = ZeroLatencyRTSP(src_rgb, name="RGB_Camera")
    cam_thm = ZeroLatencyRTSP(src_thm, name="THM_Camera")
    
    # 等待攝影機串流連線就緒：
    # 1. 若 TCP 554 可連通，優先等候雙路高清主碼流 (subtype=0) 首幀解碼
    # 2. 若任一路主碼流 (subtype=0) 逾 6 秒仍未就緒，自動將該路降級為副碼流 (subtype=1) 重試確保連線
    base_timeout = 3.5 if fallback_demo else 12.0
    rgb_downgraded = False
    thm_downgraded = False
    t_wait_start = time.time()
    while True:
        if cam_rgb.is_opened() and cam_thm.is_opened():
            break
        elapsed_w = time.time() - t_wait_start
        tcp_up = getattr(cam_rgb, "tcp_reachable", False) or getattr(cam_thm, "tcp_reachable", False)

        if tcp_up and not rgb_downgraded and elapsed_w >= 6.0 and not cam_rgb.is_opened() and "subtype=0" in src_rgb:
            rgb_downgraded = True
            fallback_rgb = src_rgb.replace("subtype=0", "subtype=1")
            print("[!] 可見光高清主碼流 (subtype=0) 回應逾時，自動切換副碼流 (subtype=1) 重試...")
            cam_rgb.release()
            src_rgb = fallback_rgb
            cam_rgb = ZeroLatencyRTSP(src_rgb, name="RGB_Camera_Sub")

        if tcp_up and not thm_downgraded and elapsed_w >= 6.0 and not cam_thm.is_opened() and "subtype=0" in src_thm:
            thm_downgraded = True
            fallback_thm = src_thm.replace("subtype=0", "subtype=1")
            print("[!] 熱成像高清主碼流 (subtype=0) 回應逾時，自動切換副碼流 (subtype=1) 重試...")
            cam_thm.release()
            src_thm = fallback_thm
            cam_thm = ZeroLatencyRTSP(src_thm, name="THM_Camera_Sub")

        cur_limit = 14.0 if tcp_up else base_timeout
        if elapsed_w >= cur_limit:
            break
        time.sleep(0.2)
        
    if not cam_rgb.is_opened() or not cam_thm.is_opened():
        if fallback_demo:
            print(f"[!] RTSP 連線逾時 -> RGB就緒={cam_rgb.is_opened()}(TCP={cam_rgb.tcp_reachable}), THM就緒={cam_thm.is_opened()}(TCP={cam_thm.tcp_reachable}) | Xavier={xavier_ip}")
            print("[!] 尚未偵測到現場 RTSP 雙光攝影機，自動切換為內建本地雙光示範影片模式 (Demo Fallback)！")
            cam_rgb.release()
            cam_thm.release()
            demo_rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
            demo_thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")
            if not os.path.exists(demo_rgb):
                demo_rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
                demo_thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")
            cam_rgb = ZeroLatencyRTSP(demo_rgb, name="Demo_RGB")
            cam_thm = ZeroLatencyRTSP(demo_thm, name="Demo_THM")
            t_wait2 = time.time()
            while time.time() - t_wait2 < 3.0:
                if cam_rgb.is_opened() and cam_thm.is_opened():
                    break
                time.sleep(0.1)
        else:
            print("❌ 錯誤: 無法同時開啟雙光攝影機串流，請檢查 Xavier Wi-Fi 與攝影機 IP！")
            cam_rgb.release()
            cam_thm.release()
            sock_tx.close()
            return

    if frame_callback is not None:
        headless = True

    CANVAS_W, CANVAS_H = 1920, 1080
    PANEL_W, PANEL_H = 960, 540
    
    tri_recorder = None
    last_rec_session_dir = None
    last_rec_report_saved = False

    def _reset_measurement_state(reason="", keep_pig_id=False, reset_timer=True):
        nonlocal session_start_t, cur_bpm, peaks_cnt
        nonlocal last_pose_ears, last_rgb_face, last_rgb_nose, last_rgb_eyes
        nonlocal last_snout_temp, last_thm_nose_box, current_pig_id, current_id_conf
        nonlocal auto_rec_target_pig, auto_rec_detect_streak, auto_rec_miss_streak, auto_rec_switch_streak
        t_sig.clear()
        sig_filtered_history.clear()
        hist_eye_l.clear()
        hist_eye_r.clear()
        hist_ear_l.clear()
        hist_ear_r.clear()
        cur_bpm = 0.0
        peaks_cnt = 0
        last_snout_temp = 0.0
        last_thm_nose_box = None
        last_pose_ears = []
        last_rgb_face = None
        last_rgb_nose = None
        last_rgb_eyes = []
        if reset_timer:
            session_start_t = time.time()
        if not keep_pig_id:
            pig_identifier.reset()
            current_pig_id = "Detecting..."
            current_id_conf = 0.0
            auto_rec_target_pig = None
            auto_rec_detect_streak = 0
            auto_rec_miss_streak = 0
            auto_rec_switch_streak = 0
        if reason:
            print(f"[*] [{reason}] 已清空呼吸率、波形、體溫累積與偵測快取，從本時刻起重新計算！")

    def _start_tri_recording():
        nonlocal tri_recorder, last_rec_session_dir, last_rec_report_saved
        if tri_recorder is not None:
            return
        tri_recorder = AsyncTriVideoRecorder(video_dir, fps=25.0)
        last_rec_session_dir = tri_recorder.session_dir
        last_rec_report_saved = False
        print("\n" + "=" * 80)
        print(f"🎬 [三路同步錄影：已啟動] 目錄: {tri_recorder.session_dir}")
        print(f"   • Quad View (四分割) : {os.path.basename(tri_recorder.quad_path)}")
        print(f"   • 可見光 RGB (原片)  : {os.path.basename(tri_recorder.rgb_path)}")
        print(f"   • 熱成像 THM (原片)  : {os.path.basename(tri_recorder.thm_path)}")
        print("=" * 80 + "\n")

    def _stop_tri_recording():
        nonlocal tri_recorder, last_rec_report_saved
        if tri_recorder is None:
            return
        if is_recording_vitals:
            _stop_and_save_vitals(is_auto=is_auto_session)
        rec = tri_recorder
        seg_dur = rec.elapsed_sec() if rec.written_frames > 0 else max(1.0, time.time() - session_start_t)
        try:
            from run_quad_view_unified import save_vitals_report
            seg_tag = f"live_session_{time.strftime('%Y%m%d_%H%M%S')}"
            save_vitals_report(
                out_dir=rec.session_dir,
                video_tag=seg_tag,
                pig_id=current_pig_id,
                id_conf=current_id_conf,
                final_bpm=cur_bpm,
                breath_cnt=peaks_cnt,
                duration_sec=max(1.0, seg_dur),
                hist_eye_l=hist_eye_l,
                hist_eye_r=hist_eye_r,
                hist_ear_l=hist_ear_l,
                hist_ear_r=hist_ear_r
            )
            last_rec_report_saved = True
        except Exception as e:
            print(f"[!] [錄影生理摘要寫入失敗] {e}")
        tri_recorder = None
        print("[*] 正在刷新緩衝區並安全儲存三路 MP4 影片，請稍候...")
        rec.stop()
        dur = rec.elapsed_sec()
        print("\n" + "=" * 80)
        print(f"🎬 [三路同步錄影：已儲存完成] 實錄時長: {dur:.1f} 秒 | 寫入幀數: {rec.written_frames} (遺失: {rec.dropped_frames})")
        for lbl, fp in [("Quad View", rec.quad_path), ("可見光 RGB", rec.rgb_path), ("熱成像 THM", rec.thm_path)]:
            sz_mb = os.path.getsize(fp) / (1024 * 1024) if os.path.exists(fp) else 0.0
            print(f"   • {lbl:<10}: {fp} ({sz_mb:.2f} MB)")
        print("=" * 80 + "\n")

    t_sig = []
    sig_filtered_history = []
    fps_history = []
    f_idx = 0
    paused = False
    session_start_t = time.time()
    cur_bpm = 0.0
    peaks_cnt = 0

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

    # 生理數據即時連續記錄狀態
    is_recording_vitals = False
    is_auto_session = False
    vitals_rec_file = None
    vitals_rec_writer = None
    vitals_rec_start_t = 0.0
    vitals_rec_count = 0
    vitals_rec_csv_path = None
    vitals_rec_tag = ""
    vitals_rec_dir = None
    vitals_vid_start_sec = None
    vitals_buf_eye_l = []
    vitals_buf_eye_r = []
    vitals_buf_ear_l = []
    vitals_buf_ear_r = []
    vitals_buf_bpm = []

    # 自動巡檢感應錄製狀態機 (Auto-Sensing Server Engine)
    auto_record_mode = auto_record
    auto_rec_target_pig = None
    auto_rec_detect_streak = 0
    auto_rec_miss_streak = 0
    auto_rec_switch_streak = 0

    if save_video:
        _start_tri_recording()

    window_name = "Pig Health Live AI Monitor [V4 Quad-View from Xavier]"
    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception:
            headless = True

    color_map = {
        'face': (100, 255, 100),
        'nose': (50, 200, 255),
        'eye': (0, 255, 200),
        'ear_root_l': (255, 200, 0),
        'ear_root_r': (255, 0, 200),
    }

    # 未知新豬隻自動建檔通知狀態
    last_registered_alert_id = None
    last_registered_alert_time = 0.0

    def _start_vitals_recording(pig_id, is_auto=False):
        nonlocal is_recording_vitals, is_auto_session, auto_rec_target_pig
        nonlocal vitals_rec_file, vitals_rec_writer, vitals_rec_start_t, vitals_rec_count
        nonlocal vitals_rec_csv_path, vitals_rec_tag, vitals_rec_dir, vitals_vid_start_sec
        if is_recording_vitals:
            return
            
        vitals_dir = (
            os.path.join(tri_recorder.session_dir, "vitals")
            if tri_recorder is not None
            else os.path.join(out_dir, "vitals_records")
        )
        vitals_rec_dir = vitals_dir
        os.makedirs(vitals_dir, exist_ok=True)
        vitals_rec_start_t = time.time()
        vitals_vid_start_sec = tri_recorder.elapsed_sec() if tri_recorder is not None else None
        vitals_rec_count = 0
        is_auto_session = is_auto
        auto_rec_target_pig = pig_id
        prefix = "auto_vitals" if is_auto else "manual_vitals"
        vitals_rec_tag = f"{prefix}_{pig_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        vitals_rec_csv_path = os.path.join(vitals_dir, f"{vitals_rec_tag}.csv")
        vitals_rec_file = open(vitals_rec_csv_path, mode="w", newline="", encoding="utf-8-sig")
        vitals_rec_writer = csv.writer(vitals_rec_file)
        vitals_rec_writer.writerow([
            "Timestamp", "Elapsed_Sec", "Frame_Idx", "Pig_ID", "ID_Confidence",
            "Left_Eye_Temp_C", "Right_Eye_Temp_C", "Left_Ear_Temp_C", "Right_Ear_Temp_C",
            "Snout_Temp_C", "Respiration_BPM", "Breath_Peaks", "Record_Mode"
        ])
        vitals_rec_file.flush()
        vitals_buf_eye_l.clear()
        vitals_buf_eye_r.clear()
        vitals_buf_ear_l.clear()
        vitals_buf_ear_r.clear()
        vitals_buf_bpm.clear()
        is_recording_vitals = True
        
        mode_text = "【自動感應啟動】" if is_auto else "【手動啟動】"
        print("\n" + "=" * 80)
        print(f"🤖 {mode_text} 偵測到目標豬隻 {pig_id}，開始量測記錄生理數據！")
        print(f"[*] 輸出檔案: {vitals_rec_csv_path}")
        if is_auto:
            print(f"[*] 模式說明: 伺服器端無人值守巡檢中，當豬隻離場或小車駛離時，系統將自動結算存檔。")
        else:
            print(f"[*] 提示: 再次按下 [T] 或 [V] 鍵即可停止儲存並產生摘要報告。")
        print("=" * 80 + "\n")

    def _stop_and_save_vitals(is_auto=False):
        nonlocal is_recording_vitals, is_auto_session, auto_rec_target_pig
        nonlocal vitals_rec_file, vitals_rec_writer, vitals_rec_start_t, vitals_rec_count
        nonlocal vitals_rec_csv_path, vitals_rec_tag, vitals_rec_dir, vitals_vid_start_sec
        if not is_recording_vitals:
            return
            
        is_recording_vitals = False
        if vitals_rec_file is not None:
            vitals_rec_file.close()
            vitals_rec_file = None
            
        rec_duration = max(0.1, time.time() - vitals_rec_start_t)
        vitals_dir = vitals_rec_dir or os.path.join(out_dir, "vitals_records")
        os.makedirs(vitals_dir, exist_ok=True)
        
        # 若自動模式下記錄時間過短且筆數小於 10 筆 (例如只是鏡頭掠過不到 1 秒)，自動清除無效空檔
        if is_auto and vitals_rec_count < 10:
            try:
                if os.path.exists(vitals_rec_csv_path):
                    os.remove(vitals_rec_csv_path)
            except Exception:
                pass
            print(f"[*] 略過過短暫態資料 (<10筆)，取消存檔。")
            return
            
        avg_el = f"{np.mean(vitals_buf_eye_l):.2f}" if vitals_buf_eye_l else "N/A"
        avg_er = f"{np.mean(vitals_buf_eye_r):.2f}" if vitals_buf_eye_r else "N/A"
        avg_al = f"{np.mean(vitals_buf_ear_l):.2f}" if vitals_buf_ear_l else "N/A"
        avg_ar = f"{np.mean(vitals_buf_ear_r):.2f}" if vitals_buf_ear_r else "N/A"
        valid_bpms = [b for b in vitals_buf_bpm if b > 0]
        avg_bpm = f"{int(np.mean(valid_bpms) + 0.5)}" if valid_bpms else "N/A"
        
        summary_txt = os.path.join(vitals_dir, f"{vitals_rec_tag}_summary.txt")
        with open(summary_txt, "w", encoding="utf-8") as sf:
            sf.write("================================================================================\n")
            sf.write(f"  {'【自動感應巡檢】' if is_auto else '【手動觸發】'}豬隻即時生理數據量測摘要報告\n")
            sf.write("================================================================================\n")
            sf.write(f"量測模式: {'自動感應無人巡檢模式 (Auto-Sensing Server Mode)' if is_auto else '手動記錄模式'}\n")
            sf.write(f"記錄起訖: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(vitals_rec_start_t))} ~ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sf.write(f"記錄時長: {rec_duration:.1f} 秒\n")
            sf.write(f"總有效筆數: {vitals_rec_count} 筆\n")
            sf.write(f"豬隻身分: {auto_rec_target_pig or current_pig_id}\n")
            sf.write(f"左眼窩均溫: {avg_el} °C\n")
            sf.write(f"右眼窩均溫: {avg_er} °C\n")
            sf.write(f"左耳根均溫: {avg_al} °C\n")
            sf.write(f"右耳根均溫: {avg_ar} °C\n")
            sf.write(f"平均呼吸率: {avg_bpm} BPM\n")
            sf.write(f"原始數據 CSV: {vitals_rec_csv_path}\n")
            sf.write("================================================================================\n")
            
        saved_pig = auto_rec_target_pig or current_pig_id
        summary_json = os.path.join(vitals_dir, f"{vitals_rec_tag}_summary.json")
        video_segment_info = None
        if tri_recorder is not None and vitals_vid_start_sec is not None:
            vid_end_sec = tri_recorder.elapsed_sec()
            start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(vitals_rec_start_t))
            end_str = time.strftime('%Y-%m-%d %H:%M:%S')
            tri_recorder.record_pig_segment(
                pig_id=saved_pig,
                rec_mode="AUTO" if is_auto else "MANUAL",
                start_str=start_str,
                end_str=end_str,
                start_vid_sec=vitals_vid_start_sec,
                end_vid_sec=vid_end_sec,
                eye_l=avg_el, eye_r=avg_er, ear_l=avg_al, ear_r=avg_ar, bpm=avg_bpm
            )
            video_segment_info = {
                "start_sec": round(vitals_vid_start_sec, 2),
                "end_sec": round(vid_end_sec, 2),
                "quad_video": tri_recorder.quad_path,
                "rgb_video": tri_recorder.rgb_path,
                "thm_video": tri_recorder.thm_path
            }
            try:
                with open(summary_txt, "a", encoding="utf-8") as sf:
                    sf.write(f"對應錄影區間: {vitals_vid_start_sec:.1f}s ~ {vid_end_sec:.1f}s\n")
                    sf.write(f"Quad View 檔: {tri_recorder.quad_path}\n")
                    sf.write(f"可見光 RGB 檔: {tri_recorder.rgb_path}\n")
                    sf.write(f"熱成像 THM 檔: {tri_recorder.thm_path}\n")
                    sf.write("================================================================================\n")
            except Exception:
                pass
        vitals_vid_start_sec = None

        summary_data = {
            "pig_id": saved_pig,
            "record_mode": "Auto-Sensing Server Mode" if is_auto else "Manual Mode",
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(vitals_rec_start_t)),
            "end_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_sec": round(rec_duration, 1),
            "total_records": vitals_rec_count,
            "averages": {
                "left_eye_temp_c": float(avg_el) if avg_el != "N/A" else None,
                "right_eye_temp_c": float(avg_er) if avg_er != "N/A" else None,
                "left_ear_temp_c": float(avg_al) if avg_al != "N/A" else None,
                "right_ear_temp_c": float(avg_ar) if avg_ar != "N/A" else None,
                "respiration_bpm": int(avg_bpm) if avg_bpm != "N/A" else None
            },
            "video_segment": video_segment_info,
            "csv_path": vitals_rec_csv_path,
            "txt_summary_path": summary_txt
        }
        with open(summary_json, "w", encoding="utf-8") as jf:
            json.dump(summary_data, jf, ensure_ascii=False, indent=4)
            
        auto_rec_target_pig = None
        
        tag_type = "🤖 [自動感應結算]" if is_auto else "■ [手動記錄完成]"
        print("\n" + "=" * 80)
        print(f"{tag_type} 豬隻 {saved_pig} 生理量測已自動結算存檔！")
        print(f"[*] 記錄時長: {rec_duration:.1f} 秒 | 總筆數: {vitals_rec_count} 筆")
        print(f"[*] 眼窩均溫: 左 {avg_el} °C / 右 {avg_er} °C")
        print(f"[*] 耳根均溫: 左 {avg_al} °C / 右 {avg_ar} °C")
        print(f"[*] 平均呼吸率: {avg_bpm} BPM")
        print(f"[*] 詳細 CSV: {vitals_rec_csv_path}")
        print(f"[*] 摘要 TXT: {summary_txt}")
        print(f"[*] 摘要 JSON: {summary_json}")
        print("=" * 80 + "\n")

    print("\n[✓] 系統就緒！已成功接入 Xavier 影像源。可操作快捷鍵：")
    print("    - [C]:       【三路同步錄影開關】同時錄製 Quad View + 可見光 RGB + 熱成像 MP4")
    print("    - [Q / ESC]: 關閉退出系統 (並安全儲存所有錄影檔案)")
    print("    - [S]:       即時保存四分割高畫質快照 (Snapshot)")
    print("    - [A]:       【自動感應開關】辨識到豬自動量測開錄，離場自動結算存檔")
    print("    - [T / V]:   【手動切換開關】手動強制 開始 / 停止 記錄生理數據 CSV")
    print("    - [SPACE]:   畫面暫停 / 繼續")
    print("    - [R]:       重設呼吸累積訊號緩衝區\n")

    canvas = None
    try:
      while True:
        if not paused:
            t0 = time.time()
            ret_r, frame_r = cam_rgb.read()
            ret_t, frame_t = cam_thm.read()
            
            if not ret_r or not ret_t or frame_r is None or frame_t is None:
                time.sleep(0.01)
                continue
                
            f_idx += 1
            cur_w_r, cur_h_r = frame_r.shape[1], frame_r.shape[0]
            cur_w_t, cur_h_t = frame_t.shape[1], frame_t.shape[0]
            if f_idx == 1:
                print(f"[✓] [串流實際解析度] RGB: {cur_w_r}x{cur_h_r} | THM: {cur_w_t}x{cur_h_t}")
            
            # 1. 可見光五官偵測 (0817 模型，fast_mode 下偶數幀推論)
            run_primary = (not fast_mode) or (f_idx % 2 == 0) or (last_rgb_face is None)
            if run_primary:
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
                    
                if rgb_face_box is not None: last_rgb_face = rgb_face_box
                elif last_rgb_face is not None: rgb_face_box = last_rgb_face
                
                if rgb_nose_box is not None: last_rgb_nose = rgb_nose_box
                elif last_rgb_nose is not None: rgb_nose_box = last_rgb_nose
                
                if len(rgb_eye_boxes) > 0: last_rgb_eyes = rgb_eye_boxes
                elif len(last_rgb_eyes) > 0: rgb_eye_boxes = last_rgb_eyes
            else:
                rgb_face_box = last_rgb_face
                rgb_nose_box = last_rgb_nose
                rgb_eye_boxes = last_rgb_eyes
                
            # 1.5 豬隻個體身分辨識 (Pig ID Recognition)
            target_face_id = rgb_face_box or last_rgb_face
            if target_face_id is not None and (f_idx % 3 == 0 or current_pig_id == "Detecting..."):
                fx1, fy1, fx2, fy2 = map(int, target_face_id[0])
                crop_face = frame_r[max(0, fy1):min(cur_h_r, fy2), max(0, fx1):min(cur_w_r, fx2)]
                current_pig_id, current_id_conf = pig_identifier.update(crop_face)
            
            # 1.6 即時目標座標回傳 Xavier (延續 ubantu_to _windows.py 的雲台與小車追蹤回饋)
            has_target = (rgb_face_box is not None)
            if has_target:
                fx1, fy1, fx2, fy2 = map(int, rgb_face_box[0])
                cx = float((fx1 + fx2) * 0.5)
                cy = float((fy1 + fy2) * 0.5)
                id_conf_val = float(rgb_face_box[1])
                target_w = int(fx2 - fx1)
                target_h = int(fy2 - fy1)
                dx = float(cx - (cur_w_r * 0.5))  # <0: 在畫面左側, >0: 在畫面右側
                dy = float(cy - (cur_h_r * 0.5))  # <0: 在畫面偏上, >0: 在畫面偏下
            else:
                cx, cy = float(cur_w_r * 0.5), float(cur_h_r * 0.5)
                id_conf_val = 0.0
                target_w, target_h = 0, 0
                dx, dy = 0.0, 0.0

            payload = {
                "has_target": has_target,
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "conf": round(id_conf_val, 2),
                "status": "TRACKING:1:0.0" if has_target else "",
                "pig_id": current_pig_id,
                "dx": round(dx, 1),
                "dy": round(dy, 1),
                "target_w": target_w,
                "target_h": target_h,
                "frame_w": cur_w_r,
                "frame_h": cur_h_r
            }
            try:
                sock_tx.sendto(json.dumps(payload).encode("utf-8"), (xavier_ip, udp_port))
            except Exception:
                pass
            
            # 2. 耳根錨定 (依 ear_mode: yolo_bbox 雙模型 或 pose 姿態關鍵點模型)
            ear_boxes = []
            target_face = rgb_face_box or last_rgb_face
            run_ear = (not fast_mode) or (f_idx % 2 == 1) or (len(last_pose_ears) == 0)
            
            if run_ear:
                if ear_mode == 'yolo_bbox' and yolo_root_model is not None:
                    res_root = yolo_root_model(frame_r, imgsz=640, conf=0.10, verbose=False)[0]
                    raw_roots = []
                    for box in res_root.boxes:
                        c = int(box.cls[0].item())
                        cname = res_root.names[c]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        if cname == 'root' or c == 2:
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
                    
                    if len(raw_roots) >= 2:
                        sorted_roots = sorted(raw_roots[:2], key=lambda x: (x[0][0] + x[0][2]) / 2.0)
                        b_l, conf_l = sorted_roots[0]
                        b_r, conf_r = sorted_roots[1]
                        ear_boxes.append(('ear_root_l', b_l, conf_l, (int((b_l[0]+b_l[2])/2), int((b_l[1]+b_l[3])/2))))
                        ear_boxes.append(('ear_root_r', b_r, conf_r, (int((b_r[0]+b_r[2])/2), int((b_r[1]+b_r[3])/2))))
                    elif len(raw_roots) == 1:
                        b_one, conf_one = raw_roots[0]
                        rcx = (b_one[0] + b_one[2]) / 2.0
                        nose_x = (rgb_nose_box[0][0] + rgb_nose_box[0][2]) / 2.0 if rgb_nose_box else (cur_w_r / 2.0)
                        center = (int((b_one[0] + b_one[2]) / 2), int((b_one[1] + b_one[3]) / 2))
                        if rcx < nose_x:
                            ear_boxes.append(('ear_root_l', b_one, conf_one, center))
                        else:
                            ear_boxes.append(('ear_root_r', b_one, conf_one, center))
                elif pose_model is not None and target_face is not None:
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
                ear_boxes = list(last_pose_ears)
                                
            if len(ear_boxes) > 0: last_pose_ears = ear_boxes
            elif len(last_pose_ears) > 0: ear_boxes = last_pose_ears
            
            # 繪製【左上】RGB 畫面
            vis_rgb = frame_r.copy()
            if rgb_face_box is not None:
                if current_id_conf >= 0.35 and current_pig_id != "Detecting...":
                    face_lbl = f"Pig ID: {current_pig_id} ({current_id_conf*100:.0f}%)"
                    face_col = (50, 255, 120)
                else:
                    face_lbl = f"Face {rgb_face_box[1]:.2f}"
                    face_col = color_map['face']
                draw_styled_box(vis_rgb, rgb_face_box[0], face_lbl, face_col)
            if rgb_nose_box is not None: draw_styled_box(vis_rgb, rgb_nose_box[0], f"Snout {rgb_nose_box[1]:.2f}", color_map['nose'])
            for eb, c in rgb_eye_boxes: draw_styled_box(vis_rgb, eb, f"Eye {c:.2f}", color_map['eye'])
            for tag, eb, conf, center in ear_boxes:
                lbl = "Ear Root L" if tag == 'ear_root_l' else "Ear Root R"
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
                if thm_nose_box is not None:
                    last_thm_nose_box = thm_nose_box
            else:
                thm_nose_box = last_thm_nose_box
                
            # 距離視差指標 ref_w
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
            if len(t_sig) > 450: t_sig.pop(0) # 保持約 15 秒即時滑窗
            
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
                    snapped_box, t_eye, hot_pt = snap_to_tear_duct(frame_t, pbox, max_search_radius=20)
                    eye_temps[tag_e] = t_eye
                    if t_eye > 20.0:
                        if tag_e == 'eye_l': hist_eye_l.append(t_eye)
                        else: hist_eye_r.append(t_eye)
                    lbl_e = "Eye L (Top10)" if tag_e == 'eye_l' else "Eye R (Top10)"
                    draw_styled_box(vis_thm, snapped_box, lbl_e, (0, 255, 0), temp=t_eye, hot_pt=hot_pt)
                    
            # 雙耳根熱成像區域溫度 (採框內最高溫 Top-10 pixels 平均)
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
            if len(t_sig) >= 60:
                sig_arr = np.array(t_sig)
                sig_valid = sig_arr[sig_arr > 0]
                if len(sig_valid) > 40:
                    detrended = sig_valid - np.mean(sig_valid)
                    est_fps = max(10.0, avg_fps)
                    b, a = butter(2, [0.2 / (est_fps / 2), min(0.95, 2.0 / (est_fps / 2))], btype='band')
                    filtered = filtfilt(b, a, detrended)
                    sig_filtered_history = filtered.tolist()
                    peaks, _ = find_peaks(filtered, distance=int(est_fps * 0.4), prominence=0.03)
                    peaks_cnt = len(peaks)
                    if len(peaks) >= 2:
                        valid_dur_sec = len(sig_valid) / max(1.0, est_fps)
                        cur_bpm = (peaks_cnt / valid_dur_sec) * 60.0
                        
            period_avgs = {
                'eye_l': float(np.mean(hist_eye_l)) if len(hist_eye_l) > 0 else None,
                'eye_r': float(np.mean(hist_eye_r)) if len(hist_eye_r) > 0 else None,
                'ear_l': float(np.mean(hist_ear_l)) if len(hist_ear_l) > 0 else None,
                'ear_r': float(np.mean(hist_ear_r)) if len(hist_ear_r) > 0 else None,
            }
            
            # 5.5 自動巡檢感應錄製決策狀態機 (Auto-Sensing Server Engine)
            if auto_record_mode:
                if has_target and current_pig_id not in ("Detecting...", "Unknown"):
                    auto_rec_detect_streak += 1
                    auto_rec_miss_streak = 0
                    
                    if not is_recording_vitals:
                        # 連續穩定偵測到特定豬隻達到門檻 (6 幀，約 0.2~0.3 秒)，自動開錄！
                        if auto_rec_detect_streak >= 6:
                            _start_vitals_recording(current_pig_id, is_auto=True)
                    else:
                        # 正在自動錄製中，若畫面豬隻變更 (例如巡檢小車轉向下一頭豬)
                        if is_auto_session and auto_rec_target_pig is not None and current_pig_id != auto_rec_target_pig:
                            auto_rec_switch_streak += 1
                            if auto_rec_switch_streak >= 12:  # 約 0.4~0.5 秒穩定切換
                                print(f"[*] [自動巡檢切換] 目標由 {auto_rec_target_pig} 變更為 {current_pig_id}，自動結算上一隻並啟動新量測！")
                                _stop_and_save_vitals(is_auto=True)
                                _start_vitals_recording(current_pig_id, is_auto=True)
                                auto_rec_switch_streak = 0
                        else:
                            auto_rec_switch_streak = 0
                else:
                    auto_rec_detect_streak = 0
                    auto_rec_switch_streak = 0
                    if is_recording_vitals and is_auto_session:
                        auto_rec_miss_streak += 1
                        # 連續遺失目標超過 60 幀 (~2~3 秒，代表豬隻離場或小車駛向下一欄)
                        if auto_rec_miss_streak >= 60:
                            print(f"[*] [自動巡檢結算] 目標豬隻 {auto_rec_target_pig} 已離場超過 2 秒，自動存檔結算並進入待命狀態！")
                            _stop_and_save_vitals(is_auto=True)
                            _reset_measurement_state(reason="", keep_pig_id=False, reset_timer=False)
                        
            # 6. 四分割縮放與合成 (960 x 540)
            rgb_interp = cv2.INTER_CUBIC if cur_w_r < PANEL_W else cv2.INTER_AREA
            thm_interp = cv2.INTER_CUBIC if cur_w_t < PANEL_W else cv2.INTER_AREA
            q_tl = cv2.resize(vis_rgb, (PANEL_W, PANEL_H), interpolation=rgb_interp)
            q_bl = cv2.resize(vis_thm, (PANEL_W, PANEL_H), interpolation=thm_interp)
            q_tr = cv2.resize(vis_fusion, (PANEL_W, PANEL_H), interpolation=rgb_interp)
            rec_sec = (time.time() - vitals_rec_start_t) if is_recording_vitals else 0.0
            q_br = draw_telemetry_dashboard(
                eye_temps, ear_temps, last_snout_temp, cur_bpm, peaks_cnt, sig_filtered_history,
                f_idx, avg_fps, is_live=True, pig_id=current_pig_id, id_conf=current_id_conf,
                period_avgs=period_avgs, is_recording=is_recording_vitals,
                rec_duration_sec=rec_sec, rec_count=vitals_rec_count,
                auto_mode=auto_record_mode, is_auto_session=is_auto_session,
                is_video_rec=(tri_recorder is not None)
            )
            
            if time.time() - last_registered_alert_time < 5.0 and last_registered_alert_id is not None:
                top_left_banner = f"[TOP-LEFT] ★ NEW PIG AUTO-REGISTERED: {last_registered_alert_id} (DATA SAVED!) ★"
                top_left_col = (0, 215, 255)
            elif (current_id_conf >= 0.35 and current_pig_id != "Detecting..."):
                top_left_banner = f"[TOP-LEFT] LIVE RGB CAMERA | Pig ID: {current_pig_id} ({current_id_conf*100:.0f}%)"
                top_left_col = (50, 255, 120)
            else:
                top_left_banner = "[TOP-LEFT] LIVE RGB CAMERA (Xavier)"
                top_left_col = (100, 255, 100)
                
            add_banner(q_tl, top_left_banner, top_left_col)
            add_banner(q_bl, "[BOTTOM-LEFT] LIVE THERMAL CAMERA | Top-10 Pixels Mean", (0, 200, 255))
            add_banner(q_tr, "[TOP-RIGHT] ALIGNED FUSION | 4x2 Dynamic Matrix", (255, 180, 0))
            if tri_recorder is not None:
                vid_sec = tri_recorder.elapsed_sec()
                vm, vs = divmod(int(vid_sec), 60)
                cv2.rectangle(q_tr, (705, 5), (952, 35), (20, 20, 170), -1)
                cv2.rectangle(q_tr, (705, 5), (952, 35), (60, 80, 255), 1)
                v_dot = (0, 0, 255) if int(vid_sec * 2) % 2 == 0 else (140, 140, 255)
                cv2.circle(q_tr, (721, 20), 6, v_dot, -1, cv2.LINE_AA)
                cv2.putText(q_tr, f"3-CAM REC [{vm:02d}:{vs:02d}]", (735, 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[0:PANEL_H, 0:PANEL_W] = q_tl
            canvas[PANEL_H:CANVAS_H, 0:PANEL_W] = q_bl
            canvas[0:PANEL_H, PANEL_W:CANVAS_W] = q_tr
            canvas[PANEL_H:CANVAS_H, PANEL_W:CANVAS_W] = q_br
            
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), (0, 215, 255), 2)
            cv2.line(canvas, (0, PANEL_H), (CANVAS_W, PANEL_H), (0, 215, 255), 2)
            
            # 7. 若正在進行生理數據記錄，每幀即時寫入 CSV 檔案
            if is_recording_vitals and vitals_rec_writer is not None:
                rec_now_ts = time.strftime('%Y-%m-%d %H:%M:%S')
                rec_elap = round(time.time() - vitals_rec_start_t, 2)
                t_el = eye_temps.get('eye_l')
                t_er = eye_temps.get('eye_r')
                t_al = ear_temps.get('ear_root_l')
                t_ar = ear_temps.get('ear_root_r')
                
                vitals_rec_writer.writerow([
                    rec_now_ts, rec_elap, f_idx, current_pig_id, f"{current_id_conf:.2f}",
                    f"{t_el:.2f}" if (t_el is not None and t_el > 0) else "",
                    f"{t_er:.2f}" if (t_er is not None and t_er > 0) else "",
                    f"{t_al:.2f}" if (t_al is not None and t_al > 0) else "",
                    f"{t_ar:.2f}" if (t_ar is not None and t_ar > 0) else "",
                    f"{last_snout_temp:.2f}" if (last_snout_temp > 0) else "",
                    f"{int(float(cur_bpm) + 0.5)}" if (cur_bpm > 0) else "",
                    peaks_cnt,
                    "AUTO" if is_auto_session else "MANUAL"
                ])
                vitals_rec_file.flush()
                vitals_rec_count += 1
                if t_el and t_el > 0: vitals_buf_eye_l.append(t_el)
                if t_er and t_er > 0: vitals_buf_eye_r.append(t_er)
                if t_al and t_al > 0: vitals_buf_ear_l.append(t_al)
                if t_ar and t_ar > 0: vitals_buf_ear_r.append(t_ar)
                if cur_bpm and cur_bpm > 0: vitals_buf_bpm.append(cur_bpm)

            # 8. 檢查是否有未知新豬隻剛被自動建檔，若是則自動同步保存全景快照與個體生理檔案
            if pig_identifier.last_registered_pig_event is not None:
                new_event = pig_identifier.last_registered_pig_event
                pig_identifier.last_registered_pig_event = None  # 消耗事件
                new_id = new_event["pig_id"]
                last_registered_alert_id = new_id
                last_registered_alert_time = time.time()
                
                new_pigs_dir = os.path.join(out_dir, "new_pigs")
                os.makedirs(new_pigs_dir, exist_ok=True)
                
                # (1) 儲存四分割全景快照
                cv2.imwrite(os.path.join(new_pigs_dir, f"{new_id}_quad_snapshot.jpg"), canvas)
                cv2.imwrite(os.path.join(pig_identifier.gallery_img_dir, f"{new_id}_quad_snapshot.jpg"), canvas)
                
                # (2) 儲存可見光全幅與熱成像全幅
                cv2.imwrite(os.path.join(new_pigs_dir, f"{new_id}_rgb_scene.jpg"), frame_r)
                cv2.imwrite(os.path.join(new_pigs_dir, f"{new_id}_thm_scene.jpg"), frame_t)
                
                # (3) 儲存初檢完整生理檔案 (JSON & TXT)
                t_el = eye_temps.get('eye_l')
                t_er = eye_temps.get('eye_r')
                t_al = ear_temps.get('ear_root_l')
                t_ar = ear_temps.get('ear_root_r')
                vitals_profile = {
                    "pig_id": new_id,
                    "registered_time": new_event.get("registered_time", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "status": "Auto-Registered Unknown Pig (Open-Set ReID)",
                    "initial_vitals": {
                        "left_eye_temp_c": round(t_el, 2) if (t_el and t_el > 0) else None,
                        "right_eye_temp_c": round(t_er, 2) if (t_er and t_er > 0) else None,
                        "left_ear_root_temp_c": round(t_al, 2) if (t_al and t_al > 0) else None,
                        "right_ear_root_temp_c": round(t_ar, 2) if (t_ar and t_ar > 0) else None,
                        "snout_temp_c": round(last_snout_temp, 2) if last_snout_temp > 0 else None,
                        "respiration_bpm": int(float(cur_bpm) + 0.5) if cur_bpm > 0 else None
                    },
                    "files": {
                        "face_crop": f"{new_id}.jpg",
                        "multi_shot_album": f"registered_pigs/{new_id}/",
                        "quad_snapshot": f"{new_id}_quad_snapshot.jpg",
                        "rgb_scene": f"{new_id}_rgb_scene.jpg",
                        "thm_scene": f"{new_id}_thm_scene.jpg"
                    }
                }
                
                prof_json_path = os.path.join(new_pigs_dir, f"{new_id}_profile.json")
                with open(prof_json_path, "w", encoding="utf-8") as pf:
                    json.dump(vitals_profile, pf, ensure_ascii=False, indent=4)
                with open(os.path.join(pig_identifier.gallery_img_dir, f"{new_id}_profile.json"), "w", encoding="utf-8") as pf:
                    json.dump(vitals_profile, pf, ensure_ascii=False, indent=4)
                    
                prof_txt_path = os.path.join(new_pigs_dir, f"{new_id}_profile.txt")
                with open(prof_txt_path, "w", encoding="utf-8") as tf:
                    tf.write("================================================================================\n")
                    tf.write(f"  新註冊豬隻個體建檔卡 - {new_id} (Pig Health Dossier)\n")
                    tf.write("================================================================================\n")
                    tf.write(f"註冊時間: {vitals_profile['registered_time']}\n")
                    tf.write(f"豬隻編號: {new_id}\n")
                    tf.write(f"身分狀態: 自動辨識入庫之未知新個體 (Auto-Registered Unknown Pig)\n")
                    tf.write("--------------------------------------------------------------------------------\n")
                    tf.write("【初始生理檢測數據】\n")
                    tf.write(f"  - 左眼窩溫度: {vitals_profile['initial_vitals']['left_eye_temp_c']} °C\n")
                    tf.write(f"  - 右眼窩溫度: {vitals_profile['initial_vitals']['right_eye_temp_c']} °C\n")
                    tf.write(f"  - 左耳根溫度: {vitals_profile['initial_vitals']['left_ear_root_temp_c']} °C\n")
                    tf.write(f"  - 右耳根溫度: {vitals_profile['initial_vitals']['right_ear_root_temp_c']} °C\n")
                    tf.write(f"  - 鼻鏡溫度:   {vitals_profile['initial_vitals']['snout_temp_c']} °C\n")
                    tf.write(f"  - 呼吸率:     {vitals_profile['initial_vitals']['respiration_bpm']} BPM\n")
                    tf.write("--------------------------------------------------------------------------------\n")
                    tf.write("【建檔影像檔案】\n")
                    tf.write(f"  - 臉部特寫截圖: parameters/registered_pigs/{new_id}.jpg\n")
                    tf.write(f"  - 四分割快照:   new_pigs/{new_id}_quad_snapshot.jpg\n")
                    tf.write(f"  - 可見光原始檔: new_pigs/{new_id}_rgb_scene.jpg\n")
                    tf.write(f"  - 熱成像原始檔: new_pigs/{new_id}_thm_scene.jpg\n")
                    tf.write("================================================================================\n")
                    
                # (4) 寫入新豬隻清冊總表 CSV
                cat_csv = os.path.join(new_pigs_dir, "registered_pigs_catalog.csv")
                cat_exists = os.path.exists(cat_csv)
                with open(cat_csv, "a", newline="", encoding="utf-8-sig") as cf:
                    cw = csv.writer(cf)
                    if not cat_exists:
                        cw.writerow(["Registered_Time", "Pig_ID", "Status", "Left_Eye_Temp_C", "Right_Eye_Temp_C", "Left_Ear_Temp_C", "Right_Ear_Temp_C", "Snout_Temp_C", "Respiration_BPM", "Face_Image", "Quad_Snapshot"])
                    cw.writerow([
                        vitals_profile['registered_time'], new_id, "Auto-Registered Unknown Pig",
                        vitals_profile['initial_vitals']['left_eye_temp_c'],
                        vitals_profile['initial_vitals']['right_eye_temp_c'],
                        vitals_profile['initial_vitals']['left_ear_root_temp_c'],
                        vitals_profile['initial_vitals']['right_ear_root_temp_c'],
                        vitals_profile['initial_vitals']['snout_temp_c'],
                        vitals_profile['initial_vitals']['respiration_bpm'],
                        f"{new_id}.jpg", f"{new_id}_quad_snapshot.jpg"
                    ])
                    
                print("\n" + "=" * 80)
                print(f"🎉 [新豬隻自動建檔成功！]")
                print(f"[*] 系統辨識為未註冊新個體，已自動賦予新編號: {new_id}")
                print(f"[*] 臉部特寫存檔: parameters/registered_pigs/{new_id}.jpg")
                print(f"[*] 四分割全景儲存: {new_pigs_dir}/{new_id}_quad_snapshot.jpg")
                print(f"[*] 初檢生理檔案: {prof_json_path}")
                print(f"[*] 新豬隻總清冊: {cat_csv}")
                print("=" * 80 + "\n")
            
            if tri_recorder is not None:
                tri_recorder.write(canvas, frame_r, frame_t)
                
            if frame_callback is not None:
                try:
                    meta = {
                        "f_idx": f_idx,
                        "avg_fps": avg_fps,
                        "pig_id": current_pig_id,
                        "id_conf": current_id_conf,
                        "cur_bpm": cur_bpm,
                        "is_recording": is_recording_vitals,
                        "auto_mode": auto_record_mode,
                        "video_recording": (tri_recorder is not None),
                        "eye_l": eye_temps.get('eye_l'),
                        "eye_r": eye_temps.get('eye_r'),
                        "ear_l": ear_temps.get('ear_root_l'),
                        "ear_r": ear_temps.get('ear_root_r'),
                        "snout": last_snout_temp,
                    }
                    frame_callback(canvas, meta)
                except Exception:
                    pass
                
            if not headless:
                try:
                    cv2.imshow(window_name, canvas)
                except Exception:
                    pass
            
            if f_idx % 30 == 0:
                t_el = f"{eye_temps['eye_l']:.1f}C" if 'eye_l' in eye_temps else "N/A"
                t_er = f"{eye_temps['eye_r']:.1f}C" if 'eye_r' in eye_temps else "N/A"
                print(f"[*] [即時監控] 影格: {f_idx} | ID: {current_pig_id} | 速度: {avg_fps:.1f} FPS | 呼吸率: {cur_bpm:.1f} BPM | 左/右眼溫: {t_el}/{t_er}")
            
            if max_frames > 0 and f_idx >= max_frames:
                print(f"[*] 已達到指定測試影格上限 ({max_frames} 幀)，自動保存快照並安全結束！")
                if tri_recorder is not None:
                    snap_dir = os.path.join(tri_recorder.session_dir, "snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    snap_path = os.path.join(snap_dir, "xavier_live_test_snapshot.jpg")
                else:
                    snap_path = os.path.join(out_dir, "xavier_live_test_snapshot.jpg")
                cv2.imwrite(snap_path, canvas)
                print(f"[✓] 測試快照已儲存: {snap_path}")
                break
            
        key = cv2.waitKey(1) & 0xFF if not headless else 0xFF
        if not headless and f_idx > 0:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[*] 視窗已關閉，安全停止監控與錄影...")
                    break
            except Exception:
                break
        if command_queue is not None and not command_queue.empty():
            try:
                cmd_item = command_queue.get_nowait()
                if isinstance(cmd_item, str) and len(cmd_item) == 1:
                    key = ord(cmd_item)
                elif cmd_item == "SPACE":
                    key = 32
                elif cmd_item == "REC_VIDEO":
                    key = ord('c')
                elif cmd_item == "QUIT":
                    key = ord('q')
            except Exception:
                pass
        elif paused and headless:
            time.sleep(0.03)
        if key in [ord('q'), ord('Q'), 27]: # 'q' 或 ESC 退出
            print("[*] 收到退出指令，關閉即時監控系統...")
            break
        elif key in [ord('c'), ord('C')]: # 'c' 切換三路同步錄影並重設呼吸率與生理量測
            if tri_recorder is None:
                if is_recording_vitals:
                    _stop_and_save_vitals(is_auto=is_auto_session)
                _reset_measurement_state(reason="按 [C] 開始新錄影")
                _start_tri_recording()
            else:
                _stop_tri_recording()
                _reset_measurement_state(reason="按 [C] 停止錄影")
        elif key in [ord('s'), ord('S')]: # 's' 截圖
            if canvas is not None:
                snap_time = time.strftime("%Y%m%d_%H%M%S")
                if tri_recorder is not None:
                    snap_dir = os.path.join(tri_recorder.session_dir, "snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    snap_name = os.path.join(snap_dir, f"live_snapshot_{snap_time}.jpg")
                else:
                    snap_name = os.path.join(out_dir, f"live_snapshot_{snap_time}.jpg")
                cv2.imwrite(snap_name, canvas)
                print(f"[✓] 已儲存即時現場快照: {snap_name}")
        elif key == 32: # 空白鍵暫停/繼續
            paused = not paused
            print(f"[*] 系統 {'暫停中' if paused else '繼續運行'}")
        elif key in [ord('a'), ord('A')]: # 'A' 切換自動感應無人巡檢模式
            auto_record_mode = not auto_record_mode
            if auto_record_mode:
                print("\n[✓] 【自動巡檢感應模式：已啟用】辨識到豬隻將自動開錄，離場自動結算存檔。")
            else:
                if is_recording_vitals and is_auto_session:
                    _stop_and_save_vitals(is_auto=True)
                print("\n[!] 【自動巡檢感應模式：已關閉】切換為純手動控制模式，可按 [T] 或 [V] 手動啟閉記錄。")
        elif key in [ord('t'), ord('T'), ord('v'), ord('V')]: # 'T' 或 'V' 手動強制 開始/停止 生理數據記錄
            if not is_recording_vitals:
                target_id = current_pig_id if current_pig_id not in ("Detecting...", "Unknown") else "Pig_Manual"
                _start_vitals_recording(target_id, is_auto=False)
            else:
                _stop_and_save_vitals(is_auto=is_auto_session)
        elif key in [ord('r'), ord('R')]: # 重設呼吸信號
            _reset_measurement_state(reason="按 [R] 手動重設", keep_pig_id=True)
    except KeyboardInterrupt:
        print("\n[*] 偵測到中斷 (Ctrl+C)，正在安全停止系統並寫入錄影檔案...")
    finally:
        cam_rgb.release()
        cam_thm.release()
        sock_tx.close()
        _stop_tri_recording()
        if is_recording_vitals:
            _stop_and_save_vitals(is_auto=is_auto_session)
        cv2.destroyAllWindows()
    
    # 結束監控時將本段期間的呼吸率、平均左右眼窩溫度、平均左右耳根溫度與豬隻身分存成檔案
    from run_quad_view_unified import save_vitals_report
    duration_sec = max(1.0, time.time() - session_start_t)
    session_report_tag = f"live_session_{time.strftime('%Y%m%d_%H%M%S')}"
    save_vitals_report(
        out_dir=out_dir,
        video_tag=session_report_tag,
        pig_id=current_pig_id,
        id_conf=current_id_conf,
        final_bpm=cur_bpm,
        breath_cnt=peaks_cnt,
        duration_sec=duration_sec,
        hist_eye_l=hist_eye_l,
        hist_eye_r=hist_eye_r,
        hist_ear_l=hist_ear_l,
        hist_ear_r=hist_ear_r
    )
    if last_rec_session_dir is not None:
        if not last_rec_report_saved:
            save_vitals_report(
                out_dir=last_rec_session_dir,
                video_tag=session_report_tag,
                pig_id=current_pig_id,
                id_conf=current_id_conf,
                final_bpm=cur_bpm,
                breath_cnt=peaks_cnt,
                duration_sec=duration_sec,
                hist_eye_l=hist_eye_l,
                hist_eye_r=hist_eye_r,
                hist_ear_l=hist_ear_l,
                hist_ear_r=hist_ear_r
            )
        print(f"[*] 本次錄影完整子資料夾 (含影片/截圖/生理數據): {last_rec_session_dir}")
    print("[✓] 現場即時監控系統已安全釋放與關閉！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 雙光現場即時智慧監控系統 (Xavier Live AI Monitor)")
    parser.add_argument("--rgb", type=str, default=DEFAULT_RGB_URL, help=f"可見光視訊源 (預設為 Xavier RTSP: {DEFAULT_RGB_URL})")
    parser.add_argument("--thm", type=str, default=DEFAULT_THM_URL, help=f"熱成像視訊源 (預設為 Xavier RTSP: {DEFAULT_THM_URL})")
    parser.add_argument("--xavier_ip", type=str, default=DEFAULT_XAVIER_IP, help=f"Xavier Wi-Fi IP (預設: {DEFAULT_XAVIER_IP})")
    parser.add_argument("--udp_port", type=int, default=DEFAULT_UDP_PORT, help=f"傳送座標之 UDP Port (預設: {DEFAULT_UDP_PORT})")
    parser.add_argument("--ear_mode", type=str, default="pose", choices=["pose", "yolo_bbox"], help="耳根偵測模式: pose (骨骼姿態, 預設) 或 yolo_bbox")
    parser.add_argument("--frames", type=int, default=0, help="限制運行影格數 (預設 0 為持續監控直到按 Q)")
    parser.add_argument("--headless", action="store_true", help="無 GUI 視窗背景測試模式")
    parser.add_argument("--save", action="store_true", help="是否同時錄製三路同步 MP4 影片 (Quad View + RGB + THM)")
    parser.add_argument("--video_dir", type=str, default=None, help="三路錄影輸出根目錄 (預設: recorded_videos)")
    parser.add_argument("--demo", action="store_true", help="啟用內建現場預錄影片模擬串流測試")
    parser.add_argument("--full", action="store_true", help="啟用全幀每幀推論 (預設為高速交替跳幀模式，現場省電又順暢)")
    parser.add_argument("--no_auto", action="store_true", help="關閉自動感應無人巡檢錄製模式 (改為純手動按鍵模式)")
    args = parser.parse_args()
    
    fast_mode = not args.full
    
    if args.demo:
        # 示範模式：使用已對齊的現場影片模擬即時雙光鏡頭
        src_rgb = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
        src_thm = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")
    else:
        src_rgb = args.rgb
        src_thm = args.thm
        
    start_live_monitor(
        src_rgb=src_rgb,
        src_thm=src_thm,
        xavier_ip=args.xavier_ip,
        udp_port=args.udp_port,
        save_video=args.save,
        fast_mode=fast_mode,
        ear_mode=args.ear_mode,
        max_frames=args.frames,
        headless=args.headless,
        auto_record=not args.no_auto,
        video_dir=args.video_dir
    )
