"""
pig_shm_engine.py — 高性能跨平台零拷貝共享記憶體（Zero-Copy IPC）引擎
設計重點：
1. 運用今日核心觀念：Process 多進程 ＋ 共享記憶體（Shared Memory）徹底繞過 Python GIL 瓶頸。
2. 雙相機 RTSP 擷取獨立進程：以無阻塞模式持續更新記憶體，消除網路攝影機累積延遲。
3. EasyOCR 背景異步進程：動態溫標解析完全脫鉤，1Hz 定時更新，不再引發影格卡頓。
4. 完美相容 Windows（底層呼叫 CreateFileMapping）與 Linux（底層呼叫 shm_open）。
"""

import time
import os
import re
import numpy as np
from multiprocessing import Process, Event, Array, RawValue
from multiprocessing.shared_memory import SharedMemory
import cv2

class SharedFrameBuffer:
    """
    高效能零拷貝雙光譜影像共享記憶體環
    RGB 與 Thermal 影像直接映射於作業系統核心記憶體中
    """
    def __init__(self, name_prefix="pig_v4", rgb_shape=(1080, 1920, 3), thm_shape=(1080, 1920, 3), is_creator=True):
        self.name_prefix = name_prefix
        self.rgb_shape = rgb_shape
        self.thm_shape = thm_shape
        self.is_creator = is_creator

        self.rgb_size = int(np.prod(rgb_shape))
        self.thm_size = int(np.prod(thm_shape))

        self.rgb_shm_name = f"{name_prefix}_rgb"
        self.thm_shm_name = f"{name_prefix}_thm"

        if is_creator:
            # 清理可能遺留的舊共享記憶體
            for name in [self.rgb_shm_name, self.thm_shm_name]:
                try:
                    old_shm = SharedMemory(name=name)
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass

            self.shm_rgb = SharedMemory(name=self.rgb_shm_name, create=True, size=self.rgb_size)
            self.shm_thm = SharedMemory(name=self.thm_shm_name, create=True, size=self.thm_size)
        else:
            self.shm_rgb = SharedMemory(name=self.rgb_shm_name)
            self.shm_thm = SharedMemory(name=self.thm_shm_name)

        # 零拷貝 NumPy View
        self.rgb_arr = np.ndarray(self.rgb_shape, dtype=np.uint8, buffer=self.shm_rgb.buf)
        self.thm_arr = np.ndarray(self.thm_shape, dtype=np.uint8, buffer=self.shm_thm.buf)

    def write_frames(self, frame_rgb, frame_thm):
        """將影像寫入共享記憶體（零跨進程序列化負擔）"""
        if frame_rgb is not None:
            if frame_rgb.shape != self.rgb_shape:
                frame_rgb = cv2.resize(frame_rgb, (self.rgb_shape[1], self.rgb_shape[0]))
            self.rgb_arr[:] = frame_rgb[:]

        if frame_thm is not None:
            if frame_thm.shape != self.thm_shape:
                frame_thm = cv2.resize(frame_thm, (self.thm_shape[1], self.thm_shape[0]))
            self.thm_arr[:] = frame_thm[:]

    def get_rgb_view(self):
        return self.rgb_arr

    def get_thm_view(self):
        return self.thm_arr

    def cleanup(self):
        try:
            self.shm_rgb.close()
            self.shm_thm.close()
            if self.is_creator:
                self.shm_rgb.unlink()
                self.shm_thm.unlink()
        except Exception:
            pass

def _rtsp_producer_worker(rgb_source, thm_source, name_prefix, rgb_shape, thm_shape, frame_counter, stop_event):
    """
    RTSP 擷取工作行程：專責持續讀取串流，隨時保持共享記憶體為最新影像，拋棄累積過期影格
    """
    shm_buf = SharedFrameBuffer(name_prefix=name_prefix, rgb_shape=rgb_shape, thm_shape=thm_shape, is_creator=False)
    
    cap_rgb = cv2.VideoCapture(rgb_source)
    cap_thm = cv2.VideoCapture(thm_source)

    # 將底層緩衝區限制為 1，避免緩衝堆疊
    cap_rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_thm.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[*] [RTSP Producer Process] 影像串流連線就緒 (PID: {os.getpid()})")

    try:
        while not stop_event.is_set():
            # 硬體層面同步抓取 (Grab)
            grab_rgb = cap_rgb.grab()
            grab_thm = cap_thm.grab()

            if not grab_rgb or not grab_thm:
                time.sleep(0.01)
                continue

            ret_rgb, frame_rgb = cap_rgb.retrieve()
            ret_thm, frame_thm = cap_thm.retrieve()

            if ret_rgb and ret_thm:
                shm_buf.write_frames(frame_rgb, frame_thm)
                with frame_counter.get_lock():
                    frame_counter.value += 1
            else:
                time.sleep(0.005)
    finally:
        cap_rgb.release()
        cap_thm.release()
        shm_buf.cleanup()
        print("[*] [RTSP Producer Process] 安全釋放完成。")

def _ocr_scale_worker(name_prefix, thm_shape, temp_scale_array, stop_event, update_interval=1.0):
    """
    EasyOCR 異步溫標解析工作行程：完全獨立於推論進程，1Hz 定時更新動態溫標
    """
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
    shm_buf = SharedFrameBuffer(name_prefix=name_prefix, thm_shape=thm_shape, is_creator=False)

    print(f"[*] [OCR Worker Process] 溫標解析模型就緒 (PID: {os.getpid()})")
    
    thm_h, thm_w = thm_shape[:2]

    def extract_num(text):
        match = re.search(r'(\d+\.\d+|\d+)', text)
        return float(match.group(1)) if match else None

    try:
        while not stop_event.is_set():
            thm_img = shm_buf.get_thm_view()
            if thm_img is not None:
                roi_top_right = thm_img[0:int(thm_h * 0.25), int(thm_w * 0.8):thm_w]
                roi_bot_right = thm_img[int(thm_h * 0.75):thm_h, int(thm_w * 0.8):thm_w]

                try:
                    top_res = reader.readtext(roi_top_right, detail=0)
                    bot_res = reader.readtext(roi_bot_right, detail=0)

                    max_t = None
                    min_t = None

                    for t in top_res:
                        val = extract_num(t)
                        if val is not None and 10 < val < 100:
                            max_t = val
                            break

                    for t in bot_res:
                        val = extract_num(t)
                        if val is not None and -20 < val < 50:
                            min_t = val
                            break

                    # 寫入共享記憶體 Array (Thread-safe)
                    if max_t is not None or min_t is not None:
                        with temp_scale_array.get_lock():
                            if max_t is not None:
                                temp_scale_array[0] = max_t
                            if min_t is not None:
                                temp_scale_array[1] = min_t
                except Exception as e:
                    pass

            time.sleep(update_interval)
    finally:
        shm_buf.cleanup()
        print("[*] [OCR Worker Process] 安全釋放完成。")

class PigZeroCopyEngine:
    """
    V4 零拷貝共享記憶體管線主控制器
    """
    def __init__(self, rgb_source, thm_source, rgb_shape=(1080, 1920, 3), thm_shape=(1080, 1920, 3), name_prefix="pig_v4"):
        self.rgb_source = rgb_source
        self.thm_source = thm_source
        self.rgb_shape = rgb_shape
        self.thm_shape = thm_shape
        self.name_prefix = name_prefix

        # 建立主共享記憶體
        self.master_shm = SharedFrameBuffer(
            name_prefix=name_prefix,
            rgb_shape=rgb_shape,
            thm_shape=thm_shape,
            is_creator=True
        )

        self.stop_event = Event()
        self.frame_counter = RawValue('Q', 0)
        # [0]: Max Temp (預設 35.0), [1]: Min Temp (預設 20.0)
        self.temp_scale_array = Array('d', [35.0, 20.0])

        self.p_rtsp = None
        self.p_ocr = None

    def start(self, enable_ocr=True):
        print(f"🚀 [Pig V4 Engine] 啟動多進程零拷貝架構 (Windows/Linux 原生共享記憶體)")
        
        # 啟動 RTSP 擷取 Process
        self.p_rtsp = Process(
            target=_rtsp_producer_worker,
            args=(self.rgb_source, self.thm_source, self.name_prefix, self.rgb_shape, self.thm_shape, self.frame_counter, self.stop_event),
            daemon=True
        )
        self.p_rtsp.start()

        # 啟動 OCR 動態溫標 Process
        if enable_ocr:
            self.p_ocr = Process(
                target=_ocr_scale_worker,
                args=(self.name_prefix, self.thm_shape, self.temp_scale_array, self.stop_event),
                daemon=True
            )
            self.p_ocr.start()

    def get_latest_frames(self):
        """讀取最新影像與當前溫標（零拷貝）"""
        rgb = self.master_shm.get_rgb_view()
        thm = self.master_shm.get_thm_view()
        count = self.frame_counter.value
        with self.temp_scale_array.get_lock():
            max_t = self.temp_scale_array[0]
            min_t = self.temp_scale_array[1]
        return rgb, thm, count, max_t, min_t

    def stop(self):
        print("\n[*] [Pig V4 Engine] 正在停止背景工作進程...")
        self.stop_event.set()
        if self.p_rtsp:
            self.p_rtsp.join(timeout=2.0)
        if self.p_ocr:
            self.p_ocr.join(timeout=2.0)
        self.master_shm.cleanup()
        print("[*] [Pig V4 Engine] 系統關閉完成，共享記憶體已回收。")
