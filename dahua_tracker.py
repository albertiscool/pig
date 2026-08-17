import cv2
import csv
import time
import queue
import threading
from ultralytics import YOLO
import ctypes

# ---------------------------------------------------------
# 大華 SDK 結構體定義
# ---------------------------------------------------------
class NET_POINT(ctypes.Structure):
    _fields_ = [("nx", ctypes.c_short), ("ny", ctypes.c_short)]

class NET_IN_RADIOMETRY_GETPOINTTEMP(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint),
        ("nChannel", ctypes.c_int),
        ("stCoordinate", NET_POINT),
    ]

class NET_OUT_RADIOMETRY_GETPOINTTEMP(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint),
        ("fTemperature", ctypes.c_float),
    ]

class DahuaNetSDKWrapper:
    def __init__(self, ip, user, pwd):
        import os
        # 載入 SDK 動態連結庫
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 針對 Python 3.8+ 安全性更新：必須明確加入 DLL 搜尋路徑
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(current_dir)
                
            dll_path = os.path.join(current_dir, "dhnetsdk.dll")
            self.sdk = ctypes.windll.LoadLibrary(dll_path)
            print("[*] 成功載入 dhnetsdk.dll！")
        except Exception as e:
            print(f"[警告] 無法載入 dhnetsdk.dll: {e}")
            self.sdk = None
            self.login_id = 0
            return
            
        # 初始化 SDK
        self.sdk.CLIENT_Init(None, 0)
        
        # 登入設備
        self.login_id = self.sdk.CLIENT_LoginEx2(
            ip.encode(), 37777, user.encode(), pwd.encode(), 
            0, None, ctypes.byref(ctypes.c_int()), ctypes.byref(ctypes.c_int())
        )
        if self.login_id == 0:
            print("[錯誤] 大華攝影機登入失敗！請檢查 IP 與帳號密碼。")

    def get_point_temperature(self, x, y, channel=2):
        """傳入影像的 x, y 座標，回傳絕對溫度"""
        if self.sdk is None or self.login_id == 0:
            return -999.0

        # 將 1920x1080 轉換為 8192 萬分比坐標系
        norm_x = int((x / 1920.0) * 8192)
        norm_y = int((y / 1080.0) * 8192)

        in_param = NET_IN_RADIOMETRY_GETPOINTTEMP()
        in_param.dwSize = ctypes.sizeof(NET_IN_RADIOMETRY_GETPOINTTEMP)
        in_param.nChannel = channel
        in_param.stCoordinate.nx = norm_x
        in_param.stCoordinate.ny = norm_y

        out_param = NET_OUT_RADIOMETRY_GETPOINTTEMP()
        out_param.dwSize = ctypes.sizeof(NET_OUT_RADIOMETRY_GETPOINTTEMP)

        COMMAND_GET_POINT_TEMP = 0x0113
        
        res = self.sdk.CLIENT_QueryDevInfo(
            self.login_id, 
            COMMAND_GET_POINT_TEMP, 
            ctypes.byref(in_param), 
            ctypes.byref(out_param), 
            None, 
            3000
        )

        if res:
            return out_param.fTemperature
        else:
            err = self.sdk.CLIENT_GetLastError()
            # 加上錯誤代碼印出，幫助除錯
            print(f"[SDK 錯誤] 測溫請求失敗 (Channel: {channel}, X: {norm_x}, Y: {norm_y})，大華錯誤碼: {err & 0x7fffffff}")
            return -999.0

    def cleanup(self):
        if self.sdk and self.login_id != 0:
            self.sdk.CLIENT_Logout(self.login_id)
            self.sdk.CLIENT_Cleanup()

# ---------------------------------------------------------
# 1. 非阻塞 RTSP 影像讀取執行緒
# ---------------------------------------------------------
class RTSPReaderThread(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 減少底層緩衝
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01) # 斷線重連機制可在此實作

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        self.cap.release()

# ---------------------------------------------------------
# 2. 非同步大華 SDK 測溫與 CSV 寫入執行緒
# ---------------------------------------------------------
class TempQueryAndSaveThread(threading.Thread):
    def __init__(self, csv_filename, sdk_wrapper):
        super().__init__()
        self.csv_filename = csv_filename
        self.sdk = sdk_wrapper
        self.task_queue = queue.Queue()
        self.running = True
        self.latest_temps = {} # 存放最新的溫度供畫面顯示 (target_id -> temp)

    def run(self):
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Target_ID', 'Center_X', 'Center_Y', 'Temperature_C'])
            
            while self.running or not self.task_queue.empty():
                try:
                    task = self.task_queue.get(timeout=0.1)
                    timestamp, target_id, x, y = task
                    
                    temperature = self.sdk.get_point_temperature(x, y)
                    self.latest_temps[target_id] = temperature # 更新供畫面讀取
                    
                    writer.writerow([timestamp, target_id, x, y, temperature])
                    self.task_queue.task_done()
                except queue.Empty:
                    continue

    def add_task(self, timestamp, target_id, x, y):
        self.task_queue.put((timestamp, target_id, x, y))

    def stop(self):
        self.running = False

# ---------------------------------------------------------
# 3. 主迴圈
# ---------------------------------------------------------
def main():
    # TODO: 替換為您的實際設備資訊
    RTSP_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"
    CSV_FILE = "temperature_log.csv"
    VIDEO_OUT = "output.mp4"
    
    print("[*] 正在載入 YOLO 模型...")
    model = YOLO("yolov8n.pt") 
    
    print("[*] 正在初始化大華 SDK...")
    dahua_sdk = DahuaNetSDKWrapper(ip="192.168.1.108", user="admin", pwd="sp123456")
    
    print("[*] 啟動 RTSP 與測溫執行緒...")
    reader = RTSPReaderThread(RTSP_URL)
    reader.start()
    
    temp_saver = TempQueryAndSaveThread(CSV_FILE, dahua_sdk)
    temp_saver.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(VIDEO_OUT, fourcc, 30.0, (1920, 1080))

    print("[*] 系統啟動成功！按 'q' 結束。")
    try:
        while True:
            frame = reader.get_latest_frame()
            if frame is None:
                continue

            current_time = time.time()
            
            results = model(frame, verbose=False)[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                temp_saver.add_task(current_time, cls_id, cx, cy)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # 從背景執行緒取得最新測到的溫度並顯示在畫面上
                last_temp = temp_saver.latest_temps.get(cls_id, None)
                if last_temp is not None:
                    if last_temp == -999.0:
                        text = "Temp: N/A"
                    else:
                        text = f"Temp: {last_temp:.1f} C"
                    cv2.putText(frame, text, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out_video.write(frame)
            cv2.imshow("Thermal YOLO", cv2.resize(frame, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\n[*] 正在安全關閉系統...")
        reader.stop()
        temp_saver.stop()
        reader.join()
        temp_saver.join()
        out_video.release()
        cv2.destroyAllWindows()
        dahua_sdk.cleanup()
        print("[*] 關閉完成。")

if __name__ == "__main__":
    main()
