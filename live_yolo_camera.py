import cv2
import os
import time
from ultralytics import YOLO

# ==========================================
# 參數設定區
# ==========================================
# 攝影機 RTSP 網址 (目前預設為可見光 Channel 1)
RTSP_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# 模型路徑：改回使用純抓框的 YOLO 模型 (不使用關鍵點)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "..", "parameters", "bestv12.pt")

def main():
    print("=" * 50)
    print("   攝影機即時 YOLO 偵測測試工具 (純抓框)")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"[錯誤] 找不到模型檔案: {MODEL_PATH}")
        print("請打開程式碼修改 MODEL_PATH 為正確的路徑。")
        return

    print(f"[*] 載入模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"[*] 連線攝影機: {RTSP_URL}...")
    cap = cv2.VideoCapture(RTSP_URL)
    
    # 降低緩衝區，確保即時性
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[錯誤] 無法開啟攝影機，請檢查網路或 RTSP 網址。")
        return

    print("[*] 開始即時推論，按下 'q' 鍵退出。")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("串流中斷或讀取失敗。")
            break

        # 將影像送進 YOLO 模型推論
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

        # 取得 YOLO 內建畫好的影像 (會畫出基本的點跟框)
        annotated_frame = results[0].plot()

        # 計算並顯示即時 FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 縮小顯示
        display_frame = cv2.resize(annotated_frame, (1024, 576))
        
        cv2.imshow("Live YOLO Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
