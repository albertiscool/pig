import cv2
import time
from datetime import datetime

# 替換為實際的帳號密碼
RTSP_RGB = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
RTSP_THM = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

cap_rgb = cv2.VideoCapture(RTSP_RGB)
cap_thm = cv2.VideoCapture(RTSP_THM)

if not cap_rgb.isOpened() or not cap_thm.isOpened():
    print("無法連線到攝影機，請檢查網路或 RTSP 網址")
    exit()

# 為了減少延遲，將緩衝區設為最小值
cap_rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap_thm.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 自動抓取攝影機輸出的解析度與 FPS
w_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
h_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_rgb = int(cap_rgb.get(cv2.CAP_PROP_FPS))
if fps_rgb == 0: fps_rgb = 30

w_thm = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
h_thm = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_thm = int(cap_thm.get(cv2.CAP_PROP_FPS))
if fps_thm == 0: fps_thm = 30

# 以當前時間作為檔名的一部分，避免覆蓋
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_rgb = f'rgb_record_{timestamp}.mp4'
filename_thm = f'thm_record_{timestamp}.mp4'

# 設定輸出影片參數 (H.264 格式，副檔名 .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_rgb = cv2.VideoWriter(filename_rgb, fourcc, fps_rgb, (w_rgb, h_rgb))
out_thm = cv2.VideoWriter(filename_thm, fourcc, fps_thm, (w_thm, h_thm))

print(f"開始錄製 60 秒影片...")
print(f"  - 可見光: {filename_rgb} ({w_rgb}x{h_rgb}, {fps_rgb}fps)")
print(f"  - 熱成像: {filename_thm} ({w_thm}x{h_thm}, {fps_thm}fps)")

start_time = time.time()

while True:
    # 為了讓兩支影片時間軸對齊，先同時送出抓取指令 (grab)
    cap_rgb.grab()
    cap_thm.grab()
    
    # 再將畫面拿出來 (retrieve)
    ret_rgb, frame_rgb = cap_rgb.retrieve()
    ret_thm, frame_thm = cap_thm.retrieve()

    if not ret_rgb or not ret_thm:
        print("串流中斷！")
        break

    # 寫入影格到對應的影片檔
    out_rgb.write(frame_rgb)
    out_thm.write(frame_thm)

    # 顯示即時畫面，將兩張圖縮放後並排顯示 (非必要)
    vis_rgb = cv2.resize(frame_rgb, (640, 480))
    vis_thm = cv2.resize(frame_thm, (640, 480))
    cv2.imshow('Live Dual Record', cv2.hconcat([vis_rgb, vis_thm]))
    
    # 檢查時間是否超過 60 秒
    elapsed = time.time() - start_time
    if elapsed >= 60:
        print("60 秒錄製完成，檔案已儲存！")
        break

    # 按下 'q' 鍵可提早中斷錄影
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("使用者手動中斷錄影")
        break

# 釋放資源並關閉視窗
cap_rgb.release()
cap_thm.release()
out_rgb.release()
out_thm.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
