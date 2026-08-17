import cv2

# ==========================================
# 參數設定區 (請隨時依照畫面右側的數字手動修改)
# ==========================================
# 請替換成您真實的 RTSP 網址
RTSP_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

MIN_TEMP = 23.6  # 畫面右側色條最下方的最低溫度
MAX_TEMP = 45.1  # 畫面右側色條最上方的最高溫度

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    # 降低緩衝以減少延遲
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not cap.isOpened():
        print("[錯誤] 無法連接到 RTSP 影像，請檢查網址或網路設定。")
        return

    print("[*] 開始測試畫面中心點溫度... 點擊影像視窗並按 'q' 結束。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(10)
            continue
            
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        
        # 1. 將彩色畫面轉換為灰階 (亮度)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 取得中心點的亮度值 (0~255)
        # 為了避免單一像素剛好是雜訊，我們取中心 3x3 範圍求平均亮度
        center_roi = gray[cy-1:cy+2, cx-1:cx+2]
        pixel_intensity = center_roi.mean()
        
        # 3. 線性對照換算成溫度
        # 公式: 溫度 = 最小值 + (亮度 / 255) * (最大值 - 最小值)
        temp = MIN_TEMP + (pixel_intensity / 255.0) * (MAX_TEMP - MIN_TEMP)
        
        # 4. 在畫面上繪製十字準星與文字
        # 畫準心 (綠色)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        
        # 準備要印出的文字
        info_text1 = f"Pixel: {pixel_intensity:.1f} / 255"
        info_text2 = f"Temp: {temp:.2f} C"
        
        # 將文字印在準心旁邊 (黃色)
        cv2.putText(frame, info_text1, (cx + 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, info_text2, (cx + 30, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 顯示縮放後的畫面 (避免 1080p 畫面太大超出版面)
        cv2.imshow("Center Temp Test", cv2.resize(frame, (960, 540)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
