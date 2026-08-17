import cv2
import time
import re
import numpy as np
import easyocr

# ==========================================
# 參數設定區
# ==========================================
# 請替換成您真實的 RTSP 網址
RTSP_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

# 預設的溫度上下限 (當程式剛啟動，或是字體被遮擋導致 OCR 失敗時的備用值)
current_min_temp = 23.0
current_max_temp = 32.0

def extract_temperature(text):
    """用正則表達式從 OCR 辨識結果中萃取出純數字"""
    # 尋找類似 "32.1" 或 "23.6" 的數字
    match = re.search(r'(\d+\.\d+|\d+)', text)
    if match:
        return float(match.group(1))
    return None

def main():
    global current_min_temp, current_max_temp
    
    print("[*] 正在載入 EasyOCR AI 文字辨識模型 (初次執行會下載模型，請稍候)...")
    # 開啟 gpu=True 讓它用您的 RTX 5070 Ti 飆速辨識！
    reader = easyocr.Reader(['en'], gpu=True)
    
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[錯誤] 無法連接到 RTSP 影像。")
        return

    print("[*] 開始測試 (動態 OCR 模式)... 點擊影像視窗並按 'q' 結束。")
    
    last_ocr_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(10)
            continue
            
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        
        # ===================================================
        # [AI OCR 動態偵測] 
        # 為了避免畫面卡頓，我們不需要每幀都做 OCR
        # 設定「每隔 1 秒鐘」掃描一次右邊跳動的數字即可
        # ===================================================
        current_time = time.time()
        if current_time - last_ocr_time > 1.0:
            # 擷取右上角與右下角的局部畫面 (只讓 AI 看這兩塊，速度最快)
            # 抓取右邊 40% 寬度，並避開最上方的「時間字體」，往下移去抓最高/最低溫
            roi_top_right = frame[int(height*0.15):int(height*0.35), int(width*0.6):width]
            roi_bot_right = frame[int(height*0.65):int(height*0.85), int(width*0.6):width]
            
            # 放大兩倍來提升 OCR 對於小字體的辨識率 (很重要！)
            roi_top_enlarged = cv2.resize(roi_top_right, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            roi_bot_enlarged = cv2.resize(roi_bot_right, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            
            # 呼叫 EasyOCR 讀字
            top_results = reader.readtext(roi_top_enlarged, detail=0)
            bot_results = reader.readtext(roi_bot_enlarged, detail=0)
            
            # 從右上角的文字中找出最高溫
            for text in top_results:
                val = extract_temperature(text)
                if val is not None and 10 < val < 100: # 確認數字合理
                    current_max_temp = val
                    break
                    
            # 從右下角的文字中找出最低溫
            for text in bot_results:
                val = extract_temperature(text)
                if val is not None and -20 < val < 50: # 確認數字合理
                    current_min_temp = val
                    break
                    
            last_ocr_time = current_time

        # ===================================================
        # [亮度換算溫度]
        # ===================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center_roi = gray[cy-1:cy+2, cx-1:cx+2]
        pixel_intensity = center_roi.mean()
        
        # 動態線性對照換算溫度 (上下限數值會跟著 OCR 自動更新)
        temp = current_min_temp + (pixel_intensity / 255.0) * (current_max_temp - current_min_temp)
        
        # ===================================================
        # [繪製畫面]
        # ===================================================
        # 畫準心 (綠色)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        
        # 在畫面上標示目前 AI 讀取到的上下限 (紅色與藍色)
        cv2.putText(frame, f"AI Max: {current_max_temp} C", (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"AI Min: {current_min_temp} C", (width - 250, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 在十字準心旁邊印出亮度和換算後的溫度 (分開兩行，拉開間距避免重疊)
        info_text1 = f"Pixel: {pixel_intensity:.1f}/255"
        cv2.putText(frame, info_text1, (cx + 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Temp: {temp:.1f} C", (cx + 30, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 在畫面上畫出 AI 正在偷看的範圍 (除錯用)，並顯示當前抓到的數值
        cv2.rectangle(frame, (int(width*0.6), int(height*0.15)), (width, int(height*0.35)), (0, 0, 255), 2)
        cv2.putText(frame, f"Max: {current_max_temp}", (int(width*0.6)+5, int(height*0.15)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.rectangle(frame, (int(width*0.6), int(height*0.65)), (width, int(height*0.85)), (255, 0, 0), 2)
        cv2.putText(frame, f"Min: {current_min_temp}", (int(width*0.6)+5, int(height*0.85)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 顯示影像
        cv2.imshow("Center Temp Tracker (OCR)", cv2.resize(frame, (960, 540)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
