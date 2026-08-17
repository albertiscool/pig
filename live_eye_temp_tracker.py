import os
import sys
import cv2
import time
import numpy as np
import torch
import re
import csv
from datetime import datetime
import easyocr
from ultralytics import YOLO

# ---------------------------------------------------------
# 從現有的主程式直接匯入 YOLO 與 P2P 映射模型
# (這樣就不需要重複寫上百行的模型架構代碼)
# ---------------------------------------------------------
from video_pipeline2 import (
    SwinTransformerP2P, 
    process_yolo_detection_and_sort_points, 
    get_model_warp_matrix_inference, 
    get_warped_box,
    P2P_MODEL_PATH,
    YOLOV8_KEYPOINT_PATH,
    IMAGE_SIZE,
    CONF_THRESHOLD
)

# ==========================================
# 參數設定區 (請依照現場實際網址修改)
# ==========================================
RGB_STREAM_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
THM_STREAM_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

# 您也可以傳入影片檔進行離線測試：
# RGB_STREAM_URL = "data/test_vid/484-6_11410181546_可見光_正面(30秒呼吸34次).mp4"
# THM_STREAM_URL = "data/test_vid/484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4"

CSV_SAVE_PATH = "field_eye_temp_records.csv"

def extract_temperature(text):
    match = re.search(r'(\d+\.\d+|\d+)', text)
    if match:
        return float(match.group(1))
    return None

def main():
    print("=" * 60)
    print("  現場豬隻即時眼部體溫追蹤系統 (Live Tracker)")
    print("=" * 60)

    # 1. 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在載入 YOLO 與 Swin P2P 模型 (Device: {device})...")
    detect_model = YOLO(YOLOV8_KEYPOINT_PATH)
    
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
    if device.type == 'cuda': p2p_model.half()
    p2p_model.eval()

    print("[*] 正在載入 EasyOCR 動態溫標辨識模型...")
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    # 2. 開啟影像串流
    print("[*] 正在連線攝影機串流...")
    cap_rgb = cv2.VideoCapture(RGB_STREAM_URL)
    cap_thm = cv2.VideoCapture(THM_STREAM_URL)
    
    # 降低緩衝區，確保即時性
    cap_rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_thm.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_rgb.isOpened() or not cap_thm.isOpened():
        print("[錯誤] 無法開啟攝影機，請檢查 RTSP 網址或網路連線！")
        return

    # 3. 初始化 CSV 記錄檔
    if not os.path.exists(CSV_SAVE_PATH):
        with open(CSV_SAVE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Left_Eye_Temp', 'Right_Eye_Temp', 'Max_Scale', 'Min_Scale'])

    # 變數初始化
    current_max_temp = 35.0
    current_min_temp = 20.0
    last_ocr_time = 0
    
    print("\n[系統準備就緒] 視窗即將跳出。")
    print("操作說明:")
    print("  - 按 's' 鍵：將當前的眼溫寫入 CSV 記錄檔。")
    print("  - 按 'q' 鍵：離開程式。")

    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_thm, frame_thm = cap_thm.read()
        
        if not ret_rgb or not ret_thm:
            # 若是播放影片檔結束，或是網路斷線
            cv2.waitKey(10)
            continue
            
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
        # Step B: YOLO 眼睛偵測 & Swin P2P 空間映射
        # ---------------------------------------------------------
        yolo_res = detect_model.predict(frame_rgb, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
        detect_map = process_yolo_detection_and_sort_points(yolo_res)
        
        # 將 RGB 偵測到的座標，算出一個變形矩陣 M 來映射到 THM
        M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, p2p_model, device)
        
        gray_thm = cv2.cvtColor(frame_thm, cv2.COLOR_BGR2GRAY)
        
        eye_temps_data = {'Left Eye': None, 'Right Eye': None}

        # ---------------------------------------------------------
        # Step C: 擷取映射過去的眼睛 ROI，並換算溫度
        # ---------------------------------------------------------
        if M is not None:
            for eye_id, eye_name in [(0, 'Left Eye'), (1, 'Right Eye')]:
                if eye_id in detect_map:
                    # 在可見光畫面畫出原本的追蹤框 (藍色)
                    orig_box = detect_map[eye_id][1]
                    cv2.rectangle(frame_rgb, (orig_box[0], orig_box[1]), (orig_box[2], orig_box[3]), (255, 0, 0), 2)
                    cv2.putText(frame_rgb, eye_name, (orig_box[0], orig_box[1]-5), 0, 0.6, (255, 0, 0), 2)
                
                    # 取得映射到熱影像上的座標
                    lx1, ly1, lx2, ly2 = get_warped_box(orig_box, M)
                    
                    lx1, ly1 = max(0, lx1), max(0, ly1)
                    lx2, ly2 = min(thm_w, lx2), min(thm_h, ly2)
                    
                    if (lx2 - lx1) > 2 and (ly2 - ly1) > 2:
                        # 將框框往內縮 2 像素，避免周圍的豬皮或背景干擾，純取眼睛中心
                        inner = 2
                        cx1, cy1 = max(0, lx1 + inner), max(0, ly1 + inner)
                        cx2, cy2 = min(thm_w, lx2 - inner), min(thm_h, ly2 - inner)
                        
                        roi_crop = gray_thm[cy1:cy2, cx1:cx2]
                        if roi_crop.size > 0:
                            # 由於眼睛通常是臉部最高溫的區域，我們取前 10% 最亮的像素點平均值
                            p90 = np.percentile(roi_crop, 90)
                            hot_pixels = roi_crop[roi_crop >= p90]
                            pixel_intensity = hot_pixels.mean()
                            
                            # 帶入 OCR 動態溫標公式
                            temp = current_min_temp + (pixel_intensity / 255.0) * (current_max_temp - current_min_temp)
                            eye_temps_data[eye_name] = temp
                            
                            # 在熱影像上畫出映射框與溫度 (黃色)
                            cv2.rectangle(frame_thm, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                            cv2.putText(frame_thm, f"{temp:.1f} C", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
        status_bar = np.zeros((60, 1600, 3), dtype=np.uint8)
        left_temp = f"{eye_temps_data['Left Eye']:.1f}C" if eye_temps_data['Left Eye'] else "N/A"
        right_temp = f"{eye_temps_data['Right Eye']:.1f}C" if eye_temps_data['Right Eye'] else "N/A"
        
        cv2.putText(status_bar, f"Left Eye: {left_temp} | Right Eye: {right_temp}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(status_bar, "[S]: Save Data to CSV | [Q]: Quit", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        final_ui = np.vstack((combined_view, status_bar))
        
        cv2.imshow("Field Eye Temp Tracker", final_ui)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 按下 's' 將數據存入 CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            l_val = round(eye_temps_data['Left Eye'], 2) if eye_temps_data['Left Eye'] else ""
            r_val = round(eye_temps_data['Right Eye'], 2) if eye_temps_data['Right Eye'] else ""
            with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, l_val, r_val, current_max_temp, current_min_temp])
            print(f"[{timestamp}] 儲存成功！左眼: {l_val}C, 右眼: {r_val}C")

    cap_rgb.release()
    cap_thm.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
