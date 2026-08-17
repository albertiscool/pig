import matplotlib
matplotlib.use('Agg')  # 🌟 加入這行！強制使用無介面背景渲染引擎

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
# --- 參數設定區 (請確認路徑正確) ---
csv_path = os.path.join(ROOT_DIR, "data", "test_output", "roi_boxes.csv")
video_path = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

def plot_raw_signal():
    print("讀取資料中...")
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    
    raw_signals = []
    
    for index, row in df.iterrows():
        ret, frame = cap.read()
        if not ret:
            break
            
        x1, y1, x2, y2 = int(row['x1_th']), int(row['y1_th']), int(row['x2_th']), int(row['y2_th'])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
            raw_signals.append(np.mean(roi_gray))
        else:
            raw_signals.append(0)
            
    cap.release()
    
    print("正在繪製波浪圖...")
    plt.figure(figsize=(12, 4))
    plt.plot(raw_signals, label='Nose ROI Mean Pixel', color='blue', linewidth=1.5)
    plt.title("Raw Respiratory Signal (Before FFT/Filter)")
    plt.xlabel("Frame")
    plt.ylabel("Mean Pixel Value (0-255)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_img = "debug_signal_plot.png"
    plt.savefig(out_img)
    print(f"✅ 繪製完成！圖片已儲存為 {out_img}")
    
    # 🌟 記得把 plt.show() 刪掉或註解掉！
    # plt.show() 

if __name__ == "__main__":
    plot_raw_signal()