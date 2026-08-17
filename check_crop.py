import cv2
import pandas as pd

# 讀取剛剛產生的 CSV
csv_path = r"data/test_output/roi_boxes.csv"
video_path = r"data\test_vid\478-2_11410181559_熱成像_正面(30秒呼吸46次).mp4"

try:
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if ret:
        print(f"🎥 熱成像影片原始解析度: {frame.shape[1]} x {frame.shape[0]} (寬x高)")
        
        row = df.iloc[0]
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        print(f"✂️ 嘗試裁切的座標: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # 進行裁切
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            print("❌ 慘了！截圖為空，座標完全超出了影片邊界！")
        else:
            cv2.imwrite("debug_crop.jpg", crop)
            print("✅ 截圖成功！請打開資料夾看 'debug_crop.jpg'，確認它是不是豬鼻子。")
            # 算一下平均亮度
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            print(f"🌡️ 該區域的平均亮度(溫度)為: {gray.mean():.2f} (如果是 0.00 代表全黑)")
    else:
        print("❌ 無法讀取熱成像影片，請檢查路徑。")
except Exception as e:
    print(f"❌ 發生錯誤: {e}")