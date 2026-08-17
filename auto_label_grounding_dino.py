# -*- coding: utf-8 -*-
"""
auto_label_grounding_dino.py
使用 Zero-Shot (YOLO-World / Grounding Open-Vocabulary) 模型進行自動標註：
1. 輸入提示詞 Prompt: ["pig eye", "pig snout", "pig ear", "pig face"]
2. 針對測試影片進行 Zero-Shot 自動偵測
3. 輸出包含彩色標籤框之影片與圖片
4. 自動導出標準 YOLO 格式 .txt 標註檔 (x_center, y_center, width, height)
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLOWorld

# 路徑設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_auto_labeled")
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "annotated_images")
TXT_OUT_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(TXT_OUT_DIR, exist_ok=True)

# 測試影片路徑
TEST_VIDEO_PATH = os.path.join(ROOT_DIR, "rgb_record_20260722_113719.mp4")

# 定義 Zero-Shot 提示詞與類別對照 (英文為 AI 文字理解提示，中文為顯示標籤)
CLASSES_PROMPT = ["pig eye", "pig snout", "pig ear", "pig face"]
CLASS_NAME_MAP = {
    "pig eye": "Eye",
    "pig snout": "Nose/Snout",
    "pig ear": "Ear",
    "pig face": "Face"
}

# 類別色彩表 (BGR)
COLOR_MAP = {
    "pig eye": (255, 0, 0),     # 藍色 (眼睛)
    "pig snout": (0, 255, 255), # 黃色 (鼻子)
    "pig ear": (0, 0, 255),     # 紅色 (耳朵)
    "pig face": (0, 255, 0)     # 綠色 (豬臉)
}

def main():
    print("=" * 60)
    print("  Zero-Shot (YOLO-World / Grounding) 自動標註與資料庫測試")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用運算裝置 (Device): {device}")

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[錯誤] 找不到測試影片: {TEST_VIDEO_PATH}")
        return

    print(f"[*] 載入 Zero-Shot 開放詞彙模型 (yolov8m-worldv2.pt)...")
    # 載入 YOLO-World 預訓練模型
    model = YOLOWorld("yolov8m-worldv2.pt")

    # 設定自訂提示詞 (Set Custom Prompts)
    print(f"[*] 設定自動標註 Prompt 類別: {CLASSES_PROMPT}...")
    model.set_classes(CLASSES_PROMPT)

    # 開啟測試影片
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = os.path.join(OUTPUT_DIR, "auto_labeled_20260722_113719.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    print(f"[*] 開始處理測試影片: {os.path.basename(TEST_VIDEO_PATH)}")
    print(f"  - 影片規格: {w}x{h}, {fps:.1f} FPS, 總影格數: {total_frames}")

    frame_count = 0
    saved_images_count = 0
    start_time = time.time()

    # 門檻設定
    CONF_THRES = 0.20  # Zero-Shot 信心門檻

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis_frame = frame.copy()

        # 推論
        results = model.predict(frame, imgsz=640, conf=CONF_THRES, verbose=False, device=device)[0]

        txt_lines = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                prompt_name = CLASSES_PROMPT[cls_id] if cls_id < len(CLASSES_PROMPT) else f"cls_{cls_id}"
                display_name = CLASS_NAME_MAP.get(prompt_name, prompt_name)
                color = COLOR_MAP.get(prompt_name, (0, 255, 255))

                # 取得 bounding box (x1, y1, x2, y2)
                b = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = b

                # 畫在影片畫面上
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{display_name} {conf:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 轉為 YOLO 標準 txt 標註格式 (x_center, y_center, width, height 歸一化)
                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                txt_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        # 寫入輸出影片
        writer.write(vis_frame)

        # 每隔 15 影格抽取一張作為訓練/預覽圖片與匯出對應 .txt 檔
        if frame_count % 15 == 0:
            saved_images_count += 1
            base_name = f"frame_{saved_images_count:04d}"
            img_save_path = os.path.join(IMG_OUT_DIR, f"{base_name}.jpg")
            txt_save_path = os.path.join(TXT_OUT_DIR, f"{base_name}.txt")

            cv2.imwrite(img_save_path, vis_frame)
            with open(txt_save_path, "w", encoding="utf-8") as f:
                f.writelines(txt_lines)

        if frame_count % 60 == 0 or frame_count == total_frames:
            sys.stdout.write(f"\r  進度: {frame_count}/{total_frames} 影格 (已自動標註並抽樣 {saved_images_count} 張照片/txt檔)")
            sys.stdout.flush()

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    print(f"\n\n" + "=" * 60)
    print("  🎉 Zero-Shot 自動標註測試完成！")
    print(f"  - 處理時間: {elapsed:.1f} 秒 ({frame_count / max(1, elapsed):.1f} FPS)")
    print(f"  - 標註影片儲存於: {out_video_path}")
    print(f"  - 抽樣標註圖片儲存於: {IMG_OUT_DIR}")
    print(f"  - YOLO .txt 標註檔儲存於: {TXT_OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
