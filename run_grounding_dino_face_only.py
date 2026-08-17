# -*- coding: utf-8 -*-
"""
run_grounding_dino_face_only.py
使用 Grounding DINO 專門對另一部現場錄製影片 (rgb_record_20260722_114107.mp4)
進行「純豬臉 (pig face)」之單一目標 Zero-Shot 標註與 NMS 邊界框過濾。
"""

import os
import sys
import time
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 路徑設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 挑選另一部現場錄影檔 (20260722_114107)
VIDEO_PATH = os.path.join(ROOT_DIR, "rgb_record_20260722_114107.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_grounding_dino_face_only")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "sample_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 只針對豬臉提示詞 (Single Class Prompt)
TEXT_PROMPT = "a pig face."
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.35

# 顏色定義: 綠色框 (Pig Face)
COLOR_FACE = (0, 255, 0)

def main():
    print("=" * 60)
    print("  Grounding DINO「純豬臉 (pig face)」單一標籤測試")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 運算裝置 (Device): {device}")

    if not os.path.exists(VIDEO_PATH):
        print(f"[錯誤] 找不到影片: {VIDEO_PATH}")
        return

    print(f"[*] 載入 Grounding DINO 模型: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = os.path.join(OUTPUT_DIR, "face_only_20260722_114107.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    print(f"[*] 開始處理測試影片: {os.path.basename(VIDEO_PATH)}")
    print(f"  - 提示詞: '{TEXT_PROMPT}'")
    print(f"  - 規格: {w}x{h}, {fps:.1f} FPS, 總影格數: {total_frames}\n")

    frame_count = 0
    saved_samples = 0
    start_total_time = time.time()

    sample_interval = max(1, total_frames // 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        cv_img = frame.copy()

        # PIL 轉換
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 模型預處理與推論
        inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 後處理
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=CONF_THRESHOLD,
            text_threshold=CONF_THRESHOLD,
            target_sizes=[(h, w)]
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        txt_lines = []

        # NMS 過濾重複豬臉框
        if len(boxes) > 0:
            boxes_xywh = []
            for b in boxes:
                x1, y1, x2, y2 = b
                boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD)

            if len(indices) > 0:
                for idx_item in indices:
                    i_idx = idx_item[0] if isinstance(idx_item, (list, np.ndarray)) else idx_item
                    x1, y1, x2, y2 = map(int, boxes[i_idx])
                    score = scores[i_idx]

                    # 畫綠色豬臉框
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), COLOR_FACE, 2)
                    cv2.putText(cv_img, f"Face {score:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FACE, 2)

                    # YOLO txt 格式 (0 代表 pig face)
                    x_center = ((x1 + x2) / 2.0) / w
                    y_center = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / float(w)
                    bh = (y2 - y1) / float(h)
                    txt_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        # 寫入影片
        writer.write(cv_img)

        # 寫入 YOLO txt 檔
        txt_path = os.path.join(LABEL_DIR, f"frame_{frame_count:04d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(txt_lines)

        # 抽樣 10 張圖片預覽
        if frame_count % sample_interval == 0 or frame_count == 1:
            saved_samples += 1
            sample_img_path = os.path.join(SAMPLE_DIR, f"sample_face_{saved_samples:02d}.jpg")
            cv2.imwrite(sample_img_path, cv_img)

        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_total_time
            current_fps = frame_count / max(1, elapsed)
            sys.stdout.write(f"\r  進度: {frame_count}/{total_frames} 影格 ({current_fps:.1f} FPS)")
            sys.stdout.flush()

    cap.release()
    writer.release()
    total_elapsed = time.time() - start_total_time

    print("\n" + "=" * 60)
    print("  🎉 專用「純豬臉 (pig face)」Grounding DINO 標註完成！")
    print(f"  - 總用時: {total_elapsed:.1f} 秒 (平均 {total_frames / max(1, total_elapsed):.1f} FPS)")
    print(f"  - 豬臉標註影片: {out_video_path}")
    print(f"  - 抽樣 10 張特寫預覽檔存於: {SAMPLE_DIR}")
    print(f"  - 1808 個 YOLO 豬臉 .txt 檔存於: {LABEL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
