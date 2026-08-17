# -*- coding: utf-8 -*-
"""
run_full_grounding_dino_video.py
使用正宗 Grounding DINO (IDEA-Research/grounding-dino-tiny) + NMS 邊界框過濾
完整跑完一整隻現場錄製影片 (1807 影格)：
1. 輸出高畫質標註影片: output_grounding_dino_real/full_annotated_20260722_113719.mp4
2. 匯出每一影格的標準 YOLO .txt 標註檔至 output_grounding_dino_real/full_labels/
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
VIDEO_PATH = os.path.join(ROOT_DIR, "rgb_record_20260722_113719.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_grounding_dino_real")
LABEL_DIR = os.path.join(OUTPUT_DIR, "full_labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "a pig eye. a pig snout. a pig ear. a pig face."

# 色彩繪製標籤 (BGR)
COLOR_MAP = {
    "pig eye": (255, 0, 0),     # 藍色 (眼睛)
    "pig snout": (0, 255, 255), # 黃色 (鼻子)
    "pig ear": (0, 0, 255),     # 紅色 (耳朵)
    "pig face": (0, 255, 0)     # 綠色 (豬臉)
}

CLASS_ID_MAP = {
    "pig eye": 0,
    "pig snout": 1,
    "pig ear": 2,
    "pig face": 3
}

def main():
    print("=" * 60)
    print("  Grounding DINO 全影片完整 Zero-Shot 標註 (含 NMS 過濾)")
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

    out_video_path = os.path.join(OUTPUT_DIR, "full_annotated_20260722_113719.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    print(f"[*] 開始完整處理影片: {os.path.basename(VIDEO_PATH)}")
    print(f"  - 解析度: {w}x{h}, 幀率: {fps:.1f} FPS, 總影格數: {total_frames}")

    frame_count = 0
    start_total_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        cv_img = frame.copy()

        # PIL 轉換
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 預處理與模型推論
        inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 後處理
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(h, w)]
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        txt_lines = []

        # 應用 NMS 過濾重複框
        if len(boxes) > 0:
            boxes_xywh = []
            for b in boxes:
                x1, y1, x2, y2 = b
                boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold=0.20, nms_threshold=0.35)

            if len(indices) > 0:
                for idx_item in indices:
                    i_idx = idx_item[0] if isinstance(idx_item, (list, np.ndarray)) else idx_item
                    x1, y1, x2, y2 = map(int, boxes[i_idx])
                    score = scores[i_idx]
                    label = labels[i_idx]

                    # 決定顏色
                    color = (0, 255, 255)
                    cid = 1
                    for k, c in COLOR_MAP.items():
                        if k in label:
                            color = c
                            cid = CLASS_ID_MAP[k]
                            break

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cv_img, f"{label} {score:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # 轉成 YOLO txt 格式
                    x_center = ((x1 + x2) / 2.0) / w
                    y_center = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / float(w)
                    bh = (y2 - y1) / float(h)
                    txt_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        # 寫入影片
        writer.write(cv_img)

        # 寫入 txt 標註檔
        txt_path = os.path.join(LABEL_DIR, f"frame_{frame_count:04d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(txt_lines)

        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_total_time
            current_fps = frame_count / max(1, elapsed)
            sys.stdout.write(f"\r  處理進度: {frame_count}/{total_frames} 影格 ({current_fps:.1f} FPS) - 已生成 {frame_count} 個 YOLO txt 標註檔")
            sys.stdout.flush()

    cap.release()
    writer.release()
    total_elapsed = time.time() - start_total_time

    print("\n" + "=" * 60)
    print("  🎉 Grounding DINO 全影片完整處理與標註完成！")
    print(f"  - 總處理時間: {total_elapsed:.1f} 秒 (平均 {total_frames / max(1, total_elapsed):.1f} FPS)")
    print(f"  - 完整標註影片: {out_video_path}")
    print(f"  - 1807 個 YOLO .txt 標註檔儲存於: {LABEL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
