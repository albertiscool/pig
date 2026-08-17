# -*- coding: utf-8 -*-
"""
run_real_grounding_dino.py
使用 IDEA-Research 正宗的 Grounding DINO 模型 (Transformers)
進行真正的 Zero-Shot Open-Vocabulary 物件偵測測試：
提示詞 Text Prompt: "a pig eye. a pig snout. a pig ear. a pig face."
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
INPUT_DIR = os.path.join(ROOT_DIR, "scratch", "sample_frames")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_grounding_dino_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grounding DINO 模型名稱 (使用最熱門的 grounding-dino-tiny / base)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 文字 Prompt (Grounding DINO 規範：各名詞間使用句號 . 分隔)
TEXT_PROMPT = "a pig eye. a pig snout. a pig ear. a pig face."

# 色彩繪製標籤
COLOR_MAP = {
    "pig eye": (255, 0, 0),     # 藍色
    "pig snout": (0, 255, 255), # 黃色
    "pig ear": (0, 0, 255),     # 紅色
    "pig face": (0, 255, 0)     # 綠色
}

def main():
    print("=" * 60)
    print("  正宗 Grounding DINO (IDEA-Research) Zero-Shot 測試")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用運算裝置: {device}")

    print(f"[*] 自 Hugging Face 載入 Grounding DINO 模型: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    print(f"[*] 設定提示詞 Prompt: '{TEXT_PROMPT}'\n")

    for i in range(1, 11):
        filename = f"frame_{i:02d}.jpg"
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[錯誤] 找不到圖片: {filepath}")
            continue

        image_pil = Image.open(filepath).convert("RGB")
        w, h = image_pil.size

        # 模型推論預處理
        inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)

        start_t = time.time()
        with torch.no_grad():
            outputs = model(**inputs)

        # 後處理解析結果
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(h, w)]
        )[0]

        elapsed = time.time() - start_t

        # 載入 OpenCV 圖片進行繪圖
        cv_img = cv2.imread(filepath)

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        # 使用 NMS (Non-Maximum Suppression) 過濾重複重疊的框
        if len(boxes) > 0:
            # 將 boxes 轉為 [x, y, w, h] 格式供 NMSBoxes 使用
            boxes_xywh = []
            for b in boxes:
                x1, y1, x2, y2 = b
                boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold=0.20, nms_threshold=0.35)

            det_count = 0
            if len(indices) > 0:
                for idx_item in indices:
                    i_idx = idx_item[0] if isinstance(idx_item, (list, np.ndarray)) else idx_item
                    x1, y1, x2, y2 = map(int, boxes[i_idx])
                    score = scores[i_idx]
                    label = labels[i_idx]
                    det_count += 1

                    # 決定顏色
                    color = (0, 255, 255)
                    for k, c in COLOR_MAP.items():
                        if k in label:
                            color = c
                            break

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cv_img, f"{label} {score:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        out_name = f"dino_real_{i:02d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, cv_img)

        print(f"  [✅ {i}/10] 處理 {filename} -> 抓到 {det_count} 個標籤 (用時: {elapsed:.2f}s)")

    print("\n" + "=" * 60)
    print(f"  🎉 正宗 Grounding DINO 測試完成！10 張標示圖片已存至:\n  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
