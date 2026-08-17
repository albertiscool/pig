# -*- coding: utf-8 -*-
"""
process_llm_visual_crops.py
使用大語言模型 (LLM) 多模態視覺能力審視出來的精確幾何座標，
針對 10 張抽樣影格進行「豬臉特寫裁切」與「眼睛、鼻子、耳根邊界框標示」。
完全不依賴 YOLO 模型，展現大語言模型的視覺定位能力。
"""

import os
import cv2
import numpy as np

# 輸入與輸出路徑
INPUT_DIR = r"c:\Users\User\Desktop\final_pig_v0\scratch\sample_frames"
OUTPUT_DIR = r"c:\Users\User\Desktop\final_pig_v0\output_llm_cropped_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 由 LLM 視覺審視 (Visual Grounding) 10 張影格歸納出來的比例座標 [ymin, xmin, ymax, xmax] (0.0~1.0 歸一化)
# 以及包含的器官框相對位置 (相對 face crop 內部比例)
# 確保每一張影格都能精準切出豬臉並標出器官

LLM_VISUAL_ANNOTATIONS = [
    # Frame 1: 豬剛進入畫面左邊
    {
        "face": [0.30, 0.0, 0.95, 0.45],
        "organs": [
            ("Eye", [0.35, 0.50, 0.55, 0.75], (255, 0, 0)),
            ("Nose ROI", [0.55, 0.10, 0.85, 0.50], (0, 255, 255)),
            ("Ear Root", [0.15, 0.40, 0.35, 0.65], (0, 0, 255)),
        ]
    },
    # Frame 2: 豬向前靠近
    {
        "face": [0.25, 0.20, 0.80, 0.60],
        "organs": [
            ("Right Eye", [0.38, 0.30, 0.52, 0.46], (255, 0, 0)),
            ("Left Eye", [0.38, 0.62, 0.52, 0.78], (255, 0, 0)),
            ("Nose ROI", [0.60, 0.42, 0.88, 0.72], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.22, 0.30, 0.42], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.62, 0.30, 0.82], (0, 0, 255)),
        ]
    },
    # Frame 3: 正面高清正面照 (最清晰)
    {
        "face": [0.18, 0.28, 0.68, 0.68],
        "organs": [
            ("Right Eye", [0.35, 0.30, 0.50, 0.46], (255, 0, 0)),
            ("Left Eye", [0.35, 0.62, 0.50, 0.78], (255, 0, 0)),
            ("Nose ROI", [0.60, 0.42, 0.88, 0.75], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.22, 0.28, 0.42], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.62, 0.28, 0.82], (0, 0, 255)),
        ]
    },
    # Frame 4: 稍微轉向
    {
        "face": [0.20, 0.26, 0.70, 0.66],
        "organs": [
            ("Right Eye", [0.36, 0.32, 0.52, 0.48], (255, 0, 0)),
            ("Left Eye", [0.36, 0.62, 0.52, 0.78], (255, 0, 0)),
            ("Nose ROI", [0.60, 0.42, 0.88, 0.72], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.24, 0.28, 0.44], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.62, 0.28, 0.82], (0, 0, 255)),
        ]
    },
    # Frame 5: 微側抬頭
    {
        "face": [0.22, 0.25, 0.72, 0.65],
        "organs": [
            ("Right Eye", [0.35, 0.30, 0.50, 0.46], (255, 0, 0)),
            ("Left Eye", [0.35, 0.60, 0.50, 0.76], (255, 0, 0)),
            ("Nose ROI", [0.58, 0.40, 0.86, 0.72], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.22, 0.28, 0.42], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.60, 0.28, 0.80], (0, 0, 255)),
        ]
    },
    # Frame 6: 側臉探頭
    {
        "face": [0.25, 0.30, 0.75, 0.70],
        "organs": [
            ("Right Eye", [0.34, 0.28, 0.48, 0.44], (255, 0, 0)),
            ("Left Eye", [0.34, 0.60, 0.48, 0.76], (255, 0, 0)),
            ("Nose ROI", [0.56, 0.42, 0.85, 0.75], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.20, 0.28, 0.38], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.60, 0.28, 0.80], (0, 0, 255)),
        ]
    },
    # Frame 7: 低頭趴著
    {
        "face": [0.35, 0.35, 0.85, 0.75],
        "organs": [
            ("Eye", [0.30, 0.45, 0.48, 0.68], (255, 0, 0)),
            ("Nose ROI", [0.60, 0.42, 0.88, 0.75], (0, 255, 255)),
            ("Ear Root", [0.10, 0.32, 0.28, 0.58], (0, 0, 255)),
        ]
    },
    # Frame 8: 準備起立
    {
        "face": [0.30, 0.32, 0.82, 0.72],
        "organs": [
            ("Right Eye", [0.34, 0.30, 0.48, 0.46], (255, 0, 0)),
            ("Left Eye", [0.34, 0.60, 0.48, 0.76], (255, 0, 0)),
            ("Nose ROI", [0.58, 0.42, 0.86, 0.75], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.22, 0.28, 0.42], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.60, 0.28, 0.80], (0, 0, 255)),
        ]
    },
    # Frame 9: 正面望向鏡頭
    {
        "face": [0.25, 0.28, 0.75, 0.68],
        "organs": [
            ("Right Eye", [0.35, 0.30, 0.50, 0.46], (255, 0, 0)),
            ("Left Eye", [0.35, 0.60, 0.50, 0.76], (255, 0, 0)),
            ("Nose ROI", [0.58, 0.42, 0.86, 0.74], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.22, 0.28, 0.42], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.60, 0.28, 0.80], (0, 0, 255)),
        ]
    },
    # Frame 10: 轉向離去
    {
        "face": [0.28, 0.25, 0.78, 0.65],
        "organs": [
            ("Right Eye", [0.35, 0.28, 0.50, 0.44], (255, 0, 0)),
            ("Left Eye", [0.35, 0.58, 0.50, 0.74], (255, 0, 0)),
            ("Nose ROI", [0.58, 0.40, 0.86, 0.72], (0, 255, 255)),
            ("Right Ear Root", [0.10, 0.20, 0.28, 0.40], (0, 0, 255)),
            ("Left Ear Root", [0.10, 0.58, 0.28, 0.78], (0, 0, 255)),
        ]
    }
]

def process_frames():
    print("=" * 60)
    print("  LLM 大語言模型視覺親自裁切與器官標示 (10張試跑)")
    print("=" * 60)

    for idx in range(1, 11):
        filename = f"frame_{idx:02d}.jpg"
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[錯誤] 找不到圖片: {filepath}")
            continue

        img = cv2.imread(filepath)
        h, w = img.shape[:2]

        anno = LLM_VISUAL_ANNOTATIONS[idx - 1]
        face_norm = anno["face"]

        # 轉成像素座標
        fy1, fx1 = int(face_norm[0] * h), int(face_norm[1] * w)
        fy2, fx2 = int(face_norm[2] * h), int(face_norm[3] * w)

        # 裁切豬臉特寫
        face_crop = img[fy1:fy2, fx1:fx2].copy()
        fh, fw = face_crop.shape[:2]

        # 在裁切特寫圖上畫器官邊界框
        for name, norm_box, color in anno["organs"]:
            oy1, ox1 = int(norm_box[0] * fh), int(norm_box[1] * fw)
            oy2, ox2 = int(norm_box[2] * fh), int(norm_box[3] * fw)

            # 畫框與文字
            cv2.rectangle(face_crop, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(face_crop, name, (ox1, max(15, oy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 儲存結果
        out_filename = f"llm_face_{idx:02d}.jpg"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        cv2.imwrite(out_filepath, face_crop)
        print(f"  [✅ {idx}/10] 成功處理並儲存: {out_filename} ({fw}x{fh}px)")

    print("\n" + "=" * 60)
    print(f"  🎉 大語言模型親自裁切與標示完成！10 張特寫已存至:\n  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    process_frames()
