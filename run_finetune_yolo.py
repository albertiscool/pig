# -*- coding: utf-8 -*-
"""
run_finetune_yolo.py
使用 Grounding DINO 自動標註的新資料集 (dataset_dino_new/dataset.yaml)
微調原本的 nose_eyes_face.pt 模型。
類別包含:
  0: eye
  1: face
  2: nose
"""

import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "nose_eyes_face.pt")
YAML_PATH = os.path.join(ROOT_DIR, "dataset_dino_new", "dataset.yaml")

def main():
    print("=" * 60)
    print("  🔥 開始微調 YOLO 模型 (nose_eyes_face.pt)")
    print("=" * 60)
    print(f"[*] 原本模型路徑: {MODEL_PATH}")
    print(f"[*] 資料集設定檔: {YAML_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[錯誤] 找不到原始權重: {MODEL_PATH}")
        return

    if not os.path.exists(YAML_PATH):
        print(f"[錯誤] 找不到資料集設定檔: {YAML_PATH}")
        return

    # 1. 載入原本訓練好的模型權重
    model = YOLO(MODEL_PATH)

    # 2. 開始微調 (Fine-tuning)
    print("[*] 開始執行微調訓練 (30 Epochs, lr0=0.001)...")
    results = model.train(
        data=YAML_PATH,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.001,          # 微調專用小學習率
        lrf=0.01,
        freeze=10,          # 凍結前 10 層 Backbone
        device=0,           # GPU 裝置
        name="pig_dino_finetune_3labels",
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("🎉 微調完成！")
    print("  - 新的模型權重已儲存於: runs/detect/pig_dino_finetune_3labels/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
