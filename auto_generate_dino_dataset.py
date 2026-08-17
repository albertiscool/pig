# -*- coding: utf-8 -*-
"""
auto_generate_dino_dataset.py
自動處理全部 12 部 RGB 現場錄製影片：
1. 均勻對每部影片抽樣約 30~35 影格 (總計約 400 張圖)
2. 使用 Grounding DINO + NMS 進行零樣本自動標註
   指定標籤與類別 ID:
   0: eye (pig eye)
   1: face (pig face)
   2: nose (pig nose)
3. 自動過濾無標註的無效影格
4. 自動按 80% : 10% : 10% 比例切分為 train / val / test 目錄
5. 自動生成對應的 dataset.yaml 設定檔
"""

import os
import glob
import random
import shutil
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 路徑設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DATASET_DIR = os.path.join(ROOT_DIR, "dataset_dino_new")

# 清理並建立目錄
if os.path.exists(OUTPUT_DATASET_DIR):
    shutil.rmtree(OUTPUT_DATASET_DIR)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "labels", split), exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "a pig eye. a pig face. a pig nose."

CLASS_ID_MAP = {
    "pig eye": 0,
    "eye": 0,
    "pig face": 1,
    "face": 1,
    "pig nose": 2,
    "nose": 2
}

def main():
    print("=" * 60)
    print("  🚀 開始自動建立 Grounding DINO 微調資料集 (類別: 0:eye, 1:face, 2:nose)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 運算裝置 (Device): {device}")

    # 尋找所有 rgb_record_*.mp4 影片
    video_paths = sorted(glob.glob(os.path.join(ROOT_DIR, "rgb_record_*.mp4")))
    if not video_paths:
        print("[錯誤] 在根目錄下找不到 rgb_record_*.mp4 影片！")
        return

    print(f"[*] 找到 {len(video_paths)} 部現場 RGB 影片：")
    for v in video_paths:
        print(f"  - {os.path.basename(v)}")

    print(f"\n[*] 載入 Grounding DINO 模型: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    temp_pool = []  # 存放 (img_path, txt_path)

    temp_dir = os.path.join(OUTPUT_DATASET_DIR, "temp_pool")
    os.makedirs(temp_dir, exist_ok=True)

    frames_per_video = 33  # 每部影片抽約 33 張 => 12 * 33 ≈ 400 張

    total_valid_samples = 0

    for v_idx, vid_path in enumerate(video_paths):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            cap.release()
            continue

        step = max(1, total_frames // frames_per_video)
        sample_indices = set(range(0, total_frames, step))

        print(f"\n[{v_idx+1}/{len(video_paths)}] 處理影片: {vid_name} (總影格: {total_frames}, 步長: {step})")

        current_frame = 0
        video_valid_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in sample_indices:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)

                inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

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
                            label = labels[i_idx]

                            cid = None
                            for k, c_code in CLASS_ID_MAP.items():
                                if k in label:
                                    cid = c_code
                                    break
                            
                            if cid is None:
                                continue

                            x_center = ((x1 + x2) / 2.0) / w
                            y_center = ((y1 + y2) / 2.0) / h
                            bw = (x2 - x1) / float(w)
                            bh = (y2 - y1) / float(h)
                            txt_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

                # 如果有成功偵測到目標，存入 temp pool
                if len(txt_lines) > 0:
                    sample_name = f"{vid_name}_f{current_frame:04d}"
                    img_out_path = os.path.join(temp_dir, f"{sample_name}.jpg")
                    txt_out_path = os.path.join(temp_dir, f"{sample_name}.txt")

                    cv2.imwrite(img_out_path, frame)
                    with open(txt_out_path, "w", encoding="utf-8") as f:
                        f.writelines(txt_lines)

                    temp_pool.append((img_out_path, txt_out_path, sample_name))
                    video_valid_count += 1
                    total_valid_samples += 1

            current_frame += 1

        cap.release()
        print(f"  └─ 影片處理完成，有效標註樣本數: {video_valid_count} 張")

    print("\n" + "=" * 60)
    print(f"[*] 抽樣標註完成！共獲得 {total_valid_samples} 張有效標註樣本。")
    print("[*] 開始劃分 Train (80%) / Val (10%) / Test (10%)...")

    # 打亂數據以均勻隨機劃分
    random.seed(42)
    random.shuffle(temp_pool)

    n_total = len(temp_pool)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_samples = temp_pool[:n_train]
    val_samples = temp_pool[n_train:n_train + n_val]
    test_samples = temp_pool[n_train + n_val:]

    def copy_samples(samples, split_name):
        for img_p, txt_p, name in samples:
            target_img = os.path.join(OUTPUT_DATASET_DIR, "images", split_name, f"{name}.jpg")
            target_txt = os.path.join(OUTPUT_DATASET_DIR, "labels", split_name, f"{name}.txt")
            shutil.copy(img_p, target_img)
            shutil.copy(txt_p, target_txt)

    copy_samples(train_samples, "train")
    copy_samples(val_samples, "val")
    copy_samples(test_samples, "test")

    # 清理臨時池
    shutil.rmtree(temp_dir)

    # 建立 dataset.yaml
    yaml_dataset_path = OUTPUT_DATASET_DIR.replace('\\', '/')
    yaml_content = f"""path: {yaml_dataset_path}
train: images/train
val: images/val
test: images/test

names:
  0: eye
  1: face
  2: nose
"""
    yaml_path = os.path.join(OUTPUT_DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("=" * 60)
    print("🎉 資料集建置完成！詳細統計如下：")
    print(f"  - 訓練集 (Train): {len(train_samples)} 張")
    print(f"  - 驗證集 (Val):   {len(val_samples)} 張")
    print(f"  - 測試集 (Test):  {len(test_samples)} 張")
    print(f"  - 總圖片數:       {n_total} 張")
    print(f"  - 資料集目錄:     {OUTPUT_DATASET_DIR}")
    print(f"  - YOLO 導出檔:    {yaml_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
