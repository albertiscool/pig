import cv2
import os
from ultralytics import YOLO
import numpy as np

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
def extract_faces_from_video(
    video_path,
    model_path,
    output_root=os.path.join(ROOT_DIR, "dataset_split", "unknown"),
    conf_threshold=0.5,
    max_faces=5000,
    save_every_n_faces=5,
    feature_ratio=0.1
):
    """
    使用 YOLOv8 模型從影片中偵測豬臉，依照結構儲存：
        unknown/feature/<pig_id>/
        unknown/test/<pig_id>/
    隨機或平均選取 10% 臉當 feature，其餘放 test。

    Args:
        video_path (str): 影片路徑
        model_path (str): YOLO 模型 (.pt)
        output_root (str): unknown 根資料夾
        conf_threshold (float): YOLO 信心閾值
        max_faces (int): 每部影片最多儲存臉的數量
        save_every_n_faces (int): 每幾個偵測到臉的 frame 才儲存一次
        feature_ratio (float): 取多少比例的臉作為 feature (預設 0.1)
    """

    # === 取影片名稱與豬ID ===
    video_name = os.path.basename(video_path)
    pig_id = video_name.split("_")[0]  # e.g. "752-2" from "752-2_12345_可見光.mp4"

    # === 建立 feature/test 資料夾 ===
    feature_dir = os.path.join(output_root, "feature", pig_id)
    test_dir = os.path.join(output_root, "test", pig_id)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # === 載入 YOLO 模型 ===
    model = YOLO(model_path)

    # === 開啟影片 ===
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detected_frame_count = 0
    saved_faces = []

    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_path}")
        return

    print(f"🎥 開始處理影片：{video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO 偵測
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) > 0:
            detected_frame_count += 1

            # 每 N 次偵測到臉的 frame 才儲存一次
            if detected_frame_count % save_every_n_faces != 0:
                continue

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                saved_faces.append(face_crop)

                if len(saved_faces) >= max_faces:
                    break

        if len(saved_faces) >= max_faces:
            break

    cap.release()

    if len(saved_faces) == 0:
        print(f"⚠️ 沒有在影片中偵測到豬臉：{video_name}")
        return

    # === 分配 feature/test ===
    n_faces = len(saved_faces)
    n_feature = max(1, int(n_faces * feature_ratio))

    # 均勻抽樣 index，例如取 10%、等距
    step = max(1, n_faces // n_feature)
    feature_indices = set(range(0, n_faces, step))
    print(f"📊 總共偵測 {n_faces} 張臉，取 {len(feature_indices)} 張為 feature")

    # === 儲存圖片 ===
    for i, face_crop in enumerate(saved_faces):
        save_dir = feature_dir if i in feature_indices else test_dir
        save_path = os.path.join(save_dir, f"{pig_id}_{i:04d}.jpg")
        cv2.imwrite(save_path, face_crop)

    print(f"✅ 完成！共儲存 {n_faces} 張臉")
    print(f"📁 Feature：{feature_dir}")
    print(f"📁 Test：{test_dir}")


# === 使用範例 ===
if __name__ == "__main__":
    video_path = os.path.join(ROOT_DIR, "train111", "face_trained", "unknown_pig", "478-2_11410181600_可見光_正面(30秒呼吸51次).mp4")
    model_path = os.path.join(ROOT_DIR, "runs", "detect", "train60", "weights", "best.pt")

    extract_faces_from_video(
        video_path=video_path,
        model_path=model_path,
        output_root=os.path.join(ROOT_DIR, "dataset_split", "unknown"),
        conf_threshold=0.55,
        max_faces=500,
        save_every_n_faces=5,  # 每偵測到第5個有臉的 frame 才儲存
        feature_ratio=0.1      # 取10%當feature
    )