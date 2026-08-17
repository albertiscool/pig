# -*- coding: utf-8 -*-
"""
batch_process_recorded.py
批次處理現場錄製的 RGB 與熱成像影片：
1. 使用 parameters/nose_eyes_face.pt YOLO 偵測模型（眼睛、鼻子、豬臉框）
2. 使用 parameters/thm/swin_p2p_best_1873.pth 計算空間變形矩陣 M
3. 將 RGB 的 YOLO 偵測框空間映射（Warp）至熱成像影片上
4. 自動將處理好的 RGB 影片、熱成像影片與並排對照影片儲存至指定資料夾
"""

import os
import sys
import glob
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

# 載入主專案的核心架構與類別
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SCRIPT_DIR)

from video_pipeline2 import (
    SwinTransformerP2P,
    IMAGE_SIZE,
    CONF_THRESHOLD,
    P2P_CLS_IDS,
    YOLO_EYE_ID,
    YOLO_NOSE_ID,
    get_warped_box
)

# 定義模型與資料夾路徑
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_1873.pth")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_recorded_processed")

# 將預設門檻降至 0.15，適應現場錄製影片較小/較遠的特徵器官
CONF_THRESHOLD = 0.15

def process_yolo_detection_and_sort_points(yolo_results):
    """
    從 nose_eyes_face.pt 的偵測結果中抽取點與框供 P2P 使用
    P2P_CLS_IDS: 0=左眼, 1=右眼, 2=鼻子
    """
    eye_pts = []
    nose_pts = []
    if hasattr(yolo_results, 'boxes') and len(yolo_results.boxes) > 0:
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD:
                continue
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_pt = np.array([x_center, y_center], dtype=np.float32)
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            if cls_id == YOLO_EYE_ID:
                eye_pts.append({'pt': center_pt, 'bbox': bbox, 'x': x_center})
            elif cls_id == YOLO_NOSE_ID:
                nose_pts.append({'pt': center_pt, 'bbox': bbox})

    final_pts = {}
    if nose_pts:
        final_pts[2] = (nose_pts[0]['pt'], nose_pts[0]['bbox'])
    if len(eye_pts) >= 2:
        eye_pts.sort(key=lambda item: item['x'])
        final_pts[0] = (eye_pts[0]['pt'], eye_pts[0]['bbox'])
        final_pts[1] = (eye_pts[1]['pt'], eye_pts[1]['bbox'])
    elif len(eye_pts) == 1 and 2 in final_pts:
        single, nose = eye_pts[0], final_pts[2][0]
        final_pts[1 if single['x'] > nose[0] else 0] = (single['pt'], single['bbox'])
    elif len(eye_pts) == 1:
        final_pts[0] = (eye_pts[0]['pt'], eye_pts[0]['bbox'])

    return final_pts

def get_model_warp_matrix_inference(rgb_img, thm_img, detect_map, p2p_model, device):
    """計算 Swin P2P 仿射/單應性變形矩陣 M"""
    h_orig, w_orig = rgb_img.shape[:2]
    thm_h, thm_w = thm_img.shape[:2]

    pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
    if not pts_found:
        return None

    rgb_pts_in = [pts_found[cid] if cid in pts_found else [0.0, 0.0] for cid in P2P_CLS_IDS]
    model_in = [[p[0] * IMAGE_SIZE / w_orig, p[1] * IMAGE_SIZE / h_orig] for p in rgb_pts_in]

    t_rgb = torch.as_tensor(rgb_img).permute(2, 0, 1).to(device, non_blocking=True).float() / 255.0
    t_thm = torch.as_tensor(thm_img).permute(2, 0, 1).to(device, non_blocking=True).float() / 255.0
    if device.type == 'cuda':
        t_rgb, t_thm = t_rgb.half(), t_thm.half()

    t_rgb = F.interpolate(t_rgb.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    t_thm = F.interpolate(t_thm.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)

    t_pts = torch.tensor([model_in], dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)

    with torch.inference_mode():
        pred = p2p_model(t_rgb.unsqueeze(0), t_thm.unsqueeze(0), t_pts).cpu().float().numpy()[0]

    dst_pts = [[pred[i][0] * thm_w / IMAGE_SIZE, pred[i][1] * thm_h / IMAGE_SIZE] for i in range(3) if i in pts_found]
    src_np = np.float32([pts_found[i] for i in range(3) if i in pts_found])
    dst_np = np.float32(dst_pts)

    M = None
    if len(src_np) >= 3:
        M = np.vstack([cv2.getAffineTransform(src_np[:3], dst_np[:3]), [0, 0, 1]])
    elif len(src_np) == 2:
        M_p, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
        if M_p is not None:
            M = np.vstack([M_p, [0, 0, 1]])
    elif len(src_np) == 1:
        dx, dy = dst_np[0] - src_np[0]
        M = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    return M

def main():
    print("=" * 60)
    print("  現場錄製影片批次處理 (YOLO nose_eyes_face.pt + Swin P2P 1873)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用裝置 (Device): {device}")

    # 1. 載入模型
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"[錯誤] 找不到 YOLO 模型: {YOLO_MODEL_PATH}")
        return
    if not os.path.exists(P2P_MODEL_PATH):
        print(f"[錯誤] 找不到 Swin P2P 模型: {P2P_MODEL_PATH}")
        return

    print(f"[*] 載入 YOLO 模型: {YOLO_MODEL_PATH}...")
    detect_model = YOLO(YOLO_MODEL_PATH)

    print(f"[*] 載入 Swin P2P 模型: {P2P_MODEL_PATH}...")
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
    if device.type == 'cuda':
        p2p_model.half()
    p2p_model.eval()

    # 2. 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] 處理結果將儲存至: {OUTPUT_DIR}\n")

    # 3. 搜尋根目錄下所有現場錄製的影片對
    rgb_files = sorted(glob.glob(os.path.join(ROOT_DIR, "rgb_record_*.mp4")))
    if not rgb_files:
        print("[提示] 根目錄下找不到 rgb_record_*.mp4 影片檔案！")
        return

    video_pairs = []
    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        timestamp = filename.replace("rgb_record_", "").replace(".mp4", "")
        thm_path = os.path.join(ROOT_DIR, f"thm_record_{timestamp}.mp4")
        if os.path.exists(thm_path):
            video_pairs.append((timestamp, rgb_path, thm_path))

    print(f"[*] 共找到 {len(video_pairs)} 組成對的現場錄製影片。")

    # 定義類別色彩表 (BGR)
    COLOR_MAP = {
        0: (255, 0, 0),     # Eye: 藍色
        1: (0, 255, 0),     # Face: 綠色
        2: (0, 255, 255),   # Nose: 黃色
    }

    for idx, (timestamp, rgb_path, thm_path) in enumerate(video_pairs, 1):
        print(f"\n[{idx}/{len(video_pairs)}] 正在處理時間戳: {timestamp}")
        print(f"  - RGB: {os.path.basename(rgb_path)}")
        print(f"  - THM: {os.path.basename(thm_path)}")

        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_thm = cv2.VideoCapture(thm_path)

        if not cap_rgb.isOpened() or not cap_thm.isOpened():
            print("  [錯誤] 無法開啟影片檔，跳過。")
            continue

        w_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0

        w_thm = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_thm = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_thm = cap_thm.get(cv2.CAP_PROP_FPS) or 30.0

        total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

        out_rgb_path = os.path.join(OUTPUT_DIR, f"processed_rgb_{timestamp}.mp4")
        out_thm_path = os.path.join(OUTPUT_DIR, f"processed_thm_{timestamp}.mp4")
        out_comb_path = os.path.join(OUTPUT_DIR, f"processed_combined_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_rgb = cv2.VideoWriter(out_rgb_path, fourcc, fps_rgb, (w_rgb, h_rgb))
        writer_thm = cv2.VideoWriter(out_thm_path, fourcc, fps_thm, (w_thm, h_thm))
        writer_comb = cv2.VideoWriter(out_comb_path, fourcc, fps_rgb, (1600, 600))

        frame_count = 0
        start_t = time.time()

        while True:
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_thm, frame_thm = cap_thm.read()

            if not ret_rgb or not ret_thm:
                break

            frame_count += 1

            # 複製畫面進行繪製
            vis_rgb = frame_rgb.copy()
            vis_thm = frame_thm.copy()

            # YOLO 推論
            yolo_res = detect_model.predict(frame_rgb, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device)[0]

            # 抓點算 P2P 變形矩陣 M
            detect_map = process_yolo_detection_and_sort_points(yolo_res)
            M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, p2p_model, device)

            # 在 RGB 畫 YOLO 框，並在 THM 畫變形映射框
            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                for box in yolo_res.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < CONF_THRESHOLD:
                        continue

                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = detect_model.names.get(cls_id, f"cls_{cls_id}")
                    color = COLOR_MAP.get(cls_id, (0, 255, 255))

                    b = box.xyxy[0].cpu().numpy().astype(int)
                    rx1, ry1, rx2, ry2 = b

                    # 1. 畫在 RGB 上
                    cv2.rectangle(vis_rgb, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(vis_rgb, f"{cls_name} {conf:.2f}", (rx1, max(0, ry1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 2. 透過 M 變形映射到 THM 上
                    if M is not None:
                        wx1, wy1, wx2, wy2 = get_warped_box(b, M)
                        wx1, wy1 = max(0, wx1), max(0, wy1)
                        wx2, wy2 = min(w_thm, wx2), min(h_thm, wy2)

                        if (wx2 - wx1) > 2 and (wy2 - wy1) > 2:
                            cv2.rectangle(vis_thm, (wx1, wy1), (wx2, wy2), color, 2)
                            cv2.putText(vis_thm, f"{cls_name} (mapped)", (wx1, max(0, wy1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 寫入單個 RGB 與 THM 影片
            writer_rgb.write(vis_rgb)
            writer_thm.write(vis_thm)

            # 製作並排影片 (Resize 為 800x600，拼成 1600x600)
            vis_rgb_800 = cv2.resize(vis_rgb, (800, 600))
            vis_thm_800 = cv2.resize(vis_thm, (800, 600))
            combined = np.hstack((vis_rgb_800, vis_thm_800))
            writer_comb.write(combined)

            if frame_count % 30 == 0 or frame_count == total_frames:
                sys.stdout.write(f"\r  進度: {frame_count}/{total_frames} 幀")
                sys.stdout.flush()

        cap_rgb.release()
        cap_thm.release()
        writer_rgb.release()
        writer_thm.release()
        writer_comb.release()

        elapsed_sec = time.time() - start_t
        print(f"\n  ✅ 處理完成！用時: {elapsed_sec:.1f}秒 ({frame_count / max(1, elapsed_sec):.1f} FPS)")

    print("\n" + "=" * 60)
    print("  🎉 全部 12 組現場錄製影片批次處理完成！")
    print(f"  檔案已儲存至: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
