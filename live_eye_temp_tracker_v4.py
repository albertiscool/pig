"""
live_eye_temp_tracker_v4.py — V4 零拷貝共享記憶體版現場豬隻眼溫即時追蹤系統
特點：
- 採用 pig_shm_engine (Zero-Copy Shared Memory IPC)，徹底分離 RTSP 解碼與 AI 推論
- EasyOCR 背景進程異步執行，消滅影格停頓與延遲累積
- 支援 Windows 原生多進程執行與 Linux / 樹莓派環境
"""

import os
import sys
import time
import csv
from datetime import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from video_pipeline2 import (
    SwinTransformerP2P,
    process_yolo_detection_and_sort_points,
    get_model_warp_matrix_inference,
    get_warped_box,
    P2P_MODEL_PATH,
    YOLOV8_KEYPOINT_PATH,
    IMAGE_SIZE,
    CONF_THRESHOLD
)

from pig_shm_engine import PigZeroCopyEngine

# ==========================================
# 參數設定區 (可填入實際 RTSP 網址或測試影片)
# ==========================================
DEFAULT_RGB_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
DEFAULT_THM_URL = "rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0"

CSV_SAVE_PATH = "field_eye_temp_records_v4.csv"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pig Monitoring System V4 - Zero-Copy Live Tracker")
    parser.add_argument("--rgb", type=str, default=DEFAULT_RGB_URL, help="RGB RTSP URL or video path")
    parser.add_argument("--thm", type=str, default=DEFAULT_THM_URL, help="Thermal RTSP URL or video path")
    parser.add_argument("--no_ocr", action="store_true", help="Disable background OCR process")
    args = parser.parse_args()

    print("=" * 65)
    print("  🐷 豬隻多模態即時眼溫追蹤系統 V4 (Zero-Copy IPC Architecture)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 載入 AI 模型中... (運算裝置: {device})")
    
    detect_model = YOLO(YOLOV8_KEYPOINT_PATH)
    p2p_model = SwinTransformerP2P().to(device)
    if os.path.exists(P2P_MODEL_PATH):
        p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
        if device.type == 'cuda':
            p2p_model.half()
        p2p_model.eval()
        print("✅ Swin Transformer P2P 空間映射權重載入成功！")
    else:
        print(f"⚠️ 找不到 P2P 權重 ({P2P_MODEL_PATH})，將僅執行 YOLO 關鍵點偵測。")

    # 初始化零拷貝共享記憶體引擎
    engine = PigZeroCopyEngine(
        rgb_source=args.rgb,
        thm_source=args.thm,
        rgb_shape=(1080, 1920, 3),
        thm_shape=(1080, 1920, 3),
        name_prefix="pig_v4_live"
    )
    engine.start(enable_ocr=not args.no_ocr)

    # 初始化 CSV 記錄檔
    if not os.path.exists(CSV_SAVE_PATH):
        with open(CSV_SAVE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Left_Eye_Temp', 'Right_Eye_Temp', 'Max_Scale', 'Min_Scale', 'FPS'])

    print("\n[系統準備就緒] 零拷貝影像管線運行中。")
    print("操作快捷鍵:")
    print("  - [S] 鍵：手動將當前眼溫與溫標寫入 CSV。")
    print("  - [Q] 鍵：安全關閉多進程管線並退出。")

    last_frame_count = -1
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # 從共享記憶體中取得最新影像視圖（耗時近乎 0ms）
            frame_rgb, frame_thm, frame_count, max_temp, min_temp = engine.get_latest_frames()

            if frame_count == last_frame_count:
                # 等待新影格寫入，避免無謂消耗 CPU
                time.sleep(0.005)
                continue
            last_frame_count = frame_count

            # 計算即時推論 FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # 複製一份供 OpenCV 繪製標註與顯示（保護共享記憶體主幹）
            vis_rgb = frame_rgb.copy()
            vis_thm = frame_thm.copy()

            thm_h, thm_w = vis_thm.shape[:2]

            # ---------------------------------------------------------
            # AI 推論：YOLO 眼睛關鍵點與 Swin P2P 變形矩陣
            # ---------------------------------------------------------
            yolo_res = detect_model.predict(
                frame_rgb,
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                verbose=False,
                device=device,
                half=(device.type == 'cuda')
            )[0]
            detect_map = process_yolo_detection_and_sort_points(yolo_res)

            M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, p2p_model, device)
            gray_thm = cv2.cvtColor(frame_thm, cv2.COLOR_BGR2GRAY)

            eye_temps = {'Left Eye': None, 'Right Eye': None}

            if M is not None:
                for eye_id, eye_name in [(0, 'Left Eye'), (1, 'Right Eye')]:
                    if eye_id in detect_map:
                        orig_box = detect_map[eye_id][1]
                        cv2.rectangle(vis_rgb, (orig_box[0], orig_box[1]), (orig_box[2], orig_box[3]), (255, 120, 0), 2)
                        cv2.putText(vis_rgb, eye_name, (orig_box[0], orig_box[1] - 8), 0, 0.6, (255, 120, 0), 2)

                        lx1, ly1, lx2, ly2 = get_warped_box(orig_box, M)
                        lx1, ly1 = max(0, lx1), max(0, ly1)
                        lx2, ly2 = min(thm_w, lx2), min(thm_h, ly2)

                        if (lx2 - lx1) > 2 and (ly2 - ly1) > 2:
                            inner = 2
                            cx1, cy1 = max(0, lx1 + inner), max(0, ly1 + inner)
                            cx2, cy2 = min(thm_w, lx2 - inner), min(thm_h, ly2 - inner)
                            roi_crop = gray_thm[cy1:cy2, cx1:cx2]

                            if roi_crop.size > 0:
                                p90 = np.percentile(roi_crop, 90)
                                hot_pixels = roi_crop[roi_crop >= p90]
                                pixel_val = hot_pixels.mean()

                                temp = min_temp + (pixel_val / 255.0) * (max_temp - min_temp)
                                eye_temps[eye_name] = temp

                                cv2.rectangle(vis_thm, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                                cv2.putText(vis_thm, f"{temp:.1f} C", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # ---------------------------------------------------------
            # 渲染 UI 面板
            # ---------------------------------------------------------
            cv2.putText(vis_thm, f"OCR Max: {max_temp:.1f} C", (thm_w - 240, 40), 0, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_thm, f"OCR Min: {min_temp:.1f} C", (thm_w - 240, thm_h - 40), 0, 0.7, (255, 0, 0), 2)

            disp_rgb = cv2.resize(vis_rgb, (800, 600))
            disp_thm = cv2.resize(vis_thm, (800, 600))
            combined_view = np.hstack((disp_rgb, disp_thm))

            status_bar = np.zeros((65, 1600, 3), dtype=np.uint8)
            l_str = f"{eye_temps['Left Eye']:.1f}°C" if eye_temps['Left Eye'] else "N/A"
            r_str = f"{eye_temps['Right Eye']:.1f}°C" if eye_temps['Right Eye'] else "N/A"

            cv2.putText(status_bar, f"V4 Zero-Copy SHM | FPS: {fps:.1f} | L-Eye: {l_str} | R-Eye: {r_str}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(status_bar, "[S]: Record CSV | [Q]: Quit", (1180, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            final_dashboard = np.vstack((combined_view, status_bar))
            cv2.imshow("Pig Monitor V4 - Zero-Copy Live Tracker", final_dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                l_val = round(eye_temps['Left Eye'], 2) if eye_temps['Left Eye'] else ""
                r_val = round(eye_temps['Right Eye'], 2) if eye_temps['Right Eye'] else ""
                with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, l_val, r_val, max_temp, min_temp, round(fps, 1)])
                print(f"[{timestamp}] 數據已記錄至 CSV：左眼 {l_str}，右眼 {r_str} (FPS: {fps:.1f})")

    finally:
        engine.stop()
        cv2.destroyAllWindows()
        print("[*] 系統已安全退出。")

if __name__ == "__main__":
    main()
