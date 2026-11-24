# -*- coding: utf-8 -*-
"""
batch_run_onsite_quad_view.py
================================================================================
🚀 批次自動推論「所有現場錄製影片 (20260722 全系列共 11 部)」生成 Quad-View 四分割影片
使用使用者 PPT 指定的純 YOLO 模型架構 (ear_mode="yolo_bbox"):
1. 可見光五官: parameters/nose_eyes_face_0817.pt (YOLOv8)
2. 可見光耳根: parameters/best_v12_eye_nose_root.pt (YOLOv12)
3. 熱成像鼻鏡: parameters/thm/best_thermal_nose.pt (YOLO)
4. 雙光對齊: 4x2 動態視差轉換矩陣 + 方案二淚腺熱點自動吸附
5. 生理看板: 呼吸率示波器 + 眼/耳/鼻即時體溫看板 + 65% 身分保護
================================================================================
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from run_quad_view_unified import run_pipeline

V3_DIR = os.path.join(PROJECT_ROOT, "v3")
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "test_video_outputs", "v4_all_onsite_quad_view")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 11 部現場錄製影片清單及其最佳起始校正影格 (RGB offset = 30 幀)
ONSITE_VIDEOS = [
    {"tag": "111318", "start_rgb": 60,  "start_thm": 30,  "desc": "現場豬隻進食/側臉 (277幀/短片)"},
    {"tag": "111352", "start_rgb": 60,  "start_thm": 30,  "desc": "現場正面近距站立 (1807幀)"},
    {"tag": "111506", "start_rgb": 230, "start_thm": 200, "desc": "現場粉紅豬正對鏡頭標準觀測 (1808幀)"},
    {"tag": "113528", "start_rgb": 180, "start_thm": 150, "desc": "現場中距站立 (876幀)"},
    {"tag": "113652", "start_rgb": 60,  "start_thm": 30,  "desc": "現場移動巡視 (696幀)"},
    {"tag": "113719", "start_rgb": 280, "start_thm": 250, "desc": "現場黃金長時序監測 (1807幀)"},
    {"tag": "113837", "start_rgb": 60,  "start_thm": 30,  "desc": "現場角落觀測 (823幀)"},
    {"tag": "113908", "start_rgb": 90,  "start_thm": 60,  "desc": "現場豬隻抬頭觀測 (1808幀)"},
    {"tag": "114107", "start_rgb": 60,  "start_thm": 30,  "desc": "現場欄杆側邊觀測 (1808幀)"},
    {"tag": "114220", "start_rgb": 60,  "start_thm": 30,  "desc": "現場豬隻進食觀測 (922幀)"},
    {"tag": "114333", "start_rgb": 60,  "start_thm": 30,  "desc": "現場正面低頭觀測 (1249幀)"},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=600, help="每部影片處理影格數 (預設 600 幀 / 20 秒)")
    parser.add_argument("--full", action="store_true", help="全片處理至結束 (忽略 --frames 上限)")
    parser.add_argument("--ear_mode", type=str, default="yolo_bbox", choices=["yolo_bbox", "best2_all"], 
                        help="YOLO 模型模式: yolo_bbox (PPT指定: 0817+best_v12) 或 best2_all")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 啟動【所有現場錄製影片 (全系列 11 部)】Quad-View 批次推論渲染任務")
    print(f"[*] 使用模型: PPT 指定純 YOLO 模型架構 ({args.ear_mode})")
    print(f"[*] 輸出目錄: {OUTPUT_BASE_DIR}")
    print(f"[*] 模式: {'全片推論至片尾' if args.full else f'每部基準 {args.frames} 幀 (~20秒)'}")
    print("=" * 80)

    total_t0 = time.time()
    results = []

    for idx, item in enumerate(ONSITE_VIDEOS, 1):
        tag = item["tag"]
        rgb_file = f"rgb_record_20260722_{tag}.mp4"
        thm_file = f"thm_record_20260722_{tag}.mp4"
        rgb_path = os.path.join(V3_DIR, rgb_file)
        thm_path = os.path.join(V3_DIR, thm_file)

        if not os.path.exists(rgb_path) or not os.path.exists(thm_path):
            print(f"[!] 警告: 找不到檔案 {rgb_path} 或 {thm_path}，跳過。")
            continue

        c_r = cv2.VideoCapture(rgb_path)
        total_len = int(c_r.get(cv2.CAP_PROP_FRAME_COUNT))
        c_r.release()

        frames_to_run = (total_len - item["start_rgb"]) if args.full else min(args.frames, total_len - item["start_rgb"])
        if frames_to_run <= 30:
            frames_to_run = total_len

        print(f"\n" + "-" * 80)
        print(f"🎬 [{idx}/{len(ONSITE_VIDEOS)}] 正在處理現場影片: 20260722_{tag} ({item['desc']})")
        print(f"    - 起始影格: RGB={item['start_rgb']}, THM={item['start_thm']} | 處理長度: {frames_to_run} 幀")
        print("-" * 80)

        t_start = time.time()
        video_tag = f"v4_site_{tag}"
        
        # 呼叫 run_quad_view_unified 執行四分割推論管線
        run_pipeline(
            rgb_path=rgb_path,
            thm_path=thm_path,
            out_dir=OUTPUT_BASE_DIR,
            video_tag=video_tag,
            start_frame_rgb=item["start_rgb"],
            start_frame_thm=item["start_thm"],
            num_frames=frames_to_run,
            fast_mode=True,
            ear_mode=args.ear_mode
        )
        t_cost = time.time() - t_start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out_video_name = f"{video_tag}_quad_view_yolo_root.mp4" if args.ear_mode == "yolo_bbox" else f"{video_tag}_quad_view_best2_all.mp4"
        out_video_file = os.path.join(OUTPUT_BASE_DIR, out_video_name)
        out_snap_file  = os.path.join(OUTPUT_BASE_DIR, out_video_name.replace(".mp4", ".jpg").replace("quad_view", "quad_snapshot"))
        out_plot_file  = os.path.join(OUTPUT_BASE_DIR, out_video_name.replace(".mp4", ".png").replace("quad_view", "quad_waveform"))

        sz_mb = os.path.getsize(out_video_file) / 1024 / 1024 if os.path.exists(out_video_file) else 0.0

        results.append({
            "tag": tag,
            "desc": item["desc"],
            "frames": frames_to_run,
            "time_sec": round(t_cost, 1),
            "fps": round(frames_to_run / max(1e-3, t_cost), 1),
            "video_path": out_video_file,
            "snap_path": out_snap_file,
            "plot_path": out_plot_file,
            "size_mb": round(sz_mb, 2)
        })

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"🎉 全部 11 部現場影片 Quad-View 渲染完成！總耗時: {total_elapsed/60:.2f} 分鐘")
    print("=" * 80)

    # 輸出彙整 Markdown 報告
    summary_md_path = os.path.join(OUTPUT_BASE_DIR, "onsite_quad_view_batch_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# 現場錄製影片 (11 部) Quad-View 純 YOLO 評測影片彙整清單\n\n")
        f.write(f"- **推論模型**: PPT 指定純 YOLO 模型管線 (`{args.ear_mode}`)\n")
        f.write(f"- **總影片數**: {len(results)} 部\n")
        f.write(f"- **總耗時**: {total_elapsed/60:.2f} 分鐘\n\n")
        f.write("| 編號 | 影片代號 | 案場情境說明 | 影格數 / 長度 | 速度 | 檔案大小 | Quad-View 影片檔連結 |\n")
        f.write("| :---: | :---: | :--- | :---: | :---: | :---: | :--- |\n")
        for i, r in enumerate(results, 1):
            v_basename = os.path.basename(r["video_path"])
            f.write(f"| {i} | `20260722_{r['tag']}` | {r['desc']} | {r['frames']} 幀 ({r['frames']/30:.1f}s) | {r['fps']} FPS | {r['size_mb']} MB | [{v_basename}](file:///{r['video_path'].replace(chr(92), '/')}) |\n")

    print(f"[✓] 彙整報告已儲存至: {summary_md_path}")

if __name__ == "__main__":
    main()
