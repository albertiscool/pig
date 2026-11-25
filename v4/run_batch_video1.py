# -*- coding: utf-8 -*-
"""
run_batch_video1.py
================================================================================
批次執行 C:\\Users\\User\\Desktop\\final_pig_v0\\v3\\data\\test_vid\\video1 下的所有影片對
- 自動掃描 0930、1016、1018 所有配對影片 (可見光 + 熱成像)
- 若已有四分割影片結果則自動略過，未跑過的自動執行
- 採用 YOLO11-Pose 耳根幾何模式 + 4x2 智慧對齊與熱成像淚腺自動吸附微調
- 輸出四分割 MP4、快照 JPG、呼吸波形 PNG、逐幀生理 CSV、結算總表 TXT/JSON
================================================================================
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from run_quad_view_unified import run_pipeline

BASE_DIR = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "video1")
OUT_BASE = os.path.join(SCRIPT_DIR, "test_video_outputs")

VIDEO_TASKS = [
    {
        "tag": "Pig_388-4",
        "folder": "0930",
        "rgb_name": "388-4(前30秒)_11409301550_可見光_正面.mp4",
        "thm_name": "388-4(前30秒)_11409301550_熱成像_正面.mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_388_4_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_388_4_quad_view", "Pig_388-4_quad_view.mp4"),
        "frames": 900,
        "note": "388-4 前30秒 (900幀)"
    },
    {
        "tag": "Pig_471-4",
        "folder": "1016",
        "rgb_name": "471-4_11410161521_可見光_正面(10秒呼吸21次).mp4",
        "thm_name": "471-4_11410161521_熱成像_正面(10秒呼吸21次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_471_4_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_471_4_quad_view", "Pig_471-4_quad_view.mp4"),
        "frames": 900,
        "note": "471-4 10秒呼吸21次 (30秒900幀)"
    },
    {
        "tag": "Pig_472-8",
        "folder": "1016",
        "rgb_name": "472-8_11410161509_可見光_正面(30秒呼吸60次).mp4",
        "thm_name": "472-8_11410161509_熱成像_正面(30秒呼吸60次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_472_8_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_472_8_quad_view", "Pig_472-8_quad_view.mp4"),
        "frames": 900,
        "note": "472-8 30秒呼吸60次 (900幀)"
    },
    {
        "tag": "Pig_478-2_1016",
        "folder": "1016",
        "rgb_name": "478-2_11410161457_可見光_正面(30秒呼吸68次).mp4",
        "thm_name": "478-2_11410161457_熱成像_正面(30秒呼吸68次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_478_2_1016_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_478_2_1016_quad_view", "Pig_478-2_1016_quad_view.mp4"),
        "frames": 900,
        "note": "478-2 1016場次 30秒呼吸68次 (900幀)"
    },
    {
        "tag": "Pig_472-10",
        "folder": "1018",
        "rgb_name": "472-10_11410181618_可見光_正面(30秒呼吸56次).mp4",
        "thm_name": "472-10_11410181618_熱成像_正面(30秒呼吸56次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_472_10_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_472_10_quad_view", "Pig_472-10_quad_view.mp4"),
        "frames": 900,
        "note": "472-10 1018場次 30秒呼吸56次 (先前已跑過)"
    },
    {
        "tag": "Pig_484-6",
        "folder": "1018",
        "rgb_name": "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4",
        "thm_name": "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_484_6_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_484_6_quad_view", "Pig_484-6_quad_view.mp4"),
        "frames": 900,
        "note": "484-6 1018場次 30秒呼吸34次 (先前已跑過)"
    },
    {
        "tag": "Pig_484-6_51",
        "folder": "1018",
        "rgb_name": "484-6_11410181547_可見光_正面(30秒呼吸51次).mp4",
        "thm_name": "484-6_11410181547_熱成像_正面(30秒呼吸51次).mp4",
        "out_dir": os.path.join(OUT_BASE, "v4_484_6_51_quad_view"),
        "check_file": os.path.join(OUT_BASE, "v4_484_6_51_quad_view", "Pig_484-6_51_quad_view.mp4"),
        "frames": 900,
        "note": "484-6 1018場次 30秒呼吸51次 (先前已跑過)"
    }
]

def main():
    print("=" * 80)
    print("🚀 啟動 test_vid/video1 全自動批次推論流水線")
    print(f"[*] 掃描目錄: {BASE_DIR}")
    print(f"[*] 任務總數: {len(VIDEO_TASKS)} 組影片")
    print("=" * 80)

    for i, task in enumerate(VIDEO_TASKS, 1):
        rgb_path = os.path.join(BASE_DIR, "video", task["folder"], task["rgb_name"])
        thm_path = os.path.join(BASE_DIR, "video", task["folder"], task["thm_name"])
        
        # 容錯處理：如果 video1 下沒有 video/ 子目錄直接放日期
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(BASE_DIR, task["folder"], task["rgb_name"])
            thm_path = os.path.join(BASE_DIR, task["folder"], task["thm_name"])
            
        print(f"\n[{i}/{len(VIDEO_TASKS)}] 檢查: {task['tag']} ({task['note']})")
        if not os.path.exists(rgb_path) or not os.path.exists(thm_path):
            print(f"  ❌ 找不到檔案:\n    RGB: {rgb_path}\n    THM: {thm_path}")
            continue

        if os.path.exists(task["check_file"]):
            sz = os.path.getsize(task["check_file"])
            print(f"  ⏭ 已有成果檔 ({sz/1024/1024:.1f} MB)，自動略過: {task['check_file']}")
            continue

        print(f"  ▶ 開始執行推論 (900 幀 @ 30fps = 30秒)...")
        t_start = time.time()
        run_pipeline(
            rgb_path=rgb_path,
            thm_path=thm_path,
            out_dir=task["out_dir"],
            video_tag=task["tag"],
            start_frame_rgb=0,
            start_frame_thm=0,
            num_frames=task["frames"],
            fast_mode=True,
            ear_mode="pose"
        )
        print(f"  ✓ [{task['tag']}] 執行完成！耗時: {time.time()-t_start:.1f} 秒")

    print("\n" + "=" * 80)
    print("🎉 test_vid/video1 所有影片批次處理完畢！")
    print("=" * 80)

if __name__ == "__main__":
    main()
