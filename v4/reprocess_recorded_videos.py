# -*- coding: utf-8 -*-
"""
reprocess_recorded_videos.py
================================================================================
從 recorded_videos 內已錄存的原始雙光影片 (*_rgb.mp4 + *_thm.mp4) 重新推論，
每段影片皆從第 0 幀清空呼吸訊號與體溫累計並重算 BPM、波形圖、逐幀 CSV 與四分割影片，
輸出至各錄影資料夾底下的 recalculated\ 子目錄 (不覆蓋原始檔案)。

使用範例:
  python v4/reprocess_recorded_videos.py --rec recorded_videos/rec_20261111_120047
  python v4/reprocess_recorded_videos.py --date 20261111 --min_sec 10
================================================================================
"""

import os
import sys
import glob
import argparse
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_quad_view_unified import run_pipeline


def find_video_pair(rec_dir):
    tag = os.path.basename(os.path.normpath(rec_dir))
    rgb_path = os.path.join(rec_dir, f"{tag}_rgb.mp4")
    thm_path = os.path.join(rec_dir, f"{tag}_thm.mp4")
    if not (os.path.exists(rgb_path) and os.path.exists(thm_path)):
        rgb_candidates = sorted(glob.glob(os.path.join(rec_dir, "*_rgb.mp4")))
        thm_candidates = sorted(glob.glob(os.path.join(rec_dir, "*_thm.mp4")))
        if rgb_candidates and thm_candidates:
            rgb_path = rgb_candidates[0]
            thm_path = thm_candidates[0]
        else:
            return None, None, 0, 0.0
    cap_r = cv2.VideoCapture(rgb_path)
    cap_t = cv2.VideoCapture(thm_path)
    if not cap_r.isOpened() or not cap_t.isOpened():
        cap_r.release()
        cap_t.release()
        return None, None, 0, 0.0
    n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap_r.get(cv2.CAP_PROP_FPS) or 25.0)
    cap_r.release()
    cap_t.release()
    n_frames = min(n_r, n_t)
    dur_sec = n_frames / max(1.0, fps)
    return rgb_path, thm_path, n_frames, dur_sec


def is_already_recalculated(rec_dir):
    out_dir = os.path.join(rec_dir, "recalculated")
    if not os.path.isdir(out_dir):
        return False
    return any(fn.endswith("_vitals_summary.json") for fn in os.listdir(out_dir)) and any(
        fn.endswith("_quad_view.mp4") for fn in os.listdir(out_dir)
    )


def reprocess_folder(rec_dir, ear_mode="pose", fast_mode=True, sig_window=450, skip_existing=False):
    tag = os.path.basename(os.path.normpath(rec_dir))
    if skip_existing and is_already_recalculated(rec_dir):
        print(f"[*] 略過 (已有重算結果): {tag}")
        return True
    rgb_path, thm_path, n_frames, dur_sec = find_video_pair(rec_dir)
    if rgb_path is None or n_frames <= 0:
        print(f"[!] 略過 (無法開啟或影片損毀): {rec_dir}")
        return False
    out_dir = os.path.join(rec_dir, "recalculated")
    print(f"[*] 重新計算 {tag} | 幀數: {n_frames} 幀 ({dur_sec:.1f} 秒) -> {out_dir}")
    run_pipeline(
        rgb_path=rgb_path,
        thm_path=thm_path,
        out_dir=out_dir,
        video_tag=f"{tag}_recalc",
        start_frame_rgb=0,
        start_frame_thm=0,
        num_frames=n_frames,
        fast_mode=fast_mode,
        ear_mode=ear_mode,
        auto_register=False,
        sig_window=sig_window,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser(description="從原始錄影重算呼吸率與四分割畫面")
    parser.add_argument("--rec", type=str, nargs="*", default=None, help="指定一或多個錄影資料夾路徑")
    parser.add_argument("--root", type=str, default=os.path.join(PROJECT_ROOT, "recorded_videos"), help="recorded_videos 根目錄")
    parser.add_argument("--date", type=str, default=None, help="依日期批次重算 (例如 20261111)")
    parser.add_argument("--min_sec", type=float, default=5.0, help="批次模式下僅處理至少幾秒以上的片段 (預設 5 秒)")
    parser.add_argument("--ear_mode", type=str, default="pose", choices=["pose", "yolo_bbox"], help="耳根偵測模式 (預設 pose)")
    parser.add_argument("--full", action="store_true", help="啟用全幀推論")
    parser.add_argument("--sig_window", type=int, default=450, help="呼吸率滑動視窗幀數 (0 代表整支影片累積)")
    parser.add_argument("--skip_existing", action="store_true", help="略過已產生 recalculated 的資料夾")
    args = parser.parse_args()

    targets = []
    if args.rec:
        for p in args.rec:
            full_p = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            if os.path.isdir(full_p):
                targets.append(full_p)
            else:
                print(f"[!] 找不到資料夾: {p}")
    else:
        prefix = f"rec_{args.date}_" if args.date else "rec_"
        for name in sorted(os.listdir(args.root)):
            if name.startswith(prefix):
                full_p = os.path.join(args.root, name)
                if os.path.isdir(full_p):
                    _, _, n_frames, dur_sec = find_video_pair(full_p)
                    if n_frames > 0 and dur_sec >= args.min_sec:
                        targets.append(full_p)

    if not targets:
        print("[!] 找不到符合條件且可讀取的錄影資料夾。")
        return

    print(f"[*] 共找到 {len(targets)} 個錄影資料夾準備重算...")
    ok_cnt = 0
    for d in targets:
        if reprocess_folder(
            d,
            ear_mode=args.ear_mode,
            fast_mode=not args.full,
            sig_window=(args.sig_window if args.sig_window > 0 else None),
            skip_existing=args.skip_existing,
        ):
            ok_cnt += 1
    print(f"[✓] 全部處理完成: {ok_cnt}/{len(targets)} 個資料夾已重新產生呼吸率與四分割結果。")


if __name__ == "__main__":
    main()
