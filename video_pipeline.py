"""
video_pipeline.py  —  離線影片版管線 (整合 auto_rr_pipeline)
自動尋找影片並呼叫 auto_rr_pipeline 進行全自動分析。

用法：
  python video_pipeline.py                          # 互動式選擇影片
  python video_pipeline.py --pig_id 484-6           # 自動搜尋對應的 RGB/熱成像影片
  python video_pipeline.py --rgb <RGB路徑> --thm <熱成像路徑>   # 手動指定兩支影片
"""

import os
import argparse
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 預設使用專案相對的 data/test_vid（v2 在子資料夾時會回退到上層）
TEST_VID_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "test_vid"))

def find_video_pair(pig_id_query):
    """根據豬隻 ID 自動搜尋 data/test_vid 中對應的 RGB 和熱成像影片"""
    if not os.path.isdir(TEST_VID_DIR):
        print(f"[錯誤] 找不到測試影片目錄: {TEST_VID_DIR}")
        return None, None

    all_files = os.listdir(TEST_VID_DIR)
    rgb_path, thm_path = None, None

    for fname in sorted(all_files, key=lambda x: (not x.endswith('.mp4'), x)):
        if not fname.lower().endswith(('.mp4', '.avi')): continue
        # 比對 pig_id（檔名前綴）
        name_lower = fname.lower()
        if pig_id_query.lower() not in name_lower.split('_')[0].lower():
            # 也嘗試直接包含比對 (例如 "15棟_295-6")
            if pig_id_query.lower() not in name_lower:
                continue

        full_path = os.path.join(TEST_VID_DIR, fname)
        if '可見光' in fname or 'rgb' in name_lower:
            if rgb_path is None:  # 只取第一個匹配到的
                rgb_path = full_path
        elif '熱成像' in fname or 'thermal' in name_lower or 'thm' in name_lower:
            if thm_path is None:
                thm_path = full_path

    return rgb_path, thm_path

def list_available_videos():
    """列出 data/test_vid 中所有可用的影片組合"""
    if not os.path.isdir(TEST_VID_DIR):
        print(f"[錯誤] 找不到測試影片目錄: {TEST_VID_DIR}")
        return {}

    # 排序時讓 .mp4 排在前面，這樣有重複檔名時會優先取用到 .mp4
    all_files = sorted(os.listdir(TEST_VID_DIR), key=lambda x: (not x.endswith('.mp4'), x))
    pairs = {}  # pig_id -> {'rgb': path, 'thm': path}

    for fname in all_files:
        if not fname.lower().endswith(('.mp4', '.avi')): continue
        # 取得 pig_id
        parts = fname.split('_')
        # 有些檔名是 "15棟_295-6_..." 格式
        pig_id = parts[0]
        if '棟' in pig_id and len(parts) > 1:
            pig_id = parts[1]  # 取 295-6 部分

        if pig_id not in pairs:
            pairs[pig_id] = {'rgb': None, 'thm': None, 'rgb_name': None, 'thm_name': None}

        full_path = os.path.join(TEST_VID_DIR, fname)
        if '可見光' in fname:
            if pairs[pig_id]['rgb'] is None:
                pairs[pig_id]['rgb'] = full_path
                pairs[pig_id]['rgb_name'] = fname
        elif '熱成像' in fname:
            if pairs[pig_id]['thm'] is None:
                pairs[pig_id]['thm'] = full_path
                pairs[pig_id]['thm_name'] = fname

    return pairs

def main():
    parser = argparse.ArgumentParser(description='離線影片版豬隻辨識與呼吸率分析管線 (整合 auto_rr_pipeline)')
    parser.add_argument('--rgb', type=str, default=None, help='RGB (可見光) 影片路徑')
    parser.add_argument('--thm', type=str, default=None, help='Thermal (熱成像) 影片路徑')
    parser.add_argument('--pig_id', type=str, default=None, help='豬隻 ID，自動搜尋 data/test_vid 中對應影片')
    parser.add_argument('--out_root', type=str, default="data/test_output", help='輸出目錄')
    
    # 為了相容原本的參數，接受但不處理部分參數
    parser.add_argument('--save_video', action='store_true', help='相容用，不具作用')
    parser.add_argument('--collect_faces', action='store_true', help='是否蒐集未知豬臉')
    parser.add_argument('--no_display', action='store_true', help='相容用，不具作用')
    parser.add_argument('--export_csv', action='store_true', help='相容用，不具作用')
    
    args = parser.parse_args()

    rgb_path = args.rgb
    thm_path = args.thm

    # === 模式判斷 ===
    if rgb_path and thm_path:
        pass
    elif args.pig_id:
        rgb_path, thm_path = find_video_pair(args.pig_id)
        if rgb_path is None or thm_path is None:
            print(f"[錯誤] 找不到 pig_id={args.pig_id} 的完整影片組合")
            return
    else:
        pairs = list_available_videos()
        if not pairs:
            print("[錯誤] data/test_vid 目錄中找不到任何影片。")
            print("       請使用 --rgb 和 --thm 手動指定影片路徑。")
            return

        print("\n" + "=" * 60)
        print("  離線影片管線 - 可用的測試影片組合")
        print("=" * 60)
        valid_ids = []
        for i, (pid, info) in enumerate(sorted(pairs.items())):
            has_rgb = "✓" if info['rgb'] else "✗"
            has_thm = "✓" if info['thm'] else "✗"
            status = "可用" if info['rgb'] and info['thm'] else "缺少影片"
            print(f"  [{i+1}] {pid:>10s}  |  RGB: {has_rgb}  |  THM: {has_thm}  |  {status}")
            valid_ids.append(pid)

        print("=" * 60)
        choice = input("\n>> 請輸入編號選擇豬隻 (或直接輸入 pig_id): ").strip()

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(valid_ids):
                selected_id = valid_ids[idx]
            else:
                print("[錯誤] 無效的編號。")
                return
        else:
            selected_id = choice

        if selected_id in pairs:
            rgb_path = pairs[selected_id]['rgb']
            thm_path = pairs[selected_id]['thm']
        else:
            # 嘗試用 find_video_pair 搜尋
            rgb_path, thm_path = find_video_pair(selected_id)

        if rgb_path is None or thm_path is None:
            print(f"[錯誤] 找不到 {selected_id} 的完整影片組合")
            return

    # === 驗證影片檔案 ===
    for p, label in [(rgb_path, "RGB"), (thm_path, "THM")]:
        if not os.path.isfile(p):
            print(f"[錯誤] {label} 影片檔案不存在: {p}")
            return

    print("\n" + "=" * 60)
    print("  準備執行 auto_rr_pipeline")
    print("=" * 60)
    print(f"  RGB 影片: {os.path.basename(rgb_path)}")
    print(f"  THM 影片: {os.path.basename(thm_path)}")
    print("=" * 60)

    if not args.collect_faces:
        ans = input("\n>> 是否要蒐集未知豬臉 (Collect Unknown Faces)? [y/N]: ").strip().lower()
        if ans in ['y', 'yes']:
            args.collect_faces = True

    # === 呼叫 auto_rr_pipeline.py ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    auto_rr_path = os.path.join(script_dir, "auto_rr_pipeline.py")
    
    cmd = [
        sys.executable, auto_rr_path,
        "--rgb", rgb_path,
        "--thm", thm_path,
        "--out-root", args.out_root
    ]
    if args.collect_faces:
        cmd.append("--collect-faces")
    
    print(f"執行指令: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print("\n>> 處理完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n[錯誤] auto_rr_pipeline 執行失敗: {e}")

if __name__ == "__main__":
    main()
