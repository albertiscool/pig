# -*- coding: utf-8 -*-
import subprocess
import os
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(cmd_list, step_name):
    print(f"\n--- [正在執行 {step_name}] ---")
    print(f"指令: {' '.join(cmd_list)}")
    try:
        # 使用 sys.executable 確保使用與目前相同的 Python 環境執行子腳本
        subprocess.run([sys.executable] + cmd_list, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} 執行失敗: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="豬隻呼吸率全自動優化分析流水線")
    parser.add_argument("--rgb", required=True, help="可見光影片路徑")
    parser.add_argument("--thm", required=True, help="熱成像影片路徑")
    parser.add_argument("--out-root", default="data/test_output", help="輸出根目錄")
    args = parser.parse_args()

    rgb_path = os.path.abspath(args.rgb)
    thm_path = os.path.abspath(args.thm)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    
    # 定義中間產物路徑
    roi_csv = os.path.join(out_root, "roi_boxes.csv")
    m1_dir = os.path.join(out_root, "out_m1")
    m2_dir = os.path.join(out_root, "out_m2")
    m3_dir = os.path.join(out_root, "out_m3")
    final_rr_dir = os.path.join(out_root, "rr_final_fusion")

    # Step 1: 執行 vid_test2_optim.py (身分標籤 + Swin 配準 + 產出鼻部 ROI)
    # 明確傳遞參數，確保使用 auto_rr_pipeline.py 同一個 v0 資料夾下的子腳本
    step1_cmd = [
        os.path.join(SCRIPT_DIR, "vid_test2_optim.py"), 
        "--rgb", rgb_path, 
        "--thm", thm_path, 
        "--out-dir", out_root
    ]
    if not run_step(step1_cmd, "Step 1: 座標配準與 ROI 生成"): return

    # 驗證 ROI 檔案是否存在
    if not os.path.exists(roi_csv):
        print(f"❌ 找不到 ROI 檔案: {roi_csv}，請檢查 vid_test2.py 是否正確儲存檔案。")
        return

    # Step 2: 執行 m1, m2, m3 訊號提取
    # m1: 亮度差異法
    run_step([
        os.path.join(SCRIPT_DIR, "roi_signal_m1.py"), "--thermal", thm_path, "--roi-csv", roi_csv, "--out-dir", m1_dir,
        "--butterworth-order", "6", "--inner-shrink", "3"
    ], "Step 2.1: m1 亮度分析")

    # m2: 冷區分割法 (使用 G 通道優化)
    run_step([
        os.path.join(SCRIPT_DIR, "roi_signal_m2.py"), "--thermal", thm_path, "--roi-csv", roi_csv, "--out-dir", m2_dir,
        "--gray-mode", "green", "--th-mode", "otsu", "--min-area", "0.02"
    ], "Step 2.2: m2 冷區分割")

    # m3: 高斯中心加權法
    run_step([
        os.path.join(SCRIPT_DIR, "roi_signal_m3.py"), "--thermal", thm_path, "--roi-csv", roi_csv, "--out-dir", m3_dir,
        "--anchor", "roi_center", "--sigma-ratio", "0.25", "--inner-band-ratio", "0.6", "1.0"
    ], "Step 2.3: m3 溫差加權")

    # Step 3: 多通道融合優化分析 (rr_fft_sliding.py)
    """
    run_step([
        "rr_fft_sliding.py",
        "--inputs", m1_dir, m2_dir, m3_dir,
        "--mode", "multi_vote",
        "--out-dir", final_rr_dir,
        "--dual-band",             
        "--snr-min", "4.0",        
        "--agree-df", "0.17",      
        "--median-win", "3",       
        "--jump-limit-bpm", "20.0",
        "--per-channel-plots"      
    ], "Step 3: 多通道融合 RR 估計")
    """
    run_step([
        os.path.join(SCRIPT_DIR, "rr_fft_sliding.py"),
        "--inputs", m1_dir, m2_dir, m3_dir,
        "--mode", "multi_vote",
        "--out-dir", final_rr_dir,
        "--dual-band",             
        "--snr-min", "1.5",        
        "--agree-df", "0.5",      
        "--median-win", "1",       
        "--jump-limit-bpm", "50.0",
        "--per-channel-plots"      
    ], "Step 3: 多通道融合 RR 估計")

    print(f"\n✅ 全自動優化流程執行完畢！")
    print(f"📍 最終呼吸率趨勢圖請見: {os.path.join(final_rr_dir, 'rr_trend.png')}")

if __name__ == "__main__":
    main()