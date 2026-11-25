# -*- coding: utf-8 -*-
"""
export_tensorrt.py
專為 NVIDIA Jetson Xavier 開發之 TensorRT 引擎一鍵轉換工具：
在 Xavier 上將 PyTorch (.pt) 模型導出為極速半精度 TensorRT (.engine)，
推論幀率預期可提升 2 ~ 3 倍以上！

使用範例：
    python export_tensorrt.py --model parameters/nose_eyes_face_0817.pt --half
"""

import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Jetson Xavier TensorRT 一鍵導出工具")
    parser.add_argument("--model", default="parameters/nose_eyes_face_0817.pt", help="輸入的 .pt 權重檔案")
    parser.add_argument("--imgsz", type=int, default=640, help="推論輸入影像尺寸 (預設 640)")
    parser.add_argument("--half", action="store_true", default=True, help="啟用 FP16 半精度加速 (強烈推薦)")
    parser.add_argument("--device", default="0", help="CUDA 設備代號 (預設 0)")
    args = parser.parse_args()

    model_p = os.path.abspath(args.model)
    if not os.path.exists(model_p):
        print(f"❌ 找不到模型檔案: {model_p}")
        return

    print("=" * 65)
    print("  🚀 [Jetson TensorRT] 開始導出模型為 .engine 格式")
    print("=" * 65)
    print(f"[*] 來源模型: {model_p}")
    print(f"[*] 尺寸 (imgsz): {args.imgsz}")
    print(f"[*] 半精度 (FP16): {args.half}")
    print(f"[*] 設備: {args.device}")

    model = YOLO(model_p)

    # 執行導出
    print("[*] 正在透過 TensorRT 構建推論引擎 (在 Xavier 上預計需時 2~5 分鐘)...")
    exported_file = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("🎉 TensorRT 導出成功！")
    print(f"📁 引擎檔案: {exported_file}")
    print("💡 在 Xavier 上直接將該 .engine 傳入系統即可享受極限 FPS！")
    print("=" * 65)

if __name__ == "__main__":
    main()
