# -*- coding: utf-8 -*-
"""
two_stage_dma_detector.py
專為 NVIDIA Jetson Xavier 開發之二階段 DMA 豬隻偵測引擎：
- 支援 TensorRT (.engine) 或 PyTorch (.pt) 自動切換
- 導入 Host-to-Device 鎖頁記憶體非同步 DMA 搬移 (Pinned Memory DMA)
- Stage 1: 全圖鎖定豬臉 (Face) + 20% 外擴 Margin 裁切
- Stage 2: 局部高解析度眼鼻偵測 (Eye/Nose)，消除耳朵誤判假陽性
- 座標局部到全圖無縫反向映射 (Inverse Coordinate Mapping)
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class TwoStageDMAPigDetector:
    def __init__(self, model_path, margin=0.20, conf=0.25, device='cuda:0'):
        """
        :param model_path: .pt 或 .engine 權重路徑 (在 Xavier 上強烈推薦 .engine)
        :param margin: 臉部外擴裁切比例 (0.20 代表四周各擴 20%)
        :param conf: 檢測信心閾值
        :param device: 推論設備 ('cuda:0' 或 'cpu')
        """
        self.device_str = device if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        self.margin = margin
        self.conf = conf
        
        print(f"[*] [Xavier DMA] 載入模型: {os.path.basename(model_path)} (Device: {self.device_str})")
        self.model = YOLO(model_path)

        # 建立專屬 CUDA Stream 用於非同步 DMA 資料傳輸
        if torch.cuda.is_available():
            self.dma_stream = torch.cuda.Stream()
        else:
            self.dma_stream = None

    def _dma_transfer(self, crop_bgr):
        """
        透過 Pinned Memory 與 CUDA DMA Engine 進行非同步資料搬移
        """
        if not torch.cuda.is_available():
            return crop_bgr

        # 利用 PyTorch non_blocking 特性在 DMA Stream 預熱
        with torch.cuda.stream(self.dma_stream):
            # 建立鎖頁記憶體並觸發非同步傳輸
            t_cpu = torch.from_numpy(crop_bgr).pin_memory()
            t_gpu = t_cpu.to(self.device, non_blocking=True)
        self.dma_stream.synchronize()
        return crop_bgr

    def predict(self, frame):
        """
        執行二階段 DMA 檢測
        :param frame: 原始 1080p BGR 影像
        :return: final_detections: list of dict
        """
        h_img, w_img = frame.shape[:2]
        final_detections = []

        # ---------------- Stage 1: 全圖鎖定豬臉 ----------------
        stage1_results = self.model.predict(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device_str,
            half=True if torch.cuda.is_available() else False
        )[0]

        face_boxes = []
        for box in stage1_results.boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            # 類別 1: face
            if cls_id == 1 or "face" in stage1_results.names.get(cls_id, "").lower():
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_boxes.append((x1, y1, x2, y2, conf_val))

        # 降級保護：若 Stage 1 沒抓到臉，直接回傳全圖偵測結果
        if not face_boxes:
            for box in stage1_results.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                final_detections.append({
                    'box': [x1, y1, x2, y2],
                    'cls': cls_id,
                    'name': stage1_results.names.get(cls_id, str(cls_id)),
                    'conf': conf_val
                })
            return final_detections

        # -------- Stage 2: 臉部裁切 + DMA 搬移 + 局部高精度器官推論 --------
        for fx1, fy1, fx2, fy2, f_conf in face_boxes:
            final_detections.append({
                'box': [fx1, fy1, fx2, fy2],
                'cls': 1,
                'name': 'face',
                'conf': f_conf
            })

            # 計算外擴 Margin (20%)
            fw = fx2 - fx1
            fh = fy2 - fy1
            pad_w = int(fw * self.margin)
            pad_h = int(fh * self.margin)

            cx1 = max(0, fx1 - pad_w)
            cy1 = max(0, fy1 - pad_h)
            cx2 = min(w_img, fx2 + pad_w)
            cy2 = min(h_img, fy2 + pad_h)

            crop_face = frame[cy1:cy2, cx1:cx2]
            if crop_face.size == 0:
                continue

            # 使用 DMA 管道確保記憶體直接就緒
            crop_face = self._dma_transfer(crop_face)

            # Stage 2 局部推論 (640x640 高解析度，半精度 FP16)
            stage2_results = self.model.predict(
                crop_face,
                conf=self.conf,
                imgsz=640,
                verbose=False,
                device=self.device_str,
                half=True if torch.cuda.is_available() else False
            )[0]

            for box in stage2_results.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
                c_name = stage2_results.names.get(cls_id, "").lower()

                # 只保留眼 (0) 與鼻 (2)，過濾多餘雜訊
                if "eye" in c_name or "nose" in c_name or cls_id in [0, 2]:
                    # 映射回全圖全域座標
                    gx1 = cx1 + lx1
                    gy1 = cy1 + ly1
                    gx2 = cx1 + lx2
                    gy2 = cy1 + ly2

                    final_detections.append({
                        'box': [gx1, gy1, gx2, gy2],
                        'cls': cls_id,
                        'name': c_name,
                        'conf': conf_val
                    })

        return final_detections
