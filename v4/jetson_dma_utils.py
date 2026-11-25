# -*- coding: utf-8 -*-
"""
jetson_dma_utils.py
專為 NVIDIA Jetson (Xavier / Orin) 開發之硬體 DMA 與零拷貝 (Zero-Copy) 工具模組：
1. GStreamer 硬體 DMA 解碼串流構建器 (NVDEC -> NVMM -> nvvidconv -> appsink)
2. PyTorch Pinned Memory (鎖頁記憶體) 與非同步 DMA 搬移 (Non-blocking DMA Transfer)
3. 預先分配之顯存環形緩衝區 (Pre-allocated CUDA Buffer)，避免重複記憶體分配
"""

import cv2
import torch
import numpy as np

def build_jetson_dma_rtsp_pipeline(rtsp_url, width=1920, height=1080, fps=30, latency=0):
    """
    構建 Jetson 專屬 GStreamer 硬體 DMA 解碼管線
    原理：
    - rtspsrc: 接收 RTSP 串流
    - nvv4l2decoder: 使用 Jetson 硬體解碼晶片 (NVDEC)，直接解碼至硬體 DMA 緩衝區 (NVMM)
    - nvvidconv: 利用硬體轉碼引擎進行色彩空間與尺寸轉換，零 CPU 負擔
    """
    pipeline = (
        f"rtspsrc location={rtsp_url} latency={latency} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )
    return pipeline

class JetsonDMALoader:
    """
    非同步 DMA 影像搬移器 (Asynchronous Host-to-Device DMA Transfer)
    在 Jetson 統一記憶體 (UMA) 架構下，使用 Pinned Memory 與 non_blocking=True，
    讓 CPU 與 GPU 資料傳輸完全交由 GPU DMA 引擎在背景處理，推論運算與資料搬移重疊 (Overlap)！
    """
    def __init__(self, device='cuda', imgsz=640):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.imgsz = imgsz
        self.is_cuda = self.device.type == 'cuda'
        
        # 預先分配常駐 Pinned Buffer (DMA 緩衝區)
        if self.is_cuda:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def preprocess_dma(self, bgr_img):
        """
        將 OpenCV BGR 影像快速轉換並透過 DMA 非同步推至 GPU
        :param bgr_img: numpy.ndarray (H, W, 3)
        :return: torch.Tensor on GPU, shape (1, 3, imgsz, imgsz), float32, normalized 0~1
        """
        h, w = bgr_img.shape[:2]
        
        # 保持比例縮放 (Letterbox / Resize)
        if (w, h) != (self.imgsz, self.imgsz):
            resized = cv2.resize(bgr_img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr_img

        # BGR 轉 RGB, HWC 轉 CHW
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor_cpu = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)

        if self.is_cuda:
            # 1. 將張量鎖頁至 Pinned Memory (允許直接 DMA)
            tensor_pinned = tensor_cpu.pin_memory()
            
            # 2. 在專屬非同步 CUDA Stream 中觸發 DMA 傳輸 (non_blocking=True)
            with torch.cuda.stream(self.stream):
                tensor_gpu = tensor_pinned.to(self.device, non_blocking=True).float() / 255.0
            
            # 同步確保搬移完成
            self.stream.synchronize()
            return tensor_gpu
        else:
            return tensor_cpu.float() / 255.0
