"""
video_pipeline2.py — 核心模型定義與工具函式庫
提供 SwinTransformerP2P 模型架構、YOLO 關鍵點後處理與跨光譜 Homography 變形估計。
相容 Windows 與 Linux 環境。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    parent_dir = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    if os.path.exists(os.path.join(parent_dir, "parameters")):
        ROOT_DIR = parent_dir

# 預設權重路徑
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_KEYPOINT_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")
YOLOV8_FACE_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")

IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2]  # 0: 左眼, 1: 右眼, 2: 鼻子
ID_NAMES = ["Left_Eye", "Right_Eye", "Nose"]
CONF_THRESHOLD = 0.5
YOLO_EYE_ID = 0
YOLO_NOSE_ID = 2

class SwinBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.features = swin.features
        self.proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, feat_dim)
        )

    def forward(self, x):
        x = self.features(x)  # (B, 20, 20, 768)
        x = self.proj(x)      # (B, 20, 20, 256)
        return x.permute(0, 3, 1, 2)  # (B, 256, 20, 20)

class SwinTransformerP2P(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_size = 640
        self.backbone = SwinBackbone(feat_dim=feat_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos_embed = nn.Linear(2, feat_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def bilinear_sample(self, fmap, pts):
        B, C, H, W = fmap.shape
        x = pts[..., 0] / self.img_size * 2 - 1
        y = pts[..., 1] / self.img_size * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)
        feat = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear')
        return feat.squeeze(3).permute(0, 2, 1)

    def forward(self, rgb, thm, rgb_pts):
        feat_rgb = self.backbone(rgb)
        feat_thm = self.backbone(thm)
        memory = feat_thm.flatten(2).permute(0, 2, 1)
        query_feat = self.bilinear_sample(feat_rgb, rgb_pts)
        query_pos = self.query_pos_embed(rgb_pts / self.img_size)
        query = query_feat + query_pos
        output = self.decoder(query, memory)
        return self.regression_head(output)

def process_yolo_detection_and_sort_points(yolo_results):
    """解析 YOLO 偵測結果並對左眼、右眼、鼻子進行幾何排序"""
    eye_pts = []
    nose_pts = []
    if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD:
                continue
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
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
    """透過 Swin Transformer P2P 模型預測熱成像對應點，並計算轉換矩陣 M"""
    h_orig, w_orig = rgb_img.shape[:2]
    thm_h, thm_w = thm_img.shape[:2]

    pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
    if not pts_found:
        return None

    rgb_pts_in = [pts_found[cid] if cid in pts_found else [0.0, 0.0] for cid in P2P_CLS_IDS]
    model_in = [[p[0] * IMAGE_SIZE / w_orig, p[1] * IMAGE_SIZE / h_orig] for p in rgb_pts_in]

    t_rgb = torch.from_numpy(rgb_img).permute(2, 0, 1).to(device, non_blocking=True)
    if device.type == 'cuda':
        t_rgb = t_rgb.half()
    else:
        t_rgb = t_rgb.float()
    t_rgb = t_rgb / 255.0

    t_thm = torch.from_numpy(thm_img).permute(2, 0, 1).to(device, non_blocking=True)
    if device.type == 'cuda':
        t_thm = t_thm.half()
    else:
        t_thm = t_thm.float()
    t_thm = t_thm / 255.0

    t_rgb = F.interpolate(t_rgb.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    t_thm = F.interpolate(t_thm.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)

    t_pts = torch.tensor([model_in], dtype=torch.float16 if device.type == 'cuda' else torch.float32, device=device)

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

def get_warped_box(box, matrix):
    """將 RGB 框座標透過變形矩陣 M 映射至 Thermal 座標系統，回傳 (x1, y1, x2, y2)"""
    pts_rgb = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
    pts_poly = cv2.perspectiveTransform(pts_rgb, matrix).reshape(-1, 2)
    x1, y1 = np.min(pts_poly[:, 0]), np.min(pts_poly[:, 1])
    x2, y2 = np.max(pts_poly[:, 0]), np.max(pts_poly[:, 1])
    return int(x1), int(y1), int(x2), int(y2)
