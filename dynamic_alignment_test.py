import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from ultralytics import YOLO
from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# -------------------------------------------------------------
# --- MODEL DEFINITION (Swin Transformer P2P) ---
# -------------------------------------------------------------
class SwinBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.features = swin.features 
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, feat_dim))
    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 3, 1, 2)

class SwinTransformerP2P(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=6):
        super().__init__()
        self.feat_dim, self.img_size = feat_dim, 640
        self.backbone = SwinBackbone(feat_dim=feat_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos_embed = nn.Linear(2, feat_dim)
        self.regression_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 2))

    def bilinear_sample(self, fmap, pts):
        B, C, H, W = fmap.shape
        x, y = pts[..., 0] / self.img_size * 2 - 1, pts[..., 1] / self.img_size * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2) 
        feat = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear')
        return feat.squeeze(3).permute(0, 2, 1)

    def forward(self, rgb, thm, rgb_pts):
        feat_rgb, feat_thm = self.backbone(rgb), self.backbone(thm)
        memory = feat_thm.flatten(2).permute(0, 2, 1)
        query = self.bilinear_sample(feat_rgb, rgb_pts) + self.query_pos_embed(rgb_pts / self.img_size)
        return self.regression_head(self.decoder(query, memory))

# --------------------
# --- CONFIGURATION ---
# --------------------
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_DETECT_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "best_yang.pt")
RGB_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
THM_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5
FUSION_ALPHA = 0.6
ID_NAMES = ["Left_Eye", "Right_Eye", "Nose"]

# -------------------------------------------------------------
# --- 工具函數 ---
# -------------------------------------------------------------
def add_frame_text(img, frame, title, t_type=""):
    img_out = img.copy()
    cv2.rectangle(img_out, (0, 0), (img_out.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img_out, f"{title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_out, f"Frame: {frame} {t_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img_out

def run_test_integrated():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = YOLO(YOLOV8_DETECT_PATH)
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device))
    p2p_model.eval()

    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_thm = cv2.VideoCapture(THM_VIDEO_PATH)
    
    thm_w = int(cap_thm.get(3))
    thm_h = int(cap_thm.get(4))
    fps = cap_thm.get(5) or 30
    
    # 建立視窗並允許縮放
    cv2.namedWindow("Alignment Check", cv2.WINDOW_NORMAL)

    frame_count = 0
    while cap_rgb.isOpened():
        ret1, frame_rgb = cap_rgb.read(); ret2, frame_thm = cap_thm.read()
        if not ret1 or not ret2: break
        frame_count += 1

        # 1. YOLO 偵測與物理校正
        res = detect_model.predict(frame_rgb, conf=CONF_THRESHOLD, verbose=False)[0]
        detect_map = {}
        for box in res.boxes:
            cid = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if cid in [0, 1, 2]:
                detect_map[cid] = ([(x1+x2)/2, (y1+y2)/2], [x1, y1, x2, y2])

        # 物理校正：確保 ID 正確 (眼睛在鼻子上方，左眼在左)
        if 2 in detect_map:
            for eid in [0, 1]:
                if eid in detect_map and detect_map[eid][0][1] > detect_map[2][0][1]:
                    detect_map[eid], detect_map[2] = detect_map[2], detect_map[eid]
        if 0 in detect_map and 1 in detect_map:
            if detect_map[0][0][0] > detect_map[1][0][0]:
                detect_map[0], detect_map[1] = detect_map[1], detect_map[0]

        # 2. 計算變換矩陣 M
        M = None
        t_type = "None"
        if len(detect_map) > 0:
            h_r, w_r = frame_rgb.shape[:2]
            model_input = [detect_map[cid][0] if cid in detect_map else [0.0, 0.0] for cid in [0, 1, 2]]
            model_input_scaled = [[p[0] * IMAGE_SIZE / w_r, p[1] * IMAGE_SIZE / h_r] for p in model_input]

            rgb_t = torch.from_numpy(cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE)).transpose(2,0,1)).float().to(device).unsqueeze(0) / 255.0
            thm_t = torch.from_numpy(cv2.resize(frame_thm, (IMAGE_SIZE, IMAGE_SIZE)).transpose(2,0,1)).float().to(device).unsqueeze(0) / 255.0
            pts_t = torch.tensor([model_input_scaled], dtype=torch.float32).to(device)

            with torch.no_grad():
                pred_thm = p2p_model(rgb_t, thm_t, pts_t).cpu().numpy()[0]

            src_pts = np.float32([detect_map[cid][0] for cid in detect_map.keys()])
            dst_pts = np.float32([[pred_thm[i][0] * thm_w / IMAGE_SIZE, pred_thm[i][1] * thm_h / IMAGE_SIZE] 
                                  for i, cid in enumerate([0, 1, 2]) if cid in detect_map])

            if len(src_pts) >= 3:
                M = np.vstack([cv2.getAffineTransform(src_pts[:3], dst_pts[:3]), [0, 0, 1]]); t_type = "Affine"
            elif len(src_pts) == 2:
                M_p, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                M = np.vstack([M_p, [0, 0, 1]]); t_type = "Similarity"

        # 3. 繪製與視覺化檢查
        vis_thm = frame_thm.copy()
        if M is not None:
            # 取得鼻子 (ID=2) 的投影
            if 2 in detect_map:
                box_rgb = detect_map[2][1]
                pts_rgb = np.float32([[box_rgb[0], box_rgb[1]], [box_rgb[2], box_rgb[1]], 
                                      [box_rgb[2], box_rgb[3]], [box_rgb[0], box_rgb[3]]]).reshape(-1, 1, 2)
                pts_thm = cv2.perspectiveTransform(pts_rgb, M).reshape(-1, 2)
                
                # --- [視覺化 A]: 紅色多邊形 (P2P 透視結果) ---
                cv2.polylines(vis_thm, [np.int32(pts_thm)], True, (0, 0, 255), 2, cv2.LINE_AA)
                
                # --- [視覺化 B]: 綠色矩形 (CSV 儲存結果) ---
                tx1, ty1 = np.min(pts_thm[:, 0]), np.min(pts_thm[:, 1])
                tx2, ty2 = np.max(pts_thm[:, 0]), np.max(pts_thm[:, 1])
                cv2.rectangle(vis_thm, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 255, 0), 2)
                
                cv2.putText(vis_thm, "RED: P2P Poly", (10, thm_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis_thm, "GREEN: CSV Box", (10, thm_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 融合畫面供參考
            warped_rgb = cv2.warpPerspective(frame_rgb, M, (thm_w, thm_h))
            blended = cv2.addWeighted(warped_rgb, FUSION_ALPHA, frame_thm, 1.0 - FUSION_ALPHA, 0)
            
            # 組合畫面展示
            vis_thm = add_frame_text(vis_thm, frame_count, "ROI Prediction", t_type)
            blended = add_frame_text(blended, frame_count, "Fusion Check", t_type)
            
            # 合併兩個顯示畫面 (水平拼接)
            h_concat = np.hstack([blended, vis_thm])
            cv2.imshow("Alignment Check", h_concat)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_rgb.release(); cap_thm.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test_integrated()