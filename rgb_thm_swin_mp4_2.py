import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import pandas as pd  # 新增 pandas 用於儲存 CSV
from ultralytics import YOLO
import json
import time
from collections import defaultdict 
import sys

# -------------------------------------------------------------
# --- MODEL DEFINITION (Swin Transformer P2P) ---
# -------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
        pos = torch.stack([x, y], dim=-1)
        self.register_buffer('pos', pos)
        self.proj = nn.Linear(2, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        pos_embed = self.proj(self.pos).permute(2, 0, 1).unsqueeze(0)
        return x + pos_embed

from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 3, 1, 2)

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

# --------------------
# --- CONFIGURATION (請務必檢查以下路徑) ---
# --------------------
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_DETECT_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "best_yang.pt")
YOLOV8_SEG_MODEL_PATH = (
    os.path.join(ROOT_DIR, "parameters", "best.pt")
)

# 💥 影片輸入路徑 💥
RGB_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
THM_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")
OUTPUT_DIR = r"data/test_output" # CSV 輸出位置

# Fixed Parameters
IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2] 
NUM_POINTS = len(P2P_CLS_IDS)
FUSION_ALPHA = 0.6 
ID_NAMES = ["Left_Eye", "Right_Eye", "Nose"] 
CONF_THRESHOLD = 0.5 
YOLO_EYE_ID = 0  
YOLO_NOSE_ID = 2 

# 定義固定視窗名稱
WINDOW_NAMES = [
    "1. RGB (YOLO BBox)",
    "2. THM (Red: Poly / Green: CSV Box)",
    "3. Warped RGB BBox Check",
    f"4. Blended Check (Alpha={FUSION_ALPHA})"
]

# -------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# -------------------------------------------------------------

def add_frame_text(img, frame, title, transform_type=""):
    img_out = img.copy()
    if len(img.shape) == 2: 
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    
    h_out, w_out, _ = img_out.shape
    cv2.rectangle(img_out, (0, 0), (w_out, 80), (0, 0, 0), -1) 
    
    cv2.putText(img_out, f"{title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_out, f"Frame: {frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if transform_type:
        cv2.putText(img_out, f"Transform: {transform_type}", (w_out - 300, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img_out

def process_yolo_detection_and_sort_points(yolo_results):
    eye_pts = [] 
    nose_pts = [] 
    
    if hasattr(yolo_results, 'boxes'):
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD: continue
            if box.cls.numel() == 0 or box.xyxy.numel() == 0: continue
            
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
    elif len(eye_pts) == 1 and final_pts.get(2) is not None:
        single_eye = eye_pts[0]
        nose_center = final_pts[2][0]
        if single_eye['x'] > nose_center[0]:
            final_pts[1] = (single_eye['pt'], single_eye['bbox']) # 右眼
        else:
            final_pts[0] = (single_eye['pt'], single_eye['bbox']) # 左眼
    elif len(eye_pts) == 1:
        final_pts[0] = (eye_pts[0]['pt'], eye_pts[0]['bbox']) 
            
    return final_pts

def get_model_warp_matrix_inference(rgb_img_orig, thm_img_orig, detected_results_map, p2p_model):
    model_device = next(p2p_model.parameters()).device
    h_orig, w_orig, _ = rgb_img_orig.shape
    thm_h, thm_w, _ = thm_img_orig.shape
    
    pts_found = {cid: pt for cid, (pt, _) in detected_results_map.items()}
    valid_count = len(pts_found)
    if valid_count == 0: return None, None, None, "None"

    rgb_pts_model_input = []
    src_pts_orig = []
    for cls_id in P2P_CLS_IDS:
        if cls_id in pts_found:
            x, y = pts_found[cls_id]
            src_pts_orig.append([x, y])
            rgb_pts_model_input.append([x * IMAGE_SIZE / w_orig, y * IMAGE_SIZE / h_orig])
        else:
            rgb_pts_model_input.append([0.0, 0.0])

    rgb_pts_tensor = torch.tensor([rgb_pts_model_input], dtype=torch.float32).to(model_device)
    rgb_resized = cv2.resize(rgb_img_orig, (IMAGE_SIZE, IMAGE_SIZE))
    thm_resized = cv2.resize(thm_img_orig, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
    thm_tensor = torch.from_numpy(thm_resized.transpose(2, 0, 1)).float() / 255.0

    with torch.no_grad():
        thm_pts_pred_scaled = p2p_model(
            rgb_tensor.unsqueeze(0).to(model_device),
            thm_tensor.unsqueeze(0).to(model_device),
            rgb_pts_tensor
        ).cpu().numpy()[0]

    dst_pts = []
    for i, cls_id in enumerate(P2P_CLS_IDS):
        if cls_id in pts_found:
            dst_pts.append([
                thm_pts_pred_scaled[i][0] * thm_w / IMAGE_SIZE,
                thm_pts_pred_scaled[i][1] * thm_h / IMAGE_SIZE
            ])

    src_pts = np.float32(src_pts_orig)
    dst_pts = np.float32(dst_pts)

    M = None
    transform_type = "None"
    if valid_count >= 3:
        M_aff = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        M = np.vstack([M_aff, [0, 0, 1]]) # 轉為 3x3 透視矩陣
        transform_type = "Affine"
    elif valid_count == 2:
        M_p, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M_p is not None:
            M = np.vstack([M_p, [0, 0, 1]])
            transform_type = "Similarity"
    elif valid_count == 1:
        dx = dst_pts[0][0] - src_pts[0][0]
        dy = dst_pts[0][1] - src_pts[0][1]
        M = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        transform_type = "Translation"

    return M, dst_pts, pts_found, transform_type

def draw_bboxes_and_keypoints(img, initial_map, key_points_map, color, thickness=3):
    img_out = img.copy()
    for cls_id in key_points_map.keys():
        center_pt_raw = key_points_map[cls_id]
        if cls_id in initial_map and color == (0, 255, 0): 
             _, bbox_raw = initial_map[cls_id]
             x1, y1, x2, y2 = bbox_raw.astype(int)
             cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)

        x_c, y_c = int(center_pt_raw[0]), int(center_pt_raw[1])
        cv2.circle(img_out, (x_c, y_c), 7, color, -1) 
        label = f"{ID_NAMES[cls_id]}(ID={cls_id})"
        cv2.putText(img_out, label, (x_c + 10, y_c - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
    return img_out

# -------------------------------------------------------------
# --- VIDEO INFERENCE FUNCTION (MAIN) ---
# -------------------------------------------------------------

def inference_video():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        detect_model = YOLO(YOLOV8_DETECT_PATH) 
        p2p_model = SwinTransformerP2P().to(device)
        p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
        p2p_model.eval()
        print("✅ 所有模型載入成功。")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}. 終止執行。")
        return
    
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_thm = cv2.VideoCapture(THM_VIDEO_PATH)

    if not cap_rgb.isOpened() or not cap_thm.isOpened():
        print(f"❌ 錯誤: 無法開啟影片。")
        return

    thm_w = int(cap_thm.get(cv2.CAP_PROP_FRAME_WIDTH))
    thm_h = int(cap_thm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_thm.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps) if fps > 0 else 1

    # 初始化 CSV 紀錄清單
    roi_records = []

    # Video Writers
    SAVE_VIDEO = True
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_blended = None
    if SAVE_VIDEO:
        writer_blended = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, "out_blended.mp4"), fourcc, fps, (thm_w, thm_h)
        )

    frame_count = 0
    for name in WINDOW_NAMES:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    while cap_rgb.isOpened() and cap_thm.isOpened():
        ret_rgb, rgb_img_orig = cap_rgb.read()
        ret_thm, thm_img_orig = cap_thm.read()
        if not ret_rgb or not ret_thm: break
        frame_count += 1

        # A. RGB 端偵測
        yolo_results = detect_model.predict(rgb_img_orig, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device)[0]
        initial_detection_map = process_yolo_detection_and_sort_points(yolo_results)
        
        # --- 物理校正 ---
        if 2 in initial_detection_map:
            nose_y_rgb = initial_detection_map[2][0][1]
            for eye_id in [0, 1]:
                if eye_id in initial_detection_map:
                    if initial_detection_map[eye_id][0][1] > nose_y_rgb:
                        initial_detection_map[eye_id], initial_detection_map[2] = \
                            initial_detection_map[2], initial_detection_map[eye_id]
        
        if 0 in initial_detection_map and 1 in initial_detection_map:
            if initial_detection_map[0][0][0] > initial_detection_map[1][0][0]:
                initial_detection_map[0], initial_detection_map[1] = \
                    initial_detection_map[1], initial_detection_map[0]

        # B. Transformer 配準推理
        M_rgb_to_thm, dst_pts_pred_np_raw, rgb_center_pts_raw, current_transform_type = get_model_warp_matrix_inference(
            rgb_img_orig, thm_img_orig, initial_detection_map, p2p_model
        )
        
        # 繪製畫面初始化
        rgb_result_img = draw_bboxes_and_keypoints(rgb_img_orig.copy(), initial_detection_map, rgb_center_pts_raw, (0, 255, 0))
        rgb_result_img = add_frame_text(rgb_result_img, frame_count, WINDOW_NAMES[0], current_transform_type)
        
        thm_result_img = thm_img_orig.copy()
        
        if M_rgb_to_thm is None:
            # 偵測失敗時的處理
            pass
        else:
            valid_cls_ids = list(initial_detection_map.keys())
            thm_predicted_pts = {cid: dst_pts_pred_np_raw[i] for i, cid in enumerate(valid_cls_ids)}

            warped_rgb = cv2.warpPerspective(rgb_img_orig, M_rgb_to_thm, (thm_w, thm_h))
            
            # --- 關鍵修改：處理鼻部 ROI 並儲存 ---
            if 2 in initial_detection_map:
                # 1. 取得 RGB 鼻部框
                _, bbox_rgb = initial_detection_map[2]
                pts_rgb = np.float32([
                    [bbox_rgb[0], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[1]], 
                    [bbox_rgb[2], bbox_rgb[3]], [bbox_rgb[0], bbox_rgb[3]]
                ]).reshape(-1, 1, 2)
                
                # 2. 投影到 Thermal 座標系
                pts_poly = cv2.perspectiveTransform(pts_rgb, M_rgb_to_thm).reshape(-1, 2)
                
                # 3. 計算最小外接矩形 (CSV 用)
                tx1 = np.min(pts_poly[:, 0])
                ty1 = np.min(pts_poly[:, 1])
                tx2 = np.max(pts_poly[:, 0])
                ty2 = np.max(pts_poly[:, 1])

                # 4. 存入紀錄 (Frame, x1, y1, x2, y2)
                roi_records.append([frame_count, tx1, ty1, tx2, ty2])

                # 5. [視覺化 A] 畫紅色多邊形 (P2P 原型)
                cv2.polylines(thm_result_img, [np.int32(pts_poly)], True, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 6. [視覺化 B] 畫綠色矩形 (CSV 儲存框)
                cv2.rectangle(thm_result_img, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 255, 0), 2)
                
                # 標示
                cv2.putText(thm_result_img, "Red: P2P Poly", (10, thm_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(thm_result_img, "Green: CSV Box", (10, thm_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            thm_result_img = add_frame_text(thm_result_img, frame_count, WINDOW_NAMES[1], current_transform_type)

            # Blended 檢查
            blended_check = cv2.addWeighted(warped_rgb, FUSION_ALPHA, thm_img_orig, 1.0 - FUSION_ALPHA, 0)
            blended_check = add_frame_text(blended_check, frame_count, WINDOW_NAMES[3], current_transform_type)
            
            # Warped Check (Yellow Polygon)
            warped_bbox_check = warped_rgb.copy()
            for cls_id, (_, bbox_rgb) in initial_detection_map.items():
                pts_rgb = np.float32([[bbox_rgb[0], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[3]], [bbox_rgb[0], bbox_rgb[3]]]).reshape(-1, 1, 2)
                pts_poly_w = np.int32([cv2.perspectiveTransform(pts_rgb, M_rgb_to_thm).reshape(-1, 2)])
                cv2.polylines(warped_bbox_check, pts_poly_w, True, (0, 255, 255), 3) 
            warped_bbox_check = add_frame_text(warped_bbox_check, frame_count, WINDOW_NAMES[2], current_transform_type)
            
            # 顯示
            rgb_show = cv2.resize(rgb_result_img, (thm_w, thm_h))
            cv2.imshow(WINDOW_NAMES[0], rgb_show)
            cv2.imshow(WINDOW_NAMES[1], thm_result_img)
            cv2.imshow(WINDOW_NAMES[2], warped_bbox_check)
            cv2.imshow(WINDOW_NAMES[3], blended_check)

            if SAVE_VIDEO:
                writer_blended.write(blended_check)

        if cv2.waitKey(delay) & 0xFF == ord('q'): break

    # --- 程式結束後儲存 CSV ---
    roi_csv_path = os.path.join(OUTPUT_DIR, "roi_boxes.csv")
    df = pd.DataFrame(roi_records, columns=['frame', 'x1', 'y1', 'x2', 'y2'])
    df.to_csv(roi_csv_path, index=False)
    
    cap_rgb.release()
    cap_thm.release()
    if SAVE_VIDEO:
        writer_blended.release()
    
    print(f"🎉 影片已輸出，ROI 檔案已儲存至: {roi_csv_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_video()