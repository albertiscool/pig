import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from ultralytics import YOLO
import json
import time
from collections import defaultdict 
import sys

# -------------------------------------------------------------
# --- MODEL DEFINITION (已替換為 Transformer 架構) ---
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
        x = self.features(x)  # (B, 20, 20, 768)
        x = self.proj(x)      # (B, 20, 20, 256)
        return x.permute(0, 3, 1, 2) # (B, 256, 20, 20)

class SwinTransformerP2P(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_size = 640
        self.backbone = SwinBackbone(feat_dim=feat_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, 
            nhead=nhead, 
            dim_feedforward=1024,  # <--- 加入這一行，必須跟訓練時一致
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
# RGB_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "15棟_344-5_11407021541_可見光_正面2.mp4")
# THM_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "15棟_344-5_11407021541_熱成像_正面2.mp4")
RGB_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
THM_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

# Fixed Parameters
IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2] # P2P 模型所需的最終 ID 結構 (0: 左眼, 1: 右眼, 2: 鼻子)
NUM_POINTS = len(P2P_CLS_IDS)
FUSION_ALPHA = 0.6 
ID_NAMES = ["Left_Eye", "Right_Eye", "Nose"] 
TARGET_DISPLAY_SIZE = (400, 400) 
CONF_THRESHOLD = 0.5 
# --- YOLO 輸出 ID ---
YOLO_EYE_ID = 0  # 合併後的眼睛 ID
YOLO_NOSE_ID = 2 # 鼻子 ID
# --------------------

# 定義固定視窗名稱
WINDOW_NAMES = [
    "1. RGB (YOLO BBox)",
    "2. THM (P2P Predicted Polygon - Red)",
    "3. Warped RGB BBox Check (Yellow Polygon)",
    f"4. Blended Check (Alpha={FUSION_ALPHA})"
]

# -------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# -------------------------------------------------------------

def generate_blank_image(target_h, target_w):
    """生成一張空白（黑色）的圖像用於替換缺失的幀。"""
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)

def add_frame_text(img, frame, title, transform_type=""):
    """在圖像上添加幀數、標題和變換類型。"""
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
    """
    核心邏輯：處理 YOLO 的合併輸出，重建 P2P 結構 (0:左眼, 1:右眼, 2:鼻子)。
    """
    eye_pts = [] 
    nose_pts = [] 
    
    if hasattr(yolo_results, 'boxes'):
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD: 
                continue
            if box.cls.numel() == 0 or box.xyxy.numel() == 0:
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
    
    # 1. 處理鼻子 (P2P ID 2) - 只取第一個鼻子
    if nose_pts:
        final_pts[2] = (nose_pts[0]['pt'], nose_pts[0]['bbox']) 
        
    # 2. 處理眼睛 (P2P ID 0 & 1)
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

def get_segmentation_mask(model, img_orig):
    if model is None: return None, None
    results = model(img_orig, verbose=False, device=model.device)[0] 
    if results and results.masks is not None and len(results.masks.data) > 0:
        mask_data = results.masks.data[0].cpu().numpy() 
        mask_orig = cv2.resize(mask_data, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_orig = (mask_orig > 0).astype(np.uint8) * 255
        if results.boxes.xyxy.numel() > 0:
             box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        else:
             box = np.array([0, 0, img_orig.shape[1], img_orig.shape[0]], dtype=np.int32)
        return mask_orig, box
    return None, None 

def get_model_warp_matrix_inference(rgb_img_orig, thm_img_orig, detected_results_map, p2p_model):
    """使用 P2P 模型預測點動態計算仿射/相似/平移變換矩陣"""
    
    model_device = next(p2p_model.parameters()).device
    h_orig, w_orig, _ = rgb_img_orig.shape
    thm_h, thm_w, _ = thm_img_orig.shape
    
    pts_found = {cid: pt for cid, (pt, _) in detected_results_map.items()}
    valid_count = len(pts_found)

    if valid_count == 0:
        return None, None, None, "None"

    # ⭐️ 這裡修正了 NameError 與 Tensor 形狀 ⭐️
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
        M = np.vstack([M_aff, [0, 0, 1]])
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
        
        # 繪製 BBox (僅用於 RGB 影像，傳入顏色為綠色時)
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

def crop_and_resize_outline_by_bbox(img_outline, x1, y1, x2, y2, target_size, margin=50):
    if img_outline is None: return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    h, w, _ = img_outline.shape
    x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin), min(h, y2 + margin)
    cropped = img_outline[y1:y2, x1:x2]
    if cropped.size == 0: return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    return cv2.resize(cropped, target_size)

# -------------------------------------------------------------
# --- VIDEO INFERENCE FUNCTION (MAIN) ---
# -------------------------------------------------------------

def inference_video():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    
    try:
        detect_model = YOLO(YOLOV8_DETECT_PATH) 
        p2p_model = SwinTransformerP2P().to(device)
        p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
        p2p_model.eval()
        try:
            seg_model = YOLO(YOLOV8_SEG_MODEL_PATH)
        except Exception:
            seg_model = None
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

    # ===============================
    # 🎥 Video Writers (1,2,4 視窗)
    # ===============================
    SAVE_VIDEO = True
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_rgb = None
    writer_thm = None
    writer_blended = None

    if SAVE_VIDEO:
        writer_rgb = cv2.VideoWriter(
            "out_rgb.mp4", fourcc, fps, (thm_w, thm_h)
        )
        writer_thm = cv2.VideoWriter(
            "out_thm.mp4", fourcc, fps, (thm_w, thm_h)
        )
        writer_blended = cv2.VideoWriter(
            "out_blended.mp4", fourcc, fps, (thm_w, thm_h)
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
        
        # --- ⭐️ RGB 端的物理校正 ⭐️ ---
        # 1. 確保眼睛在鼻子上方 (Y 座標校正)
        if 2 in initial_detection_map:
            nose_y_rgb = initial_detection_map[2][0][1]
            for eye_id in [0, 1]:
                if eye_id in initial_detection_map:
                    if initial_detection_map[eye_id][0][1] > nose_y_rgb:
                        # 交換 ID 座標與 BBox (眼睛與鼻子標反)
                        initial_detection_map[eye_id], initial_detection_map[2] = \
                            initial_detection_map[2], initial_detection_map[eye_id]
        
        # 2. 確保左眼在右眼左邊 (X 座標校正)
        if 0 in initial_detection_map and 1 in initial_detection_map:
            if initial_detection_map[0][0][0] > initial_detection_map[1][0][0]:
                initial_detection_map[0], initial_detection_map[1] = \
                    initial_detection_map[1], initial_detection_map[0]

        # B. Transformer 配準推理
        M_rgb_to_thm, dst_pts_pred_np_raw, rgb_center_pts_raw, current_transform_type = get_model_warp_matrix_inference(
            rgb_img_orig, thm_img_orig, initial_detection_map, p2p_model
        )
        
        if M_rgb_to_thm is None:
            cv2.imshow(WINDOW_NAMES[0], add_frame_text(rgb_img_orig.copy(), frame_count, WINDOW_NAMES[0], "Matrix Fail"))
            cv2.imshow(WINDOW_NAMES[1], add_frame_text(thm_img_orig.copy(), frame_count, WINDOW_NAMES[1], "Matrix Fail"))
        else:
            # 1. 建立熱像圖預測點位
            valid_cls_ids = list(initial_detection_map.keys())
            thm_predicted_pts = {cid: dst_pts_pred_np_raw[i] for i, cid in enumerate(valid_cls_ids)}

            # --- ⭐️ THM 端的物理校正 ⭐️ ---
            # 1. 確保預測的眼睛在鼻子上方 (Y 座標校正)
            if 2 in thm_predicted_pts:
                nose_y_thm = thm_predicted_pts[2][1]
                for eye_id in [0, 1]:
                    if eye_id in thm_predicted_pts:
                        if thm_predicted_pts[eye_id][1] > nose_y_thm:
                            thm_predicted_pts[eye_id], thm_predicted_pts[2] = \
                                thm_predicted_pts[2], thm_predicted_pts[eye_id]
            
            # 2. 確保左眼在右眼左邊 (X 座標校正)
            if 0 in thm_predicted_pts and 1 in thm_predicted_pts:
                if thm_predicted_pts[0][0] > thm_predicted_pts[1][0]:
                    thm_predicted_pts[0], thm_predicted_pts[1] = \
                        thm_predicted_pts[1], thm_predicted_pts[0]

            # 2. 影像處理與繪製
            rgb_result_img = draw_bboxes_and_keypoints(
                rgb_img_orig.copy(), initial_detection_map, rgb_center_pts_raw, (0, 255, 0)
            )
            rgb_result_img = add_frame_text(rgb_result_img, frame_count, WINDOW_NAMES[0], current_transform_type)
            
            thm_result_img = thm_img_orig.copy()
            warped_rgb = cv2.warpPerspective(rgb_img_orig, M_rgb_to_thm, (thm_w, thm_h))
            
            # 繪製 Thermal 視窗的多邊形
            for cls_id, (_, bbox_rgb) in initial_detection_map.items():
                pts_rgb = np.float32([[bbox_rgb[0], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[3]], [bbox_rgb[0], bbox_rgb[3]]]).reshape(-1, 1, 2)
                pts_poly = np.int32([cv2.perspectiveTransform(pts_rgb, M_rgb_to_thm).reshape(-1, 2)])
                cv2.polylines(thm_result_img, pts_poly, True, (0, 0, 255), 3) 
            
            # 繪製 Thermal 視窗的點位 (傳入校正後的 thm_predicted_pts)
            thm_result_img = draw_bboxes_and_keypoints(thm_result_img, initial_map={}, key_points_map=thm_predicted_pts, color=(0, 0, 255))
            thm_result_img = add_frame_text(thm_result_img, frame_count, WINDOW_NAMES[1], current_transform_type)

            blended_check = cv2.addWeighted(warped_rgb, FUSION_ALPHA, thm_img_orig, 1.0 - FUSION_ALPHA, 0)
            blended_check = add_frame_text(blended_check, frame_count, WINDOW_NAMES[3], current_transform_type)
            
            warped_bbox_check = warped_rgb.copy()
            for cls_id, (_, bbox_rgb) in initial_detection_map.items():
                pts_rgb = np.float32([[bbox_rgb[0], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[1]], [bbox_rgb[2], bbox_rgb[3]], [bbox_rgb[0], bbox_rgb[3]]]).reshape(-1, 1, 2)
                pts_poly = np.int32([cv2.perspectiveTransform(pts_rgb, M_rgb_to_thm).reshape(-1, 2)])
                cv2.polylines(warped_bbox_check, pts_poly, True, (0, 255, 255), 3) 
            
            # 繪製 Warped 視窗的點位
            warped_bbox_check = draw_bboxes_and_keypoints(warped_bbox_check, initial_map={}, key_points_map=thm_predicted_pts, color=(0, 0, 255))
            warped_bbox_check = add_frame_text(warped_bbox_check, frame_count, WINDOW_NAMES[2], current_transform_type)
            
        rgb_show = cv2.resize(rgb_result_img, (thm_w, thm_h))

        cv2.imshow(WINDOW_NAMES[0], rgb_show)
        cv2.imshow(WINDOW_NAMES[1], thm_result_img)
        cv2.imshow(WINDOW_NAMES[2], warped_bbox_check)
        cv2.imshow(WINDOW_NAMES[3], blended_check)

        if SAVE_VIDEO:
            writer_rgb.write(rgb_show)          # 視窗 1
            writer_thm.write(thm_result_img)    # 視窗 2
            writer_blended.write(blended_check) # 視窗 4


        if cv2.waitKey(delay) & 0xFF == ord('q'): break

    cap_rgb.release()
    cap_thm.release()

    if SAVE_VIDEO:
        writer_rgb.release()
        writer_thm.release()
        writer_blended.release()
        print("🎉 影片已輸出：out_rgb.mp4 / out_thm.mp4 / out_blended.mp4")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_video()