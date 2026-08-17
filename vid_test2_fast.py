# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import pandas as pd
import argparse
import time
from ultralytics import YOLO
from torchvision import models
from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
# -------------------------------------------------------------
# --- MODEL DEFINITION 1: Swin Transformer P2P ---
# -------------------------------------------------------------
class SwinBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.features = swin.features 
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, feat_dim))
    def forward(self, x):
        x = self.features(x); x = self.proj(x)
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
        x = pts[..., 0] / self.img_size * 2 - 1
        y = pts[..., 1] / self.img_size * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2) 
        feat = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear')
        return feat.squeeze(3).permute(0, 2, 1)
    def forward(self, rgb, thm, rgb_pts):
        feat_rgb, feat_thm = self.backbone(rgb), self.backbone(thm)
        memory = feat_thm.flatten(2).permute(0, 2, 1)
        query = self.bilinear_sample(feat_rgb, rgb_pts) + self.query_pos_embed(rgb_pts / self.img_size)
        return self.regression_head(self.decoder(query, memory))

# -------------------------------------------------------------
# --- MODEL DEFINITION 2: ECBHybridModel (Pig ID) ---
# -------------------------------------------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def extract_features(self, x):
        feat_cnn = self.cnn_model(x) 
        return F.normalize(feat_cnn, dim=1)

def get_pig_id(name: str):
    return name.split('_')[0]

# --------------------
# --- CONFIGURATION ---
# --------------------
P2P_MODEL_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth")
YOLOV8_KEYPOINT_PATH = os.path.join(ROOT_DIR, "parameters", "bestv12.pt")
YOLOV8_FACE_PATH = os.path.join(ROOT_DIR, "parameters", "nose_eyes_face.pt")  

IMAGE_SIZE = 640
P2P_CLS_IDS = [0, 1, 2] 
CONF_THRESHOLD = 0.5 
YOLO_EYE_ID = 0  
YOLO_NOSE_ID = 2 
ID_CHECK_INTERVAL = 15

# -------------------------------------------------------------
# --- UTILITY FUNCTIONS ---
# -------------------------------------------------------------
def process_yolo_detection_and_sort_points(yolo_results):
    eye_pts = [] 
    nose_pts = [] 
    if hasattr(yolo_results, 'boxes'):
        for box in yolo_results.boxes:
            if box.conf[0] < CONF_THRESHOLD: continue
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_pt = np.array([x_center, y_center], dtype=np.float32)
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            if cls_id == YOLO_EYE_ID: 
                eye_pts.append({'pt': center_pt, 'bbox': bbox, 'x': x_center})
            elif cls_id == YOLO_NOSE_ID: 
                nose_pts.append({'pt': center_pt, 'bbox': bbox})
    final_pts = {}
    if nose_pts: final_pts[2] = (nose_pts[0]['pt'], nose_pts[0]['bbox']) 
    if len(eye_pts) >= 2:
        eye_pts.sort(key=lambda item: item['x'])
        final_pts[0], final_pts[1] = (eye_pts[0]['pt'], eye_pts[0]['bbox']), (eye_pts[1]['pt'], eye_pts[1]['bbox'])
    elif len(eye_pts) == 1 and 2 in final_pts:
        single, nose = eye_pts[0], final_pts[2][0]
        final_pts[1 if single['x'] > nose[0] else 0] = (single['pt'], single['bbox'])
    elif len(eye_pts) == 1:
        final_pts[0] = (eye_pts[0]['pt'], eye_pts[0]['bbox']) 
    return final_pts

def get_model_warp_matrix_inference(rgb_img, thm_img, detect_map, p2p_model, device):
    h_orig, w_orig = rgb_img.shape[:2]
    thm_h, thm_w = thm_img.shape[:2]
    
    pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
    if not pts_found: return None, None, {}, "None"

    rgb_pts_in = [pts_found[cid] if cid in pts_found else [0.0,0.0] for cid in P2P_CLS_IDS]
    model_in = [[p[0]*IMAGE_SIZE/w_orig, p[1]*IMAGE_SIZE/h_orig] for p in rgb_pts_in]

    t_rgb = torch.as_tensor(rgb_img).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
    t_thm = torch.as_tensor(thm_img).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
    
    t_rgb = F.interpolate(t_rgb.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    t_thm = F.interpolate(t_thm.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    
    t_pts = torch.tensor([model_in], dtype=torch.float32).to(device).half()

    with torch.inference_mode():
        pred = p2p_model(t_rgb.unsqueeze(0), t_thm.unsqueeze(0), t_pts).cpu().float().numpy()[0]

    dst_pts = [[pred[i][0]*thm_w/IMAGE_SIZE, pred[i][1]*thm_h/IMAGE_SIZE] for i in range(3) if i in pts_found]
    src_np = np.float32([pts_found[i] for i in range(3) if i in pts_found])
    dst_np = np.float32(dst_pts)

    M, t_type = None, "None"
    if len(src_np) >= 3:
        M = np.vstack([cv2.getAffineTransform(src_np[:3], dst_np[:3]), [0,0,1]])
        t_type = "Affine"
    elif len(src_np) == 2:
        M_p, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
        if M_p is not None: M = np.vstack([M_p, [0,0,1]])
        t_type = "Similarity"
    elif len(src_np) == 1:
        dx, dy = dst_np[0] - src_np[0]
        M = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
        t_type = "Translation"
        
    return M, dst_np, pts_found, t_type

# -------------------------------------------------------------
# --- MAIN EXECUTION ---
# -------------------------------------------------------------

def run_pipeline_step(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Step 1: Background Fast Mode] Processing: {os.path.basename(args.rgb)}")
    print(f"   Using Device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        detect_model = YOLO(YOLOV8_KEYPOINT_PATH) 
        p2p_model = SwinTransformerP2P().to(device)
        p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=device, weights_only=True))
        if device.type == 'cuda': p2p_model.half() 
        p2p_model.eval()

        face_model = None
        if os.path.exists(YOLOV8_FACE_PATH):
            face_model = YOLO(YOLOV8_FACE_PATH)
            print("✅ 載入豬臉偵測模型成功！")
        else:
            print("⚠️ 未找到 face.pt，將跳過身分識別。")

        id_model = None
        if face_model and os.path.exists(args.id_model) and os.path.exists(args.id_feat):
            id_checkpoint = torch.load(args.id_model, map_location=device)
            num_classes = len(id_checkpoint["classes"]) if "classes" in id_checkpoint else id_checkpoint['model']['classifier_cnn.weight'].shape[0]
            
            id_model = ECBHybridModel(num_classes=num_classes).to(device)
            id_model.load_state_dict(id_checkpoint["model"])
            if device.type == 'cuda': id_model.half() 
            id_model.eval()

            id_features = np.load(args.id_feat)
            id_labels = np.load(args.id_label)
            id_features_torch = torch.tensor(id_features, dtype=torch.float16 if device.type=='cuda' else torch.float32, device=device)
            
            mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            if device.type == 'cuda':
                mean_tensor, std_tensor = mean_tensor.half(), std_tensor.half()
            
            print("✅ 載入身分特徵比對模型成功！")

    except Exception as e:
        print(f"❌ 模型載入發生錯誤: {e}"); return

    cap_rgb = cv2.VideoCapture(args.rgb)
    cap_thm = cv2.VideoCapture(args.thm)
    if not cap_rgb.isOpened() or not cap_thm.isOpened(): return

    roi_records = []
    frame_count = 0
    current_pig_id = "Unknown"

    start_total_time = time.time()
    
    # 🌟 幫助轉換並產生新框的工具函式
    def get_warped_box(box, matrix):
        pts_rgb = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
        pts_poly = cv2.perspectiveTransform(pts_rgb, matrix).reshape(-1, 2)
        x1, y1 = np.min(pts_poly[:,0]), np.min(pts_poly[:,1])
        x2, y2 = np.max(pts_poly[:,0]), np.max(pts_poly[:,1])
        return pts_poly, int(x1), int(y1), int(x2), int(y2)

    while True:
        ret_rgb, rgb_img = cap_rgb.read()
        ret_thm, thm_img = cap_thm.read()
        if not ret_rgb or not ret_thm: break
        
        current_frame = frame_count 
        frame_count += 1

        yolo_res = detect_model.predict(rgb_img, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
        detect_map = process_yolo_detection_and_sort_points(yolo_res)
        
        if 2 in detect_map:
            nose_y = detect_map[2][0][1]
            for eid in [0,1]:
                if eid in detect_map and detect_map[eid][0][1] > nose_y:
                    detect_map[eid], detect_map[2] = detect_map[2], detect_map[eid]
        if 0 in detect_map and 1 in detect_map and detect_map[0][0][0] > detect_map[1][0][0]:
             detect_map[0], detect_map[1] = detect_map[1], detect_map[0]

        M, dst_pts, raw_pts, t_type = get_model_warp_matrix_inference(rgb_img, thm_img, detect_map, p2p_model, device)
        
        if id_model and face_model and (current_frame % ID_CHECK_INTERVAL == 0):
            face_res = face_model.predict(
                rgb_img,
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                classes=[1],
                verbose=False,
                device=device,
                half=(device.type=='cuda')
            )[0]
            face_box = None
            for box in face_res.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if cls_id == 1:
                    face_box = box.xyxy[0].cpu().numpy()
                    break
            if face_box is None and len(face_res.boxes) > 0:
                face_box = face_res.boxes[0].xyxy[0].cpu().numpy()
            if face_box is not None:
                x1_f, y1_f, x2_f, y2_f = map(int, face_box)
                rgb_h, rgb_w = rgb_img.shape[:2]
                
                pad_x, pad_y = int((x2_f - x1_f) * 0.1), int((y2_f - y1_f) * 0.1)
                x1_f, y1_f = max(0, x1_f - pad_x), max(0, y1_f - pad_y)
                x2_f, y2_f = min(rgb_w, x2_f + pad_x), min(rgb_h, y2_f + pad_y)
                
                face_crop = rgb_img[y1_f:y2_f, x1_f:x2_f]
                
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    
                    t_face = torch.as_tensor(face_rgb).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
                    t_face = t_face.unsqueeze(0)
                    t_face = F.interpolate(t_face, size=(224, 224), mode='bilinear', align_corners=False)
                    img_tensor_id = (t_face - mean_tensor) / std_tensor
                    
                    with torch.inference_mode():
                        embedding_id = id_model.extract_features(img_tensor_id)
                        sim_id = F.cosine_similarity(embedding_id, id_features_torch)
                        best_idx = torch.argmax(sim_id).item()
                        current_pig_id = get_pig_id(str(id_labels[best_idx]))

        if M is not None:
            # 取得座標即可，完全不要畫圖
            nx1, ny1, nx2, ny2 = 0, 0, 0, 0
            lx1, ly1, lx2, ly2 = 0, 0, 0, 0
            rx1, ry1, rx2, ry2 = 0, 0, 0, 0
            has_nose = False
            
            if 2 in detect_map:
                _, box = detect_map[2]
                _, nx1, ny1, nx2, ny2 = get_warped_box(box, M)
                has_nose = True

            if 0 in detect_map:
                _, box = detect_map[0]
                _, lx1, ly1, lx2, ly2 = get_warped_box(box, M)

            if 1 in detect_map:
                _, box = detect_map[1]
                _, rx1, ry1, rx2, ry2 = get_warped_box(box, M)

            if has_nose:
                roi_records.append([current_frame, 1, nx1, ny1, nx2, ny2, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2, current_pig_id])
                
        # 加一點進度提示（每 100 幀閃過一次）
        if current_frame % 100 == 0 and current_frame > 0:
            print(f"   [Background] Processed {current_frame} frames...")

    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    csv_path = os.path.join(args.out_dir, "roi_boxes.csv")
    df = pd.DataFrame(roi_records, columns=[
        'frame_th', 'valid', 
        'x1_th', 'y1_th', 'x2_th', 'y2_th',      
        'leye_x1', 'leye_y1', 'leye_x2', 'leye_y2', 
        'reye_x1', 'reye_y1', 'reye_x2', 'reye_y2', 
        'pig_id'
    ])
    df.to_csv(csv_path, index=False)
    
    cap_rgb.release(); cap_thm.release()
    print(f"\n✅ [Step 1 Done] 背景高速純推理模式完成！")
    print(f"📍 ROI CSV 已儲存至: {csv_path}")
    print(f"⏱️ 處理總耗時: {total_time:.2f} 秒 | 總幀數: {frame_count}")
    print(f"🚀 平均滿載 FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, help="Path to RGB video")
    parser.add_argument("--thm", type=str, help="Path to Thermal video")
    parser.add_argument("--out-dir", type=str, default="data/test_output", help="Output directory")
    
    parser.add_argument("--id-model", type=str, default=os.path.join(ROOT_DIR, "pig_ecb_best.pth"))
    parser.add_argument("--id-feat", type=str, default=os.path.join(ROOT_DIR, "pig_features_ecb.npy"))
    parser.add_argument("--id-label", type=str, default=os.path.join(ROOT_DIR, "pig_labels_ecb.npy"))
    
    args = parser.parse_args()

    if args.rgb is None:
        args.rgb = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
        args.thm = os.path.join(ROOT_DIR, "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")

    run_pipeline_step(args)