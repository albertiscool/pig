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
from threading import Thread
import queue
from ultralytics import YOLO
from torchvision import models
from torchvision.models import swin_t, Swin_T_Weights

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
# 🚀 【Batch 推論設定】 專為 5090 等高階計算卡設計，一次送多張圖片運算
BATCH_SIZE = 128

# -------------------------------------------------------------
# --- THREADED VIDEO I/O ---
# -------------------------------------------------------------
class ThreadedVideoReader:
    def __init__(self, path, queue_size=256):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        if self.cap.isOpened():
            self.thread.start()
            
    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    self.q.put((False, None))
                    return
                self.q.put((ret, frame))
            else:
                time.sleep(0.002)
                
    def read(self):
        if self.q.empty() and self.stopped:
            return False, None
        return self.q.get()
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def get(self, prop_id):
        return self.cap.get(prop_id)
        
    def stop(self):
        self.stopped = True
        
    def release(self):
        self.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

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
YOLOV8_KEYPOINT_PATH = os.path.join(ROOT_DIR, "parameters", "thm", "best_yang.pt")
YOLOV8_FACE_PATH = os.path.join(ROOT_DIR, "parameters", "face.pt")  

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

def get_warped_box(box, matrix):
    pts_rgb = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
    pts_poly = cv2.perspectiveTransform(pts_rgb, matrix).reshape(-1, 2)
    x1, y1 = np.min(pts_poly[:,0]), np.min(pts_poly[:,1])
    x2, y2 = np.max(pts_poly[:,0]), np.max(pts_poly[:,1])
    return pts_poly, int(x1), int(y1), int(x2), int(y2)

# -------------------------------------------------------------
# --- MAIN EXECUTION ---
# -------------------------------------------------------------

def run_pipeline_step(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Step 1: Batch Mode = {BATCH_SIZE}] Processing: {os.path.basename(args.rgb)}")
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

    # 使用多執行緒影片讀取器緩解 I/O 瓶頸
    cap_rgb = ThreadedVideoReader(args.rgb, queue_size=BATCH_SIZE*3)
    cap_thm = ThreadedVideoReader(args.thm, queue_size=BATCH_SIZE*3)
    
    if not cap_rgb.isOpened() or not cap_thm.isOpened(): return

    roi_records = []
    frame_count = 0
    current_pig_id = "Unknown"

    start_total_time = time.time()
    
    while True:
        rgb_batch = []
        thm_batch = []
        frames_idx = []
        
        # 一次抓取 BATCH_SIZE 張畫面
        for _ in range(BATCH_SIZE):
            ret_r, rgb_img = cap_rgb.read()
            ret_t, thm_img = cap_thm.read()
            if not ret_r or not ret_t: break
            rgb_batch.append(rgb_img)
            thm_batch.append(thm_img)
            frames_idx.append(frame_count)
            frame_count += 1
            
        if len(rgb_batch) == 0: break
        
        # 1. 批量 YOLO 預測 (將 List of 畫面丟入，真正榨出顯卡平行運算效能)
        yolo_res_list = detect_model.predict(rgb_batch, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))
        
        batch_t_rgb = []
        batch_t_thm = []
        batch_t_pts = []
        detect_maps = []
        
        # 準備 P2P 的張量 Batch
        for i in range(len(rgb_batch)):
            rgb_img = rgb_batch[i]
            thm_img = thm_batch[i]
            
            detect_map = process_yolo_detection_and_sort_points(yolo_res_list[i])
            
            # 對齊點位
            if 2 in detect_map:
                nose_y = detect_map[2][0][1]
                for eid in [0,1]:
                    if eid in detect_map and detect_map[eid][0][1] > nose_y:
                        detect_map[eid], detect_map[2] = detect_map[2], detect_map[eid]
            if 0 in detect_map and 1 in detect_map and detect_map[0][0][0] > detect_map[1][0][0]:
                 detect_map[0], detect_map[1] = detect_map[1], detect_map[0]

            detect_maps.append(detect_map)

            pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
            rgb_pts_in = [pts_found[cid] if cid in pts_found else [0.0,0.0] for cid in P2P_CLS_IDS]
            
            h_orig, w_orig = rgb_img.shape[:2]
            model_in = [[p[0]*IMAGE_SIZE/w_orig, p[1]*IMAGE_SIZE/h_orig] for p in rgb_pts_in]
            
            t_rgb = torch.from_numpy(rgb_img).permute(2,0,1).half() / 255.0
            t_thm = torch.from_numpy(thm_img).permute(2,0,1).half() / 255.0
            t_pts = torch.tensor(model_in, dtype=torch.float32).half()
            
            batch_t_rgb.append(t_rgb)
            batch_t_thm.append(t_thm)
            batch_t_pts.append(t_pts)
            
        # 將資料庫堆疊成為 [B, C, H, W] 透過 PCI-e 一次傳輸送入 GPU
        batch_t_rgb_tensor = torch.stack(batch_t_rgb).to(device, non_blocking=True)
        batch_t_thm_tensor = torch.stack(batch_t_thm).to(device, non_blocking=True)
        batch_t_pts_tensor = torch.stack(batch_t_pts).to(device, non_blocking=True)
        
        batch_t_rgb_tensor = F.interpolate(batch_t_rgb_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        batch_t_thm_tensor = F.interpolate(batch_t_thm_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        
        # 2. 批量 P2P 變換神經網路推論
        with torch.inference_mode():
            preds = p2p_model(batch_t_rgb_tensor, batch_t_thm_tensor, batch_t_pts_tensor).cpu().float().numpy()
            
        # 3. 卸載資料與計算幾何 M 矩陣 (CPU 輕量級分配)
        for i in range(len(rgb_batch)):
            current_frame = frames_idx[i]
            rgb_img = rgb_batch[i]
            detect_map = detect_maps[i]
            pts_found = {cid: pt for cid, (pt, _) in detect_map.items()}
            
            if not pts_found: continue
            
            # 定期身份識別：不影響主要 Batch pipeline，這只需要偶爾看單張
            if id_model and face_model and (current_frame % ID_CHECK_INTERVAL == 0):
                face_res = face_model.predict(rgb_img, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False, device=device, half=(device.type=='cuda'))[0]
                if len(face_res.boxes) > 0:
                    best_box = face_res.boxes[0].xyxy[0].cpu().numpy()
                    x1_f, y1_f, x2_f, y2_f = map(int, best_box)
                    rgb_h, rgb_w = rgb_img.shape[:2]
                    
                    pad_x, pad_y = int((x2_f - x1_f) * 0.1), int((y2_f - y1_f) * 0.1)
                    x1_f, y1_f = max(0, x1_f - pad_x), max(0, y1_f - pad_y)
                    x2_f, y2_f = min(rgb_w, x2_f + pad_x), min(rgb_h, y2_f + pad_y)
                    
                    face_crop = rgb_img[y1_f:y2_f, x1_f:x2_f]
                    
                    if face_crop.size > 0:
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        t_face = torch.from_numpy(face_rgb).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
                        t_face = t_face.unsqueeze(0)
                        t_face = F.interpolate(t_face, size=(224, 224), mode='bilinear', align_corners=False)
                        img_tensor_id = (t_face - mean_tensor) / std_tensor
                        
                        with torch.inference_mode():
                            embedding_id = id_model.extract_features(img_tensor_id)
                            sim_id = F.cosine_similarity(embedding_id, id_features_torch)
                            best_idx = torch.argmax(sim_id).item()
                            current_pig_id = get_pig_id(str(id_labels[best_idx]))

            pred = preds[i]
            thm_h, thm_w = thm_batch[i].shape[:2]
            
            dst_pts = [[pred[j][0]*thm_w/IMAGE_SIZE, pred[j][1]*thm_h/IMAGE_SIZE] for j in range(3) if j in pts_found]
            src_np = np.float32([pts_found[j] for j in range(3) if j in pts_found])
            dst_np = np.float32(dst_pts)
            
            M = None
            if len(src_np) >= 3:
                M = np.vstack([cv2.getAffineTransform(src_np[:3], dst_np[:3]), [0,0,1]])
            elif len(src_np) == 2:
                M_p, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
                if M_p is not None: M = np.vstack([M_p, [0,0,1]])
            elif len(src_np) == 1:
                dx, dy = dst_np[0] - src_np[0]
                M = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
                
            if M is not None:
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
                    
        # 顯示處理進度
        print(f"   [Batch {BATCH_SIZE}] Processed {frame_count} frames... GPU Utilization Maxed")

    # 結算
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
    print(f"\n✅ [Step 1 Done] 5090 Batch 平行極速引擎執行完畢！")
    print(f"📍 ROI CSV 已儲存至: {csv_path}")
    print(f"⏱️ 處理總耗時: {total_time:.2f} 秒 | 總幀數: {frame_count}")
    print(f"🚀 電腦瓶頸解除後 平均 FPS: {avg_fps:.2f}")

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