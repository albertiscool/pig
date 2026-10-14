import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# --------------------------------------------------------
# 1. 模型架構定義 (ECB 與 Swin Transformer)
# --------------------------------------------------------

class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(768, num_classes)
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(512, num_classes)

    def extract_features(self, x):
        feat_cnn = self.cnn_model(x)
        return F.normalize(feat_cnn, dim=1)

class SwinBackbone(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        swin = models.swin_t(weights=None)
        self.features = swin.features 
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, feat_dim))
    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 3, 1, 2)

class SwinTransformerP2P(nn.Module):
    def __init__(self, feat_dim=256, nhead=8, num_layers=6):
        super().__init__()
        self.backbone = SwinBackbone(feat_dim=feat_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos_embed = nn.Linear(2, feat_dim)
        self.regression_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 2))

    def bilinear_sample(self, fmap, pts, img_size=640):
        B, C, H, W = fmap.shape
        x = pts[..., 0] / img_size * 2 - 1
        y = pts[..., 1] / img_size * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2) 
        feat = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear')
        return feat.squeeze(3).permute(0, 2, 1)

    def forward(self, rgb, thm, rgb_pts, img_size=640):
        feat_rgb = self.backbone(rgb)
        feat_thm = self.backbone(thm)
        memory = feat_thm.flatten(2).permute(0, 2, 1)
        query_feat = self.bilinear_sample(feat_rgb, rgb_pts, img_size)
        query_pos = self.query_pos_embed(rgb_pts / img_size)
        query = query_feat + query_pos
        output = self.decoder(query, memory)
        return self.regression_head(output)

# --------------------------------------------------------
# 2. 核心功能：座標映射與動態繪圖 (修正 dst_dst 錯誤)
# --------------------------------------------------------

def run_optimized_registration(rgb_path, thm_path, out_dir, weights):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    
    print("🚀 載入模型中...")
    yolo_face = YOLO(weights['face'])
    yolo_yang = YOLO(weights['yang']) 
    
    p2p_model = SwinTransformerP2P().to(device)
    p2p_model.load_state_dict(torch.load(weights['p2p'], map_location=device))
    p2p_model.eval()
    
    ckpt = torch.load(weights['ecb'], map_location=device)
    recognizer = ECBHybridModel(num_classes=len(ckpt["classes"])).to(device)
    recognizer.load_state_dict(ckpt["model"])
    recognizer.eval()
    
    known_features = torch.tensor(np.load(weights['feat']), dtype=torch.float32).to(device)
    known_labels = np.load(weights['labels'])

    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_thm = cv2.VideoCapture(thm_path)
    w_rgb, h_rgb = int(cap_rgb.get(3)), int(cap_rgb.get(4))
    w_thm, h_thm = int(cap_thm.get(3)), int(cap_thm.get(4))
    fps = int(cap_rgb.get(5)) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mix = cv2.VideoWriter(os.path.join(out_dir, 'out_blended.mp4'), fourcc, fps, (w_thm, h_thm))

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while cap_rgb.isOpened():
        ret1, frame_rgb = cap_rgb.read()
        ret2, frame_thm = cap_thm.read()
        if not ret1 or not ret2: break
        
        vis_rgb, vis_thm = frame_rgb.copy(), frame_thm.copy()

        # 1. 偵測臉部與身份
        face_results = yolo_face.predict(frame_rgb, conf=0.5, verbose=False)
        for box in face_results[0].boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            face_img = frame_rgb[by1:by2, bx1:bx2]
            if face_img.size == 0: continue
            
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = recognizer.extract_features(face_tensor)
                sim = F.cosine_similarity(emb, known_features)
                max_sim, best_idx = torch.max(sim, 0)
                name = str(known_labels[best_idx.item()]).split('_')[0] if max_sim > 0.7 else "Unknown"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # 2. 關鍵點偵測與【物理校正】
            yang_results = yolo_yang.predict(frame_rgb, imgsz=640, conf=0.4, verbose=False)[0]
            raw_pts = {} # 暫存 YOLO 原始點位
            eye_list = []
            
            for y_box in yang_results.boxes:
                yx1, yy1, yx2, yy2 = y_box.xyxy[0].cpu().numpy()
                ycid = int(y_box.cls[0].item())
                
                if ycid == 2: # 鼻子 ID
                    # 記錄鼻子的四個頂點座標
                    nose_pts_rgb = np.float32([[yx1, yy1], [yx2, yy1], [yx2, yy2], [yx1, yy2]]).reshape(-1, 1, 2)
                    
                    if M is not None:
                        # 將鼻子的框也投影到熱成像上
                        nose_dst_poly = cv2.transform(nose_pts_rgb, M)
                        # 在 vis_thm 上畫出黃色的鼻子框
                        cv2.polylines(vis_thm, [np.int32(nose_dst_poly)], True, (0, 255, 255), 2)

            # 重構 P2P 所需結構 (0:左眼, 1:右眼, 2:鼻子)
            final_rgb_pts = {}
            if len(eye_list) >= 2:
                eye_list.sort(key=lambda x: x[0]) # 按 X 軸排序區分左右
                final_rgb_pts[0], final_rgb_pts[1] = eye_list[0], eye_list[1]
            elif len(eye_list) == 1:
                final_rgb_pts[0] = eye_list[0]
            
            if 2 in raw_pts:
                final_rgb_pts[2] = raw_pts[2]

            # 3. Transformer 映射計算
            M = None
            if len(final_rgb_pts) >= 1:
                img_size = 640
                model_input = []
                # 嚴格按照 [0, 1, 2] 填入模型
                for cid in [0, 1, 2]:
                    if cid in final_rgb_pts:
                        pt = final_rgb_pts[cid]
                        model_input.append([pt[0] * img_size / w_rgb, pt[1] * img_size / h_rgb])
                    else:
                        model_input.append([0.0, 0.0])
                
                rgb_t = torch.from_numpy(cv2.resize(frame_rgb, (640, 640)).transpose(2,0,1)).float().to(device).unsqueeze(0) / 255.0
                thm_t = torch.from_numpy(cv2.resize(frame_thm, (640, 640)).transpose(2,0,1)).float().to(device).unsqueeze(0) / 255.0
                pts_t = torch.tensor([model_input], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    pred_thm = p2p_model(rgb_t, thm_t, pts_t).cpu().numpy()[0]
                
                valid_src, valid_dst = [], []
                for i, cid in enumerate([0, 1, 2]):
                    if cid in final_rgb_pts:
                        valid_src.append(final_rgb_pts[cid])
                        valid_dst.append([pred_thm[i][0] * w_thm / img_size, pred_thm[i][1] * h_thm / img_size])
                
                # 計算轉換矩陣 (保持穩健性)
                src_np, dst_np = np.float32(valid_src), np.float32(valid_dst)
                if len(src_np) >= 3:
                    M = cv2.getAffineTransform(src_np[:3], dst_np[:3])
                elif len(src_np) == 2:
                    M, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
                elif len(src_np) == 1:
                    dx, dy = dst_np[0] - src_np[0]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])

            # 4. 繪製與投影
            label_text = f"{name} ({max_sim:.2f})"
            cv2.rectangle(vis_rgb, (bx1, by1), (bx2, by2), color, 4)
            cv2.putText(vis_rgb, label_text, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if M is not None:
                # 使用透視/仿射變換投影 BBox 四個角
                face_pts_rgb = np.float32([[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]]).reshape(-1,1,2)
                # 因為 M 是 2x3，用 cv2.transform；如果是 3x3 則用 perspectiveTransform
                dst_poly = cv2.transform(face_pts_rgb, M) 
                cv2.polylines(vis_thm, [np.int32(dst_poly)], True, color, 4, cv2.LINE_AA)
                cv2.putText(vis_thm, name, (int(dst_poly[0][0][0]), int(dst_poly[0][0][1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 5. 混合與顯示
        if M is not None:
            warped_rgb = cv2.warpAffine(vis_rgb, M, (w_thm, h_thm))
            blended = cv2.addWeighted(vis_thm, 0.5, warped_rgb, 0.5, 0)
        else:
            blended = vis_thm

        out_mix.write(blended)
        cv2.imshow("Pig Registration System", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_rgb.release(); cap_thm.release(); out_mix.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    _new_thm = os.path.join(ROOT_DIR, "parameters", "thm", "best_thermal_nose.pt")
    weights_config = {
        'face': os.path.join(ROOT_DIR, "parameters", "face.pt"),
        'yang': _new_thm if os.path.exists(_new_thm) else os.path.join(ROOT_DIR, "parameters", "thm", "best_yang.pt"),
        'p2p':  os.path.join(ROOT_DIR, "parameters", "thm", "swin_p2p_best_re.pth"),
        'ecb':  os.path.join(ROOT_DIR, "parameters", "pig_ecb_best.pth"),
        'feat': os.path.join(ROOT_DIR, "parameters", "pig_features_ecb.npy"),
        'labels': os.path.join(ROOT_DIR, "parameters", "pig_labels_ecb.npy")
    }
    run_optimized_registration(
        rgb_path = os.path.join(ROOT_DIR, "data", "test_vid", "15棟_295-6_11407031549_可見光_正面.mp4"),
        thm_path = os.path.join(ROOT_DIR, "data", "test_vid", "15棟_295-6_11407031549_熱成像_正面.mp4"),
        out_dir  = os.path.join(ROOT_DIR, "data", "test_output"),
        weights  = weights_config
    )