# -*- coding: utf-8 -*-
"""
pig_identity.py
================================================================================
🐷 豬隻個體身分辨識與未知個體自動註冊模組 (Pig ReID & Auto-Registration)
================================================================================"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from collections import deque, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ID_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "pig_ecb_best.pth")
DEFAULT_GALLERY_PATH  = os.path.join(SCRIPT_DIR, "parameters", "registered_pigs_gallery.pt")
DEFAULT_GALLERY_DIR   = os.path.join(SCRIPT_DIR, "parameters", "registered_pigs")


class ECBHybridModel(nn.Module):
    """
    與訓練時架構完全一致的 ECBHybridModel (ViT-B/16 + ResNet18)
    推論階段使用 CNN (ResNet18) 分支進行極速特徵提取與分類 (單次推論僅 ~1.8ms)
    """
    def __init__(self, num_classes):
        super().__init__()
        # ViT 分支 (初始化以供權重對齊載入)
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        # CNN 主力推論分支
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def forward(self, x):
        feat_cnn = self.cnn_model(x)
        logits_cnn = self.classifier_cnn(feat_cnn)
        return logits_cnn

    def extract_features(self, x):
        feat_cnn = self.cnn_model(x)
        logits_cnn = self.classifier_cnn(feat_cnn)
        feat_norm = F.normalize(feat_cnn.float(), p=2, dim=1)
        return logits_cnn, feat_norm


class PigIdentifier:
    """
    豬隻個體身分即時辨識器 (支援未知豬隻開局自動註冊與特徵庫比對)
    """
    def __init__(self, model_path=DEFAULT_ID_MODEL_PATH, device="cuda", half=True,
                 conf_threshold=0.65, history_len=15, auto_register=True,
                 gallery_path=DEFAULT_GALLERY_PATH,
                 # Calibrated from external shaojiang eval: EER=0.8983, Youden=0.9109 (0.82 caused 87.5% FPR)
                 sim_threshold=0.90):
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.half = half and (self.device == "cuda")
        self.conf_threshold = conf_threshold
        self.sim_threshold = sim_threshold
        self.auto_register = auto_register
        self.gallery_path = gallery_path
        self.gallery_img_dir = DEFAULT_GALLERY_DIR
        os.makedirs(self.gallery_img_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到豬隻身分辨識模型權重: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        self.classes = ckpt.get("classes", [])
        num_classes = len(self.classes) if len(self.classes) > 0 else 17
        
        self.model = ECBHybridModel(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        if self.half:
            self.model.half()
            
        # 載入已註冊的新豬隻特徵庫 (Open-Set Registered Gallery)
        self.registered_gallery = {}
        if os.path.exists(self.gallery_path):
            try:
                loaded_gal = torch.load(self.gallery_path, map_location="cpu")
                if isinstance(loaded_gal, dict):
                    self.registered_gallery = loaded_gal
            except Exception:
                self.registered_gallery = {}
                
        # 影像前處理 Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 滑動窗口歷史隊列與未知豬隻註冊緩衝
        self.history = deque(maxlen=history_len)
        self.current_id = "Detecting..."
        self.current_conf = 0.0
        self.unknown_feats = []
        self.session_locked_id = None
        self.last_registered_pig_event = None

        # 多角度多照片收集緩衝 (Multi-Shot Face Gallery for New Pigs)
        self.multishot_crops = {}  # {pig_id: [(crop_img, sharpness)]}
        self.multishot_feats = {}  # {pig_id: [feat1, feat2, ...]}
        self.target_shots = 5      # 每個新個體自動收集 5 張多角度高品質特寫
        
        # 預熱
        dummy = torch.zeros((1, 3, 224, 224), device=self.device, dtype=torch.float16 if self.half else torch.float32)
        with torch.no_grad():
            _ = self.model(dummy)
            
        print(f"[*] [豬隻身分辨識就緒] 已載入 {num_classes} 隻訓練個體 + {len(self.registered_gallery)} 隻註冊個體 | 門檻: {self.conf_threshold:.2f} | 自動註冊: {'啟用' if self.auto_register else '停用'}")

    def _predict_with_feat(self, face_crop_bgr):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None, 0.0, None
            
        h, w = face_crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return None, 0.0, None

        crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        if self.half:
            tensor = tensor.half()
            
        with torch.no_grad():
            logits, feat_norm = self.model.extract_features(tensor)
            probs = F.softmax(logits.float(), dim=1)[0]
            top_idx = torch.argmax(probs).item()
            conf = float(probs[top_idx].item())
            
        feat_cpu = feat_norm[0].cpu()
        raw_id = self.classes[top_idx] if top_idx < len(self.classes) else f"Pig_{top_idx}"
        
        # 1. 若達原始 17 隻訓練集門檻，直接回傳已知 ID
        if conf >= self.conf_threshold:
            return raw_id, conf, feat_cpu
            
        # 2. 比對已註冊的新豬隻特徵庫 (Cosine Similarity)
        if len(self.registered_gallery) > 0:
            best_reg_id = None
            best_sim = -1.0
            for reg_id, reg_feat in self.registered_gallery.items():
                sim = float(torch.dot(feat_cpu, reg_feat.view(-1)).item())
                if sim > best_sim:
                    best_sim = sim
                    best_reg_id = reg_id
            if best_sim >= self.sim_threshold:
                return best_reg_id, best_sim, feat_cpu
                
        return "Unknown", conf, feat_cpu

    def predict(self, face_crop_bgr):
        pred_id, conf, _ = self._predict_with_feat(face_crop_bgr)
        return pred_id, conf

    def register_new_pig(self, feat_list, face_crop_bgr=None, custom_id=None):
        """
        將未知豬隻特徵向量平均正規化後，正式註冊為新豬隻 ID 並寫入磁碟
        """
        if len(feat_list) == 0:
            return None
        stacked = torch.stack(feat_list, dim=0)
        mean_feat = F.normalize(torch.mean(stacked, dim=0, keepdim=True), p=2, dim=1)[0].cpu()
        
        if custom_id is not None:
            new_id = custom_id
        else:
            next_idx = len(self.registered_gallery) + 1
            while f"Pig_Reg_{next_idx:03d}" in self.registered_gallery:
                next_idx += 1
            new_id = f"Pig_Reg_{next_idx:03d}"
            
        self.registered_gallery[new_id] = mean_feat
        reg_time = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            torch.save(self.registered_gallery, self.gallery_path)
            
            # 建立多角度相簿目錄並儲存首張代表特寫
            album_dir = os.path.join(self.gallery_img_dir, new_id)
            os.makedirs(album_dir, exist_ok=True)
            if face_crop_bgr is not None and face_crop_bgr.size > 0:
                img_path = os.path.join(self.gallery_img_dir, f"{new_id}.jpg")
                cv2.imwrite(img_path, face_crop_bgr)
                cv2.imwrite(os.path.join(album_dir, "face_01.jpg"), face_crop_bgr)
                gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
                s_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                self.multishot_crops[new_id] = [(face_crop_bgr.copy(), s_val)]
                self.multishot_feats[new_id] = list(feat_list)
                
            # 建立個體資料記錄與註冊清冊總表
            import json
            import csv
            info = {
                "pig_id": new_id,
                "registered_time": reg_time,
                "face_image": f"{new_id}.jpg",
                "embedding_dim": int(mean_feat.shape[0]),
                "status": "Auto-Registered Unknown Pig",
                "album_dir": f"registered_pigs/{new_id}/"
            }
            self.last_registered_pig_event = info

            profile_path = os.path.join(self.gallery_img_dir, f"{new_id}_profile.json")
            with open(profile_path, "w", encoding="utf-8") as pf:
                json.dump(info, pf, ensure_ascii=False, indent=4)

            catalog_csv = os.path.join(self.gallery_img_dir, "registered_pigs_catalog.csv")
            csv_exists = os.path.exists(catalog_csv)
            with open(catalog_csv, "a", newline="", encoding="utf-8-sig") as cf:
                writer = csv.writer(cf)
                if not csv_exists:
                    writer.writerow(["Registered_Time", "Pig_ID", "Status", "Face_Image_File", "Embedding_Dim"])
                writer.writerow([reg_time, new_id, "Auto-Registered Unknown Pig", f"{new_id}.jpg", mean_feat.shape[0]])
        except Exception as e:
            print(f"[!] 儲存註冊特徵庫/檔案失敗: {e}")
            
        print(f"[✓] [新豬隻自動建檔] 偵測到未知個體，已自動建立新身分: {new_id} 並保存臉部特寫與個體檔案！(總庫存: {len(self.registered_gallery)} 隻)")
        return new_id

    def _collect_multishot_sample(self, pig_id, face_crop_bgr, feat_cpu):
        """
        為新註冊豬隻在鏡頭前自動持續採集多角度特寫 (最多 target_shots 張)，並動態更新特徵重心向量
        """
        if face_crop_bgr is None or feat_cpu is None:
            return
            
        cur_list = self.multishot_crops.get(pig_id, [])
        if len(cur_list) >= self.target_shots:
            return
            
        h, w = face_crop_bgr.shape[:2]
        if h < 45 or w < 45:
            return
            
        # 計算清晰度評分 (Laplacian 方差，低於 35 代表動態模糊/晃動)
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 35.0:
            return
            
        album_dir = os.path.join(self.gallery_img_dir, pig_id)
        os.makedirs(album_dir, exist_ok=True)
        
        self.multishot_crops.setdefault(pig_id, []).append((face_crop_bgr.copy(), sharpness))
        self.multishot_feats.setdefault(pig_id, []).append(feat_cpu)
        shot_idx = len(self.multishot_crops[pig_id])
        
        shot_path = os.path.join(album_dir, f"face_{shot_idx:02d}.jpg")
        cv2.imwrite(shot_path, face_crop_bgr)
        
        # 動態計算特徵重心 (Centroid) 並正規化更新特徵庫
        feats_all = self.multishot_feats[pig_id]
        stacked = torch.stack(feats_all, dim=0)
        mean_feat = F.normalize(torch.mean(stacked, dim=0, keepdim=True), p=2, dim=1)[0].cpu()
        self.registered_gallery[pig_id] = mean_feat
        try:
            torch.save(self.registered_gallery, self.gallery_path)
        except Exception:
            pass
            
        # 始終挑選目前清晰度最高的一張作為對外代表大頭照
        best_crop = max(self.multishot_crops[pig_id], key=lambda x: x[1])[0]
        cv2.imwrite(os.path.join(self.gallery_img_dir, f"{pig_id}.jpg"), best_crop)
        
        if shot_idx == self.target_shots:
            print(f"[★ 多角度特徵庫建立完成] {pig_id} 已成功採集 {self.target_shots} 張不同角度高清特寫，特徵重心融合升級！")

    def update(self, face_crop_bgr):
        """
        更新當前辨識結果；若開局碰到 Unknown 豬隻且啟用 auto_register，自動為其註冊新 ID
        """
        pred_id, conf, feat_cpu = self._predict_with_feat(face_crop_bgr)
        if pred_id is None:
            return self.current_id, self.current_conf
            
        if pred_id == "Unknown":
            if self.auto_register and self.session_locked_id is None and feat_cpu is not None:
                self.unknown_feats.append(feat_cpu)
                # 累積 3 次穩定臉部特徵即立刻完成註冊
                if len(self.unknown_feats) >= 3:
                    new_id = self.register_new_pig(self.unknown_feats, face_crop_bgr)
                    self.unknown_feats.clear()
                    if new_id is not None:
                        self.session_locked_id = new_id
                        self.history.clear()
                        for _ in range(5):
                            self.history.append((new_id, 0.95))
                        self.current_id = new_id
                        self.current_conf = 0.95
                        return self.current_id, self.current_conf
            elif self.session_locked_id is not None:
                # 若本次為新註冊豬隻，持續採集不同影格之高清多角度特寫 (最多 target_shots 張)
                if str(self.session_locked_id).startswith("Pig_Reg_"):
                    self._collect_multishot_sample(self.session_locked_id, face_crop_bgr, feat_cpu)
                return self.current_id, self.current_conf
        else:
            # 若成功辨識為已知或已註冊 ID，清空未知累積並鎖定本次量測身分
            if len(self.unknown_feats) > 0:
                self.unknown_feats.clear()
            if self.session_locked_id is None:
                self.session_locked_id = pred_id
            if str(self.session_locked_id).startswith("Pig_Reg_"):
                self._collect_multishot_sample(self.session_locked_id, face_crop_bgr, feat_cpu)
                
        self.history.append((pred_id, conf))
        
        if len(self.history) > 0:
            id_counts = Counter([item[0] for item in self.history])
            top_id, count = id_counts.most_common(1)[0]
            confs = [item[1] for item in self.history if item[0] == top_id]
            avg_conf = float(np.mean(confs))
            
            if count >= max(1, len(self.history) // 3):
                self.current_id = top_id
                self.current_conf = avg_conf
                
        return self.current_id, self.current_conf

    def reset(self):
        """重設辨識緩衝區"""
        self.history.clear()
        self.unknown_feats.clear()
        self.session_locked_id = None
        self.current_id = "Detecting..."
        self.multishot_crops.clear()
        self.multishot_feats.clear()
        self.current_conf = 0.0
