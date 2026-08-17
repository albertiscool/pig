import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters（或其他共用資源），回退到專案根目錄
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
# --------------------------------------------------------
# 1. 定義新的 ECB Hybrid Model (需與訓練時一致)
# --------------------------------------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # ========== Branch 1: ViT Encoder (E1) ==========
        # 即使推論不用，載入權重時仍需定義結構以免報錯
        self.vit_model = models.vit_b_16(weights=None) 
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        # ========== Branch 2: CNN Encoder (E2) ==========
        # 這是我們註冊/辨識時主要使用的分支
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity() # 這樣輸出就是 512 維特徵
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def forward(self, x):
        # 訓練時的 forward (回傳 logits)
        feat_vit = self.vit_model(x)
        logits_vit = self.classifier_vit(feat_vit)
        feat_cnn = self.cnn_model(x)
        logits_cnn = self.classifier_cnn(feat_cnn)
        return logits_vit, logits_cnn

    def extract_features(self, x):
        """
        🔥 新增此函式：專門用於註冊與測試
        根據論文 Testing phase，只使用 CNN Encoder (E2) 提取特徵
        """
        feat_cnn = self.cnn_model(x) # (B, 512)
        return F.normalize(feat_cnn, dim=1) # 正規化很重要

# --------------------------------------------------------
# 2. 註冊 Unknown 豬隻功能
# --------------------------------------------------------
def register_unknown_pigs(
    model_path,
    unknown_feature_dir,
    known_feature_path,
    known_label_path,
    save_dir="./",
    diff_threshold=0.35
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------
    # 圖片前處理 (需與訓練時一致)
    # ------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet/ViT 標準輸入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ------------------------
    # 載入 unknown dataset
    # ------------------------
    dataset = datasets.ImageFolder(unknown_feature_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    pig_names = dataset.classes
    print(f"🐷 Unknown 豬數量: {len(pig_names)} → {pig_names}")

    # ------------------------
    # 載入模型
    # ------------------------
    print(f"🔄 載入模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 取得類別數量以初始化模型
    # 假設 checkpoint 裡有存 classes (上次訓練程式有存)
    if "classes" in checkpoint:
        num_known_pigs = len(checkpoint["classes"])
    else:
        # 如果沒存，需手動指定或根據權重推算 (這裡預設先給個數字，如果不影響 load_state_dict)
        # 建議確保訓練程式有存 "classes"
        print("⚠️ Checkpoint 中無 classes 資訊，嘗試從 classifier_cnn 推斷...")
        num_known_pigs = checkpoint['model']['classifier_cnn.weight'].shape[0]

    print(f"📦 模型已知分類數: {num_known_pigs}")

    # 初始化新模型
    model = ECBHybridModel(num_classes=num_known_pigs).to(device)
    
    # 載入權重
    model.load_state_dict(checkpoint["model"])
    model.eval()

    features_per_pig = {name: [] for name in pig_names}

    # ------------------------
    # Step 1️⃣ 提取每張圖 embedding (使用 extract_features)
    # ------------------------
    print("🚀 開始提取特徵 (使用 CNN 分支)...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="🔍 生成特徵"):
            imgs = imgs.to(device)
            # 關鍵修改：呼叫 extract_features 而不是 model(imgs)
            embeddings = model.extract_features(imgs)
            
            for emb, lbl in zip(embeddings, labels):
                pig_name = pig_names[lbl.item()]
                features_per_pig[pig_name].append(emb.cpu().numpy())

    new_features, new_labels = [], []

    # ------------------------
    # Step 2️⃣ 聚合每隻豬的 embedding
    # ------------------------
    for pig_name, feats in features_per_pig.items():
        if len(feats) == 0:
            continue

        feats = np.stack(feats) # (N, 512)
        
        # 計算平均向量
        mean_vec = np.mean(feats, axis=0)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-8

        # 計算與平均向量的 cosine 相似度
        sims = np.dot(feats, mean_vec) / (np.linalg.norm(feats, axis=1) * np.linalg.norm(mean_vec) + 1e-8)
        diffs = 1 - sims 

        # 找出差異大的 embedding
        large_diff_idx = np.where(diffs > diff_threshold)[0]

        # 儲存主要 embedding
        new_features.append(mean_vec)
        new_labels.append(pig_name)

        if len(large_diff_idx) > 0:
            print(f"⚠️ {pig_name}: 有 {len(large_diff_idx)} 張圖與平均差異過大，新增額外特徵向量")
            extra_vec = np.mean(feats[large_diff_idx], axis=0)
            extra_vec /= np.linalg.norm(extra_vec) + 1e-8
            new_features.append(extra_vec)
            new_labels.append(pig_name + "_v2")

        print(f"✅ {pig_name}: 已生成 {len(feats)} 張 → {len(new_features) - len(new_labels) + 1 if len(large_diff_idx)==0 else 2} 向量")

    new_features = np.stack(new_features)

    # ------------------------
    # Step 3️⃣ 儲存資料庫 (建議不與舊混用)
    # ------------------------
    # 注意：因為模型架構變了 (ResNet18 backbone)，特徵空間完全不同。
    # 舊的 feature file (64 dim) 與新的 (512 dim) 不能合併。
    # 這裡預設直接儲存新的，若要合併需確保 known_feature_path 也是用新模型跑出來的。
    
    save_feat_name = "pig_features_ecb.npy"
    save_label_name = "pig_labels_ecb.npy"

    if os.path.exists(os.path.join(save_dir, save_feat_name)):
        print("⚠️ 偵測到現有 ECB 特徵庫，進行合併...")
        try:
            old_features = np.load(os.path.join(save_dir, save_feat_name))
            old_labels = np.load(os.path.join(save_dir, save_label_name))
            
            if old_features.shape[1] != new_features.shape[1]:
                print(f"❌ 維度不符！舊: {old_features.shape[1]}, 新: {new_features.shape[1]}。將強制覆蓋舊檔。")
                all_features = new_features
                all_labels = np.array(new_labels)
            else:
                all_features = np.concatenate([old_features, new_features], axis=0)
                all_labels = np.concatenate([old_labels, np.array(new_labels)], axis=0)
        except Exception as e:
            print(f"❌ 讀取舊檔失敗: {e}，將建立新檔。")
            all_features = new_features
            all_labels = np.array(new_labels)
    else:
        all_features = new_features
        all_labels = np.array(new_labels)
        print("🆕 建立新的 feature 資料庫")

    np.save(os.path.join(save_dir, save_feat_name), all_features)
    np.save(os.path.join(save_dir, save_label_name), all_labels)

    print(f"💾 已儲存更新後的特徵庫至 {save_dir}")
    print(f"📊 總筆數: {len(all_labels)}, 特徵維度: {all_features.shape[1]}")

# ------------------------
# 執行範例
# ------------------------
if __name__ == "__main__":
    register_unknown_pigs(
        model_path=os.path.join(ROOT_DIR, "pig_ecb_best.pth"), # 改成您的新模型路徑
        unknown_feature_dir=os.path.join(ROOT_DIR, "dataset_split", "unknown", "feature"),
        known_feature_path=None, # 設為 None 代表不讀取舊的 (因為維度不同)
        known_label_path=None,
        save_dir=r"C:\Users\User\Desktop\final_pig_o",
        diff_threshold=0.35
    )