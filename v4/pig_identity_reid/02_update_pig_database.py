import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# 如果 v1 底下沒有 parameters，回退到上一層
if not os.path.exists(os.path.join(ROOT_DIR, "parameters")):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))

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

def main():
    print("="*50)
    print("🐷 特徵庫擴充程式 (Update Pig Database)")
    print("="*50)
    
    # 預設路徑：抓取剛剛 video_pipeline 儲存的目錄
    script_dir = os.path.abspath(os.path.dirname(__file__))
    collected_dir = os.path.join(script_dir, "data", "test_output", "collected_faces")
    
    if not os.path.exists(collected_dir) or len(os.listdir(collected_dir)) == 0:
        print("❌ 沒有找到需要新增的臉部資料。")
        print(f"請確認 [{collected_dir}] 資料夾內有以豬隻 ID 命名的子資料夾。")
        return

    # Check for parameters dir
    param_dir = os.path.join(ROOT_DIR, "parameters")
    model_path = os.path.join(param_dir, "pig_ecb_best.pth")
    feat_path = os.path.join(param_dir, "pig_features_ecb.npy")
    label_path = os.path.join(param_dir, "pig_labels_ecb.npy")

    # Load existing features
    if os.path.exists(feat_path) and os.path.exists(label_path):
        old_features = np.load(feat_path)
        old_labels = np.load(label_path)
    else:
        print("❌ 找不到現有的特徵庫 (.npy)，請確認 parameters 目錄下的檔案。")
        return

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 載入模型中 ({device})...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = len(checkpoint["classes"]) if "classes" in checkpoint else checkpoint['model']['classifier_cnn.weight'].shape[0]
        
        model = ECBHybridModel(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint["model"])
        if device == "cuda":
            model.half()
        model.eval()
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(collected_dir, transform=transform)
    if len(dataset) == 0:
        print("❌ 沒有找到圖片。")
        return
        
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    pig_names = dataset.classes

    features_per_pig = {name: [] for name in pig_names}

    print("\n🚀 開始提取新豬隻特徵...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if device == "cuda":
                imgs = imgs.half()
            embeddings = model.extract_features(imgs)
            for emb, lbl in zip(embeddings, labels):
                pig_name = pig_names[lbl.item()]
                features_per_pig[pig_name].append(emb.cpu().float().numpy())

    new_features = []
    new_labels = []

    print("\n" + "-"*30)
    for pig_name, feats in features_per_pig.items():
        if len(feats) == 0: continue
        feats = np.stack(feats)
        # 平均融合：將所有照片的特徵融合為單一個最具代表性的 Embedding
        mean_vec = np.mean(feats, axis=0)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-8
        new_features.append(mean_vec)
        new_labels.append(pig_name)
        print(f"✅ 加入豬隻 [{pig_name}] 的特徵向量 (彙整 {len(feats)} 張大頭照)")
    print("-"*30)

    if len(new_features) == 0:
        return

    new_features = np.stack(new_features)
    
    # 檢查是否已經存在相同名字的，避免重複疊加
    final_features = list(old_features)
    final_labels = list(old_labels)
    
    for nf, nl in zip(new_features, new_labels):
        if nl in final_labels:
            print(f"⚠️ 發現資料庫中已有 [{nl}] 的紀錄。目前採取【直接新增另一組向量】策略，以增加其辦識穩定度。")
        final_features.append(nf)
        final_labels.append(nl)

    final_features = np.stack(final_features)
    final_labels = np.array(final_labels)

    # 備份舊檔案
    shutil.copy(feat_path, feat_path + ".bak")
    shutil.copy(label_path, label_path + ".bak")

    # 儲存
    np.save(feat_path, final_features)
    np.save(label_path, final_labels)

    print("\n💾 特徵庫已成功更新！")
    print(f"   - 新增筆數: {len(new_labels)}")
    print(f"   - 目前資料庫總筆數: {len(final_labels)}")
    print(f"   - 舊檔案已備份為 .bak")
    
    ans = input("\n>> 是否要清理剛剛蒐集完畢的臉部資料夾，避免下次重複匯入？ [y/N]: ").strip().lower()
    if ans in ['y', 'yes']:
        shutil.rmtree(collected_dir)
        print("🗑️ 已清理 collected_faces 目錄。")

if __name__ == "__main__":
    main()
