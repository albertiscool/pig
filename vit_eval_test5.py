import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# -----------------------------
# 1. 必須定義與訓練時相同的 ECBHybridModel
# -----------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ViT Branch (E1) [cite: 31, 35]
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(768, num_classes)

        # CNN Branch (E2) [cite: 34, 36]
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(512, num_classes)

    def forward(self, x):
        # 測試時根據論文，我們主要觀察 CNN 分支 
        feat_vit = self.vit_model(x)
        logits_vit = self.classifier_vit(feat_vit)
        feat_cnn = self.cnn_model(x)
        logits_cnn = self.classifier_cnn(feat_cnn)
        return logits_vit, logits_cnn

# -----------------------------
# 2. 測試主流程
# -----------------------------
def test_accuracy(dataset_root, model_path, device="cuda"):
    # 載入 Checkpoint
    print(f"🔄 載入模型權重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 從儲存的資訊中取得豬隻類別列表 
    class_names = checkpoint["classes"]
    num_classes = len(class_names)
    print(f"🐷 辨識類別數量: {num_classes}")

    # 初始化模型並載入權重
    model = ECBHybridModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 資料前處理 (需與訓練時的 transform_eval 一致)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 載入測試集
    test_dir = os.path.join(dataset_root, "test_known")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ 找不到測試資料夾: {test_dir}")
        
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 開始評估
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    print(f"🚀 開始測試 {len(test_dataset)} 張圖片...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 根據論文測試階段僅使用 CNN 分支輸出 
            _, logits_cnn = model(imgs)
            
            preds = torch.argmax(logits_cnn, dim=1)
            
            # 統計總體
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # 統計各類別
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # -----------------------------
    # 3. 輸出結果
    # -----------------------------
    print("\n" + "="*30)
    print(f"📊 總體測試準確率: {100 * correct / total:.2f}% ({correct}/{total})")
    print("="*30)

    print("\n📈 各類別詳細報告:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"豬隻 ID {class_names[i]:<10}: {acc:>6.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"豬隻 ID {class_names[i]:<10}: 無測試資料")

if __name__ == "__main__":
    # 請修改以下路徑為您的實際路徑
    DATASET_ROOT = os.path.join(ROOT_DIR, "dataset_split")
    MODEL_PATH = "pig_ecb_best.pth"
    
    test_accuracy(DATASET_ROOT, MODEL_PATH)