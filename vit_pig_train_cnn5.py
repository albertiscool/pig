# ==============================================================
# train_pig_ecb_method.py
# 基於論文: Learning CNN on ViT: A Hybrid Model 
# ==============================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# -----------------------------
# 1. 雙分支混合模型 (Hybrid ViT + CNN)
# -----------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # ========== Branch 1: ViT Encoder (E1)  ==========
        # 論文使用 ViT-B/16。若顯卡記憶體不足，可改用 weights=None 或較小的 ViT
        print("🏗️ Loading ViT-B/16 (E1)...")
        self.vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        # 替換掉原本的分類頭，改為 Identity，我們自己接分類器
        self.vit_model.heads = nn.Identity()
        
        # Classifier F1 for ViT [cite: 31]
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        # ========== Branch 2: CNN Encoder (E2)  ==========
        # 論文使用 ResNet，這裡沿用您原本的 ResNet18
        print("🏗️ Loading ResNet-18 (E2)...")
        self.cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        # 移除原本的 fc 層
        self.cnn_model.fc = nn.Identity()
        
        # Classifier F2 for CNN [cite: 34]
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def forward(self, x):
        # --- ViT Flow ---
        # ViT 需要固定的輸入尺寸 (通常 224x224)，且計算量較大
        feat_vit = self.vit_model(x)       # (B, 768)
        logits_vit = self.classifier_vit(feat_vit)

        # --- CNN Flow ---
        feat_cnn = self.cnn_model(x)       # (B, 512)
        logits_cnn = self.classifier_cnn(feat_cnn)

        return logits_vit, logits_cnn

# -----------------------------
# 2. 評估函式 (Testing Phase)
# -----------------------------
def evaluate(model, dataloader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"→ {desc}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 根據論文 Methodology - Testing Phase [cite: 110]
            # 測試時只使用 CNN 分支 (E2 + F2) 進行預測
            _, logits_cnn = model(imgs)
            
            # 計算 Loss (僅計算 CNN 的 Loss 作為參考)
            loss = criterion(logits_cnn, labels)
            total_loss += loss.item()

            # 預測
            preds = torch.argmax(logits_cnn, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# -----------------------------
# 3. 訓練主流程
# -----------------------------
def train_model(
    dataset_root,
    num_epochs=50,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    device="cuda",
    pth_path=None
):
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test_known")

    # =====================
    # 資料增強 (保持不變)
    # =====================
    # ViT 對輸入尺寸較敏感，建議固定為 224
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 統計數據
        transforms.RandomErasing(p=0.4)
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_eval)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"🐷 已知豬數量: {num_classes}")

    # =====================
    # 初始化 ECB Hybrid Model
    # =====================
    model = ECBHybridModel(num_classes=num_classes).to(device)
    
    # 根據論文公式 ，使用標準 CrossEntropyLoss
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 優化器同時優化兩個分支
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    start_epoch = 0
    best_val_loss = float("inf")

    if pth_path and os.path.exists(pth_path):
        checkpoint = torch.load(pth_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"🔄 從 {pth_path} 繼續訓練，起始 epoch={start_epoch}")

    # =====================
    # 訓練循環 (Supervised + Co-training style)
    # =====================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                # 取得兩個分支的輸出
                logits_vit, logits_cnn = model(imgs)

                # 1. Supervised Training Loss 
                # 同時訓練 ViT 和 CNN 進行正確分類
                loss_vit = criterion_ce(logits_vit, labels)
                loss_cnn = criterion_ce(logits_cnn, labels)
                
                # 2. Consistency Loss (模擬 Co-training/Minimize Discrepancy)
                # 強制兩個模型的輸出分佈相似 (即使在有標籤數據上這也有助於正則化)
                # 使用 KL Divergence 或 L1 Loss 來拉近兩者預測
                probs_vit = F.softmax(logits_vit, dim=1)
                probs_cnn = F.softmax(logits_cnn, dim=1)
                loss_consistency = F.l1_loss(probs_cnn, probs_vit) # 論文中的 d(p1, p2) 

                # 總 Loss: 監督 loss + 一致性約束
                # 如果是純監督任務，consistency 可以設小一點權重
                loss = loss_vit + loss_cnn + 0.1 * loss_consistency

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(train_loss=total_loss / (loop.n + 1), 
                             v_loss=loss_vit.item(), 
                             c_loss=loss_cnn.item())

        avg_train_loss = total_loss / len(train_loader)

        # evaluate (使用 CNN 分支評估)
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion_ce, device, "Val")
        avg_test_loss, test_acc = evaluate(model, test_loader, criterion_ce, device, "TestKnown")

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} (acc={val_acc:.3f}) | TestKnown: {avg_test_loss:.4f} (acc={test_acc:.3f})")

        scheduler.step()

        # Save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch+1,
                "model": model.state_dict(),
                "classes": train_dataset.classes
            }, "pig_ecb_best.pth")
            print(f"💾 儲存最佳模型：pig_ecb_best.pth")

    print("✅ 訓練完成")

if __name__ == "__main__":
    train_model(
        dataset_root=os.path.join(ROOT_DIR, "dataset_split"),
        num_epochs=20,
        batch_size=8, # ViT 較佔記憶體，若 OOM 請調小
        device="cuda"
    )