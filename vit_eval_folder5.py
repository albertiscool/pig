import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 設定後端，避免在無螢幕環境報錯
matplotlib.use('Agg')

# --------------------------------------------------------
# 1. 必須定義 ECBHybridModel (需與訓練/註冊時一致)
# --------------------------------------------------------
class ECBHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ViT Branch (Testing 時只用來載入權重避免報錯，不參與運算)
        self.vit_model = models.vit_b_16(weights=None)
        self.vit_head_dim = self.vit_model.heads.head.in_features
        self.vit_model.heads = nn.Identity()
        self.classifier_vit = nn.Linear(self.vit_head_dim, num_classes)

        # CNN Branch (Testing 主力)
        self.cnn_model = models.resnet18(weights=None)
        self.cnn_fc_dim = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()
        self.classifier_cnn = nn.Linear(self.cnn_fc_dim, num_classes)

    def extract_features(self, x):
        """
        測試階段專用：只使用 CNN Encoder 提取特徵
        """
        feat_cnn = self.cnn_model(x)  # (B, 512)
        return F.normalize(feat_cnn, dim=1) # 正規化

# --------------------------------------------------------
# 工具函式
# --------------------------------------------------------
def get_pig_id(name: str):
    """回傳豬的主要ID（取 '_' 前面的部分）"""
    return name.split('_')[0]

def evaluate_unknown_folder(
    test_root,
    model_path,
    feature_path,
    label_path,
    device="cuda",
    visualize_num=6
):
    print(f"🔄 載入模型權重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 取得類別數量
    if "classes" in checkpoint:
        num_classes = len(checkpoint["classes"])
    else:
        # 嘗試從權重形狀推斷
        num_classes = checkpoint['model']['classifier_cnn.weight'].shape[0]
    
    # 初始化模型
    model = ECBHybridModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 載入 feature 庫
    print(f"📂 載入特徵庫: {feature_path}")
    features = np.load(feature_path)
    labels = np.load(label_path)
    features_torch = torch.tensor(features, dtype=torch.float32, device=device)
    
    print(f"   -> 特徵維度: {features.shape} (應為 N x 512)")

    # 圖片前處理 (配合新模型使用 ImageNet 標準)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []  # 儲存結果
    correct = 0
    total = 0

    print(f"\n🚀 開始測試資料夾：{test_root}\n")

    # 遍歷測試資料夾
    # 假設資料結構為 test_root/pig_name/image.jpg
    # 或是 test_root/image.jpg (視您的資料夾結構而定，這邊沿用您原本的雙層結構)
    
    # 如果 test_root 下面直接是圖片，請修改這裡的迴圈邏輯
    # 這裡預設結構為: test_root/正確答案豬名/圖片.jpg
    dirs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    
    if len(dirs) == 0:
        print("⚠️ 警告：未發現子資料夾，假設根目錄下即為圖片，嘗試以檔名解析標籤...")
        # 簡易適配：如果沒有子資料夾，就直接讀檔
        # 這裡保留原本邏輯，只處理有子資料夾的情況
    
    for pig_label in sorted(dirs):
        pig_folder = os.path.join(test_root, pig_label)
        
        for img_name in tqdm(os.listdir(pig_folder), desc=f"🐷 {pig_label}"):
            img_path = os.path.join(pig_folder, img_name)
            
            # 簡單過濾非圖片檔案
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except:
                print(f"⚠️ 無法開啟圖片: {img_path}")
                continue

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                # 🔥 關鍵修改：使用 extract_features 取得 CNN 特徵
                embedding = model.extract_features(img_tensor)
                
                # 計算相似度
                sim = F.cosine_similarity(embedding, features_torch)
                best_idx = torch.argmax(sim).item()
                pred_label = labels[best_idx]
                similarity = sim[best_idx].item()

            total += 1
            
            # 比對 ID (忽略 _v2 等後綴)
            true_id = get_pig_id(str(pig_label))
            pred_id = get_pig_id(str(pred_label))
            
            is_correct = (true_id == pred_id)
            
            if is_correct:
                correct += 1
                result_mark = "✅"
            else:
                result_mark = "❌"

            # Log (可選，避免刷屏可註解掉 print)
            # print(f"{result_mark} | 真: {pig_label} | 測: {pred_label} ({similarity:.4f})")

            results.append({
                "path": img_path,
                "true": pig_label,
                "pred": pred_label,
                "sim": similarity,
                "correct": is_correct
            })

    acc = correct / total if total > 0 else 0
    print("\n" + "="*30)
    print(f"📊 測試完成！總共 {total} 張圖片")
    print(f"🎯 準確率: {acc*100:.2f}%")
    print("="*30)

    # ========================
    # 顯示並儲存結果圖
    # ========================
    if total > 0:
        print(f"\n🖼️ 正在生成結果示意圖 (隨機 {visualize_num} 張)...")
        visualize_cnt = min(len(results), visualize_num)
        sample_results = random.sample(results, visualize_cnt)
        
        plt.figure(figsize=(15, 10))

        # 嘗試載入字體
        try:
            # Windows 常見字體路徑，若在 Linux 可改為 /usr/share/fonts/...
            font = ImageFont.truetype("arial.ttf", size=24)
        except:
            font = ImageFont.load_default()

        for i, res in enumerate(sample_results, 1):
            img = Image.open(res["path"]).convert("RGB")
            draw = ImageDraw.Draw(img)

            text_lines = [
                f"True: {res['true']}",
                f"Pred: {res['pred']}",
                f"Sim: {res['sim']:.3f}"
            ]

            # 繪製文字背景與內容
            x_pos, y_pos = 10, 10
            for line in text_lines:
                # 取得文字大小 (兼容舊版 Pillow 使用 getsize，新版用 getbbox)
                try:
                    left, top, right, bottom = font.getbbox(line)
                    w, h = right - left, bottom - top
                except AttributeError:
                    w, h = font.getsize(line)
                
                # 畫半透明黑底
                draw.rectangle([x_pos-2, y_pos-2, x_pos+w+2, y_pos+h+2], fill=(0, 0, 0, 160))
                
                # 畫文字 (綠色正確，紅色錯誤)
                color = (0, 255, 0) if res["correct"] else (255, 100, 100)
                draw.text((x_pos, y_pos), line, fill=color, font=font)
                y_pos += h + 5

            plt.subplot(2, 3, i)
            plt.imshow(img)
            plt.axis("off")
            plt.title("Correct" if res["correct"] else "Wrong", 
                      color="green" if res["correct"] else "red")

        plt.tight_layout()
        save_path = "visual_results_ecb.jpg"
        plt.savefig(save_path)
        print(f"📸 結果圖已儲存至: {save_path}")

if __name__ == "__main__":
    # 請確保這裡的路徑是指向 "ECB" 新模型與 "新生成的特徵庫"
    evaluate_unknown_folder(
        test_root=os.path.join(ROOT_DIR, "dataset_split", "unknown", "test"),
        model_path=os.path.join(ROOT_DIR, "pig_ecb_best.pth"),
        feature_path=os.path.join(ROOT_DIR, "pig_features_ecb.npy"),
        label_path=os.path.join(ROOT_DIR, "pig_labels_ecb.npy"),
    )