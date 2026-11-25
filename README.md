# 🐷 豬隻智慧監控與隔欄抗遮擋辨識系統 — `v1`（最新：V3 抗遮擋偵測 + P0 抗遮擋輕量 Re-ID）

本分支為目前最新整合版本（於 `v0` 基礎上加入 **P0 豬臉 Re-ID 抗遮擋優化**、**嚴格時間零洩漏資料切分**、**`42.8 MB` 純 CNN 輕量化權重**與**校準後開集拒識門檻**）。

---

## 🆕 `v1` 相較 `v0` 的核心升級

1. **重建時間序列零洩漏 17 類 Re-ID 資料切分 (`reid_p0_occlusion/`)**
   - 修正 `reid_ecb_dataset_split2` 類別錯位與洩漏問題，採 `70% Train | 5% Buffer | 10% Val | 5% Buffer | 10% Test`（`4,221 / 604 / 609` 張，重疊率 `0%`）與獨立 `386` 張本地未見未知豬測試集。
2. **AI 柵欄遮擋增強 + 度量化學習重訓 (`ECB + AI-Fence + BNNeck + ArcFace + Triplet + KD`)**
   - 訓練期以 4 種 AI 柵欄去背圖層隨機遮擋（15%–40%），驗證與測試期嚴格使用 2 種**未見過的圖層**（`wire_grate`、`diag_gate`）。
   - **閉集未見鐵欄 Top-1 準確率**：`83.91%` → **`93.76%`**（真類平均信心度 `49.53%` → **`92.95%`**）。
   - **外部未見豬 (`shaojiang`) 類間分離度 $\Delta$**：`+0.0714` → **`+0.2123`**（擴大 3 倍）；`FPR@0.82` 由 `88.54%` 驟降至 **`1.04%`**；鐵欄遮擋下 `ROC-AUC` 由 `0.6932` → **`0.7571`**。
   - **困難探針測試**：`295-6` 垂直鐵欄預測機率 `63.64%` → **`99.61%`**；`pig_16` 斜角柵欄辨識 Margin `+0.0255` → **`+0.1166`**。
3. **輕量化純 CNN 推論權重匯出 (`models/pig_reid_p0_cnn_only.pth`)**
   - 抽離推論期用不到的 ViT-B/16 分支，權重由 `370.17 MB` 縮減為 **`42.75 MB`（-88.4%）**，與完整模型 Embedding 最大誤差為 **`0.00e+00`**，RTX 4090 FP16 單張推論僅 `2.233 ms`。
4. **開集門檻校準**
   - `v4/pig_identity.py` 預設 `sim_threshold` 已由 `0.82` 校準為 `0.90`（對應原有 `pig_ecb_best.pth`）。
   - 若部署 `models/pig_reid_p0_cnn_only.pth`（`ArcMarginProduct(s=16.0)`），請改用最大餘弦門檻 `max(logit / 16.0) >= 0.45`（本地未知拒識 `98.96%`／外場 `100.00%`）或 Softmax `>= 0.97`，開集新豬比對設 `sim_threshold = 0.71`。

---

## 📁 目錄架構與視覺化報告

```text
v1/
├── README.md                          # 本說明文件
├── requirements.txt
├── models/
│   ├── yolo_v3_occlusion_best.pt      # YOLO11s-SEAM-EMA 隔欄豬臉偵測權重 (19.2 MB)
│   └── pig_reid_p0_cnn_only.pth       # P0 純 CNN 輕量化 Re-ID 推論權重 (42.8 MB)
├── reid_p0_occlusion/                 # P0 AI 柵欄增強、BNNeck+ArcFace+Triplet 與訓練腳本
├── reid_p0_results/                   # P0 完整實驗數據報表與 4 張視覺化圖表
│   └── README.md                      # 👉 GitHub 圖文視覺化報告 (含 3 方消融對比)
├── yolo_v3_occlusion/                 # V3 隔欄抗遮擋偵測模組與報表
├── v4/                                # V4 雙光四分割融合系統與開集自動註冊
├── xavier_edge/                       # Jetson Xavier DMA 硬體加速部署程式碼
└── external_eval_scripts/             # 外部公開資料集與 AI 障礙圖壓力測試腳本
```

👉 **完整 3 方消融數據與視覺圖表請見：[reid_p0_results/README.md](./reid_p0_results/README.md)**
