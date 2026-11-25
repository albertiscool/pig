# 🐷 智慧豬隻健康監控與體溫特徵辨識系統 (Pig Health & Temperature Vision System)

本專案提供基於雙鏡頭 (RGB 可見光 + Thermal 熱成像) 之即時豬隻偵測、隔欄抗遮擋識別、眼耳體溫提取、呼吸率分析與個體 Re-ID 身分辨識管線。

---

## 🌿 分支架構說明 (Branch Navigation)

為保持版本清晰並避免不同階段混雜，本專案保留 **`main`、`v0`、`v1`、`v2`、`v3`** 主要分支（早期歷史版本皆已封存於 `archive/*` Git Tags）：

| 分支名稱 (Branch) | 版本定位與核心內容 |
| :--- | :--- |
| **`v0`** | **先前研發成果總結基準 (Baseline Summary)**：整合早期熱成像呼吸率演算法、V4 雙光四分割融合系統與開集自動註冊、Jetson Xavier DMA 邊緣部署程式碼、外部 Re-ID 基準測試，以及 V3 `YOLO11s-SEAM-EMA` 隔欄抗遮擋豬臉偵測器（`19.2 MB`）。 |
| **`v1`** | **P0 抗遮擋 Re-ID (CNN 輕量版)**：在 `v0` 基礎上加入時間零洩漏 17 類切分、AI 柵欄遮擋增強、`BNNeck + ArcFace + Triplet + KD` 消融與 **`42.8 MB` 純 CNN 輕量推論權重**。 |
| **`v2`** | **Foundation DINOv2-S 抗遮擋 Re-ID (`42.11 MB`)**：導入 **DINOv2-ViT-S/14 + GeM Token Pooling + Sub-Center (K=3) AdaFace-ArcFace + Relational Anchor-KD**，外部 Shaojiang 未見遮擋豬隻驗證 **ROC-AUC 0.8432**（vs `v1` 0.7571）、**Top-5 Rank-1 58.33%**（vs `v1` 29.17%）、EER 降至 **25.00%**，同時收錄 `MegaDescriptor-T` 與 `Frozen DINOv2+Adapter`（Clean Rank-1 **81.25%**）完整消融。 |
| **`v3`** | **YOLO11m-BiFPN-EMA-SEAM 隔欄豬臉偵測 (`40.69 MB`)**：在 `v2` 基礎上升級 **BiFPN + EMA + SEAM** 與 **Focaler-Shape-WIoUv3** 損失，結合 `20260722` 真實白豬舍 CCTV 影像與嚴格零洩漏 `20261110` 困難負樣本硬化（`V3.5-HN`），在 Unseen Heavy-Fence 達 **mAP@0.5 99.05% / mAP@0.5:0.95 68.27%**，救回 OOD 白豬欄杆難例（`0.7295` vs `V3` 漏檢）並將巡檢人員誤報從 `10` 壓至 `2`。 |

---

## 🚀 切換分支指令 (Switching Branches)

```bash
git checkout v0   # 先前總結基準 (V3 偵測器 + V4 雙光系統)
git checkout v1   # P0 CNN 抗遮擋 Re-ID
git checkout v2   # DINOv2-S + GeM + SubCenter AdaArcFace 抗遮擋 Re-ID (推薦 Re-ID)
git checkout v3   # YOLO11m-BiFPN-EMA-SEAM + HN 抗遮擋多品種偵測 (推薦 Detector)
```

---

## 🗄️ 歷史封存標籤 (Archived Legacy Tags)

先前各階段獨立快照已完整保留為唯讀 Git Tags，隨時可透過 `git checkout <tag>` 檢視：

| Tag 名稱 | 原分支與內容說明 |
| :--- | :--- |
| `archive/legacy-v0` | 初版熱成像對齊與 FFT 呼吸率基準 |
| `archive/legacy-v1` | 初版 ViT 特徵萃取與特徵庫管理 |
| `archive/legacy-v2` | RTSP 雙攝影機同步與 YOLOv12 即時推論 |
| `archive/legacy-v3` | Grounding DINO 自動標註與 YOLO 微調 |
| `archive/legacy-v4` | Zero-Copy Shared Memory IPC 與 ROS 2 邊緣架構 |
| `archive/legacy-xavier` | Jetson Xavier DMA 與雙光視差投影初版 |
| `archive/legacy-v5-workstation` | 另一台工作站之 V3 遮擋資料與實驗快照 |

---

## 🛠️ 系統共用資源說明
* 大型訓練資料集、現場雙光影片（`*.mp4` / `*.dav`）與超過 100 MB 之原始訓練 Checkpoint 不納入 Git 版控，僅收錄輕量化推論權重（`models/` 內均 < 50 MB）。
