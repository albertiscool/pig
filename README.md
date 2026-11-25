# 🐷 智慧豬隻健康監控與體溫特徵辨識系統 (Pig Health & Temperature Vision System)

本專案提供基於雙鏡頭 (RGB 可見光 + Thermal 熱成像) 之即時豬隻偵測、隔欄抗遮擋識別、眼耳體溫提取、呼吸率分析與個體 Re-ID 身分辨識管線。

---

## 🌿 分支架構說明 (Branch Navigation)

為保持版本清晰並避免不同階段混雜，本專案僅保留 **`main`、`v0`、`v1`** 三個主要分支（更早期的歷史版本皆已封存於 `archive/*` Git Tags）：

| 分支名稱 (Branch) | 版本定位與核心內容 |
| :--- | :--- |
| **`v0`** | **先前研發成果總結基準 (Baseline Summary)**：整合早期熱成像呼吸率演算法、V4 雙光四分割融合系統與開集自動註冊、Jetson Xavier DMA 邊緣部署程式碼、外部 Re-ID 基準測試，以及 V3 `YOLO11s-SEAM-EMA` 隔欄抗遮擋豬臉偵測器（`19.2 MB`）。 |
| **`v1`** | **目前最新版本 (Latest — P0 Occlusion Re-ID)**：在 `v0` 基礎上加入 **P0 豬臉抗遮擋 Re-ID**（時間零洩漏 17 類切分、AI 柵欄遮擋增強、`BNNeck + ArcFace + Triplet + KD` 3 方消融與視覺化報告）、**`42.8 MB` 純 CNN 輕量推論權重**與校準後開集拒識門檻。 |

---

## 🚀 切換分支指令 (Switching Branches)

```bash
# 查看先前各代系統總結基準 (v0)
git checkout v0

# 切換至目前最新版本 (v1：含 P0 抗遮擋 Re-ID 與輕量推論權重)
git checkout v1
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
