# 🐷 智慧豬隻健康監控與體溫特徵辨識系統 (Pig Health & Temperature Vision System)

本專案提供基於雙鏡頭 (RGB 可見光 + Thermal 熱成像) 之即時豬隻偵測、特徵定位 (眼、耳、鼻、臉)、體溫提取、呼吸率分析與自動標註微調管線。

---

## 🌿 分支架構說明 (Branch Navigation)

為避免各版本混雜於單一目錄造成混亂，本專案採用 **獨立版本分支 (Orphan Branches)** 進行組織管理：

| 分支名稱 (Branch) | 版本重點與核心技術 |
| :--- | :--- |
| **`v0`** | **基礎版本**：熱成像動態對齊矩陣、Swin-Transformer 熱成像特徵分析、FFT 呼吸頻率基準。 |
| **`v1`** | **特徵比對與資料庫**：ViT (Vision Transformer) 豬隻臉部特徵提取、未知豬隻比對與特徵庫更新。 |
| **`v2`** | **即時雙鏡頭串流**：RTSP 雙攝影機硬體同步擷取 (`SyncedCapture`)、YOLOv12 即時推論、現場體溫追蹤。 |
| **`v3`** | **最新版本**：Grounding DINO 零樣本 (Zero-Shot) 自動標註與 NMS 去重、多影片均勻抽樣資料集生成 (`dataset_dino_new`)、YOLO 微調管線 (`run_finetune_yolo.py`)。 |

---

## 🚀 切換分支指令 (Switching Branches)

若要切換至指定版本查看並執行對應程式碼：

```bash
# 切換至 v0
git checkout v0

# 切換至 v1
git checkout v1

# 切換至 v2
git checkout v2

# 切換至 v3 (最新自動標註與微調版本)
git checkout v3
```

---

## 🛠️ 系統共用資源說明
* **模型權重檔 (`parameters/`)** 與 **現場錄影檔 (`*.mp4`)** 檔案容量龐大，不建議直接納入 Git 版控。
* 建議在本機根目錄下保留 `parameters/` 資料夾，即可供所有版本分支共同讀取。
