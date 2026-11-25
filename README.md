# 🐷 豬隻智慧監控與隔欄辨識系統 — `v0`（先前各代研發成果總結基準）

本分支將先前各階段（`legacy v0`～`v4`、`Jetson Xavier` 邊緣部署與 `V3 YOLO11s-SEAM-EMA` 抗遮擋偵測）整理收錄為統一的 **Baseline 基準版本 (`v0`)**，作為後續 `v1`（P0 Re-ID 抗遮擋強化）之對照基準。

> 💡 若需查閱更早期的單獨歷史快照，本專案已完整保留於 Git Tags：`archive/legacy-v0`、`archive/legacy-v1`、`archive/legacy-v2`、`archive/legacy-v3`、`archive/legacy-v4`、`archive/legacy-xavier`、`archive/legacy-v5-workstation`。

---

## 📈 一、系統演進總結（Legacy → V4 → Xavier → V3 Occlusion）

1. **熱像對齊與呼吸率基準（`legacy v0` → `v4/respiration_rate_v0/`）**：建立熱成像鼻孔 ROI 訊號抽取（M1/M2/M3）與滑窗 FFT 呼吸率估計。
2. **豬臉五官偵測與 Re-ID 雛形（`legacy v1`–`v3` → `v4/pig_identity_reid/`）**：以 `ECBHybridModel`（ViT-B/16 + ResNet-18）搭配 `0817` 五官 YOLO 權重，並建立 Grounding DINO 自動標註微調管線。
3. **雙光四分割融合與自動註冊引擎（`v4/`）**：
   - `run_quad_view_unified.py` / `run_quad_view_live.py`：採用 `4x2` 視差補償矩陣取代早期 Swin P2P，整合眼窩（Top-10 淚腺熱點吸附）／耳根測溫、鼻孔帶通濾波呼吸率與四分割看板。
   - `pig_identity.py`：加入未知豬開集自動註冊（`Pig_Reg_001`～`006`）、Laplacian 模糊過濾與多角度重心融合（原始設定 `conf_threshold=0.65`, `sim_threshold=0.82`）。
4. **Jetson Xavier 邊緣硬體優化（`xavier_edge/`）**：整合 GStreamer NVDEC DMA 取流、Pinned Memory 非同步搬移與二階段局部器官推論（`TwoStageDMAPigDetector`）。
5. **V3 隔欄抗遮擋豬臉偵測器（`yolo_v3_occlusion/` + `models/yolo_v3_occlusion_best.pt`）**：
   - 以 `YOLO11s` 為骨幹，於 P3/P4/P5 偵測頭前加入 `EMA_SEAM`（跨欄協調注意力＋通道指數增強）並於訓練期替換為 `Shape-IoU` 損失。
   - 重度未見鐵欄測試 `mAP@0.5` 由 `17.66%` 提升至 **`99.43%`**（`mAP@0.5:0.95` 由 `5.27%` → **`68.35%`**），現場 8 部實拍影片 Top-20 獨立框達 `0/20` 背景誤檢。
6. **外部 Re-ID 與 AI 障礙圖壓力測試發現（`external_eval_scripts/`）**：
   - 在公開 `shaojiang/pig-images` 乾淨資料集上，原有 `pig_ecb_best.pth`（`ECBHybridModel`）未知豬拒識 `@0.65` 達 `TRR=100%`、Top-5 重心 `Rank-1=72.92%`（`ROC-AUC=0.9024`），但不同豬相似度均值高達 `0.8639`（導致 `sim_threshold=0.82` 下 FPR 高達 `87.5%`）。
   - 在 AI 生成垂直柵欄遮擋下，`295-6` Re-ID 信心度由 `78.68%` 降至 `63.63%`，促使後續 **`v1`（P0 Re-ID 抗遮擋訓練）** 誕生。

---

## 📁 二、目錄架構

```text
v0/
├── README.md                          # 本總結文件
├── requirements.txt
├── models/
│   └── yolo_v3_occlusion_best.pt      # V3 隔欄抗遮擋豬臉偵測權重 (19.2 MB)
├── v4/                                # V4 雙光四分割系統、Re-ID 與呼吸率模組
├── yolo_v3_occlusion/                 # V3 YOLO11s-SEAM-EMA + Shape-IoU 程式碼與報表
├── xavier_edge/                       # Jetson Xavier DMA 與 TensorRT 部署程式及基準圖
└── external_eval_scripts/             # 外部 Shaojiang 資料集與 AI 困難樣本壓力測試腳本
```
