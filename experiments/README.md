# 🐷 豬隻臉部抗遮擋偵測（YOLO）與外部個體辨識（Re-ID）實驗總覽

本文件彙整針對「**隔著豬舍柵欄／障礙物精準框選豬臉（YOLO）**」與「**外部場域豬臉個體辨識（Re-ID）**」所完成的完整實驗、架構實作、AI 困難樣本合成管線與量化／現場影片驗收結果。

> **專案邊界說明**：所有實驗均獨立存放於 `external_test_datasets/` 與 `experiments/`，**未改動**任何線上推論主程式或既有的正式模型權重（`v4/parameters/` 與 `pig_id_package/models/`）。

---

## 📌 一、整體研發動機與演進脈絡

1. **外部資料集實測現有 Re-ID（[shaojiang_pig_faces](file:///C:/Users/User/Desktop/final_pig_v0/external_test_datasets/shaojiang_pig_faces)）**
   - 以公開多豬個體資料集 Hugging Face `shaojiang/pig-images` 評估既有 `ECBHybridModel`（ResNet18 推論分支，[pig_ecb_best.pth](file:///C:/Users/User/Desktop/final_pig_v0/v4/parameters/pig_ecb_best.pth)）。
   - 發現前段偵測器單靠 `face` 類別在陌生豬舍易誤抓紅色欄板、豬臀部與水窪；以五官關鍵點過濾後（11 隻未見新豬、48 張查詢圖）：
     - 未知豬隻拒識（`@0.65`）：**FAR = 0.00%（TRR = 100.00%）**
     - Top-5 清晰度重心 Re-ID：**Rank-1 = 72.92%（多張建檔子集 `78.79%`）**、**ROC-AUC = `0.9024`**（單張對單張 `0.9329`）。
2. **AI 障礙物壓力測試（[generated_hard_samples](file:///C:/Users/User/Desktop/final_pig_v0/external_test_datasets/generated_hard_samples)）**
   - 以生成式編輯在豬臉前加入擬真柵欄與鋼管後，現有模型出現明顯掉分：
     - `295-6` 的 Re-ID 信心度從 **`78.68%` 降至 `63.63%`**（跌破線上 `0.65` 門檻）；
     - 監視器畫面 `nose_eyes_face_0817.pt` 的 `face` 信心度從 **`0.897` 暴跌至 `0.505`**；
     - 在零豬臉的「豬臀＋紅欄板」負樣本圖上誤抓出 `nose 0.447`。
3. **選擇「訓練期強化」而非「推論期 GAN 修補」（對比 *PigFRIS* 2025）**
   - *PigFRIS*（*Animals*, 2025）推論階段需額外執行柵欄分割與 AOT-GAN 影像修補，不適合現場 25 fps 即時多模型巡檢。
   - 本方案將 AI 生成成本全部放在**離線訓練階段**，透過論文級注意力模組與寫實遮擋增強讓單一 YOLO 直接穿透欄杆偵測，達到**推論期零額外架構延遲**。
4. **發現原始 `face.pt` 容量瓶頸並完成 V1 → V2 → V3 三代迭代**
   - 經檢查，`face.pt` 與早期 V1/V2 實為僅 **`2.57M` 參數的 `YOLOv12n`**，對深色衣物、人腿與黑色網椅容易誤檢（V2 在 21 部實拍影片驗收時雖救回隔欄豬臉，但也誤抓人腿 `0.85`、黑網椅 `0.71`、空鐵欄 `0.67`）。
   - **V3** 正式升級為 **`YOLO11s`（`9.88M` 參數，COCO 預訓練轉移 `378/568` 層權重）＋ 論文遮擋模組（`EMA` + `SEAM` + `Shape-IoU`）＋ 實拍無豬負樣本＋ 跨品種白豬／6 種 AI 障礙圖層**，一次解決背景誤檢與隔欄偵測問題。

---

## 🧠 二、V3 論文核心架構實作（`YOLO11s-SEAM-EMA + Shape-IoU`）

實作程式碼位於 [custom_occlusion_modules.py](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/custom_occlusion_modules.py)，網路設定位於 [yolo11s_seam_ema.yaml](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/yolo11s_seam_ema.yaml)：

| 參考論文與年份 | 採用模組 / 機制 | 插入位置與作用機制 |
| :--- | :--- | :--- |
| **YOLO-FaceV2** (*Pattern Recognition*) | **`SEAM`**（Separated & Enhancement Attention Module） | 置於 P3/P4/P5 進入 `Detect` 頭前；以深度可分離殘差卷積 + `exp(sigmoid)` 通道指數增強，放大未被柵欄遮住的五官局部紋理。 |
| **DESA-YOLO / YOLO-ESO** (*MDPI*, 2026) | **`EMA`**（Efficient Multi-Scale Attention） | 以水平＋垂直座標池化與跨空間注意力，跨越橫向／垂直鐵欄縫隙關聯上下左右露出來的豬臉區域；搭配可學習殘差閘門 `gate`（初值 `0.25`）穩定保留 COCO 特徵。 |
| **DGS-YOLO** (*MDPI*, Jan 2026) | **`Shape-IoU`** 邊框回歸損失 | 於 Runtime 替換 Ultralytics `BboxLoss.forward`，依據 Ground Truth 框長寬比例加權位置與形狀懲罰，改善部分遮擋導致的框偏移與鬆散問題。 |

---

## 🛠️ 三、各版本資料配置與設計演進（嚴格零資料洩漏）

| 版本 | 模型骨幹／策略 | 訓練集構成 | 測試隔離原則 |
| :--- | :--- | :--- | :--- |
| **Baseline** | `YOLOv12n` (`face.pt`, `2.57M`) | 原始系統內建權重 | 未經本次微調，作為統一基準 |
| **V1** | `YOLOv12n` 微調<br>(`epochs=15`, `lr0=0.002`) | `1,520` 張（1,000 黑豬＋400 程式合成欄杆＋120 外部紅欄板負樣本） | `test_clean` (180)、`test_occluded` (180 重度未見格柵)、`test_hard_neg` (23) 完全隔離 |
| **V2** | `YOLOv12n` 溫和微調<br>(`epochs=12`, `lr0=0.0005`) | `5,512` 張（4,092 全量黑豬＋1,000 張 4 種 AI 去背柵欄＠`47.1%` 遮擋＋200 白豬合成＋220 負樣本） | 同上標準測試集＋未見 AI 生成圖＋21 部實拍影片抽樣稽核（1,890 幀） |
| **V3 (最新)** | **`YOLO11s-SEAM-EMA` + `Shape-IoU`**<br>(`9.88M`, `epochs=20`, `lr0=0.0015`) | **`5,942` 張**：<br>• `4,092` 全量真實黑豬 (`v3/data/face_train/train`)<br>• `1,000` **6 種 AI 障礙圖層**（生鏽鐵欄、交叉鋼管、鐵絲網格、紅洞洞板、生鏽鐵鍊與飲水餵飼管、斜撐鐵門，控制於 **`29.9%`** 合理遮擋）<br>• `250` 跨品種粉白豬（含 HF `UniqueData/pigs-detection-dataset` `27` 原圖＋`27` 遮擋圖＋`shaojiang` 白豬）<br>• `600` 困難負樣本（含 **`338` 張**來自 `20261110` 實驗室／走道＋`103456` 零豬影格、**2 張**新 AI 豬臀背影與工人褲管雨鞋走廊圖、紅欄板與空豬舍合成場景） | **嚴格保留 8 部 `20261111` 現場豬舍影片**完全不進訓練，專作實拍盲測（720 幀） |

---

## 📊 四、完整測試集與現場實拍影片結果對比

### 1. 標準隔離測試集數據（來自 [v3_full_report.txt](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/v3_full_report.txt)）

| 評估基準 | Baseline (`face.pt`) | V1 (`YOLOv12n`) | V2 (`YOLOv12n+AI`) | **V3 (`YOLO11s-SEAM-EMA`)** |
| :--- | :---: | :---: | :---: | :---: |
| **乾淨測試集 — Precision / Recall** | `99.97% / 100.00%` | `96.70% / 100.00%` | `99.97% / 100.00%` | **`99.97% / 100.00%`** |
| **乾淨測試集 — mAP@0.5** | `99.50%` | `99.18%` | `99.50%` | **`99.50%`** |
| **乾淨測試集 — mAP@0.5:0.95** | `85.72%` | `73.38%` | `85.04%` | **`84.84%`** |
| **重度未見鐵欄測試 — Precision / Recall** | `25.55% / 37.22%` | `78.35% / 80.44%` | `100.00% / 99.25%` | **`97.25% / 98.36%`** |
| **重度未見鐵欄測試 — mAP@0.5** | `17.66%` | `84.58%` | `99.50%` | **`99.43%`** *(+81.76%)* |
| **重度未見鐵欄測試 — mAP@0.5:0.95** | `5.27%` | `50.72%` | `61.55%` | **`68.35%`** *(+63.08% vs Base, +6.80% vs V2)* |
| **外部困難負樣本誤檢率 (`@0.45`)** | `8.70%` (3 框) | `0.00%` (0 框) | `4.35%` (1 框 `0.831`) | **`4.35%`** (1 框 `0.641`) |
| **未見 AI 黑豬 `295-6` 鐵欄圖 (`conf`)** | `0.5737` | `0.2370` | `0.7987` | **`0.8562`** |
| **未見 AI 白豬 CCTV 鐵欄圖 (`conf`)** | `0.3565` | `[]` | `0.5600` | `[]` |

### 2. 現場 8 部未見過 `20261111` 實拍影片盲測（720 幀，`@0.45`）

| 比較項目 | V4 Live (`nose_eyes_face_0817.pt`) | Baseline (`face.pt`) | V2 (`YOLOv12n+AI`) | **V3 (`YOLO11s-SEAM-EMA`)** |
| :--- | :---: | :---: | :---: | :---: |
| **720 幀總偵測框數** | `125` | `147` | `153` *(含 40 可疑)* | **`134`** *(22 未獲他模佐證，`83.6%` 一致)* |
| **巡檢員褲管／人影片段 (`104223`, 90 幀)** | `9` | `7` | `13` *(含多處人腿誤框)* | **`0` 誤檢** |
| **白豬隔欄困難片段 (`113922` / `113613`)** | `7` / `5` | `2` / `2` | `1` / `0` | **`13` / `8`** *(`@0.35` 達 `21` / `11` 框)* |
| **Top-20 獨立「可疑框」人工肉眼檢查** | — | — | 大多為人腿／黑網椅／螢幕／空欄杆 | **`0 / 20` 背景誤檢** *(約 15 張為真實隔欄豬臉，約 5 張為真豬耳側)* |

---

## 📁 五、檔案與重點成果索引

### A. 推薦直接使用的最佳模型（V3）
- **訓練完成權重**：[best.pt](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/runs/yolo11s_seam_ema_v3/weights/best.pt)
- **自訂模組定義（載入 V3 前必匯入）**：[custom_occlusion_modules.py](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/custom_occlusion_modules.py)
- **訓練與完整評測腳本**：[train_and_eval_v3.py](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/train_and_eval_v3.py)

### B. 視覺驗證圖與實測影片
- **V4 vs V2 vs V3 現場實拍對比圖**：[v3_vs_v2_vs_v4_field_grid.jpg](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/v3_vs_v2_vs_v4_field_grid.jpg)
- **V3 現場獨立偵測 20 格實檢圖**：[v3_field_suspicious_grid.jpg](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/v3_field_suspicious_grid.jpg)
- **隔欄黑豬並排對比實測影片（左 V4 / 右 V3，570 幀）**：[rec_20261111_120218_v4_vs_v3_check.mp4](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/rec_20261111_120218_v4_vs_v3_check.mp4)
- **V3 訓練樣本預覽與基準長條圖**：[v3_dataset_preview_grid.jpg](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/v3_dataset_preview_grid.jpg) ／ [v3_metrics_chart.png](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/v3_metrics_chart.png)
- **6 種 AI 障礙綠幕去背圖層資料夾**：[ai_overlays_rgba](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo11s_seam_ema_v3/ai_overlays_rgba)

### C. 先前外部 Re-ID 與 V1/V2 實驗紀錄
- **外部豬臉 Re-ID 評估報告與拼圖**：[clean_evaluation_report.txt](file:///C:/Users/User/Desktop/final_pig_v0/external_test_datasets/shaojiang_pig_faces/clean_evaluation_report.txt) ／ [clean_crops_grid.jpg](file:///C:/Users/User/Desktop/final_pig_v0/external_test_datasets/shaojiang_pig_faces/clean_crops_grid.jpg)
- **V1 實驗報告**：[experiment_report.txt](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo_fence_occlusion_exp/experiment_report.txt)
- **V2 實驗報告與 21 部影片稽核**：[v2_experiment_report.txt](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo_fence_occlusion_v2/v2_experiment_report.txt) ／ [field_audit_report.txt](file:///C:/Users/User/Desktop/final_pig_v0/experiments/yolo_fence_occlusion_v2/field_audit_report.txt)

---

## 💻 六、如何在你的 Python 系統載入 V3 模型（注意事項）

1. **Windows 匯入順序限制**：本機環境需將 `import torch` 放在所有腳本第 1 行（在 `ultralytics`、`cv2` 之前），否則可能觸發 `WinError 4551`（`shm.dll` 應用程式控制原則阻擋）。
2. **必須先註冊自訂層 (`EMA_SEAM`)**：

```python
import torch  # 必須放第一行
import sys
sys.path.insert(0, r"C:\Users\User\Desktop\final_pig_v0\experiments\yolo11s_seam_ema_v3")

from custom_occlusion_modules import register_custom_modules
from ultralytics import YOLO

register_custom_modules()  # 註冊 EMA / SEAM / EMA_SEAM 模組
model = YOLO(r"C:\Users\User\Desktop\final_pig_v0\experiments\yolo11s_seam_ema_v3\runs\yolo11s_seam_ema_v3\weights\best.pt")

results = model.predict(frame, imgsz=640, conf=0.45, verbose=False)
```

---

## 🔭 七、後續可繼續推進的方向

1. **併入更多公開多品種豬臉資料集**：若後續從 Roboflow 下載 CC BY 4.0 公開豬臉包（如 `trainforge/pig-face-detection`、`davixpixie/pig-face-detection`）並補入更多白豬背影負樣本，可進一步改善大寬幅白豬場景保守問題，並壓低殘餘豬臀／豬耳誤框。
2. **推論端雙模型互補過濾**：在正式即時系統（`v4/run_quad_view_live.py`）中，可將 V3 的隔欄 `face` 框結合既有 `nose_eyes_face_0817.pt` 的 `nose`／`eye` 關鍵點做雙重驗證，確保只有**露臉豬隻**才會進入體溫計算與 Re-ID 自動註冊。
3. **後續展開 Re-ID 抗遮擋訓練**：待 YOLO 偵測框定版後，即可進入下一階段對 `ECBHybridModel`（`pig_ecb_best.pth`）進行身分保真式遮擋微調與門檻校準。
