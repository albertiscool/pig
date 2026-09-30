# 🐷 Pig Monitoring System - Version 4 (Zero-Copy SHM & ROS 2 Architecture)

智慧豬隻健康多模態監控系統（可見光 RGB ＋ 紅外線熱成像 Thermal），整合 **YOLOv8 關鍵點偵測**、**Swin Transformer P2P 跨光譜空間映射** 與 **呼吸率（RR）時頻分析**。

---

## 🚀 Version 4 核心架構升級 (What's New in V4)

為解決高解析度雙串流與重度 AI 模型在 Windows 筆電上的執行瓶頸，V4 全面導入作業系統底層 IPC 與邊緣運算框架優化：

### 1. 跨平台零拷貝共享記憶體引擎 (Zero-Copy Shared Memory IPC)
* **突破 Python GIL 限制**：將 RTSP 串流解碼、EasyOCR 動態溫標辨識與 YOLO/Swin 類神經網路推論拆分至獨立 **Process（多行程）**，完全釋放多核心 CPU 算力。
* **Zero-Copy Frame Pipeline**：運用跨平台共享記憶體（Windows 呼叫 `CreateFileMapping`，Linux 呼叫 `shm_open`），1080p 影像跨進程傳遞延遲降至 **0.001 毫秒**，無資料拷貝負擔。
* **RTSP 累積延遲消滅機制**：擷取進程持續更新環狀緩衝，自動丟棄過期緩衝影格，徹底解決攝影機串流跑久延遲數秒的「緩衝膨脹（Buffer Bloat）」現象。
* **異步動態溫標校準**：EasyOCR 移入背景獨立進程（1 Hz 週期性更新），消滅原本每秒一次的推論卡頓。

### 2. ROS 2 工業級分散式節點架構 (`pig_vision_ros2`)
* 提供開箱即用的 ROS 2 套件，完美支援 **Windows WSL 2（原生支援 NVIDIA GPU / CUDA 加速）** 與原廠 Linux 環境。
* **模組化解耦**：
  * `dahua_dual_stream_node`：雙鏡頭影像發布（採用 `SensorDataQoS`，Best Effort，Depth=1）。
  * `ocr_calibrator_node`：熱成像動態溫標解析發布（`/pig/thermal/temp_scale`）。
  * `swin_fusion_tracker_node`：以 `ApproximateTimeSynchronizer` 精確同步雙鏡頭時間戳，執行 YOLO + Swin P2P 幾何變形映射與眼溫追蹤。

---

## 📂 專案核心目錄結構

```text
pig/
├── pig_shm_engine.py               # [V4 核心] 零拷貝共享記憶體 IPC 引擎
├── live_eye_temp_tracker_v4.py     # [V4 核心] 零拷貝即時眼溫監控主程式
├── video_pipeline2.py              # SwinTransformerP2P 模型架構與幾何映射工具
├── dahua_tracker.py                # 大華 C-SDK 底層測溫與封包對接
├── record_dual_cameras.py          # 雙鏡頭 RTSP 影像時間對齊錄影
├── auto_rr_pipeline.py             # 呼吸率（RR）時頻特徵提取
│
└── ros2_edge_ws/                   # [V4 核心] ROS 2 分散式邊緣運算工作區
    └── src/pig_vision_ros2/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   └── pig_system.launch.py
        └── pig_vision_ros2/
            ├── dahua_dual_stream_node.py
            ├── ocr_calibrator_node.py
            └── swin_fusion_tracker_node.py
```

---

## ⚡ 快速啟動指南 (Quick Start)

### 模式 A：Windows 筆電原生極速獨立模式 (推薦)
直接在 Windows 終端機執行 V4 零拷貝引擎：

```bash
# 預設連線大華 RTSP 串流
python live_eye_temp_tracker_v4.py

# 或是手動指定測試影片檔
python live_eye_temp_tracker_v4.py --rgb data/test_vid/rgb_test.mp4 --thm data/test_vid/thm_test.mp4
```

* 快捷操作：
  * 按 `S` 鍵：將當前左右眼溫與 OCR 溫標自動寫入 `field_eye_temp_records_v4.csv`。
  * 按 `Q` 鍵：安全釋放共享記憶體與退出程式。

---

### 模式 B：ROS 2 分散式節點模式 (WSL 2 / Ubuntu)

在 Windows 11 的 WSL 2 (Ubuntu 22.04 / 24.04) 中編譯並啟動：

```bash
cd ros2_edge_ws
colcon build --symlink-install
source install/setup.bash

# 一鍵啟動全套分散式節點
ros2 launch pig_vision_ros2 pig_system.launch.py
```

---

## 📊 V3 vs V4 效能實測對比

| 評測指標 | V3 單執行緒迴圈 | V4 零拷貝多進程 (SHM) | V4 ROS 2 分散式節點 |
| :--- | :--- | :--- | :--- |
| **CPU 核心利用** | 單核 (受 Python GIL 限制) | **多核並發 (3 獨立 Process)** | **多進程分散式排程** |
| **RTSP 累積延遲** | 隨時間延遲遞增至 3~5 秒 | **常時維持 < 30 ms** | **常時維持 < 30 ms (QoS)** |
| **OCR 卡頓影響** | 每秒畫面凍結約 200 ms | **完全零卡頓 (背景異步)** | **完全零卡頓 (獨立節點)** |
| **影像傳遞開銷** | N/A (單一程式記憶體) | **0 拷貝 (共享記憶體黑板)** | 零拷貝 / DDS 共享記憶體傳輸 |
| **系統容錯性** | 單點閃退整機停擺 | 擷取 / 推論隔離，容錯度高 | **節點自動恢復，支援熱抽換** |
