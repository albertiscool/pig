# 🐷 Pig Health, Thermal & Occlusion-Robust Vision System (`v5-occlusion-workstation`)

> **AI Agent Handover & High-Performance Workstation Guide**  
> This branch packages the complete **V4 Dual-Camera Thermal/Respiration Monitoring System**, the **V3 Occlusion-Robust Pig Face Detector (`YOLO11s-SEAM-EMA + Shape-IoU`)**, **5 paired field RGB + Thermal pig videos (`20261111`)**, **5,942 training images**, **held-out benchmark suites**, and **all required model checkpoints** so an AI agent or developer on a higher-performance workstation can reproduce results, scale training, and integrate end-to-end improvements cleanly.

---

## 1. Project Overview & Current Status

The repository contains three interconnected computer-vision subsystems designed for pig-pen monitoring behind fence bars and feeders:

1. **V3 Occlusion-Robust Pig Face Detector (`experiments/yolo11s_seam_ema_v3/`)**
   - Architecture: **`YOLO11s` backbone/neck (`9.88M` params) + `EMA_SEAM` attention blocks (`P3/P4/P5`) + runtime `Shape-IoU` bbox regression loss**, defined in [`yolo11s_seam_ema.yaml`](experiments/yolo11s_seam_ema_v3/yolo11s_seam_ema.yaml) and [`custom_occlusion_modules.py`](experiments/yolo11s_seam_ema_v3/custom_occlusion_modules.py).
   - Trained offline (`5,942` images) with 6 AI-generated fence/pipe/chain chroma-keyed occluders (`29.9%` mean bbox occlusion), multi-breed white pigs (`external_hf_uniquedata/`), and real empty-barn/human-leg hard negatives — achieving **zero extra inference latency vs. a GAN-inpainting pipeline**.
   - **Important**: Always call `register_custom_modules()` before loading `best.pt`, otherwise Ultralytics cannot deserialize the `EMA_SEAM` layers.

2. **V4 Dual-Camera RGB + Thermal Physiological Pipeline (`v4/`)**
   - Synchronized dual-stream (RGB visible + Thermal IR) eye-socket / ear-root temperature extraction, `4x2` homography thermal projection ([`thermal_projection.py`](v4/thermal_projection.py)), `0.2–2.0 Hz` bandpass respiratory waveform BPM estimation, and 4-panel quad-view rendering.
   - Live entry point: [`v4/run_quad_view_live.py`](v4/run_quad_view_live.py); offline dual-video reprocessing entry points: [`v4/run_quad_view_unified.py`](v4/run_quad_view_unified.py) and [`v4/reprocess_recorded_videos.py`](v4/reprocess_recorded_videos.py).

3. **V4 Open-Set & Closed-Set Pig Identity Re-ID (`v4/pig_identity.py`, `v4/pig_identity_reid/`)**
   - **`ECBHybridModel`** (`v4/parameters/pig_ecb_best.pth`, `370.15 MB`): dual-branch `ViT-B/16` (training co-regularization teacher) + `ResNet-18` (512-D normalized embedding / fast inference branch).
   - Paired with `registered_pigs/` gallery (`Pig_Reg_001` – `Pig_Reg_006`), `conf_threshold=0.65` unknown-pig rejection, and open-set evaluation benchmarks under `external_test_datasets/shaojiang_pig_faces/`.

---

## 2. Quick Start & Environment Setup (New Workstation)

### A. Clone with Git LFS (Mandatory)

Five files exceed GitHub's 100 MB limit and are stored in **Git LFS** (`.gitattributes`, `~881.7 MB` total). Make sure `git-lfs` is installed **before** running scripts, otherwise those 5 files will remain `~130`-byte pointer text files and fail to load:

```bash
git lfs install
git clone -b v5-occlusion-workstation --single-branch https://github.com/albertiscool/pig.git
cd pig
git lfs pull
```

Verify that all 5 LFS binaries downloaded completely (every size should be `106 MB`–`370 MB`, NOT `130 B`):

```bash
git lfs ls-files -s
# Expected outputs:
# * recorded_videos/rec_20261111_113613/rec_20261111_113613_rgb.mp4 (136 MB)
# * recorded_videos/rec_20261111_113613/rec_20261111_113613_thm.mp4 (106 MB)
# * recorded_videos/rec_20261111_113922/rec_20261111_113922_rgb.mp4 (135 MB)
# * recorded_videos/rec_20261111_113922/rec_20261111_113922_thm.mp4 (134 MB)
# * v4/parameters/pig_ecb_best.pth (370 MB)
```

### B. Python Dependencies

Python 3.10–3.12 with CUDA-enabled PyTorch is recommended:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows Import Order Warning**: On strict Windows setups, `import torch` **must appear on line 1** before `ultralytics`, `cv2`, or `numpy` to prevent `WinError 4551` (`torch\lib\shm.dll` policy block).  
> **Custom V3 Layer Registration Warning**: Always call `register_custom_modules()` (and `patch_ultralytics_shape_iou()` when training) from [`custom_occlusion_modules.py`](experiments/yolo11s_seam_ema_v3/custom_occlusion_modules.py) prior to `YOLO(...best.pt)`.

---

## 3. Complete Directory Map & Key Scripts

| Path | Contents & Role |
| :--- | :--- |
| [`requirements.txt`](requirements.txt) | Core Python package dependencies. |
| [`v4/run_quad_view_unified.py`](v4/run_quad_view_unified.py) | Main offline RGB + Thermal fusion, eye/ear temperature & respiration BPM pipeline (`run_pipeline`). |
| [`v4/reprocess_recorded_videos.py`](v4/reprocess_recorded_videos.py) | CLI batch reprocessor reading `recorded_videos/<rec_id>/*_{rgb,thm}.mp4` from frame 0 and resetting BPM/temperature state. |
| [`v4/run_quad_view_live.py`](v4/run_quad_view_live.py) | Real-time RTSP dual-stream cockpit + UDP Xavier pan/tilt coordinate feedback. |
| [`v4/pig_identity.py`](v4/pig_identity.py) | `ECBHybridModel` definition & `PigIdentifier` open-set gallery matcher / auto-registration engine. |
| [`v4/thermal_projection.py`](v4/thermal_projection.py) | Fast `4x2` homography & affine RGB-to-Thermal coordinate mapper using `rgb_to_thermal_mapping.npz`. |
| [`v4/pig_identity_reid/`](v4/pig_identity_reid/) | Standalone Re-ID gallery extraction, DB update, evaluation (`03_evaluate_pig_id_folder.py`), and `ECBHybridModel` training (`05_train_reid_model.py`). |
| [`v4/respiration_rate_v0/`](v4/respiration_rate_v0/) | Baseline FFT, sliding window, and thermal nostril ROI signal scripts. |
| [`v4/parameters/`](v4/parameters/) | Active YOLO, Pose, Thermal, and Re-ID weights plus `registered_pigs/` profiles and shots. |
| [`experiments/README.md`](experiments/README.md) | Detailed Chinese R&D writeup covering V1 → V2 → V3 iterations, design rationale vs. *PigFRIS* (2025), and field audit notes. |
| [`experiments/yolo11s_seam_ema_v3/`](experiments/yolo11s_seam_ema_v3/) | **Main V3 occlusion-robust pig face package**: `custom_occlusion_modules.py`, `yolo11s_seam_ema.yaml`, `run_v3_workstation.py`, `train_and_eval_v3.py`, `data/` (`5,942` train + `180` val), `ai_overlays_rgba/`, `ai_new_assets/`, `external_hf_uniquedata/`, and `runs/yolo11s_seam_ema_v3/weights/best.pt`. |
| [`experiments/yolo_fence_occlusion_exp/data/`](experiments/yolo_fence_occlusion_exp/data/) | Held-out standard benchmark splits: `test_clean/` (180), `test_occluded/` (180 heavy synthetic bars), `test_hard_neg/` (23 empty barn / butt / red-board images). |
| [`external_test_datasets/generated_hard_samples/`](external_test_datasets/generated_hard_samples/) | Generative fence stress-test samples (`01_reid_295-6_vertical_fence.png` .. `04_yolo_hard_negative_butt_redboard.png`) + `probe_report.txt`. |
| [`external_test_datasets/shaojiang_pig_faces/`](external_test_datasets/shaojiang_pig_faces/) | Clean landmark-verified unseen external pig Re-ID set (`clean_gallery/`, `clean_query/`) + `clean_evaluation_report.txt`. |
| [`recorded_videos/`](recorded_videos/) | 5 paired farm RGB + Thermal recordings from `2026-11-11` (`113613`, `113922`, `115925`, `120047`, `120218`) + [`all_recalculated_summary.csv`](recorded_videos/all_recalculated_summary.csv). |

---

## 4. Model Weights & Git LFS Inventory

All required inference/training weights live either in `v4/parameters/` or `experiments/yolo11s_seam_ema_v3/runs/`.

| File Path | Approx. Size | Storage | Role |
| :--- | :---: | :---: | :--- |
| `experiments/yolo11s_seam_ema_v3/runs/yolo11s_seam_ema_v3/weights/best.pt` | `19.19 MB` | Git | **Best V3 Occlusion-Robust Pig Face Detector** (`YOLO11s-SEAM-EMA`). Requires `register_custom_modules()`. |
| `v4/parameters/pig_ecb_best.pth` | `370.15 MB` | **Git LFS** | `ECBHybridModel` (`ViT-B/16 + ResNet18`, 17 closed classes + 512-D embeddings). |
| `v4/parameters/nose_eyes_face_0817.pt` | `21.48 MB` | Git | V4 primary RGB multi-class detector (`face`, `nose`, `eye`). |
| `v4/parameters/pose_best.pt` | `40.35 MB` | Git | RGB anatomical pose keypoint model (ear-root keypoints KP 5 / KP 6). |
| `v4/parameters/best_v12_eye_nose_root.pt` | `5.38 MB` | Git | YOLOv12 ear-root / eye / nose bbox detector (`yolo_bbox` mode). |
| `v4/parameters/face.pt` | `5.26 MB` | Git | Legacy YOLOv12n face baseline (`2.57M` params). |
| `v4/parameters/thm/best_thermal_nose.pt` | `49.69 MB` | Git | Thermal IR nostril detector for breathing signal extraction. |
| `v4/parameters/thm/rgb_to_thermal_mapping.npz` | `< 1 MB` | Git | Calibrated `4x2` / affine spatial alignment matrix (`RGB -> Thermal`). |
| `v4/parameters/registered_pigs/` & `registered_pigs_gallery.pt` | `~2 MB` | Git | Registered pig face gallery (`Pig_Reg_001` – `Pig_Reg_006`). |
| `recorded_videos/rec_20261111_113613/*.mp4` (`_rgb`, `_thm`) | `136 + 106 MB` | **Git LFS** | Held-out farm video (`Pig_Reg_006`, difficult fence white pig). |
| `recorded_videos/rec_20261111_113922/*.mp4` (`_rgb`, `_thm`) | `135 + 134 MB` | **Git LFS** | Held-out farm video (`Pig_Reg_006`, difficult fence white pig, 45 BPM). |
| `recorded_videos/rec_20261111_{115925,120047,120218}/*.mp4` | `13–50 MB` ea | Git | 3 additional paired farm RGB + Thermal clips (`21–36` BPM; `120218` is heavily occluded behind bars). |

*(Note: Obsolete 130 MB `swin_p2p_*.pth` legacy alignment weights, ONNX mirrors, `runs/.../last.pt`, and raw `best(2).zip` were intentionally excluded to keep the branch self-contained and within GitHub's 1 GB free LFS budget.)*

---

## 5. Ready-to-Run Workstation Commands

### A. Evaluate V3 (`YOLO11s-SEAM-EMA`) on Clean, Occluded, Hard-Neg & AI Stress Benchmarks

```bash
python experiments/yolo11s_seam_ema_v3/run_v3_workstation.py --mode eval
```

### B. Run V3 Inference on Any Field Video or Image Folder

```bash
python experiments/yolo11s_seam_ema_v3/run_v3_workstation.py \
  --mode infer \
  --source recorded_videos/rec_20261111_120218/rec_20261111_120218_rgb.mp4 \
  --conf 0.45 \
  --out runs_workstation/infer/120218_v3.mp4
```

### C. Continue Fine-Tuning or Scale Up on the Workstation GPU

```bash
# Continue fine-tuning existing V3 weights on packaged data (5,942 train / 180 val):
python experiments/yolo11s_seam_ema_v3/run_v3_workstation.py \
  --mode train --epochs 50 --batch 32 --imgsz 640 --lr0 0.0008 --name v3_ft50

# Scale to a larger YAML (or re-train from COCO yolo11s.pt with custom EMA_SEAM + Shape-IoU):
python experiments/yolo11s_seam_ema_v3/run_v3_workstation.py \
  --mode train --arch-yaml experiments/yolo11s_seam_ema_v3/yolo11s_seam_ema.yaml \
  --weights yolo11s.pt --epochs 60 --batch 32 --imgsz 960 --name v3_scale_960
```

### D. Batch Reprocess the 5 Paired Field Videos (`RGB + Thermal -> BPM + Quad-View`)

Outputs are written under `recorded_videos/<rec_id>/recalculated/` (ignored by `.gitignore` so the repo stays clean):

```bash
python v4/reprocess_recorded_videos.py --date 20261111 --min_sec 10
```

*(Note on legacy scripts: [`experiments/yolo11s_seam_ema_v3/train_and_eval_v3.py`](experiments/yolo11s_seam_ema_v3/train_and_eval_v3.py) documents the full V1→V3 dataset synthesis pipeline, while `run_v3_workstation.py` is the self-contained script for the packaged dataset/benchmarks. When running [`external_test_datasets/eval_clean_pig_faces.py`](external_test_datasets/eval_clean_pig_faces.py) on another PC, update `PROJECT_ROOT` & `DET_FACE_PT_PATH` to point to `v4/parameters/face.pt`.)*

---

## 6. Verified Benchmark Baselines (Reference Before Optimizing)

From [`v3_full_report.txt`](experiments/yolo11s_seam_ema_v3/v3_full_report.txt), [`clean_evaluation_report.txt`](external_test_datasets/shaojiang_pig_faces/clean_evaluation_report.txt), [`probe_report.txt`](external_test_datasets/generated_hard_samples/probe_report.txt), and [`all_recalculated_summary.csv`](recorded_videos/all_recalculated_summary.csv):

### YOLO Pig Face Detection (`Baseline face.pt` vs. `V2` vs. `V3 YOLO11s-SEAM-EMA`)

| Benchmark (`imgsz=640`) | Baseline `face.pt` | V2 (`YOLOv12n+AI`) | **V3 (`YOLO11s-SEAM-EMA`)** |
| :--- | :---: | :---: | :---: |
| **`test_clean` (180 imgs)** Precision / Recall | `99.97%` / `100.00%` | `99.97%` / `100.00%` | **`99.97%` / `100.00%`** |
| **`test_clean`** mAP@0.5 / mAP@0.5:0.95 | `99.50%` / `85.72%` | `99.50%` / `85.04%` | **`99.50%` / `84.84%`** |
| **`test_occluded` (180 heavy fence imgs)** P / R | `25.55%` / `37.22%` | `100.00%` / `99.25%` | **`97.25%` / `98.36%`** |
| **`test_occluded`** mAP@0.5 / mAP@0.5:0.95 | `17.66%` / `5.27%` | `99.50%` / `61.55%` | **`99.43%` / `68.35%`** |
| **`test_hard_neg` (23 imgs)** Img FP `@0.25` / `@0.45` | `13.04%` (5b) / `8.70%` (3b) | `17.39%` (4b) / `4.35%` (1b) | **`13.04%` (3b) / `4.35%` (1b)** |
| **AI Fence Stress (`01_reid_295-6`)** face conf | `0.5737` | `0.7987` | **`0.8562`** |
| **8 Held-Out `20261111` Farm Videos (720f `@0.45`)** | `147` boxes (`7` in `104223` human clip) | `153` boxes (`40` susp, `13` in `104223` human legs) | **`134` boxes (`0` human-leg FPs; Top-20 independent audit: `0/20` background FPs)** |

### Included 5 Field Video Reference Baselines (`all_recalculated_summary.csv`)

| Recording ID | Current V4 Pig ID (Conf) | Duration | Recalculated BPM (Peaks) | Note |
| :--- | :--- | :---: | :---: | :--- |
| `rec_20261111_113613` | `Pig_Reg_006` (`0.836`) | `164.2 s` | `28` BPM (`36`) | White pig behind pen bars (`0.666–0.731` V3 fence faces) |
| `rec_20261111_113922` | `Pig_Reg_006` (`0.896`) | `327.7 s` | `45` BPM (`102`) | White pig behind pen bars (`V3` raises hits vs `V4`: `13` vs `7` in 90f sample) |
| `rec_20261111_115925` | `Pig_Reg_006` (`0.880`) | `50.5 s` | `21` BPM (`15`) | Clean / side pen recording |
| `rec_20261111_120047` | `Pig_Reg_006` (`0.928`) | `60.6 s` | `21` BPM (`17`) | Clean / partial rail recording |
| `rec_20261111_120218` | `Unknown` (`0.349`) | `22.8 s` | `36` BPM (`9`) | **Priority occluded stress clip**: V4 Live misses face (`0.00`/fr in full scan), V3 recovers behind-bar faces (`0.498–0.751`) |

### Current `ECBHybridModel` Re-ID Baselines (`pig_ecb_best.pth`, Without Occlusion Training Yet)

- External unseen `shaojiang/pig-images` (11 pigs, 44 gallery / 48 query): Unknown rejection `@0.65` gives **`FAR = 0.00%` (`TRR = 100.00%`)**, Top-5 centroid **`Rank-1 = 72.92%`** (multi-shot **`78.79%`**), **`ROC-AUC = 0.9024`**.
- Under AI fence occlusion on `295-6` (`01_reid_295-6_vertical_fence.png`), Re-ID confidence drops from **`78.68%` → `63.63%`** (just below the `0.65` gate).

---

## 7. AI Agent Next-Step Mission Brief

If you are the AI agent taking over on the high-performance GPU workstation, here are the recommended next milestones in priority order:

1. **Scale Up & Broaden the Occlusion-Robust Face Detector (V3 → V4 Detector)**
   - First verify baseline reproduction with `python experiments/yolo11s_seam_ema_v3/run_v3_workstation.py --mode eval`.
   - On a workstation GPU (e.g., 24+ GB VRAM), test **scaling `YOLO11m-SEAM-EMA` or `YOLO11l-SEAM-EMA`** (adjusting `scales` + neck channel widths `[128, 256, 512]` in a new YAML to match backbone output channels, or increasing `imgsz=960`/`epochs=60+`) and incorporate additional public multi-breed pig-face datasets + white-pig rear/ear hard negatives to fix conservative misses on wide-angle white pig scenes (`02_yolo_fullframe_metal_bars.png`) while keeping `test_hard_neg` near `0%`.
2. **Integrate V3 Fence-Robust Face Detector into the Live / Offline V4 Dual-Camera Pipeline**
   - Update [`v4/run_quad_view_unified.py`](v4/run_quad_view_unified.py) and [`v4/run_quad_view_live.py`](v4/run_quad_view_live.py) to run V3 (`best.pt` + `register_custom_modules()`) alongside [`v4/parameters/nose_eyes_face_0817.pt`](v4/parameters/nose_eyes_face_0817.pt): use V3 to recall occluded faces behind pen bars (e.g., `rec_20261111_120218`), and cross-validate `nose` / `eye` keypoints before feeding crops to Re-ID and temperature tracking.
3. **Train Occlusion-Robust Pig Re-ID (`ECBHybridModel`)**
   - Once YOLO localization behind fences is locked in, use [`v4/pig_identity_reid/05_train_reid_model.py`](v4/pig_identity_reid/05_train_reid_model.py) together with the 6 chroma-keyed occluder PNGs in [`experiments/yolo11s_seam_ema_v3/ai_overlays_rgba/`](experiments/yolo11s_seam_ema_v3/ai_overlays_rgba/) to fine-tune `pig_ecb_best.pth` with identity-preserving bar occlusions and recalibrate `conf_threshold` / `sim_threshold` in [`v4/pig_identity.py`](v4/pig_identity.py).
