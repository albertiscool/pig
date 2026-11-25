import torch  # Must be imported first on Windows to avoid WinError 4551 (shm.dll)

import os
import sys
import cv2
import time
import glob
import tempfile
import argparse

EXP_V3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_V3_DIR, "..", ".."))
if EXP_V3_DIR not in sys.path:
    sys.path.insert(0, EXP_V3_DIR)

from custom_occlusion_modules import register_custom_modules, patch_ultralytics_shape_iou
from ultralytics import YOLO

DEFAULT_V3_WEIGHTS = os.path.join(PROJECT_ROOT, "02_models", "yolo_v3_occlusion_best.pt")
DEFAULT_ARCH_YAML = os.path.join(EXP_V3_DIR, "yolo11s_seam_ema.yaml")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "03_datasets", "yolo_face_v3_train_val")
DEFAULT_TEST_ROOT = os.path.join(PROJECT_ROOT, "03_datasets", "yolo_face_benchmarks")
DEFAULT_AI_HARD_DIR = os.path.join(PROJECT_ROOT, "03_datasets", "external_test_datasets", "generated_hard_samples")
DEFAULT_PROJECT_OUT = os.path.join(PROJECT_ROOT, "runs_workstation")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".dav")


def make_temp_dataset_yaml(dataset_dir: str, train_sub: str = "train/images", val_sub: str = "val/images") -> str:
    abs_dir = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_text = (
        f"path: {abs_dir}\n"
        f"train: {train_sub}\n"
        f"val: {val_sub}\n"
        f"nc: 1\n"
        f"names: ['face']\n"
    )
    fd, tmp_path = tempfile.mkstemp(prefix="v3_data_", suffix=".yaml")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return tmp_path


def resolve_device(device_arg: str):
    if device_arg != "auto":
        if device_arg.isdigit():
            return int(device_arg)
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


def run_eval(args):
    device = resolve_device(args.device)
    weights = os.path.abspath(args.weights)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[EVAL] Loading weights: {weights} (device={device})")
    model = YOLO(weights)

    clean_dir = os.path.join(DEFAULT_TEST_ROOT, "test_clean")
    occ_dir = os.path.join(DEFAULT_TEST_ROOT, "test_occluded")
    neg_dir = os.path.join(DEFAULT_TEST_ROOT, "test_hard_neg", "images")
    if not os.path.isdir(neg_dir):
        neg_dir = os.path.join(DEFAULT_TEST_ROOT, "test_hard_neg")

    tmp_yamls = []
    try:
        yaml_clean = make_temp_dataset_yaml(clean_dir, train_sub="images", val_sub="images")
        yaml_occ = make_temp_dataset_yaml(occ_dir, train_sub="images", val_sub="images")
        tmp_yamls.extend([yaml_clean, yaml_occ])

        print("\n[1/4] Evaluating on test_clean (180 images)...")
        res_clean = model.val(
            data=yaml_clean,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            verbose=False,
            plots=False,
        )

        print("[2/4] Evaluating on test_occluded (180 unseen heavy fence images)...")
        res_occ = model.val(
            data=yaml_occ,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            verbose=False,
            plots=False,
        )
    finally:
        for yp in tmp_yamls:
            if os.path.exists(yp):
                os.remove(yp)

    print("[3/4] Evaluating on test_hard_neg (23 external hard-negative scenes)...")
    neg_files = sorted(
        os.path.join(neg_dir, fn)
        for fn in os.listdir(neg_dir)
        if fn.lower().endswith(IMAGE_EXTS)
    )
    fp25_imgs, fp25_boxes = 0, 0
    fp45_imgs, fp45_boxes = 0, 0
    fp_details = {}
    for nf in neg_files:
        r = model.predict(nf, imgsz=args.imgsz, conf=0.25, device=device, verbose=False)[0]
        confs = [float(b.conf) for b in r.boxes]
        if confs:
            fp25_imgs += 1
            fp25_boxes += len(confs)
            fp_details[os.path.basename(nf)] = [round(c, 3) for c in confs]
        c45 = [c for c in confs if c >= 0.45]
        if c45:
            fp45_imgs += 1
            fp45_boxes += len(c45)

    print("[4/4] Evaluating on external_test_datasets/generated_hard_samples...")
    ai_imgs = [
        "01_reid_295-6_vertical_fence.png",
        "02_yolo_fullframe_metal_bars.png",
        "03_reid_pig16_angled_rails.png",
        "04_yolo_hard_negative_butt_redboard.png",
    ]
    ai_summary = {}
    for fn in ai_imgs:
        p = os.path.join(DEFAULT_AI_HARD_DIR, fn)
        if os.path.exists(p):
            r = model.predict(p, imgsz=args.imgsz, conf=0.20, device=device, verbose=False)[0]
            ai_summary[fn] = [(round(float(b.conf), 4), [int(x) for x in b.xyxy[0].tolist()]) for b in r.boxes]

    n_neg = max(1, len(neg_files))
    print("\n" + "=" * 78)
    print(f"Workstation V3 Evaluation Summary | Weights: {os.path.basename(weights)}")
    print("=" * 78)
    print(
        f"test_clean    (n=180) | P={res_clean.box.mp*100:6.2f}% | R={res_clean.box.mr*100:6.2f}% "
        f"| mAP@0.5={res_clean.box.map50*100:6.2f}% | mAP@0.5:0.95={res_clean.box.map*100:6.2f}%"
    )
    print(
        f"test_occluded (n=180) | P={res_occ.box.mp*100:6.2f}% | R={res_occ.box.mr*100:6.2f}% "
        f"| mAP@0.5={res_occ.box.map50*100:6.2f}% | mAP@0.5:0.95={res_occ.box.map*100:6.2f}%"
    )
    print(
        f"test_hard_neg (n={len(neg_files):>2})  | ImgFP@0.25={fp25_imgs/n_neg*100:5.2f}% ({fp25_boxes} boxes) "
        f"| ImgFP@0.45={fp45_imgs/n_neg*100:5.2f}% ({fp45_boxes} boxes)"
    )
    if fp_details:
        print(f"  Hard-Neg FP@0.25 details: {fp_details}")
    print("\nGenerated Hard Samples (@conf>=0.20):")
    for fn, dets in ai_summary.items():
        print(f"  {fn}: {dets}")
    print("=" * 78)


def run_train(args):
    device = resolve_device(args.device)
    patch_ultralytics_shape_iou()

    data_input = os.path.abspath(args.data)
    tmp_yaml = None
    if os.path.isdir(data_input):
        tmp_yaml = make_temp_dataset_yaml(data_input, train_sub="train/images", val_sub="val/images")
        data_yaml = tmp_yaml
    elif os.path.isfile(data_input):
        data_yaml = data_input
    else:
        raise FileNotFoundError(f"Training data path not found: {data_input}")

    try:
        if args.arch_yaml:
            arch_yaml = os.path.abspath(args.arch_yaml)
            if not os.path.exists(arch_yaml):
                raise FileNotFoundError(f"Architecture YAML not found: {arch_yaml}")
            model = YOLO(arch_yaml)
            if args.weights and args.weights.lower() != "none":
                print(f"[TRAIN] Building from {arch_yaml} and transferring weights from {args.weights}")
                model = model.load(args.weights)
            else:
                print(f"[TRAIN] Building from scratch using {arch_yaml}")
        else:
            weights = os.path.abspath(args.weights)
            if not os.path.exists(weights):
                raise FileNotFoundError(f"Starting checkpoint not found: {weights}")
            print(f"[TRAIN] Fine-tuning existing checkpoint: {weights}")
            model = YOLO(weights)

        os.makedirs(args.project, exist_ok=True)
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            cos_lr=True,
            erasing=0.15,
            mosaic=0.7,
            close_mosaic=min(5, max(1, args.epochs // 4)),
            workers=args.workers,
            cache=args.cache,
            amp=True,
            device=device,
            project=os.path.abspath(args.project),
            name=args.name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    finally:
        if tmp_yaml and os.path.exists(tmp_yaml):
            os.remove(tmp_yaml)


def draw_detections(frame, boxes, color=(0, 230, 0)):
    vis = frame.copy()
    thick = max(2, int(round(min(vis.shape[:2]) / 260.0)))
    fscale = max(0.55, min(vis.shape[:2]) / 820.0)
    for conf, xyxy in boxes:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(
            vis,
            f"face {conf:.2f}",
            (x1 + 4, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fscale,
            color,
            thick,
            cv2.LINE_AA,
        )
    return vis


def run_infer(args):
    if not args.source:
        raise ValueError("--source is required when --mode infer")

    device = resolve_device(args.device)
    weights = os.path.abspath(args.weights)
    source = os.path.abspath(args.source)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(weights)
    t0 = time.time()

    if os.path.isfile(source) and source.lower().endswith(VIDEO_EXTS):
        out_path = args.out
        if not out_path:
            out_dir = os.path.join(DEFAULT_PROJECT_OUT, "infer")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(source))[0]
            out_path = os.path.join(out_dir, f"{stem}_v3_infer.mp4")
        else:
            out_path = os.path.abspath(out_path)
            if os.path.isdir(out_path) or not out_path.lower().endswith(".mp4"):
                os.makedirs(out_path, exist_ok=True)
                stem = os.path.splitext(os.path.basename(source))[0]
                out_path = os.path.join(out_path, f"{stem}_v3_infer.mp4")
            else:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        for fourcc_str in ("mp4v", "avc1", "XVID"):
            cand = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
            if cand.isOpened():
                writer = cand
                break
        if writer is None:
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

        frame_cnt, box_cnt = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
                boxes = [(float(b.conf), b.xyxy[0].tolist()) for b in res.boxes]
                box_cnt += len(boxes)
                frame_cnt += 1
                vis = draw_detections(frame, boxes)
                cv2.putText(
                    vis,
                    f"V3 YOLO11s-SEAM-EMA | frame={frame_cnt} | faces={len(boxes)} | conf>={args.conf:.2f}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (120, 255, 140),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(vis)
                if args.max_frames > 0 and frame_cnt >= args.max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        dt = time.time() - t0
        print(
            f"[INFER VIDEO DONE] frames={frame_cnt} | total_faces={box_cnt} "
            f"| avg={box_cnt / max(1, frame_cnt):.3f}/frame | elapsed={dt:.1f}s | saved={out_path}"
        )
        return

    # Image file or directory
    if os.path.isdir(source):
        img_paths = sorted(
            p for p in glob.glob(os.path.join(source, "*")) if p.lower().endswith(IMAGE_EXTS)
        )
    else:
        img_paths = [source]

    if not img_paths:
        raise FileNotFoundError(f"No images found at: {source}")

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(DEFAULT_PROJECT_OUT, "infer")
    os.makedirs(out_dir, exist_ok=True)

    total_boxes = 0
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        boxes = [(float(b.conf), b.xyxy[0].tolist()) for b in res.boxes]
        total_boxes += len(boxes)
        vis = draw_detections(img, boxes)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), vis)

    dt = time.time() - t0
    print(
        f"[INFER IMAGES DONE] images={len(img_paths)} | total_faces={total_boxes} "
        f"| elapsed={dt:.1f}s | saved_dir={out_dir}"
    )


def parse_args():
    default_workers = 0 if os.name == "nt" else 4
    parser = argparse.ArgumentParser(
        description="Cross-platform workstation runner for V3 Occlusion-Robust Pig Face Detector (YOLO11s-SEAM-EMA + Shape-IoU)"
    )
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train", "infer"], help="Execution mode")
    parser.add_argument("--weights", type=str, default=DEFAULT_V3_WEIGHTS, help="Model checkpoint path (.pt) or pretrained weights when --arch-yaml is set")
    parser.add_argument("--arch-yaml", type=str, default=None, help="Optional model architecture YAML (e.g. yolo11s_seam_ema.yaml) to build fresh or transfer from COCO weights")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Dataset directory (containing train/ and val/) or YAML file for --mode train")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.05, help="Final OneCycle/Cosine LR ratio")
    parser.add_argument("--warmup-epochs", type=float, default=2.0, help="Warmup epochs")
    parser.add_argument("--cache", action="store_true", help="Enable RAM image caching during training")
    parser.add_argument("--device", type=str, default="auto", help="CUDA device id (e.g. 0), 'cpu', or 'auto'")
    parser.add_argument("--workers", type=int, default=default_workers, help="DataLoader workers (default 0 on Windows, 4 on Linux)")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT_OUT, help="Training output root directory")
    parser.add_argument("--name", type=str, default="v3_workstation_exp", help="Training run name")
    parser.add_argument("--source", type=str, default=None, help="Input video/image/folder for --mode infer")
    parser.add_argument("--out", type=str, default=None, help="Output file or directory for --mode infer")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold for --mode infer")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for video inference (0 = full video)")
    return parser.parse_args()


def main():
    args = parse_args()
    register_custom_modules()
    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args)


if __name__ == "__main__":
    main()
