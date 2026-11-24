import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = r"C:\Users\User\Desktop\final_pig_v0"
V4_DIR = os.path.join(PROJECT_ROOT, "v4")
DET_PATH = os.path.join(V4_DIR, "parameters", "nose_eyes_face_0817.pt")
REID_PATH = os.path.join(V4_DIR, "parameters", "pig_ecb_best.pth")
BRAIN_DIR = r"C:\Users\User\.gemini\antigravity\brain\c44ac0e4-dd1c-4cd9-9334-ef9209e06714"
OUT_DIR = os.path.join(PROJECT_ROOT, "external_test_datasets", "generated_hard_samples")
CLEAN_GALLERY_DIR = os.path.join(PROJECT_ROOT, "external_test_datasets", "shaojiang_pig_faces", "clean_gallery")

SOURCES_AND_OUTS = [
    (
        "01_reid_295-6_vertical_fence.png",
        os.path.join(BRAIN_DIR, "pig_295_6_0001_fence_bars_v2_1795255781774.jpg"),
        os.path.join(PROJECT_ROOT, "pig_id_package", "test_samples", "295-6", "295-6_0001.jpg"),
    ),
    (
        "02_yolo_fullframe_metal_bars.png",
        os.path.join(BRAIN_DIR, "pig_fence_occluded_cctv_1795255760223.jpg"),
        os.path.join(PROJECT_ROOT, "pig_id_package", "test_samples", "sample_full_pig_photo.jpg"),
    ),
    (
        "03_reid_pig16_angled_rails.png",
        os.path.join(BRAIN_DIR, "pig_16_train_1005_pen_rails_1795255768514.jpg"),
        os.path.join(CLEAN_GALLERY_DIR, "pig_16", "train_1005.jpg"),
    ),
    (
        "04_yolo_hard_negative_butt_redboard.png",
        os.path.join(BRAIN_DIR, "cctv_two_pigs_rear_view_clean_1795255853092.jpg"),
        None,
    ),
]


def draw_yolo_boxes(img_bgr, res, names):
    vis = img_bgr.copy()
    colors = {"face": (0, 220, 0), "nose": (0, 140, 255), "eye": (255, 200, 0)}
    h, w = vis.shape[:2]
    thick = max(2, int(round(min(h, w) / 220.0)))
    font_s = max(0.5, min(h, w) / 700.0)
    dets = []
    for b in res.boxes:
        cls_name = names[int(b.cls)]
        conf = float(b.conf)
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        dets.append((cls_name, round(conf, 3), [x1, y1, x2, y2]))
        c = colors.get(cls_name, (255, 0, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, thick)
        lbl = f"{cls_name} {conf:.2f}"
        cv2.putText(vis, lbl, (x1 + 3, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_s, c, thick, cv2.LINE_AA)
    return vis, dets


def make_panel(img_bgr, title, subtitle, target_w=460, target_h=300):
    bar_h = 54
    ih, iw = img_bgr.shape[:2]
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h + bar_h, target_w, 3), 24, dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    canvas[target_h:, :] = (32, 32, 32)
    cv2.putText(canvas, title, (10, target_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, target_h + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 230, 180), 1, cv2.LINE_AA)
    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    saved_map = {}
    for out_name, src_gen, src_orig in SOURCES_AND_OUTS:
        dst = os.path.join(OUT_DIR, out_name)
        Image.open(src_gen).save(dst, format="PNG")
        saved_map[out_name] = (dst, src_orig)

    sys.path.insert(0, V4_DIR)
    from pig_identity import ECBHybridModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = (device.type == "cuda")

    det = YOLO(DET_PATH)
    det.to(device)

    ckpt = torch.load(REID_PATH, map_location=device, weights_only=False)
    classes = ckpt.get("classes", [])
    model = ECBHybridModel(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if use_half:
        model.half()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def infer_reid(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        t = tfm(rgb).unsqueeze(0).to(device)
        if use_half:
            t = t.half()
        with torch.no_grad():
            logits, feat = model.extract_features(t)
            probs = F.softmax(logits.float(), dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            p295_idx = classes.index("295-6") if "295-6" in classes else -1
            p295_prob = float(probs[p295_idx].item()) if p295_idx >= 0 else 0.0
        return classes[top_idx], float(probs[top_idx].item()), p295_prob, feat[0].cpu()

    report = [
        "================================================================================",
        "Before-vs-After Probe on Generated Hard Samples (YOLO + ECBHybridModel FP16)",
        "================================================================================",
    ]
    rows = []

    # --- Sample 1: 295-6 vertical fence ---
    gen1_path, orig1_path = saved_map["01_reid_295-6_vertical_fence.png"]
    orig1, gen1 = cv2.imread(orig1_path), cv2.imread(gen1_path)
    y_o1 = det.predict(orig1, imgsz=640, verbose=False)[0]
    y_g1 = det.predict(gen1, imgsz=640, verbose=False)[0]
    vis_o1, det_o1 = draw_yolo_boxes(orig1, y_o1, det.names)
    vis_g1, det_g1 = draw_yolo_boxes(gen1, y_g1, det.names)
    id_o1, c_o1, p295_o1, f_o1 = infer_reid(orig1)
    id_g1, c_g1, p295_g1, f_g1 = infer_reid(gen1)
    sim1 = float(torch.dot(f_o1, f_g1).item())
    report += [
        "[Sample 1] 295-6 Closed-Set + Fence Occlusion (01_reid_295-6_vertical_fence.png)",
        f"  Original : ReID={id_o1} (conf={c_o1*100:.2f}%, 295-6_prob={p295_o1*100:.2f}%) | YOLO={det_o1}",
        f"  Occluded : ReID={id_g1} (conf={c_g1*100:.2f}%, 295-6_prob={p295_g1*100:.2f}%) | YOLO={det_g1}",
        f"  Orig vs Occluded Cosine Similarity : {sim1:.4f}",
        "",
    ]
    rows.append([
        make_panel(vis_o1, "Sample 1 Orig (295-6_0001)", f"ReID: {id_o1} ({c_o1*100:.1f}%) | Det: {len(det_o1)} boxes"),
        make_panel(vis_g1, "Sample 1 Gen (+Vertical Fence)", f"ReID: {id_g1} ({c_g1*100:.1f}%) | CosSim={sim1:.3f}"),
    ])

    # --- Sample 2: Full-Frame CCTV Fence ---
    gen2_path, orig2_path = saved_map["02_yolo_fullframe_metal_bars.png"]
    orig2, gen2 = cv2.imread(orig2_path), cv2.imread(gen2_path)
    y_o2 = det.predict(orig2, imgsz=640, verbose=False)[0]
    y_g2 = det.predict(gen2, imgsz=640, verbose=False)[0]
    vis_o2, det_o2 = draw_yolo_boxes(orig2, y_o2, det.names)
    vis_g2, det_g2 = draw_yolo_boxes(gen2, y_g2, det.names)
    # Also crop detected/center face to compare face detection boxes
    report += [
        "[Sample 2] Full-Frame CCTV YOLO Detection + Fence (02_yolo_fullframe_metal_bars.png)",
        f"  Original ({orig2.shape[1]}x{orig2.shape[0]}): YOLO={det_o2}",
        f"  Occluded ({gen2.shape[1]}x{gen2.shape[0]}): YOLO={det_g2}",
        "",
    ]
    rows.append([
        make_panel(vis_o2, "Sample 2 Orig (CCTV Full Frame)", f"YOLO dets: {[(d[0], d[1]) for d in det_o2]}"),
        make_panel(vis_g2, "Sample 2 Gen (+Rusted Fence Bars)", f"YOLO dets: {[(d[0], d[1]) for d in det_g2]}"),
    ])

    # --- Sample 3: pig_16 Angled Rails (Open-Set) ---
    gen3_path, orig3_path = saved_map["03_reid_pig16_angled_rails.png"]
    orig3, gen3 = cv2.imread(orig3_path), cv2.imread(gen3_path)
    y_o3 = det.predict(orig3, imgsz=640, verbose=False)[0]
    y_g3 = det.predict(gen3, imgsz=640, verbose=False)[0]
    vis_o3, det_o3 = draw_yolo_boxes(orig3, y_o3, det.names)
    vis_g3, det_g3 = draw_yolo_boxes(gen3, y_g3, det.names)
    id_o3, c_o3, _, f_o3 = infer_reid(orig3)
    id_g3, c_g3, _, f_g3 = infer_reid(gen3)
    sim3_pair = float(torch.dot(f_o3, f_g3).item())

    # Build clean gallery centroids for the 11 pigs (excluding train_1005 from pig_16 centroid so zero leakage!)
    centroids = {}
    for pdir in sorted(glob.glob(os.path.join(CLEAN_GALLERY_DIR, "pig_*"))):
        pname = os.path.basename(pdir)
        imgs = [p for p in sorted(glob.glob(os.path.join(pdir, "*.jpg"))) if os.path.abspath(p) != os.path.abspath(orig3_path)]
        if not imgs:
            continue
        scored = []
        for p in imgs:
            im = cv2.imread(p)
            sh = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            scored.append((sh, p))
        scored.sort(reverse=True)
        feats = [infer_reid(cv2.imread(p))[3] for _, p in scored[:5]]
        centroids[pname] = F.normalize(torch.mean(torch.stack(feats), dim=0, keepdim=True), p=2, dim=1)[0]

    rank_o3 = sorted([(k, float(torch.dot(f_o3, v).item())) for k, v in centroids.items()], key=lambda x: -x[1])
    rank_g3 = sorted([(k, float(torch.dot(f_g3, v).item())) for k, v in centroids.items()], key=lambda x: -x[1])
    report += [
        "[Sample 3] External Unseen pig_16 Open-Set + Angled Rails (03_reid_pig16_angled_rails.png)",
        f"  Orig vs Occluded Cosine Similarity : {sim3_pair:.4f}",
        f"  Orig Top-3 Open-Set Match          : {[(k, round(v, 4)) for k, v in rank_o3[:3]]}",
        f"  Occluded Top-3 Open-Set Match      : {[(k, round(v, 4)) for k, v in rank_g3[:3]]}",
        f"  YOLO Original : {det_o3}",
        f"  YOLO Occluded : {det_g3}",
        "",
    ]
    rows.append([
        make_panel(vis_o3, "Sample 3 Orig (pig_16 train_1005)", f"Top1={rank_o3[0][0]} ({rank_o3[0][1]:.3f})"),
        make_panel(vis_g3, "Sample 3 Gen (+Angled Steel Rails)", f"Top1={rank_g3[0][0]} ({rank_g3[0][1]:.3f}) | Sim={sim3_pair:.3f}"),
    ])

    # --- Sample 4: YOLO Hard Negative ---
    gen4_path, _ = saved_map["04_yolo_hard_negative_butt_redboard.png"]
    gen4 = cv2.imread(gen4_path)
    y_g4 = det.predict(gen4, imgsz=640, verbose=False)[0]
    y_g4_low = det.predict(gen4, imgsz=640, conf=0.15, verbose=False)[0]
    vis_g4, det_g4 = draw_yolo_boxes(gen4, y_g4, det.names)
    det_g4_low = [(det.names[int(b.cls)], round(float(b.conf), 3)) for b in y_g4_low.boxes]
    report += [
        "[Sample 4] YOLO Hard Negative Farm Scene (04_yolo_hard_negative_butt_redboard.png)",
        f"  YOLO detections (default conf>=0.25) : {det_g4}",
        f"  YOLO detections (low conf>=0.15)     : {det_g4_low}",
        "================================================================================",
    ]
    rows.append([
        make_panel(vis_g4, "Sample 4 Gen (YOLO Hard Negative)", "Target: Empty .txt label (2 pig rears + red divider)"),
        make_panel(vis_g4, "Sample 4 YOLO Probe (conf>=0.25)", f"False Positives={len(det_g4)} (conf>=0.15: {len(det_g4_low)})"),
    ])

    # Stack 4x2 grid cleanly
    grid_rows = [np.hstack([r[0], np.full((r[0].shape[0], 10, 3), 12, dtype=np.uint8), r[1]]) for r in rows]
    sep = np.full((10, grid_rows[0].shape[1], 3), 12, dtype=np.uint8)
    grid = grid_rows[0]
    for gr in grid_rows[1:]:
        grid = np.vstack([grid, sep, gr])

    grid_out = os.path.join(OUT_DIR, "preview_comparison_grid.jpg")
    cv2.imwrite(grid_out, grid)

    report_txt = "\n".join(report)
    with open(os.path.join(OUT_DIR, "probe_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")
    print(report_txt)
    print(f"\nSaved comparison grid: {grid_out}")


if __name__ == "__main__":
    main()
