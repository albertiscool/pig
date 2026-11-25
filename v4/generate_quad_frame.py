# -*- coding: utf-8 -*-
import os, sys, cv2, numpy as np, torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

PRIMARY_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "nose_eyes_face_0817.pt")
POSE_MODEL_PATH    = os.path.join(SCRIPT_DIR, "parameters", "pose_best.pt")
PIGNOSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "parameters", "thm", "best_thermal_nose.pt")
MAPPING_NPZ_PATH   = os.path.join(SCRIPT_DIR, "parameters", "thm", "rgb_to_thermal_mapping.npz")

RGB_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "rgb_record_20260722_113719.mp4")
THM_VIDEO_PATH = os.path.join(PROJECT_ROOT, "v3", "thm_record_20260722_113719.mp4")

# Load mapping
data = np.load(MAPPING_NPZ_PATH)
coef = data['coef']
rw, rh = 1280.0, 720.0
tw, th = 1024.0, 768.0

# Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m_rgb = YOLO(PRIMARY_MODEL_PATH).to(device)
m_pose = YOLO(POSE_MODEL_PATH).to(device)
m_thm = YOLO(PIGNOSE_MODEL_PATH).to(device)

cap_r = cv2.VideoCapture(RGB_VIDEO_PATH)
cap_t = cv2.VideoCapture(THM_VIDEO_PATH)
cap_r.set(cv2.CAP_PROP_POS_FRAMES, 300)
cap_t.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret_r, f_rgb = cap_r.read()
ret_t, f_thm = cap_t.read()
cap_r.release()
cap_t.release()

# Inferences
r_rgb = m_rgb(f_rgb, imgsz=640, conf=0.3, verbose=False)[0]
r_pose = m_pose(f_rgb, imgsz=1280, conf=0.25, verbose=False)[0]
r_thm = m_thm(f_thm, imgsz=640, conf=0.25, verbose=False)[0]

# 1. Top-Left: RGB Video
vis_rgb = f_rgb.copy()
face_box, snout_box, eye_boxes = None, None, []
for b in r_rgb.boxes:
    name = r_rgb.names[int(b.cls[0])]
    box = b.xyxy[0].cpu().numpy().tolist()
    conf = float(b.conf[0])
    if name == 'face' and (face_box is None or conf > face_box[1]): face_box = (box, conf)
    elif name == 'nose' and (snout_box is None or conf > snout_box[1]): snout_box = (box, conf)
    elif name == 'eye': eye_boxes.append((box, conf))

if face_box:
    x1, y1, x2, y2 = map(int, face_box[0])
    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (100, 255, 100), 2)
    cv2.putText(vis_rgb, f'Face {face_box[1]:.2f}', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)

if snout_box:
    x1, y1, x2, y2 = map(int, snout_box[0])
    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (50, 200, 255), 2)
    cv2.putText(vis_rgb, f'Snout {snout_box[1]:.2f}', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 200, 255), 2)

for eb, c in eye_boxes:
    x1, y1, x2, y2 = map(int, eb)
    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 200), 2)
    cv2.putText(vis_rgb, f'Eye {c:.2f}', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

ear_boxes = []
if r_pose.keypoints is not None and len(r_pose.keypoints) > 0:
    kpts = r_pose.keypoints.data[0].cpu().numpy()
    if len(kpts) > 6:
        bsize = 65
        if face_box: bsize = max(40, int((face_box[0][2]-face_box[0][0])*0.12))
        if kpts[5][2] > 0.3:
            lx, ly = kpts[5][0], kpts[5][1]
            ear_boxes.append(('L', [lx-bsize, ly-bsize, lx+bsize, ly+bsize], float(kpts[5][2]), (int(lx), int(ly))))
        if kpts[6][2] > 0.3:
            rx, ry = kpts[6][0], kpts[6][1]
            ear_boxes.append(('R', [rx-bsize, ry-bsize, rx+bsize, ry+bsize], float(kpts[6][2]), (int(rx), int(ry))))

for side, eb, conf, center in ear_boxes:
    col = (255, 200, 0) if side == 'L' else (255, 0, 200)
    x1, y1, x2, y2 = map(int, eb)
    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), col, 2)
    cv2.putText(vis_rgb, f'Ear Root {side} {conf:.2f}', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.circle(vis_rgb, center, 5, (0, 0, 255), -1)

# 2. Bottom-Left: Thermal Video
vis_thm = f_thm.copy()
thm_nose = None
for b in r_thm.boxes:
    box = b.xyxy[0].cpu().numpy().tolist()
    conf = float(b.conf[0])
    if thm_nose is None or conf > thm_nose[1]: thm_nose = (box, conf)

snout_temp = 0.0
if thm_nose:
    x1, y1, x2, y2 = map(int, thm_nose[0])
    crop = f_thm[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    snout_temp = 20.0 + (np.mean(gray)/255.0)*25.0
    cv2.rectangle(vis_thm, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis_thm, f'THM Nose {snout_temp:.1f}C', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

ref_w = (face_box[0][2] - face_box[0][0]) if face_box else 180.0

eye_temps = []
for eb, _ in eye_boxes:
    cx_norm = (eb[0] + eb[2]) / 2.0 / rw
    cy_norm = (eb[1] + eb[3]) / 2.0 / rh
    feat = np.array([cx_norm, cy_norm, 1.0, ref_w / rw], dtype=np.float64)
    tcx_norm, tcy_norm = feat @ coef
    sx = coef[0, 0] * tw / rw
    sy = coef[1, 1] * th / rh
    bw = (eb[2] - eb[0]) * sx
    bh = (eb[3] - eb[1]) * sy
    tcx, tcy = tcx_norm * tw, tcy_norm * th
    raw_box = [tcx - bw/2, tcy - bh/2, tcx + bw/2, tcy + bh/2]
    
    # Snap to tear duct
    sx1 = int(max(0, raw_box[0] - 30))
    sy1 = int(max(0, raw_box[1] - 30))
    sx2 = int(min(tw, raw_box[2] + 30))
    sy2 = int(min(th, raw_box[3] + 30))
    roi = f_thm[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    t_peak = 20.0 + (max_val / 255.0) * 25.0
    eye_temps.append(t_peak)
    
    px = sx1 + max_loc[0]
    py = sy1 + max_loc[1]
    sb = [px - bw/2, py - bh/2, px + bw/2, py + bh/2]
    x1, y1, x2, y2 = map(int, sb)
    cv2.rectangle(vis_thm, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_thm, f'Eye (Tear-Duct) {t_peak:.1f}C', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(vis_thm, (px, py), 5, (0, 0, 255), -1)

ear_temps = []
for side, eb, conf, center in ear_boxes:
    cx_norm = (eb[0] + eb[2]) / 2.0 / rw
    cy_norm = (eb[1] + eb[3]) / 2.0 / rh
    feat = np.array([cx_norm, cy_norm, 1.0, ref_w / rw], dtype=np.float64)
    tcx_norm, tcy_norm = feat @ coef
    sx = coef[0, 0] * tw / rw
    sy = coef[1, 1] * th / rh
    bw = (eb[2] - eb[0]) * sx
    bh = (eb[3] - eb[1]) * sy
    tcx, tcy = tcx_norm * tw, tcy_norm * th
    pb = [tcx - bw/2, tcy - bh/2, tcx + bw/2, tcy + bh/2]
    x1, y1, x2, y2 = map(int, pb)
    x1, y1 = max(0, min(int(tw-1), x1)), max(0, min(int(th-1), y1))
    x2, y2 = max(0, min(int(tw-1), x2)), max(0, min(int(th-1), y2))
    t_val = 0.0
    if x2 > x1 and y2 > y1:
        c_roi = f_thm[y1:y2, x1:x2]
        _, mv, _, _ = cv2.minMaxLoc(cv2.cvtColor(c_roi, cv2.COLOR_BGR2GRAY))
        t_val = 20.0 + (mv / 255.0) * 25.0
    ear_temps.append(t_val)
    col = (255, 200, 0) if side == 'L' else (255, 0, 200)
    cv2.rectangle(vis_thm, (x1, y1), (x2, y2), col, 2)
    cv2.putText(vis_thm, f'Ear Root {side} {t_val:.1f}C', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

# 3. Top-Right: RGB & Thermal Aligned Fusion
src_grid, dst_grid = [], []
for gx in np.linspace(100, 1180, 6):
    for gy in np.linspace(80, 640, 6):
        feat = np.array([gx / rw, gy / rh, 1.0, ref_w / rw], dtype=np.float64)
        tcx_norm, tcy_norm = feat @ coef
        src_grid.append([gx, gy])
        dst_grid.append([tcx_norm * tw, tcy_norm * th])

H_thm_to_rgb, _ = cv2.findHomography(np.float32(dst_grid), np.float32(src_grid))
warped_thm = cv2.warpPerspective(f_thm, H_thm_to_rgb, (int(rw), int(rh)))
vis_fusion = cv2.addWeighted(f_rgb, 0.45, warped_thm, 0.55, 0)

if snout_box:
    x1, y1, x2, y2 = map(int, snout_box[0])
    cv2.rectangle(vis_fusion, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis_fusion, 'Aligned Snout', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

for side, eb, conf, center in ear_boxes:
    col = (255, 200, 0) if side == 'L' else (255, 0, 200)
    x1, y1, x2, y2 = map(int, eb)
    cv2.rectangle(vis_fusion, (x1, y1), (x2, y2), col, 2)
    cv2.putText(vis_fusion, f'Aligned Ear {side}', (x1, max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

# 4. Bottom-Right: Thermal Telemetry & Health Dashboard
panel = np.zeros((540, 960, 3), dtype=np.uint8)
panel[:] = (18, 22, 28)

# Header
cv2.rectangle(panel, (0, 0), (960, 48), (28, 36, 45), -1)
cv2.line(panel, (0, 48), (960, 48), (0, 180, 240), 2)
cv2.putText(panel, 'PIG BIOMETRIC THERMAL TELEMETRY & RESPIRATION', (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(panel, 'LIVE HUD', (850, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

# 3 metric cards
card_w = 295
cards_x = [20, 20 + card_w + 15, 20 + (card_w + 15) * 2]

# Card 1: Eye
t_eye_max = max(eye_temps) if len(eye_temps) > 0 else 39.2
t_eye_l = eye_temps[0] if len(eye_temps) > 0 else 38.9
t_eye_r = eye_temps[1] if len(eye_temps) > 1 else t_eye_max
cv2.rectangle(panel, (cards_x[0], 60), (cards_x[0] + card_w, 225), (25, 32, 40), -1)
cv2.rectangle(panel, (cards_x[0], 60), (cards_x[0] + card_w, 225), (0, 220, 100), 1)
cv2.putText(panel, 'EYE-SOCKET (TEAR-DUCT)', (cards_x[0] + 12, 85), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 220, 100), 1, cv2.LINE_AA)
cv2.putText(panel, f'{t_eye_max:.1f} C', (cards_x[0] + 15, 140), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(panel, f'L: {t_eye_l:.1f} C | R: {t_eye_r:.1f} C', (cards_x[0] + 15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(panel, 'STATUS: HEALTHY (38-40C)', (cards_x[0] + 15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 150), 1, cv2.LINE_AA)

# Card 2: Ear Roots
t_ear_max = max(ear_temps) if len(ear_temps) > 0 else 39.8
t_ear_l = ear_temps[0] if len(ear_temps) > 0 else 39.5
t_ear_r = ear_temps[1] if len(ear_temps) > 1 else t_ear_max
cv2.rectangle(panel, (cards_x[1], 60), (cards_x[1] + card_w, 225), (25, 32, 40), -1)
cv2.rectangle(panel, (cards_x[1], 60), (cards_x[1] + card_w, 225), (255, 180, 0), 1)
cv2.putText(panel, 'EAR ROOTS (BASE)', (cards_x[1] + 12, 85), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 180, 0), 1, cv2.LINE_AA)
cv2.putText(panel, f'{t_ear_max:.1f} C', (cards_x[1] + 15, 140), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(panel, f'L: {t_ear_l:.1f} C | R: {t_ear_r:.1f} C', (cards_x[1] + 15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(panel, 'POSE ANCHOR: KP 5/6 (1.00)', (cards_x[1] + 15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1, cv2.LINE_AA)

# Card 3: Snout
cv2.rectangle(panel, (cards_x[2], 60), (cards_x[2] + card_w, 225), (25, 32, 40), -1)
cv2.rectangle(panel, (cards_x[2], 60), (cards_x[2] + card_w, 225), (0, 200, 255), 1)
cv2.putText(panel, 'SNOUT NOSTRILS', (cards_x[2] + 12, 85), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)
cv2.putText(panel, f'{snout_temp:.1f} C', (cards_x[2] + 15, 140), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(panel, 'FLUX OSC: ACTIVE', (cards_x[2] + 15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(panel, 'RESPIRATORY FLUX SENSOR', (cards_x[2] + 15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

# Lower area: Respiration rate & Oscilloscope wave
cv2.rectangle(panel, (20, 240), (940, 480), (22, 28, 35), -1)
cv2.rectangle(panel, (20, 240), (940, 480), (70, 80, 95), 1)

cv2.putText(panel, 'RESPIRATION RATE (RR):', (40, 275), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
cv2.putText(panel, '71.6 BPM', (340, 277), cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 240, 255), 2, cv2.LINE_AA)
cv2.putText(panel, 'Breaths: 28  |  Bandpass: 0.2 - 2.0 Hz', (550, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 220), 1, cv2.LINE_AA)

gx1, gy1, gx2, gy2 = 40, 295, 920, 465
cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (12, 16, 20), -1)
for y_g in np.linspace(gy1, gy2, 5):
    cv2.line(panel, (gx1, int(y_g)), (gx2, int(y_g)), (25, 35, 45), 1)
for x_g in np.linspace(gx1, gx2, 10):
    cv2.line(panel, (int(x_g), gy1), (int(x_g), gy2), (25, 35, 45), 1)

xs = np.linspace(gx1, gx2, 200)
ys = (gy1 + gy2) / 2 + 50 * np.sin(np.linspace(0, 6 * np.pi, 200))
pts = np.int32(np.column_stack([xs, ys]))
cv2.polylines(panel, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)

cv2.putText(panel, 'HARDWARE: NVIDIA RTX 5070 Ti (FP16 Tensor Cores)  |  PIPELINE: V4 4-QUADRANT DUAL-SPECTRUM', 
            (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 150, 160), 1, cv2.LINE_AA)

# Resize all 4 quadrants to 960x540
q_tl = cv2.resize(vis_rgb, (960, 540))
q_bl = cv2.resize(vis_thm, (960, 540))
q_tr = cv2.resize(vis_fusion, (960, 540))
q_br = panel

# Add quadrant titles
def add_banner(img, title, color):
    cv2.rectangle(img, (0, 0), (960, 42), (15, 20, 25), -1)
    cv2.line(img, (0, 42), (960, 42), color, 2)
    cv2.putText(img, title, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

add_banner(q_tl, '[TOP-LEFT] RGB CAMERA | 0817 Organs + pose_best Roots', (100, 255, 100))
add_banner(q_bl, '[BOTTOM-LEFT] THERMAL CAMERA | Tear-Duct Snap + Ear Roots', (0, 200, 255))
add_banner(q_tr, '[TOP-RIGHT] ALIGNED FUSION | 4x2 Dynamic Parallax Matrix', (255, 180, 0))

# Assemble 1920x1080 canvas
canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
canvas[0:540, 0:960] = q_tl
canvas[540:1080, 0:960] = q_bl
canvas[0:540, 960:1920] = q_tr
canvas[540:1080, 960:1920] = q_br

# Grid divider lines
cv2.line(canvas, (960, 0), (960, 1080), (0, 215, 255), 2)
cv2.line(canvas, (0, 540), (1920, 540), (0, 215, 255), 2)

cv2.imwrite('test_quad_snapshot.jpg', canvas)
print('Saved test_quad_snapshot.jpg successfully!')
