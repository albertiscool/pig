"""
同一台雙鏡頭同時錄的 RGB / 熱像影片：用鼻子框算 RGB -> 熱像的座標轉換，
再把 RGB 模型偵測到的眼睛框投影到熱像上，輸出每一幀的熱像眼睛框與對照影片。

步驟：
  1. 逐幀偵測：熱像 -> 05_yolov8m_FINAL（鼻子）；RGB -> rgb_pt/weights2（eye / nose / root；--rgb-weights 可換）
  2. 對時：兩部影片幀數相同，但同一幀的畫面時間戳記，熱像比 RGB 快 1 秒
     （兩部影片都人工確認過）。從時間戳記秒數跳號的相位算出精確到幀的位移
  3. 視差修正的轉換：兩顆鏡頭並排，物體越近、兩邊位置差越大（實測熱像 - RGB 的
     水平位置差與鼻框大小的相關係數 0.74），單一 homography 套不上。所以用
       熱像中心 = A · [RGB 中心 x, RGB 中心 y, 1, RGB 鼻框寬]   （正規化座標）
     框寬那一項就是「距離越近、偏移越大」的視差項。用 RANSAC 剔除配錯的點
     （大多是 RGB 模型把豬蹄當鼻子），每個 1 秒區塊輪流當驗證做交叉驗證
  4. 眼睛框：中心用同一幀 RGB 鼻框寬當距離指標套轉換，寬高依比例縮放

限制：
  - 眼睛比鼻子離鏡頭遠一點（豬鼻突出），用鼻框寬當眼睛的距離指標，視差會略為高估
  - 熱像沒有眼睛的標準答案，眼睛框的誤差只能用鼻子的驗證誤差估計

輸出（calibration/video_20260722/）：
    detections_<tag>.json            兩邊逐幀偵測結果（--reuse-detections 可跳過推論）
    mapping.json                     對時位移、轉換係數、驗證誤差
    thermal_eye_boxes_<tag>.json     每一幀（熱像幀號）的熱像眼睛框
    eye_overlay_<tag>.mp4            左 RGB（RGB 權重的眼睛、鼻子框）、右熱像（熱像權重的鼻子框 + 投影過來的眼睛框）

用法：
    venv/Scripts/python scripts/video_eye_to_thermal.py              # data/pig_video_rgb 裡所有有對應熱像的影片
    venv/Scripts/python scripts/video_eye_to_thermal.py --reuse-detections
    venv/Scripts/python scripts/video_eye_to_thermal.py --reuse-detections --eye-conf 0.3   # 輸出檔名加 _eye0.3
"""

import argparse
import glob
import io
import json
import os
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(LOCAL_DIR, ".."))
ROOT = r"D:\pig\pig_project" if os.path.exists(r"D:\pig\pig_project") else LOCAL_DIR

RGB_VIDEO = os.path.join(ROOT, "data", "pig_video_rgb", "rgb_record_20260722_{}.mp4")
THM_VIDEO = os.path.join(ROOT, "data", "pig_video", "thm_record_20260722_{}.mp4")
# 有 RGB 也有同時間熱像的影片（檔名的時間 tag，例如 111352）
TAGS = sorted(t for t in (os.path.basename(p)[-10:-4] for p in glob.glob(RGB_VIDEO.format("*")))
              if os.path.exists(THM_VIDEO.format(t)))

# 優先使用 xavier/parameters 本地模型
_local_thm_w = os.path.join(LOCAL_DIR, "parameters", "thm", "best_thermal_nose.pt")
_fallback_thm_w = os.path.join(ROOT, "work_dirs", "05_yolov8m_FINAL", "weights", "best.pt")
THERMAL_WEIGHTS = _local_thm_w if os.path.exists(_local_thm_w) else _fallback_thm_w

_local_rgb_w = os.path.join(LOCAL_DIR, "parameters", "nose_eyes_face_0817.pt")
_fallback_rgb_w = os.path.join(ROOT, "rgb_pt", "weights2", "best.pt")
RGB_WEIGHTS = _local_rgb_w if os.path.exists(_local_rgb_w) else _fallback_rgb_w

OUT_DIR = os.path.join(LOCAL_DIR, "calibration", "video_20260722")

RW, RH, TW, TH = 1280, 720, 1024, 768
FPS = 30
CONF_DET = 0.25          # 偵測結果先存到這個門檻，之後再篩
CONF_RGB_NOSE = 0.3      # 算轉換用的 RGB 鼻子門檻（RGB 模型在這批影片上信心偏低）
CONF_THM_NOSE = 0.5      # 算轉換用的熱像鼻子門檻
CONF_EYE = 0.5           # 要投影的眼睛框門檻
RANSAC_THR = 0.04        # 正規化座標，約熱像 40 px
STAMP_LEAD_S = 1         # 同一幀時，熱像時間戳記比 RGB 快幾秒（人工看畫面確認）
# 時間戳記秒數最後一位的位置（y1, y2, x1, x2）
STAMP_ROI = {"rgb": (25, 55, 1125, 1148), "thm": (28, 62, 868, 893)}


# ---------------------------------------------------------------- 偵測
def boxes_of(result, cls_id=None):
    out = []
    for b in result.boxes:
        if cls_id is not None and int(b.cls[0]) != cls_id:
            continue
        out.append({"xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()], "conf": round(float(b.conf[0]), 3)})
    return sorted(out, key=lambda d: -d["conf"])


def detect(tag, rgb_weights, old=None):
    """逐幀偵測。old 是之前存的偵測結果：熱像那邊跟 RGB 權重無關，有就直接沿用，只重跑 RGB。"""
    from ultralytics import YOLO
    rgb_model = YOLO(rgb_weights)
    ids = {name: k for k, name in rgb_model.names.items()}
    rgb_sz = int(rgb_model.overrides.get("imgsz") or 640)  # 用權重訓練時的解析度
    thm_model = None if old and old.get("thm") else YOLO(THERMAL_WEIGHTS)
    cr, ct = cv2.VideoCapture(RGB_VIDEO.format(tag)), cv2.VideoCapture(THM_VIDEO.format(tag))
    rgb, thm = [], ([] if thm_model else old["thm"])
    while True:
        ok1, fr = cr.read()
        ok2, ft = ct.read() if thm_model else (False, None)
        if not ok1 and not ok2:
            break
        if ok1:
            r = rgb_model.predict(fr, imgsz=rgb_sz, conf=CONF_DET, verbose=False)[0]
            rgb.append({"nose": boxes_of(r, ids["nose"]), "eyes": boxes_of(r, ids["eye"]),
                        "face": boxes_of(r, ids["face"]) if "face" in ids else []})
        if ok2:
            t = thm_model.predict(ft, imgsz=960, conf=CONF_DET, verbose=False)[0]
            thm.append({"nose": boxes_of(t)})
    d = {"rgb": rgb, "thm": thm, "rgb_weights": os.path.relpath(rgb_weights, ROOT)}
    with open(os.path.join(OUT_DIR, f"detections_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"[{tag}] 偵測完成：RGB {len(rgb)} 幀、熱像 {len(thm)} 幀")
    return d


# ---------------------------------------------------------------- 對時
def stamp_phase(path, roi):
    """秒數最後一位每秒跳一次；回傳跳號落在每秒第幾幀（0–29）。
    只取「白字且旁邊有黑邊」的像素，排除白色背景的干擾。"""
    y1, y2, x1, x2 = roi
    cap = cv2.VideoCapture(path)
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        c = f[y1:y2, x1:x2]
        near_dark = cv2.dilate((c.max(axis=2) < 90).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m = (c.min(axis=2) > 190) & near_dark
        d.append(0.0 if prev is None else np.logical_xor(m, prev).mean())
        prev = m
    d = np.array(d)
    return int(np.argmax([d[p::FPS].mean() for p in range(FPS)]))


def frame_shift(tag):
    """RGB 第 i 幀對應熱像第 i + shift 幀。
    RGB 在第 pr 幀跳到下一秒 S；熱像早 STAMP_LEAD_S 秒，在第 pt - STAMP_LEAD_S * FPS 幀就跳到 S。"""
    pr = stamp_phase(RGB_VIDEO.format(tag), STAMP_ROI["rgb"])
    pt = stamp_phase(THM_VIDEO.format(tag), STAMP_ROI["thm"])
    return (pt - STAMP_LEAD_S * FPS) - pr, pr, pt


# ---------------------------------------------------------------- 轉換
def nose_pairs(dets, shifts):
    rows = []
    for tag, d in dets.items():
        for i, r in enumerate(d["rgb"]):
            j = i + shifts[tag]
            if not 0 <= j < len(d["thm"]):
                continue
            rn = [b for b in r["nose"] if b["conf"] >= CONF_RGB_NOSE]
            tn = [b for b in d["thm"][j]["nose"] if b["conf"] >= CONF_THM_NOSE]
            if rn and tn:
                rows.append({"tag": tag, "i": i, "j": j, "rgb": rn[0]["xyxy"], "thm": tn[0]["xyxy"]})
    return rows


def rgb_feat(box, w_ref=None):
    """正規化 RGB 特徵 [cx, cy, 1, 距離指標(框寬)]；w_ref 給了就用它當距離指標。"""
    x1, y1, x2, y2 = box
    w = (x2 - x1) if w_ref is None else w_ref
    return [(x1 + x2) / 2 / RW, (y1 + y2) / 2 / RH, 1.0, w / RW]


def thm_center(box):
    return [(box[0] + box[2]) / 2 / TW, (box[1] + box[3]) / 2 / TH]


def ransac(A, Y, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(A), A.shape[1] + 1, replace=False)
        c, *_ = np.linalg.lstsq(A[idx], Y[idx], rcond=None)
        inl = np.linalg.norm(A @ c - Y, axis=1) < RANSAC_THR
        if best is None or inl.sum() > best.sum():
            best = inl
    c, *_ = np.linalg.lstsq(A[best], Y[best], rcond=None)
    return c, best


def map_box(coef, box, w_ref):
    """RGB 框 -> 熱像框：中心套轉換（距離指標用 w_ref），寬高依線性項縮放。"""
    cx, cy = np.array(rgb_feat(box, w_ref)) @ coef
    sx, sy = coef[0, 0] * TW / RW, coef[1, 1] * TH / RH
    w, h = (box[2] - box[0]) * sx, (box[3] - box[1]) * sy
    cx, cy = cx * TW, cy * TH
    return [float(np.clip(cx - w / 2, 0, TW)), float(np.clip(cy - h / 2, 0, TH)),
            float(np.clip(cx + w / 2, 0, TW)), float(np.clip(cy + h / 2, 0, TH))]


def depth_ref(rgb_dets, i, nose_face, window=15):
    """眼睛的距離指標（RGB 鼻框寬）：同一幀的鼻框 > 前後 window 幀內最近的鼻框 > 同一幀臉框寬 × 鼻/臉寬比例。"""
    for k in sorted(range(max(0, i - window), min(len(rgb_dets), i + window + 1)), key=lambda k: abs(k - i)):
        nose = [b for b in rgb_dets[k]["nose"] if b["conf"] >= CONF_RGB_NOSE]
        if nose:
            return nose[0]["xyxy"][2] - nose[0]["xyxy"][0], "nose" if k == i else f"nose@{k - i:+d}"
    face = [b for b in rgb_dets[i]["face"] if b["conf"] >= 0.3]
    if face:
        return (face[0]["xyxy"][2] - face[0]["xyxy"][0]) * nose_face, "face"
    return None, None


def load_mapping(path=os.path.join(OUT_DIR, "rgb_to_thermal_mapping.npz")):
    """載入轉換權重，回傳 coef（4x2），配合 map_box(coef, RGB 框, RGB 鼻框寬) 使用。"""
    return np.load(path)["coef"]


def draw(img, xyxy, color, label, th=3):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    cv2.putText(img, label, (x1, max(y1 - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-detections", action="store_true")
    ap.add_argument("--eye-conf", type=float, default=CONF_EYE, help="要投影的 RGB 眼睛框門檻")
    ap.add_argument("--rgb-weights", default=RGB_WEIGHTS, help="RGB 模型權重（需有 eye、nose 類別）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    sfx = "" if args.eye_conf == CONF_EYE else f"_eye{args.eye_conf:g}"  # 非預設門檻另存，不覆蓋

    dets = {}
    for tag in TAGS:
        p = os.path.join(OUT_DIR, f"detections_{tag}.json")
        old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
        same_rgb = old is not None and old.get("rgb_weights", os.path.join("rgb_pt", "weights", "best.pt")) == \
            os.path.relpath(args.rgb_weights, ROOT)
        dets[tag] = old if args.reuse_detections and same_rgb else detect(tag, args.rgb_weights, old)
        d = dets[tag]
        for thr in sorted({0.3, args.eye_conf, CONF_EYE}):
            n_eye = sum(any(b["conf"] >= thr for b in r["eyes"]) for r in d["rgb"])
            print(f"[{tag}] RGB 有眼睛（≥{thr}）：{n_eye} / {len(d['rgb'])} 幀")

    shifts = {}
    for tag in TAGS:
        s, pr, pt = frame_shift(tag)
        shifts[tag] = s
        print(f"[{tag}] 時間戳記跳號：RGB 每秒第 {pr} 幀、熱像第 {pt} 幀 → RGB 第 i 幀對熱像第 i{s:+d} 幀")

    rows = nose_pairs(dets, shifts)
    A = np.array([rgb_feat(r["rgb"]) for r in rows]); Y = np.array([thm_center(r["thm"]) for r in rows])
    coef, inl = ransac(A, Y)
    print(f"\n鼻子配對 {len(rows)} 組，RANSAC 內點 {inl.sum()}（{inl.mean():.0%}）；其餘大多是 RGB 模型把豬蹄當鼻子")

    # 留一秒區塊交叉驗證（只看全體擬合的內點，配錯的不算）
    blocks = np.array([f"{r['tag']}_{r['i'] // FPS}" for r in rows])
    errs = []
    for b in np.unique(blocks):
        tr, te = blocks != b, (blocks == b) & inl
        if te.any():
            c, _ = ransac(A[tr], Y[tr])
            errs.extend(np.linalg.norm((A[te] @ c - Y[te]) * [TW, TH], axis=1))
    errs = np.array(errs)
    print(f"留一秒交叉驗證（{len(errs)} 組）：鼻子中心誤差 中位數 {np.median(errs):.1f} px、"
          f"平均 {errs.mean():.1f} px、90% {np.percentile(errs, 90):.1f} px（熱像 1024x768）")
    print("轉換係數（列：cx, cy, 常數, 框寬；行：熱像 cx, cy）：\n", np.round(coef, 3))

    # 轉換的「權重檔」：載入方式見 load_mapping()
    np.savez(os.path.join(OUT_DIR, "rgb_to_thermal_mapping.npz"), coef=coef, rgb_size=[RW, RH], thm_size=[TW, TH],
             cv_center_err_median_px=float(np.median(errs)))

    # 留一部影片驗證：用其他影片算轉換，套到沒參與的那一部（看能不能直接用在新影片）
    tags_arr = np.array([r["tag"] for r in rows])
    per_video = {}
    for tag in TAGS:
        te = (tags_arr == tag) & inl
        lovo = None
        if te.any() and (tags_arr != tag).sum() >= 10:
            c, _ = ransac(A[tags_arr != tag], Y[tags_arr != tag])
            e = np.linalg.norm((A[te] @ c - Y[te]) * [TW, TH], axis=1)
            lovo = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
        per_video[tag] = {"pairs": int((tags_arr == tag).sum()), "inliers": int(te.sum()), "lovo_center_err_px": lovo}
    all_lovo = [v["lovo_center_err_px"]["median"] for v in per_video.values() if v["lovo_center_err_px"]]
    if all_lovo:
        print(f"留一部影片驗證：各影片鼻子中心誤差中位數 {min(all_lovo):.1f}–{max(all_lovo):.1f} px")

    # 眼睛的距離指標：同一幀 RGB 鼻框寬；沒有鼻子就用臉框寬 × 鼻/臉寬比例
    ratio = [(n[0]["xyxy"][2] - n[0]["xyxy"][0]) / (f[0]["xyxy"][2] - f[0]["xyxy"][0])
             for d in dets.values() for r in d["rgb"] for n, f in [(r["nose"], r["face"])]
             if n and f and n[0]["conf"] >= 0.5 and f[0]["conf"] >= 0.5]
    nose_face = float(np.median(ratio)) if ratio else 0.3

    summary = {}
    for tag in TAGS:
        d, s = dets[tag], shifts[tag]
        cr, ct = cv2.VideoCapture(RGB_VIDEO.format(tag)), cv2.VideoCapture(THM_VIDEO.format(tag))
        rgb_frames = []
        while True:
            ok, f = cr.read()
            if not ok:
                break
            rgb_frames.append(f)
        out = cv2.VideoWriter(os.path.join(OUT_DIR, f"eye_overlay_{tag}{sfx}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (TW, TH // 2))  # 並排後縮成一半，省空間
        results = []
        j = 0
        while True:
            ok, ft = ct.read()
            if not ok:
                break
            i = j - s
            fr = rgb_frames[i].copy() if 0 <= i < len(rgb_frames) else np.zeros((RH, RW, 3), np.uint8)
            eyes = []
            if 0 <= i < len(d["rgb"]):
                r = d["rgb"][i]
                nose = [b for b in r["nose"] if b["conf"] >= CONF_RGB_NOSE]
                w_ref, ref = depth_ref(d["rgb"], i, nose_face)
                for e in [b for b in r["eyes"] if b["conf"] >= args.eye_conf][:2]:
                    draw(fr, e["xyxy"], (0, 220, 0), f"eye {e['conf']:.2f}")
                    if w_ref is None:
                        eyes.append({"thermal_eye": None, "status": "no_depth_ref", "rgb_eye": e["xyxy"],
                                     "rgb_conf": e["conf"]})
                        continue
                    box = map_box(coef, e["xyxy"], w_ref)
                    cx, cy = np.array(rgb_feat(e["xyxy"], w_ref)) @ coef
                    if not (0 <= cx <= 1 and 0 <= cy <= 1):  # 眼睛中心落在熱像視野外
                        eyes.append({"thermal_eye": None, "status": "outside_thermal_view",
                                     "projected_center": [round(cx * TW, 1), round(cy * TH, 1)],
                                     "rgb_eye": e["xyxy"], "rgb_conf": e["conf"], "depth_ref": ref})
                        continue
                    eyes.append({"thermal_eye": [round(v, 1) for v in box], "status": "ok", "rgb_eye": e["xyxy"],
                                 "rgb_conf": e["conf"], "depth_ref": ref})
                    draw(ft, box, (0, 255, 255), "eye (from RGB)")
                if nose:
                    draw(fr, nose[0]["xyxy"], (255, 120, 0), f"nose {nose[0]['conf']:.2f}")
            tn = [b for b in d["thm"][j]["nose"] if b["conf"] >= CONF_THM_NOSE] if j < len(d["thm"]) else []
            if tn:
                draw(ft, tn[0]["xyxy"], (0, 0, 255), f"thermal nose {tn[0]['conf']:.2f}")
            results.append({"thm_frame": j, "rgb_frame": i, "eyes": eyes})
            cv2.putText(fr, "RGB  green=eye  blue=nose", (20, RH - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(ft, "Thermal  red=nose (thermal model)  yellow=eye (from RGB)",
                        (20, TH - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.write(cv2.resize(np.hstack([cv2.resize(fr, (TW, TH)), ft]), (TW, TH // 2)))
            j += 1
        out.release()
        with open(os.path.join(OUT_DIR, f"thermal_eye_boxes_{tag}{sfx}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        st = {}
        for r in results:
            for e in r["eyes"]:
                st[e["status"]] = st.get(e["status"], 0) + 1
        n = sum(any(e["status"] == "ok" for e in r["eyes"]) for r in results)
        # 比較可信的：RGB 眼睛信心 ≥ 0.4，且距離指標是同一幀的鼻框
        n_rel = sum(any(e["status"] == "ok" and e["rgb_conf"] >= 0.4 and e.get("depth_ref") == "nose"
                        for e in r["eyes"]) for r in results)
        summary[tag] = {"frames": len(results), "frames_with_thermal_eye": n, "frames_with_reliable_thermal_eye": n_rel,
                        "eye_status_counts": st, **per_video[tag], "frame_shift": s}
        print(f"[{tag}] 有熱像眼睛框的幀：{n} / {len(results)}；RGB 眼睛框狀態 {st}"
              f" → eye_overlay_{tag}{sfx}.mp4、thermal_eye_boxes_{tag}{sfx}.json")

    json.dump({"frame_shift": shifts, "stamp_lead_s": STAMP_LEAD_S, "coef": coef.tolist(),
               "coef_rows": ["rgb_cx", "rgb_cy", "const", "rgb_nose_w"], "coef_cols": ["thm_cx", "thm_cy"],
               "normalized_by": {"rgb": [RW, RH], "thm": [TW, TH]}, "n_pairs": len(rows), "n_inliers": int(inl.sum()),
               "cv_center_err_px": {"median": float(np.median(errs)), "mean": float(errs.mean()),
                                    "p90": float(np.percentile(errs, 90))},
               "eye_conf": args.eye_conf, "nose_face_width_ratio": nose_face, "per_video": summary},
              open(os.path.join(OUT_DIR, f"mapping{sfx}.json"), "w"), indent=2)

    print("\n影片      幀數  位移  鼻子配對  內點  留一部驗證誤差  RGB眼睛  畫面外  熱像眼睛框  較可信")
    for tag, v in summary.items():
        st, lo = v["eye_status_counts"], v["lovo_center_err_px"]
        err = f"{lo['median']:.1f} px" if lo else "—"
        print(f"{tag}  {v['frames']:>5}  {v['frame_shift']:>4}  {v['pairs']:>8}  {v['inliers']:>4}  {err:>14}  "
              f"{sum(st.values()):>7}  {st.get('outside_thermal_view', 0):>6}  {v['frames_with_thermal_eye']:>10}  "
              f"{v['frames_with_reliable_thermal_eye']:>6}")


if __name__ == "__main__":
    main()
