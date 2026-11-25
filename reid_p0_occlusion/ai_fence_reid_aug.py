import torch  # Import first on Windows to avoid WinError 4551
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

OVERLAY_DIR_DEFAULT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "03_datasets",
        "ai_occlusion_assets",
        "ai_overlays_rgba",
    )
)

TRAIN_OVERLAY_NAMES = [
    "ai_asset_01_rusted_bars_rgba.png",
    "ai_asset_02_cross_pipes_rgba.png",
    "ai_asset_04_red_divider_rgba.png",
    "ai_asset_05_chain_pipe_rgba.png",
]

TEST_UNSEEN_OVERLAY_NAMES = [
    "ai_asset_03_wire_grate_rgba.png",
    "ai_asset_06_diag_gate_rgba.png",
]


def load_overlays(names, overlay_dir=OVERLAY_DIR_DEFAULT, max_side=384):
    """Load RGBA fence overlays using PIL (safe on non-ASCII Windows paths)."""
    loaded = []
    for name in names:
        p = os.path.join(overlay_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"AI overlay not found: {p}")
        im = Image.open(p).convert("RGBA")
        if max_side is not None and max(im.size) > max_side:
            ow, oh = im.size
            s = max_side / float(max(ow, oh))
            im = im.resize((max(16, int(ow * s)), max(16, int(oh * s))), Image.Resampling.BILINEAR)
        loaded.append((name, im))
    return loaded


def _transform_overlay_patch(rgba_img, target_w, target_h, rng):
    """Create a randomly scaled, rotated, jittered RGBA numpy array (H, W, 4) uint8."""
    ow, oh = rgba_img.size
    scale = float(rng.uniform(0.55, 1.35))
    base_side = max(target_w, target_h)
    new_w = max(32, int(round(base_side * scale * (ow / max(ow, oh)))))
    new_h = max(32, int(round(base_side * scale * (oh / max(ow, oh)))))
    patch = rgba_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.25:
        patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    angle = float(rng.uniform(-18.0, 18.0))
    patch = patch.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

    pw, ph = patch.size
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    min_vis_x = max(16, int(0.15 * target_w))
    min_vis_y = max(16, int(0.15 * target_h))
    dx = int(rng.integers(-pw + min_vis_x, target_w - min_vis_x + 1))
    dy = int(rng.integers(-ph + min_vis_y, target_h - min_vis_y + 1))
    canvas.paste(patch, (dx, dy))

    arr = np.asarray(canvas, dtype=np.uint8).copy()
    # Fast vectorized brightness/contrast jitter on RGB channels only
    alpha_mask = arr[:, :, 3] > 0
    if np.any(alpha_mask):
        gain = float(rng.uniform(0.75, 1.22))
        bias = float(rng.uniform(-14.0, 14.0))
        rgb = arr[:, :, :3].astype(np.float32) * gain + bias
        arr[:, :, :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        if rng.random() < 0.30:
            arr[:, :, :3] = cv2.GaussianBlur(arr[:, :, :3], (3, 3), 0.8)
    return arr


def apply_ai_fence_occlusion(pil_img, overlays, rng, occ_range=(0.15, 0.42), max_tries=8):
    """
    Alpha-blend a realistic AI fence overlay onto a PIL RGB pig face image,
    clamping effective alpha occlusion ratio into `occ_range` (never full cover).
    Returns: (occluded_pil_rgb, occ_fraction, overlay_name)
    """
    base_rgb = pil_img.convert("RGB")
    w, h = base_rgb.size
    target_mid = 0.5 * (occ_range[0] + occ_range[1])

    best_arr = None
    best_frac = None
    best_name = None
    best_dist = 1e9

    for _ in range(max_tries):
        name, ov = overlays[int(rng.integers(0, len(overlays)))]
        arr = _transform_overlay_patch(ov, w, h, rng)
        frac = float((arr[:, :, 3] > 30).mean())

        if occ_range[0] <= frac <= occ_range[1]:
            best_arr, best_frac, best_name = arr, frac, name
            break

        penalty = 0.35 if frac > 0.50 else 0.0
        dist = abs(frac - target_mid) + penalty
        if dist < best_dist:
            best_dist = dist
            best_arr, best_frac, best_name = arr, frac, name

    if best_frac is not None and best_frac > 0.50:
        cut_side = int(rng.integers(0, 4))
        if cut_side == 0:
            best_arr[:, : int(w * 0.55), 3] = 0
        elif cut_side == 1:
            best_arr[:, int(w * 0.45) :, 3] = 0
        elif cut_side == 2:
            best_arr[: int(h * 0.55), :, 3] = 0
        else:
            best_arr[int(h * 0.45) :, :, 3] = 0
        best_frac = float((best_arr[:, :, 3] > 30).mean())

    # Fast contact shadow along fence edges via cv2.dilate + GaussianBlur
    alpha_u8 = best_arr[:, :, 3]
    dilated = cv2.dilate(alpha_u8, np.ones((5, 5), dtype=np.uint8), iterations=1)
    dilated = cv2.GaussianBlur(dilated, (5, 5), 1.8)
    shadow_np = np.clip((dilated.astype(np.float32) - alpha_u8.astype(np.float32)) / 255.0, 0.0, 1.0)

    base_np = np.asarray(base_rgb, dtype=np.float32)
    base_shadowed = base_np * (1.0 - 0.28 * shadow_np[:, :, None])
    alpha_f = (alpha_u8.astype(np.float32) / 255.0)[:, :, None]
    ov_rgb = best_arr[:, :, :3].astype(np.float32)
    composed = np.clip(ov_rgb * alpha_f + base_shadowed * (1.0 - alpha_f), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(composed, mode="RGB"), float(best_frac), best_name


class RandomAIFenceOcclusion:
    """Torchvision-compatible PIL transform applying realistic AI fence overlays."""

    def __init__(self, overlays, p=0.60, occ_range=(0.15, 0.40), seed=None):
        self.overlays = overlays
        self.p = float(p)
        self.occ_range = occ_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if self.rng.random() >= self.p:
            return img
        occ_img, _, _ = apply_ai_fence_occlusion(img, self.overlays, self.rng, occ_range=self.occ_range)
        return occ_img
