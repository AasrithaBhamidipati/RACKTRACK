# rack_pipeline_all_in_one_updated.py
# ---------------------------------------------------------
# 1) SWITCH: Accuracy-focused grouping with robust detection & best image selection
#    -> outputs ONLY best image per group to: <OUTPUT_ROOT>/segments/switch/
#
# 2) SEGMENTATION (others): rack, cables, ports (connected_port, empty_port)
#    -> crops saved to flat class folders: <OUTPUT_ROOT>/segments/<class>/...
#
# Notes:
# - No audit images are produced
# - No groups_index.json is written
# - Switch are NOT segmented separately again (avoid duplication)
#
# Requirements: ultralytics, opencv-python, pillow, imagehash, numpy

import os
import glob
import json
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash
import torch

# Use CUDA GPU if available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Warning: Running on CPU. Processing will be much slower. Consider using a GPU for faster inference.")

# ============================
# Paths (HARD-CODED)
# ============================
INPUT_DIR    = r"D:\HeroPageBuilder\HeroPageBuilder\files\multiple-images"                 # folder of rack frames
OUTPUT_ROOT  = r"D:\HeroPageBuilder\HeroPageBuilder\segmented_output"         # root output

# Your trained models
MODEL_PATHS = {
    "cables": r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\cable_best.pt",
    "port":   r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\port_count.pt",
    "rack":   r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\rack_best.pt",
    "switch": r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\switch_patch.pt",  # used by SWITCH grouping pipeline only
}
 
# prefer this index first; will still wait until first image WITH detections to seed
SEED_INDEX = 0

# ============================
# Primary detection knobs (PASS 1: strict) for switch grouping
# ============================
CONF_THRESHOLD = 0.18
IOU_NMS        = 0.50
MAX_DET        = 100  # Reduced from 200
IMG_SIZE       = 960  # Reduced from 1280 for faster processing

# Tiling
TILE_W, TILE_H = 640, 640  # Reduced from 960x960
TILE_OVERLAP   = 0.15  # Reduced from 25% to 15%

# TTA - Reduced augmentations for faster processing
USE_FLIP       = False  # Disabled for speed
SCALES         = [1.0]  # Only use original scale

# Class handling for switch model
SWITCH_CLASS_NAMES = {"switch", "network_switch", "Switch"}
PATCH_CLASS_NAMES  = {"patch", "patchpanel", "patch_panel", "Patch_Panel", "Patch-Panel"}

# Post filters (full-frame normalized) for PASS 1
MIN_AREA_RATIO_P1      = 0.001
MIN_LONG_ASPECT_P1     = 1.5
MIN_EDGE_DENSITY_P1    = 0.01
MIN_HORIZ_LINECOUNT_P1 = 3

# Fallback (PASS 2) for switch grouping
FALLBACK_CONF          = 0.25  # Slightly reduced to compensate for fewer scales
FALLBACK_SCALES        = [1.0, 1.25]  # Reduced scales
FALLBACK_MIN_AREA      = 0.0012
FALLBACK_MIN_LONG_ASP  = 1.4
FALLBACK_MIN_EDGE_DENS = 0.012
FALLBACK_MIN_HLINES    = 3

# ±90° TTA final fallback
USE_ROTATE_90_TTA      = False  # Disabled for speed

# Disable audits & indices as requested
AUDIT                  = False

# ============================
# Grouping (same weights)
# ============================
HASH_BITS          = 32  # Reduced from 64
ROTATION_STEP      = 10  # Increased from 5 (fewer angles to check)
TEST_FLIPS         = [False]  # Removed flip testing
ORB_N_FEATURES     = 500  # Reduced from 1000
ORB_DISTANCE_LIMIT = 40
SSIM_SIZE          = 256
H_HIST_BINS        = 50
S_HIST_BINS        = 60
W_ORB, W_SSIM, W_HIST, W_PHASH = 0.40, 0.30, 0.20, 0.10

# ============================
# Segmentation (non-switch)
# ============================
FINAL_CLASSES = ['cables', 'rack', 'patch_panel', 'switch', 'connected_port', 'empty_port']
CONF_THRESHOLDS = {"rack": 0.6, "switch": 0.44, "cables": 0.18, "patch_panel": 0.2}  # switch threshold unused here
MARGIN = 10
MIN_DIM = 128

# ============================
# Final outputs
# ============================
OUTPUT_SEGMENTS_DIR = os.path.join(OUTPUT_ROOT, "")  # flat class folders incl. switch/

# =============== Utils ===============
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def list_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*{e}"), recursive=True))
    return sorted(files)

def read_bgr(path: str) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def clamp_box(x1, y1, x2, y2, W, H):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def crop_with_margin(bgr, box, margin=10, min_dim=128):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
    cl = clamp_box(x1, y1, x2, y2, W, H)
    if cl is None: return None
    x1, y1, x2, y2 = cl
    crop = bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < min_dim or cw < min_dim:
        crop = cv2.resize(crop, (max(cw, min_dim), max(ch, min_dim)), interpolation=cv2.INTER_CUBIC)
    return crop

def draw_box(img, box, color=(0, 255, 0), lbl="switch", conf=None):
    if not AUDIT: return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    t = f"{lbl}" + (f" {conf:.2f}" if conf is not None else "")
    cv2.putText(img, t, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def save_audit(image, path):
    # Disabled as per request
    return

# =============== YOLO ===============
def load_yolo(path):
    from ultralytics import YOLO
    import torch
    model = YOLO(path)
    if DEVICE == "cuda:0":
        model.to(DEVICE)
    return model

def model_classes(model):
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}

def class_ids_by_names(model, wanted: set):
    names = model_classes(model)
    wl = {w.lower() for w in wanted}
    return [cid for cid, name in names.items() if name.lower() in wl]

# =============== Tiling & Fusion (for switch) ===============
def iter_tiles(W, H, tw, th, overlap):
    sx = max(1, int(tw * (1 - overlap)))
    sy = max(1, int(th * (1 - overlap)))
    for y in range(0, max(1, H - th + 1), sy):
        for x in range(0, max(1, W - tw + 1), sx):
            yield x, y, min(x + tw, W), min(y + th, H)
    if (H % sy) != 0:
        y = max(0, H - th)
        for x in range(0, max(1, W - tw + 1), sx):
            yield x, y, min(x + tw, W), min(y + th, H)
    if (W % sx) != 0:
        x = max(0, W - tw)
        for y in range(0, max(1, H - th + 1), sy):
            yield x, y, min(x + tw, W), min(y + th, H)
    yield max(0, W - tw), max(0, H - th), W, H

def soft_nms_merge(boxes, scores, iou_thr=IOU_NMS, score_thr=0.001, sigma=0.5):
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    N = len(scores)
    keep = []
    idxs = list(range(N))
    while idxs:
        i = int(np.argmax(scores[idxs])); i = idxs.pop(i)
        if scores[i] < score_thr: break
        keep.append(i)
        suppressed = []
        for j in idxs:
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            iw, ih = max(0, x2 - x1), max(0, y2 - y1)
            inter = iw * ih
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou = inter / (a + b - inter + 1e-9)
            if iou > 0:
                scores[j] *= np.exp(-(iou ** 2) / sigma)
            if scores[j] < score_thr:
                suppressed.append(j)
        for j in suppressed:
            if j in idxs: idxs.remove(j)
    return keep

def weighted_boxes_fusion(boxes, scores):
    if not boxes: return [], []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    used = np.zeros(len(boxes), dtype=bool)
    fused_boxes, fused_scores = [], []
    for i in np.argsort(-scores):
        if used[i]: continue
        cluster = [i]; used[i] = True
        for j in range(len(boxes)):
            if used[j]: continue
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            iw, ih = max(0, x2 - x1), max(0, y2 - y1)
            inter = iw * ih
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou = inter / (a + b - inter + 1e-9)
            if iou >= IOU_NMS:
                cluster.append(j); used[j] = True
        ws = scores[cluster]; ws = ws / (ws.sum() + 1e-9)
        bb = boxes[cluster]
        fused = (bb * ws[:, None]).sum(axis=0)
        fused_boxes.append(fused.tolist())
        fused_scores.append(float(scores[cluster].max()))
    return fused_boxes, fused_scores

# =============== Rotation mapping (for switch) ===============
def rotate_img(img, direction: str):
    if direction == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def map_box_from_rotated(box, orig_shape, direction: str):
    H, W = orig_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    if direction == "cw":
        nx1, ny1 = W - y2, x1
        nx2, ny2 = W - y1, x2
    elif direction == "ccw":
        nx1, ny1 = y1, H - x2
        nx2, ny2 = y2, H - x1
    else:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    if nx1 > nx2: nx1, nx2 = nx2, nx1
    if ny1 > ny2: ny1, ny2 = ny2, ny1
    return [nx1, ny1, nx2, ny2]

# =============== Core detection passes (for switch) ===============
def _detect_switch_fullframe(bgr: np.ndarray, sw_model, conf, scales) -> List[Tuple[List[int], float]]:
    H, W = bgr.shape[:2]
    rois = [(0, 0, W, H)]
    sw_ids = set(class_ids_by_names(sw_model, SWITCH_CLASS_NAMES))
    patch_ids = set(class_ids_by_names(sw_model, PATCH_CLASS_NAMES))
    all_boxes, all_scores = [], []

    for (rx1, ry1, rx2, ry2) in rois:
        roi = bgr[ry1:ry2, rx1:rx2]
        RH, RW = roi.shape[:2]
        for x1, y1, x2, y2 in iter_tiles(RW, RH, TILE_W, TILE_H, TILE_OVERLAP):
            tile = roi[y1:y2, x1:x2]
            for sc in scales:
                tile_resized = tile if sc == 1.0 else cv2.resize(tile, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                aug_variants = [(tile_resized, False)]
                if USE_FLIP: aug_variants.append((cv2.flip(tile_resized, 1), True))
                rs = sw_model.predict(
                    source=[v for v, _ in aug_variants],
                    conf=conf, iou=IOU_NMS, verbose=False,
                    imgsz=IMG_SIZE, agnostic_nms=True, max_det=MAX_DET
                )
                for (img_aug, flipped), r in zip(aug_variants, rs):
                    for box, cls_id, scf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        cls_id = int(cls_id)
                        if cls_id in patch_ids:  # ignore patch panels in grouping
                            continue
                        if sw_ids and cls_id not in sw_ids:
                            continue
                        bx = np.array(box.tolist(), dtype=float)
                        if flipped:
                            w_aug = img_aug.shape[1]
                            x1f, y1f, x2f, y2f = bx
                            bx = np.array([w_aug - x2f, y1f, w_aug - x1f, y2f], dtype=float)
                        if sc != 1.0:
                            bx /= sc
                        bx[0] += x1; bx[2] += x1
                        bx[1] += y1; bx[3] += y1
                        bx[0] += rx1; bx[2] += rx1
                        bx[1] += ry1; bx[3] += ry1
                        bb = clamp_box(bx[0], bx[1], bx[2], bx[3], W, H)
                        if bb is None: continue
                        all_boxes.append(bb)
                        all_scores.append(float(scf))

    if not all_boxes:
        return []

    keep = soft_nms_merge(all_boxes, all_scores, iou_thr=IOU_NMS, sigma=0.5)
    pr_boxes = [all_boxes[i] for i in keep]
    pr_scores = [all_scores[i] for i in keep]
    fused_boxes, fused_scores = weighted_boxes_fusion(pr_boxes, pr_scores)
    out = []
    for b, s in zip(fused_boxes, fused_scores):
        bb = clamp_box(*b, W, H)
        if bb: out.append((bb, s))
    return out

def _detect_switch_rotated(bgr: np.ndarray, sw_model, conf, scales, direction: str):
    rot = rotate_img(bgr, direction)
    dets = _detect_switch_fullframe(rot, sw_model, conf, scales)
    out = []
    for bb, s in dets:
        mapped = map_box_from_rotated(bb, bgr.shape, direction)
        if mapped:
            out.append((mapped, s))
    return out

# =============== Post-filters (for switch) ===============
def passes_switch_filters(bgr: np.ndarray, box: List[int],
                          min_area, min_aspect, min_edge_dens, min_hlines) -> bool:
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    area_ratio = (bw * bh) / float(W * H)
    if area_ratio < min_area:
        return False

    long_edge = max(bw, bh)
    short_edge = max(1, min(bw, bh))
    if (long_edge / short_edge) < min_aspect:
        return False

    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 20, 20)
    gray = cv2.equalizeHist(gray)

    edges = cv2.Canny(gray, 60, 150)
    ed = edges.mean() / 255.0
    if ed < min_edge_dens:
        return False

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=int(crop.shape[1] * 0.35), maxLineGap=8)
    hlines = 0
    if lines is not None:
        for l in lines:
            x1l, y1l, x2l, y2l = l[0]
            dy, dx = abs(y2l - y1l), abs(x2l - x1l)
            if dx > 0 and dy / dx < 0.2:
                hlines += 1
    if hlines < min_hlines:
        return False

    return True

def detect_switch_crops_with_pass(img_bgr: np.ndarray, sw_model,
                                  conf, scales,
                                  min_area, min_aspect, min_edge, min_hlines,
                                  audit_tag: str) -> List[Dict]:
    dets = _detect_switch_fullframe(img_bgr, sw_model, conf, scales)
    kept = []
    for bb, s in dets:
        if passes_switch_filters(img_bgr, bb, min_area, min_aspect, min_edge, min_hlines):
            crop = crop_with_margin(img_bgr, bb, margin=10, min_dim=128)
            if crop is not None:
                kept.append({"crop_bgr": crop, "conf": s, "bbox": tuple(bb)})
    return kept

def detect_switch_crops_improved(img_bgr: np.ndarray, sw_model, base_tag: str) -> List[Dict]:
    kept = detect_switch_crops_with_pass(
        img_bgr, sw_model,
        CONF_THRESHOLD, SCALES,
        MIN_AREA_RATIO_P1, MIN_LONG_ASPECT_P1, MIN_EDGE_DENSITY_P1, MIN_HORIZ_LINECOUNT_P1,
        audit_tag=f"{base_tag}_p1"
    )
    if kept: return kept

    kept = detect_switch_crops_with_pass(
        img_bgr, sw_model,
        FALLBACK_CONF, FALLBACK_SCALES,
        FALLBACK_MIN_AREA, FALLBACK_MIN_LONG_ASP, FALLBACK_MIN_EDGE_DENS, FALLBACK_MIN_HLINES,
        audit_tag=f"{base_tag}_p2"
    )
    if kept: return kept

    if USE_ROTATE_90_TTA:
        for direction in ("cw", "ccw"):
            dets = _detect_switch_rotated(img_bgr, sw_model, FALLBACK_CONF, FALLBACK_SCALES, direction)
            kept3 = []
            for bb, s in dets:
                if passes_switch_filters(img_bgr, bb, FALLBACK_MIN_AREA, FALLBACK_MIN_LONG_ASP,
                                         FALLBACK_MIN_EDGE_DENS, FALLBACK_MIN_HLINES):
                    crop = crop_with_margin(img_bgr, bb, margin=10, min_dim=128)
                    if crop is not None:
                        kept3.append({"crop_bgr": crop, "conf": s, "bbox": tuple(bb)})
            if kept3: return kept3
    return []

# =============== Grouping (unchanged similarity logic) ===============
orb = cv2.ORB_create(ORB_N_FEATURES)

def variant_hashes(pil_img: Image.Image):
    for flip in TEST_FLIPS:
        base = ImageOps.mirror(pil_img) if flip else pil_img
        for ang in range(0, 360, ROTATION_STEP):
            yield imagehash.phash(base.rotate(ang, expand=True))

def min_phash_delta(pil_img: Image.Image, rep_hashes: List[imagehash.ImageHash]) -> int:
    if not rep_hashes: return HASH_BITS
    best = 10**9
    for vh in variant_hashes(pil_img):
        for rh in rep_hashes:
            d = abs(vh - rh)
            if d < best: best = d
    return best

def orb_des_pil(pil_img: Image.Image) -> Optional[np.ndarray]:
    gray = np.array(pil_img.convert("L"))
    _, des = orb.detectAndCompute(gray, None)
    return des

def orb_good_count(d1: Optional[np.ndarray], d2: Optional[np.ndarray]) -> int:
    if d1 is None or d2 is None: return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1, d2)
    if not m: return 0
    return sum(1 for x in m if x.distance < ORB_DISTANCE_LIMIT)

def max_orb_good_vs_reps(d_new: Optional[np.ndarray], rep_orb: List[np.ndarray]) -> int:
    return 0 if d_new is None else max((orb_good_count(d_new, r) for r in rep_orb if r is not None), default=0)

def _ssim_pair(g1: np.ndarray, g2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5); mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return max(0.0, min(1.0, float(ssim_map.mean() + 1) / 2.0))

def ssim_256_gray(pil_a: Image.Image, pil_b: Image.Image) -> float:
    a = np.array(pil_a.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float32)
    b = np.array(pil_b.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float32)
    return _ssim_pair(a, b)

def hs_hist(pil_img: Image.Image):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_HIST_BINS, S_HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def hist_corr(a, b) -> float:
    c = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (c + 1) / 2.0))

def composite_score(group, pil_img: Image.Image) -> float:
    des_new = orb_des_pil(pil_img)
    orb_norm = min(1.0, max_orb_good_vs_reps(des_new, group.rep_orb) / 60.0)
    ssim_val = max((ssim_256_gray(pil_img, rp) for rp in group.rep_pils), default=0.0)
    hist_val = max((hist_corr(hs_hist(pil_img), hr) for hr in group.rep_hists), default=0.0)
    ph_d = min_phash_delta(pil_img, group.rep_hashes)
    ph_sim = 1.0 - min(1.0, ph_d / float(HASH_BITS))
    return W_ORB * orb_norm + W_SSIM * ssim_val + W_HIST * hist_val + W_PHASH * ph_sim

def find_best_group(groups, pil_img: Image.Image):
    if not groups: return -1, -1.0
    best_i, best_s = -1, -1.0
    for i, g in enumerate(groups):
        s = composite_score(g, pil_img)
        if s > best_s: best_s, best_i = s, i
    return best_i, best_s

# ======== In-memory group (no per-group folders) ========
class SwitchGroup:
    def __init__(self, gid: int):
        self.gid = f"group_{gid:04d}"
        self.rep_pils, self.rep_hashes, self.rep_orb, self.rep_hists = [], [], [], []
        self.members: List[Dict] = []  # { "src": str, "crop_bgr": np.ndarray }

    def add_member_crop(self, src_img_path: str, crop_bgr: np.ndarray):
        self.members.append({"src": src_img_path, "crop_bgr": crop_bgr})

    def add_rep(self, pil_img: Image.Image, max_reps=3):
        if len(self.rep_pils) >= max_reps: return
        self.rep_pils.append(pil_img.copy())
        self.rep_hashes.append(imagehash.phash(pil_img))
        self.rep_orb.append(orb_des_pil(pil_img))
        self.rep_hists.append(hs_hist(pil_img))

    def choose_best(self) -> Optional[np.ndarray]:
        if not self.members:
            return None
        best_idx, best_score = -1, -1.0
        for i, m in enumerate(self.members):
            crop = m["crop_bgr"]
            s = quality_score(crop)
            if s > best_score:
                best_score, best_idx = s, i
        return None if best_idx == -1 else self.members[best_idx]["crop_bgr"]

# =============== Quality scoring for "correct image" ===============
def quality_score(img_bgr: np.ndarray, is_patch_panel: bool = False) -> float:
    """Score image quality based on sharpness, aspect ratio, and other metrics.
    For patch panels, emphasizes horizontal lines and wide aspect ratio."""
    if img_bgr is None or img_bgr.size == 0:
        return -1.0
    H, W = img_bgr.shape[:2]
    aspect = max(W, H) / max(1.0, min(W, H))
    
    if is_patch_panel:
        # Patch panels should be wide and have many horizontal lines
        if aspect < 1.6:
            return -1.0  # Too narrow for a patch panel
        
        # Count horizontal lines
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 20, 20)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean()) / 255.0
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                               minLineLength=int(max(20, W * 0.25)), maxLineGap=8)
        hlines = 0
        if lines is not None:
            for l in lines:
                x1l, y1l, x2l, y2l = l[0]
                dy, dx = abs(y2l - y1l), abs(x2l - x1l)
                if dx > 0 and (dy / dx) < 0.25:
                    hlines += 1
        
        # Score components for patch panels
        line_score = min(1.0, hlines / 10.0)  # Normalize line count
        edge_score = min(1.0, edge_density * 100)
        aspect_score = min(1.0, (aspect - 1.6) / 2.0)
        size_score = (W * H) / 1e6
        
        # Weight the scores with emphasis on lines and aspect ratio
        return (0.35 * line_score) + (0.25 * edge_score) + (0.25 * aspect_score) + (0.15 * size_score)
    else:
        # Original scoring for switches
        aspect_penalty = 1.0 / (1.0 + max(0.0, aspect - 2.0))
        size_score = (W * H) / 1e6
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = sharp / 1000.0
        contrast = min(1.5, float(np.std(gray)) / 64.0)
        return (0.15 * aspect_penalty) + (0.40 * size_score) + (0.35 * sharp_score) + (0.10 * contrast)

# =============== Segmentation helpers (non-switch) ===============
def load_models(model_paths: Dict[str, str]):
    from ultralytics import YOLO
    return {k: YOLO(v) for k, v in model_paths.items()}

def crop_and_save(region, path):
    h, w = region.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        region = cv2.resize(region, (max(w, MIN_DIM), max(h, MIN_DIM)), interpolation=cv2.INTER_CUBIC)
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def run_other_segmentations(img_bgr: np.ndarray, img_name: str, models: Dict[str, any]):
    """
    Segment rack, cables, ports and save into flat class folders under OUTPUT_SEGMENTS_DIR.
    Switch is intentionally SKIPPED here (handled by grouping pipeline).
    """
    H, W = img_bgr.shape[:2]

    # Ensure class folders exist
    for cls in ["rack", "cables", "patch_panel", "connected_port", "empty_port", "switch"]:
        ensure_dir(os.path.join(OUTPUT_SEGMENTS_DIR, cls))

    # 1) RACK
    rack_model = models["rack"]
    r_conf = CONF_THRESHOLDS.get("rack", 0.2)
    r_res = rack_model(img_bgr, conf=r_conf, verbose=False)[0]
    rack_boxes = []
    for box, cls_id in zip(r_res.boxes.xyxy, r_res.boxes.cls):
        clsn = rack_model.names[int(cls_id)].lower()
        if clsn != "rack":
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
        x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
        crop = img_bgr[y1:y2, x1:x2]
        out_path = os.path.join(OUTPUT_SEGMENTS_DIR, "rack", f"{img_name}_rack_{len(rack_boxes)}.jpg")
        crop_and_save(crop, out_path)
        rack_boxes.append((x1, y1, x2, y2))

    # 2) CABLES
    cables_model = models["cables"]
    c_conf = CONF_THRESHOLDS.get("cables", 0.2)
    c_res = cables_model(img_bgr, conf=c_conf, verbose=False)[0]
    c_count = 0
    for box, cls_id in zip(c_res.boxes.xyxy, c_res.boxes.cls):
        clsn = cables_model.names[int(cls_id)].lower()
        if "cable" not in clsn:  # be tolerant to naming like 'cables'
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
        x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
        crop = img_bgr[y1:y2, x1:x2]
        out_path = os.path.join(OUTPUT_SEGMENTS_DIR, "cables", f"{img_name}_cables_{c_count}.jpg")
        crop_and_save(crop, out_path)
        c_count += 1

    # 3) PORTS (connected_port / empty_port) and PATCH_PANEL
    port_model = models["port"]
    p_conf = CONF_THRESHOLDS.get("port", 0.2)  # default if not present
    p_res = port_model(img_bgr, conf=p_conf, verbose=False)[0]
    p_count = {"connected_port": 0, "empty_port": 0, "patch_panel": 0}

    for box, cls_id in zip(p_res.boxes.xyxy, p_res.boxes.cls):
        raw = port_model.names[int(cls_id)].lower()
        # Normalize to expected final classes
        if raw in ("connected_port", "port_connected", "connected"):
            clsn = "connected_port"
        elif raw in ("empty_port", "port_empty", "empty"):
            clsn = "empty_port"
        elif raw in ("patch_panel", "patchpanel", "patch-panel", "patch"):
            clsn = "patch_panel"
        else:
            continue

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
        x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
        crop = img_bgr[y1:y2, x1:x2]
        idx = p_count[clsn]
        out_path = os.path.join(OUTPUT_SEGMENTS_DIR, clsn, f"{img_name}_{clsn}_{idx}.jpg")
        crop_and_save(crop, out_path)
        p_count[clsn] += 1

# =============== Main ===============
def main():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(OUTPUT_SEGMENTS_DIR)
    ensure_dir(os.path.join(OUTPUT_SEGMENTS_DIR, "switch"))  # switch now live under segments/

    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"⚠️ No images found in {INPUT_DIR}")
        return

    if SEED_INDEX < 0 or SEED_INDEX >= len(imgs):
        raise IndexError(f"SEED_INDEX {SEED_INDEX} out of range for {len(imgs)} images.")

    seed_path = imgs[SEED_INDEX]
    ordered = [seed_path] + imgs[:SEED_INDEX] + imgs[SEED_INDEX + 1:]

    print("Loading YOLO models ...")
    sw_model = load_yolo(MODEL_PATHS["switch"])  # for grouping
    other_models = load_models({
        "cables": MODEL_PATHS["cables"],
        "port":   MODEL_PATHS["port"],
        "rack":   MODEL_PATHS["rack"],
    })
    try:
        print(f"[main] switch model classes: {sw_model.names}")
    except Exception:
        pass
    try:
        for k, m in other_models.items():
            try: print(f"[main] loaded model '{k}' classes: {m.names}")
            except Exception: pass
    except Exception:
        pass

    all_switch_crops = []  # collect detected switches across all images (in-memory)
    all_patch_crops = []
    groups: List[SwitchGroup] = []  # will be populated in grouping pass after loop
    next_gid = 1
    MIN_GROUP_SCORE = 0.5  # threshold for grouping similarity
    rack_json = {}

    for idx, p in enumerate(ordered, start=1):
        img_name = os.path.splitext(os.path.basename(p))[0]
        img_key = os.path.basename(p)
        print(f"\nRack {idx}/{len(ordered)}: {img_key}")
        bgr = read_bgr(p)
        if bgr is None:
            print("  ⚠️ Could not read, skipping.")
            continue

        # Improved JSON format and restore saving segmented images
        rack_json[img_key] = {
            "filename": img_key,
            "rack_bbox": None,
            "switches": [],
            "patch_panels": [],
            "cables": [],
        }

        # ---- SWITCH & PATCH PANEL DETECTION (using switch model) ----
        sw_model_classes = model_classes(sw_model)
        sw_ids = set(class_ids_by_names(sw_model, SWITCH_CLASS_NAMES))
        patch_ids = set(class_ids_by_names(sw_model, PATCH_CLASS_NAMES))

        # Enhanced patch panel detection with quality scoring
        def detect_with_quality(bgr_img, conf_threshold, is_patch_panel=False):
            results = sw_model(bgr_img, conf=conf_threshold, iou=IOU_NMS, verbose=False,
                           imgsz=IMG_SIZE, agnostic_nms=True, max_det=MAX_DET)[0]
            try:
                print(f"[detect_with_quality] raw detections: {len(results.boxes.xyxy)} (conf={conf_threshold}, patch={is_patch_panel})")
            except Exception:
                pass
            kept = []
            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                cid = int(cls_id)
                bbox = list(map(int, box))
                crop = crop_with_margin(bgr_img, bbox, margin=10, min_dim=128)
                if crop is not None:
                    quality = quality_score(crop, is_patch_panel=is_patch_panel)
                    if quality > 0.4:  # Minimum quality threshold
                        kept.append((box, cls_id, conf, quality))
            try:
                print(f"[detect_with_quality] kept {len(kept)} boxes after quality filtering (patch={is_patch_panel})")
            except Exception:
                pass
            return kept

        # Run primary detection (try strict then fallback if none)
        sw_results = []
        patch_results = []

        # Try strict threshold first
        sw_detections = detect_with_quality(bgr, CONF_THRESHOLD, is_patch_panel=False)
        patch_detections = detect_with_quality(bgr, CONF_THRESHOLD, is_patch_panel=True)

        # If no detections, try fallback threshold
        if not sw_detections and not patch_detections:
            sw_detections = detect_with_quality(bgr, FALLBACK_CONF, is_patch_panel=False)
            patch_detections = detect_with_quality(bgr, FALLBACK_CONF, is_patch_panel=True)

        # Create a results list compatible with the rest of the code
        sw_results = []
        for det in sw_detections + patch_detections:
            box, cls_id, conf = det[:3]
            if isinstance(box, torch.Tensor):
                box = box.cpu()
            if isinstance(cls_id, torch.Tensor):
                cls_id = cls_id.cpu()
            if isinstance(conf, torch.Tensor):
                conf = float(conf.cpu())
            sw_results.append((box, cls_id, conf))

        # Record detections into rack_json and keep crops in memory for later grouping
        switches_in_image = []
        for i, (box, cls_id, conf) in enumerate(sw_results):
            cid = int(cls_id)
            bbox = list(map(int, box))
            name = sw_model_classes.get(cid, str(cid)).lower()
            crop = crop_with_margin(bgr, bbox, margin=10, min_dim=128)

            # Prepare JSON entry (segmented_filename left None because we don't save per-detection here)
            if cid in sw_ids:
                switch_entry = {
                    "id": f"switch_{len(rack_json[img_key]['switches'])}",
                    "bbox": bbox,
                    "confidence": float(conf),
                    "parent_image": img_key,
                    "segmented_filename": None,
                    "empty_ports": [],
                    "connected_ports": []
                }
                rack_json[img_key]["switches"].append(switch_entry)
                switches_in_image.append(switch_entry)

                # keep detected crop in memory for grouping later (only if crop available)
                if crop is not None:
                    all_switch_crops.append({
                        "src": img_key,
                        "crop_bgr": crop,
                        "bbox": bbox,
                        "conf": float(conf)
                    })

            elif cid in patch_ids:
                # Save per-detection patch panel crop and record it for grouping
                seg_dir = os.path.join(OUTPUT_SEGMENTS_DIR, "patch_panel")
                ensure_dir(seg_dir)
                seg_idx = len(rack_json[img_key]["patch_panels"]) if rack_json[img_key]["patch_panels"] is not None else 0
                seg_filename = f"{img_name}_patch_{seg_idx}.jpg"
                out_path = os.path.join(seg_dir, seg_filename)
                if crop is not None:
                    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rack_json[img_key]["patch_panels"].append({
                    "id": f"patch_panel_{seg_idx}",
                    "bbox": bbox,
                    "confidence": float(conf),
                    "parent_image": img_key,
                    "segmented_filename": seg_filename,
                    "empty_ports": [],
                    "connected_ports": []
                })
                # collect for grouping across images
                all_patch_crops.append({
                    "src": img_key,
                    "crop_bgr": crop,
                    "bbox": bbox,
                    "conf": float(conf),
                    "segmented_filename": seg_filename,
                })

        # END: switch detection for this image - grouping postponed until after all images processed

        # ---- OTHER SEGMENTATIONS (rack, cables, ports) ----
        H, W = bgr.shape[:2]
        # RACK
        rack_model = other_models["rack"]
        r_conf = CONF_THRESHOLDS.get("rack", 0.2)
        r_res = rack_model(bgr, conf=r_conf, verbose=False)[0]
        for box, cls_id in zip(r_res.boxes.xyxy, r_res.boxes.cls):
            clsn = rack_model.names[int(cls_id)].lower()
            if clsn == "rack":
                bbox = list(map(int, box))
                seg_filename = None
                crop = crop_with_margin(bgr, bbox, margin=10, min_dim=128)
                out_dir = os.path.join(OUTPUT_SEGMENTS_DIR, "rack")
                ensure_dir(out_dir)
                seg_filename = f"{img_name}_rack.jpg"
                out_path = os.path.join(out_dir, seg_filename)
                if crop is not None:
                    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rack_json[img_key]["rack_bbox"] = bbox
                rack_json[img_key]["rack_segmented_filename"] = seg_filename

        # CABLES
        cables_model = other_models["cables"]
        c_conf = CONF_THRESHOLDS.get("cables", 0.2)
        c_res = cables_model(bgr, conf=c_conf, verbose=False)[0]
        for i, (box, cls_id) in enumerate(zip(c_res.boxes.xyxy, c_res.boxes.cls)):
            clsn = cables_model.names[int(cls_id)].lower()
            if "cable" in clsn:
                bbox = list(map(int, box))
                seg_filename = f"{img_name}_cable_{i}.jpg"
                crop = crop_with_margin(bgr, bbox, margin=10, min_dim=128)
                out_dir = os.path.join(OUTPUT_SEGMENTS_DIR, "cables")
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, seg_filename)
                if crop is not None:
                    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rack_json[img_key]["cables"].append({"bbox": bbox, "parent_image": img_key, "segmented_filename": seg_filename})

        # PORTS (using port model)
        port_model = other_models["port"]
        p_conf = CONF_THRESHOLDS.get("port", 0.2)
        p_res = port_model(bgr, conf=p_conf, verbose=False)[0]
        ports_for_json = []
        for i, (box, cls_id) in enumerate(zip(p_res.boxes.xyxy, p_res.boxes.cls)):
            raw = port_model.names[int(cls_id)].lower()
            bbox = list(map(int, box))
            port_type = None
            if raw in ("connected_port", "port_connected", "connected"):
                port_type = "connected_port"
            elif raw in ("empty_port", "port_empty", "empty"):
                port_type = "empty_port"
            if port_type:
                crop = crop_with_margin(bgr, bbox, margin=10, min_dim=128)
                out_dir = os.path.join(OUTPUT_SEGMENTS_DIR, port_type)
                ensure_dir(out_dir)
                seg_filename = f"{img_name}_{port_type}_{i}.jpg"
                out_path = os.path.join(out_dir, seg_filename)
                if crop is not None:
                    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                ports_for_json.append({"type": port_type, "bbox": bbox, "parent_image": img_key, "segmented_filename": seg_filename})

        # Assign ports to switches and patch_panels by proximity
        for port in ports_for_json:
            bbox = port["bbox"]
            seg_filename = port["segmented_filename"]
            port_cx = (bbox[0] + bbox[2]) / 2
            port_cy = (bbox[1] + bbox[3]) / 2
            min_dist = float('inf')
            target = None
            # Check switches
            for sw in rack_json[img_key]["switches"]:
                sw_bbox = sw["bbox"]
                sw_cx = (sw_bbox[0] + sw_bbox[2]) / 2
                sw_cy = (sw_bbox[1] + sw_bbox[3]) / 2
                dist = ((sw_cx - port_cx) ** 2 + (sw_cy - port_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target = sw
            # Check patch_panels
            for pp in rack_json[img_key]["patch_panels"]:
                pp_bbox = pp["bbox"]
                pp_cx = (pp_bbox[0] + pp_bbox[2]) / 2
                pp_cy = (pp_bbox[1] + pp_bbox[3]) / 2
                dist = ((pp_cx - port_cx) ** 2 + (pp_cy - port_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target = pp
            # Assign port to closest component
            if target is not None:
                # Determine parent type and id
                if "id" in target:
                    if target["id"].startswith("switch_"):
                        parent_type = "switch"
                    elif target["id"].startswith("patch_panel_"):
                        parent_type = "patch_panel"
                    else:
                        parent_type = None
                    parent_id = target["id"]
                else:
                    parent_type = None
                    parent_id = None
                port_entry = {
                    "bbox": bbox,
                    "parent_image": img_key,
                    "segmented_filename": seg_filename,
                    "parent_type": parent_type,
                    "parent_id": parent_id
                }
                if port["type"] == "connected_port":
                    target["connected_ports"].append(port_entry)
                else:
                    target["empty_ports"].append(port_entry)

    # ---------------------------
    # Group all detected switch crops (run once after all images processed)
    # ---------------------------
    print(f"\nGrouping {len(all_switch_crops)} detected switch crops into similar groups...")
    groups = []
    next_gid = 1
    MIN_GROUP_SCORE = MIN_GROUP_SCORE if 'MIN_GROUP_SCORE' in globals() else 0.5

    for idx_sc, item in enumerate(all_switch_crops):
        crop = item["crop_bgr"]
        pil = bgr_to_pil(crop)
        # if no groups yet -> seed first group
        if not groups:
            g = SwitchGroup(next_gid)
            next_gid += 1
            g.add_member_crop(item["src"], crop)
            g.add_rep(pil)
            groups.append(g)
            continue

        # find best existing group
        best_i, best_s = find_best_group(groups, pil)

        # If similarity is high enough, add to that group; otherwise create new group
        if best_i != -1 and best_s >= MIN_GROUP_SCORE:
            groups[best_i].add_member_crop(item["src"], crop)
            if len(groups[best_i].rep_pils) < 3:
                groups[best_i].add_rep(pil)
        else:
            g = SwitchGroup(next_gid)
            next_gid += 1
            g.add_member_crop(item["src"], crop)
            g.add_rep(pil)
            groups.append(g)

    print(f"Grouping complete. {len(groups)} groups created.")

    # Save best image per group and all unique switches into segments/switch/
    switch_dir = os.path.join(OUTPUT_SEGMENTS_DIR, "switch")
    ensure_dir(switch_dir)
    switch_idx = 1
    for g in groups:
        if len(g.members) > 1:
            # Similar group: save only best
            best_crop = g.choose_best()
            if best_crop is not None:
                out_name = f"switch_{switch_idx:04d}.jpg"
                out_path = os.path.join(switch_dir, out_name)
                cv2.imwrite(out_path, best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                switch_idx += 1
        else:
            # Unique group: save the single member
            if g.members:
                crop = g.members[0]["crop_bgr"]
                out_name = f"switch_{switch_idx:04d}.jpg"
                out_path = os.path.join(switch_dir, out_name)
                cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                switch_idx += 1

    # --- Group patch panels collected across images and save best per group ---
    if all_patch_crops:
        print(f"\nGrouping {len(all_patch_crops)} detected patch panel crops into similar groups...")
        patch_groups: List[SwitchGroup] = []
        next_pg = 1
        MIN_GROUP_SCORE_PP = 0.6

        for item in all_patch_crops:
            crop = item.get("crop_bgr")
            if crop is None: continue
            pil = bgr_to_pil(crop)
            if not patch_groups:
                g = SwitchGroup(next_pg); next_pg += 1
                g.add_member_crop(item.get("src", ""), crop)
                g.add_rep(pil)
                patch_groups.append(g)
                continue
            best_i, best_s = find_best_group(patch_groups, pil)
            if best_i != -1 and best_s >= MIN_GROUP_SCORE_PP:
                patch_groups[best_i].add_member_crop(item.get("src", ""), crop)
                if len(patch_groups[best_i].rep_pils) < 3:
                    patch_groups[best_i].add_rep(pil)
            else:
                g = SwitchGroup(next_pg); next_pg += 1
                g.add_member_crop(item.get("src", ""), crop)
                g.add_rep(pil)
                patch_groups.append(g)

        # save grouped bests
        patch_dir = os.path.join(OUTPUT_SEGMENTS_DIR, "patch_panel")
        ensure_dir(patch_dir)
        pp_idx = 1
        for g in patch_groups:
            best = g.choose_best()
            if best is None: continue
            out_name = f"patch_panel_{pp_idx:04d}.jpg"
            out_path = os.path.join(patch_dir, out_name)
            cv2.imwrite(out_path, best, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            pp_idx += 1
        print(f"Patch-panel grouping complete. {len(patch_groups)} groups saved to {patch_dir}")

    # Save JSON output
    json_path = os.path.join(OUTPUT_ROOT, "rack_hierarchy.json")
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w") as f:
        json.dump(rack_json, f, indent=2)
    print(f"\nJSON hierarchy saved to {json_path}")

    # Summary
    print(f"\nDone.")
    print(f"   Switches → {os.path.join(OUTPUT_SEGMENTS_DIR, 'switch')}")
    print(f"   Other segments → {OUTPUT_SEGMENTS_DIR}")

if __name__ == "__main__":
    import shutil
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multiple rack images")
    parser.add_argument("input_dir", type=str, nargs='?', default=None, help="Input directory with images")
    parser.add_argument("output_dir", type=str, nargs='?', default=None, help="Output directory")
    args = parser.parse_args()
    
    # Override globals with command-line arguments if provided
    if args.input_dir:
        INPUT_DIR = args.input_dir
    if args.output_dir:
        # Output directly to provided directory (no extra nesting)
        OUTPUT_ROOT = args.output_dir
        OUTPUT_SEGMENTS_DIR = OUTPUT_ROOT
    
    print(f"[multii] INPUT_DIR={INPUT_DIR}")
    print(f"[multii] OUTPUT_ROOT={OUTPUT_ROOT}")
    
    # Clear only class folders (not the entire output directory to preserve Results folder)
    class_folders = ['rack', 'cables', 'patch_panel', 'switch', 'connected_port', 'empty_port']
    for cls in class_folders:
        cls_path = os.path.join(OUTPUT_ROOT, cls)
        if os.path.exists(cls_path):
            shutil.rmtree(cls_path)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    main()
