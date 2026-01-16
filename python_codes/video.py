

# app_video_grouping_single_segments_fast_with_manifest.py
import os
import shutil
import json
from typing import List, Tuple, Dict, Optional
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash

import easyocr
import argparse
from ultralytics import YOLO

########################################
# ===========================
# --- Config (speed-tuned) ---
# ===========================
# ===========================
RESULTS_FOLDER  = r"D:\HeroPageBuilder\HeroPageBuilder\segmented_output"

# Your trained models
MODEL_PATHS = {
    "cables": r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\cable_best.pt",
    "port":   r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\port_count.pt",
    "rack":   r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\rack_best.pt",
    "switch": r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\switch_patch.pt",  # used by SWITCH grouping pipeline only
    "port":   r"D:\HeroPageBuilder\HeroPageBuilder\Trained_Models\port_best.pt",
}
# Video frame selection
TARGET_FRAME_CANDIDATES = 12
TOPK_SWITCH_FRAMES      = 5
USE_OCR_FOR_SCORING     = False

# Switch detection/grouping knobs
CONF_THRESHOLD = 0.18
IOU_NMS        = 0.3
MAX_DET        = 100
IMG_SIZE       = 1024

# Tiling
TILE_W, TILE_H = 1200, 1200
TILE_OVERLAP   = 0.15

# TTA / scales
USE_FLIP       = True
SCALES         = [1.0, 0.75, 1.25, 1.5]

# Class names
SWITCH_CLASS_NAMES = {"switch", "network_switch", "Switch", "network", "equipment"}
PATCH_CLASS_NAMES  = {"patch", "patchpanel", "patch_panel", "Patch_Panel", "Patch-Panel"}

# Post-filters
MIN_AREA_RATIO_P1      = 0.00005
MIN_LONG_ASPECT_P1     = 1.0
MIN_EDGE_DENSITY_P1    = 0.002
MIN_HORIZ_LINECOUNT_P1 = 0

# Fallbacks
FALLBACK_CONF          = 0.05
FALLBACK_SCALES        = [1.0, 1.25, 1.5]
FALLBACK_MIN_AREA      = 0.0001
FALLBACK_MIN_LONG_ASP  = 1.0
FALLBACK_MIN_EDGE_DENS = 0.003
FALLBACK_MIN_HLINES    = 0

USE_ROTATE_90_TTA      = False

# Grouping weights
HASH_BITS          = 32
ROTATION_STEP      = 15
TEST_FLIPS         = [False]
ORB_N_FEATURES     = 500
ORB_DISTANCE_LIMIT = 60
SSIM_SIZE          = 128
H_HIST_BINS        = 25
S_HIST_BINS        = 30
W_ORB, W_SSIM, W_HIST, W_PHASH = 0.50, 0.25, 0.15, 0.10

FINAL_CLASSES = ['cables', 'rack', 'patch_panel', 'switch', 'connected_port', 'empty_port']
CONF_THRESHOLDS = {"rack": 0.6, "switch": 0.44, "cables": 0.18}
MARGIN = 10
MIN_DIM = 128

import torch

# Auto-select device: prefer CUDA when available for much faster inference
YOLO_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if YOLO_DEVICE == "cpu":
    print("Warning: YOLO_DEVICE set to CPU. Inference will be slower. Install CUDA-enabled PyTorch and drivers for GPU acceleration.")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Define supported video file extensions
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.mpeg', '.mpg', '.webm', '.m4v')

def resolve_input_video(path_or_dir: str) -> str:
    """If path_or_dir is a directory, find a video file inside and return its path.
    If it's a file, return it unchanged. Raises SystemExit on error.
    """
    if os.path.isdir(path_or_dir):
        entries = [f for f in os.listdir(path_or_dir) if f and f.lower().endswith(VIDEO_EXTS)]
        if not entries:
            print(f"ERROR: No video files found in directory: {path_or_dir}")
            raise SystemExit(1)
        if len(entries) > 1:
            print(f"WARNING: Multiple video files found in '{path_or_dir}'. Using the first one: {entries[0]}")
        chosen = os.path.join(path_or_dir, entries[0])
        return os.path.abspath(chosen)
    if os.path.isfile(path_or_dir):
        return os.path.abspath(path_or_dir)
    print(f"ERROR: Video path does not exist: {path_or_dir}")
    raise SystemExit(1)

# EasyOCR init
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def get_ocr_score(frame: np.ndarray) -> float:
    if not USE_OCR_FOR_SCORING: return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = ocr_reader.readtext(gray)
    text = " ".join([res[1] for res in results])
    return float(len(text.strip()))

def pick_top_frames(video_path: str, out_dir: str, num_candidates: int = TARGET_FRAME_CANDIDATES,
                    final_k: int = TOPK_SWITCH_FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release(); return []
    step = max(total_frames // num_candidates, 1)
    candidates = []
    frame_idx = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            sharp = get_sharpness(frame)
            score = sharp / 100.0
            candidates.append((score, frame.copy(), frame_idx))
        frame_idx += 1
    cap.release()
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[: max(final_k * 5, final_k)]
    if USE_OCR_FOR_SCORING:
        rescored = []
        for sc, fr, fi in candidates:
            ocr_score = get_ocr_score(fr)
            combined = (2.0 * ocr_score) + sc
            rescored.append((combined, fr, fi))
        candidates = sorted(rescored, key=lambda x: x[0], reverse=True)[:final_k]
    else:
        candidates = candidates[:final_k]
    print(f"[pick_top_frames] sampled {len(candidates)} frames (final_k={final_k})")
    return [frame for (_, frame, _) in candidates]

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def clamp_box(x1, y1, x2, y2, W, H):
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(W - 1, int(round(x2))), min(H - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1: return None
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

def iter_tiles(W, H, tw, th, overlap):
    sx = max(1, int(tw * (1 - overlap))); sy = max(1, int(th * (1 - overlap)))
    for y in range(0, max(1, H - th + 1), sy):
        for x in range(0, max(1, W - tw + 1), sx):
            yield x, y, min(x + tw, W), min(y + th, H)
    if (H % sy) != 0:
        y = max(0, H - th)
        for x in range(0, max(1, W - tw + 1), sx): yield x, y, min(x + tw, W), min(y + th, H)
    if (W % sx) != 0:
        x = max(0, W - tw)
        for y in range(0, max(1, H - th + 1), sy): yield x, y, min(x + tw, W), min(y + th, H)
    yield max(0, W - tw), max(0, H - th), W, H

def soft_nms_merge(boxes, scores, iou_thr=IOU_NMS, score_thr=0.001, sigma=0.5):
    boxes = np.array(boxes, dtype=float); scores = np.array(scores, dtype=float)
    keep = []; idxs = list(range(len(scores)))
    while idxs:
        i = int(np.argmax(scores[idxs])); i = idxs.pop(i)
        if scores[i] < score_thr: break
        keep.append(i); suppressed = []
        for j in idxs:
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            iw, ih = max(0, x2 - x1), max(0, y2 - y1)
            inter = iw * ih
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou = inter / (a + b - inter + 1e-9)
            if iou > 0: scores[j] *= np.exp(-(iou ** 2) / sigma)
            if scores[j] < score_thr: suppressed.append(j)
        for j in suppressed:
            if j in idxs: idxs.remove(j)
    return keep

def weighted_boxes_fusion(boxes, scores):
    if not boxes: return [], []
    boxes = np.array(boxes, dtype=float); scores = np.array(scores, dtype=float)
    used = np.zeros(len(boxes), dtype=bool); fused_boxes, fused_scores = [], []
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
            if iou >= IOU_NMS: cluster.append(j); used[j] = True
        ws = scores[cluster]; ws = ws / (ws.sum() + 1e-9)
        bb = boxes[cluster]; fused = (bb * ws[:, None]).sum(axis=0)
        fused_boxes.append(fused.tolist()); fused_scores.append(float(scores[cluster].max()))
    return fused_boxes, fused_scores

def rotate_img(img, direction: str):
    if direction == "cw": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if direction == "ccw": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def map_box_from_rotated(box, orig_shape, direction: str):
    H, W = orig_shape[:2]; x1, y1, x2, y2 = map(int, box)
    if direction == "cw":
        nx1, ny1 = W - y2, x1; nx2, ny2 = W - y1, x2
    elif direction == "ccw":
        nx1, ny1 = y1, H - x2; nx2, ny2 = y2, H - x1
    else:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    if nx1 > nx2: nx1, nx2 = nx2, nx1
    if ny1 > ny2: ny1, ny2 = ny2, ny1
    return [nx1, ny1, nx2, ny2]

def model_classes(model):
    names = model.names
    if isinstance(names, dict): return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}

def class_ids_by_names(model, wanted: set):
    names = model_classes(model); wl = {w.lower() for w in wanted}
    return [cid for cid, name in names.items() if name.lower() in wl]

# ---------------------------
# Detection functions (patch panels and switches kept separate)
# ---------------------------
def _detect_patch_panels_fullframe(bgr: np.ndarray, model, conf) -> List[Tuple[List[int], float]]:
    """Detect only patch panels in a frame. Uses enhanced detection with additional filtering for patch panel characteristics."""
    H, W = bgr.shape[:2]
    patch_ids = set(class_ids_by_names(model, PATCH_CLASS_NAMES))
    try:
        # Stage 1: Full frame detection
        rs = model.predict(source=[bgr], conf=conf, iou=IOU_NMS,
                          imgsz=min(IMG_SIZE, max(W, H)), verbose=False, max_det=MAX_DET, device=YOLO_DEVICE)
        if rs and len(rs) > 0:
            r = rs[0]
            boxes, scores = [], []
            for box, cls_id, scf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                cls_id = int(cls_id)
                # Only accept patch panel classes
                if patch_ids and cls_id not in patch_ids: continue
                bb = clamp_box(*box.tolist(), W, H)
                if not bb: continue
                
                # Extra validation for patch panel characteristics
                crop = crop_with_margin(bgr, bb, margin=MARGIN, min_dim=MIN_DIM)
                if crop is None: continue
                
                # Score the patch panel based on characteristics
                q_score = quality_score(crop, is_patch_panel=True)
                if q_score <= 0: continue  # Skip if it doesn't meet patch panel criteria
                
                # Use a weighted combination of detection confidence and quality score
                combined_score = 0.6 * float(scf) + 0.4 * q_score
                boxes.append(bb)
                scores.append(combined_score)
            
            if boxes:
                keep = soft_nms_merge(boxes, scores, iou_thr=IOU_NMS, sigma=0.5)
                pr_boxes = [boxes[i] for i in keep]
                pr_scores = [scores[i] for i in keep]
                fused_boxes, fused_scores = weighted_boxes_fusion(pr_boxes, pr_scores)
                out = []
                for b, s in zip(fused_boxes, fused_scores):
                    bb = clamp_box(*b, W, H)
                    if bb: out.append((bb, s))
                if out: return out
    except Exception as e:
        print("[_detect_patch_panels_fullframe] detection failed:", e)
    return []

def _detect_switches_fullframe(bgr: np.ndarray, sw_model, conf, scales) -> List[Tuple[List[int], float]]:
    """Detect only switches in a frame. Keeps switch detection pure by filtering for switch classes only."""
    H, W = bgr.shape[:2]
    sw_ids = set(class_ids_by_names(sw_model, SWITCH_CLASS_NAMES))
    
    # Stage 1: full-frame detection
    try:
        rs = sw_model.predict(source=[bgr], conf=conf, iou=IOU_NMS,
                              imgsz=min(IMG_SIZE, max(W, H)), verbose=False, max_det=MAX_DET, device=YOLO_DEVICE)
        if rs and len(rs) > 0:
            r = rs[0]
            try:
                print(f"[_detect_switches_fullframe] full-frame raw dets: {len(r.boxes.xyxy)}")
            except Exception:
                pass
            boxes, scores = [], []
            for box, cls_id, scf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                cls_id = int(cls_id)
                # Only accept pure switch classes - skip anything that's not a switch
                if not sw_ids or cls_id not in sw_ids: continue
                bb = clamp_box(*box.tolist(), W, H)
                if bb: boxes.append(bb); scores.append(float(scf))
            if boxes:
                print(f"[_detect_switches_fullframe] full-frame filtered dets: {len(boxes)} (scores: {scores[:5]})")
                keep = soft_nms_merge(boxes, scores, iou_thr=IOU_NMS, sigma=0.5)
                pr_boxes = [boxes[i] for i in keep]; pr_scores = [scores[i] for i in keep]
                fused_boxes, fused_scores = weighted_boxes_fusion(pr_boxes, pr_scores)
                out = []
                for b, s in zip(fused_boxes, fused_scores):
                    bb = clamp_box(*b, W, H)
                    if bb: out.append((bb, s))
                if out: return out
    except Exception as e:
        print("[_detect_switches_fullframe] full-frame detect failed (continuing to tiles):", e)

    # Stage 2: tiled fallback (also only looking for switches)
    all_boxes, all_scores = [], []
    rois = [(0, 0, W, H)]
    for (rx1, ry1, rx2, ry2) in rois:
        roi = bgr[ry1:ry2, rx1:rx2]; RH, RW = roi.shape[:2]
        for x1, y1, x2, y2 in iter_tiles(RW, RH, TILE_W, TILE_H, TILE_OVERLAP):
            tile = roi[y1:y2, x1:x2]
            for sc in scales:
                tile_resized = tile if sc == 1.0 else cv2.resize(tile, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                try:
                    rs = sw_model.predict(source=[tile_resized], conf=conf, iou=IOU_NMS,
                                          imgsz=IMG_SIZE, verbose=False, max_det=MAX_DET, device=YOLO_DEVICE)
                except Exception as e:
                    print("[_detect_switches_fullframe] tile predict failed:", e)
                    continue
                
                if not rs: continue
                r = rs[0]
                try:
                    print(f"[_detect_switches_fullframe] tile raw dets: {len(r.boxes.xyxy)}")
                except Exception:
                    pass
                for box, cls_id, scf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    cls_id = int(cls_id)
                    # Only accept pure switch classes in tiles too
                    if not sw_ids or cls_id not in sw_ids: continue
                    bx = np.array(box.tolist(), dtype=float)
                    if sc != 1.0: bx /= sc
                    bx[0] += x1; bx[2] += x1; bx[1] += y1; bx[3] += y1
                    bx[0] += rx1; bx[2] += rx1; bx[1] += ry1; bx[3] += ry1
                    bb = clamp_box(bx[0], bx[1], bx[2], bx[3], W, H)
                    if bb is None: continue
                    all_boxes.append(bb)
                    all_scores.append(float(scf))

    # Merge tile detections
    if not all_boxes: return []
    keep = soft_nms_merge(all_boxes, all_scores, iou_thr=IOU_NMS, sigma=0.5)
    pr_boxes = [all_boxes[i] for i in keep]
    pr_scores = [all_scores[i] for i in keep]
    fused_boxes, fused_scores = weighted_boxes_fusion(pr_boxes, pr_scores)
    
    # Final output
    out = []
    for b, s in zip(fused_boxes, fused_scores):
        bb = clamp_box(*b, W, H)
        if bb: out.append((bb, s))
    return out

def passes_switch_filters(bgr: np.ndarray, box: List[int], min_area, min_aspect, min_edge_dens, min_hlines) -> bool:
    H, W = bgr.shape[:2]; x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    area_ratio = (bw * bh) / float(W * H)
    if area_ratio < min_area: return False
    long_edge = max(bw, bh); short_edge = max(1, min(bw, bh))
    if (long_edge / short_edge) < min_aspect: return False
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0: return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 20, 20); gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 60, 150); ed = edges.mean() / 255.0
    if ed < min_edge_dens: return False
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=int(crop.shape[1] * 0.35), maxLineGap=8)
    hlines = 0
    if lines is not None:
        for l in lines:
            x1l, y1l, x2l, y2l = l[0]
            dy, dx = abs(y2l - y1l), abs(x2l - x1l)
            if dx > 0 and dy / dx < 0.2: hlines += 1
    if hlines < min_hlines: return False
    return True


def is_probably_patch_panel(bgr: np.ndarray) -> bool:
    """Heuristic to decide if a crop likely contains a patch panel.
    Uses horizontal line count, edge density and aspect ratio.
    """
    try:
        if bgr is None or bgr.size == 0:
            return False
        H, W = bgr.shape[:2]
        bw, bh = W, H
        area = bw * bh
        if area <= 0:
            return False
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        # patch panels are typically wide and have many horizontal rows
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 20, 20)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean()) / 255.0
        # count horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=int(max(20, bw * 0.25)), maxLineGap=8)
        hlines = 0
        if lines is not None:
            for l in lines:
                x1l, y1l, x2l, y2l = l[0]
                dy, dx = abs(y2l - y1l), abs(x2l - x1l)
                if dx > 0 and (dy / dx) < 0.25:
                    hlines += 1
        # heuristics thresholds (tunable)
        if aspect >= 1.6 and hlines >= 3 and edge_density > 0.0015:
            return True
        return False
    except Exception:
        return False

def detect_switch_crops_with_pass(img_bgr: np.ndarray, sw_model, conf, scales, min_area, min_aspect, min_edge, min_hlines) -> List[Dict]:
    dets = _detect_switches_fullframe(img_bgr, sw_model, conf, scales)
    kept = []
    for bb, s in dets:
        if passes_switch_filters(img_bgr, bb, min_area, min_aspect, min_edge, min_hlines):
            crop = crop_with_margin(img_bgr, bb, margin=10, min_dim=MIN_DIM)
            if crop is not None:
                kept.append({"crop_bgr": crop, "conf": s, "bbox": tuple(bb)})
    return kept

def detect_switch_crops_improved(img_bgr: np.ndarray, sw_model) -> List[Dict]:
    kept = detect_switch_crops_with_pass(img_bgr, sw_model, CONF_THRESHOLD, SCALES,
                                         MIN_AREA_RATIO_P1, MIN_LONG_ASPECT_P1, MIN_EDGE_DENSITY_P1, MIN_HORIZ_LINECOUNT_P1)
    if kept: return kept
    kept = detect_switch_crops_with_pass(img_bgr, sw_model, FALLBACK_CONF, FALLBACK_SCALES,
                                         FALLBACK_MIN_AREA, FALLBACK_MIN_LONG_ASP, FALLBACK_MIN_EDGE_DENS, FALLBACK_MIN_HLINES)
    if kept: return kept
    if USE_ROTATE_90_TTA:
        for direction in ("cw", "ccw"):
            rot = rotate_img(img_bgr, direction)
            dets = _detect_switches_fullframe(rot, sw_model, FALLBACK_CONF, FALLBACK_SCALES)
            kept3 = []
            for bb, s in dets:
                mapped = map_box_from_rotated(bb, img_bgr.shape, direction)
                if mapped and passes_switch_filters(img_bgr, mapped, FALLBACK_MIN_AREA, FALLBACK_MIN_LONG_ASP,
                                                    FALLBACK_MIN_EDGE_DENS, FALLBACK_MIN_HLINES):
                    crop = crop_with_margin(img_bgr, mapped, margin=10, min_dim=MIN_DIM)
                    if crop is not None:
                        kept3.append({"crop_bgr": crop, "conf": s, "bbox": tuple(mapped)})
            if kept3: return kept3
    return []

# ---------------------------
# Grouping helpers & class
# ---------------------------
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
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX); return hist

def hist_corr(a, b) -> float:
    c = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (c + 1) / 2.0))

def composite_score(group, pil_img: Image.Image) -> float:
    des_new = orb_des_pil(pil_img)
    orb_norm = min(1.0, max_orb_good_vs_reps(des_new, group.rep_orb) / 30.0)
    ssim_val = max((ssim_256_gray(pil_img, rp) for rp in group.rep_pils), default=0.0)
    hist_val = max((hist_corr(hs_hist(pil_img), hr) for hr in group.rep_hists), default=0.0)
    ph_d = min_phash_delta(pil_img, group.rep_hashes)
    ph_sim = 1.0 - min(1.0, ph_d / float(HASH_BITS))
    score = W_ORB * orb_norm + W_SSIM * ssim_val + W_HIST * hist_val + W_PHASH * ph_sim
    return score

def find_best_group(groups, pil_img: Image.Image):
    if not groups: return -1, -1.0
    best_i, best_s = -1, -1.0
    for i, g in enumerate(groups):
        s = composite_score(g, pil_img)
        if s > best_s: best_s, best_i = s, i
    if best_s < 0.6: return -1, best_s
    return best_i, best_s

class SwitchGroup:
    def __init__(self, gid: int):
        self.gid = f"group_{gid:04d}"
        self.rep_pils, self.rep_hashes, self.rep_orb, self.rep_hists = [], [], [], []
        # members: list of dicts { src: str, crop_bgr: np.ndarray, conf: float, bbox: tuple }
        self.members: List[Dict] = []

    def add_member_crop(self, src_img_path: str, crop_bgr: np.ndarray, conf: float = 0.0, bbox: Tuple[int, int, int, int] = ()):
        self.members.append({"src": src_img_path, "crop_bgr": crop_bgr, "conf": float(conf), "bbox": tuple(bbox)})

    def add_rep(self, pil_img: Image.Image, max_reps=3):
        if len(self.rep_pils) >= max_reps: return
        self.rep_pils.append(pil_img.copy())
        self.rep_hashes.append(imagehash.phash(pil_img))
        self.rep_orb.append(orb_des_pil(pil_img))
        self.rep_hists.append(hs_hist(pil_img))

    def choose_best(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if not self.members:
            return None, None
        best_idx, best_score = -1, -1.0
        for i, m in enumerate(self.members):
            crop = m["crop_bgr"]
            s = quality_score(crop)
            if s > best_score:
                best_score, best_idx = s, i
        if best_idx == -1:
            return None, None
        return self.members[best_idx]["crop_bgr"], best_idx

def quality_score(img_bgr: np.ndarray, is_patch_panel: bool = False) -> float:
    """Score image quality based on sharpness, aspect ratio, and other metrics.
    For patch panels, emphasizes horizontal lines and wide aspect ratio."""
    if img_bgr is None or img_bgr.size == 0: return -1.0
    H, W = img_bgr.shape[:2]
    aspect = max(W, H) / max(1.0, min(W, H))
    
    if is_patch_panel:
        # Patch panels should be wide and have many horizontal lines
        if aspect < 1.6: return -1.0  # Too narrow for a patch panel
        
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

# ---------------------------
# Other segmentation helpers (unchanged)
# ---------------------------
def load_models(model_paths: Dict[str, str]):
    models = {}
    for k, p in model_paths.items():
        try:
            m = YOLO(p)
            if YOLO_DEVICE:
                try: m.model.to(YOLO_DEVICE)
                except Exception: pass
            models[k] = m; print(f"[load_models] loaded {k} from {p}")
        except Exception as e:
            print(f"[load_models] failed to load {k} from {p}: {e}")
    return models

def crop_and_save(region, path):
    h, w = region.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        region = cv2.resize(region, (max(w, MIN_DIM), max(h, MIN_DIM)), interpolation=cv2.INTER_CUBIC)
    ensure_dir(os.path.dirname(path)); cv2.imwrite(path, region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def run_other_segmentations(img_bgr: np.ndarray, out_segments_dir: str, img_stub: str, models: Dict[str, YOLO]):
    ensure_dir(out_segments_dir)
    for cls in ["rack", "cables", "patch_panel", "connected_port", "empty_port"]:
        ensure_dir(os.path.join(out_segments_dir, cls))
    H, W = img_bgr.shape[:2]

    # We'll collect structured entries for each class to mirror single.py's output format
    entries = {"rack": [], "switch": [], "patch_panel": [], "cables": [], "connected_port": [], "empty_port": []}

    # RACK
    rack_model = models.get("rack")
    if rack_model:
        r_conf = CONF_THRESHOLDS.get("rack", 0.2)
        try:
            r_res = rack_model(img_bgr, conf=r_conf, verbose=False)[0]; rack_idx = 0
            for box, cls_id, conf in zip(r_res.boxes.xyxy, r_res.boxes.cls, r_res.boxes.conf):
                clsn = rack_model.names[int(cls_id)].lower()
                if clsn != "rack": continue
                x1, y1, x2, y2 = map(int, box); x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
                crop = img_bgr[y1:y2, x1:x2]
                seg_filename = f"{img_stub}_rack_{rack_idx}.jpg"
                out_path = os.path.join(out_segments_dir, "rack", seg_filename)
                crop_and_save(crop, out_path)
                entries["rack"].append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "parent_image": img_stub, "segmented_filename": seg_filename})
                rack_idx += 1
        except Exception as e:
            print("[run_other_segmentations] rack predict failed:", e)

    # CABLES
    cables_model = models.get("cables")
    if cables_model:
        c_conf = CONF_THRESHOLDS.get("cables", 0.2)
        try:
            c_res = cables_model(img_bgr, conf=c_conf, verbose=False)[0]; cable_idx = 0
            for box, cls_id, conf in zip(c_res.boxes.xyxy, c_res.boxes.cls, c_res.boxes.conf):
                clsn = cables_model.names[int(cls_id)].lower()
                if "cable" not in clsn: continue
                x1, y1, x2, y2 = map(int, box); x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
                crop = img_bgr[y1:y2, x1:x2]
                seg_filename = f"{img_stub}_cables_{cable_idx}.jpg"
                out_path = os.path.join(out_segments_dir, "cables", seg_filename)
                crop_and_save(crop, out_path)
                entries["cables"].append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "parent_image": img_stub, "segmented_filename": seg_filename})
                cable_idx += 1
        except Exception as e:
            print("[run_other_segmentations] cables predict failed:", e)

    # PORTS & PATCH_PANEL
    port_model = models.get("port")
    if port_model:
        p_conf = CONF_THRESHOLDS.get("port", 0.2)
        try:
            p_res = port_model(img_bgr, conf=p_conf, verbose=False)[0]; p_count = {"connected_port": 0, "empty_port": 0, "patch_panel": 0}
            for box, cls_id, conf in zip(p_res.boxes.xyxy, p_res.boxes.cls, p_res.boxes.conf):
                raw = port_model.names[int(cls_id)].lower()
                if raw in ("connected_port", "port_connected", "connected"): clsn = "connected_port"
                elif raw in ("empty_port", "port_empty", "empty"): clsn = "empty_port"
                elif raw in ("patch_panel", "patchpanel", "patch-panel", "patch"): clsn = "patch_panel"
                else: continue
                x1, y1, x2, y2 = map(int, box); x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
                crop = img_bgr[y1:y2, x1:x2]; idx = p_count[clsn]
                seg_filename = f"{img_stub}_{clsn}_{idx}.jpg"
                out_path = os.path.join(out_segments_dir, clsn, seg_filename)
                crop_and_save(crop, out_path)
                entries[clsn].append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "parent_image": img_stub, "segmented_filename": seg_filename})
                p_count[clsn] += 1
        except Exception as e:
            print("[run_other_segmentations] port predict failed:", e)

    # If port model didn't detect patch panels, try to use the switch model (if provided)
    # Note: some trained switch models include patch_panel as a class; use it as a fallback
    sw_model = models.get("switch")
    try:
        if sw_model:
            # basic full-frame predict for patch panel class names
            p_res = sw_model(img_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
            pp_idx = 0
            for box, cls_id, conf in zip(p_res.boxes.xyxy, p_res.boxes.cls, p_res.boxes.conf):
                clsn = str(sw_model.names[int(cls_id)]).lower()
                if clsn in (name.lower() for name in PATCH_CLASS_NAMES):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                    x2, y2 = min(W, x2 + MARGIN), min(H, y2 + MARGIN)
                    crop = img_bgr[y1:y2, x1:x2]
                    seg_filename = f"{img_stub}_patch_panel_{pp_idx}.jpg"
                    out_path = os.path.join(out_segments_dir, "patch_panel", seg_filename)
                    crop_and_save(crop, out_path)
                    entries["patch_panel"].append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "parent_image": img_stub, "segmented_filename": seg_filename})
                    pp_idx += 1
    except Exception as e:
        # non-fatal; keep whatever entries we have
        print("[run_other_segmentations] switch-model-based patch_panel predict failed:", e)

    return entries

# ---------------------------
# Main: detection -> grouping -> save best/unique -> save other segments -> manifest
# ---------------------------
if __name__ == "__main__":
    # CLI: allow choosing mode and output directory
    ap = argparse.ArgumentParser(description="Detect and group switches and/or patch panels from a video")
    ap.add_argument("--detect", choices=["patch", "switch", "both"], default="both",
                    help="Which targets to detect: 'patch' (patch panels only), 'switch' (switches only), or 'both'")
    ap.add_argument("video_path", type=str, help="Path to the input video file")
    ap.add_argument("output_dir", type=str, nargs='?', default=None, help="Output directory (optional)")
    args = ap.parse_args()
    
    # Use provided output_dir or fall back to RESULTS_FOLDER
    if args.output_dir:
        # Output directly to provided directory (no extra nesting)
        output_root = args.output_dir
    else:
        output_root = os.path.join(RESULTS_FOLDER)
    
    segments_dir = os.path.join(output_root)
    switches_out = os.path.join(segments_dir, "switch")
    # create only the switches folder (no subfolders)
    ensure_dir(switches_out)
    ensure_dir(segments_dir)

    # Defer extracting frames until we've parsed CLI args and validated the video path
    print("[main] loading models...")
    sw_model = None
    try:
        sw_model = YOLO(MODEL_PATHS["switch"])
        if YOLO_DEVICE:
            try: sw_model.model.to(YOLO_DEVICE)
            except Exception: pass
        print("[main] loaded switch model")
        try:
            print(f"[main] switch model classes: {sw_model.names}")
        except Exception:
            pass
    except Exception as e:
        print("[main] failed to load switch model:", e)
    
    # Validate video path
    video_path = resolve_input_video(args.video_path)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    detect_mode = args.detect

    other_models = load_models({"rack": MODEL_PATHS["rack"], "cables": MODEL_PATHS["cables"], "port": MODEL_PATHS["port"], "switch": MODEL_PATHS["switch"]})
    try:
        for k, m in other_models.items():
            try: print(f"[main] loaded model '{k}' classes: {m.names}")
            except Exception: pass
    except Exception:
        pass

    # We'll collect detected switches (crops) in-memory, then group and write only best/unique outputs.
    all_switch_crops = []  # collect detected switches across all images (in-memory)
    # also prepare to collect patch panel crops and track the best one
    all_patch_crops = []
    best_patch_panel = {"crop": None, "score": -1.0, "info": None}  # Track best patch panel

    manifest = {
        "video": os.path.basename(video_path),
        "switch_groups": [],
        "patch_panel_groups": [],
        "other_segments": {}
    }

    # Structured JSON matching single.py format
    structured_json = {
        "filename": os.path.splitext(os.path.basename(video_path))[0],
        "rack_bbox": None,
        "rack_segmented_filename": None,
        "switches": [],
        "patch_panels": [],
        "cables": [],
        "connected_ports": [],
        "empty_ports": []
    }

    # CLI: allow choosing mode: patch-only, switch-only, or both
    ap = argparse.ArgumentParser(description="Detect and group switches and/or patch panels from a video")
    ap.add_argument("--detect", choices=["patch", "switch", "both"], default="both",
                    help="Which targets to detect: 'patch' (patch panels only), 'switch' (switches only), or 'both'")
    ap.add_argument("video_path", type=str, help="Path to the input video file")
    args = ap.parse_args()
    
    # Validate video path
    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    detect_mode = args.detect
    video_path = args.video_path

    print(f"Processing video: {video_path}")
    print(f"Output directory: {segments_dir}")
    print(f"Detect mode: {detect_mode}")

    # Extract top frames from the provided video path
    top_frames = pick_top_frames(video_path, out_dir=output_root, num_candidates=TARGET_FRAME_CANDIDATES, final_k=TOPK_SWITCH_FRAMES)
    if not top_frames:
        print("Could not extract usable frames from the video.")
        sys.exit(1)

    # Detection flows based on detect_mode:
    # - switch: detect and group only switches
    # - patch: detect and group only patch panels
    # - both: detect and group both types independently (no mixing between classes)
    
    # First detect switches if requested
    if sw_model is not None and detect_mode in ("switch", "both"):
        print("[main] Starting switch detection...")
        # Detect switches across frames using pure switch detection
        for fi, frame_bgr in enumerate(top_frames, start=1):
            print(f"[main] Frame {fi}: detect switches...")
            switches = detect_switch_crops_improved(frame_bgr, sw_model)
            print(f"[main] Frame {fi}: found {len(switches)} switch candidates")
            for si, sw in enumerate(switches):
                crop = sw["crop_bgr"]
                all_switch_crops.append({
                    "src": f"frame_{fi:02d}",
                    "crop_bgr": crop,
                    "bbox": sw.get("bbox", ()),
                    "conf": float(sw.get("conf", 0.0))
                })
        print(f"[main] total detected switch crops: {len(all_switch_crops)}")

        # When running in 'both' mode, we now keep detections in their original classes
        # (no reclassification/moving between pools) to ensure pure classification

        # --- Grouping: create in-memory groups from all_switch_crops ---
        groups: List[SwitchGroup] = []
        next_gid = 1
        MIN_GROUP_SCORE = 0.8  # threshold for grouping similarity

        for idx_sc, item in enumerate(all_switch_crops):
            crop = item["crop_bgr"]
            pil = bgr_to_pil(crop)
            # if no groups yet -> seed first group
            if not groups:
                g = SwitchGroup(next_gid)
                next_gid += 1
                g.add_member_crop(item["src"], crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                g.add_rep(pil)
                groups.append(g)
                continue

            # find best existing group
            best_i, best_s = find_best_group(groups, pil)

            # If similarity is high enough, add to that group; otherwise create new group
            if best_i != -1 and best_s >= MIN_GROUP_SCORE:
                groups[best_i].add_member_crop(item["src"], crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                if len(groups[best_i].rep_pils) < 3:
                    groups[best_i].add_rep(pil)
            else:
                g = SwitchGroup(next_gid)
                next_gid += 1
                g.add_member_crop(item["src"], crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                g.add_rep(pil)
                groups.append(g)

        print(f"[main] grouping complete. total groups: {len(groups)}")

        # Save best image per group and all unique switches into segments/switch/
        switch_dir = switches_out
        ensure_dir(switch_dir)
        switch_idx = 1

        for g in groups:
            group_entry = {
                "gid": g.gid,
                "member_count": len(g.members),
                "members": [
                    {"src": m["src"], "conf": float(m.get("conf", 0.0)), "bbox": m.get("bbox", ())}
                    for m in g.members
                ],
                "saved": None
            }

            best_crop, best_idx = g.choose_best()
            if best_crop is not None and best_idx is not None:
                out_name = f"switch_{switch_idx:04d}.jpg"
                out_path = os.path.join(switch_dir, out_name)
                cv2.imwrite(out_path, best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rel = os.path.relpath(out_path, output_root).replace("\\", "/")
                saved_info = {
                    "filename": out_name,
                    "relpath": rel,
                    "member_index": int(best_idx),
                    "member_conf": float(g.members[best_idx].get("conf", 0.0)),
                    "type": "best" if len(g.members) > 1 else "unique"
                }
                group_entry["saved"] = saved_info
                switch_idx += 1
                # Also add to structured JSON switches list (use bbox/conf from the member)
                try:
                    member = g.members[best_idx]
                    mbbox = list(member.get("bbox", ())) if member.get("bbox") else None
                    mconf = float(member.get("conf", 0.0))
                    structured_json["switches"].append({
                        "bbox": mbbox,
                        "confidence": mconf,
                        "parent_image": member.get("src", ""),
                        "segmented_filename": out_name
                    })
                except Exception:
                    pass
            else:
                group_entry["saved"] = None

            manifest["switch_groups"].append(group_entry)
    else:
        if sw_model is None:
            print("[main] switch model not loaded; skipping switch detection & grouping")
        else:
            print(f"[main] skipping switch detection because detect_mode='{detect_mode}'")

    # 4) OTHER segmentation for each top frame (rack, cables, ports)
    # Collect patch panel crops across frames for grouping
    for fi, frame_bgr in enumerate(top_frames, start=1):
        img_stub = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{fi:02d}"
        entries = run_other_segmentations(frame_bgr, segments_dir, img_stub, other_models)
        # Merge entries into structured_json
        # Rack: pick first detected rack if not already set
        racks = entries.get("rack", [])
        if racks and structured_json.get("rack_bbox") is None:
            first = racks[0]
            structured_json["rack_bbox"] = first.get("bbox")
            structured_json["rack_segmented_filename"] = first.get("segmented_filename")
        # Cables
        for e in entries.get("cables", []):
            structured_json["cables"].append(e)
        # Patch panels: append entries but also collect crops for grouping (only when collecting patches)
        if detect_mode in ("patch", "both"):
            for e in entries.get("patch_panel", []):
                structured_json["patch_panels"].append(e)
                try:
                    seg_file = e.get("segmented_filename")
                    seg_path = os.path.join(segments_dir, "patch_panel", seg_file)
                    if os.path.exists(seg_path):
                        crop = cv2.imread(seg_path)
                        if crop is not None:
                            all_patch_crops.append({
                                "src": img_stub,
                                "crop_bgr": crop,
                                "bbox": e.get("bbox"),
                                "conf": float(e.get("confidence", 0.0)),
                                "segmented_filename": seg_file,
                            })
                except Exception as exc:
                    print("[main] failed to load patch panel segmented file for grouping:", exc)
        else:
            # If not collecting patches, skip adding patch entries so switch-only mode remains pure
            pass
        # Connected ports
        for e in entries.get("connected_port", []):
            structured_json["connected_ports"].append(e)
        # Empty ports
        for e in entries.get("empty_port", []):
            structured_json["empty_ports"].append(e)

    # --- Grouping for patch panels collected across frames ---
    # Use same grouping strategy as switches but for patch panel detections
    if detect_mode in ("patch", "both") and sw_model is not None:
        print("[main] Starting patch panel detection...")
        # First collect direct patch panel detections from the model
        for fi, frame_bgr in enumerate(top_frames, start=1):
            patches = _detect_patch_panels_fullframe(frame_bgr, sw_model, CONF_THRESHOLD)
            for bbox, conf in patches:
                crop = crop_with_margin(frame_bgr, bbox, margin=MARGIN, min_dim=MIN_DIM)
                if crop is not None:
                    all_patch_crops.append({
                        "src": f"frame_{fi:02d}",
                        "crop_bgr": crop,
                        "bbox": bbox,
                        "conf": float(conf)
                    })
        print(f"[main] Found {len(all_patch_crops)} patch panel candidates")

    # Find the single best patch panel among all detections
    if all_patch_crops:
        print("[main] Selecting best patch panel...")
        patch_dir = os.path.join(segments_dir, "patch_panel")
        ensure_dir(patch_dir)
        structured_json["patch_panels"] = []  # Clear existing entries
        # Group patch panels and select best from each group
        print("[main] Grouping patch panel detections...")
        patch_groups: List[SwitchGroup] = []  # reuse SwitchGroup for consistent logic
        next_pg = 1
        MIN_GROUP_SCORE_PP = 0.8  # Threshold for group similarity

        # First, group similar patch panels
        for item in all_patch_crops:
                crop = item["crop_bgr"]
                pil = bgr_to_pil(crop)
                if not patch_groups:
                    g = SwitchGroup(next_pg); next_pg += 1
                    g.add_member_crop(item.get("src", ""), crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                    g.add_rep(pil)
                    patch_groups.append(g)
                    continue
                best_i, best_s = find_best_group(patch_groups, pil)
                if best_i != -1 and best_s >= MIN_GROUP_SCORE_PP:
                    patch_groups[best_i].add_member_crop(item.get("src", ""), crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                    if len(patch_groups[best_i].rep_pils) < 3:
                        patch_groups[best_i].add_rep(pil)
                else:
                    g = SwitchGroup(next_pg); next_pg += 1
                    g.add_member_crop(item.get("src", ""), crop, conf=item.get("conf", 0.0), bbox=item.get("bbox", ()))
                    g.add_rep(pil)
                    patch_groups.append(g)

            # Save best patch panel from each group
        print(f"[main] Found {len(patch_groups)} distinct patch panel groups")
        for group_idx, g in enumerate(patch_groups, 1):
                # Score all members in the group
                best_member_score = -1.0
                best_member_idx = -1
                best_member_crop = None
            
                for idx, member in enumerate(g.members):
                    crop = member["crop_bgr"]
                    # Calculate comprehensive score
                    quality = quality_score(crop, is_patch_panel=True)
                    conf = float(member.get("conf", 0.0))
                    # Weighted combination of quality and confidence
                    final_score = 0.6 * quality + 0.4 * conf
                
                    if final_score > best_member_score:
                        best_member_score = final_score
                        best_member_idx = idx
                        best_member_crop = crop

                if best_member_crop is not None and best_member_idx >= 0:
                    # Save the best member from this group
                    out_name = f"best_group_{group_idx:02d}.jpg"
                    out_path = os.path.join(patch_dir, out_name)
                    cv2.imwrite(out_path, best_member_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    # Add to manifest
                    group_entry = {
                        "gid": g.gid,
                        "member_count": len(g.members),
                        "members": [{"src": m["src"], "conf": float(m.get("conf", 0.0)), 
                                   "bbox": m.get("bbox", ())} for m in g.members],
                        "saved": {
                            "filename": out_name,
                            "relpath": os.path.relpath(out_path, output_root).replace("\\", "/"),
                            "member_index": best_member_idx,
                            "member_conf": float(g.members[best_member_idx].get("conf", 0.0)),
                            "type": "best_in_group",
                            "quality_score": best_member_score
                        }
                    }
                    manifest["patch_panel_groups"].append(group_entry)

                    # Add to structured JSON
                    best_member = g.members[best_member_idx]
                    structured_json["patch_panels"].append({
                        "bbox": list(best_member.get("bbox", ())) if best_member.get("bbox") else None,
                        "confidence": float(best_member.get("conf", 0.0)),
                        "parent_image": best_member.get("src", ""),
                        "segmented_filename": out_name,
                        "quality_score": best_member_score,
                        "group_id": g.gid
                    })

        # Gather other_segments file lists for manifest
    other_segments = {}
    if os.path.isdir(segments_dir):
        for cls in FINAL_CLASSES:
            cls_dir = os.path.join(segments_dir, cls)
            if not os.path.isdir(cls_dir):
                other_segments[cls] = []
                continue
            files = sorted([os.path.relpath(os.path.join(cls_dir, f), output_root).replace("\\", "/")
                            for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
            other_segments[cls] = files

    manifest["other_segments"] = other_segments

    # Save manifest
    manifest_path = os.path.join(output_root, "manifest.json")
    ensure_dir(os.path.dirname(manifest_path))
    with open(manifest_path, "w", encoding="utf-8") as jf:
        json.dump(manifest, jf, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    # Save structured JSON (single.py-compatible format)
    structured_path = os.path.join(output_root, f"{structured_json.get('filename')}_structured.json")
    try:
        with open(structured_path, "w", encoding="utf-8") as sf:
            json.dump(structured_json, sf, indent=2)
        print(f"Structured JSON saved to: {structured_path}")
    except Exception as e:
        print(f"Failed to save structured JSON: {e}")

    # Summary
    print(f"\nDone.")
    print(f"   Switches → {os.path.join(segments_dir, 'switch')}")
    print(f"   Other segments → {segments_dir}")
