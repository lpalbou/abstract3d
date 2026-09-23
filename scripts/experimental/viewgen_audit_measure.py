#!/usr/bin/env python
"""Viewgen anatomy audit — feature-row measurement (adversarial audit, 2026-07-21).

Measures facial-feature ROW positions on bust images with SUBJECT-AGNOSTIC
signals only (no face detector): silhouette landmarks (top of head, shoulder
flare, profile leading-edge extrema), the dark-glasses band, and skin-fraction
row profiles. Works on four image kinds:

  photo  — the real front photo (RGBA matte, or RGB -> robust segmentation)
  rgba   — a generated view (RGBA matte from the pipeline)
  clay   — a clay render from the offscreen renderer (near-white background)

Reported landmarks (rows, image coordinates), each as a fraction of
  * head height (top-of-head -> chin), the anatomy-proportion axis, and
  * subject bbox height, the axis Hunyuan3D-2mv's per-view recenter actually
    equalizes (same feature, same row law).

Landmark definitions (all measured, none assumed):
  top       first row with silhouette width > 2% of bbox width
  shoulder  first row below the head where smoothed width > 1.45x head width
  glasses   photo/rgba: most prominent dark band (relative luma < 0.24) in the
            upper head; clay: topmost leading-edge protrusion shelf
  nose_tip  profile only: global leading-edge maximum below the glasses band
  nose_base profile: leading-edge minimum within 0.25 head-heights under the
            nose tip; front photo: horizontal-edge energy peak between the
            glasses band and the mouth region (flagged low-confidence)
  mouth     profile: next leading-edge maximum below nose_base (lip bump)
  chin      profile: next leading-edge maximum below mouth; front: center-
            column skin-fraction transition at the beard -> neck boundary
Front-photo nose_base/mouth are reported with confidence="low" and are NOT
used for acceptance math; view-vs-guide offsets always compare the SAME
extractor on both images so extractor bias cancels to first order.

Every measurement writes an annotated overlay PNG so a human (or the auditor)
can verify each detected row against the pixels before trusting the table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = Path("/tmp/viewgen_audit")
OVERLAYS = OUT / "overlays"


def load_subject(path: str, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """(rgba float 0..1 HxWx4, mask bool). kind: photo|rgba|clay."""
    from PIL import Image

    image = Image.open(path)
    if kind == "clay":
        from abstract3d.reference_generation import clay_silhouette

        mask = np.asarray(clay_silhouette(image))
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        rgba[:, :, 3] = mask.astype(np.float32)
        return rgba, mask
    if image.mode != "RGBA" or image.getchannel("A").getextrema()[0] == 255:
        from abstract3d.segmentation import remove_background_robust

        image = remove_background_robust(image)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return rgba, rgba[:, :, 3] > 0.5


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(x.astype(np.float64), sigma, mode="nearest")


def head_geometry(mask: np.ndarray) -> Dict[str, Any]:
    """top / shoulder rows + head width from the silhouette width profile."""
    height, width = mask.shape
    widths = mask.sum(axis=1).astype(np.float64)
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.nonzero(widths > 0.02 * (cols[-1] - cols[0] + 1))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    span = bottom - top + 1
    smoothed = _smooth(widths, max(2.0, span * 0.008))
    # Head width: median over the upper head band, before any shoulder flare.
    band0 = smoothed[top + int(0.05 * span): top + int(0.24 * span)]
    head_w0 = float(np.median(band0)) if len(band0) else float(smoothed[top])
    shoulder = bottom
    for row in range(top + int(0.20 * span), bottom):
        if smoothed[row] > 1.45 * head_w0:
            shoulder = row
            break
    refine = smoothed[top + int(0.04 * span): max(top + int(0.06 * span), shoulder - int(0.02 * span))]
    head_width = float(np.median(refine)) if len(refine) else head_w0
    return {
        "top": top, "bottom": bottom, "shoulder": int(shoulder),
        "head_width": head_width, "bbox_rows": (top, bottom),
        "bbox_cols": (int(cols[0]), int(cols[-1])),
    }


def face_direction(mask: np.ndarray, geo: Dict[str, Any]) -> str:
    """'left'|'right': the head-side whose edge profile varies more (features)."""
    top, shoulder = geo["top"], geo["shoulder"]
    band = mask[top + (shoulder - top) // 3: shoulder]
    lefts, rights = [], []
    for row in band:
        idx = np.nonzero(row)[0]
        if len(idx):
            lefts.append(idx[0])
            rights.append(idx[-1])
    lefts, rights = np.asarray(lefts, float), np.asarray(rights, float)
    if len(lefts) < 8:
        return "unknown"
    return "left" if np.std(np.diff(lefts)) > np.std(np.diff(rights)) else "right"


def leading_edge(mask: np.ndarray, side: str) -> np.ndarray:
    """Per-row frontmost x on the face side; NaN where the row is empty."""
    height = mask.shape[0]
    edge = np.full(height, np.nan)
    for row in range(height):
        idx = np.nonzero(mask[row])[0]
        if len(idx):
            edge[row] = idx[0] if side == "left" else idx[-1]
    return edge


def _relative_luma(rgba: np.ndarray) -> np.ndarray:
    return 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]


def dark_band(rgba: np.ndarray, mask: np.ndarray, geo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Glasses band: dark band with SKIN directly BELOW it (cheeks).

    Hair is equally dark on this subject class, but hair has hair or
    background below it, never a skin band. Scoring each candidate dark
    row by (dark fraction) x (skin fraction in the 0.06-0.20 head-heights
    below) separates the lens band from the crown plateau on every view
    kind that carries color. Clay renders use the geometric shelf instead.
    """
    top, shoulder = geo["top"], geo["shoulder"]
    head_h = shoulder - top
    luma = _relative_luma(rgba)
    dark = (luma < 0.24) & mask
    skin_rows = skin_fraction_rows(rgba, mask)
    skin_s = _smooth(skin_rows, max(1.5, head_h * 0.01))
    rows = slice(top + int(0.10 * head_h), top + int(0.75 * head_h))
    frac = np.zeros(rgba.shape[0])
    for row in range(rows.start, rows.stop):
        total = int(mask[row].sum())
        if total:
            frac[row] = dark[row].sum() / total
    if frac.max() <= 0.0:
        return None
    smoothed = _smooth(frac, max(1.5, head_h * 0.01))
    best, best_score = None, 0.0
    for row in range(rows.start, rows.stop):
        lo = row + int(0.06 * head_h)
        hi = row + int(0.20 * head_h)
        skin_below = float(skin_s[lo:hi].max()) if hi > lo and hi < len(skin_s) else 0.0
        score = smoothed[row] * skin_below
        if score > best_score:
            best, best_score = row, score
    if best is None or best_score <= 0.0:
        return None
    peak = smoothed[best]
    lo = best
    while lo > rows.start and smoothed[lo - 1] > 0.62 * peak:
        lo -= 1
    hi = best
    while hi < rows.stop - 1 and smoothed[hi + 1] > 0.62 * peak:
        hi += 1
    return {"row": int(best), "band": (int(lo), int(hi)),
            "peak_frac": round(float(peak), 3)}


def clay_glasses_shelf(mask: np.ndarray, side: str, geo: Dict[str, Any]) -> Optional[int]:
    """Clay glasses row: topmost strong leading-edge protrusion shelf."""
    top, shoulder = geo["top"], geo["shoulder"]
    head_h = shoulder - top
    edge = leading_edge(mask, side)
    band = slice(top + int(0.10 * head_h), top + int(0.60 * head_h))
    segment = edge[band]
    if np.all(np.isnan(segment)):
        return None
    protrusion = (np.nanmax(segment) - segment) if side == "right" else (segment - np.nanmin(segment))
    protrusion = -protrusion  # larger = more protruded toward the face side
    smoothed = _smooth(np.nan_to_num(protrusion, nan=-1e9), max(1.5, head_h * 0.01))
    threshold = np.nanpercentile(smoothed[np.isfinite(smoothed)], 92)
    for row in range(len(smoothed)):
        if smoothed[row] >= threshold:
            return int(band.start + row)
    return None


def profile_landmarks(mask: np.ndarray, side: str, geo: Dict[str, Any],
                      glasses_row: Optional[int]) -> Dict[str, Optional[int]]:
    """nose_tip / nose_base / mouth / chin from leading-edge extrema.

    Search windows are scaled by an ANTHROPOMETRIC head-height estimate
    anchored on two robust landmarks — top-of-head and the glasses band
    (eye line sits at ~0.45 of head height on adults) — because the
    shoulder flare row varies with shirt cut and pose and inflates any
    shoulder-derived scale (measured: 768px frame, shoulder-scale put the
    lip-bump search 97px wide and locked onto the collar).
    """
    top, shoulder = geo["top"], geo["shoulder"]
    if glasses_row is None:
        return {"nose_tip": None, "nose_base": None, "mouth": None, "chin": None}
    est_head = max(20.0, (glasses_row - top) / 0.45)
    edge = leading_edge(mask, side)
    signed = edge if side == "right" else -edge  # larger = more protruded
    smoothed = _smooth(np.nan_to_num(signed, nan=-1e9), max(1.8, est_head * 0.008))
    floor = min(mask.shape[0] - 1, int(top + 1.25 * est_head))

    def window_max(lo: int, hi: int) -> Optional[int]:
        lo, hi = int(max(top, lo)), int(min(floor, hi))
        if hi <= lo:
            return None
        segment = smoothed[lo:hi]
        if not np.isfinite(segment).any():
            return None
        return int(lo + np.argmax(segment))

    def window_min(lo: int, hi: int) -> Optional[int]:
        lo, hi = int(max(top, lo)), int(min(floor, hi))
        if hi <= lo:
            return None
        segment = smoothed[lo:hi].copy()
        segment[~np.isfinite(segment)] = 1e9
        if not (segment < 1e8).any():
            return None
        return int(lo + np.argmin(segment))

    nose_tip = window_max(glasses_row + int(0.02 * est_head),
                          glasses_row + int(0.35 * est_head))
    if nose_tip is None:
        return {"nose_tip": None, "nose_base": None, "mouth": None, "chin": None}
    nose_base = window_min(nose_tip + 2, nose_tip + int(0.14 * est_head))
    mouth = window_max(nose_base + 2, nose_base + int(0.14 * est_head)) if nose_base else None
    chin = None
    if mouth is not None:
        lip_valley = window_min(mouth + 2, mouth + int(0.10 * est_head))
        if lip_valley is not None:
            chin = window_max(lip_valley + 1, lip_valley + int(0.20 * est_head))
    return {"nose_tip": nose_tip, "nose_base": nose_base, "mouth": mouth, "chin": chin}


def width_minimum_chin(mask: np.ndarray, geo: Dict[str, Any],
                       glasses_row: Optional[int]) -> Optional[int]:
    """Front/back chin proxy: the NECK width minimum between face and
    shoulders. On a front bust the silhouette narrows from the jaw down
    to the neck and flares to the shoulders; the minimum row is the
    face-bottom / neck transition (measured on the front photo: min at
    row 580 with the beard bottom at ~578 visually). Beard-inclusive by
    construction, exactly like the profile-contour chin bump."""
    top, shoulder = geo["top"], geo["shoulder"]
    head_h = shoulder - top
    start = (glasses_row + int(0.25 * head_h)) if glasses_row else (top + int(0.55 * head_h))
    widths = _smooth(mask.sum(axis=1).astype(np.float64), max(2.0, head_h * 0.012))
    stop = shoulder - max(2, int(0.02 * head_h))
    if stop <= start:
        return None
    return int(start + np.argmin(widths[start:stop]))


def skin_fraction_rows(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row skin-like fraction (LAB bands) over the mask foreground."""
    from skimage import color as skcolor

    lab = skcolor.rgb2lab(rgba[:, :, :3])
    skin = (
        mask
        & (lab[:, :, 0] > 35) & (lab[:, :, 0] < 88)
        & (lab[:, :, 1] > 4) & (lab[:, :, 1] < 36)
        & (lab[:, :, 2] > 6) & (lab[:, :, 2] < 42)
    )
    height = mask.shape[0]
    out = np.zeros(height)
    for row in range(height):
        total = int(mask[row].sum())
        if total:
            out[row] = skin[row].sum() / total
    return out


def front_landmarks(rgba: np.ndarray, mask: np.ndarray, geo: Dict[str, Any],
                    glasses: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Front-photo nose_base/mouth (low confidence) + chin (skin transition)."""
    top, shoulder = geo["top"], geo["shoulder"]
    head_h = shoulder - top
    c0, c1 = geo["bbox_cols"]
    center = slice(c0 + int(0.38 * (c1 - c0)), c0 + int(0.62 * (c1 - c0)))
    luma = _relative_luma(rgba)
    edges = np.abs(np.diff(luma, axis=0))
    glasses_bottom = glasses["band"][1] if glasses else top + int(0.45 * head_h)

    chin = width_minimum_chin(mask, geo, glasses["row"] if glasses else None)

    # nose base / mouth: horizontal-edge energy peaks in the center columns
    # between the glasses band and the chin (nostril line, then lip line).
    # LOW CONFIDENCE on a bearded subject; excluded from acceptance math.
    nose_base = mouth = None
    if chin is not None and chin > glasses_bottom + 8:
        band = slice(glasses_bottom + int(0.05 * head_h), chin - 3)
        energy = np.zeros(mask.shape[0])
        for row in range(band.start, min(band.stop, edges.shape[0])):
            row_mask = mask[row, center]
            if row_mask.any():
                energy[row] = float(edges[row, center][row_mask].mean())
        smoothed_e = _smooth(energy, max(1.5, head_h * 0.008))
        peaks = [row for row in range(band.start + 1, band.stop - 1)
                 if smoothed_e[row] >= smoothed_e[row - 1]
                 and smoothed_e[row] >= smoothed_e[row + 1]]
        peaks.sort(key=lambda row: -smoothed_e[row])
        strongest = sorted(peaks[:2])
        if strongest:
            nose_base = int(strongest[0])
            mouth = int(strongest[1]) if len(strongest) > 1 else None
    return {"nose_base": nose_base, "mouth": mouth, "chin": chin, "confidence": "low"}


def measure(path: str, kind: str, label: str, facing: Optional[str] = None) -> Dict[str, Any]:
    """Full landmark record for one image + annotated overlay PNG."""
    from PIL import Image, ImageDraw

    rgba, mask = load_subject(path, kind)
    geo = head_geometry(mask)
    detected_side = face_direction(mask, geo)
    # Facing comes from the LABEL (the clay camera convention: side_left
    # renders at azimuth +90 with the face pointing image-left); the
    # heuristic detector is recorded as a diagnostic only — dark curly
    # hair defeats edge-variance heuristics, and a wrong facing would be
    # obvious in the overlay while silently poisoning every landmark.
    label_facing = {"side_left": "left", "side_right": "right"}.get(label)
    side = facing or label_facing or detected_side
    record: Dict[str, Any] = {
        "path": path, "kind": kind, "label": label,
        "facing_used": side, "facing_detected": detected_side,
        "top": geo["top"], "shoulder": geo["shoulder"], "bottom": geo["bottom"],
        "head_width_px": round(geo["head_width"], 1),
    }
    glasses = None
    if kind == "clay":
        shelf = clay_glasses_shelf(mask, side, geo) if side in ("left", "right") else None
        glasses = {"row": shelf, "band": (shelf, shelf), "peak_frac": None} if shelf else None
    else:
        glasses = dark_band(rgba, mask, geo)
    record["glasses_row"] = glasses["row"] if glasses else None

    if label == "back":
        landmarks: Dict[str, Optional[int]] = {"nose_tip": None, "nose_base": None,
                                               "mouth": None, "chin": None}
    elif label == "front":
        front = front_landmarks(rgba, mask, geo, glasses)
        landmarks = {"nose_tip": None, "nose_base": front["nose_base"],
                     "mouth": front["mouth"], "chin": front["chin"]}
        record["front_confidence"] = front["confidence"]
    else:
        landmarks = profile_landmarks(mask, side, geo, record["glasses_row"])
    record.update(landmarks)

    # Nose protrusion (x axis): leading-edge advance of the nose tip over
    # the nose base, in pixels — the "bulbous nose" number. Positive =
    # protrudes toward the face side. Profiles only.
    if landmarks.get("nose_tip") is not None and landmarks.get("nose_base") is not None \
            and side in ("left", "right"):
        edge = leading_edge(mask, side)
        tip_x, base_x = edge[landmarks["nose_tip"]], edge[landmarks["nose_base"]]
        if np.isfinite(tip_x) and np.isfinite(base_x):
            advance = (base_x - tip_x) if side == "left" else (tip_x - base_x)
            record["nose_protrusion_px"] = round(float(advance), 1)

    # Normalizations.
    top = geo["top"]
    chin = landmarks.get("chin")
    head_h = (chin - top) if chin else None
    span = geo["bottom"] - top
    fractions: Dict[str, Dict[str, Optional[float]]] = {}
    for name in ("glasses_row", "nose_tip", "nose_base", "mouth", "chin", "shoulder"):
        row = record.get(name)
        fractions[name] = {
            "of_head": round((row - top) / head_h, 4) if (row is not None and head_h) else None,
            "of_subject": round((row - top) / span, 4) if row is not None else None,
        }
    record["fractions"] = fractions
    record["head_height_px"] = head_h
    record["subject_span_px"] = span

    OVERLAYS.mkdir(parents=True, exist_ok=True)
    display = Image.fromarray((rgba * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(display)
    palette = {"glasses_row": (255, 80, 80), "nose_tip": (255, 170, 0),
               "nose_base": (0, 200, 90), "mouth": (60, 130, 255),
               "chin": (200, 60, 220), "shoulder": (120, 120, 120), "top": (0, 0, 0)}
    for name in ("top", "glasses_row", "nose_tip", "nose_base", "mouth", "chin", "shoulder"):
        row = geo["top"] if name == "top" else record.get(name)
        if row is None:
            continue
        draw.line([(0, row), (display.width, row)], fill=palette[name], width=2)
        draw.text((6, max(0, row - 14)), name.replace("_row", ""), fill=palette[name])
    overlay_path = OVERLAYS / f"{Path(path).stem}__{label}.png"
    display.save(overlay_path)
    record["overlay"] = str(overlay_path)
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path,
                        help="JSON list of {path, kind, label, facing?}")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    results = [measure(row["path"], row["kind"], row["label"], row.get("facing"))
               for row in spec]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(json.dumps([{k: r[k] for k in ("label", "kind", "top", "glasses_row",
                                          "nose_tip", "nose_base", "mouth", "chin",
                                          "shoulder", "facing_used")}
                      for r in results], indent=1))


if __name__ == "__main__":
    main()
