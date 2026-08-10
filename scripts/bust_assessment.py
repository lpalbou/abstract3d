#!/usr/bin/env python
"""Bust reconstruction assessment harness: renders + objective metrics.

Scores a bust-generation attempt so an iteration loop can measure progress
instead of eyeballing thumbnails (a double-mouth geometry defect once shipped
because verification used 256px thumbnails). Produces:

  1. High-res clay (geometry-only) and textured renders at 8 azimuths,
     elevation 10, plus a face close-up crop pair (clay + textured) taken
     from dedicated 2x-resolution front renders.
  2. Geometry metrics (watertight, body count, euler number, dihedral-angle
     RMS, vertex/face counts) and a DOUBLE-FEATURE detector: the
     autocorrelation of the horizontal-edge-energy row profile of the face
     region on the front clay render. A duplicated horizontal feature (double
     mouth / eye line) shows as a strong non-zero-lag autocorrelation peak at
     the duplication distance; a healthy face decays monotonically there.
  3. Texture metrics: coverage stats from bundle metadata.json (when
     present) and per-view mean saturation / luminance variance of the
     subject region from the textured renders (detects the "unpainted dark
     back" failure).
  4. <output-dir>/scorecard.json + scorecard.md.

Exit codes: 0 on success (assessment tool, not a gate — a suspect mesh still
exits 0); 2 when inputs are missing.

Duplication-detector calibration (2026-07-20, moderngl, size=1024, elev=10):
  known-bad   out/laurent-bust-redo/e2_2mv_explicit_refs (double mouth):
              peak ratio 0.071 at lag 46 px = 6.1% of subject height
  known-clean abstractcore/untracked/webcam3d_demo/reconstruction.glb:
              strongest local peak ratio 0.034
  -> default threshold 0.05 sits between the two measurements.

Usage:
  python scripts/bust_assessment.py \
      --bundle out/laurent-bust-redo/e2_2mv_explicit_refs \
      --output-dir out/assessment/e2_2mv_explicit_refs

  python scripts/bust_assessment.py \
      --glb /path/to/reconstruction.glb --output-dir out/assessment/recon
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEP_HINT = (
    "bust_assessment.py needs numpy + Pillow + trimesh (and moderngl or "
    'matplotlib for renders). Install them with: pip install "abstract3d[mesh]"'
)

try:
    import numpy as np
    from PIL import Image
except ImportError as e:  # pragma: no cover - exercised only on broken envs
    raise ImportError(_DEP_HINT) from e


# ---------------------------------------------------------------------------
# Spec constants (see module docstring for the calibration numbers).
# ---------------------------------------------------------------------------

FULL_VIEW_AZIMUTHS: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
DEFAULT_ELEVATION_DEG = 10.0
DEFAULT_RENDER_SIZE = 1024
FACE_REGION_FRACTION = 0.55  # face region = top 55% of the subject's rows
# Non-zero-lag peak search window, as fractions of the subject height.
DUPLICATION_LAG_WINDOW = (0.03, 0.15)
# Calibrated between the known-bad (0.071) and known-cleaner (0.034) meshes.
DEFAULT_DUPLICATION_PEAK_RATIO_THRESHOLD = 0.05
# Foreground = any channel differing from the border-median background by
# more than this (uint8 scale). Works for both renderer backgrounds
# (moderngl 0.95/0.95/0.93, matplotlib #f3f2ee) without hardcoding either.
BACKGROUND_DIFF_THRESHOLD = 14.0
# "Unpainted dark back" verdict thresholds (luminance in 0..1). Healthy
# measured backs: mean 0.143/var 0.0117 (e2_2mv bundle) and mean 0.231/var
# 0.0084 (webcam3d reconstruction) — both clear these with margin.
BACK_UNPAINTED_MAX_MEAN_LUMINANCE = 0.10
BACK_UNPAINTED_MAX_LUMINANCE_VARIANCE = 0.0025


class InputError(RuntimeError):
    """Missing/invalid inputs — the only condition that exits non-zero."""


# ---------------------------------------------------------------------------
# Subject segmentation (shared by the detector, crops, and texture stats).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectBBox:
    row0: int
    row1: int  # exclusive
    col0: int
    col1: int  # exclusive

    @property
    def height(self) -> int:
        return self.row1 - self.row0

    @property
    def width(self) -> int:
        return self.col1 - self.col0


def subject_mask(image: Image.Image) -> "np.ndarray":
    """Boolean foreground mask via background differencing.

    The background color is estimated from the border pixels (median), so the
    mask is renderer-agnostic instead of hardcoding one clear color.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    return np.abs(rgb - background).max(axis=2) > BACKGROUND_DIFF_THRESHOLD


def _longest_true_run(flags: "np.ndarray") -> Optional[Tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return None
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def subject_bbox(mask: "np.ndarray") -> SubjectBBox:
    """Largest contiguous row/column span of the foreground mask.

    Longest-run (instead of min/max of any foreground pixel) keeps stray
    antialiasing specks or floating debris from inflating the subject box.
    """
    height, width = mask.shape
    min_row_px = max(2, int(round(0.002 * width)))
    rows = _longest_true_run(mask.sum(axis=1) >= min_row_px)
    if rows is None:
        raise ValueError(
            "No subject found in render (foreground mask is empty). "
            "The mesh may be empty or rendered entirely out of frame."
        )
    row0, row1 = rows
    min_col_px = max(2, int(round(0.002 * (row1 - row0))))
    cols = _longest_true_run(mask[row0:row1].sum(axis=0) >= min_col_px)
    if cols is None:  # pragma: no cover - rows found implies columns exist
        raise ValueError("No subject columns found in render.")
    col0, col1 = cols
    return SubjectBBox(row0=row0, row1=row1, col0=col0, col1=col1)


# ---------------------------------------------------------------------------
# Double-feature detector (general-purpose horizontal-feature duplication).
# ---------------------------------------------------------------------------


def horizontal_edge_row_profile(
    image: Image.Image,
    mask: "np.ndarray",
    bbox: SubjectBBox,
    *,
    face_fraction: float = FACE_REGION_FRACTION,
) -> Tuple["np.ndarray", Tuple[int, int]]:
    """Per-row mean horizontal-edge energy over the face region.

    Horizontal features (mouth line, eye line, brow) produce luminance
    gradients along the vertical axis, so the energy is |d(gray)/dy| averaged
    over subject pixels of each row (mean, not sum, so wide rows such as
    shoulders do not dominate narrow face rows).
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    grad_y = np.abs(np.diff(gray, axis=0))  # row r = boundary between r and r+1
    edge_mask = mask[:-1] & mask[1:]
    column_selector = np.zeros(mask.shape[1], dtype=bool)
    column_selector[bbox.col0 : bbox.col1] = True

    face_row_end = bbox.row0 + int(round(face_fraction * bbox.height))
    face_row_end = min(face_row_end, grad_y.shape[0])
    selected = edge_mask[bbox.row0 : face_row_end] & column_selector[None, :]
    counts = selected.sum(axis=1)
    sums = (grad_y[bbox.row0 : face_row_end] * selected).sum(axis=1)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return profile.astype(np.float64), (bbox.row0, face_row_end)


def normalized_autocorrelation(profile: "np.ndarray") -> "np.ndarray":
    """Autocorrelation of the mean-removed profile, normalized to lag 0.

    The mean is removed first: without it the DC component dominates and
    every lag correlates strongly, hiding the periodic signal a duplicated
    feature creates.
    """
    x = np.asarray(profile, dtype=np.float64)
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")
    ac = full[len(x) - 1 :]
    if len(ac) == 0 or ac[0] <= 1e-12:
        return np.zeros_like(ac)
    return ac / ac[0]


def find_duplication_peak(
    autocorr: "np.ndarray",
    subject_height: int,
    *,
    lag_window: Tuple[float, float] = DUPLICATION_LAG_WINDOW,
) -> Dict[str, Any]:
    """Strongest non-zero-lag LOCAL peak within the lag window.

    Local maxima only: a healthy face's autocorrelation decays monotonically
    through the window, so its window maximum sits on the boundary and is not
    a peak. A duplicated feature at distance d produces a genuine local
    maximum at lag d.
    """
    low = max(2, int(round(lag_window[0] * subject_height)))
    high = min(int(round(lag_window[1] * subject_height)), len(autocorr) - 2)
    result: Dict[str, Any] = {
        "lag_window_px": [int(low), int(high)],
        "peak_lag_px": None,
        "peak_lag_fraction_of_subject_height": None,
        "peak_ratio": None,
        "window_max_ratio": None,
    }
    if high < low:
        result["note"] = (
            "Face-region profile too short for the lag window; subject too "
            "small at this render size for a duplication verdict."
        )
        return result
    window = autocorr[low : high + 1]
    result["window_max_ratio"] = float(window.max())
    peaks = [
        (float(autocorr[k]), int(k))
        for k in range(low, high + 1)
        if autocorr[k] > autocorr[k - 1] and autocorr[k] >= autocorr[k + 1]
    ]
    if peaks:
        value, lag = max(peaks)
        result["peak_lag_px"] = lag
        result["peak_lag_fraction_of_subject_height"] = float(lag / subject_height)
        result["peak_ratio"] = value
    return result


def analyze_duplication(
    image: Image.Image,
    *,
    face_fraction: float = FACE_REGION_FRACTION,
    lag_window: Tuple[float, float] = DUPLICATION_LAG_WINDOW,
    threshold: float = DEFAULT_DUPLICATION_PEAK_RATIO_THRESHOLD,
) -> Dict[str, Any]:
    """Run the double-feature detector on a (front clay) render.

    General-purpose: it flags any horizontally-structured feature duplicated
    at a vertical offset of 3-15% of the subject height (mouths, eye lines,
    brows, collars), not a mouth-specific template.
    """
    mask = subject_mask(image)
    bbox = subject_bbox(mask)
    profile, face_rows = horizontal_edge_row_profile(
        image, mask, bbox, face_fraction=face_fraction
    )
    autocorr = normalized_autocorrelation(profile)
    peak = find_duplication_peak(autocorr, bbox.height, lag_window=lag_window)
    peak_ratio = peak.get("peak_ratio")
    suspect = bool(peak_ratio is not None and peak_ratio >= threshold)
    return {
        "subject_bbox_px": {
            "col0": bbox.col0,
            "row0": bbox.row0,
            "col1": bbox.col1,
            "row1": bbox.row1,
        },
        "subject_height_px": bbox.height,
        "face_region_rows_px": [int(face_rows[0]), int(face_rows[1])],
        "face_region_fraction": float(face_fraction),
        "row_profile": [round(float(v), 6) for v in profile],
        "autocorrelation": [round(float(v), 6) for v in autocorr],
        "threshold": float(threshold),
        "duplication_suspect": suspect,
        **peak,
    }


# ---------------------------------------------------------------------------
# Mesh loading, geometry metrics, rendering.
# ---------------------------------------------------------------------------


def _load_single_mesh(glb_path: Path):
    from abstract3d.mesh_ops import as_single_mesh, load_mesh

    return as_single_mesh(load_mesh(glb_path))


def make_clay_mesh(mesh, source_suffix: str):
    """Copy the mesh, strip visuals, and stamp the viewer-frame marker.

    glTF/GLB files are Y-up / front +Z by spec while the shared renderer's
    camera math is canonical-frame (Z-up / front +X); `mesh_ops.render_preview`
    stamps `abstract3d_export_frame='gltf_yup_front_pz'` for .glb inputs so
    `render_mesh_views` applies its one un-rotation. The clay path must
    replicate that or every clay render would lie sideways.
    """
    import trimesh

    clay = mesh.copy()
    clay.visual = trimesh.visual.ColorVisuals(clay)
    if source_suffix.lower() in {".glb", ".gltf"}:
        try:
            clay.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
        except Exception:  # pragma: no cover - trimesh metadata is a dict
            pass
    return clay


def geometry_metrics(mesh) -> Dict[str, Any]:
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    dihedral_rms_rad = float(np.sqrt(np.mean(np.square(angles)))) if angles.size else None
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "body_count": int(mesh.body_count),
        "euler_number": int(mesh.euler_number),
        "dihedral_angle_rms_rad": dihedral_rms_rad,
        "dihedral_angle_rms_deg": (
            float(np.degrees(dihedral_rms_rad)) if dihedral_rms_rad is not None else None
        ),
        "face_adjacency_count": int(angles.size),
    }


def render_clay_views(
    clay_mesh,
    renders_dir: Path,
    *,
    size: int,
    azimuths: Sequence[float],
    elevation: float,
) -> Tuple[Dict[float, Path], Dict[float, Image.Image], Optional[str]]:
    from abstract3d.rendering import get_last_render_backend, render_mesh_views

    views = render_mesh_views(
        clay_mesh, size=int(size), azimuths=tuple(azimuths), elevation=float(elevation)
    )
    if len(views) != len(azimuths):
        raise RuntimeError(
            f"Clay renderer produced {len(views)} views for {len(azimuths)} azimuths. "
            "Install moderngl or matplotlib (both in: pip install \"abstract3d[mesh]\")."
        )
    paths: Dict[float, Path] = {}
    images: Dict[float, Image.Image] = {}
    for azimuth, view in zip(azimuths, views):
        path = renders_dir / f"clay_az{int(round(azimuth)):03d}.png"
        view.save(path)
        paths[float(azimuth)] = path
        images[float(azimuth)] = view
    return paths, images, get_last_render_backend()


def render_textured_views(
    glb_path: Path,
    renders_dir: Path,
    *,
    size: int,
    azimuths: Sequence[float],
    elevation: float,
) -> Tuple[Dict[float, Path], Optional[str]]:
    """Textured full views through `mesh_ops.render_preview`, one azimuth per
    call (a single-azimuth strip is exactly one size x size frame)."""
    from abstract3d.mesh_ops import render_preview

    paths: Dict[float, Path] = {}
    renderer: Optional[str] = None
    for azimuth in azimuths:
        path = renders_dir / f"textured_az{int(round(azimuth)):03d}.png"
        report = render_preview(
            glb_path, path, size=int(size), azimuths=(float(azimuth),), elevation=float(elevation)
        )
        renderer = report.get("renderer") or renderer
        paths[float(azimuth)] = path
    return paths, renderer


def face_crop_box(bbox: SubjectBBox, image_size: Tuple[int, int], *, face_fraction: float = FACE_REGION_FRACTION) -> Tuple[int, int, int, int]:
    """Upper-front crop rectangle: top `face_fraction` of the subject rows,
    horizontally centered on the subject, square-ish (width = crop height),
    clamped to the image. Simple bbox math — no face detection."""
    width, height = image_size
    row0 = bbox.row0
    row1 = min(bbox.row0 + int(round(face_fraction * bbox.height)), height)
    crop_height = max(row1 - row0, 1)
    center_col = (bbox.col0 + bbox.col1) / 2.0
    left = int(round(center_col - crop_height / 2.0))
    right = int(round(center_col + crop_height / 2.0))
    left, right = max(0, left), min(width, right)
    return (left, row0, right, row1)


def save_face_crop(image: Image.Image, box: Tuple[int, int, int, int], out_path: Path, *, target_px: int) -> None:
    crop = image.crop(box)
    scale = float(target_px) / float(max(crop.width, crop.height, 1))
    resized = crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    resized.save(out_path)


# ---------------------------------------------------------------------------
# Texture metrics.
# ---------------------------------------------------------------------------


def texture_view_stats(image: Image.Image, mask: "np.ndarray") -> Dict[str, Any]:
    """Mean saturation + luminance stats of the subject region (0..1 scale)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    selected = rgb[mask]
    if selected.size == 0:
        return {
            "subject_pixel_count": 0,
            "mean_saturation": None,
            "mean_luminance": None,
            "luminance_variance": None,
        }
    channel_max = selected.max(axis=1)
    channel_min = selected.min(axis=1)
    saturation = np.where(channel_max > 0, (channel_max - channel_min) / np.maximum(channel_max, 1e-6), 0.0)
    luminance = selected[:, 0] * 0.299 + selected[:, 1] * 0.587 + selected[:, 2] * 0.114
    return {
        "subject_pixel_count": int(mask.sum()),
        "mean_saturation": float(saturation.mean()),
        "mean_luminance": float(luminance.mean()),
        "luminance_variance": float(luminance.var()),
    }


def metadata_texture_section(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None
    artifacts = metadata.get("texture_artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    views = artifacts.get("observed_view_stats")
    section = {
        "observed_coverage_ratio": artifacts.get("observed_coverage_ratio"),
        "texture_completion": metadata.get("texture_completion", artifacts.get("texture_completion")),
        "views": views if isinstance(views, list) else [],
    }
    if section["observed_coverage_ratio"] is None and not section["views"]:
        section["note"] = "metadata.json carries no texture_artifacts coverage stats."
    return section


# ---------------------------------------------------------------------------
# Scorecard assembly.
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert to plain JSON types (numpy scalars/arrays, Paths,
    non-finite floats -> None)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(scorecard: Dict[str, Any], path: Path) -> None:
    inputs = scorecard["inputs"]
    geometry = scorecard["geometry"]
    dup = scorecard["duplication_detector"]
    texture = scorecard["texture"]
    lines: List[str] = []
    lines.append(f"# Bust assessment: {inputs['label']}")
    lines.append("")
    lines.append(f"- generated: {scorecard['generated_at']}")
    lines.append(f"- input: `{inputs['glb_path']}` (mode: {inputs['mode']})")
    if inputs.get("metadata_path"):
        lines.append(f"- metadata: `{inputs['metadata_path']}`")
    renders = scorecard["renders"]
    lines.append(
        f"- renders: {renders['size']}px, elevation {renders['elevation_deg']} deg, "
        f"azimuths {renders['azimuths_deg']} (clay renderer: {renders['clay_renderer']}, "
        f"textured renderer: {renders['textured_renderer']})"
    )
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    if dup.get("peak_ratio") is not None:
        dup_detail = (
            f"peak ratio {_fmt(dup['peak_ratio'])} at lag {dup['peak_lag_px']} px "
            f"({_fmt(100.0 * dup['peak_lag_fraction_of_subject_height'], 1)}% of subject height)"
        )
    else:
        dup_detail = f"no non-zero-lag local peak in window {dup['lag_window_px']} px"
    verdict_word = "SUSPECT" if dup["duplication_suspect"] else "not suspected"
    lines.append(
        f"- DUPLICATION: **{verdict_word}** — {dup_detail}; threshold {_fmt(dup['threshold'], 3)}"
    )
    back = texture.get("back_view") or {}
    if back.get("mean_luminance") is not None:
        back_word = "SUSPECT" if texture.get("back_view_unpainted_suspect") else "ok"
        lines.append(
            f"- BACK-VIEW PAINT: **{back_word}** — mean luminance {_fmt(back['mean_luminance'])}, "
            f"luminance variance {_fmt(back['luminance_variance'], 5)} "
            f"(unpainted-dark-back = mean < {BACK_UNPAINTED_MAX_MEAN_LUMINANCE} "
            f"and variance < {BACK_UNPAINTED_MAX_LUMINANCE_VARIANCE})"
        )
    else:
        lines.append("- BACK-VIEW PAINT: no back view rendered")
    lines.append(
        f"- GEOMETRY: watertight={_fmt(geometry['watertight'])}, bodies={geometry['body_count']}, "
        f"euler={geometry['euler_number']}, dihedral RMS {_fmt(geometry['dihedral_angle_rms_deg'], 2)} deg"
    )
    lines.append("")

    lines.append("## Geometry")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in (
        "vertex_count",
        "face_count",
        "watertight",
        "body_count",
        "euler_number",
        "dihedral_angle_rms_deg",
        "dihedral_angle_rms_rad",
    ):
        lines.append(f"| {key} | {_fmt(geometry[key])} |")
    lines.append("")

    lines.append("## Duplication detector (front clay render)")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    lines.append(f"| subject height (px) | {dup['subject_height_px']} |")
    lines.append(f"| face region rows | {dup['face_region_rows_px']} (top {int(round(dup['face_region_fraction'] * 100))}% of subject) |")
    lines.append(f"| lag window (px) | {dup['lag_window_px']} (3-15% of subject height) |")
    lines.append(f"| peak lag (px) | {_fmt(dup['peak_lag_px'])} |")
    lines.append(f"| peak ratio | {_fmt(dup['peak_ratio'])} |")
    lines.append(f"| window max ratio | {_fmt(dup['window_max_ratio'])} |")
    lines.append(f"| threshold | {_fmt(dup['threshold'], 3)} |")
    lines.append(f"| duplication_suspect | {_fmt(dup['duplication_suspect'])} |")
    if dup.get("note"):
        lines.append(f"| note | {dup['note']} |")
    lines.append("")
    lines.append("Raw row-profile and autocorrelation curves are in scorecard.json.")
    lines.append("")

    lines.append("## Texture")
    lines.append("")
    meta_section = texture.get("metadata")
    if meta_section:
        lines.append(
            f"- observed_coverage_ratio (metadata): {_fmt(meta_section.get('observed_coverage_ratio'))}"
        )
        if meta_section.get("texture_completion") is not None:
            lines.append(f"- texture_completion (metadata): {_fmt(meta_section.get('texture_completion'))}")
        if meta_section.get("views"):
            lines.append("")
            lines.append("| view | azimuth | coverage_ratio | capture_efficiency | facing_fraction | generated |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for view in meta_section["views"]:
                lines.append(
                    f"| {view.get('label', '-')} | {_fmt(view.get('azimuth_deg'), 1)} "
                    f"| {_fmt(view.get('coverage_ratio'))} | {_fmt(view.get('capture_efficiency'))} "
                    f"| {_fmt(view.get('facing_fraction'))} | {_fmt(view.get('generated'))} |"
                )
        if meta_section.get("note"):
            lines.append(f"- note: {meta_section['note']}")
    else:
        lines.append("- no metadata.json texture stats (bare GLB input or metadata absent)")
    lines.append("")
    lines.append("Per rendered view (subject region, 0..1 scale):")
    lines.append("")
    lines.append("| azimuth | mean_saturation | mean_luminance | luminance_variance |")
    lines.append("| --- | --- | --- | --- |")
    for view in texture["views"]:
        lines.append(
            f"| {_fmt(view['azimuth_deg'], 0)} | {_fmt(view['mean_saturation'])} "
            f"| {_fmt(view['mean_luminance'])} | {_fmt(view['luminance_variance'], 5)} |"
        )
    lines.append("")

    lines.append("## Renders")
    lines.append("")
    for name, file_path in sorted(scorecard["renders"]["files"].items()):
        lines.append(f"- {name}: `{file_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Assessment pipeline.
# ---------------------------------------------------------------------------


def run_assessment(
    *,
    glb_path: Path,
    output_dir: Path,
    bundle_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_path: Optional[Path] = None,
    size: int = DEFAULT_RENDER_SIZE,
    elevation: float = DEFAULT_ELEVATION_DEG,
    azimuths: Sequence[float] = FULL_VIEW_AZIMUTHS,
    duplication_threshold: float = DEFAULT_DUPLICATION_PEAK_RATIO_THRESHOLD,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    started = time.time()
    output_dir = Path(output_dir)
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_single_mesh(glb_path)
    geometry = geometry_metrics(mesh)
    timings["load_and_geometry"] = round(time.time() - started, 3)

    # --- renders -----------------------------------------------------------
    render_start = time.time()
    clay = make_clay_mesh(mesh, glb_path.suffix)
    clay_paths, clay_images, clay_renderer = render_clay_views(
        clay, renders_dir, size=size, azimuths=azimuths, elevation=elevation
    )
    textured_paths, textured_renderer = render_textured_views(
        glb_path, renders_dir, size=size, azimuths=azimuths, elevation=elevation
    )

    # Face close-up pair from dedicated 2x-resolution front renders so the
    # crop carries real detail instead of an upscale of the full view (the
    # whole point of this harness is that low-res inspection ships defects).
    from abstract3d.mesh_ops import render_preview
    from abstract3d.rendering import render_mesh_views

    crop_render_size = int(size) * 2
    hires_clay_path = renders_dir / "clay_front_hires.png"
    hires_clay = render_mesh_views(clay, size=crop_render_size, azimuths=(0.0,), elevation=elevation)[0]
    hires_clay.save(hires_clay_path)
    hires_textured_path = renders_dir / "textured_front_hires.png"
    render_preview(glb_path, hires_textured_path, size=crop_render_size, azimuths=(0.0,), elevation=elevation)
    hires_textured = Image.open(hires_textured_path)

    hires_bbox = subject_bbox(subject_mask(hires_clay))
    crop_box = face_crop_box(hires_bbox, hires_clay.size)
    face_crop_clay_path = renders_dir / "face_crop_clay.png"
    face_crop_textured_path = renders_dir / "face_crop_textured.png"
    save_face_crop(hires_clay, crop_box, face_crop_clay_path, target_px=int(size))
    save_face_crop(hires_textured, crop_box, face_crop_textured_path, target_px=int(size))
    timings["renders"] = round(time.time() - render_start, 3)

    # --- duplication detector on the front clay render ----------------------
    detector_start = time.time()
    front_azimuth = float(azimuths[0])
    duplication = analyze_duplication(
        clay_images[front_azimuth], threshold=duplication_threshold
    )
    timings["duplication_detector"] = round(time.time() - detector_start, 3)

    # --- texture metrics -----------------------------------------------------
    texture_start = time.time()
    # Clay and textured renders share camera math over identical geometry, so
    # the clay silhouette is the subject mask for the textured view — unless
    # the two passes fell back to different renderer backends (different
    # framing), in which case each textured view masks itself.
    same_renderer = clay_renderer == textured_renderer
    texture_views: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if not same_renderer:
        warnings.append(
            f"#FALLBACK clay renderer ({clay_renderer}) != textured renderer "
            f"({textured_renderer}); textured views are masked from their own "
            "pixels instead of the clay silhouette."
        )
    for azimuth in azimuths:
        textured_image = Image.open(textured_paths[float(azimuth)])
        mask = (
            subject_mask(clay_images[float(azimuth)])
            if same_renderer
            else subject_mask(textured_image)
        )
        stats = texture_view_stats(textured_image, mask)
        texture_views.append({"azimuth_deg": float(azimuth), **stats})
    back_view = next((v for v in texture_views if v["azimuth_deg"] == 180.0), None)
    back_unpainted = bool(
        back_view is not None
        and back_view["mean_luminance"] is not None
        and back_view["mean_luminance"] < BACK_UNPAINTED_MAX_MEAN_LUMINANCE
        and back_view["luminance_variance"] < BACK_UNPAINTED_MAX_LUMINANCE_VARIANCE
    )
    texture_section = {
        "metadata": metadata_texture_section(metadata),
        "views": texture_views,
        "back_view": back_view,
        "back_view_unpainted_suspect": back_unpainted,
        "back_unpainted_thresholds": {
            "max_mean_luminance": BACK_UNPAINTED_MAX_MEAN_LUMINANCE,
            "max_luminance_variance": BACK_UNPAINTED_MAX_LUMINANCE_VARIANCE,
        },
    }
    timings["texture_metrics"] = round(time.time() - texture_start, 3)
    timings["total"] = round(time.time() - started, 3)

    render_files: Dict[str, Path] = {}
    for azimuth, p in clay_paths.items():
        render_files[f"clay_az{int(round(azimuth)):03d}"] = p
    for azimuth, p in textured_paths.items():
        render_files[f"textured_az{int(round(azimuth)):03d}"] = p
    render_files["clay_front_hires"] = hires_clay_path
    render_files["textured_front_hires"] = hires_textured_path
    render_files["face_crop_clay"] = face_crop_clay_path
    render_files["face_crop_textured"] = face_crop_textured_path

    scorecard: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "label": bundle_dir.name if bundle_dir is not None else glb_path.stem,
            "mode": "bundle" if bundle_dir is not None else "glb",
            "bundle_dir": bundle_dir,
            "glb_path": glb_path,
            "metadata_path": metadata_path,
        },
        "renders": {
            "size": int(size),
            "crop_render_size": crop_render_size,
            "elevation_deg": float(elevation),
            "azimuths_deg": [float(a) for a in azimuths],
            "clay_renderer": clay_renderer,
            "textured_renderer": textured_renderer,
            "face_crop_box_px": list(crop_box),
            "files": render_files,
        },
        "geometry": geometry,
        "duplication_detector": duplication,
        "texture": texture_section,
        "warnings": warnings,
        "timings_s": timings,
    }
    scorecard = _json_safe(scorecard)

    json_path = output_dir / "scorecard.json"
    json_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(scorecard, output_dir / "scorecard.md")
    return scorecard


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Dict[str, Any]], Optional[Path]]:
    if args.bundle:
        bundle_dir = Path(args.bundle).expanduser()
        if not bundle_dir.is_dir():
            raise InputError(f"Bundle directory not found: {bundle_dir}")
        glb_path = bundle_dir / "scene.glb"
        if not glb_path.is_file():
            contents = ", ".join(sorted(p.name for p in bundle_dir.iterdir())[:20]) or "(empty)"
            raise InputError(
                f"Bundle directory has no scene.glb: {bundle_dir} (contents: {contents}). "
                "Pass a generation bundle, or use --glb for a bare mesh file."
            )
        metadata: Optional[Dict[str, Any]] = None
        metadata_path: Optional[Path] = bundle_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded if isinstance(loaded, dict) else None
                if metadata is None:
                    print(
                        f"WARNING: {metadata_path} is not a JSON object; texture "
                        "metadata section will be empty.",
                        file=sys.stderr,
                    )
            except (OSError, json.JSONDecodeError) as e:
                print(
                    f"WARNING: could not read {metadata_path} ({e}); texture "
                    "metadata section will be empty.",
                    file=sys.stderr,
                )
                metadata = None
        else:
            print(
                f"WARNING: bundle has no metadata.json ({metadata_path}); texture "
                "metadata section will be empty.",
                file=sys.stderr,
            )
            metadata_path = None
        return glb_path, bundle_dir, metadata, metadata_path

    glb_path = Path(args.glb).expanduser()
    if not glb_path.is_file():
        raise InputError(f"GLB file not found: {glb_path}")
    return glb_path, None, None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bust_assessment",
        description="Render + score a bust reconstruction (assessment tool, not a gate).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", help="Generation bundle directory (scene.glb + metadata.json).")
    source.add_argument("--glb", help="Bare mesh file (.glb/.gltf/.obj/...) without bundle metadata.")
    parser.add_argument("--output-dir", required=True, help="Directory for renders + scorecards.")
    parser.add_argument(
        "--size", type=int, default=DEFAULT_RENDER_SIZE,
        help=f"Render + face-crop size in px (default {DEFAULT_RENDER_SIZE}).",
    )
    parser.add_argument(
        "--elevation", type=float, default=DEFAULT_ELEVATION_DEG,
        help=f"Camera elevation in degrees (default {DEFAULT_ELEVATION_DEG}).",
    )
    parser.add_argument(
        "--duplication-threshold", type=float,
        default=DEFAULT_DUPLICATION_PEAK_RATIO_THRESHOLD,
        help=(
            "Autocorrelation peak ratio at/above which duplication is suspected "
            f"(default {DEFAULT_DUPLICATION_PEAK_RATIO_THRESHOLD}; calibrated between the known-bad "
            "double-mouth bundle at 0.071 and the known-cleaner reconstruction at 0.034)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        glb_path, bundle_dir, metadata, metadata_path = _resolve_inputs(args)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    scorecard = run_assessment(
        glb_path=glb_path,
        output_dir=Path(args.output_dir).expanduser(),
        bundle_dir=bundle_dir,
        metadata=metadata,
        metadata_path=metadata_path,
        size=args.size,
        elevation=args.elevation,
        duplication_threshold=args.duplication_threshold,
    )

    dup = scorecard["duplication_detector"]
    texture = scorecard["texture"]
    back = texture.get("back_view") or {}
    meta_section = texture.get("metadata") or {}
    print(f"scorecard: {Path(args.output_dir) / 'scorecard.json'}")
    print(f"markdown:  {Path(args.output_dir) / 'scorecard.md'}")
    print(
        "duplication: suspect=%s peak_ratio=%s peak_lag_px=%s threshold=%s"
        % (dup["duplication_suspect"], _fmt(dup["peak_ratio"]), _fmt(dup["peak_lag_px"]), dup["threshold"])
    )
    print("coverage (metadata): %s" % _fmt(meta_section.get("observed_coverage_ratio")))
    print(
        "back view: mean_luminance=%s luminance_variance=%s unpainted_suspect=%s"
        % (
            _fmt(back.get("mean_luminance")),
            _fmt(back.get("luminance_variance"), 5),
            texture.get("back_view_unpainted_suspect"),
        )
    )
    for warning in scorecard.get("warnings", []):
        print(warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
