#!/usr/bin/env python
"""Viewgen recipe bench: which local model+LoRA+prompt produces views of the
SAME subject at REQUESTED angles? (S2 adversarial bench, 2026-07-21)

Arms (all LOCAL mlx-gen; provider pinned on every call — a remote fallback is
a P0 failure of the bench itself):

  A  klein i2i, production rotate prompt (baseline; reproduce, don't re-tune)
  B  klein i2i + Flux2-Klein-9B-consistency-V2 LoRA, consistency prompt
  C  klein i2i + multiple-angles-flux-klein-9b LoRA (trigger convention is
     probed first: ModelScope camera-command vs fal <sks> pose tokens)
  D  Qwen-Image-Edit-2511 + fal multiple-angles LoRA (+ Lightning 4-step)
  E  best of B/C/D re-run with clay guidance (photo primary + e20-mesh clay
     render at the target azimuth as a second reference image)

Measurements per generated view:
  1. pose honesty      head-band silhouette-IoU ruler against e20 mesh clays,
                       full signed sweep (flip-aware); |declared - measured|
  2. row consistency   row_alignment_score / best_row_shift vs source photo
                       (subject-bbox frames)
  3. subject fidelity  general overlapping-region comparison on head-height-
                       normalized bands (glasses/nose/mouth+chin): edge and
                       dark-fraction profile correlation + LAB deltas; the
                       mouth open/closed disagreement surfaces as a dark-peak
                       excess in the mouth band (one INSTANCE of the general
                       signal, not a bespoke detector)
  4. matte quality     background uniformity + alpha extractability
  5. provenance        seed/steps/model/LoRA application counts + wall time

Usage:
  viewgen_bench.py --arms A,B          generate+measure the listed arms
  viewgen_bench.py --probe-c           run the arm-C trigger-convention probe
  viewgen_bench.py --smoke             tiny LoRA-wiring smoke generation
  viewgen_bench.py --measure-only      re-measure existing PNGs (no i2i spend)
  viewgen_bench.py --sheet             (re)build contact sheets only

Outputs: out/bust-viewbench/<arm>/<slot>[_raw].png, measurements.json,
contact_sheet_<arm>.png. Sequential generations; run the whole script under
`nice`. Every generation records raw md5 + full request for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT = REPO / "out/bust-viewbench"
SOURCE_PHOTO = Path("/tmp/laurent_bust_crop.png")
MESH_BUNDLE = REPO / "out/laurent-bust-redo/e20_fixed_views"
LORA_DIR = Path.home() / "Library/Caches/mflux/loras"

KLEIN_MODEL = "AbstractFramework/flux.2-klein-9b-8bit"
QWEN_EDIT_MODEL = "AbstractFramework/qwen-image-edit-2511-8bit"
LORA_CONSISTENCY = str(LORA_DIR / "Flux2-Klein-9B-consistency-V2.safetensors")
LORA_KLEIN_ANGLES = str(LORA_DIR / "multiple-angles-flux-klein-9b.safetensors")
LORA_72POSES = str(LORA_DIR / "flux-multi-angles-v2-72poses-comfy.safetensors")
# The shipped fal LoRA uses diffusers-PEFT keys (transformer.*.lora_A/B) that
# mflux 0.17.5's QwenLoRAMapping does NOT match (0/1680 keys applied, verified
# live) — scripts/experimental/convert_qwen_angles_lora.py rewrites the keys
# to the diffusion_model.* style the mapping accepts (12/14 submodules;
# img_mod.1/txt_mod.1 have no mflux target and stay unapplied).
LORA_QWEN_ANGLES = str(LORA_DIR / "qwen-image-edit-2511-multiple-angles-lora-mfluxkeys.safetensors")
LORA_QWEN_LIGHTNING = str(
    LORA_DIR / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")

SIZE = 1024
# Subject wording: arm A reproduces the production caption noun verbatim;
# arms B..E use the operator-direction consistency wording.
PRODUCTION_NOUN = "man wearing blind face"
CONSISTENCY_CLAUSE = (
    "It is the SAME man as in the input photo, IDENTICAL person: identical "
    "face, identical closed mouth with lips together and a calm neutral "
    "expression, identical dark flat-top sunglasses, identical short dark "
    "curly hair, identical dense dark beard, identical black t-shirt. Real "
    "photograph, plain dark background, soft diffuse even lighting, sharp "
    "focus."
)
ANGLE_PHRASES = {
    "side65_left": (
        "seen from his left side at about 65 degrees from frontal, a strong "
        "three-quarter view with most of the far half of his face hidden"),
    "side65_right": (
        "seen from his right side at about 65 degrees from frontal, a strong "
        "three-quarter view with most of the far half of his face hidden"),
    "profile_left": (
        "seen in exact left side profile, a 90 degree side view, only the "
        "left side of his face visible"),
    "profile_right": (
        "seen in exact right side profile, a 90 degree side view, only the "
        "right side of his face visible"),
    "back": (
        "seen directly from behind, the back of his head and his shoulders, "
        "no part of his face visible"),
}
# fal <sks> pose-token vocabulary (Qwen 2511 angles LoRA; 45-degree lattice).
SKS_TOKENS = {
    "side65_left": "<sks> front-left quarter view eye-level shot medium shot",
    "side65_right": "<sks> front-right quarter view eye-level shot medium shot",
    "profile_left": "<sks> left side view eye-level shot medium shot",
    "profile_right": "<sks> right side view eye-level shot medium shot",
    "back": "<sks> back view eye-level shot medium shot",
}
# ModelScope camera-command convention (dx8152 family wording, English).
CAMERA_COMMANDS = {
    "side65_left": "Rotate the camera 65 degrees to the left.",
    "side65_right": "Rotate the camera 65 degrees to the right.",
    "profile_left": "Rotate the camera 90 degrees to the left.",
    "profile_right": "Rotate the camera 90 degrees to the right.",
    "back": "Rotate the camera 180 degrees to show the back.",
}
# Declared azimuth per slot in the project's signed convention (positive =
# subject's left side toward the camera; the ruler resolves actual sign).
SLOT_DECLARED = {
    "side65_left": 65.0, "side65_right": -65.0,
    "profile_left": 90.0, "profile_right": -90.0, "back": 180.0,
}
SLOT_SEEDS = {"side65_left": 11, "side65_right": 12, "profile_left": 13,
              "profile_right": 14, "back": 15}
# Arm A reproduces the production rotate lane exactly: its vocabulary has no
# 65-degree phrase (named angles are 45/90/135/180 only) — the 65-degree
# buckets are structurally unsupported, which is itself a bench finding.
ARM_A_SEEDS = {"profile_left": 52025, "profile_right": 53025, "back": 54025}


def arm_configs() -> Dict[str, Dict[str, Any]]:
    from abstract3d.backends.hunyuan3d_runtime import _geometry_view_prompt

    def consistency_prompt(slot: str) -> str:
        return f"{ANGLE_PHRASES[slot]}. {CONSISTENCY_CLAUSE}".capitalize()

    arm_a_slots = {}
    for slot, label in (("profile_left", "side_left"),
                        ("profile_right", "side_right"), ("back", "back")):
        arm_a_slots[slot] = {
            "prompt": _geometry_view_prompt(label, PRODUCTION_NOUN),
            "declared_deg": SLOT_DECLARED[slot],
            "seed": ARM_A_SEEDS[slot],
        }
    return {
        "A": {
            "title": "klein i2i, production rotate prompt (baseline)",
            "model": KLEIN_MODEL, "loras": [], "steps": 8, "guidance": None,
            "slots": arm_a_slots,
            "unsupported_slots": {
                "side65_left": "production _view_phrase vocabulary has no "
                               "65-degree entry (named angles 45/90/135/180)",
                "side65_right": "same vocabulary gap",
            },
        },
        "B": {
            "title": "klein i2i + consistency LoRA (operator recipe)",
            "model": KLEIN_MODEL,
            "loras": [{"source": LORA_CONSISTENCY, "scale": 1.0}],
            "steps": 8, "guidance": None,
            "slots": {slot: {"prompt": consistency_prompt(slot),
                             "declared_deg": SLOT_DECLARED[slot],
                             "seed": SLOT_SEEDS[slot]}
                      for slot in SLOT_DECLARED},
        },
        "C": {
            "title": "klein i2i + multiple-angles (klein9b Camera-Blocking) LoRA",
            "model": KLEIN_MODEL,
            "loras": [{"source": LORA_KLEIN_ANGLES, "scale": 1.0}],
            "steps": 8, "guidance": None,
            # Prompt convention resolved by --probe-c; default = camera command
            # + consistency clause (ModelScope training family).
            "slots": {slot: {"prompt": f"{CAMERA_COMMANDS[slot]} {CONSISTENCY_CLAUSE}",
                             "declared_deg": SLOT_DECLARED[slot],
                             "seed": SLOT_SEEDS[slot]}
                      for slot in SLOT_DECLARED},
        },
        "D": {
            "title": "Qwen-Image-Edit-2511 + fal multiple-angles LoRA (+Lightning 4step)",
            "model": QWEN_EDIT_MODEL,
            "loras": [{"source": LORA_QWEN_ANGLES, "scale": 1.0},
                      {"source": LORA_QWEN_LIGHTNING, "scale": 1.0}],
            "steps": 4, "guidance": 1.0,
            # 65 degrees is OUTSIDE the 45-degree pose lattice: the nearest
            # expressible pose is the front quarter (declared 45), recorded
            # as a vocabulary gap, not silently relabeled.
            "slots": {
                "side65_left": {"prompt": SKS_TOKENS["side65_left"],
                                "declared_deg": 45.0, "target_deg": 65.0,
                                "seed": SLOT_SEEDS["side65_left"]},
                "side65_right": {"prompt": SKS_TOKENS["side65_right"],
                                 "declared_deg": -45.0, "target_deg": -65.0,
                                 "seed": SLOT_SEEDS["side65_right"]},
                "profile_left": {"prompt": SKS_TOKENS["profile_left"],
                                 "declared_deg": 90.0,
                                 "seed": SLOT_SEEDS["profile_left"]},
                "profile_right": {"prompt": SKS_TOKENS["profile_right"],
                                  "declared_deg": -90.0,
                                  "seed": SLOT_SEEDS["profile_right"]},
                "back": {"prompt": SKS_TOKENS["back"], "declared_deg": 180.0,
                         "seed": SLOT_SEEDS["back"]},
            },
        },
        # Arm E is materialized by --make-e ARM after B/C/D are measured.
    }


# --------------------------------------------------------------------------
# Local, instrumented generation (provider pinned; asset metadata kept so the
# LoRA application report and the local-route proof land in measurements.json)
# --------------------------------------------------------------------------

class LocalI2I:
    def __init__(self) -> None:
        from abstract3d.image_composition import _abstractvision_capability

        self._cap = _abstractvision_capability(None)

    def generate(self, prompt: str, image_bytes: bytes, *, model: str,
                 seed: int, steps: int, guidance: Optional[float],
                 loras: Sequence[Dict[str, Any]],
                 reference_images: Sequence[bytes] = (),
                 negative_prompt: Optional[str] = None,
                 width: int = SIZE, height: int = SIZE) -> Tuple[bytes, Dict[str, Any], float]:
        from abstractvision.vision_manager import VisionManager

        binding = self._cap._resolve_backend_binding(provider="mlx-gen", model=model)
        backend = self._cap._activate_request_backend(binding)
        backend_name = type(backend).__name__
        if "mflux" not in backend_name.lower() and "mlx" not in backend_name.lower():
            raise RuntimeError(
                f"PRIVACY P0: resolved backend {backend_name!r} is not the "
                "local mlx-gen backend — refusing to generate.")
        extra: Dict[str, Any] = {"width": int(width), "height": int(height)}
        if reference_images:
            extra["reference_images"] = [bytes(b) for b in reference_images]
        kwargs: Dict[str, Any] = {"seed": int(seed), "steps": int(steps), "extra": extra}
        if guidance is not None:
            kwargs["guidance_scale"] = float(guidance)
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if loras:
            kwargs["lora_adapters"] = [dict(item) for item in loras]
        vm = VisionManager(backend=backend)
        started = time.perf_counter()
        asset = vm.edit_image(str(prompt), image=bytes(image_bytes), **kwargs)
        seconds = time.perf_counter() - started
        metadata = dict(getattr(asset, "metadata", {}) or {})
        if str(metadata.get("source") or "") != "mlx-gen":
            raise RuntimeError(
                f"PRIVACY P0: generation metadata source={metadata.get('source')!r} "
                "is not mlx-gen — the image may have left the machine.")
        keep = {k: metadata.get(k) for k in (
            "source", "model", "base_model", "quantization_bits", "seed",
            "steps", "reference_image_count", "edit_mode",
            "requested_lora_adapters", "lora_paths", "lora_scales",
            "lora_applied_file_count", "lora_applied_target_count")
            if k in metadata}
        return bytes(asset.data), keep, seconds


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def load_bench_mesh():
    """The e20 bake mesh in the canonical frame — geometry.glb verbatim, the
    exact load texture_forensics_ruler2.py (the instrument that produced the
    operator's 65/-67.5/180 baseline numbers) uses. Verified this session:
    geometry.glb and the rotated scene.glb load are silhouette-identical
    (IoU >= 0.999 at 0/65/-90)."""
    import trimesh

    return trimesh.load(str(MESH_BUNDLE / "geometry.glb"), force="mesh",
                        process=False)


class PoseRuler:
    """Audit-grade azimuth estimate, reused VERBATIM from the viewgen audit
    (`viewgen_audit_analyze.estimate_azimuth`: production
    register_matte_to_clay on the full bust + head-band-only IoU, swept
    +/-55 deg around the DECLARED azimuth with a 2.5 deg refine) — the
    instrument whose numbers the operator's brief quotes; cross-validated
    this session on viewgen_fixed: +65 / -67.5 / +170 vs the brief's
    65 / -67.5 / 180 (back sweeps are ~flat, per the audit).

    Two bench additions, both diagnostics rather than replacements:
      * a MIRRORED sweep around -declared; if it beats the declared-side
        sweep by a clear margin the view is flagged side_flip (a full-circle
        sweep was tried first and rejected: hair-mass silhouette ambiguity
        produces spurious far-angle maxima — instrument dead-end recorded in
        the bench report);
      * boundary censoring: when the argmax sits on the sweep edge the
        estimate is a bound, not a measurement.
    """

    def __init__(self, mesh) -> None:
        self._mesh = mesh

    def estimate(self, matte_path: Path, declared_deg: float) -> Dict[str, Any]:
        from viewgen_audit_analyze import estimate_azimuth

        row = estimate_azimuth(str(matte_path), self._mesh, float(declared_deg))
        sweep = {float(k): v for k, v in row["sweep"].items()}
        measured = float(row["estimated_deg"])
        out: Dict[str, Any] = {
            "declared_deg": float(declared_deg),
            "measured_deg": measured,
            "pose_error_deg": round(abs(float(declared_deg) - measured), 1),
            "iou_at_measured": row["iou_at_estimate"],
            "iou_at_declared": row["iou_at_nominal"],
            "sweep": {f"{k:+.1f}": round(v, 4) for k, v in sorted(sweep.items())},
        }
        lo, hi = min(sweep), max(sweep)
        out["sweep_boundary_hit"] = bool(measured <= lo + 2.6 or measured >= hi - 2.6)
        # Mirrored-declaration flip check (side views only).
        if abs(declared_deg) < 155.0:
            mirrored = estimate_azimuth(str(matte_path), self._mesh,
                                        -float(declared_deg))
            out["mirror_measured_deg"] = float(mirrored["estimated_deg"])
            out["mirror_iou"] = mirrored["iou_at_estimate"]
            out["side_flip"] = bool(
                mirrored["iou_at_estimate"] > row["iou_at_estimate"] + 0.03)
        else:
            out["side_flip"] = False
        return out


def matte_quality(raw_image, matte) -> Dict[str, Any]:
    """Background uniformity of the RAW generation + alpha extractability."""
    import numpy as np

    rgb = np.asarray(raw_image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(matte)[:, :, 3]
    fg = alpha > 128
    h, w = fg.shape
    border = np.ones((h, w), dtype=bool)
    rows = np.nonzero(fg.any(axis=1))[0]
    cols = np.nonzero(fg.any(axis=0))[0]
    if len(rows) and len(cols):
        top = max(0, rows[0] - 8)
        bottom = min(h, rows[-1] + 8)
        left = max(0, cols[0] - 8)
        right = min(w, cols[-1] + 8)
        border[top:bottom, left:right] = False
    bg = rgb[border] if border.any() else rgb[~fg]
    semi = ((alpha > 8) & (alpha < 248)).sum() / alpha.size
    return {
        "foreground_frac": round(float(fg.mean()), 4),
        "background_rgb_std": round(float(bg.std(axis=0).mean()), 2) if len(bg) else None,
        "background_rgb_mean": [round(float(v), 1) for v in bg.mean(axis=0)] if len(bg) else None,
        "semi_transparent_frac": round(float(semi), 5),
        "alpha_ok": bool(0.05 < float(fg.mean()) < 0.75),
    }


def _band_signals(rgba: Any, mask: Any, rows: Tuple[int, int]) -> Dict[str, Any]:
    """Generic per-row signals over the foreground of a row band."""
    import numpy as np

    lo, hi = max(0, rows[0]), min(mask.shape[0], rows[1])
    if hi - lo < 4:
        return {}
    luma = 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]
    grad = np.abs(np.diff(luma, axis=0, prepend=luma[:1]))
    from skimage import color as skcolor

    lab = skcolor.rgb2lab(rgba[:, :, :3])
    dark = (luma < 0.24)
    out = {"dark": [], "edge": [], "L": [], "a": [], "b": []}
    for row in range(lo, hi):
        sel = mask[row]
        n = int(sel.sum())
        if not n:
            for key in out:
                out[key].append(np.nan)
            continue
        out["dark"].append(float(dark[row][sel].sum()) / n)
        out["edge"].append(float(grad[row][sel].mean()))
        out["L"].append(float(lab[row, :, 0][sel].mean()))
        out["a"].append(float(lab[row, :, 1][sel].mean()))
        out["b"].append(float(lab[row, :, 2][sel].mean()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _profile_corr(a, b) -> Optional[float]:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 4 or len(b) < 4:
        return None
    xb = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    valid = np.isfinite(a) & np.isfinite(xb)
    if valid.sum() < 4:
        return None
    a, xb = a[valid], xb[valid]
    if a.std() < 1e-9 or xb.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, xb)[0, 1]), 3)


def region_fidelity(view_matte, view_record: Dict[str, Any],
                    src_matte, src_record: Dict[str, Any]) -> Dict[str, Any]:
    """Shared-visibility band comparison in head-normalized rows.

    Bands are defined once, in head units u = (row - top)/(chin - top), from
    the SOURCE's measured landmarks, then mapped into each image through its
    OWN (top, chin) anchors — the row-law-compatible way to compare a front
    photo with a rotated view. Signals are subject-agnostic per-row stats.
    """
    import numpy as np

    def anchors(record):
        top = record.get("top")
        chin = record.get("chin") or record.get("shoulder")
        return top, chin

    s_top, s_chin = anchors(src_record)
    v_top, v_chin = anchors(view_record)
    if None in (s_top, s_chin, v_top, v_chin) or s_chin <= s_top or v_chin <= v_top:
        return {"error": "missing head anchors"}

    def u_of(record, row, default):
        if row is None:
            return default
        top, chin = anchors(record)
        return (row - top) / (chin - top)

    u_glasses = u_of(src_record, src_record.get("glasses_row"), 0.45)
    u_nose_base = u_of(src_record, src_record.get("nose_base"), 0.72)
    bands_u = {
        "glasses": (u_glasses - 0.07, u_glasses + 0.11),
        "nose": (u_glasses + 0.11, u_nose_base),
        "mouth_chin": (u_nose_base, 1.02),
    }

    def rows_of(record, u_pair):
        top, chin = anchors(record)
        span = chin - top
        return (int(top + u_pair[0] * span), int(top + u_pair[1] * span))

    src_rgba = np.asarray(src_matte.convert("RGBA"), dtype=np.float32) / 255.0
    view_rgba = np.asarray(view_matte.convert("RGBA"), dtype=np.float32) / 255.0
    src_mask = src_rgba[:, :, 3] > 0.5
    view_mask = view_rgba[:, :, 3] > 0.5

    out: Dict[str, Any] = {}
    for name, u_pair in bands_u.items():
        s_sig = _band_signals(src_rgba, src_mask, rows_of(src_record, u_pair))
        v_sig = _band_signals(view_rgba, view_mask, rows_of(view_record, u_pair))
        if not s_sig or not v_sig:
            out[name] = {"error": "band empty"}
            continue
        dl = np.nanmean(v_sig["L"]) - np.nanmean(s_sig["L"])
        dab = float(np.hypot(np.nanmean(v_sig["a"]) - np.nanmean(s_sig["a"]),
                             np.nanmean(v_sig["b"]) - np.nanmean(s_sig["b"])))
        row = {
            "edge_corr": _profile_corr(s_sig["edge"], v_sig["edge"]),
            "dark_corr": _profile_corr(s_sig["dark"], v_sig["dark"]),
            "delta_L": round(float(dl), 1),
            "delta_ab": round(dab, 1),
        }
        if name == "mouth_chin":
            # The parted-lips instance of the general comparison: an interior
            # dark peak (a lip gap) the source's closed mouth does not have.
            interior = slice(2, max(3, len(v_sig["dark"]) - 2))
            v_peak = float(np.nanmax(v_sig["dark"][interior])) if len(v_sig["dark"]) > 5 else 0.0
            s_peak = float(np.nanmax(s_sig["dark"][interior])) if len(s_sig["dark"]) > 5 else 0.0
            row["mouth_dark_peak_view"] = round(v_peak, 3)
            row["mouth_dark_peak_source"] = round(s_peak, 3)
            row["mouth_dark_excess"] = round(v_peak - s_peak, 3)
        out[name] = row
    return out


def measure_view(raw_path: Path, matte_path: Path, slot: str,
                 declared_deg: float, ruler: PoseRuler,
                 src_matte, src_record: Dict[str, Any],
                 src_profile) -> Dict[str, Any]:
    import numpy as np
    from PIL import Image

    from abstract3d.view_consistency import (
        best_row_shift,
        row_alignment_score,
        row_profile,
    )
    from viewgen_audit_measure import dark_band, head_geometry, measure

    raw = Image.open(raw_path)
    matte = Image.open(matte_path).convert("RGBA")
    label = {"side65_left": "side_left", "profile_left": "side_left",
             "side65_right": "side_right", "profile_right": "side_right",
             "back": "back"}[slot]
    row: Dict[str, Any] = {"slot": slot, "label": label}

    row["matte"] = matte_quality(raw, matte)
    row["pose"] = ruler.estimate(matte_path, declared_deg)

    # Row consistency vs the source photo, both in subject-bbox frames.
    def bbox(image):
        alpha = np.asarray(image)[:, :, 3]
        rows_i = np.nonzero((alpha > 128).any(axis=1))[0]
        cols_i = np.nonzero((alpha > 128).any(axis=0))[0]
        return (int(cols_i[0]), int(rows_i[0]), int(cols_i[-1]) + 1, int(rows_i[-1]) + 1)

    try:
        view_profile = row_profile(matte, bbox=bbox(matte))
        row["row_alignment_score"] = round(float(
            row_alignment_score(view_profile, src_profile)), 3)
        shift = int(best_row_shift(view_profile, src_profile, max_shift_frac=0.18))
        row["row_shift_frac"] = round(shift / max(1, len(view_profile)), 4)
    except Exception as exc:
        row["row_alignment_error"] = f"{type(exc).__name__}: {exc}"

    # Landmarks + region fidelity (skip face bands on back views).
    try:
        record = measure(str(matte_path), "rgba", label)
        row["landmarks"] = {k: record.get(k) for k in (
            "top", "glasses_row", "nose_tip", "nose_base", "mouth", "chin",
            "shoulder", "head_height_px")}
        if label != "back":
            row["fidelity"] = region_fidelity(matte, record, src_matte, src_record)
        rgba = np.asarray(matte.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        glasses = dark_band(rgba, mask, head_geometry(mask)) if label != "back" else None
        row["glasses"] = ({"present": glasses is not None,
                           **({"row": glasses["row"],
                               "peak_frac": glasses["peak_frac"]} if glasses else {})})
    except Exception as exc:
        row["landmark_error"] = f"{type(exc).__name__}: {exc}"

    from viewgen_audit_ladder import tone_stats

    row["tone"] = tone_stats(matte, src_matte)
    return row


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def load_source():
    from PIL import Image

    from abstract3d.segmentation import remove_background_robust

    source = Image.open(SOURCE_PHOTO).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)
    return source


def source_png_bytes(source) -> bytes:
    buffer = io.BytesIO()
    source.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def measurements_path() -> Path:
    return OUT / "measurements.json"


def read_measurements() -> Dict[str, Any]:
    path = measurements_path()
    if path.exists():
        return json.loads(path.read_text())
    return {"arms": {}}


def write_measurements(data: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    measurements_path().write_text(json.dumps(data, indent=1))


def generate_arm(arm_id: str, config: Dict[str, Any], i2i: LocalI2I,
                 source_bytes: bytes, clay_refs: Optional[Dict[str, bytes]] = None,
                 force: bool = False) -> None:
    from abstract3d.segmentation import remove_background_robust
    from PIL import Image

    arm_dir = OUT / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    data = read_measurements()
    arm_row = data["arms"].setdefault(arm_id, {})
    arm_row["config"] = {
        "title": config["title"], "model": config["model"],
        "loras": config["loras"], "steps": config["steps"],
        "guidance": config["guidance"], "size": SIZE,
        **({"unsupported_slots": config["unsupported_slots"]}
           if config.get("unsupported_slots") else {}),
    }
    views = arm_row.setdefault("views", {})
    for slot, spec in config["slots"].items():
        raw_path = arm_dir / f"{slot}_raw.png"
        if raw_path.exists() and not force:
            print(f"[{arm_id}/{slot}] exists, skipping generation", flush=True)
            continue
        refs: List[bytes] = []
        if clay_refs and slot in clay_refs:
            refs = [clay_refs[slot]]
        print(f"[{arm_id}/{slot}] generating (seed {spec['seed']}, "
              f"steps {config['steps']})...", flush=True)
        try:
            payload, metadata, seconds = i2i.generate(
                spec["prompt"], source_bytes, model=config["model"],
                seed=spec["seed"], steps=config["steps"],
                guidance=config["guidance"], loras=config["loras"],
                reference_images=refs)
        except Exception as exc:
            views[slot] = {"generation_error": f"{type(exc).__name__}: {exc}",
                           "request": {k: spec[k] for k in ("prompt", "seed", "declared_deg")}}
            write_measurements(data)
            print(f"[{arm_id}/{slot}] FAILED: {exc}", flush=True)
            continue
        raw_path.write_bytes(payload)
        matte = remove_background_robust(Image.open(io.BytesIO(payload)))
        matte.save(arm_dir / f"{slot}.png")
        views[slot] = {
            "request": {"prompt": spec["prompt"], "seed": spec["seed"],
                        "declared_deg": spec["declared_deg"],
                        **({"target_deg": spec["target_deg"]}
                           if "target_deg" in spec else {}),
                        "reference_images": len(refs)},
            "generation": {**metadata, "seconds": round(seconds, 1),
                           "raw_md5": hashlib.md5(payload).hexdigest()},
        }
        write_measurements(data)
        print(f"[{arm_id}/{slot}] done in {seconds:.0f}s "
              f"(lora files applied: {metadata.get('lora_applied_file_count')})",
              flush=True)


def measure_arm(arm_id: str, config: Dict[str, Any], ruler: PoseRuler,
                source, src_record, src_profile) -> None:
    arm_dir = OUT / arm_id
    data = read_measurements()
    arm_row = data["arms"].setdefault(arm_id, {})
    views = arm_row.setdefault("views", {})
    for slot, spec in config["slots"].items():
        raw_path = arm_dir / f"{slot}_raw.png"
        matte_path = arm_dir / f"{slot}.png"
        if not matte_path.exists():
            continue
        print(f"[{arm_id}/{slot}] measuring...", flush=True)
        row = views.setdefault(slot, {})
        row["measurement"] = measure_view(
            raw_path, matte_path, slot, spec["declared_deg"], ruler,
            source, src_record, src_profile)
        write_measurements(data)
        pose = row["measurement"]["pose"]
        print(f"[{arm_id}/{slot}] declared {pose['declared_deg']:+.0f} -> "
              f"measured {pose['measured_deg']:+.1f} "
              f"(err {pose['pose_error_deg']}, flip={pose['side_flip']})",
              flush=True)


def build_contact_sheet(arm_id: str, config: Dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    arm_dir = OUT / arm_id
    data = read_measurements()
    views = data["arms"].get(arm_id, {}).get("views", {})
    slots = [s for s in config["slots"] if (arm_dir / f"{s}.png").exists()]
    if not slots:
        return
    cell = 384
    sheet = Image.new("RGB", (cell * len(slots), cell + 64), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, slot in enumerate(slots):
        tile = Image.open(arm_dir / f"{slot}.png").convert("RGB")
        tile.thumbnail((cell, cell))
        sheet.paste(tile, (i * cell + (cell - tile.width) // 2, 8))
        meta = views.get(slot, {}).get("measurement", {})
        pose = meta.get("pose", {})
        text = (f"{slot}\ndecl {pose.get('declared_deg')} meas "
                f"{pose.get('measured_deg')} err {pose.get('pose_error_deg')}")
        draw.text((i * cell + 8, cell + 12), text, fill=(230, 230, 230))
    sheet.save(OUT / f"contact_sheet_{arm_id}.png")
    print(f"contact sheet -> {OUT / f'contact_sheet_{arm_id}.png'}", flush=True)


def render_clay_refs(slots: Dict[str, Any]) -> Dict[str, bytes]:
    from abstract3d.rendering import render_mesh_views

    mesh = load_bench_mesh()
    out: Dict[str, bytes] = {}
    clay_dir = OUT / "clay_refs"
    clay_dir.mkdir(parents=True, exist_ok=True)
    for slot, spec in slots.items():
        azimuth = spec["declared_deg"]
        clay = render_mesh_views(mesh, size=SIZE, azimuths=[float(azimuth)],
                                 elevation=0.0)[0].convert("RGB")
        buffer = io.BytesIO()
        clay.save(buffer, format="PNG")
        out[slot] = buffer.getvalue()
        clay.save(clay_dir / f"{slot}.png")
    return out


def smoke(i2i: LocalI2I, source_bytes: bytes) -> None:
    """Tiny LoRA-wiring smoke: 512px, 4 steps, consistency LoRA. Verifies the
    lora_adapters request path reaches mflux (applied file/target counts)."""
    payload, metadata, seconds = i2i.generate(
        "The same man, identical face, seen from the front.", source_bytes,
        model=KLEIN_MODEL, seed=7, steps=4, guidance=None,
        loras=[{"source": LORA_CONSISTENCY, "scale": 1.0}],
        width=512, height=512)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_consistency.png").write_bytes(payload)
    print(json.dumps({"seconds": round(seconds, 1), **metadata}, indent=1))


def probe_c(i2i: LocalI2I, source_bytes: bytes) -> None:
    """Arm-C trigger-convention probe: one slot, two prompt conventions."""
    probes = {
        "c_probe_command": f"{CAMERA_COMMANDS['profile_left']} {CONSISTENCY_CLAUSE}",
        "c_probe_sks": SKS_TOKENS["profile_left"],
    }
    probe_dir = OUT / "C_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in probes.items():
        print(f"[{name}] generating...", flush=True)
        payload, metadata, seconds = i2i.generate(
            prompt, source_bytes, model=KLEIN_MODEL, seed=13, steps=8,
            guidance=None, loras=[{"source": LORA_KLEIN_ANGLES, "scale": 1.0}])
        (probe_dir / f"{name}.png").write_bytes(payload)
        print(f"[{name}] {seconds:.0f}s lora_applied="
              f"{metadata.get('lora_applied_file_count')}", flush=True)


def make_arm_e(base_arm: str) -> Dict[str, Any]:
    configs = arm_configs()
    base = configs[base_arm]
    slots = {}
    for slot in SLOT_DECLARED:
        base_slot = base["slots"].get(slot)
        prompt = (base_slot or {}).get("prompt", "")
        clay_clause = (
            " A second reference image shows an untextured gray 3D model of "
            "the SAME man in exactly the requested pose: match that pose, "
            "framing and silhouette exactly, and keep his facial features at "
            "the same heights as in the model.")
        slots[slot] = {
            "prompt": (prompt + clay_clause) if prompt else clay_clause.strip(),
            "declared_deg": SLOT_DECLARED[slot],
            "seed": SLOT_SEEDS[slot],
        }
    return {
        "title": f"arm {base_arm} + clay guidance (photo primary, e20 clay reference)",
        "model": base["model"], "loras": base["loras"],
        "steps": base["steps"], "guidance": base["guidance"],
        "slots": slots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="", help="comma list, e.g. A,B,C,D")
    parser.add_argument("--make-e", default="", metavar="ARM",
                        help="materialize+run arm E from the given base arm")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe-c", action="store_true")
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--sheet", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even when the PNG exists")
    args = parser.parse_args()

    configs = arm_configs()
    if args.make_e:
        configs["E"] = make_arm_e(args.make_e)
        data = read_measurements()
        data["arms"].setdefault("E", {})["base_arm"] = args.make_e
        write_measurements(data)
    else:
        # Rebuild E from the recorded base arm so --measure-only/--sheet
        # runs cover it without regenerating.
        recorded_base = read_measurements()["arms"].get("E", {}).get("base_arm")
        if recorded_base:
            configs["E"] = make_arm_e(recorded_base)

    arm_ids = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    if args.make_e and "E" not in arm_ids:
        arm_ids.append("E")

    needs_generation = (args.smoke or args.probe_c
                        or (arm_ids and not args.measure_only))
    i2i = LocalI2I() if needs_generation else None
    source = load_source()
    source_bytes = source_png_bytes(source)

    if args.smoke:
        smoke(i2i, source_bytes)
        return
    if args.probe_c:
        probe_c(i2i, source_bytes)
        return

    if not arm_ids and not args.sheet:
        parser.error("nothing to do: pass --arms, --make-e, --smoke, "
                     "--probe-c or --sheet")

    if arm_ids and not args.measure_only:
        for arm_id in arm_ids:
            config = configs[arm_id]
            clay_refs = render_clay_refs(config["slots"]) if arm_id == "E" else None
            generate_arm(arm_id, config, i2i, source_bytes, clay_refs,
                         force=args.force)

    if arm_ids:
        from viewgen_audit_measure import measure

        from abstract3d.view_consistency import row_profile

        src_record = measure(str(SOURCE_PHOTO), "photo", "front")
        import numpy as np

        alpha = np.asarray(source)[:, :, 3]
        rows = np.nonzero((alpha > 128).any(axis=1))[0]
        cols = np.nonzero((alpha > 128).any(axis=0))[0]
        src_profile = row_profile(source, bbox=(int(cols[0]), int(rows[0]),
                                                int(cols[-1]) + 1, int(rows[-1]) + 1))
        ruler = PoseRuler(load_bench_mesh())
        data = read_measurements()
        data["source"] = {"path": str(SOURCE_PHOTO),
                          "landmarks": {k: src_record.get(k) for k in (
                              "top", "glasses_row", "nose_base", "mouth",
                              "chin", "shoulder", "head_height_px")}}
        write_measurements(data)
        for arm_id in arm_ids:
            measure_arm(arm_id, configs[arm_id], ruler, source, src_record,
                        src_profile)

    for arm_id in (arm_ids or [a for a in configs if (OUT / a).exists()]):
        if arm_id in configs:
            build_contact_sheet(arm_id, configs[arm_id])


if __name__ == "__main__":
    main()
