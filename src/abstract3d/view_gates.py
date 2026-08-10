"""Synthesized-view quality gates: matte cleanliness + subject identity.

Closes the e22_oneshot_v2 hole (2026-07-22, control experiment
out/bust/control_cleanviews_384): the loop ladder ACCEPTED a side view
with (a) a different man's face (thinner, goatee for a full beard,
different glasses) and (b) catastrophic matte debris (black paint-streak
curtains, torn interior alpha, a building-facade ghost across the shirt)
— pose honesty passed (the pose WAS correct ~90 deg) and row consistency
passed (the correction family explained the rows); nothing judged matte
cleanliness or subject identity, and pass 2 carved the debris into the
mesh (strategy_v2 W3's gap, first slice + a matte gate).

Both gates follow the `pose_gate` contract (reference_generation):
``gate(view_rgba, *, label, azimuth_deg, elevation_deg) -> verdict`` with
``verdict["passed"]`` plus loud reasons; a crashing gate ABSTAINS
(passed=True + note) rather than burning the ladder — the pose_unmeasured
doctrine. Verdicts land on the attempt rows (loop_refgen_attempts.jsonl).

MATTE CLEANLINESS — two general channels, both relative to the SOURCE
photo (the subject's own baseline, never a subject-specific rule):

1. interior semi-transparency: semi-transparent alpha AWAY from the
   silhouette rim (alpha tears INSIDE the body). Clean mattes carry
   semi-transparency only at the rim (hair wisps); torn-debris draws
   carry it mid-body.
2. content edge density: Canny edges of the scale-smoothed luminance over
   the interior, per interior pixel, as a RATIO to the source photo's own
   edge density. A bust has a bounded content-edge budget (features +
   folds + hair); paint-streak/facade debris is edge soup at 3-5x the
   subject's own density. Palette/LAB channels measurably CANNOT carry
   this class: the ladder tone-matches every candidate toward the source,
   so the debris is in-gamut by construction (measured 2026-07-22: the
   "straw-blond" thatch reads L*=24 after tone match, NN-distance to the
   source gamut ~2).

Calibration (scripts/experimental/view_gate_calibration*.py + the
edge-density table in /tmp/edge_density_cal.json, 2026-07-22; every
number measured on disk evidence):

  view                                  int_semi  edge_ratio  verdict
  REJECT e22v2 synthesized side_left     0.151     1.99       tears
  REJECT e22v2 synthesized back          0.028     3.40       edges
  REJECT e22v2 synthesized side_right    0.087     5.43       both
  REJECT e22v2 windowed side_left        0.011     3.06       edges
  REJECT e22v2 windowed back             0.031     4.01       edges
  PASS   bench E (5 views)              <=0.0015  <=1.00      clean
  PASS   bench B (5 views)              <=0.0248  <=1.50      clean
  PASS   e20 viewgen_fixed side_left     0.011     1.00       clean
  PASS   e20 viewgen_fixed side_right    0.015     1.35       clean
  PASS   e20 viewgen_fixed back          0.026     2.01       clean

  INTERIOR_SEMI_MAX 0.05: rejects >= 0.087 (1.7x above), passes <= 0.026
  (1.9x below). EDGE_DENSITY_RATIO_MAX 2.5: rejects >= 3.06 (1.2x above),
  passes <= 2.01 (1.2x below) — the geometric mean of the gap. The e20-era
  "harmonized" set-A side view (decapitated crop, e11 era) measures 2.93
  and correctly rejects.

SUBJECT IDENTITY — strategy_v2 C7's named fallback (DINO-global): cosine
similarity between DINOv2 embeddings of the anatomical head crop of the
view and of the source photo. Composition/row/palette statistics
measurably CANNOT carry identity (measured 2026-07-22: every such
statistic ranks a same-man PROFILE farther from the front photo than the
tone-matched wrong-man view). Face-visibility rule: the channel applies
only when |azimuth| <= IDENTITY_MAX_ABS_AZIMUTH_DEG — measured on backs,
same-man backs score at wrong-man level (0.23-0.26), so a back must
abstain (strategy C7 abstention).

Calibration (facebook/dinov2-base, head-crop on neutral gray):

  view                                   cosine   verdict
  REJECT e22v2 side_left (wrong man)     0.2796   reject
  PASS   bench E profiles/side65        0.4976-0.6190  pass
  PASS   bench B profiles/side65        0.5369-0.6958  pass
  PASS   e20 viewgen_fixed sides        0.5807-0.5863  pass
  (backs, abstained)                    0.2303-0.2624  n/a

  IDENTITY_MIN_COSINE 0.38: reject at 0.28 (0.74x), pass floor 0.498
  (1.3x). The embedder runs on CPU only (the DiT owns MPS) and loads
  lazily once per process; an unavailable model ABSTAINS LOUDLY
  (identity_unmeasured in the verdict + report), never silently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# Matte cleanliness (calibration table in the module docstring).
MATTE_INTERIOR_SEMI_MAX = 0.05
MATTE_EDGE_DENSITY_RATIO_MAX = 2.5
# Floor on the source's own edge density before the ratio: a pathologically
# smooth source must not turn ordinary texture into an infinite ratio.
EDGE_DENSITY_SOURCE_FLOOR = 0.004
# Alpha thresholds: subject mask at 0.5; "support" down to 0.06 so the rim
# band covers soft matte edges; semi-transparent = (0.06, 0.94).
_ALPHA_SUBJECT = 0.5
_ALPHA_SUPPORT = 0.06
_ALPHA_OPAQUE = 0.94
# Scale-proportional constants (fractions of the subject's pixel height):
# rim/erosion width 0.8%, luminance smoothing sigma h/300 (min 1.5 px) —
# the smoothing kills single-pixel generator grain (bench arm B) while
# keeping the mid-frequency debris strokes the gate exists to catch.
_RIM_FRACTION = 0.008
_SMOOTH_FRACTION = 1.0 / 300.0
# Canny thresholds on the smoothed 8-bit luminance. Absolute values are
# meaningful here because the ladder tone-matches candidates toward the
# source before gating and the source is a matted photo.
_CANNY_LO = 20
_CANNY_HI = 60

# Subject identity (calibration table in the module docstring).
IDENTITY_MIN_COSINE = 0.38
IDENTITY_MAX_ABS_AZIMUTH_DEG = 100.0
DINO_MODEL_ID = "facebook/dinov2-base"

# Process-level embedder cache: one CPU model per process, loaded lazily
# at the first gated candidate (never at gate construction — unit tests
# and mocked ladders must not pay a 330 MB model load).
_EMBEDDER_CACHE: Dict[str, Any] = {}


def _subject_height(mask: Any) -> int:
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("empty subject mask")
    return int(rows[-1] - rows[0] + 1)


def matte_cleanliness_stats(view_rgba: Any) -> Dict[str, float]:
    """Interior-semi fraction + content edge density for one RGBA image.

    Pure measurement (no thresholds): the gate compares a view's numbers
    against the source photo's own (see `build_matte_cleanliness_gate`).
    """
    import cv2
    import numpy as np
    from scipy.ndimage import binary_erosion, gaussian_filter

    arr = np.asarray(view_rgba.convert("RGBA"), dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    mask = alpha > _ALPHA_SUBJECT
    height = _subject_height(mask)
    rim_iterations = max(2, int(round(_RIM_FRACTION * height)))

    support = alpha > _ALPHA_SUPPORT
    semi = (alpha > _ALPHA_SUPPORT) & (alpha < _ALPHA_OPAQUE)
    rim = support & ~binary_erosion(support, iterations=rim_iterations)
    interior_semi = float((semi & ~rim).sum() / max(mask.sum(), 1))

    interior = binary_erosion(mask, iterations=rim_iterations)
    lum = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    smoothed = gaussian_filter(lum, max(1.5, height * _SMOOTH_FRACTION))
    edges = cv2.Canny(
        np.clip(smoothed, 0, 255).astype(np.uint8), _CANNY_LO, _CANNY_HI
    )
    edges[~interior] = 0
    interior_px = int(interior.sum())
    return {
        "interior_semi_frac": round(interior_semi, 5),
        "edge_density": round(float((edges > 0).sum()) / max(interior_px, 1), 5),
        "interior_px": interior_px,
        "subject_height_px": height,
    }


def build_matte_cleanliness_gate(
    source_rgba: Any,
    *,
    interior_semi_max: float = MATTE_INTERIOR_SEMI_MAX,
    edge_density_ratio_max: float = MATTE_EDGE_DENSITY_RATIO_MAX,
) -> Callable[..., Dict[str, Any]]:
    """Matte/alpha cleanliness acceptance gate bound to the source photo.

    The source's own matte + edge statistics set the baseline (subject-
    agnostic: a plaid-shirt subject raises its own edge budget), computed
    once at build. Verdicts carry both channels, the thresholds, and a
    loud reason on rejection.
    """
    try:
        source_stats: Optional[Dict[str, float]] = matte_cleanliness_stats(source_rgba)
    except Exception as exc:  # pragma: no cover - degenerate source
        source_stats = None
        source_error = f"{type(exc).__name__}: {exc}"

    def matte_gate(view_rgba: Any, *, label: str, azimuth_deg: float,
                   elevation_deg: float = 0.0) -> Dict[str, Any]:
        del label, azimuth_deg, elevation_deg  # gate is view-content-only
        if source_stats is None:
            return {
                "measured": False, "passed": True,
                "note": ("matte_unmeasured: source matte statistics "
                         f"unavailable ({source_error})"),
            }
        try:
            stats = matte_cleanliness_stats(view_rgba)
        except Exception as exc:
            return {
                "measured": False, "passed": True,
                "note": f"matte_unmeasured: {type(exc).__name__}: {exc}",
            }
        ratio = stats["edge_density"] / max(
            source_stats["edge_density"], EDGE_DENSITY_SOURCE_FLOOR
        )
        verdict: Dict[str, Any] = {
            "measured": True,
            "interior_semi_frac": stats["interior_semi_frac"],
            "interior_semi_max": float(interior_semi_max),
            "edge_density": stats["edge_density"],
            "source_edge_density": source_stats["edge_density"],
            "edge_density_ratio": round(ratio, 3),
            "edge_density_ratio_max": float(edge_density_ratio_max),
        }
        reasons = []
        if stats["interior_semi_frac"] > float(interior_semi_max):
            reasons.append(
                "torn interior alpha: semi-transparent fraction away from "
                f"the silhouette rim {stats['interior_semi_frac']} > "
                f"{float(interior_semi_max)}"
            )
        if ratio > float(edge_density_ratio_max):
            reasons.append(
                "content edge density "
                f"{stats['edge_density']} is {round(ratio, 2)}x the source "
                f"photo's own {source_stats['edge_density']} (> "
                f"{float(edge_density_ratio_max)}x): paint-streak/debris "
                "texture the subject does not carry"
            )
        verdict["passed"] = not reasons
        if reasons:
            verdict["reason"] = (
                "matte cleanliness: " + "; ".join(reasons)
                + " — rejected; redrawing within the attempt ladder"
            )
        return verdict

    return matte_gate


def _resolve_dino_embedder() -> Callable[[Any], Any]:
    """Process-cached CPU embedder over DINO_MODEL_ID.

    Returns a callable mapping a PIL RGB image -> L2-normalized 1-D torch
    tensor. Raises (loudly) when transformers/torch or the checkpoint are
    unavailable — the gate converts that into a recorded abstention.
    """
    cached = _EMBEDDER_CACHE.get(DINO_MODEL_ID)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    model = model.to("cpu").eval()  # MPS belongs to the DiT/i2i pools

    def embed(image: Any) -> Any:
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            out = model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            vector = pooled[0] if pooled is not None else out.last_hidden_state[0, 0]
            return torch.nn.functional.normalize(vector, dim=0)

    _EMBEDDER_CACHE[DINO_MODEL_ID] = embed
    return embed


def subject_head_crop(rgba: Any) -> Any:
    """Anatomical head crop (window-anchor head span, subject columns)
    composited on neutral gray — the identity embedding's input frame.

    Gray compositing keeps the matte from leaking background statistics
    into the embedding; the head span keeps torso/clothing from dominating
    a global embedding.
    """
    import numpy as np
    from PIL import Image

    from .loop_conditioning import window_anchors

    anchors = window_anchors(rgba)
    arr = np.asarray(rgba.convert("RGBA"))
    mask = arr[..., 3] > 128
    band = mask[anchors["head_top"]:anchors["shoulder"]]
    cols = np.flatnonzero(band.any(axis=0))
    if cols.size == 0:
        raise ValueError("no subject columns in the head span")
    crop = arr[anchors["head_top"]:anchors["shoulder"], cols[0]:cols[-1] + 1]
    base = np.full(crop.shape[:2] + (3,), 128, dtype=np.float32)
    alpha = crop[..., 3:4].astype(np.float32) / 255.0
    rgb = crop[..., :3].astype(np.float32) * alpha + base * (1.0 - alpha)
    return Image.fromarray(rgb.astype(np.uint8))


def build_subject_identity_gate(
    source_rgba: Any,
    *,
    min_cosine: float = IDENTITY_MIN_COSINE,
    max_abs_azimuth_deg: float = IDENTITY_MAX_ABS_AZIMUTH_DEG,
    embedder: Optional[Callable[[Any], Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Same-subject acceptance gate (strategy C7's DINO-global fallback).

    `embedder` injection exists for tests; production resolves the cached
    CPU DINOv2 lazily at the FIRST gated candidate. The source embedding
    is computed once and reused. Views beyond `max_abs_azimuth_deg`
    abstain (measured: same-man backs score at wrong-man level — there is
    no face to witness).
    """
    state: Dict[str, Any] = {"source_embedding": None, "embed": embedder}

    def identity_gate(view_rgba: Any, *, label: str, azimuth_deg: float,
                      elevation_deg: float = 0.0) -> Dict[str, Any]:
        del label
        azimuth = abs(((float(azimuth_deg) + 180.0) % 360.0) - 180.0)
        if azimuth > float(max_abs_azimuth_deg) or abs(float(elevation_deg)) > 30.0:
            return {
                "measured": False, "passed": True,
                "note": (
                    "identity_abstained: the identity channel applies only "
                    f"where the subject's front is visible (|azimuth| <= "
                    f"{float(max_abs_azimuth_deg)} deg, got "
                    f"{round(azimuth, 1)}); measured same-subject backs "
                    "score at wrong-subject level"
                ),
            }
        try:
            embed = state["embed"]
            if embed is None:
                embed = _resolve_dino_embedder()
                state["embed"] = embed
            if state["source_embedding"] is None:
                state["source_embedding"] = embed(subject_head_crop(source_rgba))
            view_embedding = embed(subject_head_crop(view_rgba))
            cosine = float(state["source_embedding"] @ view_embedding)
        except Exception as exc:
            # LOUD abstention (pose_unmeasured doctrine): a missing or
            # crashing embedder must not burn the ladder, but the record
            # must say the identity channel did not run.
            return {
                "measured": False, "passed": True,
                "note": (
                    "identity_unmeasured: the identity embedder is "
                    f"unavailable ({type(exc).__name__}: {exc}). Install "
                    f"transformers/torch and pre-download {DINO_MODEL_ID} "
                    "for the same-subject gate."
                ),
            }
        verdict: Dict[str, Any] = {
            "measured": True,
            "model": DINO_MODEL_ID,
            "cosine": round(cosine, 4),
            "min_cosine": float(min_cosine),
            "passed": cosine >= float(min_cosine),
        }
        if not verdict["passed"]:
            verdict["reason"] = (
                f"subject identity: head-crop embedding cosine {round(cosine, 4)} "
                f"vs the source photo is below the calibrated floor "
                f"{float(min_cosine)} ({DINO_MODEL_ID}) — a different subject "
                "(measured wrong-man 0.28 vs same-man >= 0.50); rejected — "
                "redrawing within the attempt ladder"
            )
        return verdict

    return identity_gate


__all__ = [
    "DINO_MODEL_ID",
    "EDGE_DENSITY_SOURCE_FLOOR",
    "IDENTITY_MAX_ABS_AZIMUTH_DEG",
    "IDENTITY_MIN_COSINE",
    "MATTE_EDGE_DENSITY_RATIO_MAX",
    "MATTE_INTERIOR_SEMI_MAX",
    "build_matte_cleanliness_gate",
    "build_subject_identity_gate",
    "matte_cleanliness_stats",
    "subject_head_crop",
]
