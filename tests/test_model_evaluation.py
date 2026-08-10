"""Unit tests for the general model evaluation (synthetic cases, no GL).

The contract under test (docs/research/evaluation_strategy_v2.md):
  - structure agreement must fail on pasted/fabricated content and pass on
    exposure-shifted but correctly-placed content;
  - chroma disagreement must be exposure-immune but color-sensitive;
  - the banded palette must flag out-of-palette regions and forgive global
    exposure changes;
  - mirror symmetry must be exact for symmetric silhouettes and drop for a
    one-sided bulge;
  - cavity mass must respond to local valleys, not uniform darkness;
  - profile articulation must separate articulated from soft profiles;
  - aggregation must take the WORST angle (min semantics), never a mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract3d.model_evaluation import (  # noqa: E402
    EvaluationConfig,
    aggregate_verdict,
    build_palette,
    cavity_fraction,
    chroma_disagreement,
    edge_row_profile,
    face_band_mask,
    mirror_symmetry_iou,
    ncc,
    profile_articulation,
    refine_row_alignment,
    structure_agreement,
    subject_bbox,
    _measure,
)


# ---------------------------------------------------------------- fixtures


def textured_face(size: int = 320, seed: int = 3) -> np.ndarray:
    """Synthetic photo-like luminance: smooth base + horizontal feature bars
    + fine texture (a stand-in for a face with feature lines and stubble)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 140 + 40 * np.sin(yy / 37.0) + 20 * np.cos(xx / 53.0)
    for row, depth in ((size // 3, 60), (size // 2, 45), (2 * size // 3, 70)):
        band = np.exp(-((yy - row) ** 2) / (2 * 3.0**2))
        base = base - depth * band
    return base + rng.normal(0, 4.0, (size, size))


def full_region(size: int = 320) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


# ---------------------------------------------------------------- structure


def test_structure_agreement_identical_images_pass():
    img = textured_face()
    res = structure_agreement(img, img, full_region(), patch_px=40, search_px=6)
    assert res["disagree_frac"] == 0.0


def test_structure_agreement_exposure_shift_still_passes():
    img = textured_face()
    darker = img * 0.55 + 10.0  # global exposure/tone change
    res = structure_agreement(darker, img, full_region(), patch_px=40, search_px=6)
    assert res["disagree_frac"] < 0.10


def test_structure_agreement_small_misalignment_tolerated():
    img = textured_face()
    shifted = np.roll(img, 4, axis=0)  # within the search radius
    res = structure_agreement(shifted, img, full_region(), patch_px=40, search_px=6)
    assert res["disagree_frac"] < 0.15


def test_structure_agreement_pasted_feature_fails():
    """A doctored render with a pasted feature block (ghost content) must
    raise the disagreeing fraction over the clean baseline."""
    img = textured_face()
    doctored = img.copy()
    # paste a strong alien structure (dense vertical bars) into a quadrant
    yy, xx = np.mgrid[0:80, 0:120]
    doctored[40:120, 60:180] = 200 - 90 * ((xx // 6) % 2)
    clean = structure_agreement(img, img, full_region(), patch_px=40, search_px=6)
    res = structure_agreement(doctored, img, full_region(), patch_px=40, search_px=6)
    assert res["disagree_frac"] > clean["disagree_frac"] + 0.05
    # the disagreeing patches must include the pasted area
    bad = [(r, c) for r, c, s in res["patches"] if s < 0.25]
    assert any(0 <= r <= 120 and 40 <= c <= 180 for r, c in bad)


def test_structure_agreement_missing_feature_fails():
    """Content missing where the photo has structure (soft/erased region)."""
    img = textured_face()
    erased = img.copy()
    erased[80:160, 80:240] = float(img.mean())  # flatten a band
    res = structure_agreement(erased, img, full_region(), patch_px=40, search_px=6)
    assert res["disagree_frac"] > 0.05


# ---------------------------------------------------------------- chroma


def test_chroma_exposure_immune_but_color_sensitive():
    """A 2x exposure change must stay well under the acceptance threshold
    (LAB a,b drift a little with RGB gain — near-immunity is the contract);
    an actually wrong color must blow far past it."""
    config = EvaluationConfig()
    rgb = np.zeros((64, 64, 3)) + [120.0, 90.0, 70.0]
    darker = rgb * 0.5
    region = np.ones((64, 64), dtype=bool)
    exposure_p95 = chroma_disagreement(darker, rgb, region)["p95"]
    assert exposure_p95 < config.tex_chroma_max
    wrong = rgb.copy()
    wrong[16:48, 16:48] = [90.0, 90.0, 220.0]  # blue patch on skin-like base
    wrong_p95 = chroma_disagreement(wrong, rgb, region)["p95"]
    assert wrong_p95 > config.tex_chroma_max
    assert wrong_p95 > 3.0 * exposure_p95


# ---------------------------------------------------------------- palette


def _photo_like(size: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Two-stratum subject: dark 'shirt' bottom, skin-tone top."""
    rgb = np.zeros((size, size, 3))
    rgb[: size // 2] = [180.0, 140.0, 110.0]
    rgb[size // 2 :] = [40.0, 40.0, 45.0]
    rng = np.random.default_rng(5)
    rgb += rng.normal(0, 6.0, rgb.shape)
    mask = np.ones((size, size), dtype=bool)
    return np.clip(rgb, 0, 255), mask


def test_palette_accepts_exposure_change_rejects_alien_color():
    rgb, mask = _photo_like()
    palette = build_palette(rgb, mask, bands=4)
    exposure = palette.exposure_map(rgb * 0.6, mask)
    ok = palette.outlier_fraction(rgb * 0.6, mask, exposure)
    assert ok < 0.02
    smeared = rgb * 0.6
    smeared[130:180, 40:200] = [235.0, 235.0, 240.0]  # white smear on 'shirt'
    bad = palette.outlier_fraction(smeared, mask, exposure)
    assert bad > 0.05


def test_palette_band_locality():
    """Skin color is legal at the skin band but alien deep inside the shirt
    strata (beyond the +-1 band slack)."""
    rgb, mask = _photo_like(240)
    palette = build_palette(rgb, mask, bands=6)
    doctored = rgb.copy()
    doctored[210:236, 60:180] = [180.0, 140.0, 110.0]  # skin at the bottom band
    exposure = (1.0, 0.0)
    assert palette.outlier_fraction(doctored, mask, exposure) > palette.outlier_fraction(
        rgb, mask, exposure
    )


# ---------------------------------------------------------------- symmetry


def _bust_mask(size: int = 200, bulge: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size / 2
    head = (yy - 60) ** 2 / 45**2 + (xx - cx) ** 2 / 38**2 <= 1.0
    torso = (yy > 100) & (np.abs(xx - cx) < 70) & (yy < 185)
    mask = head | torso
    if bulge:
        blob = (yy - 70) ** 2 / 22**2 + (xx - (cx + 44)) ** 2 / bulge**2 <= 1.0
        mask |= blob
    return mask


def test_mirror_symmetry_exact_for_symmetric_pair():
    mask = _bust_mask()
    assert mirror_symmetry_iou(mask, mask[:, ::-1]) == pytest.approx(1.0)


def test_mirror_symmetry_fails_one_sided_bulge():
    """A mirrored-asymmetric silhouette must fail mesh plausibility."""
    config = EvaluationConfig()
    plain = _bust_mask()
    bulged = _bust_mask(bulge=30)
    # +az sees the bulge on one side, -az (mirrored plain view) does not
    iou = mirror_symmetry_iou(bulged, plain[:, ::-1])
    assert iou < config.mesh_symmetry_iou_min
    assert mirror_symmetry_iou(plain, plain[:, ::-1]) > config.mesh_symmetry_iou_min


# ---------------------------------------------------------------- cavities


def test_cavity_fraction_responds_to_local_valleys_not_uniform_darkness():
    size = 200
    mask = np.zeros((size, size), dtype=bool)
    mask[20:180, 40:160] = True
    flat = np.where(mask, 0.30, 1.0)  # uniformly dark: no local valley
    assert cavity_fraction(flat, mask, (0.1, 0.9)) == 0.0
    slotted = flat.copy()
    slotted[90:98, 60:140] = 0.05  # carved slot (open-mouth class)
    assert cavity_fraction(slotted, mask, (0.1, 0.9)) > 0.01


# ---------------------------------------------------------------- profile


def _profile_mask(articulated: bool, size: int = 240) -> np.ndarray:
    """Profile-view silhouette: back edge straight; leading edge either
    articulated (several protrusions) or soft (one smooth bump)."""
    mask = np.zeros((size, size), dtype=bool)
    rows = np.arange(size)
    lead = np.full(size, 90.0)
    bump = lambda c, w, d: d * np.exp(-((rows - c) ** 2) / (2 * w**2))
    if articulated:
        lead -= bump(70, 7, 14) + bump(110, 6, 26) + bump(140, 5, 12) + bump(165, 7, 16)
    else:
        lead -= bump(110, 28, 26)
    for r in range(20, 220):
        mask[r, int(lead[r]) : 190] = True
    return mask


def test_profile_articulation_separates_soft_from_articulated():
    config = EvaluationConfig()
    strong = profile_articulation(_profile_mask(True), (0.1, 0.9))
    soft = profile_articulation(_profile_mask(False), (0.1, 0.9))
    assert strong > soft
    assert soft < config.mesh_articulation_min


# ---------------------------------------------------------------- alignment


def test_refine_row_alignment_recovers_shift():
    """render_profile(x) = photo_profile(x + 0.08): the search must find the
    map photo(x*scale+shift) ~ render(x), i.e. shift ~ +0.08, and score it
    near-perfectly."""
    rng = np.random.default_rng(11)
    base = np.convolve(rng.random(256), np.ones(5) / 5, mode="same")
    xs = np.linspace(0.0, 1.0, 256)
    render = np.interp(xs + 0.08, xs, base)
    scale, shift, score = refine_row_alignment(base, render)
    assert score > 0.9
    assert shift == pytest.approx(0.08, abs=0.02)
    assert scale == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------- helpers


def test_edge_row_profile_marks_feature_rows():
    img = textured_face()
    mask = np.ones(img.shape, dtype=bool)
    prof = edge_row_profile(img, mask)
    # feature bars at 1/3, 1/2, 2/3 of height must be local maxima regions
    n = len(prof)
    for frac in (1 / 3, 1 / 2, 2 / 3):
        idx = int(frac * n)
        window = prof[max(idx - 12, 0) : idx + 12]
        assert window.max() > 2.0 * np.median(prof)


def test_subject_bbox_and_face_band():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:80, 10:70] = True
    assert subject_bbox(mask) == (20, 80, 10, 70)
    band = face_band_mask(mask, (0.2, 0.5))
    rows = np.flatnonzero(band.any(axis=1))
    assert rows.min() == 20 + int(0.2 * 60)
    assert rows.max() == 20 + int(0.5 * 60) - 1


def test_ncc_bounds():
    a = np.sin(np.linspace(0, 6, 100))
    assert ncc(a, a) == pytest.approx(1.0)
    assert ncc(a, -a) == pytest.approx(-1.0)


# ---------------------------------------------------------------- verdict


def test_measure_worst_angle_is_min_not_mean():
    """Nine good angles must not outvote one bad angle."""
    per_angle = {str(a): 0.05 for a in range(9)}
    per_angle["bad"] = 0.90
    m = _measure(per_angle, threshold=0.32, direction="max")
    assert m["pass"] is False
    assert m["worst_angle"] == "bad"
    # mean would have passed — the whole point
    assert float(np.mean(list(per_angle.values()))) < 0.32


def test_measure_min_direction():
    m = _measure({"0": 0.95, "30": 0.85}, threshold=0.9, direction="min")
    assert m["pass"] is False and m["worst_angle"] == "30"


def test_measure_all_nan_fails_loudly():
    m = _measure({"0": float("nan")}, threshold=0.5, direction="min")
    assert m["pass"] is False and "note" in m


def test_aggregate_verdict_named_subverdicts():
    measures = {
        "texture": {
            "source_structure": _measure({"0": 0.5}, 0.32, "max"),  # fail
            "palette_outliers": _measure({"0": 0.001}, 0.015, "max"),
        },
        "mesh": {
            "mirror_symmetry": _measure({"30": 0.96}, 0.93, "min"),
        },
    }
    verdict = aggregate_verdict(measures)
    assert verdict["mesh_ok"] is True
    assert verdict["texture_ok"] is False
    assert verdict["accept"] is False
    assert verdict["failing_measures"] == ["texture.source_structure@0"]
    assert verdict["worst_angle"]["texture"]["source_structure"]["value"] == 0.5


def test_aggregate_verdict_accepts_only_when_both_axes_pass():
    good = {
        "texture": {"palette_outliers": _measure({"0": 0.001}, 0.015, "max")},
        "mesh": {"mirror_symmetry": _measure({"30": 0.96}, 0.93, "min")},
    }
    assert aggregate_verdict(good)["accept"] is True


# ---------------------------------------------------------------- config


def test_config_is_frozen_and_complete():
    config = EvaluationConfig()
    with pytest.raises(Exception):
        config.render_size = 1  # type: ignore[misc]
    # every calibrated threshold is exposed for the one-shot loop to inspect
    for name in (
        "tex_structure_max", "tex_chroma_max", "tex_palette_max",
        "mesh_symmetry_iou_min", "mesh_articulation_min",
        "mesh_cavity_max", "mesh_silhouette_floor",
    ):
        assert hasattr(config, name)
