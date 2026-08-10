"""Texture-fidelity gate: the acceptance oracle for generated references.

The gate's job is asymmetric: catch MATERIAL SMOOTHING (carved relief
replaced by glaze — the shipped failure mode) while never punishing excess
texture or legitimately smooth subjects. These tests pin those properties
on synthetic images where ground truth is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from abstract3d.material_gates import texture_fidelity


def _textured(size: int = 400, amplitude: float = 40.0, seed: int = 3) -> Image.Image:
    """Mid-gray disc carrying band-limited noise in the 2-8 px relief band."""

    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    noise = rng.normal(0.0, 1.0, (size, size))
    band = gaussian_filter(noise, 1.0) - gaussian_filter(noise, 3.0)
    band = band / max(np.abs(band).std(), 1e-6)
    base = np.full((size, size), 128.0) + amplitude * band
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    for channel in range(3):
        rgba[:, :, channel] = np.clip(base, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.45) ** 2
    rgba[:, :, 3] = np.where(disc, 255, 0)
    return Image.fromarray(rgba, "RGBA")


def _smooth(size: int = 400) -> Image.Image:
    return _textured(size=size, amplitude=0.0)


def test_smoothed_generation_fails_textured_source() -> None:
    result = texture_fidelity(_smooth(), _textured())
    assert not result["passed"]
    assert not result["floor"]
    assert result["relief_ratio"] < 0.5


def test_faithful_generation_passes() -> None:
    result = texture_fidelity(_textured(seed=9), _textured(seed=3))
    assert result["passed"]


def test_smooth_source_auto_passes_any_generation() -> None:
    # A porcelain vase MUST generate smooth: no relief requirement applies.
    result = texture_fidelity(_smooth(), _smooth())
    assert result["passed"]
    assert "smooth source" in str(result.get("reason"))


def test_excess_texture_never_rejects() -> None:
    # One-sided by design: more relief than the source is not a defect.
    result = texture_fidelity(_textured(amplitude=80.0), _textured(amplitude=30.0))
    assert result["passed"]


def test_empty_matte_floor_accepts() -> None:
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    result = texture_fidelity(blank, _textured())
    assert result["passed"] and result["floor"]


# -- G2 baked speculars: dark-subject recalibration (backlog 0019) -------------
#
# Root-caused on the e22 refusal (2026-07-21): the lightness-only predicate
# conflated honestly lit SKIN on dark-dominant subjects (median L 15-25;
# skin at L 75-85 sits 60+ above it, chromatic) with true gloss (near-white,
# achromatic). Corpus measurement: plausible dark bust views' hot pixels
# carry chroma p10 17.9-26.5; gloss cores measure chroma <= ~5. The gate now
# requires hot pixels to be BOTH bright and achromatic (chroma < 14), and
# self-calibrates the pass line against the source's own near-white floor.


def _dark_subject(size: int = 400, *, patch_color=None, patch_fraction: float = 0.0,
                  base=(38, 30, 26)) -> Image.Image:
    """Dark disc (median L ~ 15) with an optional square patch of
    `patch_color` covering `patch_fraction` of the foreground."""

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.45) ** 2
    rgba[disc] = (*base, 255)
    if patch_color is not None and patch_fraction > 0:
        side = int(np.sqrt(patch_fraction * disc.sum()))
        c0 = size // 2 - side // 2
        rgba[c0:c0 + side, c0:c0 + side, :3] = patch_color
    return Image.fromarray(rgba, "RGBA")


def test_speculars_dark_subject_lit_skin_passes() -> None:
    """The 0019 failure class: a dark-median subject with a large CHROMATIC
    bright region (lit skin, RGB (224,172,140): L~75, chroma~26) must pass —
    the lightness-only gate mass-rejected exactly this (5/5 e11 back draws,
    then every e22 angle)."""
    from abstract3d.material_gates import gate_baked_speculars

    generated = _dark_subject(patch_color=(224, 172, 140), patch_fraction=0.08)
    result = gate_baked_speculars(generated)
    assert result["passed"] is True
    assert result["worst_blob_fraction"] <= 0.005


def test_speculars_gloss_field_still_rejected_pure_and_warm_white() -> None:
    """True baked speculars stay caught: near-white blobs (pure AND
    warm-tinted white — both under chroma 14) above the blob budget."""
    from abstract3d.material_gates import gate_baked_speculars

    for color in ((246, 246, 246), (250, 244, 236)):
        generated = _dark_subject(patch_color=color, patch_fraction=0.02)
        result = gate_baked_speculars(generated)
        assert result["passed"] is False, color
        assert result["worst_blob_fraction"] > 0.005


def test_speculars_source_self_calibration_raises_the_floor() -> None:
    """A subject whose SOURCE photo itself carries a legitimate near-white
    region (white lettering on a dark shirt) must not have its generations
    rejected for reproducing it: the source's own worst blob under the same
    predicate sets the pass line (x2 headroom)."""
    from abstract3d.material_gates import gate_baked_speculars

    source = _dark_subject(patch_color=(240, 240, 240), patch_fraction=0.015)
    generated = _dark_subject(patch_color=(244, 244, 244), patch_fraction=0.02)
    without_source = gate_baked_speculars(generated)
    with_source = gate_baked_speculars(generated, source=source)
    assert without_source["passed"] is False
    assert with_source["passed"] is True
    assert with_source["source_floor_blob_fraction"] >= 0.01
    # The calibration is a floor RATIO, not a blank check: a gloss field
    # far beyond the source's own floor still rejects.
    excessive = _dark_subject(patch_color=(246, 246, 246), patch_fraction=0.06)
    assert gate_baked_speculars(excessive, source=source)["passed"] is False


def test_speculars_records_calibration_numbers() -> None:
    """Failure rows must be diagnosable without a rerun (0019's ask):
    median lightness, the chroma key, and the effective pass line are on
    the record."""
    from abstract3d.material_gates import gate_baked_speculars

    result = gate_baked_speculars(
        _dark_subject(patch_color=(246, 246, 246), patch_fraction=0.02))
    assert result["median_lightness"] < 30.0
    assert result["specular_chroma_max"] == 14.0
    assert result["effective_max_blob_fraction"] == 0.005
