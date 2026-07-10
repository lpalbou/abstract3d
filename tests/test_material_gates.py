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
