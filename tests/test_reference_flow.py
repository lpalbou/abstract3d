"""Tests for the dense reference-to-source residual flow (reference_flow).

The estimator's contract, each clause anchored to a measured failure mode
from the face lane (see module docstring):

  1. injected known warps are recovered inside the evidence band;
  2. displacement is identically zero outside the validated evidence
     region (global extension damaged far turf twice);
  3. unreachable targets (content that cannot match) are rejected rather
     than warped toward;
  4. the acceptance gate returns the input image untouched when there is
     no overlap evidence at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstract3d.reference_flow import (
    _bilinear_sample,
    estimate_reference_flow,
    masked_error,
    solve_lattice_flow,
    _validation_gate,
)


def _textured_pattern(size: int, seed: int = 7) -> np.ndarray:
    """Smooth random pattern with feature-scale structure (skin + edges)."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(rng.random((size, size, 3)).astype(np.float32), (9, 9, 0))
    detail = gaussian_filter(rng.random((size, size, 3)).astype(np.float32), (2.5, 2.5, 0))
    image = 0.55 * base + 0.45 * detail
    image -= image.min()
    image /= image.max()
    return image.astype(np.float32)


def _band_weight(size: int) -> np.ndarray:
    """Vertical evidence band akin to the real overlap (front-face strip)."""
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                         indexing="ij")
    weight = np.exp(-((xx - 0.55 * size) ** 2) / (2 * (0.14 * size) ** 2))
    weight *= ((yy > 0.15 * size) & (yy < 0.88 * size)).astype(np.float32)
    return np.where(weight > 0.15, weight, 0.0).astype(np.float32)


def _grids(size: int):
    return np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                       indexing="ij")


def test_lattice_flow_recovers_injected_local_warp() -> None:
    """A localized bump warp inside the band is recovered to sub-pixel
    median accuracy, and the recovered warp restores the image agreement."""
    size = 256
    image = _textured_pattern(size)
    weight = _band_weight(size)
    yy, xx = _grids(size)

    bump = 5.0 * np.exp(-(((xx - 0.55 * size) ** 2 + (yy - 0.5 * size) ** 2)
                          / (2 * (0.08 * size) ** 2)))
    gt_x = np.zeros((size, size), np.float32)
    gt_y = bump.astype(np.float32)
    warped = _bilinear_sample(image, xx + gt_x, yy + gt_y)

    solved = solve_lattice_flow(warped, image, weight, cap_px=0.03 * size)
    # recovery target: the solver looks up content, so D must reproduce the
    # injected forward warp applied to the observation grid
    band = weight > 0
    error = np.hypot(solved["flow_x"] - gt_x, solved["flow_y"] - gt_y)
    # inverse-vs-forward differ by O(|g|^2 * curvature) << 1 px at 5 px bumps
    assert float(np.median(error[band])) <= 1.5

    err_before = masked_error(warped, image, weight)
    err_after = masked_error(warped, image, weight, solved["flow_x"], solved["flow_y"])
    assert err_after < 0.5 * err_before


def test_validation_gate_zeroes_flow_outside_evidence() -> None:
    """Whatever the lattice proposes, the gate keeps displacement at zero
    beyond its designed reach around the evidence band (one 48 px
    validation cell + one 48 px leash ring + the 12 px blur skirt): hair
    and far-side territory must never move."""
    size = 384
    image = _textured_pattern(size)
    yy, xx = _grids(size)
    # narrow band on the left third so the far side has real distance
    weight = np.exp(-((xx - 0.30 * size) ** 2) / (2 * (0.08 * size) ** 2)).astype(np.float32)
    weight *= ((yy > 0.15 * size) & (yy < 0.85 * size)).astype(np.float32)
    weight = np.where(weight > 0.15, weight, 0.0).astype(np.float32)

    gt_y = np.full((size, size), 4.0, np.float32)
    warped = _bilinear_sample(image, xx, yy + gt_y)

    solved = solve_lattice_flow(warped, image, weight, cap_px=0.03 * size)
    gate, stats = _validation_gate(warped, image, weight, solved["flow_x"], solved["flow_y"])
    assert stats["cells_kept"] > 0
    flow_x = solved["flow_x"] * gate
    flow_y = solved["flow_y"] * gate

    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(weight <= 0)
    outside = distance > (48 + 48 + 3 * 12)
    assert outside.any()
    magnitude = np.hypot(flow_x, flow_y)
    assert float(magnitude[outside].max()) < 0.05


def test_validation_gate_rejects_unreachable_content() -> None:
    """Where the target is content the reference cannot match (independent
    random pattern), cells must NOT validate: moving pixels toward an
    unreachable target only drags boundaries around."""
    size = 256
    reference = _textured_pattern(size, seed=7)
    unreachable = _textured_pattern(size, seed=99)  # different content
    weight = _band_weight(size)

    solved = solve_lattice_flow(reference, unreachable, weight, cap_px=0.03 * size)
    gate, stats = _validation_gate(reference, unreachable, weight,
                                   solved["flow_x"], solved["flow_y"])
    total_band_gate = float(gate[weight > 0].mean())
    assert stats["cells_kept"] <= 2
    assert total_band_gate < 0.2


def test_estimate_reference_flow_returns_input_without_overlap() -> None:
    """No source evidence -> the image is returned untouched (identity)."""
    from PIL import Image

    size = 128
    rng = np.random.default_rng(3)
    rgba = (rng.random((size, size, 4)) * 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    image = Image.fromarray(rgba, "RGBA")

    positions = np.zeros((64, 64, 4), np.float32)  # alpha 0: no surface
    source_projection = {
        "rgba": np.zeros((64, 64, 4), np.float32),
        "weight": np.zeros((64, 64), np.float32),
    }
    out, stats = estimate_reference_flow(
        image,
        positions_texture=positions,
        source_projection=source_projection,
        azimuth_deg=90.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        ortho_half_extent=1.0,
    )
    assert stats["applied"] is False
    assert np.array_equal(np.asarray(out), np.asarray(image))
