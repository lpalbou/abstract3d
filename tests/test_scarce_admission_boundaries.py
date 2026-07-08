"""Capability-boundary regression tests for `admit_scarce_witnesses`.

From the cycle-7 mathematical review (critic 2); constructions in
/tmp/critic2/tools/attack_c7_g1.py. Two properties pinned:

1. HARD GUARD: a dark boundary-displaced claim on a bright confident
   surround must stay refused (rules 2+3 — the flake class). This is the
   guard the cycle-7 iteration ladder measured as load-bearing; if a
   future edit weakens the like-material or dark-mass rules, this fails.

2. DOCUMENTED BOUNDARY (xfail): on a UNIFORM-MATERIAL subject the
   like-material majority rule is vacuously satisfied and the
   contradiction rule only refuses deviations beyond
   `consensus_contrast` (0.22 mean-RGB): same-material structure at
   sub-threshold contrast (e.g. 0.15) that grazing views deposit
   DISPLACED is admitted — measured 67% admission on both bright and
   dark uniform subjects. Bounded by the stretch cap (displacement
   <= ~1% of mesh scale) and the render battery downstream; documented,
   not currently distinguishable from legitimate content by any signal
   the mechanism has.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstract3d.texturing import admit_scarce_witnesses


def _scene(n=512, base_lum=0.62, candidate_fn=None):
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    positions = np.zeros((n, n, 4), np.float32)
    positions[:, :, 0] = xs * 0.01
    positions[:, :, 1] = ys * 0.01
    positions[:, :, 3] = 1.0
    surface = positions[:, :, 3] > 0

    conf_lum = np.full((n, n), base_lum, np.float32)
    orphan = (xs >= 300) & (xs < 360)
    weight = np.where(orphan, 0.0, 0.8).astype(np.float32)
    cand_lum = conf_lum if candidate_fn is None else candidate_fn(conf_lum, ys, xs)
    rgb = np.stack([np.where(orphan, cand_lum, conf_lum)] * 3, axis=2)
    rgba = np.concatenate(
        [rgb, (weight > 0).astype(np.float32)[:, :, None]], axis=2)
    scarce = np.where(orphan, 0.12, 0.0).astype(np.float32)
    projection = {"label": "v0", "rgba": rgba, "weight": weight,
                  "scarce_weight": scarce}
    return [projection], surface, positions, orphan


def test_dark_boundary_claim_on_bright_surround_stays_refused():
    projections, surface, positions, orphan = _scene(
        candidate_fn=lambda l, ys, xs: np.full_like(l, 0.13))
    stats = admit_scarce_witnesses(
        projections, surface_mask=surface, positions_texture=positions)
    assert stats["admitted_texels"] == 0


@pytest.mark.xfail(
    strict=False,
    reason="documented boundary: uniform-material subjects vacuously "
    "satisfy the like-material rule; sub-contrast displaced structure "
    "is admitted (bounded by the stretch cap and render battery)",
)
def test_uniform_material_does_not_admit_displaced_structure():
    projections, surface, positions, orphan = _scene(
        candidate_fn=lambda l, ys, xs: np.where((ys // 8) % 2 == 0,
                                                l - 0.15, l))
    stats = admit_scarce_witnesses(
        projections, surface_mask=surface, positions_texture=positions)
    assert stats["admitted_texels"] == 0
