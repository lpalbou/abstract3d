"""Capability-boundary regression tests for film-band commitment.

Findings from the cycle-2 mathematical review (critic 2). Constructions in
/tmp/critic2/tools/attack_a1_filmband.py, summarized:

1. THIN BRIGHT LEGITIMATE CONTENT inside a committed dark band (a glossy
   hair sheen a few texels wide) is vacated by every witness and re-toned
   dark: the base-witness veto cannot engage (every view that images the
   band flags it — there is no non-flagging witness), and dark-dominance
   cannot block (the sheen is far narrower than the dominance ball).
   Measured: sheen luminance 0.62 -> 0.21 with two views. This is the
   documented boundary of the bright-vacate design: content-level
   protection only exists at or above the dominance-ball scale
   (~1-2% of the mesh diagonal).

2. DARK-MAJORITY DEGENERATE SPLIT: when the dark material covers roughly
   half or more of the claimed area, the "median of the bright half"
   statistic collapses onto the dark mode, no texel classifies as dark,
   and the mechanism silently no-ops (no vacate, no retone). That is the
   SAFE direction, and this test pins it: any future rework of the
   two-mode split must keep the degenerate regime non-destructive.

3. WITNESS CORRELATION: the >=2-imaging-witness consensus counts a
   mirrored photo pair (the fabricated right profile is the left profile
   flipped) as two independent witnesses. Measured on the face proof
   asset: 29% of committed texels are witnessed by the profile pair
   alone. Not asserted here (needs a full bake); recorded in the review.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstract3d.film_band import commit_film_band, retone_film_band


def _scene(n=1024, dark_rows=(400, 800), sheen_rows=(480, 486)):
    """Flat plane atlas; two identical views (the mirrored-photo analog)
    claim everything at weight 0.8; both flag the dark band as commit
    zone with first-surface imaging everywhere and no veto rows."""
    positions = np.zeros((n, n, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    positions[:, :, 0] = xs * 0.01
    positions[:, :, 1] = ys * 0.01
    positions[:, :, 3] = 1.0
    surface = positions[:, :, 3] > 0

    albedo = np.full((n, n, 3), 0.72, dtype=np.float32)
    r0, r1 = dark_rows
    albedo[r0:r1] = 0.13
    s0, s1 = sheen_rows
    albedo[s0:s1] = 0.62

    zone = np.zeros((n, n), dtype=bool)
    zone[r0:r1] = True

    def view():
        rgba = np.concatenate([albedo, np.ones((n, n, 1), np.float32)], axis=2)
        return {
            "rgba": rgba.copy(),
            "weight": np.full((n, n), 0.8, dtype=np.float32),
            "film_band": {
                "zone_texel": zone.copy(),
                "added_texel": zone.copy(),
                "commit_texel": zone.copy(),
                "veto_texel": np.zeros((n, n), dtype=bool),
                "img_first_texel": np.ones((n, n), dtype=bool),
                "contested_texel": zone.copy(),
                "body_weight_texel": np.where(zone, 1.0, 0.0).astype(np.float32),
            },
        }

    return positions, surface, albedo, [view(), view()], (s0, s1)


@pytest.mark.xfail(
    strict=False,
    reason="known boundary: sub-ball-scale bright legitimate content inside "
    "a committed dark band is vacated and re-toned dark (glossy hair "
    "sheen class); protection starts at the dominance-ball scale",
)
def test_film_band_keeps_thin_bright_sheen():
    positions, surface, albedo, projections, (s0, s1) = _scene()
    state = commit_film_band(
        projections, surface_mask=surface, positions_texture=positions)
    assert state is not None
    sheen = np.zeros(surface.shape, dtype=bool)
    sheen[s0:s1] = True

    # the sheen's claims must survive commitment...
    for projection in projections:
        weight = np.asarray(projection["weight"])
        assert (weight[sheen] > 0).all(), "sheen claims vacated"

    # ...and the final tone must stay bright after retone
    weight_stack = np.stack([np.asarray(p["weight"]) for p in projections])
    observed = weight_stack.max(axis=0) > 0
    colors = np.concatenate(
        [albedo, np.ones((*albedo.shape[:2], 1), np.float32)], axis=2)
    out, _ = retone_film_band(
        colors, positions_texture=positions, observed_mask=observed,
        commit_mask=state["commit_mask"], body_weight=state["body_weight"])
    assert out[s0:s1, :, :3].mean() > 0.75 * albedo[s0:s1].mean()


def test_film_band_dark_majority_fails_safe():
    """Degenerate two-mode split (dark >= ~50% of claimed area) must stay a
    NO-OP: zero vacated claims and no retone. Pins the safe direction of
    the degeneracy found in review; a future split rework (e.g. Otsu) that
    starts engaging here must revisit the sheen boundary above too."""
    positions, surface, albedo, projections, _ = _scene(
        dark_rows=(240, 800), sheen_rows=(480, 486))  # dark band ~55%
    state = commit_film_band(
        projections, surface_mask=surface, positions_texture=positions)
    assert state is not None
    assert state["stats"]["vacated_claims"] == 0

    weight_stack = np.stack([np.asarray(p["weight"]) for p in projections])
    observed = weight_stack.max(axis=0) > 0
    colors = np.concatenate(
        [albedo, np.ones((*albedo.shape[:2], 1), np.float32)], axis=2)
    out, stats = retone_film_band(
        colors, positions_texture=positions, observed_mask=observed,
        commit_mask=state["commit_mask"], body_weight=state["body_weight"])
    assert stats["applied"] is False
    assert np.array_equal(out, colors)
