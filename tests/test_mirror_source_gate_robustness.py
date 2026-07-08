"""Regression: the mirror-source confidence gate must degrade gracefully.

CONFIRMED defect (texture-cycle critic 2, 2026-07-05): in
`mirror_fill_from_observed`, when the confident source population
(weight >= min_source_weight) drops below the hard-coded 500-texel floor,
the implementation silently falls back to ALL observed texels — abandoning
the gate exactly when confidence is scarcest, i.e. admitting the grazing
rim samples the gate exists to exclude (the population documented to have
fabricated a bright skin patch on the hidden crown).

Measured on the starship at 1024 (instrumented bake, /tmp/critic2/qa/
pool_{snapshot,current}.log): a small upstream weight rescale moved the
confident count 570 -> 430 across the floor and mirror coverage jumped
0.0225 -> 0.2065 of the atlas (9x) with no code change in this function —
a discontinuous cliff in bake behavior w.r.t. continuous weight changes.

The tests below pin the intended robustness properties. Fixed by
graceful degradation (top-up from the best-weighted remaining texels at
>= half the threshold, never junk; empty fill when no credible anchors
exist); these are now hard regression guards.
"""
from __future__ import annotations

import numpy as np

from abstract3d.texturing import mirror_fill_from_observed


def _plane_atlas(n: int = 96):
    """Flat two-sided test surface, mirror-symmetric across y=0.

    Left half (y < 0): observed. Right half (y > 0): unseen. Every unseen
    texel has an exact observed twin, so the ONLY thing controlling fill
    extent is the source gate.
    """
    positions = np.zeros((n, n, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    # y in [-1, 1) across columns; x rows in [0, 1); z = 0
    positions[:, :, 1] = (xs / n) * 2.0 - 1.0
    positions[:, :, 0] = ys / n
    positions[:, :, 3] = 1.0
    observed = positions[:, :, 1] < 0.0
    colors = np.zeros((n, n, 3), dtype=np.float32)
    colors[observed] = 0.6
    return positions, observed, colors


def _fill_count(weights, positions, observed, colors):
    _, mask = mirror_fill_from_observed(
        positions_texture=positions,
        observed_mask=observed,
        colors_rgb=colors,
        observed_weight=weights,
        min_source_weight=0.35,
        consensus_guard=False,
    )
    return int(mask.sum())


def test_mirror_fill_extent_is_continuous_across_the_source_floor():
    """Shrinking the confident set by ONE texel must not explode the fill.

    501 confident sources -> gate active. 499 confident sources (one texel's
    weight nudged below threshold) -> today the gate collapses and the fill
    jumps by an order of magnitude. A robust gate changes the fill by O(1)
    texels for an O(1) source change.
    """
    positions, observed, colors = _plane_atlas(96)
    rows, cols = np.nonzero(observed)

    def weights_with_confident(k):
        w = np.zeros(observed.shape, dtype=np.float32)
        w[observed] = 0.05  # grazing junk weight: must never source the fill
        w[rows[:k], cols[:k]] = 0.9
        return w

    fill_above = _fill_count(weights_with_confident(501), positions, observed, colors)
    fill_below = _fill_count(weights_with_confident(499), positions, observed, colors)
    # Robustness property: an O(1) change of the source set produces an
    # O(1) change of the fill, not a regime flip.
    assert abs(fill_below - fill_above) <= 50, (
        f"source-floor cliff: fill went {fill_above} -> {fill_below} when the "
        "confident population crossed the hard floor by 2 texels"
    )


def test_mirror_fill_never_sources_from_far_below_threshold_weights():
    """With 400 confident texels (< floor) plus a sea of 0.05-weight junk,
    the fill must be anchored by the confident texels (possibly fewer
    accepted twins), NEVER by the junk: junk-only regions must stay empty.
    """
    positions, observed, colors = _plane_atlas(96)
    rows, cols = np.nonzero(observed)
    w = np.zeros(observed.shape, dtype=np.float32)
    w[observed] = 0.05
    # confident block confined to the first 8 rows
    top = rows < 8
    w[rows[top][:400], cols[top][:400]] = 0.9
    colors2 = colors.copy()
    colors2[(w > 0) & (w < 0.35)] = 0.1  # junk texels carry a distinct color

    fill_rgb, mask = mirror_fill_from_observed(
        positions_texture=positions,
        observed_mask=observed,
        colors_rgb=colors2,
        observed_weight=w,
        min_source_weight=0.35,
        consensus_guard=False,
    )
    if mask.any():
        filled_colors = fill_rgb[mask]
        junk_sourced = (np.abs(filled_colors - 0.1) < 1e-3).all(axis=1)
        assert junk_sourced.mean() < 0.05, (
            f"{junk_sourced.mean():.0%} of mirror-filled texels were sourced "
            "from weights an order of magnitude below min_source_weight"
        )
