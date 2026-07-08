"""Capability-boundary regression tests for `reconcile_shadow_aprons`.

From the cycle-5 mathematical review (critic 2); constructions in
/tmp/critic2/tools/attack_c5_m1.py. Two properties pinned:

1. GAUGE ABSORPTION (hard guard): a pure global exposure difference
   between the source and a reference must NOT fire the mechanism — the
   pairwise gauge is measured from co-witnessed medians and must absorb
   it. If a future edit replaces the measured gauge with a constant,
   this test fails.

2. SMOOTH-OCCLUDER BOUNDARY (xfail, documented limit): a smooth dark
   foreground occluder in the SOURCE photo (out-of-focus object between
   the camera and co-witnessed bright surface) is indistinguishable from
   a cast shadow by every signal the mechanism has (smooth, dark,
   source-valid, beyond gauge). It fires and darkens the composite
   toward the occluder's tone. This is the source-authority doctrine's
   accepted trade (the same class as printing the photo's baked cast
   shadow); the test documents the boundary and flips to a guard only
   if a discriminating signal is ever added.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstract3d.gradient_compositing import reconcile_shadow_aprons


def _scene(n=512, src_dark_fn=None, exposure_shift=0.0):
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    base = np.full((n, n), 0.62, np.float32)
    src_lum = base if src_dark_fn is None else src_dark_fn(base, ys, xs)
    ref_lum = np.clip(base * np.exp(exposure_shift), 0, 1)

    def rgb_of(lum):
        return np.stack([lum, lum * 0.95, lum * 0.9], axis=2)

    positions = np.zeros((n, n, 4), np.float32)
    positions[:, :, 0] = xs * 0.01
    positions[:, :, 1] = ys * 0.01
    positions[:, :, 3] = 1.0
    return dict(
        view_rgb=[rgb_of(src_lum), rgb_of(ref_lum)],
        view_weight=[np.full((n, n), 0.05, np.float32),
                     np.full((n, n), 0.60, np.float32)],
        view_valid=[np.ones((n, n), bool), np.ones((n, n), bool)],
        observed_mask=np.ones((n, n), bool),
        positions=positions,
    )


def test_pure_exposure_shift_does_not_fire():
    """A +0.20-log brighter reference is a lighting/exposure difference,
    not a shadow: the measured pairwise gauge must absorb it entirely."""
    result = reconcile_shadow_aprons(**_scene(exposure_shift=+0.20))
    assert result is None


@pytest.mark.xfail(
    strict=False,
    reason="documented boundary: a smooth dark source-side occluder is "
    "indistinguishable from a cast shadow (source-authority doctrine); "
    "no discriminating signal exists in a single source photo",
)
def test_smooth_foreground_occluder_is_not_printed():
    def occluder(base, ys, xs):
        r2 = ((ys - 256) ** 2 + (xs - 256) ** 2) / (60.0 ** 2)
        return base * (1.0 - 0.45 * np.exp(-0.5 * r2))

    result = reconcile_shadow_aprons(**_scene(src_dark_fn=occluder))
    assert result is None, (
        "the mechanism darkened co-witnessed surface toward a source-side "
        "foreground occluder"
    )
