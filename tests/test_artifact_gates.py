"""Tests for the measured artifact-detector battery.

Corpus calibration lives in the fix-program report (/tmp/afix2; margins
in the module docstring and CHANGELOG). These tests pin the detector
MECHANICS on synthetic fixtures per artifact class:

- a stamped foreign pale blob fires the blotch detector; clean subjects
  and elongated bright strips (tail-light bands, sills) do not,
- photo-background classification only activates when the raw photo's
  backdrop is recoverable (never on matted RGBA),
- translucent axis-aligned rectangles count as patch-block cells;
  organic shapes do not,
- mid-band camouflage registers in the patchwork metric,
- the A/B battery punishes only ADDED artifacts (direction),
- the stats-based class 3/6 checks fire on the recorded incident values
  and stay quiet on healthy-fleet values (and degrade gracefully when
  stats are missing).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from abstract3d.artifact_gates import (
    ADDED_BLOTCH_BUDGET,
    evaluate_artifact_battery_ab,
    evaluate_bundle_artifact_battery,
    fill_cap_mottle_risk,
    measure_render_artifacts,
    photo_reference,
    registration_floor_check,
)

BACKGROUND = (242, 242, 237)  # the offscreen renderer's constant bg


def render_like(size: int = 512, seed: int = 3) -> np.ndarray:
    """A synthetic 'render': red-noise disc subject on the renderer bg."""

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :] = BACKGROUND
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (0.42 * size) ** 2
    body = np.zeros((size, size, 3), np.uint8)
    body[:, :, 0] = rng.integers(150, 200, (size, size))
    body[:, :, 1] = rng.integers(20, 60, (size, size))
    body[:, :, 2] = rng.integers(20, 60, (size, size))
    img[disc] = body[disc]
    return img


def test_pale_blotch_fires_on_stamped_blob() -> None:
    clean = render_like()
    stamped = clean.copy()
    # a compact pale desaturated blob (the stamp payload / rim splash)
    stamped[200:260, 210:280] = (231, 228, 225)
    clean_row = measure_render_artifacts(Image.fromarray(clean))
    stamped_row = measure_render_artifacts(Image.fromarray(stamped))
    assert clean_row["pale_blotch"]["total_frac"] == 0.0
    assert stamped_row["pale_blotch"]["max_component_frac"] > 0.005
    assert stamped_row["pale_blotch"]["count"] == 1


def test_pale_blotch_ignores_elongated_strip() -> None:
    """Bright desaturated STRIPS are legitimate vehicle features
    (tail-light bands, rocker sills - measured elongations 8-18 on the
    accept corpus); only compact-to-oval blobs may vote."""

    strip = render_like()
    strip[250:262, 120:400] = (231, 228, 225)  # elongation ~ 23
    row = measure_render_artifacts(Image.fromarray(strip))
    assert row["pale_blotch"]["total_frac"] == 0.0
    assert row["pale_blotch"]["elongated_frac"] > 0.005


def test_photo_background_classification() -> None:
    # matted RGBA: backdrop destroyed -> no bg reference
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[16:48, 16:48] = (200, 40, 40, 255)
    assert photo_reference(Image.fromarray(rgba, "RGBA"))["bg"] is None

    # raw RGB photo: border median recovers the studio backdrop
    photo = np.zeros((256, 256, 3), np.uint8)
    photo[:, :] = (228, 226, 224)
    photo[64:192, 64:192] = (190, 30, 30)
    ref = photo_reference(Image.fromarray(photo))
    assert ref["bg"] is not None

    # a blob near the backdrop color classifies as background matter
    stamped = render_like()
    stamped[200:260, 210:280] = (231, 228, 225)
    row = measure_render_artifacts(Image.fromarray(stamped), ref)
    assert row["pale_blotch"]["bg_frac"] is not None
    assert row["pale_blotch"]["bg_frac"] > 0.005


def test_rect_haze_cells_count_translucent_rectangles() -> None:
    blocked = render_like().astype(np.float32)
    # three translucent pale rectangles (patch-block cells), alpha 0.35
    overlay = np.array([200.0, 190.0, 185.0])
    for y0, x0 in ((150, 150), (150, 240), (220, 190)):
        region = blocked[y0:y0 + 45, x0:x0 + 60]
        blocked[y0:y0 + 45, x0:x0 + 60] = 0.65 * region + 0.35 * overlay
    row = measure_render_artifacts(Image.fromarray(
        np.clip(blocked, 0, 255).astype(np.uint8)))
    assert row["rect_haze"]["cells"] >= 2

    organic = render_like().astype(np.float32)
    yy, xx = np.mgrid[0:512, 0:512]
    # a diamond (bbox fill 0.5) of the same translucent haze
    diamond = (np.abs(yy - 200) + np.abs(xx - 230)) <= 40
    organic[diamond] = 0.65 * organic[diamond] + 0.35 * overlay
    row = measure_render_artifacts(Image.fromarray(
        np.clip(organic, 0, 255).astype(np.uint8)))
    assert row["rect_haze"]["cells"] == 0


def test_dark_patchwork_measures_camouflage() -> None:
    rng = np.random.default_rng(9)
    mottled = render_like().astype(np.float32)
    for _ in range(60):
        cy = int(rng.integers(140, 380))
        cx = int(rng.integers(140, 380))
        r = int(rng.integers(8, 18))
        yy, xx = np.mgrid[0:512, 0:512]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mottled[blob] *= 0.55
    clean_row = measure_render_artifacts(Image.fromarray(render_like()))
    mottled_row = measure_render_artifacts(Image.fromarray(
        np.clip(mottled, 0, 255).astype(np.uint8)))
    assert (mottled_row["dark_patchwork"]["patch_frac"]
            > clean_row["dark_patchwork"]["patch_frac"] + 0.01)


def test_ab_battery_punishes_only_added_artifacts() -> None:
    clean = Image.fromarray(render_like())
    stamped_arr = render_like()
    stamped_arr[200:260, 210:280] = (231, 228, 225)
    stamped = Image.fromarray(stamped_arr)

    # candidate ADDS the blob -> vote fires
    report = evaluate_artifact_battery_ab(
        [("v", stamped)], [("v", clean)])
    assert report["metrics"]["added_pale_blotch"]["worst"] > ADDED_BLOTCH_BUDGET
    assert any("pale blotch" in reason for reason in report["reasons"])

    # baseline carries the same blob -> inherited, not added
    report = evaluate_artifact_battery_ab(
        [("v", stamped)], [("v", stamped)])
    assert report["reasons"] == []

    # candidate REMOVES the blob -> improvement, never punished
    report = evaluate_artifact_battery_ab(
        [("v", clean)], [("v", stamped)])
    assert report["reasons"] == []
    assert report["metrics"]["added_pale_blotch"]["worst"] == 0.0


def test_standalone_battery_warns_and_never_fails() -> None:
    stamped_arr = render_like()
    stamped_arr[170:240, 180:260] = (231, 228, 225)
    stamped_arr[250:310, 260:330] = (231, 228, 225)
    report = evaluate_bundle_artifact_battery(
        [("v", Image.fromarray(stamped_arr))])
    assert any("blotch_total_frac" in warning
               for warning in report["warnings"])
    # warn-only contract: the report never carries refusal reasons
    assert "reasons" not in report

    quiet = evaluate_bundle_artifact_battery(
        [("v", Image.fromarray(render_like()))])
    assert quiet["warnings"] == []


def test_registration_floor_check_fires_on_incident_values() -> None:
    incident = {"observed_view_stats": [
        {"index": 1, "label": "front", "coverage_ratio": 0.0498,
         "capture_efficiency": 0.1663}]}
    assert registration_floor_check(incident)["fired"] is True

    healthy = {"observed_view_stats": [
        {"index": 1, "label": "front", "coverage_ratio": 0.1088,
         "capture_efficiency": 0.2944}]}  # the weakest good bundle (v7)
    assert registration_floor_check(healthy)["fired"] is False

    missing = registration_floor_check({"observed_view_stats": [
        {"index": 1, "label": "front", "coverage_ratio": 0.05}]})
    assert missing["available"] is False
    assert missing["fired"] is False


def test_fill_cap_mottle_risk_fires_on_capped_fill() -> None:
    capped = {"fill_detail": {"energy_calibration": {"scale": 3.0}},
              "leverage": {"unobservable_ratio": 0.8388}}
    assert fill_cap_mottle_risk(capped)["fired"] is True

    healthy = {"fill_detail": {"energy_calibration": {"scale": 1.109}},
               "leverage": {"unobservable_ratio": 0.8388}}
    assert fill_cap_mottle_risk(healthy)["fired"] is False

    low_fill = {"fill_detail": {"energy_calibration": {"scale": 3.0}},
                "leverage": {"unobservable_ratio": 0.17}}
    assert fill_cap_mottle_risk(low_fill)["fired"] is False

    assert fill_cap_mottle_risk({})["available"] is False
