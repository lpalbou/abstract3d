"""Tests for abstract3d.view_consistency (same-elevation row-consistency law).

Synthetic images place horizontal bars at KNOWN heights so shifts are exact
ground truth; the real-data tests measure the actual double-mouth bundle
(guarded by existence checks) and pin the misalignment this module was built
to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

from abstract3d.errors import InvalidRequestError
from abstract3d.view_consistency import (
    CONSISTENT_MAX_SHIFT_FRAC,
    CORRECTABLE_FLOOR,
    DEFAULT_ACCEPT_FLOOR,
    align_view_rows,
    best_row_shift,
    row_alignment_score,
    row_profile,
    view_consistency_report,
)

BUNDLE_DIR = Path(__file__).resolve().parents[1] / "out" / "laurent-bust-redo" / "hunyuan_multiview"
CLAY_LEFT = BUNDLE_DIR / "texture_reference_generated_side_left_clay.png"
SYN_LEFT = BUNDLE_DIR / "geometry_view_synthesized_side_left.png"

HEIGHT, WIDTH = 240, 200
BAR_ROWS = (55, 90, 130, 170)


def make_subject(
    bar_rows=BAR_ROWS,
    *,
    bar_shift: int = 0,
    outline_shift: int = 0,
    height: int = HEIGHT,
    width: int = WIDTH,
    opaque_background: bool = False,
) -> "np.ndarray":
    """Gray torso rectangle with dark horizontal bars at known rows.

    ``bar_shift`` moves only the bars (interior features); ``outline_shift``
    moves the rectangle (silhouette). Equal values = whole-subject translation,
    the defect :func:`align_view_rows` repairs. RGBA on transparent background
    by default; ``opaque_background`` exercises the border-color mask contract.
    """
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    if opaque_background:
        arr[..., :3] = 245
        arr[..., 3] = 255
    top = max(0, 30 + outline_shift)
    bottom = min(height, height - 30 + outline_shift)
    left, right = 50, width - 50
    arr[top:bottom, left:right, :3] = 200
    arr[top:bottom, left:right, 3] = 255
    for row in bar_rows:
        shifted = row + bar_shift
        if 0 <= shifted < height - 3:
            arr[shifted : shifted + 3, left + 10 : right - 10, :3] = 40
    return arr


# ---------------------------------------------------------------------------
# row_profile
# ---------------------------------------------------------------------------


def test_row_profile_peaks_at_bar_rows() -> None:
    profile = row_profile(make_subject())
    assert profile.shape == (HEIGHT,)
    assert abs(float(profile.sum()) - 1.0) < 1e-9
    assert float(profile.min()) >= 0.0
    background = float(np.median(profile[profile > 0]))
    for row in BAR_ROWS:
        window = profile[row - 4 : row + 8]
        assert float(window.max()) > 3.0 * background, f"no peak near bar row {row}"


def test_row_profile_zero_outside_subject_rows() -> None:
    profile = row_profile(make_subject())
    assert float(profile[:20].sum()) == 0.0
    assert float(profile[-20:].sum()) == 0.0


def test_row_profile_accepts_pil_and_matches_array_input() -> None:
    arr = make_subject()
    from_array = row_profile(arr)
    from_pil = row_profile(PIL_Image.fromarray(arr, "RGBA"))
    assert np.allclose(from_array, from_pil)


def test_row_profile_opaque_background_uses_border_color_mask() -> None:
    transparent = row_profile(make_subject())
    opaque = row_profile(make_subject(opaque_background=True))
    assert row_alignment_score(opaque, transparent) > 0.9


def test_row_profile_bbox_restricts_analysis() -> None:
    arr = make_subject()
    profile = row_profile(arr, bbox=(0, 0, WIDTH, 100))
    assert float(profile[100:].sum()) == 0.0
    with pytest.raises(InvalidRequestError, match="bbox"):
        row_profile(arr, bbox=(0, 0, WIDTH + 5, 100))


def test_row_profile_rejects_tiny_and_empty_images() -> None:
    with pytest.raises(InvalidRequestError, match="too small"):
        row_profile(np.zeros((8, 8, 4), dtype=np.uint8))
    fully_transparent = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(InvalidRequestError, match="No foreground"):
        row_profile(fully_transparent)


def test_row_profile_rejects_bad_shapes_and_ranges() -> None:
    with pytest.raises(InvalidRequestError, match="shaped"):
        row_profile(np.zeros((4, 4, 7), dtype=np.uint8))
    bad = np.full((64, 64, 3), 300.0)
    with pytest.raises(InvalidRequestError, match="pixel values"):
        row_profile(bad)


# ---------------------------------------------------------------------------
# row_alignment_score / best_row_shift
# ---------------------------------------------------------------------------


def test_aligned_copies_score_near_one() -> None:
    a = row_profile(make_subject())
    b = row_profile(make_subject())
    assert row_alignment_score(a, b) > 0.99
    assert best_row_shift(a, b) == 0


def test_shifted_copies_detected_within_two_rows() -> None:
    reference = row_profile(make_subject())
    for true_shift in (-24, -7, 5, 18):
        shifted = row_profile(
            make_subject(bar_shift=true_shift, outline_shift=true_shift)
        )
        # The subject sits `true_shift` rows away; moving it back = -true_shift.
        detected = best_row_shift(shifted, reference)
        assert abs(detected - (-true_shift)) <= 2, (
            f"true {true_shift}: detected {detected}"
        )
        assert row_alignment_score(shifted, reference) < DEFAULT_ACCEPT_FLOOR


def test_score_compares_different_resolutions_in_normalized_height() -> None:
    image = PIL_Image.fromarray(make_subject(), "RGBA")
    doubled = image.resize((2 * WIDTH, 2 * HEIGHT), PIL_Image.NEAREST)
    small = row_profile(image)
    big = row_profile(doubled)
    assert small.shape[0] == HEIGHT and big.shape[0] == 2 * HEIGHT
    assert row_alignment_score(small, big) > 0.8
    assert row_alignment_score(big, small) > 0.8


def test_scrambled_rows_score_low() -> None:
    reference = row_profile(make_subject())
    scrambled_img = make_subject()[np.random.default_rng(0).permutation(HEIGHT)]
    scrambled = row_profile(scrambled_img)
    assert row_alignment_score(scrambled, reference) < 0.35


def test_profile_validation_is_loud() -> None:
    good = row_profile(make_subject())
    with pytest.raises(InvalidRequestError, match="1-D"):
        row_alignment_score(np.zeros((4, 4)), good)
    with pytest.raises(InvalidRequestError, match="non-finite"):
        row_alignment_score(np.full(64, np.nan), good)
    with pytest.raises(InvalidRequestError, match="negative"):
        row_alignment_score(np.full(64, -1.0), good)
    with pytest.raises(InvalidRequestError, match="max_shift_frac"):
        best_row_shift(good, good, max_shift_frac=0.9)


# ---------------------------------------------------------------------------
# align_view_rows
# ---------------------------------------------------------------------------


def test_align_corrects_shifted_subject_to_reference_rows() -> None:
    reference = make_subject()
    for true_shift in (18, -24, 7):
        broken = make_subject(bar_shift=true_shift, outline_shift=true_shift)
        corrected, report = align_view_rows(broken, reference)
        assert report["applied"] is True
        assert abs(report["shift_rows"] - (-true_shift)) <= 2
        assert report["score_after"] > report["score_before"]
        assert report["score_after"] > 0.95
        # Bars must land back on their true rows (within interpolation slack).
        fixed = np.asarray(corrected)
        dark = np.where(
            (fixed[:, WIDTH // 2, :3] < 100).all(axis=-1)
            & (fixed[:, WIDTH // 2, 3] > 128)
        )[0]
        assert dark.size, "corrected image lost its bars"
        assert abs(int(dark[0]) - BAR_ROWS[0]) <= 2


def test_align_reveals_transparent_rows() -> None:
    broken = make_subject(bar_shift=20, outline_shift=20)
    corrected, report = align_view_rows(broken, make_subject())
    fixed = np.asarray(corrected)
    revealed = fixed[HEIGHT + report["shift_rows"] :] if report["shift_rows"] < 0 else fixed[: report["shift_rows"]]
    assert revealed.size and int(revealed[..., 3].max()) == 0


def test_align_repairs_uniform_vertical_scale_via_warp() -> None:
    # Draw the subject through a 1/1.12 y-remap: the affine defect measured
    # on real bundles (i2i re-scaled the subject vertically).
    source = make_subject().astype(np.float64)
    center = (HEIGHT - 1) / 2.0
    rows = (np.arange(HEIGHT) - center) / 1.12 + center
    low = np.floor(rows).astype(int)
    frac = (rows - low)[:, None, None]
    ok = (rows >= 0) & (rows <= HEIGHT - 1)
    resampled = (
        source[np.clip(low, 0, HEIGHT - 1)] * (1 - frac)
        + source[np.clip(low + 1, 0, HEIGHT - 1)] * frac
    )
    resampled[~ok] = 0
    corrected, report = align_view_rows(resampled.astype(np.uint8), make_subject())
    assert report["applied"] is True
    assert report["warped"] is True
    assert abs(report["scale"] - 1.12) < 0.05
    assert report["score_after"] > 0.9


def test_align_returns_original_when_no_improvement_possible() -> None:
    aligned = make_subject()
    corrected, report = align_view_rows(aligned, make_subject())
    assert report["shift_rows"] == 0
    assert report["warped"] is False
    assert report["score_after"] >= report["score_before"] > 0.99
    assert np.array_equal(np.asarray(corrected), aligned)


def test_align_report_contract_keys() -> None:
    _, report = align_view_rows(make_subject(bar_shift=10, outline_shift=10), make_subject())
    for key in ("shift_rows", "score_before", "score_after", "warped"):
        assert key in report
    assert isinstance(report["shift_rows"], int)
    assert isinstance(report["warped"], bool)


# ---------------------------------------------------------------------------
# view_consistency_report
# ---------------------------------------------------------------------------


def test_report_consistent_for_aligned_copies() -> None:
    report = view_consistency_report(make_subject(), make_subject())
    assert report["verdict"] == "consistent"
    assert report["score"] > 0.99
    assert report["best_shift_rows"] == 0
    assert report["shift_frac"] <= CONSISTENT_MAX_SHIFT_FRAC


def test_report_correctable_for_shifted_copies() -> None:
    for true_shift in (-24, 18):
        report = view_consistency_report(
            make_subject(), make_subject(bar_shift=true_shift, outline_shift=true_shift)
        )
        assert report["verdict"] == "correctable", report
        assert report["score"] < DEFAULT_ACCEPT_FLOOR
        assert abs(report["best_shift_rows"] - (-true_shift)) <= 2
        assert report["score_at_best_remap"] >= CORRECTABLE_FLOOR


def test_report_flags_interior_only_shift_the_double_mouth_shape() -> None:
    # Outline agrees, interior features sit at the wrong height — the exact
    # double-mouth defect shape. Must NOT read as consistent.
    report = view_consistency_report(make_subject(), make_subject(bar_shift=18))
    assert report["verdict"] != "consistent"
    assert abs(report["best_shift_rows"] + 18) <= 3


def test_report_inconsistent_for_scrambled_and_unrelated_content() -> None:
    scrambled = make_subject()[np.random.default_rng(0).permutation(HEIGHT)]
    report = view_consistency_report(make_subject(), scrambled)
    assert report["verdict"] == "inconsistent"
    assert report["score"] < DEFAULT_ACCEPT_FLOOR

    unrelated = make_subject(bar_rows=(70, 123, 157, 168, 182))
    report = view_consistency_report(make_subject(), unrelated)
    assert report["verdict"] == "inconsistent", report
    assert report["score"] < DEFAULT_ACCEPT_FLOOR


def test_report_accept_floor_validation_and_override() -> None:
    with pytest.raises(InvalidRequestError, match="accept_floor"):
        view_consistency_report(make_subject(), make_subject(), accept_floor=1.5)
    strict = view_consistency_report(make_subject(), make_subject(), accept_floor=1.0)
    assert strict["accept_floor"] == 1.0


# ---------------------------------------------------------------------------
# Real failing artifacts (the double-mouth bundle) — guarded, measured, pinned
# ---------------------------------------------------------------------------

requires_bundle = pytest.mark.skipif(
    not (CLAY_LEFT.exists() and SYN_LEFT.exists()),
    reason=f"real double-mouth bundle not present under {BUNDLE_DIR}",
)


@requires_bundle
def test_real_side_left_misalignment_is_nonzero_and_correctable() -> None:
    """The double-mouth evidence: the synthesized left view's features sit
    ~10% of image height BELOW the clay guide's — a same-elevation law
    violation big enough to carve two mouths."""
    clay = PIL_Image.open(CLAY_LEFT)
    synthesized = PIL_Image.open(SYN_LEFT)
    report = view_consistency_report(clay, synthesized)

    assert report["best_shift_rows"] != 0, "misalignment vanished — bundle changed?"
    assert abs(report["best_shift_rows"]) >= 20, report
    assert report["shift_frac"] >= 0.05, report
    assert report["score"] < DEFAULT_ACCEPT_FLOOR, report
    assert report["verdict"] == "correctable", report
    # Measured 2026-07-20: shift -104 rows (-0.103 of H), score 0.00,
    # remap correlation 0.67. Direction: synthesized content must move UP.
    assert report["best_shift_rows"] < 0, report


@requires_bundle
def test_real_side_left_alignment_repair_improves_score() -> None:
    clay = PIL_Image.open(CLAY_LEFT)
    synthesized = PIL_Image.open(SYN_LEFT)
    corrected, report = align_view_rows(synthesized, clay)
    assert report["applied"] is True
    assert report["shift_rows"] < 0
    assert report["score_after"] > report["score_before"]
    post = view_consistency_report(clay, corrected)
    assert abs(post["best_shift_rows"]) < abs(
        view_consistency_report(clay, synthesized)["best_shift_rows"]
    )


@requires_bundle
def test_real_aligned_texture_reference_reads_consistent() -> None:
    """Contrast case from the same bundle: the texture reference for the same
    angle IS row-aligned with its clay guide and must not be flagged."""
    clay = PIL_Image.open(CLAY_LEFT)
    texture_ref = PIL_Image.open(BUNDLE_DIR / "texture_reference_generated_side_left.png")
    report = view_consistency_report(clay, texture_ref)
    assert report["verdict"] == "consistent", report
    assert report["shift_frac"] <= CONSISTENT_MAX_SHIFT_FRAC, report


@requires_bundle
def test_real_all_synthesized_geometry_views_flagged() -> None:
    """Every synthesized geometry view in the failing bundle violates the row
    law (measured shifts 0.07-0.15 of height); all must be caught, none
    accepted as consistent."""
    for angle in ("side_left", "side_right", "back"):
        clay_path = BUNDLE_DIR / f"texture_reference_generated_{angle}_clay.png"
        syn_path = BUNDLE_DIR / f"geometry_view_synthesized_{angle}.png"
        if not (clay_path.exists() and syn_path.exists()):
            pytest.skip(f"bundle file missing for angle {angle}")
        report = view_consistency_report(PIL_Image.open(clay_path), PIL_Image.open(syn_path))
        assert report["verdict"] != "consistent", (angle, report)
        assert abs(report["best_shift_rows"]) >= 20, (angle, report)
