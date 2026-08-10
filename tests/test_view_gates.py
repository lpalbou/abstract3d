"""Unit tests for the synthesized-view quality gates (view_gates.py).

Mechanism tests are fully synthetic (same-scale source/view pairs — the
edge-density RATIO is scale-consistent when both sides share scale).
The calibration-contract tests run against the real on-disk evidence
(e22v2's accepted-garbage views MUST reject; the bench/viewgen-fixed
clean views MUST pass) and skip, loudly, on machines without the
artifacts — the thresholds themselves are calibrated at native view
scale (768-1024 px), which downsampled fixtures cannot reproduce
(measured: LANCZOS resampling triples absolute Canny density).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from abstract3d import view_gates

REPO = Path(__file__).resolve().parents[1]
SOURCE_PHOTO = Path("/tmp/laurent_front_clean4.png")
E22V2 = REPO / "out" / "bust" / "e22_oneshot_v2"
BENCH = REPO / "out" / "bust-viewbench"
VIEWGEN_FIXED = REPO / "out" / "laurent-bust-redo" / "viewgen_fixed"

_EVIDENCE_PRESENT = SOURCE_PHOTO.exists() and E22V2.exists() and BENCH.exists()


def bust_rgba(size: int = 320, *, head_top: int = 24, shoulder: int = 150,
              bottom: int = 300) -> Image.Image:
    """Synthetic bust with smooth shading (a real-ish edge budget: the
    silhouette plus a couple of feature lines)."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    head_half = size // 6
    shoulder_half = size // 2 - 12
    for row in range(head_top, bottom + 1):
        half = head_half if row < shoulder else shoulder_half
        shade = 150 + int(40 * np.sin(row / 37.0))
        arr[row, center - half:center + half] = (shade, shade - 20, shade - 30, 255)
    # two feature lines (mouth/glasses-like) so the source edge budget > 0
    arr[head_top + 40, center - head_half:center + head_half, :3] = (40, 30, 30)
    arr[head_top + 70, center - head_half:center + head_half, :3] = (40, 30, 30)
    return Image.fromarray(arr, "RGBA")


def with_interior_tears(image: Image.Image, frac_rows: float = 0.3) -> Image.Image:
    """Carve semi-transparent streaks deep inside the subject (the e22v2
    side_left torn-alpha class)."""
    arr = np.asarray(image).copy()
    mask = arr[..., 3] > 128
    rows = np.flatnonzero(mask.any(axis=1))
    r0 = rows[0] + int(0.25 * len(rows))
    r1 = rows[0] + int((0.25 + frac_rows) * len(rows))
    cols = np.flatnonzero(mask.any(axis=0))
    c0, c1 = cols[0] + 30, cols[-1] - 30
    for col in range(c0, c1, 6):  # vertical semi-transparent drips
        arr[r0:r1, col:col + 2, 3] = 120
    return Image.fromarray(arr, "RGBA")


def with_lattice_debris(image: Image.Image) -> Image.Image:
    """Paint a dense dark lattice over the subject interior (the e22v2
    back facade-ghost class: opaque alpha, edge-soup content)."""
    arr = np.asarray(image).copy()
    mask = arr[..., 3] > 128
    rows = np.flatnonzero(mask.any(axis=1))
    grid = np.zeros(arr.shape[:2], dtype=bool)
    grid[::10, :] = True
    grid[:, ::10] = True
    sel = grid & mask
    # keep the rim intact: the debris class is INTERIOR content
    arr[sel, 0] = 30
    arr[sel, 1] = 25
    arr[sel, 2] = 25
    return Image.fromarray(arr, "RGBA")


# -- matte cleanliness: mechanism ------------------------------------------------


def test_matte_stats_interior_tears_separate_from_clean() -> None:
    clean = view_gates.matte_cleanliness_stats(bust_rgba())
    torn = view_gates.matte_cleanliness_stats(with_interior_tears(bust_rgba()))
    assert clean["interior_semi_frac"] < 0.01
    assert torn["interior_semi_frac"] > view_gates.MATTE_INTERIOR_SEMI_MAX
    assert torn["interior_semi_frac"] > 3 * clean["interior_semi_frac"]


def test_matte_gate_rejects_torn_alpha_and_lattice_debris() -> None:
    source = bust_rgba()
    gate = view_gates.build_matte_cleanliness_gate(source)

    clean = gate(bust_rgba(), label="back", azimuth_deg=180.0)
    assert clean["measured"] is True
    assert clean["passed"] is True

    torn = gate(with_interior_tears(bust_rgba()), label="side_left",
                azimuth_deg=90.0)
    assert torn["passed"] is False
    assert "torn interior alpha" in torn["reason"]

    debris = gate(with_lattice_debris(bust_rgba()), label="back",
                  azimuth_deg=180.0)
    assert debris["passed"] is False
    assert "edge density" in debris["reason"]
    assert debris["edge_density_ratio"] > view_gates.MATTE_EDGE_DENSITY_RATIO_MAX


def test_matte_gate_abstains_loudly_on_unmeasurable_view() -> None:
    gate = view_gates.build_matte_cleanliness_gate(bust_rgba())
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    verdict = gate(empty, label="back", azimuth_deg=180.0)
    assert verdict["measured"] is False
    assert verdict["passed"] is True  # abstention, never a blind rejection
    assert "matte_unmeasured" in verdict["note"]


# -- subject identity: mechanism -------------------------------------------------


def _fake_embedder(vectors):
    calls = []

    def embed(image):
        calls.append(image)
        return np.asarray(vectors[min(len(calls) - 1, len(vectors) - 1)],
                          dtype=np.float64)

    embed.calls = calls
    return embed


def test_identity_gate_passes_same_subject_and_rejects_different() -> None:
    same = _fake_embedder([[1.0, 0.0], [0.9, 0.436]])  # cos ~0.9
    gate = view_gates.build_subject_identity_gate(bust_rgba(), embedder=same)
    verdict = gate(bust_rgba(), label="side_left", azimuth_deg=90.0)
    assert verdict["measured"] is True
    assert verdict["passed"] is True
    assert verdict["cosine"] > view_gates.IDENTITY_MIN_COSINE

    different = _fake_embedder([[1.0, 0.0], [0.2, 0.98]])  # cos ~0.2
    gate = view_gates.build_subject_identity_gate(bust_rgba(), embedder=different)
    verdict = gate(bust_rgba(), label="side_left", azimuth_deg=90.0)
    assert verdict["passed"] is False
    assert "different subject" in verdict["reason"]


def test_identity_gate_abstains_beyond_face_visibility() -> None:
    embed = _fake_embedder([[1.0, 0.0]])
    gate = view_gates.build_subject_identity_gate(bust_rgba(), embedder=embed)
    verdict = gate(bust_rgba(), label="back", azimuth_deg=180.0)
    assert verdict["measured"] is False
    assert verdict["passed"] is True
    assert "identity_abstained" in verdict["note"]
    assert not embed.calls  # no embedding is paid for an abstained view


def test_identity_gate_abstains_loudly_when_embedder_unavailable() -> None:
    def broken(image):
        raise RuntimeError("model not downloaded")

    gate = view_gates.build_subject_identity_gate(bust_rgba(), embedder=broken)
    verdict = gate(bust_rgba(), label="side_left", azimuth_deg=90.0)
    assert verdict["measured"] is False
    assert verdict["passed"] is True
    assert "identity_unmeasured" in verdict["note"]
    assert "model not downloaded" in verdict["note"]


def test_identity_gate_embeds_source_once_across_candidates() -> None:
    embed = _fake_embedder([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    gate = view_gates.build_subject_identity_gate(bust_rgba(), embedder=embed)
    gate(bust_rgba(), label="side_left", azimuth_deg=90.0)
    gate(bust_rgba(), label="side_right", azimuth_deg=-90.0)
    # 1 source + 2 candidates — the source embedding is cached.
    assert len(embed.calls) == 3


def test_subject_head_crop_composites_on_neutral_gray() -> None:
    # Triangle head (narrow at the top) so the crop's rectangular corners
    # fall on transparent pixels — the compositing target.
    size = 240
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    for row in range(20, 110):
        half = 6 + int(34 * (row - 20) / 90.0)
        arr[row, center - half:center + half] = (200, 180, 160, 255)
    arr[110:220, center - 100:center + 100] = (90, 80, 70, 255)  # shoulders
    crop = view_gates.subject_head_crop(Image.fromarray(arr, "RGBA"))
    out = np.asarray(crop)
    for corner in (out[0, 0], out[0, -1]):
        assert tuple(corner) == (128, 128, 128)


# -- calibration contract on the on-disk evidence --------------------------------


needs_evidence = pytest.mark.skipif(
    not _EVIDENCE_PRESENT,
    reason="calibration evidence (out/bust/e22_oneshot_v2, out/bust-viewbench, "
           "/tmp/laurent_front_clean4.png) not present on this machine",
)


@pytest.fixture(scope="module")
def evidence_gate():
    if not _EVIDENCE_PRESENT:
        pytest.skip("no calibration evidence")
    from abstract3d.segmentation import remove_background_robust

    source = remove_background_robust(Image.open(SOURCE_PHOTO).convert("RGBA"))
    return view_gates.build_matte_cleanliness_gate(source)


@needs_evidence
@pytest.mark.parametrize("name", [
    "geometry_view_synthesized_side_left.png",
    "geometry_view_synthesized_back.png",
    "geometry_view_synthesized_side_right.png",
    "geometry_view_windowed_side_left.png",
    "geometry_view_windowed_back.png",
])
def test_matte_gate_rejects_e22v2_garbage_views(evidence_gate, name) -> None:
    view = Image.open(E22V2 / name).convert("RGBA")
    azimuth = 180.0 if "back" in name else (90.0 if "side_left" in name else -90.0)
    verdict = evidence_gate(view, label=name, azimuth_deg=azimuth)
    assert verdict["passed"] is False, verdict


@needs_evidence
@pytest.mark.parametrize("relpath,azimuth", [
    ("E/back.png", 180.0), ("E/profile_left.png", 90.0),
    ("E/profile_right.png", -90.0), ("E/side65_left.png", 65.0),
    ("E/side65_right.png", -65.0),
    ("B/back.png", 180.0), ("B/profile_left.png", 90.0),
    ("B/profile_right.png", -90.0), ("B/side65_left.png", 65.0),
    ("B/side65_right.png", -65.0),
])
def test_matte_gate_passes_bench_views(evidence_gate, relpath, azimuth) -> None:
    view = Image.open(BENCH / relpath).convert("RGBA")
    verdict = evidence_gate(view, label=relpath, azimuth_deg=azimuth)
    assert verdict["passed"] is True, verdict


@needs_evidence
@pytest.mark.parametrize("name,azimuth", [
    ("side_left.png", 90.0), ("side_right.png", -90.0), ("back.png", 180.0),
])
def test_matte_gate_passes_e20_viewgen_fixed(evidence_gate, name, azimuth) -> None:
    if not VIEWGEN_FIXED.exists():
        pytest.skip("viewgen_fixed evidence absent")
    view = Image.open(VIEWGEN_FIXED / name).convert("RGBA")
    verdict = evidence_gate(view, label=name, azimuth_deg=azimuth)
    assert verdict["passed"] is True, verdict


@needs_evidence
def test_identity_gate_separates_wrong_man_from_bench_views() -> None:
    """The C7-fallback contract on the labeled pair: e22v2's wrong-identity
    side_left rejects, the bench same-man profiles pass. Skips when the
    DINOv2 checkpoint is not in the local HF cache (never downloads in a
    test)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from abstract3d.segmentation import remove_background_robust

    source = remove_background_robust(Image.open(SOURCE_PHOTO).convert("RGBA"))
    try:
        embed = view_gates._resolve_dino_embedder()
    except Exception as exc:  # pragma: no cover - cacheless machines
        pytest.skip(f"dinov2 unavailable locally: {exc}")
    gate = view_gates.build_subject_identity_gate(source, embedder=embed)

    wrong = Image.open(
        E22V2 / "geometry_view_synthesized_side_left.png").convert("RGBA")
    verdict = gate(wrong, label="side_left", azimuth_deg=90.0)
    assert verdict["measured"] is True
    assert verdict["passed"] is False, verdict

    for relpath, azimuth in [("E/profile_left.png", 90.0),
                             ("E/side65_right.png", -65.0),
                             ("B/profile_left.png", 90.0)]:
        view = Image.open(BENCH / relpath).convert("RGBA")
        verdict = gate(view, label=relpath, azimuth_deg=azimuth)
        assert verdict["measured"] is True
        assert verdict["passed"] is True, (relpath, verdict)
