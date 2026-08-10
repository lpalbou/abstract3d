"""Tests for scripts/bust_assessment.py.

Fast unit coverage for the double-feature (duplication) detector on
synthetic images, plus the scorecard JSON shape on a tiny box GLB fixture.
Render-dependent tests skip when neither moderngl nor matplotlib is
available (via importorskip); the detector tests need only numpy + Pillow.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bust_assessment.py"
_SPEC = importlib.util.spec_from_file_location("bust_assessment", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bust_assessment = importlib.util.module_from_spec(_SPEC)
# Register before exec: the script's dataclasses resolve string annotations
# through sys.modules[module.__name__] (from __future__ import annotations).
sys.modules[_SPEC.name] = bust_assessment
_SPEC.loader.exec_module(bust_assessment)


def _require_renderer() -> None:
    """Skip render-dependent tests when no preview renderer is installed."""
    try:
        import moderngl  # noqa: F401

        return
    except Exception:
        pass
    pytest.importorskip("matplotlib", reason="renders need moderngl or matplotlib")


# ---------------------------------------------------------------------------
# Synthetic fixtures for the duplication detector.
# ---------------------------------------------------------------------------

_BG = (242, 242, 237)  # matches the renderer's clear color family
_FILL = (150, 150, 150)
_BAR = (60, 60, 60)

_SUBJECT_ROWS = (60, 460)  # subject height 400 -> lag window 12..60 px
_SUBJECT_COLS = (156, 356)
_BAR_COLS = (200, 312)
_BAR_THICKNESS = 6  # < 3% of subject height, so a bar's own two edges
# (top + bottom, thickness apart) fall below the search window


def _synthetic_clay(bar_top_rows: list[int]) -> Image.Image:
    """Clay-like render: uniform subject rectangle with dark horizontal bars."""
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[:, :] = _BG
    array[_SUBJECT_ROWS[0] : _SUBJECT_ROWS[1], _SUBJECT_COLS[0] : _SUBJECT_COLS[1]] = _FILL
    for top in bar_top_rows:
        array[top : top + _BAR_THICKNESS, _BAR_COLS[0] : _BAR_COLS[1]] = _BAR
    return Image.fromarray(array, mode="RGB")


def test_detector_single_bar_is_not_suspect() -> None:
    image = _synthetic_clay(bar_top_rows=[140])

    report = bust_assessment.analyze_duplication(image)

    assert report["duplication_suspect"] is False
    # A single feature must not produce a strong non-zero-lag peak.
    if report["peak_ratio"] is not None:
        assert report["peak_ratio"] < report["threshold"]


def test_detector_doubled_bar_fires_at_known_offset() -> None:
    offset = 40  # 10% of the 400px subject height -> inside the 3-15% window
    image = _synthetic_clay(bar_top_rows=[140, 140 + offset])

    report = bust_assessment.analyze_duplication(image)

    assert report["duplication_suspect"] is True
    assert report["peak_ratio"] >= report["threshold"]
    assert report["peak_lag_px"] is not None
    assert abs(report["peak_lag_px"] - offset) <= 2
    fraction = report["peak_lag_fraction_of_subject_height"]
    assert math.isclose(fraction, offset / 400.0, rel_tol=0.1)


def test_detector_reports_raw_curves_and_geometry() -> None:
    image = _synthetic_clay(bar_top_rows=[140, 180])

    report = bust_assessment.analyze_duplication(image)

    bbox = report["subject_bbox_px"]
    assert bbox["row0"] == _SUBJECT_ROWS[0]
    assert bbox["row1"] == _SUBJECT_ROWS[1]
    assert report["subject_height_px"] == 400
    face_rows = report["face_region_rows_px"]
    assert face_rows[0] == _SUBJECT_ROWS[0]
    assert face_rows[1] == _SUBJECT_ROWS[0] + int(round(0.55 * 400))
    # Raw curves must be present for downstream inspection/re-thresholding.
    assert len(report["row_profile"]) == face_rows[1] - face_rows[0]
    assert len(report["autocorrelation"]) == len(report["row_profile"])
    assert report["autocorrelation"][0] == pytest.approx(1.0)


def test_find_duplication_peak_ignores_monotonic_decay() -> None:
    # A healthy face decays monotonically through the window: the window
    # maximum sits on the boundary and must NOT count as a peak.
    autocorr = np.array([1.0 / (1.0 + k) for k in range(120)])

    result = bust_assessment.find_duplication_peak(autocorr, subject_height=400)

    assert result["peak_lag_px"] is None
    assert result["peak_ratio"] is None
    assert result["window_max_ratio"] is not None


def test_json_safe_converts_numpy_paths_and_non_finite() -> None:
    converted = bust_assessment._json_safe(
        {
            "np_scalar": np.float32(1.5),
            "np_array": np.array([1, 2, 3]),
            "path": Path("/tmp/x"),
            "nan": float("nan"),
            "nested": [np.int64(7), {"ok": True}],
        }
    )

    assert converted == {
        "np_scalar": 1.5,
        "np_array": [1, 2, 3],
        "path": "/tmp/x",
        "nan": None,
        "nested": [7, {"ok": True}],
    }
    json.dumps(converted)  # must be serializable as-is


# ---------------------------------------------------------------------------
# Scorecard shape on a tiny GLB fixture (render-dependent).
# ---------------------------------------------------------------------------


def test_scorecard_shape_on_box_glb(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    _require_renderer()

    glb_path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 1.4, 1.8)).export(glb_path)
    output_dir = tmp_path / "assessment"

    exit_code = bust_assessment.main(
        ["--glb", str(glb_path), "--output-dir", str(output_dir), "--size", "96"]
    )

    assert exit_code == 0
    scorecard = json.loads((output_dir / "scorecard.json").read_text())
    assert (output_dir / "scorecard.md").is_file()

    geometry = scorecard["geometry"]
    for key in (
        "vertex_count",
        "face_count",
        "watertight",
        "body_count",
        "euler_number",
        "dihedral_angle_rms_deg",
    ):
        assert key in geometry, f"geometry missing {key}"
    assert geometry["face_count"] == 12
    assert geometry["watertight"] is True
    assert geometry["body_count"] == 1

    dup = scorecard["duplication_detector"]
    for key in (
        "row_profile",
        "autocorrelation",
        "lag_window_px",
        "peak_ratio",
        "threshold",
        "duplication_suspect",
    ):
        assert key in dup, f"duplication_detector missing {key}"
    assert isinstance(dup["duplication_suspect"], bool)

    texture = scorecard["texture"]
    assert texture["metadata"] is None  # bare GLB input has no bundle metadata
    assert len(texture["views"]) == 8
    for view in texture["views"]:
        for key in ("azimuth_deg", "mean_saturation", "mean_luminance", "luminance_variance"):
            assert key in view
    assert texture["back_view"]["azimuth_deg"] == 180.0
    assert isinstance(texture["back_view_unpainted_suspect"], bool)

    files = scorecard["renders"]["files"]
    expected = {f"clay_az{a:03d}" for a in (0, 45, 90, 135, 180, 225, 270, 315)}
    expected |= {f"textured_az{a:03d}" for a in (0, 45, 90, 135, 180, 225, 270, 315)}
    expected |= {"face_crop_clay", "face_crop_textured", "clay_front_hires", "textured_front_hires"}
    assert expected <= set(files)
    for name in expected:
        assert Path(files[name]).is_file(), f"missing render on disk: {name}"

    json.dumps(scorecard)  # whole scorecard must stay JSON-safe


def test_missing_inputs_exit_2(tmp_path: Path) -> None:
    exit_code = bust_assessment.main(
        ["--bundle", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 2

    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()
    exit_code = bust_assessment.main(
        ["--bundle", str(empty_bundle), "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 2

    exit_code = bust_assessment.main(
        ["--glb", str(tmp_path / "missing.glb"), "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 2
