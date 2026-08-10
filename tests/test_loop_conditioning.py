"""Unit tests for loop_conditioning: the window law, the pose-honesty
ruler, the duplication flag, and the loop-view synthesis plan fold.

No model loads anywhere: the i2i stage is scripted through a monkeypatched
generate_reference_views, view consistency through a scripted tuple, and
the pose ruler through synthetic meshes/renders (or a stub where the test
targets the FOLD, not the ruler).
"""

from __future__ import annotations

import numpy as np
import pytest

# Import torch before pymeshlab-backed helpers run (OpenMP init order on
# macOS; same rule as the backend unit suite).
import torch  # noqa: F401
import trimesh
from PIL import Image

from abstract3d import loop_conditioning as loop


def bust_image(size: int = 200, head_top: int = 20, shoulder: int = 120,
               bottom: int = 190, face_rows=()) -> Image.Image:
    """Synthetic bust: a narrow head column widening to full shoulders at
    `shoulder`, on transparent background. `face_rows` paints dark
    horizontal lines (mouth-like features) into the head region."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    head_half = size // 8
    shoulder_half = size // 2 - 8
    for row in range(head_top, bottom + 1):
        if row < shoulder:
            half = head_half
        else:
            half = shoulder_half
        arr[row, center - half : center + half] = (180, 150, 130, 255)
    for row in face_rows:
        arr[row, center - head_half : center + head_half, :3] = (40, 30, 30)
    return Image.fromarray(arr, "RGBA")


# -- window law ---------------------------------------------------------------


def test_window_anchors_and_cut_row() -> None:
    image = bust_image(head_top=20, shoulder=120)
    anchors = loop.window_anchors(image)
    assert anchors["head_top"] == 20
    assert anchors["shoulder"] == 120

    windowed, report = loop.window_view(image)
    span = report["shoulder"] - report["head_top"]
    assert report["cut_row"] == round(report["shoulder"] + loop.LOOP_WINDOW_K * span)
    alpha = np.asarray(windowed)[:, :, 3]
    assert alpha[report["cut_row"] + 1 :].max() == 0  # everything below is cut
    assert alpha[report["head_top"] : report["cut_row"]].max() == 255
    # The kept span ratio equalizes by construction: 1 / (1 + k).
    assert report["kept_ratio_shoulder"] == pytest.approx(1.0 / (1.0 + loop.LOOP_WINDOW_K), abs=0.02)


def test_window_view_refuses_empty_subject() -> None:
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="no subject pixels"):
        loop.window_view(empty)


def test_window_equalizes_normalized_shoulder_across_different_cuts() -> None:
    """The mechanism the law exists for: two views of the same anatomy cut
    at different torso depths put the shoulder at different normalized
    rows; after windowing the normalized position agrees."""
    deep = bust_image(size=240, head_top=10, shoulder=110, bottom=235)
    shallow = bust_image(size=240, head_top=10, shoulder=110, bottom=150)

    def shoulder_norm(image) -> float:
        anchors = loop.window_anchors(image)
        alpha = np.asarray(image)[:, :, 3] > 10
        rows = np.flatnonzero(alpha.any(axis=1))
        return (anchors["shoulder"] - rows[0]) / max(rows[-1] - rows[0], 1)

    before = abs(shoulder_norm(deep) - shoulder_norm(shallow))
    win_deep, _ = loop.window_view(deep)
    win_shallow, _ = loop.window_view(shallow)
    after = abs(shoulder_norm(win_deep) - shoulder_norm(win_shallow))
    assert before > 0.2  # the defect is present pre-window
    assert after < 0.02  # and gone post-window


# -- duplication flag ----------------------------------------------------------


def test_duplication_flag_fires_on_duplicated_feature_and_not_on_single() -> None:
    size = 200
    # Two mouth-like lines 12 px apart = 7% of subject height (inside the
    # 3-15% lag window) -> a genuine non-zero-lag local peak.
    doubled = bust_image(size, face_rows=(60, 61, 72, 73))
    single = bust_image(size, face_rows=(60, 61))

    flagged = loop.duplication_flag(doubled)
    clean = loop.duplication_flag(single)
    assert flagged["duplication_suspect"] is True
    assert flagged["peak_lag_px"] is not None
    assert clean["duplication_suspect"] is False


def test_duplication_flag_reports_too_small_subject_honestly() -> None:
    tiny = bust_image(size=40, head_top=4, shoulder=20, bottom=36)
    result = loop.duplication_flag(tiny)
    assert result["duplication_suspect"] is False
    assert "note" in result or result["peak_ratio"] is None


# -- pose ruler ----------------------------------------------------------------


def _bust_mesh() -> trimesh.Trimesh:
    """Bust-class geometry (the ruler's calibration domain): a head with a
    nose-like protrusion above a wide shoulder slab — the head band's
    silhouette then varies strongly with azimuth."""
    shoulders = trimesh.creation.box(extents=(0.7, 1.4, 0.5))
    head = trimesh.creation.box(extents=(0.7, 0.35, 0.7))
    head.apply_translation((0.0, 0.0, 0.65))
    nose = trimesh.creation.box(extents=(0.25, 0.12, 0.15))
    nose.apply_translation((0.45, 0.0, 0.68))
    return trimesh.util.concatenate([shoulders, head, nose])


def test_measure_view_azimuth_recovers_true_angle_and_flags_lies() -> None:
    from abstract3d.rendering import render_mesh_views

    mesh = _bust_mesh()
    true_view = render_mesh_views(mesh, size=128, azimuths=[90.0], elevation=0.0)[0].convert("RGBA")
    honest = loop.measure_view_azimuth(true_view, mesh, 90.0, render_size=128)
    assert abs(honest["measured_deg"] - 90.0) <= 10.0
    assert honest["delta_deg"] <= loop.POSE_MAX_DELTA_DEG

    # A ~50-degree three-quarter view sold as a 90-degree profile (the
    # set A incident, scaled to the synthetic subject): DECISIVELY off.
    lie_view = render_mesh_views(mesh, size=128, azimuths=[50.0], elevation=0.0)[0].convert("RGBA")
    lie = loop.measure_view_azimuth(lie_view, mesh, 90.0, render_size=128)
    assert lie["decisive"] is True
    assert lie["iou_gap"] >= loop.POSE_GATE_MIN_IOU_GAP
    assert lie["delta_deg"] > loop.POSE_MAX_DELTA_DEG
    assert abs(lie["measured_deg"] - 50.0) <= 10.0  # re-declaration target


def test_measure_view_azimuth_flat_sweep_is_never_decisive() -> None:
    """Rotation-symmetric subjects (and back views of busts) sweep flat:
    any argmax is noise — never decisive, so nothing acts on it."""
    from abstract3d.rendering import render_mesh_views

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    view = render_mesh_views(mesh, size=128, azimuths=[180.0], elevation=0.0)[0].convert("RGBA")
    result = loop.measure_view_azimuth(view, mesh, 180.0, render_size=128)
    assert result["decisive"] is False
    assert result["iou_gap"] < loop.POSE_GATE_MIN_IOU_GAP


# -- pose acceptance gate (viewgen bench 2026-07-21 section 6.3) --------------


def test_pose_acceptance_gate_uses_the_ratified_two_key_rule() -> None:
    """The RATIFIED pose contract (strategy_v2 §stage contracts, stage E):
    refuse conditioning only when delta > 20 deg AND the measurement is
    decisive (iou_gap >= 0.10). The acceptance gate must use the SAME rule
    as the eligibility fold — an earlier stricter 15-deg acceptance
    default compounded with scaffold-clay measurement noise and rejected
    honest draws (e22 forensics, 2026-07-21)."""
    assert loop.POSE_MAX_DELTA_DEG == 20.0
    assert loop.POSE_ACCEPT_MAX_DELTA_DEG == loop.POSE_MAX_DELTA_DEG
    assert loop.POSE_GATE_MIN_IOU_GAP == 0.10


def test_build_pose_acceptance_gate_two_key_semantics(monkeypatch) -> None:
    """Reject only when BOTH keys agree: decisively measured AND more than
    the acceptance threshold off. Plateau argmaxes (bust backs) pass; a
    crashing or inapplicable ruler abstains as pose_unmeasured."""
    scripted = {}
    monkeypatch.setattr(
        loop, "measure_view_azimuth",
        lambda view, mesh, nominal, **kw: dict(scripted))

    gate = loop.build_pose_acceptance_gate(object())
    assert gate is not None
    assert loop.build_pose_acceptance_gate(None) is None  # no mesh, no gate

    # Decisive + 40 off -> rejected with a loud reason naming the numbers.
    scripted.update({"decisive": True, "measured_deg": -50.0,
                     "delta_deg": 40.0, "iou_gap": 0.16})
    verdict = gate(bust_image(), label="side_right", azimuth_deg=-90.0,
                   elevation_deg=0.0)
    assert verdict["measured"] is True
    assert verdict["passed"] is False
    assert verdict["max_delta_deg"] == loop.POSE_ACCEPT_MAX_DELTA_DEG
    assert "pose honesty" in verdict["reason"]
    assert "40.0" in verdict["reason"]

    # Decisive + 10 off -> passes (arm E's champion band).
    scripted.update({"measured_deg": -80.0, "delta_deg": 10.0})
    assert gate(bust_image(), label="side_right", azimuth_deg=-90.0,
                elevation_deg=0.0)["passed"] is True

    # NON-decisive 30 off -> plateau noise, never rejects (e18/e19
    # conditioned on exactly such a back, healthy meshes).
    scripted.update({"decisive": False, "measured_deg": 210.0,
                     "delta_deg": 30.0, "iou_gap": 0.03})
    assert gate(bust_image(), label="back", azimuth_deg=180.0,
                elevation_deg=0.0)["passed"] is True

    # Elevated views: the ruler sweeps elevation-0 clays only -> abstain.
    elevated = gate(bust_image(), label="top", azimuth_deg=0.0,
                    elevation_deg=55.0)
    assert elevated["measured"] is False
    assert elevated["passed"] is True
    assert "pose_unmeasured" in elevated["note"]

    # A crashing ruler abstains loudly instead of failing the draw.
    def boom(view, mesh, nominal, **kw):
        raise RuntimeError("no clay")

    monkeypatch.setattr(loop, "measure_view_azimuth", boom)
    crashed = gate(bust_image(), label="back", azimuth_deg=180.0,
                   elevation_deg=0.0)
    assert crashed["measured"] is False
    assert crashed["passed"] is True
    assert "no clay" in crashed["note"]


def test_synthesize_loop_views_threads_pose_gate_and_reuses_ladder_pose(
    monkeypatch,
) -> None:
    """The plan fold passes a pose gate into the refgen ladder (built on
    the pass-1 mesh) and REUSES the ladder's measurement for eligibility —
    the ruler never runs twice on the same pixels."""
    ladder_pose = {
        "measured": True, "passed": True, "measurable": True,
        "decisive": True, "measured_deg": 82.5, "delta_deg": 7.5,
        "iou_gap": 0.12,
    }
    view = _plan_view("side_left", 90.0)
    view["pose"] = dict(ladder_pose)
    captured: dict = {}

    def fake(mesh0, source, **kwargs):
        captured.update(kwargs)
        return [view], {"accepted": 1, "rejected": 0, "angles": []}

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views", fake)

    def must_not_remeasure(*args, **kwargs):
        raise AssertionError(
            "the fold must reuse the acceptance ladder's measurement")

    monkeypatch.setattr(loop, "measure_view_azimuth", must_not_remeasure)

    plan = loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("side_left", 90.0, 0.0),),
        seed=5,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=_scripted_view_consistency({}),
    )

    assert callable(captured["pose_gate"])
    row = plan["views"][0]
    assert row["pose"]["source"] == "acceptance_ladder"
    assert row["measured_azimuth_deg"] == 82.5
    assert row["conditioning_eligible"] is True


# -- loop-view synthesis plan fold ----------------------------------------------


def _scripted_refgen(views):
    def fake(mesh0, source, **kwargs):
        fake.kwargs = kwargs
        return list(views), {"accepted": len(views), "rejected": 0, "angles": []}

    return fake


def _scripted_view_consistency(verdicts_by_label):
    def report(clay, view):
        label = getattr(view, "_test_label", None)
        verdict = verdicts_by_label.get(label, "consistent")
        return {
            "score": 0.6 if verdict == "consistent" else 0.0,
            "score_at_best_remap": 0.7 if verdict != "inconsistent" else 0.2,
            "verdict": verdict,
            "best_shift_rows": 0,
            "shift_frac": 0.0,
        }

    def align(view, reference):
        raise AssertionError("the loop lane never row-warps registered views")

    return report, align


def _plan_view(label: str, azimuth: float) -> dict:
    image = bust_image()
    image._test_label = label
    return {
        "label": label,
        "azimuth_deg": azimuth,
        "elevation_deg": 0.0,
        "rgba": image,
        "raw_bytes": f"raw-{label}".encode(),
        "raw_payload_md5": "0" * 32,
        "seed": 71_000,
        "clay_render": bust_image(),
        "role": "reference",
        "generated": True,
    }


def test_synthesize_loop_views_attempt_log_writes_jsonl(monkeypatch, tmp_path) -> None:
    """attempt_log_path wires a JSONL writer into the refgen ladder's
    on_attempt sink: each event lands as one line the moment it happens
    (mid-run observability + crash forensics — e22 held every per-attempt
    reason in memory for ~80 minutes and lost them all on the refusal)."""
    import json

    def fake(mesh0, source, **kwargs):
        fake.kwargs = kwargs
        sink = kwargs.get("on_attempt")
        if callable(sink):
            sink({"event": "attempt", "label": "back", "seed": 72025,
                  "failure_family": "speculars"})
            sink({"event": "angle_result", "label": "back", "accepted": False})
        return [], {"accepted": 0, "rejected": 1, "angles": []}

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views", fake)

    log_path = tmp_path / "bundle" / "loop_refgen_attempts.jsonl"
    loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("back", 180.0, 0.0),),
        seed=72_025,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=_scripted_view_consistency({}),
        attempt_log_path=log_path,
    )

    assert callable(fake.kwargs.get("on_attempt"))
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["event"] for row in lines] == ["attempt", "angle_result"]
    assert lines[0]["failure_family"] == "speculars"
    # No log path -> no sink (byte-compatible with pre-forensics callers).
    fake.kwargs = {}
    loop.synthesize_loop_views(
        object(), bust_image(), owner=object(), angles=(("back", 180.0, 0.0),),
        seed=1, subject_hint="a toy bust", person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=_scripted_view_consistency({}),
    )
    assert fake.kwargs.get("on_attempt") is None


def test_synthesize_loop_views_folds_gates_into_eligibility(monkeypatch) -> None:
    views = [
        _plan_view("back", 180.0),
        _plan_view("side_left", 90.0),
        _plan_view("side_right", -90.0),
    ]
    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views",
        _scripted_refgen(views),
    )

    poses = {
        # Bust back: plateau argmax (gap under the decisive floor) — the
        # declared angle stands for both consumers.
        "back": {"measurable": True, "decisive": False, "measured_deg": 210.0,
                 "delta_deg": 30.0, "iou_gap": 0.03},
        # Champion-class profile: decisively measured, within the gate.
        "side_left": {"measurable": True, "decisive": True, "measured_deg": 82.5,
                      "delta_deg": 7.5, "iou_gap": 0.12},
        # The set A class: sold as -90, decisively measured -50 (40 off).
        "side_right": {"measurable": True, "decisive": True, "measured_deg": -50.0,
                       "delta_deg": 40.0, "iou_gap": 0.16},
    }
    monkeypatch.setattr(
        loop, "measure_view_azimuth",
        lambda view, mesh, nominal, **kw: dict(poses[getattr(view, "_test_label")]),
    )

    plan = loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("back", 180.0, 0.0), ("side_left", 90.0, 0.0), ("side_right", -90.0, 0.0)),
        seed=72_025,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=_scripted_view_consistency({"side_right": "consistent"}),
    )

    rows = {row["label"]: row for row in plan["views"]}
    # Back: plateau argmax -> never pose-gated; declared azimuth stands
    # (e18/e19 conditioned on exactly such a back, healthy meshes).
    assert rows["back"]["conditioning_eligible"] is True
    assert rows["back"]["measured_azimuth_deg"] == 180.0
    # side_left: 7.5 deg off -> conditions, measured azimuth recorded.
    assert rows["side_left"]["conditioning_eligible"] is True
    assert rows["side_left"]["measured_azimuth_deg"] == 82.5
    # side_right: 40 deg off -> refused from CONDITIONING, re-declared to
    # the measured azimuth for the bake.
    assert rows["side_right"]["conditioning_eligible"] is False
    assert rows["side_right"]["bake_eligible"] is True
    assert "pose honesty" in rows["side_right"]["conditioning_refusal"]
    assert rows["side_right"]["measured_azimuth_deg"] == -50.0
    # Windowed variants exist exactly for the conditioning-eligible rows.
    assert rows["back"].get("windowed_rgba") is not None
    assert rows["side_left"].get("windowed_rgba") is not None
    assert rows["side_right"].get("windowed_rgba") is None
    # The front windows too, and the report carries the anchors.
    assert plan["front_windowed"] is not None
    assert set(plan["window"]["per_view"]) >= {"front", "back", "side_left"}


def test_synthesize_loop_views_rejects_garbage_class_from_both_consumers(monkeypatch) -> None:
    views = [_plan_view("back", 180.0)]
    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views",
        _scripted_refgen(views),
    )
    monkeypatch.setattr(
        loop, "measure_view_azimuth",
        lambda view, mesh, nominal, **kw: {
            "measurable": False, "measured_deg": nominal, "delta_deg": 0.0},
    )

    plan = loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("back", 180.0, 0.0),),
        seed=1,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={},
        view_consistency=_scripted_view_consistency({"back": "inconsistent"}),
    )

    row = plan["views"][0]
    assert row["conditioning_eligible"] is False
    assert row["bake_eligible"] is False
    assert "row consistency" in row["conditioning_refusal"]


def test_synthesize_loop_views_pins_identity_route_and_provider(monkeypatch) -> None:
    """Provider pinning + identity conditioning are threaded VERBATIM into
    the refgen call — the owner=None remote-default incident class."""
    captured: dict = {}

    def fake(mesh0, source, **kwargs):
        captured.update(kwargs)
        return [], {"accepted": 0, "rejected": 3, "angles": []}

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views", fake)

    loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner="THE-OWNER",
        angles=(("back", 180.0, 0.0),),
        seed=9,
        subject_hint=None,
        person_attested=True,
        image_request={"provider": "mlx-gen", "model": "klein"},
        view_consistency=_scripted_view_consistency({}),
    )

    assert captured["owner"] == "THE-OWNER"
    assert captured["conditioning"] == "identity"
    assert captured["image_request"] == {"provider": "mlx-gen", "model": "klein"}
    assert captured["person_policy"] == "proceed"
    assert captured["seed"] == 9


def test_self_verification_records_duplication_and_renders(monkeypatch) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    record, images = loop.self_verification(
        mesh, textured=False, clay_size=128, oblique_size=128)
    assert "duplication" in record
    assert record["duplication"]["duplication_suspect"] in (True, False)
    assert "verification_front_clay" in images
    # Oblique raking closeups render for both configured angles.
    assert any(name.startswith("verification_obl_") for name in images)
