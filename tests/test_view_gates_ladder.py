"""Ladder integration for the matte/identity acceptance gates: rejection
re-rolls the seed within `generate_reference_views` (same semantics as the
pose gate), verdicts land on attempt rows, and `synthesize_loop_views`
threads gates built on the source photo.

Fake i2i generators + fake gates throughout (the real calibration lives in
test_view_gates.py's evidence battery)."""

from __future__ import annotations

import io

import numpy as np
import pytest

# torch before pymeshlab-backed helpers (OpenMP init order on macOS).
import torch  # noqa: F401
import trimesh
from PIL import Image

from abstract3d import loop_conditioning as loop
from abstract3d import reference_generation as refgen


@pytest.fixture(autouse=True)
def _stub_captioner(monkeypatch):
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image", lambda image, **kw: "test object")


def sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=2, radius=0.5)


def clay_matching_generation(mesh, azimuth, color=(180, 140, 100)):
    from abstract3d.rendering import render_mesh_views

    clay = render_mesh_views(mesh, size=96, azimuths=[azimuth], elevation=0.0)[0]
    silhouette = refgen.clay_silhouette(clay)
    rgba = np.zeros((*silhouette.shape, 4), np.uint8)
    rgba[silhouette] = (*color, 255)
    return Image.fromarray(rgba, "RGBA")


def make_fake_generator(image):
    calls = []

    def generator(prompt, payload, **kwargs):
        calls.append(kwargs)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    generator.calls = calls
    return generator


def scripted_gate(verdicts, log):
    """A gate whose verdicts pop off a script; records each call."""

    def gate(view_rgba, *, label, azimuth_deg, elevation_deg=0.0):
        log.append({"label": label, "azimuth_deg": azimuth_deg})
        return dict(verdicts[min(len(log) - 1, len(verdicts) - 1)])

    return gate


def solid_rgba(color, size=96, alpha=255):
    return Image.new("RGBA", (size, size), (*color, alpha))


def _bypass_rembg(monkeypatch):
    # the fake generations already carry a clean alpha
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))


def test_matte_gate_rejection_rerolls_and_names_the_family(monkeypatch) -> None:
    mesh = sphere_mesh()
    _bypass_rembg(monkeypatch)
    generator = make_fake_generator(clay_matching_generation(mesh, 180.0))
    calls: list = []
    gate = scripted_gate(
        [{"measured": True, "passed": False,
          "reason": "matte cleanliness: torn interior alpha 0.2 > 0.05"}],
        calls)

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("back", 180.0, 0.0)],
        max_attempts=2,
        render_size=96,
        matte_gate=gate,
    )

    assert views == []
    assert report["matte_gate"] == "enabled"
    assert len(calls) == 2  # every strict-passing candidate was judged
    entry = report["angles"][0]
    assert entry["accepted"] is False
    assert "matte cleanliness" in entry["rejection_reason"]
    families = [row.get("failure_family") for row in entry["attempts"]]
    assert families == ["matte_debris", "matte_debris"]
    assert all("matte" in row for row in entry["attempts"])


def test_identity_gate_rejection_rerolls_and_names_the_family(monkeypatch) -> None:
    mesh = sphere_mesh()
    _bypass_rembg(monkeypatch)
    generator = make_fake_generator(clay_matching_generation(mesh, 90.0))
    calls: list = []
    gate = scripted_gate(
        [{"measured": True, "passed": False, "cosine": 0.21,
          "reason": "subject identity: cosine 0.21 below floor 0.38"}],
        calls)

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("side_left", 90.0, 0.0)],
        max_attempts=2,
        render_size=96,
        identity_gate=gate,
    )

    assert views == []
    assert report["identity_gate"] == "enabled"
    entry = report["angles"][0]
    assert "subject identity" in entry["rejection_reason"]
    assert "0.21" in entry["rejection_reason"]
    families = [row.get("failure_family") for row in entry["attempts"]]
    assert families == ["subject_identity", "subject_identity"]


def test_gates_pass_through_and_ride_the_accepted_view(monkeypatch) -> None:
    mesh = sphere_mesh()
    _bypass_rembg(monkeypatch)
    generator = make_fake_generator(clay_matching_generation(mesh, 90.0))
    matte_calls: list = []
    identity_calls: list = []
    matte = scripted_gate(
        [{"measured": True, "passed": True, "edge_density_ratio": 1.1}],
        matte_calls)
    identity = scripted_gate(
        [{"measured": True, "passed": True, "cosine": 0.61}],
        identity_calls)

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("side_left", 90.0, 0.0)],
        render_size=96,
        matte_gate=matte,
        identity_gate=identity,
    )

    assert len(views) == 1
    assert len(matte_calls) == 1 and len(identity_calls) == 1
    assert views[0]["matte"]["passed"] is True
    assert views[0]["identity"]["cosine"] == 0.61
    entry = report["angles"][0]
    assert entry["accepted"] is True
    assert entry["identity"]["cosine"] == 0.61


def test_identity_gate_skipped_after_matte_rejection(monkeypatch) -> None:
    """Cheapest-first ordering: a matte-rejected candidate never pays the
    identity embedding."""
    mesh = sphere_mesh()
    _bypass_rembg(monkeypatch)
    generator = make_fake_generator(clay_matching_generation(mesh, 90.0))
    matte_calls: list = []
    identity_calls: list = []
    matte = scripted_gate(
        [{"measured": True, "passed": False, "reason": "matte cleanliness: x"}],
        matte_calls)
    identity = scripted_gate(
        [{"measured": True, "passed": True}], identity_calls)

    refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("side_left", 90.0, 0.0)],
        max_attempts=1,
        render_size=96,
        matte_gate=matte,
        identity_gate=identity,
    )

    assert len(matte_calls) == 1
    assert identity_calls == []


def test_crashing_gates_abstain_loudly_and_never_burn_the_ladder(monkeypatch) -> None:
    mesh = sphere_mesh()
    _bypass_rembg(monkeypatch)
    generator = make_fake_generator(clay_matching_generation(mesh, 90.0))

    def boom(view_rgba, **kwargs):
        raise RuntimeError("gate exploded")

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("side_left", 90.0, 0.0)],
        render_size=96,
        matte_gate=boom,
        identity_gate=boom,
    )

    assert len(views) == 1  # abstention, not rejection
    entry = report["angles"][0]
    row = entry["attempts"][-1]
    assert "gate exploded" in row["matte"]["note"]
    assert "gate exploded" in row["identity"]["note"]


def test_synthesize_loop_views_threads_matte_and_identity_gates(monkeypatch) -> None:
    captured: dict = {}

    def fake(mesh0, source, **kwargs):
        captured.update(kwargs)
        return [], {"accepted": 0, "rejected": 0, "angles": []}

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views", fake)

    def bust_image() -> Image.Image:
        arr = np.zeros((200, 200, 4), dtype=np.uint8)
        arr[20:120, 80:120] = (180, 150, 130, 255)
        arr[120:190, 10:190] = (90, 80, 70, 255)
        return Image.fromarray(arr, "RGBA")

    def scripted_view_consistency():
        def report(clay, view):
            return {"score": 0.6, "verdict": "consistent",
                    "score_at_best_remap": 0.7, "best_shift_rows": 0,
                    "shift_frac": 0.0}

        def align(view, reference):
            raise AssertionError("never row-warps")

        return report, align

    loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("side_left", 90.0, 0.0),),
        seed=5,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=scripted_view_consistency(),
    )

    assert callable(captured["matte_gate"])
    assert callable(captured["identity_gate"])
    # gate CONSTRUCTION must not load any model: the identity gate is lazy
    # (this test would take ~10s + 330MB if it did).


def test_loop_fold_refuses_view_with_riding_failed_matte(monkeypatch) -> None:
    """Defensive belt: a riding FAILED verdict (only a bypassed ladder
    could produce one) refuses both consumers."""

    def bust_image() -> Image.Image:
        arr = np.zeros((200, 200, 4), dtype=np.uint8)
        arr[20:120, 80:120] = (180, 150, 130, 255)
        arr[120:190, 10:190] = (90, 80, 70, 255)
        return Image.fromarray(arr, "RGBA")

    view = {
        "label": "side_left", "azimuth_deg": 90.0, "elevation_deg": 0.0,
        "rgba": bust_image(), "raw_bytes": b"raw", "raw_payload_md5": "0" * 32,
        "seed": 71_000, "clay_render": None, "role": "reference",
        "generated": True,
        "matte": {"measured": True, "passed": False,
                  "reason": "matte cleanliness: torn interior alpha"},
    }

    def fake(mesh0, source, **kwargs):
        return [view], {"accepted": 1, "rejected": 0, "angles": []}

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views", fake)

    def scripted_view_consistency():
        def report(clay, v):
            return {"score": 0.6, "verdict": "consistent",
                    "score_at_best_remap": 0.7, "best_shift_rows": 0,
                    "shift_frac": 0.0}

        return report, lambda v, r: None

    plan = loop.synthesize_loop_views(
        object(),
        bust_image(),
        owner=object(),
        angles=(("side_left", 90.0, 0.0),),
        seed=5,
        subject_hint="a toy bust",
        person_attested=False,
        image_request={"provider": "mlx-gen"},
        view_consistency=scripted_view_consistency(),
    )

    row = plan["views"][0]
    assert row["conditioning_eligible"] is False
    assert row["bake_eligible"] is False
    assert "matte_debris" in row["conditioning_refusal"]
