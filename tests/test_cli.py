from __future__ import annotations

import json

from abstract3d import cli


def test_catalog_json_prints_rows(capsys) -> None:
    exit_code = cli.main(["catalog", "--validated-only", "--json"])

    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["model_id"] == "stabilityai/TripoSR"
    assert rows[0]["validated"] is True


def test_validate_uses_default_prompts_when_none_are_passed(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def validate_suite(self, **kwargs):
            calls.update(kwargs)
            return {"contact_sheet": "sheet.png", "stats": "stats.json", "rows": []}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(["validate", "--output-dir", str(tmp_path), "--backend", "trellis2"])

    assert exit_code == 0
    assert calls["backend_id"] == "trellis2"
    assert calls["output_dir"] == str(tmp_path)
    assert calls["image_provider"] is None
    assert calls["prompts"] == [
        "a ceramic teapot with a curved spout and matte glaze",
        "a mid-century lounge chair with walnut legs and woven fabric",
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["contact_sheet"] == "sheet.png"


def test_t23d_defaults_to_provider_neutral_image_generation(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def t23d(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls.update(kwargs)
            return {"metadata": {"ok": True}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(["t23d", "rocket", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert calls["prompt"] == "rocket"
    # Provider-neutral means ABSENT under the strict option contract: the
    # CLI forwards only explicitly-set options, and the composition layer
    # resolves its configured default when no provider/model arrives.
    assert "image_provider" not in calls
    assert "image_model" not in calls
    assert calls["image_width"] == 768
    assert calls["image_height"] == 768
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True


def test_validate_passes_guidance_and_steps(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def validate_suite(self, **kwargs):
            calls.update(kwargs)
            return {"contact_sheet": "sheet.png", "stats": "stats.json", "rows": []}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "validate",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "step1x",
            "--num-inference-steps",
            "8",
            "--guidance-scale",
            "3.0",
        ]
    )

    assert exit_code == 0
    assert calls["num_inference_steps"] == 8
    assert calls["guidance_scale"] == 3.0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stats"] == "stats.json"


def test_validate_passes_model_subfolder(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def validate_suite(self, **kwargs):
            calls.update(kwargs)
            return {"contact_sheet": "sheet.png", "stats": "stats.json", "rows": []}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "validate",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "step1x",
            "--model-subfolder",
            "Step1X-3D-Geometry-Label-1300m",
        ]
    )

    assert exit_code == 0
    assert calls["model_subfolder"] == "Step1X-3D-Geometry-Label-1300m"
    summary = json.loads(capsys.readouterr().out)
    assert summary["contact_sheet"] == "sheet.png"


def test_i23d_passes_cleanup_option(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def i23d(self, image, **kwargs):
            calls["image"] = image
            calls.update(kwargs)
            return {"metadata": {"ok": True}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "i23d",
            "rocket.png",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "triposr",
            "--cleanup",
            "presentation",
        ]
    )

    assert exit_code == 0
    assert calls["backend_id"] == "triposr"
    assert calls["cleanup"] == "presentation"
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True


def test_i23d_passes_texture_options(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def i23d(self, image, **kwargs):
            calls["image"] = image
            calls.update(kwargs)
            return {"metadata": {"ok": True}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "i23d",
            "rocket.png",
            "--output-dir",
            str(tmp_path),
            "--texture-mode",
            "baked_basecolor",
            "--texture-resolution",
            "4096",
        ]
    )

    assert exit_code == 0
    assert calls["texture_mode"] == "baked_basecolor"
    assert calls["texture_resolution"] == 4096
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True


def test_i23d_passes_texture_reference_options(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def i23d(self, image, **kwargs):
            calls["image"] = image
            calls.update(kwargs)
            return {"metadata": {"ok": True}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "i23d",
            "face.png",
            "--output-dir",
            str(tmp_path),
            "--texture-reference-image",
            "side.png",
            "--texture-reference-angle",
            "side_right",
            "--texture-reference-remove-background",
        ]
    )

    assert exit_code == 0
    assert calls["texture_reference_images"] == ["side.png"]
    assert calls["texture_reference_angles"] == ["side_right"]
    assert calls["texture_reference_remove_background"] is True
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True


def test_i23d_passes_texture_completion(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None) -> None:
            calls["backend_id"] = backend_id

        def i23d(self, image, **kwargs):
            calls["image"] = image
            calls.update(kwargs)
            return {"metadata": {"ok": True}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)

    exit_code = cli.main(
        [
            "i23d",
            "face.png",
            "--output-dir",
            str(tmp_path),
            "--texture-completion",
            "mirror_symmetry",
        ]
    )

    assert exit_code == 0
    assert calls["texture_completion"] == "mirror_symmetry"
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True


class _CaptureManager:
    """Records the kwargs the CLI forwards (shared by the quality-option
    tests below; the strict option contract is about exactly which keys
    reach the backend)."""

    calls: dict = {}

    def __init__(self, backend_id=None) -> None:
        _CaptureManager.calls = {"backend_id": backend_id}

    def i23d(self, image, **kwargs):
        _CaptureManager.calls["image"] = image
        _CaptureManager.calls.update(kwargs)
        return {"metadata": {"ok": True}}

    def t23d(self, prompt, **kwargs):
        _CaptureManager.calls["prompt"] = prompt
        _CaptureManager.calls.update(kwargs)
        return {"metadata": {"ok": True}}


def test_i23d_passes_shape_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        [
            "i23d",
            "car.png",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "hunyuan3d21",
            "--shape-candidates",
            "3",
        ]
    )

    assert exit_code == 0
    assert _CaptureManager.calls["shape_candidates"] == 3


def test_i23d_passes_angle_planning_only_when_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        [
            "i23d", "car.png", "--output-dir", str(tmp_path),
            "--backend", "hunyuan3d21",
            "--texture-reference-angle-planning", "adaptive",
        ]
    )
    assert exit_code == 0
    assert _CaptureManager.calls["texture_reference_angle_planning"] == "adaptive"

    # strict option contract: unset means ABSENT, not None
    exit_code = cli.main(
        ["i23d", "car.png", "--output-dir", str(tmp_path),
         "--backend", "hunyuan3d21"]
    )
    assert exit_code == 0
    assert "texture_reference_angle_planning" not in _CaptureManager.calls


def test_t23d_quality_preset_maps_to_shape_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    for preset, expected in (("standard", 1), ("high", 2), ("best", 3)):
        exit_code = cli.main(
            [
                "t23d",
                "a sports car",
                "--output-dir",
                str(tmp_path),
                "--backend",
                "hunyuan3d21",
                "--quality",
                preset,
            ]
        )
        assert exit_code == 0
        assert _CaptureManager.calls["shape_candidates"] == expected


def test_quality_presets_do_not_route_geometry_conditioning(monkeypatch, tmp_path) -> None:
    # Multi-view geometry conditioning is an explicit opt-in, not a preset
    # rider: the equal-seed A/B measured it as a topology-vs-concavity
    # TRADE, so no preset forwards it (strict option contract: absent,
    # not None).
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    for preset in ("standard", "high", "best"):
        exit_code = cli.main(
            [
                "i23d",
                "car.png",
                "--output-dir",
                str(tmp_path),
                "--backend",
                "hunyuan3d21",
                "--quality",
                preset,
            ]
        )
        assert exit_code == 0
        assert "geometry_conditioning" not in _CaptureManager.calls


def test_i23d_passes_geometry_conditioning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        [
            "i23d",
            "car.png",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "hunyuan3d21",
            "--geometry-conditioning",
            "multiview",
        ]
    )

    assert exit_code == 0
    assert _CaptureManager.calls["geometry_conditioning"] == "multiview"


def test_geometry_conditioning_combines_with_quality_preset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        [
            "i23d",
            "car.png",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "hunyuan3d21",
            "--quality",
            "best",
            "--geometry-conditioning",
            "multiview",
        ]
    )

    assert exit_code == 0
    assert _CaptureManager.calls["geometry_conditioning"] == "multiview"
    assert _CaptureManager.calls["shape_candidates"] == 3


def test_explicit_shape_candidates_overrides_quality_preset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        [
            "i23d",
            "car.png",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "hunyuan3d21",
            "--quality",
            "standard",
            "--shape-candidates",
            "5",
        ]
    )

    assert exit_code == 0
    assert _CaptureManager.calls["shape_candidates"] == 5


def test_no_quality_flags_forward_no_shape_candidates(monkeypatch, tmp_path) -> None:
    # Strict option contract: unset flags must be ABSENT, not None — a
    # backend without the option (e.g. triposr) must never see it.
    monkeypatch.setattr(cli, "Scene3DManager", _CaptureManager)

    exit_code = cli.main(
        ["i23d", "car.png", "--output-dir", str(tmp_path), "--backend", "triposr"]
    )

    assert exit_code == 0
    assert "shape_candidates" not in _CaptureManager.calls
    assert "quality" not in _CaptureManager.calls
    assert "geometry_conditioning" not in _CaptureManager.calls
