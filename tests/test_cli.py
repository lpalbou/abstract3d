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
