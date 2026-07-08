from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_local.py"
_SPEC = importlib.util.spec_from_file_location("validate_local", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validate_local = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_local)


def test_default_validation_output_dir_uses_label_suffix_for_step1x_default() -> None:
    root = Path("/tmp/abstract3d")

    out = validate_local._default_validation_output_dir(
        root=root,
        backend="step1x",
        model_subfolder=None,
    )

    assert out == root / "artifacts" / "validation" / "local-step1x-label"


def test_default_validation_output_dir_uses_base_suffix_for_step1x_base_geometry() -> None:
    root = Path("/tmp/abstract3d")

    out = validate_local._default_validation_output_dir(
        root=root,
        backend="abstract3d:step1x-local",
        model_subfolder="Step1X-3D-Geometry-1300m",
    )

    assert out == root / "artifacts" / "validation" / "local-step1x-base"


def test_default_validation_output_dir_keeps_triposr_name() -> None:
    root = Path("/tmp/abstract3d")

    out = validate_local._default_validation_output_dir(
        root=root,
        backend="triposr",
        model_subfolder=None,
    )

    assert out == root / "artifacts" / "validation" / "local-triposr"


def test_default_case_rss_limit_gb_guards_heavy_backends_on_any_device() -> None:
    # The guard must hold under the `auto` device default and on CPU runs:
    # heavy backends use the same or more system memory there, and the guard
    # exists precisely to protect the host from pathological cases.
    assert validate_local._default_case_rss_limit_gb(backend="step1x", device="mps") == 64.0
    assert validate_local._default_case_rss_limit_gb(backend="step1x", device="auto") == 64.0
    assert validate_local._default_case_rss_limit_gb(backend="step1x", device="cpu") == 64.0
    assert validate_local._default_case_rss_limit_gb(backend="hunyuan3d21", device="auto") == 64.0
    assert validate_local._default_case_rss_limit_gb(backend="triposr", device="mps") is None


def test_build_case_specs_numbers_cases_consistently(tmp_path) -> None:
    args = validate_local._parser().parse_args([])

    cases = validate_local._build_case_specs(
        prompts=["teapot", "chair"],
        images=["/tmp/espresso.png", "/tmp/owl.png"],
        image_prompts=["espresso", "owl"],
        output_dir=tmp_path,
        args=args,
    )

    assert [case["case_id"] for case in cases] == ["01_t23d", "02_t23d", "03_i23d", "04_i23d"]
    assert cases[2]["image"] == "/tmp/espresso.png"
    assert cases[3]["prompt"] == "owl"


def test_validate_local_parser_defaults_to_neutral_image_composition() -> None:
    args = validate_local._parser().parse_args([])

    assert args.image_provider is None
    assert args.image_model is None


def test_failure_row_creates_placeholder_contact_sheet(tmp_path) -> None:
    row = validate_local._failure_row(
        case={
            "case_id": "03_i23d",
            "mode": "i23d",
            "backend": "step1x",
            "prompt": "owl",
            "output_dir": str(tmp_path / "03_i23d"),
        },
        status="timed_out",
        error="Case exceeded timeout.",
        timeout_s=120,
    )

    assert row["status"] == "timed_out"
    assert row["case_timeout_s"] == 120
    contact_sheet = Path(row["contact_sheet_path"])
    assert contact_sheet.exists()
    image = Image.open(contact_sheet)
    assert image.width > 0 and image.height > 0


def test_run_single_case_writes_success_result(monkeypatch, tmp_path) -> None:
    class _FakeManager:
        def __init__(self, backend_id=None):
            self.backend_id = backend_id

        def t23d(self, prompt, **kwargs):
            case_dir = Path(kwargs["output_dir"])
            contact_sheet = case_dir / "contact_sheet.png"
            contact_sheet.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), "white").save(contact_sheet)
            return {
                "metadata": {
                    "backend_id": self.backend_id,
                    "contact_sheet_path": str(contact_sheet),
                    "timings_s": {"total": 1.5},
                    "memory": {"rss_bytes": 1024},
                    "vertex_count": 12,
                    "face_count": 18,
                }
            }

    monkeypatch.setattr(validate_local, "Scene3DManager", _FakeManager)
    manifest_path = tmp_path / "case_manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "step1x",
                "case_id": "01_t23d",
                "mode": "t23d",
                "prompt": "teapot",
                "output_dir": str(tmp_path / "01_t23d"),
                "image_provider": "mlx-gen",
                "image_model": "fake",
                "mc_resolution": 128,
                "device": "cpu",
                "model": None,
                "model_subfolder": None,
                "num_inference_steps": 8,
                "guidance_scale": 3.0,
            }
        ),
        encoding="utf-8",
    )

    code = validate_local._run_single_case(manifest_path, result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["status"] == "succeeded"
    assert payload["case_id"] == "01_t23d"
    assert payload["mode"] == "t23d"
    assert payload["vertex_count"] == 12


def test_run_single_case_writes_failure_result(monkeypatch, tmp_path) -> None:
    class _FakeManager:
        def __init__(self, backend_id=None):
            self.backend_id = backend_id

        def i23d(self, image, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(validate_local, "Scene3DManager", _FakeManager)
    manifest_path = tmp_path / "case_manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "step1x",
                "case_id": "03_i23d",
                "mode": "i23d",
                "prompt": "owl",
                "image": "/tmp/owl.png",
                "output_dir": str(tmp_path / "03_i23d"),
                "mc_resolution": 128,
                "device": "cpu",
                "model": None,
                "model_subfolder": None,
                "num_inference_steps": 8,
                "guidance_scale": 3.0,
                "remove_background": None,
            }
        ),
        encoding="utf-8",
    )

    code = validate_local._run_single_case(manifest_path, result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["status"] == "failed"
    assert "RuntimeError" in payload["error"]
    assert Path(payload["contact_sheet_path"]).exists()


def test_run_single_case_forwards_texture_controls(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def __init__(self, backend_id=None):
            self.backend_id = backend_id

        def i23d(self, image, **kwargs):
            calls["image"] = image
            calls.update(kwargs)
            case_dir = Path(kwargs["output_dir"])
            contact_sheet = case_dir / "contact_sheet.png"
            contact_sheet.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), "white").save(contact_sheet)
            return {
                "metadata": {
                    "backend_id": self.backend_id,
                    "contact_sheet_path": str(contact_sheet),
                    "timings_s": {"total": 1.0},
                    "memory": {"rss_bytes": 1024},
                    "vertex_count": 12,
                    "face_count": 18,
                }
            }

    monkeypatch.setattr(validate_local, "Scene3DManager", _FakeManager)
    manifest_path = tmp_path / "case_manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "triposr",
                "case_id": "03_i23d",
                "mode": "i23d",
                "prompt": "rocket",
                "image": "/tmp/rocket.png",
                "output_dir": str(tmp_path / "03_i23d"),
                "mc_resolution": 256,
                "texture_mode": "baked_basecolor",
                "texture_resolution": 2048,
                "device": "cpu",
                "model": None,
                "model_subfolder": None,
                "num_inference_steps": None,
                "guidance_scale": None,
                "remove_background": None,
            }
        ),
        encoding="utf-8",
    )

    code = validate_local._run_single_case(manifest_path, result_path)

    assert code == 0
    assert calls["texture_mode"] == "baked_basecolor"
    assert calls["texture_resolution"] == 2048


def test_run_case_subprocess_kills_case_when_rss_limit_is_exceeded(monkeypatch, tmp_path) -> None:
    class _FakeProcess:
        def __init__(self):
            self.pid = 4242
            self.returncode = None

        def poll(self):
            return None

        def communicate(self, timeout=None):
            self.returncode = -15
            return ("child stdout", "child stderr")

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    fake_process = _FakeProcess()
    monkeypatch.setattr(validate_local.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(validate_local, "_process_tree_rss_bytes", lambda pid: 2 * (1024 ** 3))
    monkeypatch.setattr(validate_local.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(validate_local.time, "sleep", lambda seconds: None)

    row = validate_local._run_case_subprocess(
        case={
            "case_id": "03_i23d",
            "mode": "i23d",
            "backend": "step1x",
            "prompt": "espresso",
            "output_dir": str(tmp_path / "03_i23d"),
            "image": "/tmp/espresso.png",
        },
        timeout_s=60,
        rss_limit_gb=1.0,
        resume=False,
    )

    assert row["status"] == "memory_guard_killed"
    assert "RSS limit" in row["error"]
    assert row["observed_rss_gib"] == 2.0
    assert row["exit_code"] == -15
    assert Path(row["contact_sheet_path"]).exists()
