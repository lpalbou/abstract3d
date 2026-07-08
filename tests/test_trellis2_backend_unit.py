from __future__ import annotations

import io
from pathlib import Path

import torch
import pytest

from PIL import Image

from abstract3d.errors import CapabilityNotSupportedError, DependencyUnavailableError, SourceBootstrapError
from abstract3d.backends.trellis2_runtime import (
    _configured_dino_source,
    _download_dino_model,
    _local_flexible_dual_grid_to_mesh,
    _resolve_source_dir,
    _select_device,
    _variant_for_model,
    Trellis2LocalBackend,
)


def test_select_device_falls_back_when_requested_accelerator_is_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Explicit accelerator requests must not return an unavailable device on
    # hosts without that accelerator (e.g. `mps` on Linux/Windows).
    assert _select_device(None, explicit="mps") == "cpu"
    assert _select_device(None, explicit="cuda") == "cpu"
    assert _select_device(None, explicit="cpu") == "cpu"
    assert _select_device(None) == "cpu"


def test_select_device_honors_available_accelerator(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _select_device(None, explicit="mps") == "mps"
    assert _select_device(None) == "mps"
    # An explicit cuda request on an mps host falls back to the best
    # available accelerator instead of crashing at tensor-placement time.
    assert _select_device(None, explicit="cuda") == "mps"


def test_variant_for_model_accepts_only_official_repo() -> None:
    variant = _variant_for_model("microsoft/TRELLIS.2-4B")
    assert variant["repo_id"] == "microsoft/TRELLIS.2-4B"
    assert variant["precision"] == "bf16/fp16"

    with pytest.raises(CapabilityNotSupportedError):
        _variant_for_model("visualbruno/TRELLIS.2-4B-FP8")


def test_configured_dino_source_accepts_official_repo_or_local_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ABSTRACT3D_TRELLIS2_DINO_MODEL", raising=False)
    assert _configured_dino_source(None) == "facebook/dinov3-vitl16-pretrain-lvd1689m"

    local_snapshot = tmp_path / "dinov3"
    local_snapshot.mkdir()
    for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
        (local_snapshot / name).write_bytes(b"x")
    monkeypatch.setenv("ABSTRACT3D_TRELLIS2_DINO_MODEL", str(local_snapshot))
    assert _configured_dino_source(None) == str(local_snapshot.resolve())


def test_configured_dino_source_rejects_unofficial_repo(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACT3D_TRELLIS2_DINO_MODEL", "someone/dinov3-int8")
    with pytest.raises(CapabilityNotSupportedError):
        _configured_dino_source(None)


def test_download_dino_model_wraps_gated_access_with_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_TRELLIS2_DINO_MODEL", raising=False)

    def _raise(*args, **kwargs):
        raise RuntimeError("403 gated")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _raise)
    with pytest.raises(DependencyUnavailableError, match="official gated DINOv3 companion model"):
        _download_dino_model(None)


def test_resolve_source_dir_accepts_only_verified_official_override(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "trellis2-official"
    package_dir = source_dir / "trellis2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / ".abstract3d-source.json").write_text(
        (
            '{\n'
            '  "repo_url": "https://github.com/microsoft/TRELLIS.2",\n'
            '  "commit": "75fbf0183001ed9876c8dbb35de6b68552ee08bd"\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ABSTRACT3D_TRELLIS2_SOURCE_DIR", str(source_dir))
    assert _resolve_source_dir(None) == source_dir.resolve()


def test_resolve_source_dir_rejects_unverified_override(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "trellis2-unverified"
    package_dir = source_dir / "trellis2"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("ABSTRACT3D_TRELLIS2_SOURCE_DIR", str(source_dir))
    with pytest.raises(SourceBootstrapError, match="not verifiable as an official pinned snapshot"):
        _resolve_source_dir(None)


def test_load_runtime_rejects_unofficial_model_before_source_bootstrap(monkeypatch) -> None:
    backend = Trellis2LocalBackend(owner=None)
    called = {"resolve_source_dir": False}

    def _fail(*args, **kwargs):
        called["resolve_source_dir"] = True
        raise AssertionError("source bootstrap should not run")

    monkeypatch.setattr("abstract3d.backends.trellis2_runtime._resolve_source_dir", _fail)
    with pytest.raises(CapabilityNotSupportedError):
        backend._load_runtime(model_id="visualbruno/TRELLIS.2-4B-FP8", device="cpu")
    assert called["resolve_source_dir"] is False


def test_t23d_passes_image_generation_kwargs_to_composition(monkeypatch) -> None:
    backend = Trellis2LocalBackend(owner=None)
    seen: dict[str, object] = {}

    def _load_runtime(*, model_id=None, device=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "microsoft/TRELLIS.2-4B"
        backend._last_runtime_stats = {"load_s": 0.12}
        return {"variant": {"precision": "bf16/fp16"}}

    def _make_source_image(prompt: str, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = dict(kwargs)
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    class _Mesh:
        vertices = [0, 1, 2]
        faces = [0, 1]

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(backend, "_make_source_image", _make_source_image)
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime._prepare_image", lambda image, **kwargs: (Image.new("RGBA", (8, 8)), Image.new("RGB", (8, 8)), False))
    monkeypatch.setattr(backend, "_get_cond", lambda runtime, source_image: {})
    monkeypatch.setattr(backend, "_sample_sparse_structure", lambda runtime, cond: object())
    monkeypatch.setattr(backend, "_sample_shape_slat", lambda runtime, cond, coords: object())
    monkeypatch.setattr(backend, "_decode_shape", lambda runtime, slat: [_Mesh()])
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime._mesh_export_bytes", lambda mesh, *, file_type: b"mesh")
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime._mesh_to_trimesh", lambda mesh: object())
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime.render_mesh_views", lambda mesh: [Image.new("RGB", (8, 8))])

    out = backend.t23d(
        "desk lamp",
        image_provider="mlx-gen",
        image_model="AbstractFramework/flux.2-klein-4b-8bit",
        image_width=640,
        image_height=512,
        image_seed=7,
    )

    assert out["format"] == "glb"
    assert seen["prompt"] == "desk lamp"
    assert seen["kwargs"] == {
        "image_provider": "mlx-gen",
        "image_model": "AbstractFramework/flux.2-klein-4b-8bit",
        "image_width": 640,
        "image_height": 512,
        "image_seed": 7,
    }


def test_trellis2_discovery_hides_t23d_when_composer_is_unavailable(monkeypatch) -> None:
    backend = Trellis2LocalBackend(owner=None)
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime.has_image_composer", lambda owner: False)
    monkeypatch.setattr("abstract3d.backends.trellis2_runtime.importlib.util.find_spec", lambda name: object())

    providers = backend.available_providers()
    operations = backend.list_operations()

    assert providers[0]["tasks"] == ["image_to_scene3d"]
    assert providers[0]["metadata"]["composition_ready"] is False
    assert [item["task"] for item in operations] == ["image_to_scene3d"]


def test_local_flexible_dual_grid_to_mesh_builds_quad_as_two_triangles() -> None:
    coords = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=torch.int32,
    )
    dual_vertices = torch.zeros((4, 3), dtype=torch.float32)
    intersected = torch.tensor(
        [
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    vertices, faces = _local_flexible_dual_grid_to_mesh(
        coords=coords,
        dual_vertices=dual_vertices,
        intersected_flag=intersected,
        split_weight=None,
        aabb=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        voxel_size=1.0,
        train=False,
    )

    assert vertices.shape == (4, 3)
    assert faces.shape == (2, 3)
