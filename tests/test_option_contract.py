"""The strict generation-option contract: unknown options fail loudly.

Backends previously ignored unrecognized keyword options, so a caller could
not distinguish "tuned and applied" from "tuned and ignored" (the CLI itself
sent diffusion knobs to the feed-forward TripoSR path). Every backend now
rejects leftovers with `InvalidRequestError` after consuming its supported
options, and the CLI forwards only explicitly-set flags.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from abstract3d import cli
from abstract3d.backends import reject_unknown_options, triposr_runtime
from abstract3d.errors import InvalidRequestError


def _png_image() -> Image.Image:
    return Image.new("RGBA", (64, 64), (180, 160, 140, 255))


class _FakeRuntime:
    def __call__(self, images, device="cpu"):
        return ["scene-code"]

    def extract_mesh(self, scene_codes, has_vertex_color, resolution=256):
        raise AssertionError("mesh extraction must not run for an invalid request")


def test_reject_unknown_options_lists_offenders() -> None:
    with pytest.raises(InvalidRequestError, match="definitely_a_typo, guidance_scale"):
        reject_unknown_options(
            "abstract3d:triposr",
            {"guidance_scale": 5.0, "definitely_a_typo": 1},
        )


def test_reject_unknown_options_allows_envelope_keys() -> None:
    reject_unknown_options(
        "abstract3d:triposr",
        {"artifact_store": object(), "run_id": "r1", "tags": {}, "metadata": {}},
    )


def test_triposr_rejects_diffusion_options(monkeypatch, tmp_path) -> None:
    backend = triposr_runtime.TripoSRBackend(owner=None)

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "stabilityai/TripoSR"
        backend._resident_chunk_size = chunk_size or 2048
        backend._last_runtime_stats = {"load_s": 0.1}
        return _FakeRuntime()

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(
        triposr_runtime, "_prepare_triposr_image",
        lambda *args, **kwargs: (
            Image.new("RGB", (64, 64), "white"),
            Image.new("RGB", (64, 64), "white"),
            False,
        ),
    )
    with pytest.raises(InvalidRequestError, match="guidance_scale"):
        backend.i23d(
            _png_image(),
            output_dir=str(tmp_path),
            device="cpu",
            texture_mode="vertex_color",
            guidance_scale=7.5,
        )


def test_triposr_rejects_typos_before_mesh_extraction(monkeypatch, tmp_path) -> None:
    backend = triposr_runtime.TripoSRBackend(owner=None)

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._last_runtime_stats = {"load_s": 0.1}
        return _FakeRuntime()

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(
        triposr_runtime, "_prepare_triposr_image",
        lambda *args, **kwargs: (
            Image.new("RGB", (64, 64), "white"),
            Image.new("RGB", (64, 64), "white"),
            False,
        ),
    )
    with pytest.raises(InvalidRequestError, match="texure_mode"):
        backend.i23d(
            _png_image(), output_dir=str(tmp_path), device="cpu",
            texure_mode="baked_basecolor",  # deliberate typo
        )


def test_cli_forwards_only_explicit_options(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    class _FakeManager:
        def __init__(self, backend_id=None):
            captured["backend_id"] = backend_id

        def i23d(self, image, **kwargs):
            captured["image"] = image
            captured["kwargs"] = kwargs
            return {"metadata": {}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)
    rc = cli.main([
        "i23d", "object.png",
        "--output-dir", str(tmp_path),
        "--backend", "triposr",
    ])
    assert rc == 0
    kwargs = captured["kwargs"]
    # No None-valued or backend-foreign tuning options leak through.
    assert "guidance_scale" not in kwargs
    assert "octree_resolution" not in kwargs
    assert "mc_resolution" not in kwargs
    assert "chunk_size" not in kwargs
    assert kwargs["output_dir"] == str(tmp_path)


def test_cli_exposes_density_controls(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    class _FakeManager:
        def __init__(self, backend_id=None):
            pass

        def i23d(self, image, **kwargs):
            captured["kwargs"] = kwargs
            return {"metadata": {}}

    monkeypatch.setattr(cli, "Scene3DManager", _FakeManager)
    rc = cli.main([
        "i23d", "object.png",
        "--output-dir", str(tmp_path),
        "--backend", "hunyuan3d21",
        "--octree-resolution", "512",
        "--max-facenum", "200000",
        "--num-inference-steps", "20",
    ])
    assert rc == 0
    assert captured["kwargs"]["octree_resolution"] == 512
    assert captured["kwargs"]["max_facenum"] == 200000
    assert captured["kwargs"]["num_inference_steps"] == 20
