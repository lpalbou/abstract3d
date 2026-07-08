from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.texture import SimpleMaterial, TextureVisuals

from abstract3d.backends import triposr_runtime as runtime


class _FakeRuntime:
    def __call__(self, images, device: str):
        _ = images, device
        return "scene-codes"

    def extract_mesh(self, scene_codes, with_texture: bool, resolution: int):
        _ = scene_codes, with_texture, resolution
        mesh = type(
            "_Mesh",
            (),
            {
                "vertices": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            },
        )()
        return [mesh]


def _png_bytes(color: tuple[int, int, int] = (210, 180, 140)) -> bytes:
    image = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_checkpoint_key_remap_matches_transformers_v5_layout() -> None:
    assert runtime._remap_triposr_checkpoint_key(
        "image_tokenizer.model.encoder.layer.11.attention.attention.query.weight"
    ) == "image_tokenizer.model.layers.11.attention.q_proj.weight"
    assert runtime._remap_triposr_checkpoint_key("decoder.weight") == "decoder.weight"


def test_resolve_source_dir_prefers_configured_checkout(monkeypatch, tmp_path) -> None:
    source_dir = tmp_path / "triposr"
    (source_dir / "tsr").mkdir(parents=True)
    (source_dir / "tsr" / "system.py").write_text("# test\n", encoding="utf-8")
    monkeypatch.setenv("ABSTRACT3D_TRIPOSR_SOURCE_DIR", str(source_dir))

    resolved = runtime._resolve_source_dir(owner=None)

    assert resolved == source_dir.resolve()


def test_generate_routes_task_aliases_without_loading_real_runtime(monkeypatch) -> None:
    backend = runtime.TripoSRBackend(owner=None)
    seen: list[tuple[str, object]] = []

    monkeypatch.setattr(backend, "t23d", lambda prompt, **kwargs: seen.append(("t23d", prompt)) or {"ok": "t23d"})
    monkeypatch.setattr(
        backend,
        "i23d",
        lambda image, **kwargs: seen.append(("i23d", image)) or {"ok": "i23d"},
    )

    assert backend.generate("cube", task="t23d") == {"ok": "t23d"}
    assert backend.generate("caption", task="i23d", image="object.png") == {"ok": "i23d"}
    assert seen == [("t23d", "cube"), ("i23d", "object.png")]


def test_i23d_writes_bundle_and_updates_metadata(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None)

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "stabilityai/TripoSR"
        backend._resident_chunk_size = chunk_size or 2048
        backend._last_runtime_stats = {"load_s": 0.1234}
        return _FakeRuntime()

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(runtime, "_prepare_triposr_image", lambda *args, **kwargs: (Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white"), False))
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: f"{file_type}-payload".encode("utf-8"))
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (96, 96), "#d2c7b8") for _ in range(4)])

    out = backend.i23d(
        Image.open(io.BytesIO(_png_bytes())),
        output_dir=str(tmp_path),
        mc_resolution=96,
        device="cpu",
        texture_mode="vertex_color",
    )

    metadata = out["metadata"]
    metadata_path = Path(metadata["metadata_path"])
    bundle_dir = Path(metadata["bundle_dir"])

    assert out["format"] == "glb"
    assert metadata["mc_resolution"] == 96
    assert metadata["timings_s"]["load"] == 0.1234
    assert metadata["timings_s"]["source_image_generation"] is None
    assert metadata["output_bytes"] == len(out["data"])
    assert bundle_dir == tmp_path.resolve()
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["output_bytes"] == len(out["data"])


def test_t23d_records_source_image_generation_time(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None, image_generator=lambda prompt, **kwargs: _png_bytes())

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "stabilityai/TripoSR"
        backend._resident_chunk_size = chunk_size or 2048
        backend._last_runtime_stats = {"load_s": 0.2}
        return _FakeRuntime()

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(runtime, "_prepare_triposr_image", lambda *args, **kwargs: (Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white"), False))
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: f"{file_type}-payload".encode("utf-8"))
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (96, 96), "#d2c7b8") for _ in range(4)])

    out = backend.t23d("a matte ceramic mug", output_dir=str(tmp_path), texture_mode="vertex_color")

    assert out["metadata"]["timings_s"]["source_image_generation"] is not None
    assert Path(out["metadata"]["contact_sheet_path"]).exists()


def test_make_source_image_does_not_force_mlx_provider_when_unconfigured() -> None:
    seen: dict[str, object] = {}

    def _generator(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = dict(kwargs)
        return _png_bytes()

    backend = runtime.TripoSRBackend(owner=None, image_generator=_generator)

    payload = backend._make_source_image("a matte ceramic mug")

    assert isinstance(payload, bytes)
    assert "provider" not in seen["kwargs"]
    assert "model" not in seen["kwargs"]
    assert seen["kwargs"]["width"] == 768
    assert seen["kwargs"]["height"] == 768


def test_make_source_image_uses_scene3d_image_config_when_present() -> None:
    seen: dict[str, object] = {}

    def _generator(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = dict(kwargs)
        return _png_bytes()

    owner = type(
        "_Owner",
        (),
        {
            "config": {
                "scene3d_image_provider": "openai-compatible",
                "scene3d_image_model": "gpt-image-1",
                "scene3d_image_width": "640",
                "scene3d_image_height": "512",
                "scene3d_image_seed": "7",
            }
        },
    )()
    backend = runtime.TripoSRBackend(owner=owner, image_generator=_generator)

    payload = backend._make_source_image("a matte ceramic mug")

    assert isinstance(payload, bytes)
    assert seen["kwargs"] == {
        "provider": "openai-compatible",
        "model": "gpt-image-1",
        "width": 640,
        "height": 512,
        "seed": 7,
    }


def test_triposr_discovery_hides_t23d_when_composer_is_unavailable(monkeypatch) -> None:
    backend = runtime.TripoSRBackend(owner=None)
    monkeypatch.setattr(runtime, "has_image_composer", lambda owner: False)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: object())

    providers = backend.available_providers()
    operations = backend.list_operations()

    assert providers[0]["tasks"] == ["image_to_scene3d"]
    assert providers[0]["metadata"]["composition_ready"] is False
    assert [item["task"] for item in operations] == ["image_to_scene3d"]


def test_i23d_defaults_to_mc256_and_records_cleanup(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None)
    seen: dict[str, object] = {}

    class _RuntimeWithResolution(_FakeRuntime):
        def extract_mesh(self, scene_codes, with_texture: bool, resolution: int):
            seen["resolution"] = resolution
            return super().extract_mesh(scene_codes, with_texture, resolution)

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "stabilityai/TripoSR"
        backend._resident_chunk_size = chunk_size or 2048
        backend._last_runtime_stats = {"load_s": 0.1234}
        return _RuntimeWithResolution()

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(runtime, "_prepare_triposr_image", lambda *args, **kwargs: (Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white"), False))
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: f"{file_type}-payload".encode("utf-8"))
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (96, 96), "#d2c7b8") for _ in range(4)])
    monkeypatch.setattr(
        runtime,
        "_tripo_postprocess_mesh",
        lambda mesh, *, cleanup_mode: (
            mesh,
            ["marching_cube_cleanup", "taubin_smooth:4"],
            [],
            {
                "settings": {"min_component_faces": 32, "hole_size": 24, "smooth_steps": 4},
                "topology_before": {"body_count": 2},
                "topology_after": {"body_count": 1},
            },
        ),
    )

    out = backend.i23d(
        Image.open(io.BytesIO(_png_bytes())),
        output_dir=str(tmp_path),
        device="cpu",
        texture_mode="vertex_color",
    )

    metadata = out["metadata"]
    assert seen["resolution"] == 256
    assert metadata["mc_resolution"] == 256
    assert metadata["cleanup_mode"] == "presentation"
    assert metadata["postprocess_cleanup"] == ["marching_cube_cleanup", "taubin_smooth:4"]
    assert metadata["surface_cleanup"] == {"min_component_faces": 32, "hole_size": 24, "smooth_steps": 4}
    assert metadata["topology"]["body_count"] == 1
    assert metadata["topology_before_cleanup"]["body_count"] == 2


def test_i23d_baked_texture_writes_texture_artifacts_and_uses_non_vertex_extract(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None)
    seen: dict[str, object] = {}

    class _RuntimeWithTextureFlag(_FakeRuntime):
        def extract_mesh(self, scene_codes, with_texture: bool, resolution: int):
            seen["with_texture"] = with_texture
            seen["resolution"] = resolution
            mesh = trimesh.Trimesh(
                vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                faces=np.asarray([[0, 1, 2]], dtype=np.int32),
                process=False,
            )
            return [mesh]

    def _load_runtime(*, model_id=None, device=None, chunk_size=None):
        backend._resident_device = "cpu"
        backend._resident_model_id = model_id or "stabilityai/TripoSR"
        backend._resident_chunk_size = chunk_size or 2048
        backend._last_runtime_stats = {"load_s": 0.1234}
        return _RuntimeWithTextureFlag()

    def _fake_bake(
        mesh,
        *,
        model,
        scene_code,
        texture_resolution,
        texture_completion="none",
        observed_views=None,
        observed_rgba=None,
    ):
        image = Image.new("RGBA", (16, 16), (220, 30, 30, 255))
        textured = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            process=False,
            visual=TextureVisuals(
                uv=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                material=SimpleMaterial(image=image),
                image=image,
            ),
        )
        assert observed_views is not None or observed_rgba is not None
        assert texture_completion == "none"
        return textured, {
            "texture_image": image,
            "uv_preview": Image.new("RGBA", (16, 16), (255, 255, 255, 255)),
            "texture_padding": 4,
            "projection_mode": "hybrid_observed_view_plus_triplane",
            "observed_coverage_ratio": 0.42,
            "observed_view_stats": [{"label": "front", "coverage_ratio": 0.42}],
            "texture_completion": texture_completion,
            "symmetry_completion": {"mode": "none", "coverage_ratio": 0.0, "applied": False},
            "uv_vertex_count": 3,
            "vertex_mapping_count": 3,
        }

    monkeypatch.setattr(backend, "_load_runtime", _load_runtime)
    monkeypatch.setattr(runtime, "_prepare_triposr_image", lambda *args, **kwargs: (Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white"), False))
    monkeypatch.setattr(runtime, "_tripo_bake_textured_mesh", _fake_bake)
    monkeypatch.setattr(runtime, "_tripo_export_obj_with_textures", lambda mesh: (b"obj-payload", {"scene.mtl": b"mtl-payload"}))
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (96, 96), "#d2c7b8") for _ in range(4)])

    out = backend.i23d(
        Image.open(io.BytesIO(_png_bytes())),
        output_dir=str(tmp_path),
        device="cpu",
        texture_mode="baked_basecolor",
        texture_resolution=1024,
    )

    metadata = out["metadata"]
    assert seen["with_texture"] is False
    assert seen["resolution"] == 256
    assert metadata["appearance_mode"] == "uv_basecolor"
    assert metadata["texture_mode"] == "baked_basecolor"
    assert metadata["texture_resolution"] == 1024
    assert metadata["uv_present"] is True
    assert metadata["material_count"] == 1
    assert metadata["timings_s"]["texture"] is not None
    assert metadata["texture_artifacts"]["projection_mode"] == "hybrid_observed_view_plus_triplane"
    assert metadata["texture_artifacts"]["observed_view_stats"] == [{"label": "front", "coverage_ratio": 0.42}]
    assert metadata["texture_artifacts"]["reference_view_count"] == 0
    assert metadata["texture_artifacts"]["texture_completion"] == "none"
    assert Path(metadata["texture_artifacts"]["texture_path"]).exists()
    assert Path(metadata["texture_artifacts"]["uv_preview_path"]).exists()
    assert Path(metadata["texture_artifacts"]["geometry_glb_path"]).exists()
    assert (tmp_path / "scene.mtl").exists()


def test_texture_edge_bleed_removes_transparent_black_padding() -> None:
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (220, 40, 30, 255))

    baked = runtime._tripo_edge_bleed_texture(image)

    assert baked.mode == "RGB"
    assert baked.getpixel((0, 0)) != (0, 0, 0)


def test_project_observed_texture_prefers_front_facing_visible_texels() -> None:
    observed = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y in range(16):
        for x in range(16):
            observed.putpixel((x, y), (255, 64, 32, 255))

    positions = np.zeros((4, 4, 4), dtype=np.float32)
    normals = np.zeros((4, 4, 4), dtype=np.float32)
    positions[:, :, 3] = 1.0
    normals[:, :, 3] = 1.0
    positions[:, :, 0] = 0.0
    normals[:, :, 0] = 1.0

    projection = runtime._tripo_project_observed_texture(
        observed,
        positions_texture=positions,
        normals_texture=normals,
    )

    assert projection["coverage_ratio"] > 0.0
    assert float(np.max(projection["weight"])) > 0.0
    assert projection["rgba"][0, 0, 0] > 0.9


def test_blend_observed_texture_merges_multiple_views() -> None:
    base = np.zeros((2, 2, 4), dtype=np.float32)
    base[:, :, 3] = 1.0
    front = np.zeros((2, 2, 4), dtype=np.float32)
    front[:, :, 0] = 1.0
    front[:, :, 3] = 1.0
    side = np.zeros((2, 2, 4), dtype=np.float32)
    side[:, :, 1] = 1.0
    side[:, :, 3] = 1.0

    blended, stats = runtime._tripo_blend_observed_texture(
        base,
        [
            {
                "rgba": front,
                "weight": np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                "coverage_ratio": 0.25,
                "label": "front",
            },
            {
                "rgba": side,
                "weight": np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
                "coverage_ratio": 0.25,
                "label": "side_right",
            },
        ],
    )

    assert blended[0, 0, 0] > 0.9
    assert blended[0, 1, 1] > 0.9
    assert stats["coverage_ratio"] == 0.5
    assert [row["label"] for row in stats["view_stats"]] == ["front", "side_right"]
    assert stats["weight_map"][0, 0] == 1.0


def test_mirror_symmetry_projection_targets_front_lateral_surfaces() -> None:
    observed = Image.new("RGBA", (32, 32), (240, 200, 180, 255))
    positions = np.zeros((2, 2, 4), dtype=np.float32)
    normals = np.zeros((2, 2, 4), dtype=np.float32)
    positions[:, :, 3] = 1.0
    normals[:, :, 3] = 1.0
    positions[:, :, 0] = np.asarray([[0.9, 0.9], [0.2, 0.2]], dtype=np.float32)
    positions[:, :, 1] = np.asarray([[0.3, -0.3], [0.02, -0.02]], dtype=np.float32)
    normals[:, :, 0] = 1.0

    projection = runtime._tripo_project_mirror_symmetry_texture(
        observed,
        positions_texture=positions,
        normals_texture=normals,
    )

    assert projection["label"] == "mirror_symmetry"
    assert float(projection["weight"][0, 0]) > float(projection["weight"][1, 0])
    assert projection["coverage_ratio"] > 0.0


def test_bake_textured_mesh_applies_mirror_symmetry_completion(monkeypatch) -> None:
    image = Image.new("RGBA", (8, 8), (220, 30, 30, 255))
    mesh = trimesh.Trimesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        process=False,
    )
    monkeypatch.setattr(
        runtime,
        "_tripo_make_texture_atlas",
        lambda mesh, texture_resolution, texture_padding: {
            "vmapping": np.asarray([0, 1, 2], dtype=np.int64),
            "indices": np.asarray([[0, 1, 2]], dtype=np.int64),
            "uvs": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        runtime,
        "_tripo_rasterize_position_atlas",
        lambda *args, **kwargs: np.dstack(
            [
                np.ones((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ]
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_tripo_rasterize_normal_atlas",
        lambda *args, **kwargs: np.dstack(
            [
                np.ones((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ]
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_tripo_positions_to_colors",
        lambda model, scene_code, positions_texture, texture_resolution: np.dstack(
            [
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ]
        ),
    )
    # Half-covered projection: only the left half of the atlas is observed,
    # so the mirror completion has real unseen texels to fill.
    half_weight = np.zeros((4, 4), dtype=np.float32)
    half_weight[:, :2] = 0.9
    monkeypatch.setattr(
        runtime,
        "_tripo_project_observed_texture",
        lambda *args, **kwargs: {
            "rgba": np.dstack(
                [
                    np.ones((4, 4), dtype=np.float32),
                    np.zeros((4, 4), dtype=np.float32),
                    np.zeros((4, 4), dtype=np.float32),
                    np.ones((4, 4), dtype=np.float32),
                ]
            ),
            "weight": half_weight,
            "coverage_ratio": 0.5,
            "label": "front",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
        },
    )
    from abstract3d import texturing as texturing_module

    monkeypatch.setattr(texturing_module, "register_view_2d", lambda mesh, observed_rgba, **kwargs: (observed_rgba, {"applied": False}))
    monkeypatch.setattr(
        texturing_module,
        "refine_registration_photometric",
        lambda mesh, observed_rgba, **kwargs: (observed_rgba, {"applied": False}),
    )
    monkeypatch.setattr(texturing_module, "erode_view_alpha", lambda observed_rgba, **kwargs: observed_rgba)
    monkeypatch.setattr(texturing_module, "mesh_mirror_symmetry_score", lambda mesh, **kwargs: 0.9)

    def _fake_mirror_fill(*, positions_texture, observed_mask, colors_rgb, **kwargs):
        fill_rgb = np.zeros_like(np.asarray(colors_rgb, dtype=np.float32))
        fill_mask = ~np.asarray(observed_mask, dtype=bool)
        fill_rgb[fill_mask] = [0.0, 1.0, 0.0]
        return fill_rgb, fill_mask

    monkeypatch.setattr(texturing_module, "mirror_fill_from_observed", _fake_mirror_fill)

    _mesh, artifacts = runtime._tripo_bake_textured_mesh(
        mesh,
        model=object(),
        scene_code=object(),
        texture_resolution=512,
        texture_completion="mirror_symmetry",
        observed_views=[{"rgba": image, "label": "front", "azimuth_deg": 0.0, "elevation_deg": 0.0}],
    )

    assert artifacts["texture_completion"] == "mirror_symmetry"
    assert artifacts["symmetry_completion"]["applied"] is True
    assert artifacts["projection_mode"] == "hybrid_observed_plus_symmetry_plus_triplane"


def test_prepare_texture_reference_views_keeps_source_and_additional_views(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_prepare_triposr_image",
        lambda image, **kwargs: (
            Image.new("RGB", (16, 16), "white"),
            Image.new("RGB", (16, 16), "white"),
            True,
            Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
        ),
    )

    prepared = runtime._prepare_texture_reference_views(
        source_observed_rgba=Image.new("RGBA", (16, 16), (0, 255, 0, 255)),
        source_preview=Image.new("RGB", (16, 16), "gray"),
        texture_reference_views=[{"image": b"fake", "label": "side_right", "azimuth_deg": -90.0, "elevation_deg": 0.0}],
        texture_reference_remove_background=None,
        foreground_ratio=0.85,
        artifact_store=None,
    )

    assert [row["label"] for row in prepared] == ["front", "side_right"]
    assert prepared[0]["role"] == "source"
    assert prepared[1]["role"] == "reference"


def test_validate_suite_writes_summary_outputs(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None)

    def _fake_case(*, case_dir: Path, mode: str) -> dict:
        case_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet = case_dir / "contact_sheet.png"
        Image.new("RGB", (120, 80), "#c0d6df").save(contact_sheet)
        return {
            "metadata": {
                "contact_sheet_path": str(contact_sheet),
                "vertex_count": 123,
                "mode": mode,
            }
        }

    monkeypatch.setattr(
        backend,
        "t23d",
        lambda prompt, **kwargs: _fake_case(case_dir=Path(kwargs["output_dir"]), mode="t23d"),
    )
    monkeypatch.setattr(
        backend,
        "i23d",
        lambda image, **kwargs: _fake_case(case_dir=Path(kwargs["output_dir"]), mode="i23d"),
    )

    summary = backend.validate_suite(
        prompts=["teapot"],
        images=[str(tmp_path / "input.png")],
        output_dir=str(tmp_path / "validation"),
    )

    assert Path(summary["contact_sheet"]).exists()
    assert Path(summary["stats"]).exists()
    rows = json.loads(Path(summary["stats"]).read_text(encoding="utf-8"))
    assert [row["case_id"] for row in rows] == ["01_t23d", "02_i23d"]


def test_validate_suite_forwards_model_and_generation_controls(monkeypatch, tmp_path) -> None:
    backend = runtime.TripoSRBackend(owner=None)
    calls: dict[str, dict] = {}

    def _fake_t23d(prompt, **kwargs):
        calls["t23d"] = {"prompt": prompt, **kwargs}
        case_dir = Path(kwargs["output_dir"])
        case_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet = case_dir / "contact_sheet.png"
        Image.new("RGB", (32, 32), "white").save(contact_sheet)
        return {"metadata": {"contact_sheet_path": str(contact_sheet)}}

    def _fake_i23d(image, **kwargs):
        calls["i23d"] = {"image": image, **kwargs}
        case_dir = Path(kwargs["output_dir"])
        case_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet = case_dir / "contact_sheet.png"
        Image.new("RGB", (32, 32), "white").save(contact_sheet)
        return {"metadata": {"contact_sheet_path": str(contact_sheet)}}

    monkeypatch.setattr(backend, "t23d", _fake_t23d)
    monkeypatch.setattr(backend, "i23d", _fake_i23d)

    backend.validate_suite(
        prompts=["teapot"],
        images=[str(tmp_path / "rocket.png")],
        output_dir=str(tmp_path / "validation"),
        model="stabilityai/TripoSR",
        model_subfolder="ignored-subfolder",
        num_inference_steps=12,
        guidance_scale=6.5,
        chunk_size=1024,
        texture_mode="baked_basecolor",
    )

    assert calls["t23d"]["model"] == "stabilityai/TripoSR"
    assert calls["t23d"]["num_inference_steps"] == 12
    assert calls["t23d"]["guidance_scale"] == 6.5
    assert calls["t23d"]["chunk_size"] == 1024
    assert calls["t23d"]["texture_mode"] == "baked_basecolor"
    assert calls["i23d"]["model"] == "stabilityai/TripoSR"
    assert calls["i23d"]["num_inference_steps"] == 12
    assert calls["i23d"]["guidance_scale"] == 6.5
    assert calls["i23d"]["chunk_size"] == 1024


def test_postprocess_recovers_outward_normals_from_inverted_input() -> None:
    # Marching-cubes output on this stack arrives with inward winding. The
    # presentation cleanup must hand back outward face winding AND outward
    # vertex normals (a stale trimesh normal cache after repair.fix_normals
    # previously kept inward vertex normals despite the corrected winding).
    sphere = trimesh.creation.icosphere(subdivisions=3)
    inverted = trimesh.Trimesh(
        sphere.vertices.copy(), sphere.faces[:, ::-1].copy(), process=False
    )

    cleaned, applied, warnings, _details = runtime._tripo_postprocess_mesh(
        inverted, cleanup_mode="presentation"
    )

    assert "fix_normals" in applied
    assert float(cleaned.volume) > 0
    vertices = np.asarray(cleaned.vertices)
    normals = np.asarray(cleaned.vertex_normals)
    outward = ((vertices - vertices.mean(axis=0)) * normals).sum(axis=1)
    assert float((outward > 0).mean()) > 0.95


def test_marching_cubes_fallback_matches_torchmcubes_axis_convention(monkeypatch) -> None:
    import sys

    import torch

    # Force the pure-Python fallback module even if a native torchmcubes
    # build exists in the environment.
    monkeypatch.delitem(sys.modules, "torchmcubes", raising=False)
    real_import = runtime.importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "torchmcubes":
            raise ImportError("forced fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", _fake_import)
    runtime._ensure_torchmcubes_compat_module()
    module = sys.modules["torchmcubes"]
    assert getattr(module, "_abstract3d_fallback", False)

    # Anisotropic ellipsoid: longest along axis 0, shortest along axis 2 in
    # index space. After the TripoSR helper post-processing (swap + scale),
    # the X extent must be largest and the mesh volume positive (outward
    # winding), exactly as with native torchmcubes.
    res = 48
    ii, jj, kk = np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing="ij")
    center = (res - 1) / 2
    d = ((ii - center) / 20.0) ** 2 + ((jj - center) / 12.0) ** 2 + ((kk - center) / 7.0) ** 2
    density = torch.tensor(1.0 - d, dtype=torch.float32)

    # Emulate the patched MarchingCubeHelper.forward call chain: the helper
    # receives -(density - threshold) and negates it again internally.
    helper_input = -density.view(res, res, res)
    level = -helper_input
    v_pos, t_pos_idx = module.marching_cubes(level.detach(), 0.0)
    v_pos = v_pos[..., [2, 1, 0]]
    v_pos = v_pos / (res - 1.0)
    mesh = trimesh.Trimesh(v_pos.numpy(), t_pos_idx.numpy(), process=False)

    assert mesh.extents[0] > mesh.extents[1] > mesh.extents[2]
    assert mesh.volume > 0

    monkeypatch.delitem(sys.modules, "torchmcubes", raising=False)


def test_cpu_atlas_rasterizer_interpolates_and_dilates() -> None:
    # One UV triangle covering the lower-left half of a 32x32 atlas with a
    # linear value ramp across its vertices.
    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.asarray([[0, 1, 2]], dtype=np.int64)
    values = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )

    atlas = runtime._tripo_rasterize_vec3_atlas_cpu(
        atlas_indices=indices,
        atlas_uvs=uvs,
        values=values,
        texture_resolution=32,
        texture_padding=2,
    )

    assert atlas.shape == (32, 32, 4)
    coverage = atlas[:, :, 3] > 0.0
    # Roughly half of the atlas plus a 2-texel dilation band must be covered.
    assert 0.45 <= float(coverage.mean()) <= 0.75
    # Barycentric interpolation: near the (1, 0) uv corner the first channel
    # dominates, and near the (0, 1) uv corner the second channel dominates.
    assert atlas[0, 30, 0] > 0.8
    assert atlas[30, 0, 1] > 0.8
    # Dilated texels just outside the diagonal edge take nearest covered values.
    assert bool(coverage[20, 20]) or bool(coverage[16, 16])


def test_atlas_rasterizer_falls_back_to_cpu_when_gl_is_unavailable(monkeypatch) -> None:
    def _raise_gl(**kwargs):
        raise RuntimeError("cannot create standalone GL context")

    monkeypatch.setattr(runtime, "_tripo_rasterize_vec3_atlas_moderngl", _raise_gl)

    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.asarray([[0, 1, 2]], dtype=np.int64)
    values = np.ones((3, 3), dtype=np.float32)

    atlas = runtime._tripo_rasterize_vec3_atlas(
        atlas_indices=indices,
        atlas_uvs=uvs,
        values=values,
        texture_resolution=16,
        texture_padding=1,
    )

    assert atlas.shape == (16, 16, 4)
    assert float((atlas[:, :, 3] > 0).mean()) > 0.3
