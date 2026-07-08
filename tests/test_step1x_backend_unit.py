from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from abstract3d.backends import step1x_runtime as runtime
from abstract3d.errors import CapabilityNotSupportedError, SourceBootstrapError


def test_step1x_rejects_unofficial_model_repo() -> None:
    try:
        runtime._model_id("someone/else")
    except CapabilityNotSupportedError:
        pass
    else:
        raise AssertionError("Expected CapabilityNotSupportedError for unofficial model repo.")


def test_step1x_rejects_non_geometry_subfolder() -> None:
    try:
        runtime._geometry_subfolder("Step1X-3D-Texture")
    except CapabilityNotSupportedError:
        pass
    else:
        raise AssertionError("Expected CapabilityNotSupportedError for non-geometry checkpoint.")


def test_step1x_accepts_official_label_geometry_subfolder() -> None:
    assert runtime._geometry_subfolder("Step1X-3D-Geometry-Label-1300m") == "Step1X-3D-Geometry-Label-1300m"


def test_step1x_defaults_to_official_label_geometry_subfolder() -> None:
    assert runtime._geometry_subfolder(None) == "Step1X-3D-Geometry-Label-1300m"


def test_step1x_verify_source_dir_requires_manifest(tmp_path) -> None:
    source_dir = tmp_path / "step1x"
    package_dir = source_dir / "step1x3d_geometry"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("from . import models\n", encoding="utf-8")

    try:
        runtime._verify_official_source_dir(source_dir)
    except SourceBootstrapError:
        pass
    else:
        raise AssertionError("Expected SourceBootstrapError when manifest is missing.")


def test_step1x_copy_and_patch_source_tree_keeps_external_source_unmodified(tmp_path) -> None:
    source_dir = tmp_path / "official"
    (source_dir / "step1x3d_geometry" / "models" / "autoencoders").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "pipelines").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2_with_registers").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "utils").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "__init__.py").write_text("from . import data, models, systems\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py").write_text("# Hunyuan\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline_utils.py").write_text("def preprocess_image(images_pil, force=False, **rembg_kwargs):\n    pass\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "utils" / "misc.py").write_text("def get_device():\n    return torch.device(f\"cuda:{get_rank()}\")\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline.py").write_text(
        "from ..conditional_encoders.dinov2_encoder import Dinov2Encoder\n"
        "import PIL.Image\n"
        "import torch\n"
        "class Step1X3DGeometryPipeline:\n"
        "    def __init__(self, visual_encoder: Dinov2Encoder):\n"
        "        self.visual_encoder = visual_encoder\n"
        "def check_inputs(image):\n"
        "    if isinstance(image, str):\n"
        "        pass\n"
        "    elif isinstance(image, (torch.Tensor, PIL.Image.Image)):\n"
        "        raise ValueError(\"Input image must be a `torch.Tensor` or `PIL.Image.Image`.\")\n",
        encoding="utf-8",
    )
    for rel in (
        source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2" / "modeling_dinov2.py",
        source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2_with_registers" / "modeling_dinov2_with_registers.py",
    ):
        rel.write_text(
            "import torch\n"
            "from transformers.pytorch_utils import (\n"
            "    find_pruneable_heads_and_indices,\n"
            "    prune_linear_layer,\n"
            ")\n"
            "logger = logging.get_logger(__name__)\n"
            "head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)\n",
            encoding="utf-8",
        )
    (source_dir / runtime._SOURCE_MANIFEST).write_text(
        json.dumps({"repo_url": runtime._STEP1X_REPO_URL, "commit": runtime._STEP1X_COMMIT}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    managed = runtime._copy_and_patch_source_tree(owner=None, source_dir=source_dir)

    assert managed != source_dir
    original_volume = (source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py").read_text(encoding="utf-8")
    managed_volume = (managed / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py").read_text(encoding="utf-8")
    assert "Hunyuan" in original_volume
    assert "Hunyuan" not in managed_volume
    manifest = json.loads((managed / runtime._SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["patchset_version"] == runtime._PATCHSET_VERSION
    assert manifest["source_origin"] == str(source_dir)


def test_step1x_patch_source_tree_rewrites_training_and_license_sensitive_files(tmp_path) -> None:
    source_dir = tmp_path / "repo"
    (source_dir / "step1x3d_geometry" / "models" / "autoencoders").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "pipelines").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2_with_registers").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "utils").mkdir(parents=True)
    (source_dir / "step1x3d_geometry" / "__init__.py").write_text("from . import data, models, systems\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py").write_text("# Hunyuan\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline_utils.py").write_text(
        "def preprocess_image(images_pil, force=False, **rembg_kwargs):\n"
        "    for i in range(len(images_pil)):\n"
        "        image = images_pil[i]\n"
        "        do_remove = True\n"
        "        if do_remove:\n"
        "            import rembg  # lazy import\n\n"
        "            if rembg_backend == \"default\":\n"
        "                image = rembg.remove(image, **rembg_kwargs)\n"
        "            else:\n"
        "                image = rembg.remove(\n"
        "                    image,\n"
        "                    session=rembg.new_session(\n"
        "                        model_name=\"bria\",\n"
        "                    ),\n"
        "                    **rembg_kwargs,\n"
        "                )\n"
        "        do_remove = do_remove or force\n",
        encoding="utf-8",
    )
    (source_dir / "step1x3d_geometry" / "utils" / "misc.py").write_text("def get_device():\n    return torch.device(f\"cuda:{get_rank()}\")\n", encoding="utf-8")
    (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline.py").write_text(
        "from ..conditional_encoders.dinov2_encoder import Dinov2Encoder\n"
        "import PIL.Image\n"
        "import torch\n"
        "class Step1X3DGeometryPipeline:\n"
        "    def __init__(self, visual_encoder: Dinov2Encoder):\n"
        "        self.visual_encoder = visual_encoder\n"
        "def check_inputs(image):\n"
        "    if isinstance(image, str):\n"
        "        pass\n"
        "    elif isinstance(image, (torch.Tensor, PIL.Image.Image)):\n"
        "        raise ValueError(\"Input image must be a `torch.Tensor` or `PIL.Image.Image`.\")\n",
        encoding="utf-8",
    )
    for rel in (
        source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2" / "modeling_dinov2.py",
        source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2_with_registers" / "modeling_dinov2_with_registers.py",
    ):
        rel.write_text(
            "import torch\n"
            "from transformers.pytorch_utils import (\n"
            "    find_pruneable_heads_and_indices,\n"
            "    prune_linear_layer,\n"
            ")\n"
            "logger = logging.get_logger(__name__)\n"
            "head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)\n",
            encoding="utf-8",
        )

    runtime._patch_step1x_source(source_dir)

    assert "from . import models" in (source_dir / "step1x3d_geometry" / "__init__.py").read_text(encoding="utf-8")
    assert "Hunyuan" not in (source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py").read_text(encoding="utf-8")
    pipeline_utils = (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline_utils.py").read_text(encoding="utf-8")
    assert "do_remove = bool(force)" in pipeline_utils
    assert "model_name=\"bria\"" not in pipeline_utils
    assert "alpha channel not empty" in pipeline_utils
    pipeline_text = (source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline.py").read_text(encoding="utf-8")
    assert "from ..conditional_encoders.base import BaseVisualEncoder" in pipeline_text
    assert "Dinov2Encoder" not in pipeline_text
    assert "BaseVisualEncoder" in pipeline_text
    assert "elif not isinstance(image, (torch.Tensor, PIL.Image.Image)):" in pipeline_text
    misc_text = (source_dir / "step1x3d_geometry" / "utils" / "misc.py").read_text(encoding="utf-8")
    assert "torch.device(\"mps\")" in misc_text
    dino_text = (source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "dinov2" / "modeling_dinov2.py").read_text(encoding="utf-8")
    assert "def find_pruneable_heads_and_indices" in dino_text
    assert "self.get_head_mask" not in dino_text


def test_step1x_t23d_composes_image_and_marks_geometry_only(monkeypatch) -> None:
    class _FakeMesh:
        vertices = [(0.0, 0.0, 0.0)] * 3
        faces = [(0, 1, 2)]

    calls: dict[str, object] = {}

    class _FakePipeline:
        def __call__(self, image, **kwargs):
            calls["image_mode"] = image.mode
            calls["kwargs"] = dict(kwargs)
            calls["pipeline_pixel"] = image.getpixel((0, 0))
            assert image.mode == "RGBA"
            return type("Out", (), {"mesh": [_FakeMesh()]})()

    backend = runtime.Step1XGeometryBackend(owner=None)

    payload = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(payload, format="PNG")

    order: list[str] = []

    monkeypatch.setattr(
        backend,
        "_make_source_image",
        lambda prompt, **kwargs: order.append("make_source_image") or payload.getvalue(),
    )
    monkeypatch.setattr(
        backend,
        "_load_runtime",
        lambda **kwargs: order.append("load_runtime") or _FakePipeline(),
    )
    monkeypatch.setattr(
        runtime,
        "_preprocess_step1x_image",
        lambda **kwargs: Image.new("RGBA", kwargs["image"].size, (0, 0, 0, 255)),
    )
    monkeypatch.setattr(runtime, "_release_mlx_generation_cache", lambda: order.append("release_mlx"))
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: b"mesh")
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (8, 8), "white")])
    backend._resident_model_id = "stepfun-ai/Step1X-3D"
    backend._resident_device = "mps"
    backend._resident_dtype = "float32"
    backend._resident_subfolder = "Step1X-3D-Geometry-1300m"
    backend._last_runtime_stats = {"load_s": 1.2, "source_dir": "/tmp/source"}

    out = backend.t23d("chair", model="stepfun-ai/Step1X-3D")

    assert out["data"] == b"mesh"
    meta = out["metadata"]
    assert meta["geometry_only"] is True
    assert meta["native_text_to_scene3d"] is False
    assert meta["composed_text_to_scene3d"] is True
    assert meta["dtype"] == "float32"
    assert meta["background_removed"] is True
    assert meta["background_removal_policy"] == "auto_generated_image"
    assert meta["guidance_scale"] == 3.0
    assert meta["patchset_version"] == runtime._PATCHSET_VERSION
    assert meta["surface_cleanup"] == {
        "do_remove_degenerate_face": False,
        "do_reduce_face": True,
        "do_remove_floater": True,
    }
    assert meta["pipeline_cleanup"] == {
        "do_remove_degenerate_face": False,
        "do_reduce_face": True,
        "do_remove_floater": True,
    }
    assert meta["cleanup_mode"] == "runtime_defaults"
    assert meta["max_facenum"] == 200000
    assert meta["label_condition"] is None
    assert meta["timings_s"]["preprocess"] is not None
    kwargs = calls["kwargs"]
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 3.0
    assert kwargs["octree_resolution"] == 192
    assert kwargs["max_facenum"] == 200000
    assert kwargs["force_remove_background"] is True
    assert kwargs["do_remove_floater"] is True
    assert kwargs["do_reduce_face"] is True
    assert calls["pipeline_pixel"] == (255, 255, 255, 255)
    assert meta["octree_resolution_policy"] == "thin_structure_prompt"
    assert order == ["make_source_image", "release_mlx", "load_runtime"]


def test_step1x_discovery_hides_t23d_when_composer_is_unavailable(monkeypatch) -> None:
    backend = runtime.Step1XGeometryBackend(owner=None)
    monkeypatch.setattr(runtime, "has_image_composer", lambda owner: False)

    providers = backend.available_providers()
    operations = backend.list_operations()

    assert providers[0]["tasks"] == ["image_to_scene3d"]
    assert providers[0]["metadata"]["composition_ready"] is False
    assert [item["task"] for item in operations] == ["image_to_scene3d"]


def test_step1x_i23d_opaque_input_auto_enables_background_removal(monkeypatch) -> None:
    class _FakeMesh:
        vertices = [(0.0, 0.0, 0.0)] * 3
        faces = [(0, 1, 2)]

    calls: dict[str, object] = {}

    class _FakePipeline:
        def __call__(self, image, **kwargs):
            calls["image_mode"] = image.mode
            calls["kwargs"] = dict(kwargs)
            return type("Out", (), {"mesh": [_FakeMesh()]})()

    backend = runtime.Step1XGeometryBackend(owner=None)
    monkeypatch.setattr(backend, "_load_runtime", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(runtime, "_load_image_payload", lambda image, artifact_store=None: Image.new("RGB", (16, 16), "white"))
    monkeypatch.setattr(
        runtime,
        "_preprocess_step1x_image",
        lambda **kwargs: kwargs["image"].convert("RGBA"),
    )
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: b"mesh")
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (8, 8), "white")])
    backend._resident_model_id = "stepfun-ai/Step1X-3D"
    backend._resident_device = "mps"
    backend._resident_dtype = "float32"
    backend._resident_subfolder = "Step1X-3D-Geometry-1300m"
    backend._last_runtime_stats = {"load_s": 1.2, "source_dir": "/tmp/source"}

    out = backend.i23d("opaque.png", model="stepfun-ai/Step1X-3D")

    meta = out["metadata"]
    assert meta["background_removed"] is True
    assert meta["background_removal_policy"] == "auto_opaque_input"
    assert meta["max_facenum"] == 200000
    kwargs = calls["kwargs"]
    assert kwargs["guidance_scale"] == 3.0
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["octree_resolution"] == 128
    assert kwargs["max_facenum"] == 200000


def test_step1x_label_geometry_auto_infers_label(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeExtractMesh:
        def __init__(self) -> None:
            self.verts = __import__("torch").tensor([[0.0, 0.0, 0.0]] * 3)
            self.faces = __import__("torch").tensor([[0, 1, 2]])

    class _FakeVAE:
        def to(self, *args, **kwargs):
            return self

        def decode(self, latents):
            calls["decode_latents"] = latents
            return latents

        def extract_geometry(self, decoded, **kwargs):
            calls["extract_kwargs"] = dict(kwargs)
            return [_FakeExtractMesh()]

    class _FakePipeline:
        def __init__(self) -> None:
            self.vae = _FakeVAE()

        def __call__(self, image, **kwargs):
            calls["kwargs"] = dict(kwargs)
            return type("Out", (), {"mesh": __import__("torch").zeros((1, 4, 4))})()

    backend = runtime.Step1XGeometryBackend(owner=None)
    monkeypatch.setattr(backend, "_load_runtime", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(runtime, "_load_image_payload", lambda image, artifact_store=None: Image.new("RGBA", (16, 16), (255, 255, 255, 0)))
    monkeypatch.setattr(runtime, "_preprocess_step1x_image", lambda **kwargs: kwargs["image"])
    monkeypatch.setattr(
        runtime,
        "_step1x_postprocess_mesh",
        lambda **kwargs: (kwargs["mesh"], ["remove_floater", "reduce_face:200000"], []),
    )
    monkeypatch.setattr(runtime, "_mesh_export_bytes", lambda mesh, *, file_type: b"mesh")
    monkeypatch.setattr(runtime, "render_mesh_views", lambda mesh: [Image.new("RGB", (8, 8), "white")])
    backend._resident_model_id = "stepfun-ai/Step1X-3D"
    backend._resident_device = "mps"
    backend._resident_dtype = "float32"
    backend._resident_subfolder = "Step1X-3D-Geometry-Label-1300m"
    backend._last_runtime_stats = {"load_s": 1.2, "source_dir": "/tmp/source"}

    out = backend.i23d("object.png", prompt="a ceramic teapot with a curved spout", model="stepfun-ai/Step1X-3D")

    meta = out["metadata"]
    assert meta["geometry_subfolder"] == "Step1X-3D-Geometry-Label-1300m"
    assert meta["label_condition"] == {"symmetry": "asymmetry", "geometry_type": "smooth"}
    assert meta["pipeline_cleanup"] == {
        "do_remove_degenerate_face": False,
        "do_reduce_face": False,
        "do_remove_floater": False,
    }
    assert meta["postprocess_cleanup"] == ["remove_floater", "reduce_face:200000"]
    assert meta["cleanup_mode"] == "label_geometry_cleanup_disabled_on_mps"
    assert meta["surface_extract_device"] == "cpu"
    assert calls["kwargs"]["label"] == {"symmetry": "asymmetry", "geometry_type": "smooth"}
    assert calls["kwargs"]["do_remove_floater"] is False
    assert calls["kwargs"]["do_reduce_face"] is False
    assert calls["kwargs"]["output_type"] == "latent"


def test_step1x_label_geometry_prefers_sharp_for_espresso_machine() -> None:
    label = runtime._infer_step1x_label(
        prompt="a studio product photo of a red espresso machine with rounded corners",
        processed_image=Image.new("RGBA", (16, 16), (255, 255, 255, 0)),
    )

    assert label == {"symmetry": "asymmetry", "geometry_type": "sharp"}


def test_step1x_label_geometry_prefers_x_sharp_for_chair() -> None:
    label = runtime._infer_step1x_label(
        prompt="a mid-century lounge chair with walnut legs and woven fabric",
        processed_image=Image.new("RGBA", (16, 16), (255, 255, 255, 0)),
    )

    assert label == {"symmetry": "x", "geometry_type": "sharp"}


def test_step1x_selects_base_fallback_for_mps_sharp_asymmetry_i23d() -> None:
    selected, policy = runtime._step1x_select_geometry_subfolder(
        requested=None,
        device="mps",
        task="image_to_scene3d",
        label={"symmetry": "asymmetry", "geometry_type": "sharp"},
    )

    assert selected == "Step1X-3D-Geometry-1300m"
    assert policy == "mps_i23d_sharp_asymmetry_base_fallback"


def test_step1x_keeps_label_default_for_mps_symmetric_chair_case() -> None:
    selected, policy = runtime._step1x_select_geometry_subfolder(
        requested=None,
        device="mps",
        task="image_to_scene3d",
        label={"symmetry": "x", "geometry_type": "sharp"},
    )

    assert selected == "Step1X-3D-Geometry-Label-1300m"
    assert policy == "default_label"


def test_step1x_keep_runtime_resident_defaults_to_false_on_mps() -> None:
    assert runtime._step1x_keep_runtime_resident(owner=None, device="mps") is False
    assert runtime._step1x_keep_runtime_resident(owner=None, device="cpu") is True


def test_step1x_mps_memory_cap_default_bytes() -> None:
    assert runtime._step1x_mps_memory_cap_bytes(owner=None) == 48 * (1024 ** 3)


def test_step1x_validate_suite_forwards_model_subfolder(monkeypatch, tmp_path) -> None:
    backend = runtime.Step1XGeometryBackend(owner=None)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        backend,
        "t23d",
        lambda prompt, **kwargs: calls.append(("t23d", dict(kwargs))) or {"metadata": {"contact_sheet_path": None}},
    )
    monkeypatch.setattr(
        backend,
        "i23d",
        lambda image, **kwargs: calls.append(("i23d", dict(kwargs))) or {"metadata": {"contact_sheet_path": None}},
    )
    monkeypatch.setattr(
        runtime,
        "stack_contact_sheets",
        lambda items, **kwargs: Image.new("RGB", (8, 8), "white"),
    )

    backend.validate_suite(
        prompts=["teapot"],
        images=["/tmp/object.png"],
        image_prompts=["a studio product photo of a red espresso machine with rounded corners"],
        output_dir=str(tmp_path),
        model="stepfun-ai/Step1X-3D",
        model_subfolder="Step1X-3D-Geometry-Label-1300m",
    )

    assert calls[0][0] == "t23d"
    assert calls[0][1]["model"] == "stepfun-ai/Step1X-3D"
    assert calls[0][1]["model_subfolder"] == "Step1X-3D-Geometry-Label-1300m"
    assert calls[1][0] == "i23d"
    assert calls[1][1]["model"] == "stepfun-ai/Step1X-3D"
    assert calls[1][1]["model_subfolder"] == "Step1X-3D-Geometry-Label-1300m"
    assert calls[1][1]["prompt"] == "a studio product photo of a red espresso machine with rounded corners"
