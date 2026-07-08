from __future__ import annotations

import tomllib
from pathlib import Path


def test_trellis2_and_platform_profiles_cover_trellis_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    base_deps = data["project"].get("dependencies", [])
    extras = data["project"]["optional-dependencies"]

    trellis2 = extras["trellis2"]
    t23d = extras["t23d"]
    apple = extras["apple"]
    gpu = extras["gpu"]
    all_apple = extras["all-apple"]
    all_gpu = extras["all-gpu"]

    assert any(dep.startswith("numpy>=") for dep in trellis2)
    assert any(dep.startswith("torchvision>=") for dep in trellis2)
    assert any(dep.startswith("safetensors>=") for dep in trellis2)
    assert any(dep.startswith("abstractvision>=") for dep in base_deps)
    assert t23d == []
    assert any(dep.startswith("numpy>=") for dep in apple)
    assert any(dep.startswith("torchvision>=") for dep in apple)
    assert any(dep.startswith("safetensors>=") for dep in apple)
    assert any(dep.startswith("abstractvision[apple]") for dep in apple)
    assert any(dep.startswith("numpy>=") for dep in gpu)
    assert any(dep.startswith("torchvision>=") for dep in gpu)
    assert any(dep.startswith("safetensors>=") for dep in gpu)
    assert any(dep.startswith("abstractvision[gpu]") for dep in gpu)
    assert any(dep.startswith("numpy>=") for dep in all_apple)
    assert any(dep.startswith("torchvision>=") for dep in all_apple)
    assert any(dep.startswith("safetensors>=") for dep in all_apple)
    assert any(dep.startswith("abstractvision[all-apple]") for dep in all_apple)
    assert any(dep.startswith("numpy>=") for dep in all_gpu)
    assert any(dep.startswith("torchvision>=") for dep in all_gpu)
    assert any(dep.startswith("safetensors>=") for dep in all_gpu)
    assert any(dep.startswith("abstractvision[all-gpu]") for dep in all_gpu)

def test_step1x_and_platform_profiles_cover_step1x_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    base_deps = data["project"].get("dependencies", [])
    extras = data["project"]["optional-dependencies"]

    step1x = extras["step1x"]
    t23d = extras["t23d"]
    apple = extras["apple"]
    gpu = extras["gpu"]
    all_apple = extras["all-apple"]
    all_gpu = extras["all-gpu"]

    assert any(dep.startswith("numpy>=") for dep in step1x)
    assert any(dep.startswith("diffusers>=") for dep in step1x)
    assert any(dep.startswith("timm>=") for dep in step1x)
    assert any(dep.startswith("onnxruntime>=") for dep in step1x)
    assert any(dep.startswith("pymeshlab>=") for dep in step1x)
    assert any(dep.startswith("abstractvision>=") for dep in base_deps)
    assert t23d == []
    assert any(dep.startswith("numpy>=") for dep in apple)
    assert any(dep.startswith("diffusers>=") for dep in apple)
    assert any(dep.startswith("timm>=") for dep in apple)
    assert any(dep.startswith("onnxruntime>=") for dep in apple)
    assert any(dep.startswith("pymeshlab>=") for dep in apple)
    assert any(dep.startswith("numpy>=") for dep in gpu)
    assert any(dep.startswith("diffusers>=") for dep in gpu)
    assert any(dep.startswith("timm>=") for dep in gpu)
    assert any(dep.startswith("onnxruntime>=") for dep in gpu)
    assert any(dep.startswith("pymeshlab>=") for dep in gpu)
    assert any(dep.startswith("numpy>=") for dep in all_apple)
    assert any(dep.startswith("diffusers>=") for dep in all_apple)
    assert any(dep.startswith("timm>=") for dep in all_apple)
    assert any(dep.startswith("onnxruntime>=") for dep in all_apple)
    assert any(dep.startswith("pymeshlab>=") for dep in all_apple)
    assert any(dep.startswith("numpy>=") for dep in all_gpu)
    assert any(dep.startswith("diffusers>=") for dep in all_gpu)
    assert any(dep.startswith("timm>=") for dep in all_gpu)
    assert any(dep.startswith("onnxruntime>=") for dep in all_gpu)
    assert any(dep.startswith("pymeshlab>=") for dep in all_gpu)

def test_triposr_and_platform_profiles_cover_triposr_texture_bake_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    base_deps = data["project"].get("dependencies", [])
    extras = data["project"]["optional-dependencies"]

    triposr = extras["triposr"]
    t23d = extras["t23d"]
    apple = extras["apple"]
    gpu = extras["gpu"]
    all_apple = extras["all-apple"]
    all_gpu = extras["all-gpu"]

    assert any(dep.startswith("numpy>=") for dep in triposr)
    assert any(dep.startswith("xatlas>=") for dep in triposr)
    assert any(dep.startswith("moderngl>=") for dep in triposr)
    assert any(dep.startswith("onnxruntime>=") for dep in triposr)
    assert not any("torchmcubes" in dep for dep in triposr)
    assert any(dep.startswith("abstractvision>=") for dep in base_deps)
    assert t23d == []
    assert any(dep.startswith("numpy>=") for dep in apple)
    assert any(dep.startswith("xatlas>=") for dep in apple)
    assert any(dep.startswith("moderngl>=") for dep in apple)
    assert any(dep.startswith("onnxruntime>=") for dep in apple)
    assert not any("torchmcubes" in dep for dep in apple)
    assert any(dep.startswith("numpy>=") for dep in gpu)
    assert any(dep.startswith("xatlas>=") for dep in gpu)
    assert any(dep.startswith("moderngl>=") for dep in gpu)
    assert any(dep.startswith("onnxruntime>=") for dep in gpu)
    assert not any("torchmcubes" in dep for dep in gpu)
    assert any(dep.startswith("numpy>=") for dep in all_apple)
    assert any(dep.startswith("xatlas>=") for dep in all_apple)
    assert any(dep.startswith("moderngl>=") for dep in all_apple)
    assert any(dep.startswith("onnxruntime>=") for dep in all_apple)
    assert not any("torchmcubes" in dep for dep in all_apple)
    assert any(dep.startswith("numpy>=") for dep in all_gpu)
    assert any(dep.startswith("xatlas>=") for dep in all_gpu)
    assert any(dep.startswith("moderngl>=") for dep in all_gpu)
    assert any(dep.startswith("onnxruntime>=") for dep in all_gpu)
    assert not any("torchmcubes" in dep for dep in all_gpu)


def test_texturing_module_dependencies_are_declared_in_texture_profiles() -> None:
    # The shared texture bake relies on scipy (EDT feathering, KD-tree fill,
    # despeckle labeling); OpenCV ships with the same profiles for the
    # upstream backend stacks and rembg. Both must be first-class
    # dependencies of every texture-capable profile, not transitive
    # accidents.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    for profile in ("triposr", "apple", "gpu", "all-apple", "all-gpu", "hunyuan3d"):
        deps = extras[profile]
        assert any(dep.startswith("scipy>=") for dep in deps), profile
        assert any(dep.startswith("opencv-python-headless>=") for dep in deps), profile


def test_hunyuan3d_extra_covers_runtime_imports() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    hunyuan = extras["hunyuan3d"]
    for prefix in (
        "torch>=",
        "diffusers>=",
        "transformers>=",
        "huggingface_hub>=",
        "trimesh>=",
        "Pillow>=",
        "rembg>=",
        "onnxruntime>=",
        "scikit-image>=",
        "opencv-python-headless>=",
        "pymeshlab>=",
        "einops>=",
        "PyYAML>=",
        "psutil>=",
        "xatlas>=",
        "moderngl>=",
        "scipy>=",
    ):
        assert any(dep.startswith(prefix) for dep in hunyuan), prefix
