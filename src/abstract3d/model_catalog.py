"""Model and backend catalog for Abstract3D."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .types import Scene3DModelSpec


_CATALOG: tuple[Scene3DModelSpec, ...] = (
    Scene3DModelSpec(
        model_id="stabilityai/TripoSR",
        provider_id="triposr",
        backend_kind="triposr",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="MIT",
        status="validated",
        source_url="https://github.com/VAST-AI-Research/TripoSR",
        validated=True,
        apple_silicon="validated",
        footprint_gb=2.0,
        model_memory_gb=6.0,
        notes="Validated locally on Apple Silicon for i23d and composed t23d via AbstractVision image synthesis.",
    ),
    Scene3DModelSpec(
        model_id="microsoft/TRELLIS-image-large",
        provider_id="trellis",
        backend_kind="trellis",
        tasks=("image_to_scene3d",),
        license="MIT",
        status="research",
        source_url="https://github.com/microsoft/TRELLIS",
        apple_silicon="cuda_only",
        footprint_gb=3.3,
        model_memory_gb=16.0,
        notes="Permissive and strong quality, but officially Linux + NVIDIA oriented.",
    ),
    Scene3DModelSpec(
        model_id="microsoft/TRELLIS-text-large",
        provider_id="trellis",
        backend_kind="trellis",
        tasks=("text_to_scene3d",),
        license="MIT",
        status="research",
        source_url="https://github.com/microsoft/TRELLIS",
        apple_silicon="cuda_only",
        footprint_gb=2.3,
        model_memory_gb=16.0,
        notes="Best coherent permissive t23d family candidate, but not validated locally here.",
    ),
    Scene3DModelSpec(
        model_id="microsoft/TRELLIS.2-4B",
        provider_id="trellis2",
        backend_kind="trellis2",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="MIT",
        status="experimental",
        source_url="https://github.com/microsoft/TRELLIS.2",
        apple_silicon="experimental",
        footprint_gb=16.2,
        model_memory_gb=24.0,
        notes=(
            "Experimental official-only TRELLIS.2 path using official Microsoft checkpoints and "
            "official companion models. The required DINOv3 companion is gated behind Meta's "
            "DINOv3 License: operators must request access and accept its terms (attribution and "
            "acceptable-use restrictions) before the backend can run."
        ),
    ),
    Scene3DModelSpec(
        model_id="stepfun-ai/Step1X-3D",
        provider_id="step1x",
        backend_kind="step1x",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="Apache-2.0",
        status="experimental",
        source_url="https://github.com/stepfun-ai/Step1X-3D",
        apple_silicon="experimental",
        footprint_gb=6.8,
        model_memory_gb=8.0,
        notes="Experimental geometry-only backend using official Step1X geometry weights with composed t23d via AbstractVision. Works locally on Apple Silicon, but the checked proof cases were slower and lower-fidelity than TripoSR.",
    ),
    Scene3DModelSpec(
        model_id="tencent/Hunyuan3D-2.1",
        provider_id="hunyuan3d21",
        backend_kind="hunyuan3d21",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="tencent-hunyuan-community",
        status="experimental",
        source_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
        apple_silicon="experimental",
        footprint_gb=7.4,
        model_memory_gb=10.0,
        notes=(
            "Official Hunyuan3D-2.1 shape stage (3.3B flow-matching DiT). Highest-quality geometry "
            "in the local catalog, but the license is territory-restricted (excludes EU/UK/South "
            "Korea) and requires explicit operator acknowledgment before the backend will run. "
            "Texture uses the shared projection bake; the official CUDA-only PaintPBR stage is out "
            "of scope."
        ),
    ),
    Scene3DModelSpec(
        model_id="tencent/Hunyuan3D-2mv",
        provider_id="hunyuan3d21",
        backend_kind="hunyuan3d21",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="tencent-hunyuan-community",
        status="experimental",
        source_url="https://huggingface.co/tencent/Hunyuan3D-2mv",
        apple_silicon="experimental",
        footprint_gb=4.9,
        model_memory_gb=8.0,
        notes=(
            "Official multi-view shape stage (2.0 family, 1.1B DiT) served by the Hunyuan3D "
            "backend. Conditions geometry on up to four views (front/left/back/right): pass "
            "texture_reference_images/angles and reference views whose angles snap to those slots "
            "also drive shape reconstruction. Same territory-restricted community license and "
            "explicit acknowledgment gate as Hunyuan3D-2.1. Runs through the vendored 2.1 source "
            "with a config namespace remap; loads verified key-exact against the checkpoint."
        ),
    ),
    Scene3DModelSpec(
        model_id="TencentARC/InstantMesh",
        provider_id="instantmesh",
        backend_kind="instantmesh",
        tasks=("image_to_scene3d",),
        license="Apache-2.0",
        status="research",
        source_url="https://github.com/TencentARC/InstantMesh",
        apple_silicon="cuda_only",
        footprint_gb=7.3,
        model_memory_gb=10.0,
        notes="Smaller permissive i23d option, still officially CUDA-centric.",
    ),
    Scene3DModelSpec(
        model_id="ashawkey/LGM",
        provider_id="lgm",
        backend_kind="lgm",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="MIT",
        status="research",
        source_url="https://github.com/3DTopia/LGM",
        apple_silicon="cuda_only",
        footprint_gb=5.0,
        model_memory_gb=10.0,
        notes="Compact permissive t23d/i23d Gaussian model, but its official stack still depends on CUDA rasterization pieces.",
    ),
    Scene3DModelSpec(
        model_id="openai/shap-e",
        provider_id="shap-e",
        backend_kind="shap-e",
        tasks=("image_to_scene3d", "text_to_scene3d"),
        license="MIT",
        status="planned",
        source_url="https://github.com/openai/shap-e",
        apple_silicon="unverified",
        notes="Direct permissive baseline for future expansion once its older dependency stack is normalized.",
    ),
)


def iter_model_specs(*, validated_only: bool = False, task: Optional[str] = None) -> Iterable[Scene3DModelSpec]:
    normalized_task = str(task or "").strip().lower().replace("-", "_")
    for spec in _CATALOG:
        if validated_only and not spec.validated:
            continue
        if normalized_task and normalized_task not in {t.replace("-", "_") for t in spec.tasks}:
            continue
        yield spec


def capability_model_records(*, task: Optional[str] = None, validated_only: bool = True) -> List[dict]:
    return [spec.to_capability_model() for spec in iter_model_specs(validated_only=validated_only, task=task)]


def catalog_rows(*, validated_only: bool = False) -> List[dict]:
    rows: List[dict] = []
    for spec in iter_model_specs(validated_only=validated_only):
        rows.append(
            {
                "model_id": spec.model_id,
                "provider_id": spec.provider_id,
                "backend_kind": spec.backend_kind,
                "tasks": list(spec.tasks),
                "license": spec.license,
                "status": spec.status,
                "validated": spec.validated,
                "apple_silicon": spec.apple_silicon,
                "footprint_gb": spec.footprint_gb,
                "model_memory_gb": spec.model_memory_gb,
                "source_url": spec.source_url,
                "notes": spec.notes,
            }
        )
    return rows
