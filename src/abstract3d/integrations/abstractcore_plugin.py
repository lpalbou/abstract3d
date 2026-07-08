"""AbstractCore capability plugin registration for Abstract3D."""

from __future__ import annotations

from ..backends import make_backend


def register(registry) -> None:
    registry.register_scene3d_backend(
        backend_id="abstract3d:triposr",
        factory=lambda owner: make_backend("abstract3d:triposr", owner),
        priority=10,
        description="Local TripoSR image-to-3D with AbstractVision-composed text-to-3D.",
        install_hint='pip install "abstract3d[triposr]" for local i23d, or "abstract3d[apple]" / "abstract3d[gpu]" for local composed t23d. Base "pip install abstract3d" keeps the lightweight AbstractVision composition contract.',
        config_hint=(
            "Optional config: scene3d_device=mps|cpu, scene3d_model_id=stabilityai/TripoSR, "
            "scene3d_triposr_source_dir=/path/to/TripoSR, "
            "scene3d_triposr_mc_resolution=256, scene3d_triposr_cleanup=presentation|none, "
            "scene3d_image_provider=<configured AbstractVision provider>, scene3d_image_model=<vision model id>, "
            "or let Abstract3D bootstrap the pinned source snapshot."
        ),
    )
    registry.register_scene3d_backend(
        backend_id="abstract3d:step1x-local",
        factory=lambda owner: make_backend("abstract3d:step1x-local", owner),
        priority=7,
        description="Experimental local Step1X geometry backend using official Step1X weights, geometry-only output, and AbstractVision-composed text-to-3D.",
        install_hint='pip install "abstract3d[step1x]" for local i23d, or "abstract3d[apple]" / "abstract3d[gpu]" for local composed t23d. Base "pip install abstract3d" keeps the lightweight AbstractVision composition contract.',
        config_hint=(
            "Optional config: scene3d_backend=abstract3d:step1x-local, scene3d_device=mps|cpu|cuda, "
            "scene3d_model_id=stepfun-ai/Step1X-3D, "
            "scene3d_step1x_source_dir=/path/to/pinned/Step1X-source, "
            "scene3d_image_provider=<configured AbstractVision provider>, scene3d_image_model=<vision model id>, "
            "scene3d_step1x_dtype=float32|float16. "
            "Apple MPS currently uses float32 for stability."
        ),
    )
    registry.register_scene3d_backend(
        backend_id="abstract3d:trellis2-local",
        factory=lambda owner: make_backend("abstract3d:trellis2-local", owner),
        priority=5,
        description="Local TRELLIS.2 image-to-3D using official Microsoft checkpoints and official companion models only.",
        install_hint='pip install "abstract3d[trellis2]" for local i23d, or "abstract3d[apple]" / "abstract3d[gpu]" for local composed t23d. Base "pip install abstract3d" keeps the lightweight AbstractVision composition contract.',
        config_hint=(
            "Optional config: scene3d_backend=abstract3d:trellis2-local, scene3d_device=mps|cpu|cuda, "
            "scene3d_model_id=microsoft/TRELLIS.2-4B, "
            "scene3d_trellis2_source_dir=/path/to/TRELLIS.2, "
            "scene3d_image_provider=<configured AbstractVision provider>, scene3d_image_model=<vision model id>, "
            "scene3d_trellis2_dino_model=/path/to/local/dinov3-snapshot or facebook/dinov3-vitl16-pretrain-lvd1689m."
        ),
    )
