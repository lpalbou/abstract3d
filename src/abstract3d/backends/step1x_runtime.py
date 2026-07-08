"""Official Step1X geometry backend with local compatibility patches."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import gc
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..artifacts import is_artifact_ref, stable_artifact_id, store_bytes
from ..errors import Abstract3DError, CapabilityNotSupportedError, DependencyUnavailableError, SourceBootstrapError
from ..image_composition import COMPOSITION_INSTALL_HINT, has_image_composer, pop_image_generation_request
from ..model_catalog import iter_model_specs
from ..rendering import render_mesh_views, stack_contact_sheets
from .triposr_runtime import (
    _cache_root,
    _default_image_generator,
    _default_text_to_image_prompt,
    _env,
    _load_image_payload,
    _mesh_export_bytes,
    _owner_cfg,
    _owner_cfg_bool,
    _owner_cfg_int,
    _write_bundle,
    _zip_bundle,
)


_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "abstract3d"
_STEP1X_REPO_URL = "https://github.com/stepfun-ai/Step1X-3D.git"
_STEP1X_COMMIT = "cb5ac944709c6c913109070c7b90c3447f57f3d4"
_SOURCE_MANIFEST = ".abstract3d-source.json"
_PATCHSET_VERSION = "step1x-geometry-compat-v4"
_OFFICIAL_MODEL_ID = "stepfun-ai/Step1X-3D"
_DEFAULT_MODEL_ID = _OFFICIAL_MODEL_ID
_MODEL_REVISION = "bf7084495b3a72222f36549b7942948aa4d9daa7"
_GEOMETRY_SUBFOLDER = "Step1X-3D-Geometry-1300m"
_GEOMETRY_LABEL_SUBFOLDER = "Step1X-3D-Geometry-Label-1300m"
_DEFAULT_GEOMETRY_SUBFOLDER = _GEOMETRY_LABEL_SUBFOLDER
_SUPPORTED_GEOMETRY_SUBFOLDERS = (_GEOMETRY_SUBFOLDER, _GEOMETRY_LABEL_SUBFOLDER)
_TASK_ALIASES = {
    "scene3d": "text_to_scene3d",
    "scene3d_generation": "text_to_scene3d",
    "text_to_scene3d": "text_to_scene3d",
    "t23d": "text_to_scene3d",
    "image_to_scene3d": "image_to_scene3d",
    "i23d": "image_to_scene3d",
}
_RUNTIME_IMPORTS = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "trimesh",
    "PIL",
    "rembg",
    "skimage",
    "pymeshlab",
    "pytorch_lightning",
    "timm",
    "cv2",
    "mcubes",
    "jaxtyping",
)
_PATCH_LOCK = threading.Lock()
_ASYMMETRY_KEYWORDS = (
    "teapot",
    "mug",
    "pitcher",
    "kettle",
    "watering can",
    "coffee machine",
    "espresso machine",
    "lamp",
    "guitar",
    "violin",
)
_X_SYMMETRY_KEYWORDS = (
    "chair",
    "armchair",
    "lounge chair",
    "stool",
    "bench",
    "sofa",
    "couch",
    "figurine",
    "owl",
    "statue",
    "vase",
    "bottle",
)
_SMOOTH_KEYWORDS = (
    "ceramic",
    "glazed",
    "glaze",
    "round",
    "rounded",
    "curved",
    "sculpture",
    "figurine",
    "owl",
    "vase",
    "bottle",
    "figurine",
    "carved",
    "wooden",
)
_SHARP_KEYWORDS = (
    "machine",
    "espresso",
    "camera",
    "computer",
    "box",
    "cabinet",
    "desk",
    "table",
    "shelf",
)
_THIN_STRUCTURE_KEYWORDS = (
    "chair",
    "armchair",
    "lounge chair",
    "stool",
    "bench",
    "table",
    "lamp",
    "tripod",
    "rack",
    "stand",
)
_UPRIGHT_FURNITURE_KEYWORDS = (
    "chair",
    "armchair",
    "lounge chair",
    "stool",
    "bench",
    "sofa",
    "couch",
    "table",
    "desk",
)


def _keyword_hits(lowered: str, keywords: Sequence[str]) -> int:
    text = str(lowered or "")
    return sum(1 for keyword in keywords if keyword in text)


def _normalize_repo_url(repo_url: str) -> str:
    text = str(repo_url or "").strip().lower()
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def _require_runtime_dependencies() -> None:
    missing = [name for name in _RUNTIME_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        raise DependencyUnavailableError(
            "Step1X geometry runtime dependencies are missing: "
            + ", ".join(sorted(missing))
            + '. Install with: pip install "abstract3d[step1x]"'
        )


def _select_device(owner: Any, explicit: Optional[str] = None) -> str:
    torch = importlib.import_module("torch")
    requested = str(explicit or _owner_cfg(owner, "scene3d_device") or _env("ABSTRACT3D_DEVICE") or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    return "cpu"


def _select_dtype(device: str, explicit: Optional[str] = None) -> str:
    requested = str(explicit or "").strip().lower()
    if requested in {"float32", "fp32"}:
        return "float32"
    if requested in {"float16", "fp16"}:
        return "float16"
    if requested in {"bfloat16", "bf16"}:
        return "bfloat16"
    if str(device).startswith("cuda"):
        return "float16"
    return "float32"


def _torch_dtype(dtype_name: str):
    torch = importlib.import_module("torch")
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype_name)]


def _owner_cfg_float(owner: Any, key: str, default: float) -> float:
    raw = _owner_cfg(owner, key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _step1x_default_num_inference_steps(owner: Any, device: str, explicit: Optional[Any]) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    configured = _owner_cfg(owner, "scene3d_step1x_num_inference_steps")
    if configured is not None:
        try:
            return max(1, int(configured))
        except Exception:
            pass
    return 8 if str(device) == "mps" else 30


def _step1x_default_octree_resolution(owner: Any, device: str, explicit: Optional[Any]) -> int:
    if explicit is not None:
        return max(32, int(explicit))
    configured = _owner_cfg(owner, "scene3d_step1x_octree_resolution")
    if configured is not None:
        try:
            return max(32, int(configured))
        except Exception:
            pass
    return 128 if str(device) == "mps" else 384


def _step1x_tuned_octree_resolution(
    *,
    prompt: str,
    device: str,
    resolved: int,
    explicit: Optional[Any],
) -> tuple[int, Optional[str]]:
    tuned = int(resolved)
    if explicit is not None or str(device) != "mps":
        return tuned, None
    lowered = str(prompt or "").strip().lower()
    if any(keyword in lowered for keyword in _THIN_STRUCTURE_KEYWORDS):
        return max(tuned, 192), "thin_structure_prompt"
    return tuned, None


def _step1x_default_guidance_scale(owner: Any, device: str, explicit: Optional[Any]) -> float:
    if explicit is not None:
        return float(explicit)
    configured = _owner_cfg(owner, "scene3d_step1x_guidance_scale")
    if configured is not None:
        try:
            return float(configured)
        except Exception:
            pass
    return 3.0 if str(device) == "mps" else 7.5


def _step1x_default_max_facenum(owner: Any, device: str, explicit: Optional[Any]) -> int:
    if explicit is not None:
        return max(1000, int(explicit))
    configured = _owner_cfg(owner, "scene3d_step1x_max_facenum")
    if configured is not None:
        try:
            return max(1000, int(configured))
        except Exception:
            pass
    return 200000 if str(device) == "mps" else 400000


def _step1x_effective_postprocess_max_facenum(
    *,
    device: str,
    resolved: int,
    explicit: Optional[Any],
    label: Optional[Mapping[str, Any]],
) -> tuple[int, str]:
    target = max(1000, int(resolved))
    if explicit is not None or str(device) != "mps":
        return target, "requested_or_non_mps"
    geometry_type = str((label or {}).get("geometry_type") or "").strip().lower()
    symmetry = str((label or {}).get("symmetry") or "").strip().lower()
    if geometry_type == "sharp" and symmetry == "asymmetry":
        return min(target, 120000), "mps_sharp_asymmetry_compact"
    if geometry_type == "smooth" and symmetry == "asymmetry":
        return min(target, 140000), "mps_smooth_asymmetry_compact"
    if geometry_type == "smooth":
        return min(target, 160000), "mps_smooth_compact"
    return target, "default"


def _step1x_keep_runtime_resident(owner: Any, device: str) -> bool:
    configured = _owner_cfg(owner, "scene3d_step1x_keep_resident")
    if configured is not None:
        return _owner_cfg_bool(owner, "scene3d_step1x_keep_resident", False)
    return str(device) != "mps"


def _step1x_mps_memory_cap_bytes(owner: Any) -> Optional[int]:
    configured = _owner_cfg(owner, "scene3d_step1x_mps_max_memory_gb") or _env("ABSTRACT3D_STEP1X_MPS_MAX_MEMORY_GB")
    if configured is None:
        configured = 48.0
    try:
        value = float(configured)
    except Exception:
        return None
    if value <= 0:
        return None
    return int(value * (1024 ** 3))


def _step1x_foreground_ratio(owner: Any, explicit: Optional[Any]) -> float:
    if explicit is not None:
        return float(explicit)
    return _owner_cfg_float(owner, "scene3d_step1x_foreground_ratio", 0.95)


def _image_has_nonempty_alpha(image: Any) -> bool:
    try:
        bands = set(image.getbands())
        if "A" not in bands:
            return False
        extrema = image.getchannel("A").getextrema()
        return isinstance(extrema, tuple) and len(extrema) == 2 and int(extrema[0]) < 255
    except Exception:
        return False


def _alpha_symmetry_score(image: Any) -> Optional[float]:
    try:
        import numpy as np

        alpha = image.getchannel("A")
        mask = np.array(alpha, dtype=np.uint8) > 0
        if not mask.any():
            return None
        flipped = np.fliplr(mask)
        intersection = int((mask & flipped).sum())
        union = int((mask | flipped).sum())
        if union <= 0:
            return None
        return float(intersection / union)
    except Exception:
        return None


def _normalize_step1x_label(value: Any) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    raw: Mapping[str, Any]
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return None
        raw = json.loads(text)
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise ValueError("Step1X label must be a mapping or a JSON object string.")

    normalized: Dict[str, str] = {}
    pose = str(raw.get("pose") or "").strip().lower()
    if pose in {"unknown", "t-pose", "a-pose"}:
        normalized["pose"] = pose

    symmetry = str(raw.get("symmetry") or "").strip().lower()
    if symmetry in {"x", "asymmetry", "y", "z"}:
        normalized["symmetry"] = "x" if symmetry == "x" else "asymmetry"

    geometry_type = str(raw.get("geometry_type") or raw.get("edge_type") or "").strip().lower()
    if geometry_type in {"normal", "smooth", "sharp"}:
        normalized["geometry_type"] = geometry_type

    return normalized or None


def _infer_step1x_label(*, prompt: str, processed_image: Any) -> Dict[str, str]:
    lowered = str(prompt or "").strip().lower()
    label: Dict[str, str] = {}
    if any(keyword in lowered for keyword in _ASYMMETRY_KEYWORDS):
        label["symmetry"] = "asymmetry"
    elif any(keyword in lowered for keyword in _X_SYMMETRY_KEYWORDS):
        label["symmetry"] = "x"
    else:
        symmetry_score = _alpha_symmetry_score(processed_image)
        label["symmetry"] = "x" if symmetry_score is not None and symmetry_score >= 0.94 else "asymmetry"

    sharp_hits = _keyword_hits(lowered, _SHARP_KEYWORDS)
    smooth_hits = _keyword_hits(lowered, _SMOOTH_KEYWORDS)
    if any(keyword in lowered for keyword in _THIN_STRUCTURE_KEYWORDS):
        label["geometry_type"] = "sharp"
    elif sharp_hits >= 2 and sharp_hits >= smooth_hits:
        label["geometry_type"] = "sharp"
    elif smooth_hits >= 2 and smooth_hits > sharp_hits:
        label["geometry_type"] = "smooth"

    return label


def _resolve_background_removal_policy(
    *,
    task: str,
    image: Any,
    explicit: Optional[bool],
) -> tuple[bool, str]:
    if explicit is True:
        return True, "explicit_true"
    if explicit is False:
        return False, "explicit_false"
    if task == "text_to_scene3d":
        return True, "auto_generated_image"
    if _image_has_nonempty_alpha(image):
        return False, "auto_alpha_mask_present"
    return True, "auto_opaque_input"


def _step1x_cleanup_flags(owner: Any) -> Dict[str, bool]:
    return {
        "do_remove_floater": _owner_cfg_bool(owner, "scene3d_step1x_remove_floater", True),
        "do_remove_degenerate_face": _owner_cfg_bool(owner, "scene3d_step1x_remove_degenerate_face", False),
        "do_reduce_face": _owner_cfg_bool(owner, "scene3d_step1x_reduce_face", True),
    }


def _step1x_cpu_extract_num_chunks(owner: Any, device: str) -> int:
    configured = _owner_cfg(owner, "scene3d_step1x_cpu_extract_num_chunks")
    if configured is not None:
        try:
            return max(1024, int(configured))
        except Exception:
            pass
    return 16384 if str(device) == "mps" else 65536


def _step1x_cpu_extract_surface_band(
    owner: Any,
    *,
    device: str,
    label: Optional[Mapping[str, Any]],
) -> float:
    configured = _owner_cfg(owner, "scene3d_step1x_cpu_extract_surface_band")
    if configured is not None:
        try:
            return max(0.05, min(0.95, float(configured)))
        except Exception:
            pass
    if str(device) != "mps":
        return 0.95
    geometry_type = str((label or {}).get("geometry_type") or "").strip().lower()
    symmetry = str((label or {}).get("symmetry") or "").strip().lower()
    if geometry_type == "sharp" and symmetry == "asymmetry":
        return 0.45
    if geometry_type == "sharp":
        return 0.55
    if geometry_type == "smooth":
        return 0.75
    return 0.65


def _preprocess_step1x_image(
    *,
    image: Any,
    force_remove_background: bool,
    foreground_ratio: float,
):
    module = importlib.import_module("step1x3d_geometry.models.pipelines.pipeline_utils")
    preprocess = getattr(module, "preprocess_image")
    return preprocess(
        image,
        force=bool(force_remove_background),
        background_color=[255, 255, 255],
        foreground_ratio=float(foreground_ratio),
    )


def _release_mlx_generation_cache() -> None:
    gc.collect()
    try:
        mx = importlib.import_module("mlx.core")
    except Exception:
        return
    try:
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    except Exception:
        pass


def _step1x_sync_device(torch: Any, device: Any) -> None:
    try:
        if str(device) == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif str(device).startswith("cuda") and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _step1x_offload_pipeline_modules(
    *,
    pipeline: Any,
    preserve_runtime: bool,
) -> Dict[str, Any]:
    torch = importlib.import_module("torch")
    details: Dict[str, Any] = {
        "preserve_runtime": bool(preserve_runtime),
        "moved_to_cpu": [],
        "memory_before_offload": _step1x_mps_memory_stats(torch),
        "memory_after_offload": None,
    }
    if preserve_runtime:
        details["memory_after_offload"] = _step1x_mps_memory_stats(torch)
        return details
    for name in ("transformer", "visual_encoder", "caption_encoder", "label_encoder"):
        module = getattr(pipeline, name, None)
        if module is None or not hasattr(module, "to"):
            continue
        try:
            module.to("cpu")
            details["moved_to_cpu"].append(name)
        except Exception:
            continue
    _step1x_release_runtime_memory(torch)
    details["memory_after_offload"] = _step1x_mps_memory_stats(torch)
    return details


def _step1x_drop_pipeline_components(pipeline: Any) -> List[str]:
    dropped: List[str] = []
    for name in ("transformer", "visual_encoder", "caption_encoder", "label_encoder", "vae"):
        if not hasattr(pipeline, name):
            continue
        try:
            setattr(pipeline, name, None)
            dropped.append(name)
        except Exception:
            continue
    return dropped


def _step1x_extract_vae_helper_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "step1x_extract_vae.py"


def _step1x_extract_mesh_with_vae_subprocess(
    *,
    latents: Any,
    source_dir: str,
    snapshot_root: str,
    geometry_subfolder: str,
    octree_resolution: int,
    num_chunks: int,
    near_surface_band: float,
    postprocess_max_facenum: int,
    cleanup_flags: Mapping[str, bool],
    canonicalize_export: bool,
    prompt: str,
) -> Any:
    import json
    from PIL import Image

    torch = importlib.import_module("torch")
    helper = _step1x_extract_vae_helper_script()
    temp_root = Path(tempfile.mkdtemp(prefix="abstract3d-step1x-vae-extract-"))
    try:
        latents_path = temp_root / "latents.pt"
        bundle_dir = temp_root / "bundle"
        torch.save(latents.detach().to("cpu", dtype=torch.float32), latents_path)
        command = [
            sys.executable,
            str(helper),
            "--source-dir",
            str(source_dir),
            "--snapshot-root",
            str(snapshot_root),
            "--geometry-subfolder",
            str(geometry_subfolder),
            "--latents",
            str(latents_path),
            "--bundle-dir",
            str(bundle_dir),
            "--octree-resolution",
            str(int(octree_resolution)),
            "--num-chunks",
            str(int(num_chunks)),
            "--near-surface-band",
            str(float(near_surface_band)),
            "--max-facenum",
            str(int(postprocess_max_facenum)),
            "--prompt",
            str(prompt or ""),
        ]
        for flag_name, enabled in (
            ("--do-remove-floater", cleanup_flags.get("do_remove_floater")),
            ("--do-remove-degenerate-face", cleanup_flags.get("do_remove_degenerate_face")),
            ("--do-reduce-face", cleanup_flags.get("do_reduce_face")),
            ("--canonicalize-export", canonicalize_export),
        ):
            if enabled:
                command.append(flag_name)
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        report = json.loads((bundle_dir / "report.json").read_text(encoding="utf-8"))
        glb_bytes = (bundle_dir / "mesh.glb").read_bytes()
        obj_bytes = (bundle_dir / "mesh.obj").read_bytes()
        views = []
        for path in sorted(bundle_dir.glob("view_*.png")):
            with Image.open(path) as image:
                views.append(image.convert("RGB"))
        return {
            "glb_bytes": glb_bytes,
            "obj_bytes": obj_bytes,
            "views": views,
            "report": report,
            "helper_stdout": result.stdout.strip() or None,
            "helper_stderr": result.stderr.strip() or None,
        }
    except subprocess.CalledProcessError as exc:
        message = exc.stderr or exc.stdout or str(exc)
        raise Abstract3DError(f"Step1X VAE extraction helper failed: {message}") from exc
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _step1x_mesh_extract_on_cpu(
    *,
    pipeline: Any,
    latents: Any,
    octree_resolution: int,
    preserve_runtime: bool,
    num_chunks: int,
    near_surface_band: float,
    source_dir: str,
    snapshot_root: str,
    geometry_subfolder: str,
    cleanup_flags: Mapping[str, bool],
    postprocess_max_facenum: int,
    canonicalize_export: bool,
    prompt: str,
):
    torch = importlib.import_module("torch")
    latents_device = getattr(latents, "device", "unknown")
    _step1x_sync_device(torch, latents_device)
    latents_cpu = latents.detach().to("cpu", dtype=torch.float32)
    del latents
    offload = _step1x_offload_pipeline_modules(pipeline=pipeline, preserve_runtime=preserve_runtime)
    can_use_vae_subprocess = (
        not preserve_runtime
        and str(latents_device).startswith("mps")
        and bool(str(source_dir).strip())
        and bool(str(snapshot_root).strip())
        and Path(str(source_dir)).exists()
        and Path(str(snapshot_root)).exists()
        and _step1x_extract_vae_helper_script().exists()
    )
    if can_use_vae_subprocess:
        offload["dropped_components"] = _step1x_drop_pipeline_components(pipeline)
        _step1x_release_runtime_memory(torch)
        offload["memory_after_parent_drop"] = _step1x_mps_memory_stats(torch)
        mesh_result = _step1x_extract_mesh_with_vae_subprocess(
            latents=latents_cpu,
            source_dir=source_dir,
            snapshot_root=snapshot_root,
            geometry_subfolder=geometry_subfolder,
            octree_resolution=int(octree_resolution),
            num_chunks=int(num_chunks),
            near_surface_band=float(near_surface_band),
            postprocess_max_facenum=int(postprocess_max_facenum),
            cleanup_flags=cleanup_flags,
            canonicalize_export=bool(canonicalize_export),
            prompt=prompt,
        )
        del latents_cpu
        _step1x_release_runtime_memory(torch)
        offload["memory_after_extract"] = _step1x_mps_memory_stats(torch)
        offload["latents_device"] = str(latents_device)
        offload["extraction_mode"] = "vae_subprocess"
        offload["helper_stdout"] = mesh_result.get("helper_stdout")
        offload["helper_stderr"] = mesh_result.get("helper_stderr")
        offload["helper_report"] = dict(mesh_result.get("report") or {})
        return mesh_result, offload
    vae = pipeline.vae.to("cpu", dtype=torch.float32)
    decoded = vae.decode(latents_cpu)
    del latents_cpu
    _step1x_release_runtime_memory(torch)
    meshes = vae.extract_geometry(
        decoded,
        surface_extractor_type=None,
        bounds=1.05,
        mc_level=0.0,
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        near_surface_band=float(near_surface_band),
        enable_pbar=False,
    )
    del decoded
    _step1x_release_runtime_memory(torch)
    offload["memory_after_extract"] = _step1x_mps_memory_stats(torch)
    offload["latents_device"] = str(latents_device)
    offload["extraction_mode"] = "in_process_vae"
    return meshes, offload


def _step1x_trimesh_from_extract_result(mesh_result: Any):
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=mesh_result.verts.cpu().numpy(),
        faces=mesh_result.faces.cpu().numpy(),
    )
    mesh.fix_normals()
    mesh.face_normals
    mesh.vertex_normals
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=(255, 255, 255),
            main_color=(255, 255, 255),
            metallicFactor=0.05,
            roughnessFactor=1.0,
        )
    )
    return mesh


def _step1x_postprocess_mesh(
    *,
    mesh: Any,
    cleanup_flags: Mapping[str, bool],
    max_facenum: int,
) -> tuple[Any, List[str], List[str]]:
    applied: List[str] = []
    warnings: List[str] = []
    try:
        module = importlib.import_module("step1x3d_geometry.models.pipelines.pipeline_utils")
        processed = mesh
        if cleanup_flags.get("do_remove_floater"):
            processed = module.remove_floater(processed)
            applied.append("remove_floater")
        processed, component_applied, component_warnings = _step1x_prune_components(processed)
        applied.extend(component_applied)
        warnings.extend(component_warnings)
        if cleanup_flags.get("do_remove_degenerate_face"):
            processed = module.remove_degenerate_face(processed)
            applied.append("remove_degenerate_face")
        if cleanup_flags.get("do_reduce_face") and int(max_facenum) > 0:
            processed = module.reduce_face(processed, int(max_facenum))
            applied.append(f"reduce_face:{int(max_facenum)}")
        processed, topology_applied, topology_warnings = _step1x_repair_mesh_topology(processed)
        applied.extend(topology_applied)
        warnings.extend(topology_warnings)
        try:
            processed = processed.smooth_shaded
            applied.append("shade_smooth")
        except Exception as exc:
            warnings.append(f"Step1X smooth shading skipped: {type(exc).__name__}: {exc}")
        return processed, applied, warnings
    except Exception as exc:
        warnings.append(f"Step1X CPU postprocess skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings


def _step1x_prune_components(mesh: Any) -> tuple[Any, List[str], List[str]]:
    applied: List[str] = []
    warnings: List[str] = []
    try:
        import trimesh

        processed = mesh
        if isinstance(processed, trimesh.Scene):
            geometries = [item for item in processed.geometry.values()]
            if geometries:
                processed = trimesh.util.concatenate(geometries)
                applied.append(f"concatenate_scene:{len(geometries)}")
        components = list(processed.split(only_watertight=False))
        if len(components) > 1:
            components = sorted(components, key=lambda item: (len(item.faces), float(item.area)), reverse=True)
            processed = components[0].copy()
            applied.append(f"keep_largest_component:{len(components)}->1")
        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()
        return processed, applied, warnings
    except Exception as exc:
        warnings.append(f"Step1X component pruning skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings


def _step1x_repair_mesh_topology(mesh: Any) -> tuple[Any, List[str], List[str]]:
    applied: List[str] = []
    warnings: List[str] = []
    try:
        import trimesh

        processed = mesh.copy() if hasattr(mesh, "copy") else mesh
        try:
            if hasattr(processed, "merge_vertices"):
                processed.merge_vertices()
                applied.append("merge_vertices")
        except Exception as exc:
            warnings.append(f"Step1X merge_vertices skipped: {type(exc).__name__}: {exc}")
        try:
            before_watertight = bool(getattr(processed, "is_watertight", False))
        except Exception:
            before_watertight = False
        try:
            filled = bool(processed.fill_holes())
            after_watertight = bool(getattr(processed, "is_watertight", False))
            if filled or (after_watertight and not before_watertight):
                applied.append("fill_holes")
        except Exception as exc:
            warnings.append(f"Step1X fill_holes skipped: {type(exc).__name__}: {exc}")
        try:
            trimesh.repair.fix_normals(processed)
            # Trimesh.invert (used by fix_inversion) preserves cached normals
            # across its cache clear, so normals cached before the repair stay
            # stale (inward) even after the winding is fixed. Clear the cache
            # so normals are recomputed from the repaired faces.
            try:
                processed._cache.clear()
            except Exception:
                pass
            _ = processed.face_normals
            _ = processed.vertex_normals
            applied.append("fix_normals")
        except Exception as exc:
            warnings.append(f"Step1X fix_normals skipped: {type(exc).__name__}: {exc}")
        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()
        return processed, applied, warnings
    except Exception as exc:
        warnings.append(f"Step1X topology repair skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings


def _step1x_canonicalize_mesh_axes(
    mesh: Any,
    *,
    prompt: Optional[str] = None,
) -> tuple[Any, List[str], List[str], Dict[str, Any]]:
    applied: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}
    try:
        import numpy as np
        import trimesh

        processed = mesh.copy() if hasattr(mesh, "copy") else mesh
        try:
            poses, probs = trimesh.poses.compute_stable_poses(processed, n_samples=5, sigma=0.0)
            if len(poses):
                processed.apply_transform(poses[0])
                applied.append("stable_pose_z_up")
                details["stable_pose_probability"] = round(float(probs[0]), 6)
        except Exception as exc:
            warnings.append(f"Step1X stable-pose alignment skipped: {type(exc).__name__}: {exc}")
        try:
            lowered = str(prompt or "").strip().lower()
            if any(keyword in lowered for keyword in _UPRIGHT_FURNITURE_KEYWORDS):
                processed.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]))
                applied.append("upright_furniture_flip_x_pi")
        except Exception as exc:
            warnings.append(f"Step1X upright furniture flip skipped: {type(exc).__name__}: {exc}")
        try:
            bounds = np.asarray(processed.bounds, dtype=np.float64)
            center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
            translation = np.array([-center_xy[0], -center_xy[1], -bounds[0, 2]], dtype=np.float64)
            processed.apply_translation(translation)
            applied.append("center_xy_ground_z")
        except Exception as exc:
            warnings.append(f"Step1X axis recenter skipped: {type(exc).__name__}: {exc}")
        return processed, applied, warnings, details
    except Exception as exc:
        warnings.append(f"Step1X axis canonicalization skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings, details


def _step1x_mesh_topology(mesh: Any) -> Dict[str, Any]:
    topology: Dict[str, Any] = {}
    try:
        topology["is_watertight"] = bool(getattr(mesh, "is_watertight"))
    except Exception:
        topology["is_watertight"] = None
    try:
        topology["body_count"] = int(getattr(mesh, "body_count"))
    except Exception:
        topology["body_count"] = None
    try:
        topology["euler_number"] = int(getattr(mesh, "euler_number"))
    except Exception:
        topology["euler_number"] = None
    return topology


def _step1x_mps_memory_stats(torch: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "mps_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
        "mps_recommended_max_bytes": None,
    }
    if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
        return stats
    try:
        stats["mps_allocated_bytes"] = int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    try:
        stats["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    try:
        stats["mps_recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
    except Exception:
        pass
    return stats


def _step1x_apply_mps_memory_cap(owner: Any, torch: Any) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "mps_memory_cap_bytes": None,
        "mps_memory_fraction": None,
        "mps_memory_cap_applied": False,
    }
    if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
        return details
    cap_bytes = _step1x_mps_memory_cap_bytes(owner)
    if cap_bytes is None:
        return details
    details["mps_memory_cap_bytes"] = int(cap_bytes)
    recommended = None
    try:
        recommended = int(torch.mps.recommended_max_memory())
    except Exception:
        recommended = None
    fraction = 0.75
    if recommended and recommended > 0:
        fraction = min(0.95, max(0.05, float(cap_bytes) / float(recommended)))
    details["mps_memory_fraction"] = round(float(fraction), 6)
    try:
        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if callable(setter):
            setter(float(fraction))
            details["mps_memory_cap_applied"] = True
    except Exception:
        pass
    return details


def _step1x_release_runtime_memory(torch: Any) -> None:
    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def _volume_decoders_source() -> str:
    return '''from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from tqdm import tqdm


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    values = input_tensor + float(alpha)
    valid = values > -9000
    mask = torch.zeros_like(values, dtype=torch.bool)
    dims = values.ndim
    for axis in range(dims):
        for shift in (-1, 1):
            shifted = torch.roll(values, shifts=shift, dims=axis)
            slicer = [slice(None)] * dims
            slicer[axis] = 0 if shift > 0 else -1
            shifted[tuple(slicer)] = values[tuple(slicer)]
            shifted = torch.where(shifted > -9000, shifted, values)
            mask |= torch.sign(shifted.to(torch.float32)) != torch.sign(values.to(torch.float32))
    return mask.to(torch.int32) * valid.to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    coords = [
        np.linspace(bbox_min[index], bbox_max[index], int(octree_resolution) + 1, dtype=np.float32)
        for index in range(3)
    ]
    xs, ys, zs = np.meshgrid(*coords, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(octree_resolution) + 1] * 3
    return xyz, grid_size, bbox_max - bbox_min


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = 384,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3], dtype=np.float32)
        bbox_max = np.array(bounds[3:6], dtype=np.float32)
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij",
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device=device, dtype=dtype).reshape(-1, 3)
        pieces = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Volume Decoding",
            disable=not enable_pbar,
        ):
            queries = xyz_samples[start : start + num_chunks, :]
            features = geo_decoder(queries=repeat(queries, "p c -> b p c", b=batch_size), latents=latents)
            pieces.append(features)
        grid_features = torch.cat(pieces, dim=1)
        grid_logits = grid_features[..., 0:1].view((batch_size, *grid_size)).float()
        return grid_logits, xyz_samples, grid_features[..., 1:], None


class HierarchicalVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 65536,
        mc_level: float = 0.0,
        octree_resolution: int = 384,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        empty_value: float = float("nan"),
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        near_surface_band = float(kwargs.get("near_surface_band", 0.95) or 0.95)
        resolutions = []
        current = int(octree_resolution)
        if current < int(min_resolution):
            resolutions.append(current)
        while current >= int(min_resolution):
            resolutions.append(current)
            current //= 2
        resolutions.reverse()
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3], dtype=np.float32)
        bbox_max = np.array(bounds[3:6], dtype=np.float32)
        bbox_size = bbox_max - bbox_min
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij",
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device=device, dtype=dtype).reshape(-1, 3)
        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones_like(dilate.weight))
        batch_size = latents.shape[0]
        pieces = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]",
            disable=not enable_pbar,
        ):
            queries = xyz_samples[start : start + num_chunks, :]
            features = geo_decoder(queries=repeat(queries, "p c -> b p c", b=batch_size), latents=latents)
            pieces.append(features)
        grid_size_np = np.array(grid_size)
        grid_features = torch.cat(pieces, dim=1).view((batch_size, grid_size_np[0], grid_size_np[1], grid_size_np[2], -1))
        grid_logits = grid_features[..., 0]
        bbox_min_tensor = torch.tensor(bbox_min, dtype=dtype, device=device)
        for depth in resolutions[1:]:
            next_size = np.array([depth + 1] * 3)
            resolution = bbox_size / depth
            next_shape = tuple(int(item) for item in next_size.tolist())
            next_index = torch.zeros(next_shape, dtype=torch.bool, device=device)
            next_logits = torch.full((batch_size, *next_shape), -10000.0, dtype=dtype, device=device)
            active = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level) > 0
            active |= grid_logits.squeeze(0).abs() < near_surface_band
            expand_num = 0 if depth == resolutions[-1] else 1
            for _ in range(expand_num):
                active = dilate(active.unsqueeze(0).unsqueeze(0).to(dtype)).squeeze(0).squeeze(0) > 0
            coarse_index = torch.nonzero(active, as_tuple=False)
            if coarse_index.numel():
                next_index[coarse_index[:, 0] * 2, coarse_index[:, 1] * 2, coarse_index[:, 2] * 2] = True
            for _ in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0).unsqueeze(0).to(dtype)).squeeze(0).squeeze(0) > 0
            next_points = torch.nonzero(next_index, as_tuple=False)
            resolution_tensor = torch.tensor(resolution, dtype=dtype, device=device)
            for start in tqdm(
                range(0, next_points.shape[0], num_chunks),
                desc=f"Hierarchical Volume Decoding [r{depth + 1}]",
                disable=not enable_pbar,
            ):
                index_chunk = next_points[start : start + num_chunks, :]
                queries = index_chunk.to(dtype=dtype) * resolution_tensor + bbox_min_tensor
                features = geo_decoder(
                    queries=repeat(queries, "p c -> b p c", b=batch_size).to(dtype),
                    latents=latents,
                )
                logits = features[..., 0]
                for batch_index in range(batch_size):
                    next_logits[batch_index, index_chunk[:, 0], index_chunk[:, 1], index_chunk[:, 2]] = logits[batch_index]
                del queries, features, logits, index_chunk
            grid_logits = next_logits
        grid_logits[grid_logits == -10000.0] = empty_value
        return grid_logits
'''


def _surface_extractors_source() -> str:
    return '''from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure


class MeshExtractResult:
    def __init__(self, verts, faces, vertex_attrs=None, res=64, compute_normals=False):
        self.verts = verts
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = None
        self.vert_normal = None
        self.res = res
        self.success = verts.shape[0] != 0 and faces.shape[0] != 0
        if compute_normals and self.success:
            self.face_normal = self.comput_face_normals()
            self.vert_normal = self.comput_v_normals()

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None

    def comput_face_normals(self):
        i0 = self.faces[..., 0].long()
        i1 = self.faces[..., 1].long()
        i2 = self.faces[..., 2].long()

        v0 = self.verts[i0, :]
        v1 = self.verts[i1, :]
        v2 = self.verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self):
        i0 = self.faces[..., 0].long()
        i1 = self.faces[..., 1].long()
        i2 = self.faces[..., 2].long()

        v0 = self.verts[i0, :]
        v1 = self.verts[i1, :]
        v2 = self.verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(self.verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(
        self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int
    ):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
        ]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                verts, faces = self.run(grid_logits[i], **kwargs)
                outputs.append(
                    MeshExtractResult(
                        verts=verts.float(),
                        faces=faces,
                        res=kwargs["octree_resolution"],
                    )
                )

            except Exception:
                import traceback

                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def _crop_active_volume(self, volume: np.ndarray, mc_level: float):
        finite = np.isfinite(volume)
        if not finite.any():
            return np.nan_to_num(volume, nan=np.float32(mc_level + 1.0)), np.zeros(3, dtype=np.int64)
        coords = np.argwhere(finite)
        lower = np.maximum(coords.min(axis=0) - 1, 0)
        upper = np.minimum(coords.max(axis=0) + 2, np.array(volume.shape))
        cropped = volume[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        cropped = np.nan_to_num(
            cropped,
            nan=np.float32(mc_level + 1.0),
            posinf=np.float32(mc_level + 1.0),
            neginf=np.float32(mc_level - 1.0),
        )
        return cropped, lower

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        volume = grid_logit.float().cpu().numpy()
        volume, offset = self._crop_active_volume(volume, mc_level)
        if float(np.nanmin(volume)) > float(mc_level) or float(np.nanmax(volume)) < float(mc_level):
            raise ValueError("Marching-cubes level is outside the extracted volume range.")
        verts, faces, normals, _ = measure.marching_cubes(volume, mc_level, method="lewiner")
        verts = verts + offset[None, :]
        grid_size, bbox_min, bbox_size = self._compute_box_stat(
            bounds, octree_resolution
        )
        verts = verts / grid_size * bbox_size + bbox_min
        verts = torch.tensor(verts, device=grid_logit.device, dtype=torch.float32)
        faces = torch.tensor(
            np.ascontiguousarray(faces), device=grid_logit.device, dtype=torch.long
        )
        faces = faces[:, [2, 1, 0]]
        return verts, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        device = grid_logit.device
        if not hasattr(self, "dmc"):
            try:
                from diso import DiffDMC
            except:
                raise ImportError(
                    "Please install diso via `pip install diso`, or set mc_algo to 'mc'"
                )
            self.dmc = DiffDMC(dtype=torch.float32).to(device)
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        grid_size, bbox_min, bbox_size = self._compute_box_stat(
            kwargs["bounds"], octree_resolution
        )
        verts = verts * kwargs["bounds"] * 2 - kwargs["bounds"]
        return verts, faces
'''


def _pipeline_utils_preprocess_source() -> str:
    return '''def preprocess_image(
    images_pil: Union[List[PIL.Image.Image], PIL.Image.Image],
    force: bool = False,
    background_color: List[int] = [255, 255, 255],
    foreground_ratio: float = 0.9,
    rembg_backend: str = "bria",
    **rembg_kwargs,
):
    r"""
    Crop and remove the background of the input image.
    """
    is_single_image = False
    if isinstance(images_pil, PIL.Image.Image):
        images_pil = [images_pil]
        is_single_image = True
    preprocessed_images = []
    for i in range(len(images_pil)):
        image = images_pil[i]
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        width, height, size = image.width, image.height, image.size
        do_remove = bool(force)
        if image.getchannel("A").getextrema()[0] < 255:
            print("alpha channel not empty, skip remove background, using alpha channel as mask")
            do_remove = False
        if do_remove:
            import rembg  # lazy import

            image = rembg.remove(image, **rembg_kwargs)

        alpha = image.getchannel("A")
        bbox = alpha.getbbox() or (0, 0, width, height)
        x1, y1, x2, y2 = bbox
        dy = max(1, y2 - y1)
        dx = max(1, x2 - x1)
        scale = min(height * foreground_ratio / dy, width * foreground_ratio / dx)
        target_h = max(1, int(dy * scale))
        target_w = max(1, int(dx * scale))

        background = PIL.Image.new("RGBA", image.size, (*background_color, 255))
        composited = PIL.Image.alpha_composite(background, image)
        cropped_image = composited.crop(bbox)
        cropped_alpha = alpha.crop(bbox)

        resized_image = cropped_image.resize((target_w, target_h))
        resized_alpha = cropped_alpha.resize((target_w, target_h))
        padded_image = PIL.Image.new("RGB", size, tuple(background_color))
        padded_alpha = PIL.Image.new("L", size, 0)
        paste_position = (
            (width - resized_image.width) // 2,
            (height - resized_image.height) // 2,
        )
        padded_image.paste(resized_image, paste_position)
        padded_alpha.paste(resized_alpha, paste_position)

        width, height = padded_image.size
        if width == height:
            padded_image.putalpha(padded_alpha)
            preprocessed_images.append(padded_image)
            continue
        square = max(width, height)
        new_image = PIL.Image.new("RGB", (square, square), tuple(background_color))
        new_alpha = PIL.Image.new("L", (square, square), 0)
        paste_position = ((square - width) // 2, (square - height) // 2)
        new_image.paste(padded_image, paste_position)
        new_alpha.paste(padded_alpha, paste_position)
        new_image.putalpha(new_alpha)
        preprocessed_images.append(new_image)

    if is_single_image:
        return preprocessed_images[0]
    return preprocessed_images
'''


def _patch_step1x_source(source_dir: Path) -> None:
    init_path = source_dir / "step1x3d_geometry" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    if "from . import data, models, systems" in init_text:
        init_text = init_text.replace("from . import data, models, systems", "from . import models", 1)
        init_path.write_text(init_text, encoding="utf-8")

    volume_path = source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "volume_decoders.py"
    volume_path.write_text(_volume_decoders_source(), encoding="utf-8")

    surface_extractors_path = source_dir / "step1x3d_geometry" / "models" / "autoencoders" / "surface_extractors.py"
    surface_extractors_path.write_text(_surface_extractors_source(), encoding="utf-8")

    pipeline_utils_path = source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline_utils.py"
    pipeline_text = pipeline_utils_path.read_text(encoding="utf-8")
    preprocess_start = pipeline_text.find("def preprocess_image(")
    preprocess_end = pipeline_text.find("\ndef load_mesh(", preprocess_start)
    if preprocess_start >= 0:
        if preprocess_end <= preprocess_start:
            preprocess_end = len(pipeline_text)
        pipeline_text = (
            pipeline_text[:preprocess_start]
            + _pipeline_utils_preprocess_source()
            + ("\n\n" + pipeline_text[preprocess_end + 1 :] if preprocess_end < len(pipeline_text) else "\n")
        )
    pipeline_text = pipeline_text.replace(
        "elif isinstance(mesh, MeshExtractResult):\n"
        "        mesh = pymeshlab.MeshSet()\n"
        "        mesh_pymeshlab = pymeshlab.Mesh(\n"
        "            vertex_matrix=mesh.verts.cpu().numpy(), face_matrix=mesh.faces.cpu().numpy()\n"
        "        )\n"
        "        mesh.add_mesh(mesh_pymeshlab, \"converted_mesh\")\n",
        "elif isinstance(mesh, MeshExtractResult):\n"
        "        raw_mesh = mesh\n"
        "        mesh = pymeshlab.MeshSet()\n"
        "        mesh_pymeshlab = pymeshlab.Mesh(\n"
        "            vertex_matrix=raw_mesh.verts.cpu().numpy(), face_matrix=raw_mesh.faces.cpu().numpy()\n"
        "        )\n"
        "        mesh.add_mesh(mesh_pymeshlab, \"converted_mesh\")\n",
        1,
    )
    pipeline_utils_path.write_text(pipeline_text, encoding="utf-8")

    pipeline_path = source_dir / "step1x3d_geometry" / "models" / "pipelines" / "pipeline.py"
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    pipeline_source = pipeline_source.replace(
        "from ..conditional_encoders.dinov2_encoder import Dinov2Encoder\n",
        "from ..conditional_encoders.base import BaseVisualEncoder\n",
        1,
    )
    pipeline_source = pipeline_source.replace(
        "visual_encoder: Dinov2Encoder",
        "visual_encoder: BaseVisualEncoder",
        1,
    )
    pipeline_source = pipeline_source.replace(
        "elif isinstance(image, (torch.Tensor, PIL.Image.Image)):",
        "elif not isinstance(image, (torch.Tensor, PIL.Image.Image)):",
        1,
    )
    pipeline_path.write_text(pipeline_source, encoding="utf-8")

    misc_path = source_dir / "step1x3d_geometry" / "utils" / "misc.py"
    if misc_path.exists():
        misc_text = misc_path.read_text(encoding="utf-8")
        old_get_device = "def get_device():\n    return torch.device(f\"cuda:{get_rank()}\")\n"
        new_get_device = (
            "def get_device():\n"
            "    if torch.cuda.is_available():\n"
            "        return torch.device(f\"cuda:{get_rank()}\")\n"
            "    if getattr(torch.backends, \"mps\", None) is not None and torch.backends.mps.is_available():\n"
            "        return torch.device(\"mps\")\n"
            "    return torch.device(\"cpu\")\n"
        )
        misc_text = misc_text.replace(old_get_device, new_get_device, 1)
        misc_path.write_text(misc_text, encoding="utf-8")

    for rel_path in (
        Path("step1x3d_geometry/models/conditional_encoders/dinov2/modeling_dinov2.py"),
        Path("step1x3d_geometry/models/conditional_encoders/dinov2_with_registers/modeling_dinov2_with_registers.py"),
    ):
        path = source_dir / rel_path
        text = path.read_text(encoding="utf-8")
        if "from transformers.pytorch_utils import (\n    find_pruneable_heads_and_indices,\n    prune_linear_layer,\n)" in text:
            text = text.replace(
                "from transformers.pytorch_utils import (\n    find_pruneable_heads_and_indices,\n    prune_linear_layer,\n)\n",
                "from transformers.pytorch_utils import prune_linear_layer\n",
                1,
            )
        if "def find_pruneable_heads_and_indices(" not in text:
            marker = "logger = logging.get_logger(__name__)\n"
            compat = """logger = logging.get_logger(__name__)\n\n\ndef find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):\n    mask = torch.ones(n_heads, head_size)\n    heads = set(heads) - already_pruned_heads\n    for head in heads:\n        head = head - sum(1 if item < head else 0 for item in already_pruned_heads)\n        mask[head] = 0\n    mask = mask.view(-1).contiguous().eq(1)\n    index = torch.arange(mask.numel(), device=mask.device)[mask].long()\n    return heads, index\n\n\ndef get_head_mask(head_mask, num_hidden_layers, is_attention_chunked: bool = False):\n    if head_mask is None:\n        return [None] * num_hidden_layers\n    if head_mask.dim() == 1:\n        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\n        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)\n    elif head_mask.dim() == 2:\n        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)\n    head_mask = head_mask.to(dtype=torch.float32)\n    if is_attention_chunked:\n        head_mask = head_mask.unsqueeze(-1)\n    return head_mask\n"""
            if marker not in text:
                raise SourceBootstrapError(f"Could not patch Step1X DINO compatibility in {path}")
            text = text.replace(marker, compat, 1)
        text = text.replace("self.get_head_mask(head_mask, self.config.num_hidden_layers)", "get_head_mask(head_mask, self.config.num_hidden_layers)")
        path.write_text(text, encoding="utf-8")

    label_encoder_path = source_dir / "step1x3d_geometry" / "models" / "conditional_encoders" / "label_encoder.py"
    if label_encoder_path.exists():
        label_text = label_encoder_path.read_text(encoding="utf-8")
        label_text = label_text.replace(
            "torch.tensor(POSE_MAPPING[label[\"pose\"][0]]).to(",
            "torch.tensor(POSE_MAPPING[label[\"pose\"][0] if isinstance(label[\"pose\"], (list, tuple)) else label[\"pose\"]]).to(",
            1,
        )
        label_text = label_text.replace(
            "GEOMETRY_QUALITY_MAPPING[label[\"geometry_type\"][0]]",
            "GEOMETRY_QUALITY_MAPPING[label[\"geometry_type\"][0] if isinstance(label[\"geometry_type\"], (list, tuple)) else label[\"geometry_type\"]]",
            1,
        )
        label_encoder_path.write_text(label_text, encoding="utf-8")


def _write_source_manifest(
    source_dir: Path,
    *,
    repo_url: str,
    commit: str,
    patchset_version: str,
    source_origin: Optional[Path] = None,
) -> None:
    payload: Dict[str, Any] = {
        "commit": commit,
        "patchset_version": patchset_version,
        "repo_url": repo_url,
    }
    if source_origin is not None:
        payload["source_origin"] = str(source_origin)
    (source_dir / _SOURCE_MANIFEST).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clone_repo(*, repo_url: str, commit: str, repo_dir: Path) -> None:
    if (repo_dir / "step1x3d_geometry" / "__init__.py").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="abstract3d-step1x-", dir=str(repo_dir.parent)))
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", repo_url, str(tmp_root / "repo")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_root / "repo"), "checkout", "--detach", commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        git_dir = tmp_root / "repo" / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)
        _patch_step1x_source(tmp_root / "repo")
        _write_source_manifest(
            tmp_root / "repo",
            repo_url=_STEP1X_REPO_URL,
            commit=_STEP1X_COMMIT,
            patchset_version=_PATCHSET_VERSION,
        )
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.move(str(tmp_root / "repo"), str(repo_dir))
    except subprocess.CalledProcessError as exc:
        raise SourceBootstrapError(exc.stderr or exc.stdout or str(exc)) from exc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _verify_official_source_dir(source_dir: Path) -> Path:
    resolved = source_dir.expanduser().resolve()
    init_path = resolved / "step1x3d_geometry" / "__init__.py"
    manifest_path = resolved / _SOURCE_MANIFEST
    if not init_path.exists():
        raise SourceBootstrapError(f"Configured Step1X source dir does not contain step1x3d_geometry: {resolved}")
    if not manifest_path.exists():
        raise SourceBootstrapError(f"Configured Step1X source dir is missing {_SOURCE_MANIFEST}: {resolved}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_url = _normalize_repo_url(manifest.get("repo_url"))
    commit = str(manifest.get("commit") or "").strip()
    if repo_url != _normalize_repo_url(_STEP1X_REPO_URL) or commit != _STEP1X_COMMIT:
        raise SourceBootstrapError(
            "Configured Step1X source dir is not the pinned official snapshot. "
            f"Expected repo {_STEP1X_REPO_URL!r} at commit {_STEP1X_COMMIT!r}."
        )
    return resolved


def _managed_source_copy_dir(owner: Any, source_dir: Path) -> Path:
    cache_root = _cache_root(owner) if owner is not None else _DEFAULT_CACHE_ROOT
    digest = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:12]
    return cache_root / "vendor" / "step1x" / f"{_STEP1X_COMMIT}-managed-{digest}"


def _copy_and_patch_source_tree(*, owner: Any, source_dir: Path) -> Path:
    managed_dir = _managed_source_copy_dir(owner, source_dir)
    manifest_path = managed_dir / _SOURCE_MANIFEST
    if managed_dir.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            _normalize_repo_url(manifest.get("repo_url")) == _normalize_repo_url(_STEP1X_REPO_URL)
            and str(manifest.get("commit") or "").strip() == _STEP1X_COMMIT
            and str(manifest.get("patchset_version") or "").strip() == _PATCHSET_VERSION
            and str(manifest.get("source_origin") or "").strip() == str(source_dir)
        ):
            return managed_dir

    managed_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="abstract3d-step1x-managed-", dir=str(managed_dir.parent)))
    try:
        staged = tmp_root / "repo"
        shutil.copytree(source_dir, staged)
        _patch_step1x_source(staged)
        _write_source_manifest(
            staged,
            repo_url=_STEP1X_REPO_URL,
            commit=_STEP1X_COMMIT,
            patchset_version=_PATCHSET_VERSION,
            source_origin=source_dir,
        )
        if managed_dir.exists():
            shutil.rmtree(managed_dir)
        shutil.move(str(staged), str(managed_dir))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return managed_dir


def _resolve_source_dir(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_step1x_source_dir") or _env("ABSTRACT3D_STEP1X_SOURCE_DIR")
    if configured:
        with _PATCH_LOCK:
            resolved = _verify_official_source_dir(Path(str(configured)))
            managed = _copy_and_patch_source_tree(owner=owner, source_dir=resolved)
        return _verify_official_source_dir(managed)
    repo_dir = (_cache_root(owner) if owner is not None else _DEFAULT_CACHE_ROOT) / "vendor" / "step1x" / _STEP1X_COMMIT
    with _PATCH_LOCK:
        _clone_repo(repo_url=_STEP1X_REPO_URL, commit=_STEP1X_COMMIT, repo_dir=repo_dir)
        _patch_step1x_source(repo_dir)
    return _verify_official_source_dir(repo_dir)


def _model_id(model: Optional[str]) -> str:
    selected = str(model or _DEFAULT_MODEL_ID).strip()
    if selected != _OFFICIAL_MODEL_ID:
        raise CapabilityNotSupportedError(
            "Only the official Step1X model repo is supported. "
            f"Expected {_OFFICIAL_MODEL_ID!r}, got {selected!r}."
        )
    return selected


def _geometry_subfolder(model_subfolder: Optional[str]) -> str:
    selected = str(model_subfolder or _DEFAULT_GEOMETRY_SUBFOLDER).strip()
    if selected not in _SUPPORTED_GEOMETRY_SUBFOLDERS:
        raise CapabilityNotSupportedError(
            "Only the official Step1X geometry checkpoints are supported. "
            f"Expected one of {list(_SUPPORTED_GEOMETRY_SUBFOLDERS)!r}, got {selected!r}."
        )
    return selected


def _step1x_select_geometry_subfolder(
    *,
    requested: Optional[str],
    device: str,
    task: str,
    label: Optional[Mapping[str, Any]],
) -> tuple[str, str]:
    if requested is not None:
        return _geometry_subfolder(requested), "explicit"
    if str(device) == "mps" and str(task) == "image_to_scene3d":
        symmetry = str((label or {}).get("symmetry") or "").strip().lower()
        geometry_type = str((label or {}).get("geometry_type") or "").strip().lower()
        if symmetry == "asymmetry" and geometry_type == "sharp":
            return _GEOMETRY_SUBFOLDER, "mps_i23d_sharp_asymmetry_base_fallback"
    return _DEFAULT_GEOMETRY_SUBFOLDER, "default_label"


def _download_geometry_snapshot(*, model_id: str, revision: str, subfolder: str) -> Path:
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    root = snapshot_download(repo_id=model_id, revision=revision, allow_patterns=[f"{subfolder}/*"])
    return Path(root).resolve()


@contextmanager
def _sys_path(path: Path):
    raw = str(path)
    inserted = raw not in sys.path
    if inserted:
        sys.path.insert(0, raw)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(raw)
            except ValueError:
                pass


class Step1XGeometryBackend:
    """Experimental local Step1X geometry backend."""

    backend_id = "abstract3d:step1x-local"

    def __init__(self, owner: Any, *, image_generator: Optional[Callable[..., Any]] = None) -> None:
        self._owner = owner
        self._image_generator = image_generator
        self._resident_pipeline = None
        self._resident_model_id: Optional[str] = None
        self._resident_device: Optional[str] = None
        self._resident_dtype: Optional[str] = None
        self._resident_subfolder: Optional[str] = None
        self._last_runtime_stats: Dict[str, Any] = {}

    def _load_runtime(
        self,
        *,
        model_id: Optional[str],
        device: Optional[str],
        dtype: Optional[str],
        model_subfolder: Optional[str],
    ):
        _require_runtime_dependencies()
        torch = importlib.import_module("torch")
        selected_model = _model_id(model_id or _owner_cfg(self._owner, "scene3d_model_id"))
        selected_subfolder = _geometry_subfolder(model_subfolder or _owner_cfg(self._owner, "scene3d_step1x_subfolder"))
        selected_device = _select_device(self._owner, device)
        selected_dtype = _select_dtype(selected_device, dtype or _owner_cfg(self._owner, "scene3d_step1x_dtype"))
        reuse = (
            self._resident_pipeline is not None
            and self._resident_model_id == selected_model
            and self._resident_device == selected_device
            and self._resident_dtype == selected_dtype
            and self._resident_subfolder == selected_subfolder
        )
        if reuse:
            return self._resident_pipeline

        started = time.perf_counter()
        source_dir = _resolve_source_dir(self._owner)
        snapshot_root = _download_geometry_snapshot(model_id=selected_model, revision=_MODEL_REVISION, subfolder=selected_subfolder)
        mps_memory_cap = _step1x_apply_mps_memory_cap(self._owner, torch)
        with _sys_path(source_dir):
            module = importlib.import_module("step1x3d_geometry.models.pipelines.pipeline")
            pipeline_cls = getattr(module, "Step1X3DGeometryPipeline")
            pipeline = pipeline_cls.from_pretrained(
                str(snapshot_root),
                subfolder=selected_subfolder,
                torch_dtype=_torch_dtype(selected_dtype),
            )
        pipeline = pipeline.to(selected_device)
        load_s = round(time.perf_counter() - started, 4)
        self._resident_pipeline = pipeline
        self._resident_model_id = selected_model
        self._resident_device = selected_device
        self._resident_dtype = selected_dtype
        self._resident_subfolder = selected_subfolder
        self._last_runtime_stats = {
            "load_s": load_s,
            "device": selected_device,
            "dtype": selected_dtype,
            "model_id": selected_model,
            "model_revision": _MODEL_REVISION,
            "subfolder": selected_subfolder,
            "snapshot_root": str(snapshot_root),
            "source_snapshot": _STEP1X_COMMIT,
            "source_dir": str(source_dir),
            **mps_memory_cap,
        }
        return pipeline

    def _clear_runtime(self) -> None:
        if self._resident_pipeline is None:
            return
        self._resident_pipeline = None
        self._resident_model_id = None
        self._resident_device = None
        self._resident_dtype = None
        self._resident_subfolder = None
        torch = importlib.import_module("torch")
        _step1x_release_runtime_memory(torch)

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.list_models(task=task, provider="step1x")
        composition_ready = has_image_composer(self._owner)
        normalized_task = _TASK_ALIASES.get(str(task).strip().lower().replace("-", "_")) if task is not None else None
        if normalized_task == "text_to_scene3d":
            tasks = ["text_to_scene3d"]
            status = "available" if composition_ready else "install_required"
            configured = composition_ready
        elif normalized_task == "image_to_scene3d":
            tasks = ["image_to_scene3d"]
            status = "available"
            configured = True
        else:
            tasks = ["image_to_scene3d"]
            if composition_ready:
                tasks.append("text_to_scene3d")
            status = "available"
            configured = True
        return [
            {
                "provider_id": "step1x",
                "backend_id": self.backend_id,
                "experimental": True,
                "tasks": tasks,
                "status": status,
                "configured": configured,
                "models": rows,
                "metadata": {
                    "text_mode": "abstractvision_composition",
                    "composition_ready": composition_ready,
                    "composition_install_hint": COMPOSITION_INSTALL_HINT,
                },
            }
        ]

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        selector = str(provider_id or provider or "").strip().lower()
        if selector and selector not in {"step1x", self.backend_id}:
            return []
        rows = []
        for spec in iter_model_specs(validated_only=False, task=task):
            if spec.provider_id != "step1x":
                continue
            payload = spec.to_capability_model()
            payload["provider_id"] = "step1x"
            rows.append(payload)
        return rows

    def list_provider_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.list_models(task=task, provider=provider, provider_id=provider_id)

    def list_operations(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        operations = []
        if has_image_composer(self._owner):
            operations.append(
                {
                    "operation_id": "text_to_scene3d",
                    "task": "text_to_scene3d",
                    "input_modalities": ["text"],
                    "output_modalities": ["scene3d"],
                    "artifact_output": True,
                    "parameter_schema": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string", "enum": ["glb", "obj", "zip"]},
                            "num_inference_steps": {"type": "integer"},
                            "mc_resolution": {"type": "integer"},
                            "device": {"type": "string"},
                        },
                    },
                }
            )
        operations.extend(
            [
            {
                "operation_id": "image_to_scene3d",
                "task": "image_to_scene3d",
                "input_modalities": ["image"],
                "output_modalities": ["scene3d"],
                "artifact_output": True,
                "parameter_schema": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["glb", "obj", "zip"]},
                        "num_inference_steps": {"type": "integer"},
                        "mc_resolution": {"type": "integer"},
                        "remove_background": {"type": "boolean"},
                        "device": {"type": "string"},
                    },
                },
            },
            ]
        )
        if task is None:
            return operations
        normalized = _TASK_ALIASES.get(str(task).strip().lower().replace("-", "_"))
        return [item for item in operations if item["task"] == normalized]

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        already_loaded = self._resident_pipeline is not None
        self._load_runtime(
            model_id=request.get("model"),
            device=request.get("device"),
            dtype=request.get("dtype"),
            model_subfolder=request.get("subfolder"),
        )
        return {
            "task": str(request.get("task") or "scene3d_generation"),
            "provider": "step1x",
            "model": self._resident_model_id,
            "backend_id": self.backend_id,
            "state": "loaded",
            "loaded": True,
            "loaded_new": not already_loaded,
            "details": dict(self._last_runtime_stats),
        }

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        _ = filters
        if self._resident_pipeline is None:
            return []
        return [
            {
                "task": "scene3d_generation",
                "provider": "step1x",
                "model": self._resident_model_id,
                "backend_id": self.backend_id,
                "state": "loaded",
                "loaded": True,
                "details": dict(self._last_runtime_stats),
            }
        ]

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        return self.list_loaded_models(filters)

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = request
        model_id = self._resident_model_id
        self._clear_runtime()
        return {
            "task": "scene3d_generation",
            "provider": "step1x",
            "model": model_id,
            "backend_id": self.backend_id,
            "state": "unloaded",
            "unloaded": True,
        }

    def _make_source_image(self, prompt: str, **kwargs: Any) -> bytes:
        generator = self._image_generator or _default_image_generator(self._owner)
        image_prompt = _default_text_to_image_prompt(prompt)
        image_request = pop_image_generation_request(self._owner, kwargs)
        result = generator(image_prompt, **image_request)
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        if isinstance(result, Mapping):
            for key in ("data", "bytes", "content"):
                content = result.get(key)
                if isinstance(content, (bytes, bytearray)):
                    return bytes(content)
        raise Abstract3DError("Text-to-image helper did not return raw image bytes.")

    def _run_generation(
        self,
        *,
        task: str,
        prompt: str,
        image: Optional[Any],
        format: str,
        artifact_store: Optional[Any],
        run_id: Optional[str],
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
        output_dir: Optional[str],
        remove_background: Optional[bool],
        num_inference_steps: Optional[int],
        octree_resolution: Optional[int],
        device: Optional[str],
        dtype: Optional[str],
        model: Optional[str],
        model_subfolder: Optional[str],
        **kwargs: Any,
    ):
        from PIL import Image
        import psutil
        import torch

        actual_task = _TASK_ALIASES.get(task, task)
        if actual_task not in {"text_to_scene3d", "image_to_scene3d"}:
            raise CapabilityNotSupportedError(f"Unsupported Step1X task: {actual_task!r}")

        image_generation_s: Optional[float] = None
        if actual_task == "text_to_scene3d":
            image_started = time.perf_counter()
            image_bytes = self._make_source_image(prompt, **kwargs)
            image_generation_s = round(time.perf_counter() - image_started, 4)
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            _release_mlx_generation_cache()
        else:
            if image is None:
                raise ValueError("image_to_scene3d requires an image input.")
            source_image = _load_image_payload(image, artifact_store=artifact_store).convert("RGBA")
        selected_device = _select_device(self._owner, device)
        selected_dtype = _select_dtype(selected_device, dtype)
        source_dir = _resolve_source_dir(self._owner)

        resolved_background, background_policy = _resolve_background_removal_policy(
            task=actual_task,
            image=source_image,
            explicit=remove_background,
        )
        foreground_ratio = _step1x_foreground_ratio(self._owner, kwargs.pop("foreground_ratio", None))
        preprocess_started = time.perf_counter()
        with _sys_path(source_dir):
            conditioned_image = _preprocess_step1x_image(
                image=source_image,
                force_remove_background=resolved_background,
                foreground_ratio=foreground_ratio,
            )
        preprocess_s = round(time.perf_counter() - preprocess_started, 4)
        source_preview = conditioned_image.convert("RGB")
        requested_subfolder = model_subfolder or _owner_cfg(self._owner, "scene3d_step1x_subfolder")
        resolved_label = _normalize_step1x_label(kwargs.pop("label", None))
        if resolved_label is None and (requested_subfolder is None or "label" in str(requested_subfolder).lower()):
            resolved_label = _infer_step1x_label(prompt=prompt, processed_image=conditioned_image)
        selected_subfolder, geometry_subfolder_policy = _step1x_select_geometry_subfolder(
            requested=requested_subfolder,
            device=selected_device,
            task=actual_task,
            label=resolved_label if isinstance(resolved_label, Mapping) else None,
        )
        if resolved_label is None and selected_subfolder == _GEOMETRY_LABEL_SUBFOLDER:
            resolved_label = _infer_step1x_label(prompt=prompt, processed_image=conditioned_image)

        pipeline = self._load_runtime(
            model_id=model,
            device=selected_device,
            dtype=selected_dtype,
            model_subfolder=selected_subfolder,
        )
        resident_model_id = self._resident_model_id or _model_id(model)
        resident_device = self._resident_device or selected_device
        resident_dtype = self._resident_dtype or selected_dtype
        resident_subfolder = self._resident_subfolder or selected_subfolder
        runtime_stats = dict(self._last_runtime_stats)
        keep_runtime_resident = _step1x_keep_runtime_resident(self._owner, resident_device)

        runtime_device = resident_device
        resolved_steps = _step1x_default_num_inference_steps(self._owner, runtime_device, num_inference_steps)
        resolved_octree = _step1x_default_octree_resolution(self._owner, runtime_device, octree_resolution)
        resolved_octree, octree_policy = _step1x_tuned_octree_resolution(
            prompt=prompt,
            device=runtime_device,
            resolved=resolved_octree,
            explicit=octree_resolution,
        )
        explicit_guidance_scale = kwargs.pop("guidance_scale", None)
        guidance_scale = _step1x_default_guidance_scale(self._owner, runtime_device, explicit_guidance_scale)
        explicit_max_facenum = kwargs.pop("max_facenum", None)
        max_facenum = _step1x_default_max_facenum(self._owner, runtime_device, explicit_max_facenum)
        cleanup_flags = _step1x_cleanup_flags(self._owner)
        pipeline_cleanup_flags = dict(cleanup_flags)
        cleanup_mode = "runtime_defaults"
        if (self._resident_subfolder or _DEFAULT_GEOMETRY_SUBFOLDER) == _GEOMETRY_LABEL_SUBFOLDER and str(runtime_device) == "mps":
            pipeline_cleanup_flags = {
                "do_remove_floater": False,
                "do_remove_degenerate_face": False,
                "do_reduce_face": False,
            }
            cleanup_mode = "label_geometry_cleanup_disabled_on_mps"
        use_cpu_surface_extract = str(runtime_device) == "mps"
        label_for_pipeline = resolved_label if resident_subfolder == _GEOMETRY_LABEL_SUBFOLDER else None
        postprocess_max_facenum, postprocess_max_facenum_policy = _step1x_effective_postprocess_max_facenum(
            device=runtime_device,
            resolved=int(max_facenum),
            explicit=explicit_max_facenum,
            label=label_for_pipeline if isinstance(label_for_pipeline, Mapping) else None,
        )
        cpu_extract_num_chunks = _step1x_cpu_extract_num_chunks(self._owner, runtime_device)
        cpu_extract_surface_band = _step1x_cpu_extract_surface_band(
            self._owner,
            device=runtime_device,
            label=label_for_pipeline if isinstance(label_for_pipeline, Mapping) else None,
        )
        seed = int(kwargs.pop("seed", kwargs.pop("image_seed", 2025)) or 2025)
        try:
            generator = torch.Generator(device=self._resident_device or "cpu")
        except (RuntimeError, ValueError):
            # Torch builds without the resident backend (for example no MPS on
            # Linux hosts) still get deterministic seeding: a CPU generator is
            # valid for every pipeline and torch build.
            generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        started = time.perf_counter()
        result = pipeline(
            source_image,
            label=label_for_pipeline,
            num_inference_steps=int(resolved_steps),
            guidance_scale=float(guidance_scale),
            octree_resolution=int(resolved_octree),
            do_remove_floater=pipeline_cleanup_flags["do_remove_floater"],
            do_remove_degenerate_face=pipeline_cleanup_flags["do_remove_degenerate_face"],
            do_reduce_face=pipeline_cleanup_flags["do_reduce_face"],
            max_facenum=int(max_facenum),
            force_remove_background=bool(resolved_background),
            foreground_ratio=float(foreground_ratio),
            generator=generator,
            output_type="latent" if use_cpu_surface_extract else "trimesh",
        )
        total_infer_s = round(time.perf_counter() - started, 4)
        mesh_stage_started = time.perf_counter()
        postprocess_applied: List[str] = []
        postprocess_warnings: List[str] = []
        surface_extract_runtime: Dict[str, Any] = {}
        helper_bundle: Optional[Dict[str, Any]] = None
        latent_payload = result.mesh
        can_extract_latent_on_cpu = use_cpu_surface_extract and hasattr(latent_payload, "detach")
        if can_extract_latent_on_cpu:
            extracted_meshes, surface_extract_runtime = _step1x_mesh_extract_on_cpu(
                pipeline=pipeline,
                latents=latent_payload,
                octree_resolution=int(resolved_octree),
                preserve_runtime=keep_runtime_resident,
                num_chunks=int(cpu_extract_num_chunks),
                near_surface_band=float(cpu_extract_surface_band),
                source_dir=str(runtime_stats.get("source_dir") or ""),
                snapshot_root=str(runtime_stats.get("snapshot_root") or ""),
                geometry_subfolder=str(resident_subfolder),
                cleanup_flags=cleanup_flags,
                postprocess_max_facenum=int(postprocess_max_facenum),
                canonicalize_export=_owner_cfg_bool(self._owner, "scene3d_step1x_canonicalize_export_axes", True),
                prompt=prompt,
            )
            if isinstance(extracted_meshes, dict) and extracted_meshes.get("glb_bytes") is not None:
                helper_bundle = extracted_meshes
            else:
                mesh_result = extracted_meshes[0] if isinstance(extracted_meshes, list) else extracted_meshes
                mesh = _step1x_trimesh_from_extract_result(mesh_result)
                mesh, postprocess_applied, postprocess_warnings = _step1x_postprocess_mesh(
                    mesh=mesh,
                    cleanup_flags=cleanup_flags,
                    max_facenum=int(postprocess_max_facenum),
                )
        else:
            mesh = latent_payload[0] if isinstance(latent_payload, list) else latent_payload
        result = None
        generator = None
        pipeline = None
        if not keep_runtime_resident:
            self._clear_runtime()
        else:
            _step1x_release_runtime_memory(torch)
        mesh_s = round(time.perf_counter() - mesh_stage_started, 4)
        export_axis_applied: List[str] = []
        export_axis_warnings: List[str] = []
        export_axis_details: Dict[str, Any] = {}
        preview_axis_applied: List[str] = []
        preview_axis_warnings: List[str] = []
        preview_axis_details: Dict[str, Any] = {}
        if helper_bundle is not None:
            report = dict(helper_bundle.get("report") or {})
            glb_bytes = bytes(helper_bundle["glb_bytes"])
            obj_bytes = bytes(helper_bundle["obj_bytes"])
            views = list(helper_bundle.get("views") or [])
            topology = dict(report.get("topology") or {})
            vertex_count = int(report.get("vertex_count") or 0)
            face_count = int(report.get("face_count") or 0)
            postprocess_applied = list(report.get("postprocess_applied") or [])
            postprocess_warnings = list(report.get("postprocess_warnings") or [])
            export_axis = dict(report.get("export_axis_canonicalization") or {})
            export_axis_applied = list(export_axis.get("applied") or [])
            export_axis_warnings = list(export_axis.get("warnings") or [])
            export_axis_details = {key: value for key, value in export_axis.items() if key not in {"applied", "warnings"}}
            preview_axis_applied = list(export_axis_applied)
            preview_axis_warnings = list(export_axis_warnings)
            preview_axis_details = dict(export_axis_details)
        else:
            mesh, final_component_applied, final_component_warnings = _step1x_prune_components(mesh)
            postprocess_applied.extend(final_component_applied)
            postprocess_warnings.extend(final_component_warnings)
            export_mesh = mesh
            if _owner_cfg_bool(self._owner, "scene3d_step1x_canonicalize_export_axes", True):
                export_mesh, export_axis_applied, export_axis_warnings, export_axis_details = _step1x_canonicalize_mesh_axes(
                    mesh,
                    prompt=prompt,
                )
                preview_mesh = export_mesh
                preview_axis_applied = list(export_axis_applied)
                preview_axis_warnings = list(export_axis_warnings)
                preview_axis_details = dict(export_axis_details)
            else:
                preview_mesh = mesh
                if _owner_cfg_bool(self._owner, "scene3d_step1x_canonicalize_preview_axes", True):
                    preview_mesh, preview_axis_applied, preview_axis_warnings, preview_axis_details = _step1x_canonicalize_mesh_axes(
                        mesh,
                        prompt=prompt,
                    )
            topology = _step1x_mesh_topology(export_mesh)
            glb_bytes = _mesh_export_bytes(export_mesh, file_type="glb")
            obj_bytes = _mesh_export_bytes(export_mesh, file_type="obj")
            views = render_mesh_views(preview_mesh)
            vertex_count = int(len(mesh.vertices))
            face_count = int(len(mesh.faces))
        primary_format = str(format or "glb").strip().lower() or "glb"
        if primary_format not in {"glb", "obj", "zip"}:
            raise ValueError("scene3d format must be one of: glb, obj, zip")
        primary_bytes = glb_bytes if primary_format == "glb" else obj_bytes
        content_type = "model/gltf-binary" if primary_format == "glb" else "text/plain"
        if primary_format == "zip":
            content_type = "application/zip"

        process = psutil.Process(os.getpid())
        runtime_meta: Dict[str, Any] = {
            "backend_id": self.backend_id,
            "provider": self.backend_id,
            "model_id": resident_model_id,
            "model_revision": _MODEL_REVISION,
            "task": actual_task,
            "device": resident_device or "cpu",
            "dtype": resident_dtype,
            "format": primary_format,
            "content_type": content_type,
            "geometry_subfolder": resident_subfolder,
            "geometry_subfolder_policy": geometry_subfolder_policy,
            "geometry_only": True,
            "native_text_to_scene3d": False,
            "composed_text_to_scene3d": actual_task == "text_to_scene3d",
            "num_inference_steps": int(resolved_steps),
            "guidance_scale": float(guidance_scale),
            "octree_resolution": int(resolved_octree),
            "max_facenum": int(max_facenum),
            "postprocess_max_facenum": int(postprocess_max_facenum),
            "postprocess_max_facenum_policy": postprocess_max_facenum_policy,
            "foreground_ratio": float(foreground_ratio),
            "vertex_count": int(vertex_count),
            "face_count": int(face_count),
            "timings_s": {
                "source_image_generation": image_generation_s,
                "inference": total_infer_s,
                "mesh": mesh_s,
                "preprocess": preprocess_s,
                "load": runtime_stats.get("load_s"),
                "total": round((image_generation_s or 0.0) + preprocess_s + total_infer_s + mesh_s, 4),
            },
            "memory": {
                "rss_bytes": int(process.memory_info().rss),
                **_step1x_mps_memory_stats(torch),
            },
            "runtime_memory": {
                "keep_resident": bool(keep_runtime_resident),
                "released_before_export": not bool(keep_runtime_resident),
                "mps_memory_cap_bytes": runtime_stats.get("mps_memory_cap_bytes"),
                "mps_memory_fraction": runtime_stats.get("mps_memory_fraction"),
                "mps_memory_cap_applied": runtime_stats.get("mps_memory_cap_applied"),
            },
            "background_removed": bool(resolved_background),
            "background_removal_policy": background_policy,
            "label_condition": dict(label_for_pipeline) if isinstance(label_for_pipeline, Mapping) else None,
            "surface_cleanup": dict(cleanup_flags),
            "pipeline_cleanup": dict(pipeline_cleanup_flags),
            "postprocess_cleanup": list(postprocess_applied),
            "postprocess_warnings": list(postprocess_warnings),
            "cleanup_mode": cleanup_mode,
            "topology": topology,
            "octree_resolution_policy": octree_policy or "default",
            "surface_extract_device": "cpu" if use_cpu_surface_extract else resident_device,
            "source_snapshot": _STEP1X_COMMIT,
            "patchset_version": _PATCHSET_VERSION,
            "source_dir": runtime_stats.get("source_dir"),
            "surface_extract_runtime": dict(surface_extract_runtime),
            "cpu_extract_num_chunks": int(cpu_extract_num_chunks),
            "cpu_extract_surface_band": float(cpu_extract_surface_band),
            "export_axis_canonicalization": {
                "applied": list(export_axis_applied),
                "warnings": list(export_axis_warnings),
                **export_axis_details,
            },
            "preview_axis_canonicalization": {
                "applied": list(preview_axis_applied),
                "warnings": list(preview_axis_warnings),
                **preview_axis_details,
            },
            "notes": [
                "Apple MPS uses float32 for stability in the current local runtime.",
                "Apple MPS uses a lower default guidance scale than upstream to reduce local instability.",
                "The Step1X runtime supports the official base and label geometry checkpoints locally.",
                "Apple MPS uses CPU-side surface extraction after denoising to avoid local accelerator failures in marching-cubes extraction.",
                "The Step1X runtime feeds raw images plus preprocessing policy into the official pipeline so conditioning is only applied once inside the model path.",
                "The supported Step1X path in abstract3d is geometry-only; texture remains out of scope.",
            ],
        }
        if resident_subfolder == _GEOMETRY_LABEL_SUBFOLDER:
            runtime_meta["notes"].append(
                "The Step1X label-geometry checkpoint is supported locally and uses auto labels unless overridden explicitly."
            )
            runtime_meta["notes"].append(
                "The Step1X label-geometry path disables upstream in-pipeline cleanup on Apple MPS because that stage is not stable enough locally."
            )
        if geometry_subfolder_policy == "mps_i23d_sharp_asymmetry_base_fallback":
            runtime_meta["notes"].append(
                "Apple MPS sharp asymmetric i23d cases default to the official base geometry checkpoint because that local operating point preserves product-shape cues better than the coarse label prior."
            )
        if not keep_runtime_resident:
            runtime_meta["notes"].append(
                "On Apple MPS, abstract3d releases the Step1X runtime before mesh export and preview rendering to reduce unified-memory pressure."
            )
        if postprocess_applied:
            runtime_meta["notes"].append(
                "The Apple MPS label-geometry path applies deterministic CPU mesh cleanup after extraction."
            )
        if octree_policy == "thin_structure_prompt":
            runtime_meta["notes"].append(
                "Thin-structure prompts on Apple MPS raise the default octree resolution to 192 for better support recovery."
            )
        if export_axis_applied:
            runtime_meta["notes"].append(
                "Exported Step1X meshes and preview renders use a stable-pose canonical frame."
            )
        elif preview_axis_applied:
            runtime_meta["notes"].append(
                "Preview renders use a stable-pose canonical frame while the exported mesh keeps the raw model pose."
            )
        runtime_meta["notes"].extend(postprocess_warnings)
        runtime_meta["notes"].extend(export_axis_warnings)
        runtime_meta["notes"].extend(preview_axis_warnings)
        if isinstance(metadata, dict) and metadata:
            runtime_meta["request_metadata"] = dict(metadata)

        bundle_root: Optional[Path] = None
        bundle_paths: Dict[str, Optional[Path]] = {}
        if output_dir:
            bundle_root = Path(output_dir).expanduser().resolve()
            bundle_paths = _write_bundle(
                root_dir=bundle_root,
                primary_format="glb" if primary_format == "zip" else primary_format,
                primary_bytes=glb_bytes if primary_format == "zip" else primary_bytes,
                obj_bytes=obj_bytes,
                source_image=source_preview,
                prompt=prompt,
                metadata=runtime_meta,
                view_images=views,
            )
            runtime_meta["bundle_dir"] = str(bundle_root)
            runtime_meta["preview_path"] = str(bundle_paths.get("preview_path")) if bundle_paths.get("preview_path") else None
            runtime_meta["contact_sheet_path"] = str(bundle_paths.get("contact_sheet_path")) if bundle_paths.get("contact_sheet_path") else None
            runtime_meta["metadata_path"] = str(bundle_paths.get("metadata_path")) if bundle_paths.get("metadata_path") else None
            runtime_meta["source_image_path"] = str(bundle_paths.get("source_path")) if bundle_paths.get("source_path") else None

        if primary_format == "zip":
            if bundle_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="abstract3d-step1x-bundle-"))
                bundle_paths = _write_bundle(
                    root_dir=temp_root,
                    primary_format="glb",
                    primary_bytes=glb_bytes,
                    obj_bytes=obj_bytes,
                    source_image=source_preview,
                    prompt=prompt,
                    metadata=runtime_meta,
                    view_images=views,
                )
                bundle_root = temp_root
            primary_bytes = _zip_bundle(bundle_root)

        runtime_meta["output_bytes"] = len(primary_bytes)
        metadata_path = bundle_paths.get("metadata_path")
        if isinstance(metadata_path, Path):
            metadata_path.write_text(json.dumps(runtime_meta, indent=2, sort_keys=True), encoding="utf-8")

        artifact_id = stable_artifact_id(primary_bytes)
        stored = (
            store_bytes(
                artifact_store,
                primary_bytes,
                content_type=content_type,
                run_id=run_id,
                tags=tags,
                artifact_id=artifact_id,
                metadata=runtime_meta,
            )
            if artifact_store is not None
            else primary_bytes
        )

        if is_artifact_ref(stored):
            ref = dict(stored)
            ref["content_type"] = content_type
            ref["format"] = primary_format
            ref["metadata"] = runtime_meta
            return ref

        return {
            "data": bytes(stored),
            "content_type": content_type,
            "mime_type": content_type,
            "format": primary_format,
            "backend_id": self.backend_id,
            "model_id": runtime_meta["model_id"],
            "metadata": runtime_meta,
        }

    def t23d(
        self,
        prompt: str,
        *,
        format: str = "glb",
        artifact_store: Optional[Any] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        return self._run_generation(
            task="text_to_scene3d",
            prompt=str(prompt or ""),
            image=None,
            format=format,
            artifact_store=artifact_store,
            run_id=run_id,
            tags=tags,
            metadata=metadata,
            output_dir=kwargs.pop("output_dir", None),
            remove_background=kwargs.pop("remove_background", None),
            num_inference_steps=kwargs.pop("num_inference_steps", None),
            octree_resolution=kwargs.pop("octree_resolution", kwargs.pop("mc_resolution", None)),
            device=kwargs.pop("device", None),
            dtype=kwargs.pop("dtype", None),
            model=kwargs.pop("model", None),
            model_subfolder=kwargs.pop("model_subfolder", None),
            **kwargs,
        )

    def i23d(
        self,
        image: Any,
        *,
        prompt: Optional[str] = None,
        format: str = "glb",
        artifact_store: Optional[Any] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        return self._run_generation(
            task="image_to_scene3d",
            prompt=str(prompt or ""),
            image=image,
            format=format,
            artifact_store=artifact_store,
            run_id=run_id,
            tags=tags,
            metadata=metadata,
            output_dir=kwargs.pop("output_dir", None),
            remove_background=kwargs.pop("remove_background", None),
            num_inference_steps=kwargs.pop("num_inference_steps", None),
            octree_resolution=kwargs.pop("octree_resolution", kwargs.pop("mc_resolution", None)),
            device=kwargs.pop("device", None),
            dtype=kwargs.pop("dtype", None),
            model=kwargs.pop("model", None),
            model_subfolder=kwargs.pop("model_subfolder", None),
            **kwargs,
        )

    def generate(
        self,
        prompt: str = "",
        *,
        task: Optional[str] = None,
        image: Optional[Any] = None,
        format: str = "glb",
        artifact_store: Optional[Any] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        normalized_task = _TASK_ALIASES.get(str(task or "text_to_scene3d").strip().lower().replace("-", "_"), "text_to_scene3d")
        if normalized_task == "image_to_scene3d":
            return self.i23d(
                image,
                prompt=prompt or None,
                format=format,
                artifact_store=artifact_store,
                run_id=run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
        return self.t23d(
            prompt,
            format=format,
            artifact_store=artifact_store,
            run_id=run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def validate_suite(
        self,
        *,
        prompts: Sequence[str],
        images: Sequence[str],
        image_prompts: Optional[Sequence[str]] = None,
        output_dir: str,
        image_model: Optional[str] = None,
        image_provider: Optional[str] = None,
        mc_resolution: int = 128,
        device: Optional[str] = None,
        remove_background: Optional[bool] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        model_subfolder: Optional[str] = None,
    ) -> Dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, Any]] = []
        sheets = []

        if not prompts and not images:
            raise ValueError("validate_suite requires at least one prompt or image.")

        for index, prompt in enumerate(prompts, start=1):
            case_dir = root / f"{index:02d}_t23d"
            out = self.t23d(
                prompt,
                output_dir=str(case_dir),
                image_model=image_model,
                image_provider=image_provider,
                mc_resolution=mc_resolution,
                device=device,
                model=model,
                model_subfolder=model_subfolder,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            meta = dict(out.get("metadata") or {}) if isinstance(out, Mapping) else {}
            meta["case_id"] = case_dir.name
            meta["mode"] = "t23d"
            summary_rows.append(meta)
            sheet_path = meta.get("contact_sheet_path")
            if isinstance(sheet_path, str) and Path(sheet_path).exists():
                sheets.append(importlib.import_module("PIL.Image").open(sheet_path).convert("RGB"))

        base_index = len(summary_rows)
        for offset, image_path in enumerate(images, start=1):
            case_dir = root / f"{base_index + offset:02d}_i23d"
            image_prompt = None
            if image_prompts and len(image_prompts) >= offset:
                image_prompt = image_prompts[offset - 1]
            out = self.i23d(
                image_path,
                prompt=image_prompt,
                output_dir=str(case_dir),
                mc_resolution=mc_resolution,
                device=device,
                remove_background=remove_background,
                model=model,
                model_subfolder=model_subfolder,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            meta = dict(out.get("metadata") or {}) if isinstance(out, Mapping) else {}
            meta["case_id"] = case_dir.name
            meta["mode"] = "i23d"
            summary_rows.append(meta)
            sheet_path = meta.get("contact_sheet_path")
            if isinstance(sheet_path, str) and Path(sheet_path).exists():
                sheets.append(importlib.import_module("PIL.Image").open(sheet_path).convert("RGB"))

        master_sheet = stack_contact_sheets(sheets, columns=2 if len(sheets) > 1 else 1)
        summary_dir = root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        sheet_path = summary_dir / "contact_sheet.png"
        master_sheet.save(sheet_path)
        stats_path = summary_dir / "stats.json"
        stats_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
        return {"contact_sheet": str(sheet_path), "stats": str(stats_path), "rows": summary_rows}


__all__ = ["Step1XGeometryBackend"]
