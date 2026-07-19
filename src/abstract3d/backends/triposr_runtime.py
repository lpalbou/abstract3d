"""Pinned TripoSR runtime integration for Abstract3D."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..artifacts import ArtifactRef, is_artifact_ref, stable_artifact_id, store_bytes
from ..errors import Abstract3DError, CapabilityNotSupportedError, DependencyUnavailableError, SourceBootstrapError
from ..image_composition import COMPOSITION_INSTALL_HINT, default_image_generator, has_image_composer, pop_image_generation_request
from ..model_catalog import capability_model_records
from ..rendering import build_case_contact_sheet, get_last_render_backend, render_mesh_views, stack_contact_sheets
from ..types import GeneratedSceneAsset

_TRIPOSR_REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
_TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"
_DEFAULT_MODEL_ID = "stabilityai/TripoSR"
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "abstract3d"
_DEFAULT_BACKGROUND = (128, 128, 128)
_DEFAULT_TRIPOSR_MC_RESOLUTION = 256
_DEFAULT_TRIPOSR_CLEANUP_MODE = "presentation"
_DEFAULT_TRIPOSR_TEXTURE_MODE = "baked_basecolor"
_DEFAULT_TRIPOSR_TEXTURE_RESOLUTION = 2048
_DEFAULT_TRIPOSR_TEXTURE_COMPLETION = "none"
_TEXTURE_REFERENCE_ANGLES: Dict[str, Tuple[float, float]] = {
    "front": (0.0, 0.0),
    "front_center": (0.0, 0.0),
    "front_left": (45.0, 0.0),
    "three_quarter_left": (45.0, 0.0),
    "quarter_left": (45.0, 0.0),
    "front_right": (-45.0, 0.0),
    "three_quarter_right": (-45.0, 0.0),
    "quarter_right": (-45.0, 0.0),
    "side_left": (90.0, 0.0),
    "left_profile": (90.0, 0.0),
    "profile_left": (90.0, 0.0),
    "side_right": (-90.0, 0.0),
    "right_profile": (-90.0, 0.0),
    "profile_right": (-90.0, 0.0),
    "back_left": (135.0, 0.0),
    "rear_left": (135.0, 0.0),
    "back_right": (-135.0, 0.0),
    "rear_right": (-135.0, 0.0),
    "back": (180.0, 0.0),
    "rear": (180.0, 0.0),
}
_REMBG_INSTALL_HINT = (
    'Background removal requires rembg with an ONNX Runtime backend. '
    'Install with: pip install "abstract3d[triposr]"'
)
_TASK_ALIASES = {
    "scene3d": "text_to_scene3d",
    "scene3d_generation": "text_to_scene3d",
    "text_to_scene3d": "text_to_scene3d",
    "t23d": "text_to_scene3d",
    "image_to_scene3d": "image_to_scene3d",
    "i23d": "image_to_scene3d",
}
# Every **kwargs option `_run_generation` consumes (composition keys ride
# IMAGE_REQUEST_KEYS separately). Used for the millisecond-cheap preflight
# rejection before model load; the post-pop `reject_unknown_options` at the
# end of `_run_generation` stays authoritative — if a new pop is added
# without extending this set, the preflight rejects it loudly (fail-closed),
# which the option-contract tests catch immediately.
_TRIPOSR_GENERATION_OPTION_KEYS = frozenset(
    {
        "cleanup",
        "texture_mode",
        "texture_resolution",
        "texture_completion",
        "texture_reference_views",
        "texture_reference_images",
        "texture_reference_angles",
        "texture_reference_remove_background",
    }
)
_RUNTIME_IMPORTS = (
    "numpy",
    "skimage",
    "torch",
    "omegaconf",
    "einops",
    "huggingface_hub",
    "transformers",
    "PIL",
    "trimesh",
    "imageio",
    "rembg",
    "matplotlib",
    "psutil",
)
_TEXTURE_RUNTIME_IMPORTS = (
    "xatlas",
    "moderngl",
)
_CHECKPOINT_KEY_MAP = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
    ("intermediate.dense.", "mlp.fc1."),
    ("output.dense.", "mlp.fc2."),
    ("layernorm_before.", "layernorm_before."),
    ("layernorm_after.", "layernorm_after."),
)
_RUNTIME_LOCK = threading.Lock()


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(str(key))
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _owner_cfg(owner: Any, key: str) -> Optional[str]:
    try:
        cfg = getattr(owner, "config", None)
        if isinstance(cfg, dict):
            value = cfg.get(key)
            if value is None:
                return None
            text = str(value).strip()
            return text or None
    except Exception:
        return None
    return None


def _owner_cfg_int(owner: Any, key: str, default: int) -> int:
    raw = _owner_cfg(owner, key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _owner_cfg_float(owner: Any, key: str, default: float) -> float:
    raw = _owner_cfg(owner, key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _owner_cfg_bool(owner: Any, key: str, default: bool = False) -> bool:
    raw = _owner_cfg(owner, key)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _cache_root(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_cache_dir") or _env("ABSTRACT3D_CACHE_DIR")
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_CACHE_ROOT


def _tripo_default_mc_resolution(owner: Any, requested: Optional[int] = None) -> int:
    if requested is not None:
        return max(32, int(requested))
    configured = _owner_cfg(owner, "scene3d_triposr_mc_resolution") or _owner_cfg(owner, "scene3d_mc_resolution") or _env("ABSTRACT3D_TRIPOSR_MC_RESOLUTION")
    if configured is None:
        return int(_DEFAULT_TRIPOSR_MC_RESOLUTION)
    try:
        return max(32, int(configured))
    except Exception:
        return int(_DEFAULT_TRIPOSR_MC_RESOLUTION)


def _tripo_cleanup_mode(owner: Any, requested: Optional[Any] = None) -> str:
    raw = requested
    if raw is None:
        raw = _owner_cfg(owner, "scene3d_triposr_cleanup") or _env("ABSTRACT3D_TRIPOSR_CLEANUP") or _DEFAULT_TRIPOSR_CLEANUP_MODE
    if isinstance(raw, bool):
        return "presentation" if raw else "none"
    normalized = str(raw or "").strip().lower()
    if normalized in {"", "auto", "default", "presentation", "clean", "cleanup"}:
        return "presentation"
    if normalized in {"none", "off", "raw", "disabled", "disable"}:
        return "none"
    return _DEFAULT_TRIPOSR_CLEANUP_MODE


def _tripo_cleanup_min_component_faces(face_count: int) -> int:
    return max(32, min(256, int(round(max(0, int(face_count)) * 0.0025))))


def _tripo_cleanup_hole_size(face_count: int) -> int:
    return max(12, min(32, int(round(max(0, int(face_count)) * 0.0009))))


def _tripo_texture_mode(owner: Any, requested: Optional[Any] = None) -> str:
    raw = requested
    if raw is None:
        raw = _owner_cfg(owner, "scene3d_triposr_texture_mode") or _env("ABSTRACT3D_TRIPOSR_TEXTURE_MODE") or _DEFAULT_TRIPOSR_TEXTURE_MODE
    normalized = str(raw or "").strip().lower()
    if normalized in {"", "vertex", "vertex_color", "vertex-colors", "default", "raw", "none"}:
        return "vertex_color"
    if normalized in {"baked", "texture", "textured", "atlas", "uv", "basecolor", "baked_basecolor"}:
        return "baked_basecolor"
    return _DEFAULT_TRIPOSR_TEXTURE_MODE


def _tripo_texture_resolution(owner: Any, requested: Optional[int] = None) -> int:
    raw = requested
    if raw is None:
        raw = _owner_cfg(owner, "scene3d_triposr_texture_resolution") or _env("ABSTRACT3D_TRIPOSR_TEXTURE_RESOLUTION") or _DEFAULT_TRIPOSR_TEXTURE_RESOLUTION
    try:
        value = int(raw)
    except Exception:
        value = int(_DEFAULT_TRIPOSR_TEXTURE_RESOLUTION)
    return max(256, min(8192, value))


def _tripo_texture_reference_remove_background(
    owner: Any,
    requested: Optional[Any] = None,
    *,
    fallback: Optional[bool] = None,
) -> Optional[bool]:
    raw = requested
    if raw is None:
        raw = _owner_cfg(owner, "scene3d_triposr_texture_reference_remove_background") or _env(
            "ABSTRACT3D_TRIPOSR_TEXTURE_REFERENCE_REMOVE_BACKGROUND"
        )
    if raw is None:
        return fallback
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto", "default"}:
        return fallback
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def _tripo_texture_completion_mode(owner: Any, requested: Optional[Any] = None) -> str:
    raw = requested
    if raw is None:
        raw = _owner_cfg(owner, "scene3d_triposr_texture_completion") or _env(
            "ABSTRACT3D_TRIPOSR_TEXTURE_COMPLETION"
        ) or _DEFAULT_TRIPOSR_TEXTURE_COMPLETION
    normalized = str(raw or "").strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "disable", "disabled", "raw", "default"}:
        return "none"
    if normalized in {"mirror_symmetry", "symmetry", "mirror", "mirrored"}:
        return "mirror_symmetry"
    if normalized in {"auto", "automatic"}:
        # Resolved inside the bake: mirror completion iff the mesh's own
        # left-right symmetry score passes the standard gate.
        return "auto"
    return _DEFAULT_TRIPOSR_TEXTURE_COMPLETION


def _tripo_parse_texture_reference_angle(raw: Any) -> Tuple[float, float, str]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1]), f"{float(raw[0]):g},{float(raw[1]):g}"
    if raw is None:
        return 0.0, 0.0, "front"
    text = str(raw).strip().lower().replace("-", "_")
    if text in _TEXTURE_REFERENCE_ANGLES:
        azimuth, elevation = _TEXTURE_REFERENCE_ANGLES[text]
        return float(azimuth), float(elevation), text
    if "," in text:
        left, right = [part.strip() for part in text.split(",", 1)]
        return float(left), float(right), f"{float(left):g},{float(right):g}"
    return float(text), 0.0, f"{float(text):g},0"


def _tripo_append_texture_reference_view(
    out: List[Dict[str, Any]],
    raw: Any,
    *,
    angle_override: Optional[Any] = None,
) -> None:
    if raw is None:
        return
    if isinstance(raw, Mapping):
        image = raw.get("image")
        if image is None and raw.get("path") is not None:
            image = raw.get("path")
        if image is None and raw.get("bytes") is not None:
            image = raw.get("bytes")
        if image is None:
            raise ValueError("texture reference view mappings must include image, path, or bytes.")
        angle_raw = (
            angle_override
            if angle_override is not None
            else raw.get("angle", raw.get("view", raw.get("label", raw.get("azimuth"))))
        )
        azimuth_deg, elevation_deg, label = _tripo_parse_texture_reference_angle(angle_raw)
        if raw.get("elevation") is not None or raw.get("elevation_deg") is not None:
            elevation_deg = float(raw.get("elevation_deg", raw.get("elevation")))
        if raw.get("azimuth") is not None or raw.get("azimuth_deg") is not None:
            azimuth_deg = float(raw.get("azimuth_deg", raw.get("azimuth")))
        out.append(
            {
                "image": image,
                "label": str(raw.get("label") or label),
                "azimuth_deg": float(azimuth_deg),
                "elevation_deg": float(elevation_deg),
            }
        )
        return
    azimuth_deg, elevation_deg, label = _tripo_parse_texture_reference_angle(angle_override)
    out.append(
        {
            "image": raw,
            "label": label,
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
        }
    )


def _tripo_normalize_texture_reference_views(
    *,
    raw_views: Optional[Any] = None,
    raw_images: Optional[Any] = None,
    raw_angles: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if raw_views is not None:
        values = [raw_views] if isinstance(raw_views, (bytes, bytearray, memoryview, str, Path, Mapping)) else list(raw_views)
        for value in values:
            _tripo_append_texture_reference_view(normalized, value)
    if raw_images is None:
        return normalized
    images = [raw_images] if isinstance(raw_images, (bytes, bytearray, memoryview, str, Path, Mapping)) else list(raw_images)
    if raw_angles is None:
        angles = [None] * len(images)
    elif isinstance(raw_angles, (str, bytes, bytearray, memoryview, Path)):
        angles = [raw_angles] * len(images)
    else:
        angles = list(raw_angles)
    if len(angles) not in {0, len(images)}:
        raise ValueError("texture_reference_angles must be empty or match texture_reference_images length.")
    if not angles:
        angles = [None] * len(images)
    for image, angle in zip(images, angles):
        _tripo_append_texture_reference_view(normalized, image, angle_override=angle)
    return normalized


def _require_runtime_dependencies() -> None:
    missing = [name for name in _RUNTIME_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        raise DependencyUnavailableError(
            "Missing Abstract3D runtime dependencies: "
            + ", ".join(sorted(missing))
            + '. Install with: pip install "abstract3d[triposr]"'
        )


def _require_texture_runtime_dependencies() -> None:
    missing = [name for name in _TEXTURE_RUNTIME_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        raise DependencyUnavailableError(
            "Missing TripoSR texture-bake dependencies: "
            + ", ".join(sorted(missing))
            + '. Install with: pip install "abstract3d[triposr]"'
        )


def _ensure_torchmcubes_compat_module() -> None:
    if "torchmcubes" in sys.modules:
        return
    try:
        importlib.import_module("torchmcubes")
        return
    except Exception:
        sys.modules.pop("torchmcubes", None)

    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    measure = importlib.import_module("skimage.measure")
    module = types.ModuleType("torchmcubes")

    def marching_cubes(level: Any, iso_level: float = 0.0):
        volume = level.detach().cpu().numpy() if hasattr(level, "detach") else np.asarray(level)
        volume = np.asarray(volume, dtype=np.float32)
        # Native torchmcubes returns empty tensors for fields with no
        # crossing at the iso level; skimage raises instead. Match the native
        # contract so degenerate scene codes yield an empty mesh (handled
        # downstream) rather than an opaque ValueError.
        if not (float(volume.min()) < float(iso_level) < float(volume.max())):
            return (
                torch.zeros((0, 3), dtype=torch.float32),
                torch.zeros((0, 3), dtype=torch.int64),
            )
        verts, faces, _normals, _values = measure.marching_cubes(
            volume,
            level=float(iso_level),
            allow_degenerate=False,
        )
        # Match the native torchmcubes contract exactly: torchmcubes treats
        # the volume as indexed [z, y, x] and returns (x, y, z) vertices,
        # while skimage returns vertices in raw (dim0, dim1, dim2) index
        # order. The TripoSR isosurface helper then swaps columns again with
        # det=-1, so the fallback must both reorder columns and reverse the
        # face vertex order to end up with the same axes and outward winding
        # as a native torchmcubes build. Getting either wrong produces
        # side-lying meshes or inward normals that silently break the
        # observed-view texture projection.
        verts_xyz = np.ascontiguousarray(np.asarray(verts, dtype=np.float32)[:, [2, 1, 0]])
        faces_flipped = np.ascontiguousarray(np.asarray(faces, dtype=np.int64)[:, ::-1])
        return (
            torch.from_numpy(verts_xyz),
            torch.from_numpy(faces_flipped),
        )

    module.marching_cubes = marching_cubes
    module.__dict__["_abstract3d_fallback"] = True
    sys.modules["torchmcubes"] = module


def _ensure_rembg_compat_module() -> None:
    if "rembg" in sys.modules:
        return
    try:
        importlib.import_module("rembg")
        return
    except KeyboardInterrupt:
        raise
    except BaseException:
        sys.modules.pop("rembg", None)

    module = types.ModuleType("rembg")

    def remove(*args: Any, **kwargs: Any):
        raise DependencyUnavailableError(_REMBG_INSTALL_HINT)

    module.remove = remove
    module.__dict__["_abstract3d_fallback"] = True
    sys.modules["rembg"] = module


def _remap_triposr_checkpoint_key(key: str) -> str:
    match = re.match(r"image_tokenizer\.model\.encoder\.layer\.(\d+)\.(.+)", str(key))
    if not match:
        return key
    index, suffix = match.groups()
    for old, new in _CHECKPOINT_KEY_MAP:
        if suffix.startswith(old):
            return f"image_tokenizer.model.layers.{index}." + suffix.replace(old, new, 1)
    return key


def _select_triposr_state_dict(model_keys: set, checkpoint: Mapping) -> Mapping:
    """Pick the checkpoint key naming that fits THIS transformers version.

    The remap translates the legacy ViT naming
    (`encoder.layer.N.attention.attention.query`) onto the newer
    `layers.N.attention.q_proj` layout — but which naming the instantiated
    model wants depends on the installed transformers version, and applying
    the remap UNCONDITIONALLY corrupted checkpoints that already matched the
    model raw (live incident: transformers 5.6.0, 192 clean keys rewritten
    into 192 misses — reported by core 2026-07-19). Reactive selection:
    measure key-set fit for the raw and remapped forms and load whichever
    fits strictly better; a tie keeps raw (never rewrite what already fits).
    """
    raw_keys = set(checkpoint.keys())
    raw_score = len(model_keys - raw_keys) + len(raw_keys - model_keys)
    if raw_score == 0:
        return checkpoint

    remapped = {_remap_triposr_checkpoint_key(key): value for key, value in checkpoint.items()}
    remapped_keys = set(remapped.keys())
    remapped_score = len(model_keys - remapped_keys) + len(remapped_keys - model_keys)
    if remapped_score < raw_score:
        return remapped
    if raw_score > 0 and remapped_score >= raw_score:
        raise Abstract3DError(
            "The TripoSR checkpoint does not fit this transformers version's model layout "
            f"in either naming (raw mismatch={raw_score}, remapped mismatch={remapped_score}). "
            "Check the pinned transformers version against the abstract3d support range."
        )
    return checkpoint


def _select_device(owner: Any, explicit: Optional[str] = None) -> str:
    requested = str(explicit or _owner_cfg(owner, "scene3d_device") or _env("ABSTRACT3D_DEVICE") or "auto").strip().lower()
    torch = importlib.import_module("torch")
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        requested = "auto"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda:0"
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda:0"
    return "cpu"


def _ensure_repo_checked_out(repo_dir: Path) -> None:
    if (repo_dir / "tsr" / "system.py").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="abstract3d-triposr-", dir=str(repo_dir.parent)))
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _TRIPOSR_REPO_URL, str(tmp_root / "repo")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_root / "repo"), "checkout", _TRIPOSR_COMMIT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.move(str(tmp_root / "repo"), str(repo_dir))
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or exc.stdout or "").strip()
        raise SourceBootstrapError(f"Failed to prepare pinned TripoSR source snapshot: {stderr}") from exc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _resolve_source_dir(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_triposr_source_dir") or _env("ABSTRACT3D_TRIPOSR_SOURCE_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not (path / "tsr" / "system.py").exists():
            raise SourceBootstrapError(f"Configured TripoSR source dir is missing tsr/system.py: {path}")
        return path
    repo_dir = _cache_root(owner) / "vendor" / "triposr" / _TRIPOSR_COMMIT
    with _RUNTIME_LOCK:
        _ensure_repo_checked_out(repo_dir)
    return repo_dir


def _import_triposr(owner: Any) -> Tuple[Any, Any]:
    repo_dir = _resolve_source_dir(owner)
    repo_s = str(repo_dir)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    _ensure_rembg_compat_module()
    _ensure_torchmcubes_compat_module()
    from tsr.models.isosurface import MarchingCubeHelper
    from tsr.system import TSR

    if getattr(MarchingCubeHelper, "_abstract3d_patched", False) is not True:
        original_forward = MarchingCubeHelper.forward

        def _patched_forward(self, level):
            level = -level.view(self.resolution, self.resolution, self.resolution)
            try:
                v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
            except Exception:
                v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
            v_pos = v_pos[..., [2, 1, 0]]
            v_pos = v_pos / (self.resolution - 1.0)
            return v_pos.to(level.device), t_pos_idx.to(level.device)

        MarchingCubeHelper.forward = _patched_forward
        MarchingCubeHelper._abstract3d_patched = True
        MarchingCubeHelper._abstract3d_original_forward = original_forward
    return TSR, MarchingCubeHelper


def _load_triposr_model(owner: Any, *, model_id: str, device: str, chunk_size: int) -> Any:
    torch = importlib.import_module("torch")
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
    TSR, _ = _import_triposr(owner)

    config_path = hf_hub_download(repo_id=model_id, filename="config.yaml")
    weight_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    checkpoint = torch.load(weight_path, map_location="cpu")
    # Reactive normalization: only rewrite checkpoint keys when the rewrite
    # measurably fits this transformers version's model better (see
    # _select_triposr_state_dict — the unconditional remap corrupted clean
    # checkpoints on versions whose layout already matches the file).
    selected = _select_triposr_state_dict(set(model.state_dict().keys()), checkpoint)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise Abstract3DError(
            "Failed to normalize the TripoSR checkpoint for this transformers version. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.renderer.set_chunk_size(int(chunk_size))
    model.to(device)
    model.eval()
    return model


def _load_image_payload(image: Any, *, artifact_store: Optional[Any] = None):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGBA")
    if isinstance(image, str) and image.strip():
        return Image.open(str(image).strip()).convert("RGBA")
    if isinstance(image, Mapping):
        if isinstance(image.get("path"), str) and str(image.get("path")).strip():
            return Image.open(str(image.get("path")).strip()).convert("RGBA")
        if is_artifact_ref(image):
            loader = getattr(artifact_store, "load", None)
            if callable(loader):
                loaded = loader(str(image["$artifact"]))
                content = getattr(loaded, "content", None)
                if isinstance(content, (bytes, bytearray)):
                    return Image.open(io.BytesIO(bytes(content))).convert("RGBA")
                data = getattr(loaded, "data", None)
                if isinstance(data, (bytes, bytearray)):
                    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")
    raise ValueError("Image input must be a file path, raw bytes, PIL image, or an artifact ref resolvable by the artifact store.")


def _pad_square(image):
    from PIL import Image

    image = image.convert("RGBA")
    size = max(image.width, image.height)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2), image)
    return canvas


def _prepare_triposr_image(image: Any, *, remove_background: Optional[bool], foreground_ratio: float, artifact_store: Optional[Any] = None):
    import numpy as np

    loaded = _load_image_payload(image, artifact_store=artifact_store)
    if loaded.mode != "RGBA":
        loaded = loaded.convert("RGBA")

    processed_rgba = loaded
    background_removed = False
    if remove_background is not False:
        alpha = loaded.getchannel("A")
        has_alpha = alpha.getextrema()[0] < 255
        if has_alpha:
            background_removed = True
        else:
            try:
                remove_background_fn = importlib.import_module("tsr.utils").remove_background
                resize_foreground_fn = importlib.import_module("tsr.utils").resize_foreground
            except Exception:
                remove_background_fn = None
                resize_foreground_fn = None
            if remove_background_fn is None or resize_foreground_fn is None:
                if remove_background is True:
                    raise DependencyUnavailableError(
                        "Background removal requested, but TripoSR preprocessing helpers are unavailable."
                    )
            else:
                try:
                    processed_rgba = remove_background_fn(loaded)
                    processed_rgba = resize_foreground_fn(processed_rgba, foreground_ratio)
                    background_removed = True
                except Exception as exc:
                    if remove_background is True:
                        raise DependencyUnavailableError(_REMBG_INSTALL_HINT) from exc
                    processed_rgba = loaded
                    background_removed = False

    if processed_rgba.mode != "RGBA":
        processed_rgba = processed_rgba.convert("RGBA")
    if not background_removed:
        processed_rgba = _pad_square(processed_rgba)

    composed = processed_rgba.convert("RGBA")
    canvas = importlib.import_module("PIL.Image").new("RGB", composed.size, _DEFAULT_BACKGROUND)
    canvas.paste(composed.convert("RGB"), mask=composed.getchannel("A"))
    return loaded.convert("RGB"), canvas, background_removed, processed_rgba


def _prepare_texture_reference_views(
    *,
    source_observed_rgba: Optional[Any],
    source_preview: Optional[Any],
    texture_reference_views: Sequence[Mapping[str, Any]],
    texture_reference_remove_background: Optional[bool],
    foreground_ratio: float,
    artifact_store: Optional[Any],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    if source_observed_rgba is not None:
        prepared.append(
            {
                "label": "front",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "rgba": source_observed_rgba,
                "source_preview": source_preview if source_preview is not None else source_observed_rgba.convert("RGB"),
                "background_removed": None,
                "role": "source",
            }
        )
    for index, view in enumerate(texture_reference_views, start=1):
        preview, _prepared_rgb, background_removed, processed_rgba = _prepare_triposr_image(
            view["image"],
            remove_background=texture_reference_remove_background,
            foreground_ratio=float(foreground_ratio),
            artifact_store=artifact_store,
        )
        prepared.append(
            {
                "label": str(view.get("label") or f"reference_{index:02d}"),
                "azimuth_deg": float(view.get("azimuth_deg", 0.0)),
                "elevation_deg": float(view.get("elevation_deg", 0.0)),
                "rgba": processed_rgba,
                "source_preview": preview,
                "background_removed": bool(background_removed),
                "role": "reference",
            }
        )
    return prepared


def _tripo_mesh_topology(mesh: Any) -> Dict[str, Any]:
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


def _tripo_mesh_to_pymeshlab(mesh: Any) -> Any:
    import numpy as np
    import pymeshlab

    colors = np.empty((0, 4), dtype=np.float64)
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if isinstance(vertex_colors, np.ndarray) and len(vertex_colors) == len(mesh.vertices):
        colors = vertex_colors.astype(np.float64)
        if colors.size and colors.max() > 1.0:
            colors = colors / 255.0
    return pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        v_color_matrix=colors,
    )


def _tripo_mesh_from_pymeshlab(mesh: Any) -> Any:
    import numpy as np
    import trimesh

    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
    processed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.has_vertex_color():
        colors = np.clip(np.asarray(mesh.vertex_color_matrix(), dtype=np.float64), 0.0, 1.0)
        if len(colors) == len(vertices):
            processed.visual.vertex_colors = (colors * 255.0).astype(np.uint8)
    return processed


def _tripo_prune_significant_components(mesh: Any) -> tuple[Any, List[str], List[str]]:
    applied: List[str] = []
    warnings: List[str] = []
    try:
        import trimesh

        processed = mesh.copy() if hasattr(mesh, "copy") else mesh
        if isinstance(processed, trimesh.Scene):
            geometries = [item for item in processed.geometry.values()]
            if geometries:
                processed = trimesh.util.concatenate(geometries)
                applied.append(f"concatenate_scene:{len(geometries)}")
        components = list(processed.split(only_watertight=False))
        if len(components) <= 1:
            return processed, applied, warnings
        components = sorted(components, key=lambda item: (len(item.faces), float(item.area)), reverse=True)
        largest_faces = max(1, int(len(components[0].faces)))
        threshold = max(64, int(round(largest_faces * 0.03)))
        kept = [item for item in components if len(item.faces) >= threshold]
        if not kept:
            kept = [components[0]]
        if len(kept) != len(components):
            applied.append(f"keep_significant_components:{len(components)}->{len(kept)}@{threshold}")
        processed = kept[0].copy() if len(kept) == 1 else trimesh.util.concatenate(kept)
        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()
        return processed, applied, warnings
    except Exception as exc:
        warnings.append(f"TripoSR significant-component pruning skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings


def _tripo_postprocess_mesh(
    mesh: Any,
    *,
    cleanup_mode: str,
) -> tuple[Any, List[str], List[str], Dict[str, Any]]:
    applied: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {
        "mode": cleanup_mode,
        "topology_before": _tripo_mesh_topology(mesh),
    }
    if cleanup_mode == "none":
        details["topology_after"] = dict(details["topology_before"])
        return mesh, applied, warnings, details

    processed = mesh.copy() if hasattr(mesh, "copy") else mesh
    try:
        import pymeshlab
        import trimesh

        if isinstance(processed, trimesh.Scene):
            geometries = [item for item in processed.geometry.values()]
            if geometries:
                processed = trimesh.util.concatenate(geometries)
                applied.append(f"concatenate_scene:{len(geometries)}")
        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()

        face_count = int(len(getattr(processed, "faces", [])))
        min_component_faces = _tripo_cleanup_min_component_faces(face_count)
        hole_size = _tripo_cleanup_hole_size(face_count)
        details["settings"] = {
            "min_component_faces": int(min_component_faces),
            "hole_size": int(hole_size),
            "smooth_steps": 4,
        }

        ms = pymeshlab.MeshSet()
        ms.add_mesh(_tripo_mesh_to_pymeshlab(processed), "triposr")
        if min_component_faces > 0:
            try:
                ms.meshing_remove_connected_component_by_face_number(
                    mincomponentsize=int(min_component_faces),
                    removeunref=True,
                )
                applied.append(f"remove_small_components:{int(min_component_faces)}")
            except Exception as exc:
                warnings.append(f"TripoSR remove_small_components skipped: {type(exc).__name__}: {exc}")
        try:
            faces_before_mc_cleanup = int(ms.current_mesh().face_number())
            ms.meshing_decimation_edge_collapse_for_marching_cube_meshes()
            faces_after_mc_cleanup = int(ms.current_mesh().face_number())
            # The MeshLab marching-cubes decimation occasionally collapses a
            # healthy mesh to a few hundred facet-shaded faces (it merges
            # near-coplanar staircase regions and the outcome is highly
            # sensitive to input noise). Losing >90% of faces is never the
            # intended "cleanup", so rebuild the set without that stage.
            # The 1024 floor only applies to meshes large enough for it to
            # signal collapse; legitimately small meshes are judged by the
            # relative loss alone.
            collapse_floor = max(min(1024, faces_before_mc_cleanup // 2), int(0.1 * faces_before_mc_cleanup))
            if faces_after_mc_cleanup < collapse_floor:
                warnings.append(
                    "TripoSR marching_cube_cleanup over-collapsed the mesh "
                    f"({faces_before_mc_cleanup} -> {faces_after_mc_cleanup} faces); stage reverted."
                )
                ms = pymeshlab.MeshSet()
                ms.add_mesh(_tripo_mesh_to_pymeshlab(processed), "triposr")
                if min_component_faces > 0:
                    try:
                        ms.meshing_remove_connected_component_by_face_number(
                            mincomponentsize=int(min_component_faces),
                            removeunref=True,
                        )
                    except Exception:
                        pass
            else:
                applied.append("marching_cube_cleanup")
        except Exception as exc:
            warnings.append(f"TripoSR marching_cube_cleanup skipped: {type(exc).__name__}: {exc}")
        try:
            ms.apply_coord_taubin_smoothing(
                lambda_=0.5,
                mu=-0.53,
                stepsmoothnum=4,
                selected=False,
            )
            applied.append("taubin_smooth:4")
        except Exception as exc:
            warnings.append(f"TripoSR taubin smoothing skipped: {type(exc).__name__}: {exc}")
        try:
            ms.meshing_repair_non_manifold_edges()
            applied.append("repair_non_manifold_edges")
        except Exception as exc:
            warnings.append(f"TripoSR non-manifold edge repair skipped: {type(exc).__name__}: {exc}")
        try:
            ms.meshing_repair_non_manifold_vertices()
            applied.append("repair_non_manifold_vertices")
        except Exception as exc:
            warnings.append(f"TripoSR non-manifold vertex repair skipped: {type(exc).__name__}: {exc}")
        try:
            ms.meshing_close_holes(
                maxholesize=int(hole_size),
                selected=False,
                newfaceselected=False,
                selfintersection=True,
                refinehole=True,
            )
            applied.append(f"close_holes:{int(hole_size)}")
        except Exception as exc:
            warnings.append(f"TripoSR close_holes skipped: {type(exc).__name__}: {exc}")
        processed = _tripo_mesh_from_pymeshlab(ms.current_mesh())
        processed, component_applied, component_warnings = _tripo_prune_significant_components(processed)
        applied.extend(component_applied)
        warnings.extend(component_warnings)
        try:
            if hasattr(processed, "merge_vertices"):
                processed.merge_vertices()
                applied.append("merge_vertices")
        except Exception as exc:
            warnings.append(f"TripoSR merge_vertices skipped: {type(exc).__name__}: {exc}")
        try:
            trimesh.repair.fix_normals(processed)
            # trimesh.repair.fix_inversion flips faces through Trimesh.invert,
            # which intentionally preserves cached face/vertex normals across
            # its cache clear. When normals were cached before the repair,
            # that preserved cache is stale (inward) even though the winding
            # is now correct, which silently breaks every consumer of
            # vertex_normals (e.g. facing weights in the texture projection).
            # Drop the cache so normals are recomputed from the fixed faces.
            try:
                processed._cache.clear()
            except Exception:
                pass
            _ = processed.face_normals
            _ = processed.vertex_normals
            applied.append("fix_normals")
        except Exception as exc:
            warnings.append(f"TripoSR fix_normals skipped: {type(exc).__name__}: {exc}")
        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()
        details["topology_after"] = _tripo_mesh_topology(processed)
        return processed, applied, warnings, details
    except Exception as exc:
        warnings.append(f"TripoSR postprocess failed and fell back to raw mesh: {type(exc).__name__}: {exc}")
        details["topology_after"] = dict(details["topology_before"])
        return mesh, applied, warnings, details


def _tripo_texture_padding(texture_resolution: int) -> int:
    return round(max(2, int(texture_resolution) / 256))


def _tripo_make_texture_atlas(mesh: Any, *, texture_resolution: int, texture_padding: int) -> Dict[str, Any]:
    import numpy as np
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    options = xatlas.PackOptions()
    options.resolution = int(texture_resolution)
    options.padding = int(texture_padding)
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": np.asarray(vmapping, dtype=np.uint32),
        "indices": np.asarray(indices, dtype=np.uint32),
        "uvs": np.asarray(uvs, dtype=np.float32),
    }


def _tripo_rasterize_vec3_atlas_cpu(
    *,
    atlas_indices: Any,
    atlas_uvs: Any,
    values: Any,
    texture_resolution: int,
    texture_padding: int,
) -> Any:
    """Pure-CPU UV-space rasterizer used when no GL context is available.

    Rasterizes barycentric-interpolated vec3 vertex attributes into the UV
    atlas, then dilates island borders by `texture_padding` texels with a
    nearest-covered-texel fill. This reproduces the coverage semantics of the
    ModernGL path (fill pass plus edge-dilation pass) without requiring a GPU
    or geometry-shader support, which keeps the baked texture path usable on
    headless Linux hosts and Windows machines without OpenGL 3.3 drivers.
    """
    import numpy as np

    resolution = int(texture_resolution)
    uvs_px = np.asarray(atlas_uvs, dtype=np.float64) * float(resolution)
    triangles = np.asarray(atlas_indices, dtype=np.int64).reshape(-1, 3)
    vertex_values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    output = np.zeros((resolution, resolution, 4), dtype=np.float32)
    if len(triangles) == 0:
        return output

    tri_uv = uvs_px[triangles]
    min_xy = np.clip(np.floor(tri_uv.min(axis=1) - 0.5).astype(np.int64), 0, resolution - 1)
    max_xy = np.clip(np.ceil(tri_uv.max(axis=1) - 0.5).astype(np.int64), 0, resolution - 1)
    for face_index in range(len(triangles)):
        x_start, y_start = min_xy[face_index]
        x_end, y_end = max_xy[face_index]
        if x_end < x_start or y_end < y_start:
            continue
        a, b, c = tri_uv[face_index]
        edge0 = b - a
        edge1 = c - a
        denom = edge0[0] * edge1[1] - edge0[1] * edge1[0]
        if abs(float(denom)) < 1e-12:
            continue
        # Pixel centers, matching the GL viewport transform (uv * resolution).
        xs = np.arange(x_start, x_end + 1, dtype=np.float64) + 0.5
        ys = np.arange(y_start, y_end + 1, dtype=np.float64) + 0.5
        px, py = np.meshgrid(xs, ys)
        dx = px - a[0]
        dy = py - a[1]
        w1 = (dx * edge1[1] - dy * edge1[0]) / denom
        w2 = (edge0[0] * dy - edge0[1] * dx) / denom
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        v0, v1, v2 = vertex_values[triangles[face_index]]
        interpolated = (
            w0[:, :, None] * v0[None, None, :]
            + w1[:, :, None] * v1[None, None, :]
            + w2[:, :, None] * v2[None, None, :]
        ).astype(np.float32)
        row_idx, col_idx = np.nonzero(inside)
        output[y_start + row_idx, x_start + col_idx, :3] = interpolated[row_idx, col_idx]
        output[y_start + row_idx, x_start + col_idx, 3] = 1.0

    if int(texture_padding) > 0:
        covered = output[:, :, 3] > 0.0
        if covered.any() and not covered.all():
            try:
                from scipy.ndimage import distance_transform_edt

                distances, nearest = distance_transform_edt(~covered, return_indices=True)
                expand = (~covered) & (distances <= float(texture_padding))
                output[expand, :3] = output[nearest[0][expand], nearest[1][expand], :3]
                output[expand, 3] = 1.0
            except Exception:
                pass
    return output


def _tripo_rasterize_vec3_atlas(
    *,
    atlas_indices: Any,
    atlas_uvs: Any,
    values: Any,
    texture_resolution: int,
    texture_padding: int,
) -> Any:
    """Rasterize vec3 vertex attributes into the UV atlas.

    Prefers the GPU ModernGL path (fast, matches the upstream TripoSR bake),
    and transparently falls back to the CPU rasterizer when no standalone GL
    context or geometry-shader support exists on the host.
    """
    try:
        return _tripo_rasterize_vec3_atlas_moderngl(
            atlas_indices=atlas_indices,
            atlas_uvs=atlas_uvs,
            values=values,
            texture_resolution=texture_resolution,
            texture_padding=texture_padding,
        )
    except Exception:
        return _tripo_rasterize_vec3_atlas_cpu(
            atlas_indices=atlas_indices,
            atlas_uvs=atlas_uvs,
            values=values,
            texture_resolution=texture_resolution,
            texture_padding=texture_padding,
        )


def _tripo_rasterize_vec3_atlas_moderngl(
    *,
    atlas_indices: Any,
    atlas_uvs: Any,
    values: Any,
    texture_resolution: int,
    texture_padding: int,
) -> Any:
    import moderngl
    import numpy as np

    ctx = moderngl.create_context(standalone=True)
    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_value;
            out vec3 v_value;
            void main() {
                v_value = in_value;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_value;
            out vec4 o_col;
            void main() {
                o_col = vec4(v_value, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_value;
            out vec3 vg_value;
            void main() {
                vg_value = in_value;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        geometry_shader="""
            #version 330
            uniform float u_resolution;
            uniform float u_dilation;
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 12) out;
            in vec3 vg_value[];
            out vec3 vf_value;
            void lineSegment(int aidx, int bidx) {
                vec2 a = gl_in[aidx].gl_Position.xy;
                vec2 b = gl_in[bidx].gl_Position.xy;
                vec3 aCol = vg_value[aidx];
                vec3 bCol = vg_value[bidx];

                vec2 dir = normalize((b - a) * u_resolution);
                vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                gl_Position = vec4(a + offset, 0.0, 1.0);
                vf_value = aCol;
                EmitVertex();
                gl_Position = vec4(a - offset, 0.0, 1.0);
                vf_value = aCol;
                EmitVertex();
                gl_Position = vec4(b + offset, 0.0, 1.0);
                vf_value = bCol;
                EmitVertex();
                gl_Position = vec4(b - offset, 0.0, 1.0);
                vf_value = bCol;
                EmitVertex();
            }
            void main() {
                lineSegment(0, 1);
                lineSegment(1, 2);
                lineSegment(2, 0);
                EndPrimitive();
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 vf_value;
            out vec4 o_col;
            void main() {
                o_col = vec4(vf_value, 1.0);
            }
        """,
    )

    uvs = np.asarray(atlas_uvs, dtype=np.float32).reshape(-1).astype("f4")
    values_buffer = np.asarray(values, dtype=np.float32).reshape(-1).astype("f4")
    indices = np.asarray(atlas_indices, dtype=np.uint32).reshape(-1).astype("i4")
    vbo_uvs = ctx.buffer(uvs.tobytes())
    vbo_values = ctx.buffer(values_buffer.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_values.bind("in_value", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    fbo = ctx.framebuffer(
        color_attachments=[
            ctx.texture((int(texture_resolution), int(texture_resolution)), 4, dtype="f4")
        ]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gs_prog["u_resolution"].value = float(texture_resolution)
    gs_prog["u_dilation"].value = float(texture_padding)
    gs_vao.render()
    basic_vao.render()
    fbo_bytes = fbo.color_attachments[0].read()
    positions_texture = np.frombuffer(fbo_bytes, dtype="f4").reshape(
        int(texture_resolution), int(texture_resolution), 4
    )
    return positions_texture


def _tripo_rasterize_position_atlas(
    mesh: Any,
    *,
    atlas_vmapping: Any,
    atlas_indices: Any,
    atlas_uvs: Any,
    texture_resolution: int,
    texture_padding: int,
) -> Any:
    import numpy as np

    values = np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(atlas_vmapping, dtype=np.int64)]
    return _tripo_rasterize_vec3_atlas(
        atlas_indices=atlas_indices,
        atlas_uvs=atlas_uvs,
        values=values,
        texture_resolution=texture_resolution,
        texture_padding=texture_padding,
    )


def _tripo_rasterize_normal_atlas(
    mesh: Any,
    *,
    atlas_vmapping: Any,
    atlas_indices: Any,
    atlas_uvs: Any,
    texture_resolution: int,
    texture_padding: int,
) -> Any:
    import numpy as np

    normals = np.asarray(getattr(mesh, "vertex_normals", None), dtype=np.float32)
    if normals.shape != np.asarray(mesh.vertices, dtype=np.float32).shape:
        mesh = mesh.copy() if hasattr(mesh, "copy") else mesh
        _ = mesh.vertex_normals
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    values = normals[np.asarray(atlas_vmapping, dtype=np.int64)]
    return _tripo_rasterize_vec3_atlas(
        atlas_indices=atlas_indices,
        atlas_uvs=atlas_uvs,
        values=values,
        texture_resolution=texture_resolution,
        texture_padding=texture_padding,
    )


def _tripo_positions_to_colors(
    model: Any,
    scene_code: Any,
    positions_texture: Any,
    *,
    texture_resolution: int,
) -> Any:
    import numpy as np
    import torch

    scene_device = getattr(scene_code, "device", None)
    scene_dtype = getattr(scene_code, "dtype", torch.float32)
    positions = torch.tensor(
        np.asarray(positions_texture, dtype=np.float32).reshape(-1, 4)[:, :-1],
        dtype=scene_dtype if isinstance(scene_dtype, torch.dtype) else torch.float32,
        device=scene_device,
    )
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(
            model.decoder,
            positions,
            scene_code,
        )
    rgb_f = queried_grid["color"].detach().cpu().numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, np.asarray(positions_texture, dtype=np.float32).reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(int(texture_resolution), int(texture_resolution), 4)


def _tripo_texture_image(colors_texture: Any):
    import numpy as np
    from PIL import Image

    rgba = np.clip(np.asarray(colors_texture, dtype=np.float32), 0.0, 1.0)
    image = Image.fromarray((rgba * 255.0).astype(np.uint8), mode="RGBA")
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def _tripo_edge_bleed_texture(texture_image: Any):
    import numpy as np
    from PIL import Image

    rgba = np.asarray(texture_image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3] > 0
    rgb = rgba[:, :, :3].copy()
    if not alpha.any():
        return texture_image.convert("RGB")
    try:
        from scipy.ndimage import distance_transform_edt

        _distances, nearest = distance_transform_edt(~alpha, return_indices=True)
        rgb[~alpha] = rgb[nearest[0][~alpha], nearest[1][~alpha]]
    except Exception:
        pass
    return Image.fromarray(rgb, mode="RGB")


def _tripo_camera_position(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
):
    import math
    import numpy as np

    azimuth_rad = math.radians(float(azimuth_deg))
    elevation_rad = math.radians(float(elevation_deg))
    return np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=np.float32,
    ) * float(camera_distance)


def _tripo_look_at_matrix(eye: Any, target: Any, up: Any):
    import numpy as np

    eye_arr = np.asarray(eye, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    up_arr = np.asarray(up, dtype=np.float32)
    forward = target_arr - eye_arr
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    up_arr = up_arr / max(float(np.linalg.norm(up_arr)), 1e-8)
    side = np.cross(forward, up_arr)
    side = side / max(float(np.linalg.norm(side)), 1e-8)
    up_arr = np.cross(side, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = side
    view[1, :3] = up_arr
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye_arr
    return view


def _tripo_render_camera_depth_map(
    mesh: Any,
    *,
    width: int,
    height: int,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float,
    projection_model: str = "perspective",
    ortho_half_extent: Optional[float] = None,
) -> Optional[Tuple[Any, float]]:
    import math
    import numpy as np

    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    try:
        import moderngl
    except Exception:
        return None

    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
    )
    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    view = _tripo_look_at_matrix(eye, target, up)
    camera_vertices = vertices @ view[:3, :3].T + view[:3, 3]
    positive_depth = -camera_vertices[:, 2]
    valid_depth = positive_depth[positive_depth > 1e-5]
    if len(valid_depth) == 0:
        return None
    far = max(float(camera_distance) + 2.0, float(valid_depth.max()) + 1.0)
    near = max(0.05, min(0.5, float(valid_depth.min()) * 0.25))
    aspect = max(float(width) / max(float(height), 1.0), 1e-6)
    projection = np.zeros((4, 4), dtype=np.float32)
    if str(projection_model) == "orthographic":
        half_h = float(ortho_half_extent or 1.0)
        half_w = half_h * aspect
        projection[0, 0] = 1.0 / half_w
        projection[1, 1] = 1.0 / half_h
        projection[2, 2] = 2.0 / (near - far)
        projection[2, 3] = (far + near) / (near - far)
        projection[3, 3] = 1.0
    else:
        focal = 1.0 / math.tan(0.5 * math.radians(float(fovy_deg)))
        projection[0, 0] = focal / aspect
        projection[1, 1] = focal
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2.0 * far * near) / (near - far)
        projection[3, 2] = -1.0
    mvp = projection @ view

    # Depth occlusion is an accuracy refinement, not a hard requirement:
    # hosts without a standalone GL context (or with GL failures mid-render)
    # fall back to facing-only visibility instead of failing the whole bake.
    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception:
        return None
    try:
        program = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 u_mvp;
                uniform mat4 u_view;
                uniform float u_far;
                out float v_linear_depth;
                void main() {
                    vec4 view_pos = u_view * vec4(in_pos, 1.0);
                    v_linear_depth = clamp((-view_pos.z) / max(u_far, 1e-6), 0.0, 1.0);
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in float v_linear_depth;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_linear_depth, 0.0, 0.0, 1.0);
                }
            """,
        )
        color = ctx.texture((int(width), int(height)), 4, dtype="f4")
        depth = ctx.depth_renderbuffer((int(width), int(height)))
        framebuffer = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        ibo = ctx.buffer(faces.astype(np.int32).tobytes())
        vao = ctx.vertex_array(program, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)
        framebuffer.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        program["u_mvp"].write(mvp.astype(np.float32).T.tobytes())
        program["u_view"].write(view.astype(np.float32).T.tobytes())
        program["u_far"].value = float(far)
        vao.render()
        raw = color.read(alignment=1)
        depth_rgba = np.frombuffer(raw, dtype=np.float32).reshape(int(height), int(width), 4)
        depth_map = np.flipud(depth_rgba[:, :, 0].copy())
        vao.release()
        ibo.release()
        vbo.release()
        framebuffer.release()
        color.release()
        depth.release()
        program.release()
        return depth_map, float(far)
    except Exception:
        return None
    finally:
        try:
            ctx.release()
        except Exception:
            pass


def _tripo_atlas_gradients(field: Any, surface: Any, jump_cap: Optional[float] = None):
    """Central differences of a per-texel field along atlas columns/rows.

    Pairs straddling UV chart boundaries are excluded two ways: both
    neighbors must be surface texels, and (when `jump_cap` is given, in the
    field's own units) the step magnitude must stay below it — packed charts
    can abut with no gap, and those cross-chart jumps are not geometry.
    Returns (d_col, d_row, ok_col, ok_row).
    """
    import numpy as np

    field_arr = np.asarray(field, dtype=np.float32)
    d_col = np.zeros_like(field_arr)
    d_row = np.zeros_like(field_arr)
    d_col[:, 1:-1] = (field_arr[:, 2:] - field_arr[:, :-2]) * 0.5
    d_row[1:-1, :] = (field_arr[2:, :] - field_arr[:-2, :]) * 0.5
    ok_col = np.zeros(field_arr.shape[:2], dtype=bool)
    ok_row = np.zeros(field_arr.shape[:2], dtype=bool)
    surface_arr = np.asarray(surface, dtype=bool)
    ok_col[:, 1:-1] = surface_arr[:, 2:] & surface_arr[:, :-2]
    ok_row[1:-1, :] = surface_arr[2:, :] & surface_arr[:-2, :]
    if jump_cap is not None:
        magnitude_col = np.linalg.norm(d_col, axis=2) if field_arr.ndim == 3 else np.abs(d_col)
        magnitude_row = np.linalg.norm(d_row, axis=2) if field_arr.ndim == 3 else np.abs(d_row)
        ok_col &= magnitude_col < float(jump_cap)
        ok_row &= magnitude_row < float(jump_cap)
    if field_arr.ndim == 3:
        d_col[~ok_col] = 0.0
        d_row[~ok_row] = 0.0
    else:
        d_col = np.where(ok_col, d_col, 0.0)
        d_row = np.where(ok_row, d_row, 0.0)
    return d_col, d_row, ok_col, ok_row


def _tripo_projection_geometry_confidence(
    *,
    positions_xyz: Any,
    normals_xyz: Any,
    surface_mask: Any,
    sample_x: Any,
    sample_y: Any,
    facing: Any,
    stretch_power: float,
    concavity_ball_frac: float,
    concavity_threshold: float,
    concavity_facing: float,
    concavity_demote: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Geometric witness-confidence factor: sampling stretch + concavity.

    STRETCH (exact, from the projector's own mapping). The projector maps
    texel (r, c) -> photo sample s(r, c) = P(p(r, c)); the 2x2 Jacobian
    J = [ds/dc, ds/dr] (photo pixels per texel step) follows directly from
    finite differences of the sample maps — no model approximation, valid
    for both camera models. Its smallest singular value sigma_min is the
    worst-direction sampling pitch: sigma_min near zero means a RUN of
    texels along that direction reads the same photo pixel, i.e. the photo
    content is smeared along the surface. That is the eye-socket/rim
    failure the facing term cannot see: a socket wall can face the camera
    acceptably while the composed texel->photo mapping collapses. Define

        nominal = median sigma_min over well-facing texels (facing > 0.7)
        stretch = nominal / max(sigma_min, eps)          [~1 nominal]
        factor  = 1 / (1 + max(stretch - 1, 0)) ** stretch_power

    stretch_power measured on the face proof asset (adversarial harness,
    all else fixed): p=0 (off) 13 failures, p=1 13 failures with
    sub-threshold dark-debris improvements, p=2 8 failures (three
    dark-debris views cleared, the az -70 eye recovered, front identity
    MAE fail cleared) with texture_qa fully green and no measurable cost
    on the single-view assets (ship/owl fill-energy within 0.005 of p=1).

    `nominal` makes the statistic invariant to photo/atlas resolution (at
    matched sampling every texel has stretch ~1 and the factor is exactly
    1, so single-view coverage is not globally re-weighted). It is the
    median of sigma_MIN, not sigma_max: UV charts can be legitimately
    anisotropic (unequal world pitch along atlas axes), and a healthy
    front-on texel on such a chart must measure stretch 1, which only
    holds when it is compared against the healthy population's own
    worst-direction pitch.

    CONCAVITY. Mean curvature from the divergence of the normal field over
    the surface, estimated from the atlases:

        div n ~ (dn/dc . dp/dc)/|dp/dc|^2 + (dn/dr . dp/dr)/|dp/dr|^2
        concavity = -0.5 * div n * (concavity_ball_frac * diagonal)

    (positive = concave with outward normals). Texels that are BOTH
    measurably concave and grazing multiply by `concavity_demote`: concave
    interiors catch stretched/misplaced content exactly where the witness
    is weakest, while well-facing concave surface (an eye seen head-on)
    keeps its claim, so legitimate socket content and shading survive.
    """
    import numpy as np

    surface = np.asarray(surface_mask, dtype=bool)
    factor = np.ones(surface.shape, dtype=np.float32)
    stats: Dict[str, Any] = {"stretch_p99": None, "concave_demoted": 0}

    covered_positions = np.asarray(positions_xyz, dtype=np.float32)[surface]
    if len(covered_positions) == 0:
        return factor, stats
    diagonal = float(
        np.linalg.norm(covered_positions.max(axis=0) - covered_positions.min(axis=0))
    )
    if diagonal <= 0.0:
        return factor, stats
    pair = surface[:, :-1] & surface[:, 1:]
    positions_arr = np.asarray(positions_xyz, dtype=np.float32)
    if pair.any():
        deltas = np.linalg.norm(
            positions_arr[:, 1:][pair] - positions_arr[:, :-1][pair], axis=1
        )
        pitch = float(np.median(deltas)) + 1e-12
    else:
        pitch = 1e-3

    dp_col, dp_row, ok_col, ok_row = _tripo_atlas_gradients(
        positions_arr, surface, jump_cap=4.0 * pitch
    )
    grad_ok = surface & ok_col & ok_row

    # --- stretch from the sample-map Jacobian ------------------------------
    if float(stretch_power) > 0.0:
        sx_col, sx_row, ok_sx_col, ok_sx_row = _tripo_atlas_gradients(sample_x, surface)
        sy_col, sy_row, _, _ = _tripo_atlas_gradients(sample_y, surface)
        # 2x2 Gram entries of J = [[sx_col, sx_row], [sy_col, sy_row]]
        e = sx_col * sx_col + sy_col * sy_col
        f = sx_col * sx_row + sy_col * sy_row
        g = sx_row * sx_row + sy_row * sy_row
        trace = e + g
        det = e * g - f * f
        disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
        sigma_min = np.sqrt(np.maximum(0.5 * (trace - disc), 0.0))
        valid = grad_ok & ok_sx_col & ok_sx_row
        well = valid & (np.asarray(facing) > 0.7)
        basis = well if well.any() else valid
        if basis.any():
            nominal = float(np.median(sigma_min[basis]))
            if nominal > 1e-9:
                stretch = np.where(valid, nominal / np.maximum(sigma_min, 1e-6), 1.0)
                excess = np.maximum(stretch - 1.0, 0.0)
                factor *= (1.0 / (1.0 + excess) ** float(stretch_power)).astype(np.float32)
                stats["stretch_p99"] = round(float(np.percentile(stretch[valid], 99)), 3)
                # exact per-texel stretch for the caller (witness-scarcity
                # rescue bounds by it); +inf where the Jacobian is
                # unmeasurable (chart edges) so consumers stay conservative
                stats["_stretch_map"] = np.where(
                    valid, stretch, np.inf).astype(np.float32)

    # --- concavity demotion -------------------------------------------------
    if float(concavity_demote) < 1.0:
        normals_arr = np.asarray(normals_xyz, dtype=np.float32)
        dn_col, dn_row, _, _ = _tripo_atlas_gradients(normals_arr, surface)
        len_col2 = np.einsum("ijc,ijc->ij", dp_col, dp_col)
        len_row2 = np.einsum("ijc,ijc->ij", dp_row, dp_row)
        ok_c = ok_col & (len_col2 > 1e-18)
        ok_r = ok_row & (len_row2 > 1e-18)
        div_col = np.where(
            ok_c,
            np.einsum("ijc,ijc->ij", dn_col, dp_col) / np.maximum(len_col2, 1e-18),
            0.0,
        )
        div_row = np.where(
            ok_r,
            np.einsum("ijc,ijc->ij", dn_row, dp_row) / np.maximum(len_row2, 1e-18),
            0.0,
        )
        denom = ok_c.astype(np.float32) + ok_r.astype(np.float32)
        divergence = np.where(
            denom > 0, (div_col + div_row) * (2.0 / np.maximum(denom, 1.0)), 0.0
        )
        concavity = (-0.5 * divergence * float(concavity_ball_frac) * diagonal).astype(
            np.float32
        )
        try:
            from scipy.ndimage import uniform_filter

            surface_f = surface.astype(np.float32)
            concavity = np.where(
                surface,
                uniform_filter(concavity * surface_f, size=5)
                / np.maximum(uniform_filter(surface_f, size=5), 1e-6),
                0.0,
            )
        except Exception:
            pass
        demote = (
            surface
            & (concavity > float(concavity_threshold))
            & (np.asarray(facing) < float(concavity_facing))
        )
        if demote.any():
            factor = np.where(demote, factor * float(concavity_demote), factor).astype(
                np.float32
            )
            stats["concave_demoted"] = int(demote.sum())
    return factor, stats


def _tripo_footprint_filtered_colors(
    *,
    image_rgba: Any,
    sample_x: Any,
    sample_y: Any,
    surface_mask: Any,
    bilinear_sampled: Any,
    engage_sigma: float = 1.0,
    max_taps: int = 5,
    min_alpha_coverage: float = 0.3,
) -> Any:
    """Footprint-aware (anisotropic area-average) color resampling for
    MINIFIED texels; magnified texels keep their exact bilinear values.

    The projector maps texel (r, c) -> photo pixel s(r, c). Where a texel's
    footprint in the photo spans MORE than one pixel (singular values of the
    Jacobian J = [ds/dc, ds/dr] above 1 — grazing incidence, or an atlas
    coarser than the photo), the bilinear gather POINT-samples one location
    inside a multi-pixel footprint: content above the texel Nyquist rate
    aliases into stable false blocks (a checkerboard reference at grazing
    lands as random constant tiles — see the regression test). The fix is
    the textbook one (mip + anisotropic probes, Feline/GPU-aniso practice):

    - Jacobian per texel from atlas-space central differences of the sample
      maps (`_tripo_atlas_gradients` — the same construction the stretch
      confidence uses, so chart edges with no measurable Jacobian keep
      their bilinear estimate).
    - The isotropic level is set by sigma_MIN (the WELL-sampled direction):
      a Gaussian-prefiltered factor-2 pyramid is sampled at the two
      bracketing levels and linearly blended, so the sharp direction is
      never blurred beyond its own sampling pitch.
    - The residual anisotropy (sigma_max / sigma_min, capped at `max_taps`)
      is integrated with Gaussian-weighted probes spaced along the MAJOR
      footprint axis in photo space.
    - Matte-edge safety: the pyramid is built on ALPHA-PREMULTIPLIED
      channels and the probe average is renormalized by its own alpha
      mass; a footprint whose alpha coverage falls under
      `min_alpha_coverage` (mostly background) keeps the bilinear estimate
      — averaging background into rim colors would repaint rims with
      matte fringe. The OUTPUT alpha is untouched by construction (the
      caller keeps bilinear alpha for validity/visibility, exactly like
      the cubic lane).

    Texels with sigma_max <= `engage_sigma` are returned BIT-IDENTICAL to
    `bilinear_sampled`: magnification needs no area average, and the mode
    stays a strict superset of the bilinear contract (refs-off paths that
    never pass the flag are structurally untouched).
    """
    import numpy as np

    sampled = np.asarray(bilinear_sampled, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
    except Exception:
        return sampled

    surface = np.asarray(surface_mask, dtype=bool)
    height, width = np.asarray(image_rgba).shape[:2]

    # --- per-texel Jacobian (photo px per texel step) ----------------------
    sx_col, sx_row, ok_col, ok_row = _tripo_atlas_gradients(sample_x, surface)
    sy_col, sy_row, _, _ = _tripo_atlas_gradients(sample_y, surface)
    e = sx_col * sx_col + sy_col * sy_col
    f = sx_col * sx_row + sy_col * sy_row
    g = sx_row * sx_row + sy_row * sy_row
    trace = e + g
    det = np.maximum(e * g - f * f, 0.0)
    disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
    sigma_max = np.sqrt(np.maximum(0.5 * (trace + disc), 0.0))
    sigma_min = np.sqrt(np.maximum(0.5 * (trace - disc), 0.0))

    needs = surface & ok_col & ok_row & (sigma_max > float(engage_sigma))
    if not needs.any():
        return sampled

    rows, cols = np.nonzero(needs)
    px = np.asarray(sample_x, dtype=np.float32)[rows, cols]
    py = np.asarray(sample_y, dtype=np.float32)[rows, cols]
    s_min = sigma_min[rows, cols]
    s_max = sigma_max[rows, cols]

    # Major footprint axis in photo space: the singular direction of J
    # attached to sigma_max. J J^T = [[e2, f2], [f2, g2]] in PHOTO coords:
    e2 = sx_col * sx_col + sx_row * sx_row
    f2 = sx_col * sy_col + sx_row * sy_row
    g2 = sy_col * sy_col + sy_row * sy_row
    ex = e2[rows, cols] - s_max**2
    fx = f2[rows, cols]
    # eigenvector of [[e2-l, f2],[f2, g2-l]] for l = sigma_max^2; when the
    # cross term vanishes the matrix is diagonal and the major axis is
    # whichever photo axis carries the larger diagonal term.
    major_is_x = e2[rows, cols] >= g2[rows, cols]
    axis_x = np.where(np.abs(fx) > 1e-9, -fx,
                      np.where(major_is_x, 1.0, 0.0))
    axis_y = np.where(np.abs(fx) > 1e-9, ex,
                      np.where(major_is_x, 0.0, 1.0))
    norm = np.sqrt(axis_x**2 + axis_y**2) + 1e-12
    axis_x, axis_y = axis_x / norm, axis_y / norm

    # --- premultiplied Gaussian pyramid ------------------------------------
    image = np.asarray(image_rgba, dtype=np.float32)
    premultiplied = np.concatenate(
        [image[:, :, :3] * image[:, :, 3:4], image[:, :, 3:4]], axis=2)
    max_level = int(np.clip(np.ceil(np.log2(max(float(s_min.max()), 1.0))),
                            0, 8))
    pyramid = [premultiplied]
    for _level in range(max_level):
        base = pyramid[-1]
        blurred = np.stack(
            [gaussian_filter(base[:, :, channel], 1.0, mode="nearest")
             for channel in range(4)], axis=2)
        pyramid.append(blurred[::2, ::2])

    # level selected by the WELL-sampled direction
    lam = np.log2(np.maximum(s_min, 1.0))
    level0 = np.clip(np.floor(lam).astype(np.int32), 0, max_level)
    level1 = np.minimum(level0 + 1, max_level)
    level_blend = np.clip(lam - level0, 0.0, 1.0).astype(np.float32)

    # --- anisotropic probes along the major axis ---------------------------
    # Texel-Nyquist prefilter along the footprint: probes spread over the
    # FULL footprint length (+/- sigma_max around the center) with Gaussian
    # weights whose effective std is half the footprint — the frequency
    # response then suppresses content above the texel sampling rate
    # (measured on the checkerboard regression: a 2x-Nyquist checker lands
    # within 0.06 of its mean) while content the texel grid CAN represent
    # passes through. A boxcar over exactly one footprint was measured
    # first and rejected: its sinc response leaks ~50% of the first
    # aliased octave, which still renders as blocks.
    taps = int(max(3, max_taps))
    tap_units = np.linspace(-1.0, 1.0, taps, dtype=np.float32)
    tap_weights = np.exp(-2.0 * tap_units**2).astype(np.float32)
    tap_weights /= tap_weights.sum()

    accumulated = np.zeros((len(rows), 4), dtype=np.float32)
    for level_sel, level_weight in ((level0, 1.0 - level_blend),
                                    (level1, level_blend)):
        if not float(np.abs(level_weight).max()) > 0.0:
            continue
        for level in np.unique(level_sel):
            member = level_sel == level
            weight_member = level_weight[member]
            if not member.any() or float(np.abs(weight_member).max()) <= 0.0:
                continue
            scale = float(2 ** int(level))
            plane = pyramid[int(level)]
            span = s_max[member]  # full +/- footprint reach, level-0 px
            tap_accum = np.zeros((int(member.sum()), 4), dtype=np.float32)
            for tap_unit, tap_weight in zip(tap_units, tap_weights):
                tap_x = px[member] + axis_x[member] * span * tap_unit
                tap_y = py[member] + axis_y[member] * span * tap_unit
                coords = np.stack([
                    (tap_y + 0.5) / scale - 0.5,
                    (tap_x + 0.5) / scale - 0.5,
                ], axis=0)
                for channel in range(4):
                    tap_accum[:, channel] += tap_weight * map_coordinates(
                        plane[:, :, channel], coords, order=1,
                        mode="nearest")
            accumulated[member] += weight_member[:, None] * tap_accum

    alpha_mass = accumulated[:, 3]
    safe = alpha_mass > float(min_alpha_coverage)
    filtered_rgb = np.where(
        safe[:, None],
        accumulated[:, :3] / np.maximum(alpha_mass, 1e-6)[:, None],
        sampled[rows, cols, :3],
    )
    out = sampled.copy()
    out[rows, cols, :3] = np.clip(filtered_rgb, 0.0, 1.0)
    return out


def _tripo_project_observed_texture(
    observed_rgba: Any,
    *,
    mesh: Optional[Any] = None,
    positions_texture: Any,
    normals_texture: Any,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    camera_distance: float = 1.9,
    fovy_deg: float = 40.0,
    facing_threshold: float = 0.2,
    depth_tolerance: float = 0.02,
    projection_model: str = "perspective",
    ortho_half_extent: Optional[float] = None,
    layered_zone_gate: bool = True,
    layered_zone_near_k: float = 3.0,
    layered_zone_gap_ratio: float = 0.03,
    layered_zone_window_frac: float = 0.03,
    layered_zone_density: float = 0.10,
    layered_zone_min_contrast: float = 0.055,
    stretch_power: float = 2.0,
    concavity_ball_frac: float = 0.02,
    concavity_threshold: float = 0.35,
    concavity_facing: float = 0.5,
    concavity_demote: float = 0.25,
    scarcity_facing_floor: float = 0.05,
    scarcity_stretch_max: float = 4.0,
    sample_filter: str = "bilinear",
) -> Dict[str, Any]:
    import math
    import numpy as np

    image_rgba = np.asarray(observed_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    height, width = image_rgba.shape[:2]

    positions = np.asarray(positions_texture, dtype=np.float32)
    normals = np.asarray(normals_texture, dtype=np.float32)
    mask = positions[:, :, 3] > 0.0
    if not mask.any():
        return {
            "rgba": np.zeros_like(positions, dtype=np.float32),
            "weight": np.zeros(positions.shape[:2], dtype=np.float32),
            "coverage_ratio": 0.0,
        }

    normals_xyz = normals[:, :, :3]
    normal_norm = np.linalg.norm(normals_xyz, axis=2, keepdims=True)
    normals_xyz = np.divide(normals_xyz, np.maximum(normal_norm, 1e-8))
    positions_xyz = positions[:, :, :3]

    camera_position = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
    )
    to_camera = camera_position[None, None, :] - positions_xyz
    to_camera_norm = np.linalg.norm(to_camera, axis=2, keepdims=True)
    view_dir = np.divide(to_camera, np.maximum(to_camera_norm, 1e-8))
    facing = np.sum(normals_xyz * view_dir, axis=2)

    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    view = _tripo_look_at_matrix(camera_position, target, up)
    positions_h = np.concatenate(
        [positions_xyz, np.ones((*positions_xyz.shape[:2], 1), dtype=np.float32)],
        axis=2,
    )
    camera_positions = positions_h @ view.T
    x_cam = camera_positions[:, :, 0]
    y_cam = camera_positions[:, :, 1]
    z_cam = camera_positions[:, :, 2]

    # The -0.5 converts window coordinates to pixel-center array indices;
    # color and depth lookups below share these coordinates, so the
    # correction keeps them mutually consistent and sub-pixel accurate.
    if str(projection_model) == "orthographic":
        # Parallel rays: photos of canonically reconstructed objects
        # (Hunyuan trains with orthographic cameras) and long-lens studio
        # portraits have negligible perspective; the pinhole model would
        # magnify near features relative to far ones and misplace them.
        half_extent = float(ortho_half_extent or 1.0)
        ortho_scale = 0.5 * float(height) / max(half_extent, 1e-6)
        sample_x = ortho_scale * x_cam + float(width) / 2.0 - 0.5
        sample_y = -ortho_scale * y_cam + float(height) / 2.0 - 0.5
    else:
        focal = 0.5 * float(height) / math.tan(0.5 * math.radians(float(fovy_deg)))
        sample_x = focal * x_cam / np.maximum(-z_cam, 1e-6) + float(width) / 2.0 - 0.5
        sample_y = -focal * y_cam / np.maximum(-z_cam, 1e-6) + float(height) / 2.0 - 0.5

    valid = (
        mask
        & (z_cam < -1e-4)
        & (sample_x >= 0.0)
        & (sample_x <= float(width - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(height - 1))
        & (facing > float(facing_threshold))
    )

    # Clip the gather indices themselves: texels outside the frustum are
    # already excluded by `valid`, but unclipped floor indices from those
    # texels (e.g. -1 or width) would wrap or crash the fancy-indexed gather
    # below before the mask is ever applied.
    x0 = np.clip(np.floor(sample_x), 0, width - 1).astype(np.int32)
    y0 = np.clip(np.floor(sample_y), 0, height - 1).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = np.clip(sample_x - x0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(sample_y - y0, 0.0, 1.0).astype(np.float32)

    c00 = image_rgba[y0, x0]
    c10 = image_rgba[y0, x1]
    c01 = image_rgba[y1, x0]
    c11 = image_rgba[y1, x1]
    sampled = (
        c00 * (1.0 - wx)[:, :, None] * (1.0 - wy)[:, :, None]
        + c10 * wx[:, :, None] * (1.0 - wy)[:, :, None]
        + c01 * (1.0 - wx)[:, :, None] * wy[:, :, None]
        + c11 * wx[:, :, None] * wy[:, :, None]
    )
    if str(sample_filter) == "cubic":
        # Interpolating-cubic COLOR sampling (B-spline with prefilter):
        # the bilinear gather is a low-pass at fractional sample offsets —
        # measured 6% of the 2-8 px relief band lost on a 768 px reference
        # projected to a 1024 atlas. Alpha and the validity/depth logic
        # keep the bilinear estimate: cubic lobes overshoot at the matte
        # edge, and visibility must stay conservative.
        #
        # NOT WIRED BY DEFAULT (measured, then parked): together with a
        # bicubic registration warp, the recovered edge sharpness raised
        # the whole-bake acceptance gate's long-strong-edge statistic by
        # the labeled chair-regression magnitude — crisp carved contours
        # and genuine handoff seams are indistinguishable to that render-
        # space metric. Re-enable per-view once the gate consumes the
        # texture-space handoff-seam ledger instead.
        try:
            from scipy.ndimage import map_coordinates, spline_filter

            coords = np.stack([sample_y, sample_x], axis=0)
            cubic = np.empty_like(sampled[:, :, :3])
            for channel in range(3):
                plane = spline_filter(
                    image_rgba[:, :, channel], order=3, mode="nearest")
                cubic[:, :, channel] = map_coordinates(
                    plane, coords, order=3, prefilter=False, mode="nearest")
            sampled = np.concatenate(
                [np.clip(cubic, 0.0, 1.0), sampled[:, :, 3:4]], axis=2)
        except Exception:
            pass
    elif str(sample_filter) == "footprint":
        # Anisotropic area-average for MINIFIED texels (see
        # `_tripo_footprint_filtered_colors`): where a texel's photo
        # footprint exceeds one pixel (grazing incidence; atlas coarser
        # than the photo), the bilinear gather point-samples inside a
        # multi-pixel footprint and content above the texel Nyquist rate
        # aliases into stable false blocks. Magnified texels keep their
        # exact bilinear values, alpha and the validity/depth logic keep
        # the bilinear estimate (visibility stays conservative), and the
        # default filter remains "bilinear" — certified refs-off paths
        # never reach this branch. Measured (A4 resolution program,
        # 2026-07-14): a 2048 atlas samples the 1024 canonical reference
        # frame at sigma ~0.8 (magnified — footprint inert by design),
        # while 1024 fleet bakes reach sigma_max ~1.9 on roof/top
        # regions, exactly the aliasing regime this filter integrates.
        sampled = _tripo_footprint_filtered_colors(
            image_rgba=image_rgba,
            sample_x=sample_x,
            sample_y=sample_y,
            surface_mask=mask,
            bilinear_sampled=sampled,
        )
    alpha = sampled[:, :, 3]

    # Strict first-surface visibility: a photo pixel may only paint the
    # nearest surface it sees. Tolerance-based depth-map tests cannot
    # provide this: generated meshes are frequently thin crusts whose
    # sheets sit a few millimeters apart (hair shells over foreheads,
    # folded films), and any tolerance loose enough to survive depth-map
    # interpolation stamps the same photo pixels onto BOTH sheets. Viewed
    # from anywhere but the source camera those duplicate stamps separate
    # into ghosted features. The z-buffer is built from the projected
    # texels themselves (exact — no cross-renderer depth mismatch), and it
    # includes EVERY projectable surface texel regardless of orientation
    # or photo alpha: a surface occludes what is behind it whether or not
    # it is paintable itself (a grazing-angle cheek must still occlude the
    # hair sheet behind it).
    occluder = mask & (z_cam < -1e-4) & (
        (sample_x >= -0.5) & (sample_x <= float(width) - 0.5)
        & (sample_y >= -0.5) & (sample_y <= float(height) - 0.5)
    )
    visibility = np.ones_like(sample_x, dtype=bool)
    witness_factor = np.ones_like(sample_x, dtype=np.float32)
    contested = np.zeros(sample_x.shape, dtype=bool)
    film_band_maps: Optional[Dict[str, Any]] = None
    if occluder.any():
        depth_world = -z_cam
        bins_x = np.clip(np.round(sample_x).astype(np.int32), 0, width - 1)
        bins_y = np.clip(np.round(sample_y).astype(np.int32), 0, height - 1)
        nearest = np.full((int(height), int(width)), np.inf, dtype=np.float32)
        np.minimum.at(nearest, (bins_y[occluder], bins_x[occluder]), depth_world[occluder])
        # The unfiltered per-bin nearest map is the first-surface estimate
        # the layered-zone statistic below needs; the min-filtered copy
        # that follows serves only the conservative visibility test.
        nearest_raw = nearest.copy()
        # Texel sampling is sparser than the pixel grid at rims: a hidden
        # sheet can own a rim bin simply because the occluding surface's
        # texel centers landed half a pixel away. Taking the 3x3
        # neighborhood minimum makes occlusion conservatively wider by one
        # pixel — the safe direction, since a wrongly rejected rim texel is
        # recovered by fill/bleed while a wrongly accepted hidden texel
        # stamps ghost features.
        try:
            from scipy.ndimage import minimum_filter

            nearest = minimum_filter(nearest, size=3, mode="nearest")
        except Exception:
            pass
        covered_positions_zb = positions_xyz[mask]
        if len(covered_positions_zb) > 0:
            diagonal_zb = float(
                np.linalg.norm(covered_positions_zb.max(axis=0) - covered_positions_zb.min(axis=0))
            )
        else:
            diagonal_zb = 1.0
        # Slope-aware bias (standard shadow-mapping practice): within the
        # 3x3 min-filter's ~2.5 px support a smooth surface's own depth
        # varies by tan(tilt) * pixel-world-size. A scalar epsilon must
        # otherwise choose between leaking hidden sheets and SELF-REJECTING
        # tilted surfaces (measured: up to 40% of genuinely visible texels
        # zeroed at 55-75 degree tilt, all demoted to milky harmonic fill).
        # `facing` is the cosine of the local tilt, so front-on surfaces
        # keep exactly the base epsilon and the two-sheet occlusion the
        # strict z-buffer exists for.
        if str(projection_model) == "orthographic":
            pixel_world = 1.0 / ortho_scale
        else:
            pixel_world = np.maximum(-z_cam, 1e-6) / focal
        slope = np.sqrt(np.clip(1.0 - facing**2, 0.0, 1.0)) / np.maximum(facing, 0.05)
        epsilon = 0.0025 * diagonal_zb + 2.5 * pixel_world * slope
        visibility = depth_world <= nearest[bins_y, bins_x] + epsilon

        # Layered-density zone gate: where the photo images a thin SECOND
        # surface a small step behind the first (hovering film shells:
        # hair wisps over a scalp, lash shelves over a lid), sub-pixel aim
        # — not content — decides which sheet each pixel stamps, and the
        # photo pixels there are themselves mixtures of both materials.
        # Stamping such regions produces salt-and-pepper mottle that
        # mirror/harmonic completion then propagates. Pixel-level gating
        # is insufficient (measured: the un-gated survivors between
        # layered pixels still anchor flakes), so the DENSITY of layered
        # samples over a window decides: where more than
        # `layered_zone_density` of the window's projected samples land a
        # small gap behind the first surface, the whole region is a
        # mixture witness and this view surrenders it to better views or
        # fill. The density is a per-sample fraction (numerator and
        # denominator both scale with texture resolution, so the zone is
        # sampling-invariant); the gap window is capped relative to the
        # mesh diagonal so ordinary front-to-back hulls (gap ~ head
        # depth) never count as layered; the near tolerance absorbs
        # one-surface sampling jitter (legitimate rough geometry stays).
        if layered_zone_gate:
            try:
                from scipy.ndimage import binary_dilation, uniform_filter

                base_epsilon = 0.0025 * diagonal_zb
                near_tol_z = float(layered_zone_near_k) * base_epsilon
                gap_max_z = float(layered_zone_gap_ratio) * diagonal_zb
                first_z = nearest_raw
                occ_y = bins_y[occluder]
                occ_x = bins_x[occluder]
                occ_d = depth_world[occluder]
                beyond_z = occ_d > (first_z[occ_y, occ_x] + near_tol_z)
                second_sample = beyond_z & (occ_d <= first_z[occ_y, occ_x] + gap_max_z)
                num = np.zeros_like(first_z)
                den = np.zeros_like(first_z)
                np.add.at(num, (occ_y[second_sample], occ_x[second_sample]), 1.0)
                np.add.at(den, (occ_y, occ_x), 1.0)
                win = max(3, int(round(float(layered_zone_window_frac) * float(min(height, width)))))
                num_s = uniform_filter(num, size=win, mode="constant")
                den_s = uniform_filter(den, size=win, mode="constant")
                with np.errstate(divide="ignore", invalid="ignore"):
                    density = np.where(den_s > 0, num_s / np.maximum(den_s, 1e-6), 0.0)
                zone_map = density > float(layered_zone_density)
                # Ambiguity only matters when the mixed materials DIFFER.
                # Layered hair-over-hair regions (the whole rear head) are
                # harmless: whichever sheet a pixel stamps, the content is
                # the same material, and surrendering them flattens the
                # rear texture into featureless fill (measured regression).
                # A layered region is surrendered only where the photo's
                # local luminance spread says the mixture has real material
                # contrast (skin vs hair ~0.1+; within-hair ~0.02).
                photo_lum = image_rgba[:, :, :3].mean(axis=2)
                photo_alpha_w = image_rgba[:, :, 3]
                lum_w = uniform_filter(photo_lum * photo_alpha_w, size=win, mode="constant")
                alpha_w = uniform_filter(photo_alpha_w, size=win, mode="constant")
                lum2_w = uniform_filter(photo_lum**2 * photo_alpha_w, size=win, mode="constant")
                local_mean = np.where(alpha_w > 0.1, lum_w / np.maximum(alpha_w, 1e-6), 0.0)
                local_var = np.where(
                    alpha_w > 0.1, lum2_w / np.maximum(alpha_w, 1e-6) - local_mean**2, 0.0
                )
                local_std = np.sqrt(np.clip(local_var, 0.0, None))
                zone_map &= local_std > float(layered_zone_min_contrast)
                zone = zone_map[bins_y, bins_x]
                witness_factor = np.where(zone, 0.0, witness_factor).astype(np.float32)
                # Contested texels: the zone dilated by half a window.
                # They may keep their own (boundary) claims, but callers
                # must not use them as mirror-completion sources — copying
                # mixture content onto hidden surface is how a few bad
                # anchors become flakes everywhere (measured: >90% of
                # detected flake islands were mirror/harmonic copies).
                zone_wide = binary_dilation(zone_map, iterations=max(1, win // 2))
                contested = zone_wide[bins_y, bins_x] & mask & (z_cam < -1e-4)

                # Film-band maps (see film_band.py): the multi-view film
                # commitment downstream extends this zone into FUSED film
                # bands the density statistic cannot see (no second sheet)
                # and needs per-view witness/consensus maps. Purely
                # additive here — weights are not touched.
                try:
                    from ..film_band import compute_view_film_maps

                    first_surface = np.zeros(mask.shape, dtype=bool)
                    occ_rows, occ_cols = np.nonzero(occluder)
                    is_first = occ_d <= first_z[occ_y, occ_x] + near_tol_z
                    first_surface[occ_rows[is_first], occ_cols[is_first]] = True
                    film_band_maps = compute_view_film_maps(
                        image_rgba01=image_rgba,
                        zone_map=zone_map,
                        density=density,
                        local_std=local_std,
                        window=win,
                        min_contrast=float(layered_zone_min_contrast),
                        bins_y=bins_y,
                        bins_x=bins_x,
                        infront=mask & (z_cam < -1e-4),
                        first_surface=first_surface,
                    )
                except Exception:
                    film_band_maps = None
            except Exception:
                pass

    strength = np.clip((facing - float(facing_threshold)) / max(1e-6, 1.0 - float(facing_threshold)), 0.0, 1.0)
    # Geometric confidence: the facing term measures local tilt only; the
    # stretch/concavity factor measures the COMPOSED texel->photo sampling
    # map (collapsed footprints at socket walls and rims) and demotes
    # concave grazing texels where misplaced content concentrates.
    geometry_factor, geometry_stats = _tripo_projection_geometry_confidence(
        positions_xyz=positions_xyz,
        normals_xyz=normals_xyz,
        surface_mask=mask,
        sample_x=sample_x,
        sample_y=sample_y,
        facing=facing,
        stretch_power=float(stretch_power),
        concavity_ball_frac=float(concavity_ball_frac),
        concavity_threshold=float(concavity_threshold),
        concavity_facing=float(concavity_facing),
        concavity_demote=float(concavity_demote),
    )
    weight = np.where(
        valid & visibility,
        alpha * (strength ** 2) * witness_factor * geometry_factor,
        0.0,
    ).astype(np.float32)

    # Reference-leverage accounting (diagnostics only — never feeds any
    # numeric path): what this photo could GEOMETRICALLY witness
    # (in-frame, first-surface unoccluded, facing above the grazing floor,
    # photo content present) versus what the quality gates let it paint.
    # The bake-level ledger aggregates these into the per-view
    # potential/painted/won stats the project owner asked to see.
    in_frame = (
        mask
        & (z_cam < -1e-4)
        & (sample_x >= 0.0)
        & (sample_x <= float(width - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(height - 1))
    )
    visible_unoccluded = in_frame & visibility
    potential = visible_unoccluded & (facing > 0.05) & (alpha > 0.5)
    zone_texel = visible_unoccluded & (witness_factor <= 0.0)

    # WITNESS-SCARCITY RESCUE CANDIDATES (per-texel, never self-admitted):
    # claims between the grazing floor and the role facing threshold,
    # bounded by the EXACT per-texel sampling stretch (the texel->photo
    # Jacobian above — facing is only a tilt proxy; stretch measures the
    # composed mapping that actually decides content quality). The
    # projector never admits these itself: the caller may admit a scarce
    # claim ONLY on texels where no view holds a strict claim ("stretched
    # content beats no content", the single-view doctrine, generalized to
    # per-texel witness scarcity — a real observation of the surface
    # outranks a symmetry guess or harmonic fill wherever it is the only
    # witness). Everything else the strict weight respects is respected
    # here too: first-surface visibility, photo alpha, the layered-zone
    # surrender (witness_factor — mixture regions are never rescued by
    # their own view), and the stretch/concavity demotion.
    stretch_map = geometry_stats.pop("_stretch_map", None)
    scarce_ok = (
        visible_unoccluded
        & (facing > float(scarcity_facing_floor))
        & ~(facing > float(facing_threshold))
    )
    if stretch_map is not None:
        scarce_ok &= stretch_map <= float(scarcity_stretch_max)
    else:
        # no measurable Jacobian (degenerate basis): stay conservative
        scarce_ok &= False
    scarce_strength = np.clip(
        (facing - float(scarcity_facing_floor))
        / max(1e-6, 1.0 - float(scarcity_facing_floor)),
        0.0,
        1.0,
    )
    scarce_weight = np.where(
        scarce_ok,
        alpha * (scarce_strength ** 2) * witness_factor * geometry_factor,
        0.0,
    ).astype(np.float32)

    projected_rgba = np.zeros_like(positions, dtype=np.float32)
    projected_rgba[:, :, :3] = sampled[:, :, :3]
    projected_rgba[:, :, 3] = np.where(weight > 0.0, positions[:, :, 3], 0.0)
    return {
        "rgba": projected_rgba,
        "weight": weight,
        "coverage_ratio": float(np.count_nonzero(weight > 0.0)) / float(np.count_nonzero(mask)),
        "label": f"{float(azimuth_deg):g},{float(elevation_deg):g}",
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "contested": contested,
        "film_band": film_band_maps,
        "geometry_confidence": geometry_stats,
        "potential": potential,
        "visible_unoccluded": visible_unoccluded,
        "zone": zone_texel,
        "facing": facing.astype(np.float16),
        "geometry_factor": geometry_factor.astype(np.float16),
        "photo_alpha": (alpha > 0.5),
        "facing_threshold": float(facing_threshold),
        "scarce_weight": scarce_weight,
    }


def _tripo_blend_observed_texture(colors_texture: Any, observed_projections: Sequence[Mapping[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np

    base = np.asarray(colors_texture, dtype=np.float32).copy()
    surface_mask = base[:, :, 3] > 0.0
    if not observed_projections:
        return base, {"coverage_ratio": 0.0, "view_stats": []}
    accum_rgb = np.zeros_like(base[:, :, :3], dtype=np.float32)
    accum_weight = np.zeros(base.shape[:2], dtype=np.float32)
    accum_alpha = np.zeros(base.shape[:2], dtype=np.float32)
    union_mask = np.zeros(base.shape[:2], dtype=bool)
    view_stats: List[Dict[str, Any]] = []
    for index, projection in enumerate(observed_projections, start=1):
        overlay = np.asarray(projection.get("rgba"), dtype=np.float32)
        weight = np.asarray(projection.get("weight"), dtype=np.float32)
        if base.shape != overlay.shape or weight.shape != base.shape[:2]:
            continue
        clipped_weight = np.clip(weight, 0.0, 1.0)
        accum_rgb += overlay[:, :, :3] * clipped_weight[:, :, None]
        accum_weight += clipped_weight
        accum_alpha = np.maximum(accum_alpha, overlay[:, :, 3])
        union_mask |= clipped_weight > 0.0
        view_stats.append(
            {
                "index": index,
                "label": str(projection.get("label") or f"reference_{index:02d}"),
                "azimuth_deg": float(projection.get("azimuth_deg", 0.0)),
                "elevation_deg": float(projection.get("elevation_deg", 0.0)),
                "coverage_ratio": round(float(projection.get("coverage_ratio") or 0.0), 4),
            }
        )
    observed = accum_weight > 1e-6
    if observed.any():
        blended_rgb = np.divide(
            accum_rgb,
            np.maximum(accum_weight[:, :, None], 1e-6),
        )
        final_weight = np.clip(accum_weight, 0.0, 1.0)
        base[:, :, :3] = np.where(
            observed[:, :, None],
            blended_rgb * final_weight[:, :, None] + base[:, :, :3] * (1.0 - final_weight[:, :, None]),
            base[:, :, :3],
        )
        base[:, :, 3] = np.maximum(base[:, :, 3], accum_alpha)
    denominator = max(int(np.count_nonzero(surface_mask)), 1)
    return base, {
        "coverage_ratio": round(float(np.count_nonzero(union_mask & surface_mask)) / float(denominator), 4),
        "view_stats": view_stats,
        "weight_map": np.clip(accum_weight, 0.0, 1.0),
    }


def _tripo_project_mirror_symmetry_texture(
    observed_rgba: Any,
    *,
    mesh: Optional[Any] = None,
    positions_texture: Any,
    normals_texture: Any,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    camera_distance: float = 1.9,
    fovy_deg: float = 40.0,
    facing_threshold: float = 0.2,
) -> Dict[str, Any]:
    import numpy as np

    positions = np.asarray(positions_texture, dtype=np.float32)
    normals = np.asarray(normals_texture, dtype=np.float32)
    mask = positions[:, :, 3] > 0.0
    if not mask.any():
        return {
            "rgba": np.zeros_like(positions, dtype=np.float32),
            "weight": np.zeros(positions.shape[:2], dtype=np.float32),
            "coverage_ratio": 0.0,
            "label": "mirror_symmetry",
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
        }

    mirrored_positions = positions.copy()
    mirrored_normals = normals.copy()
    mirrored_positions[:, :, 1] *= -1.0
    mirrored_normals[:, :, 1] *= -1.0
    projection = _tripo_project_observed_texture(
        observed_rgba,
        mesh=mesh,
        positions_texture=mirrored_positions,
        normals_texture=mirrored_normals,
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
        fovy_deg=float(fovy_deg),
        facing_threshold=float(facing_threshold),
    )

    positions_xyz = positions[:, :, :3]
    normals_xyz = normals[:, :, :3]
    normal_norm = np.linalg.norm(normals_xyz, axis=2, keepdims=True)
    normals_xyz = np.divide(normals_xyz, np.maximum(normal_norm, 1e-8))

    x_values = positions_xyz[:, :, 0][mask]
    y_values = np.abs(positions_xyz[:, :, 1][mask])
    if len(x_values) == 0 or len(y_values) == 0:
        return projection
    x_max = float(np.max(x_values))
    x_front_cutoff = float(np.quantile(x_values, 0.35))
    y_max = float(np.quantile(y_values, 0.95)) if len(y_values) > 1 else float(y_values[0])
    y_side_cutoff = float(np.quantile(y_values, 0.2)) if len(y_values) > 1 else 0.0

    frontness = np.clip(
        (positions_xyz[:, :, 0] - x_front_cutoff) / max(x_max - x_front_cutoff, 1e-6),
        0.0,
        1.0,
    )
    lateralness = np.clip(
        (np.abs(positions_xyz[:, :, 1]) - y_side_cutoff) / max(y_max - y_side_cutoff, 1e-6),
        0.0,
        1.0,
    )
    frontal_normals = np.clip((normals_xyz[:, :, 0] + 0.05) / 1.05, 0.0, 1.0)
    mirror_support = (frontness * lateralness * frontal_normals).astype(np.float32)
    projection_weight = np.asarray(projection["weight"], dtype=np.float32) * mirror_support
    promoted_fill = np.where(projection_weight > 0.0, 0.85 * mirror_support, 0.0).astype(np.float32)
    projection["weight"] = np.maximum(projection_weight, promoted_fill)
    projection["rgba"][:, :, 3] = np.where(projection["weight"] > 0.0, positions[:, :, 3], 0.0)
    denominator = max(int(np.count_nonzero(mask)), 1)
    projection["coverage_ratio"] = round(float(np.count_nonzero(projection["weight"] > 0.0)) / float(denominator), 4)
    projection["label"] = "mirror_symmetry"
    projection["azimuth_deg"] = float(azimuth_deg)
    projection["elevation_deg"] = float(elevation_deg)
    return projection


def _tripo_uv_preview(*, texture_image: Any, uvs: Any, indices: Any):
    import numpy as np
    from PIL import ImageDraw

    preview = texture_image.convert("RGBA").copy()
    draw = ImageDraw.Draw(preview, "RGBA")
    uv_values = np.asarray(uvs, dtype=np.float32)
    face_values = np.asarray(indices, dtype=np.int64)
    if len(face_values) == 0:
        return preview
    width, height = preview.size
    step = max(1, len(face_values) // 20000)
    line_width = max(1, width // 1024)
    for face in face_values[::step]:
        tri = []
        for index in face:
            u, v = uv_values[int(index)]
            tri.append((float(u) * (width - 1), float(1.0 - v) * (height - 1)))
        draw.line([tri[0], tri[1], tri[2], tri[0]], fill=(255, 255, 255, 180), width=line_width)
    return preview


def _tripo_build_textured_mesh(mesh: Any, *, bake_output: Mapping[str, Any], texture_image: Any) -> Any:
    import numpy as np
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    vmapping = np.asarray(bake_output["vmapping"], dtype=np.int64)
    indices = np.asarray(bake_output["indices"], dtype=np.int64)
    uvs = np.asarray(bake_output["uvs"], dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)[vmapping]
    kwargs: Dict[str, Any] = {
        "vertices": vertices,
        "faces": indices,
        "process": False,
        "visual": TextureVisuals(
            uv=uvs,
            # The baked atlas IS the authored surface color, so every material
            # factor a spec viewer multiplies on top of it must be identity.
            # trimesh's SimpleMaterial defaults to a 0.4 gray diffuse and its
            # GLB conversion omits metallicFactor (glTF then defaults to 1.0,
            # fully metallic): spec-compliant viewers rendered exports ~60%
            # darker and mirror-dark under image-based lighting. Explicit PBR
            # factors keep the texture bytes authoritative: white base color,
            # non-metallic, fully rough (pure diffuse response).
            material=PBRMaterial(
                baseColorTexture=texture_image,
                baseColorFactor=(255, 255, 255, 255),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
        ),
    }
    vertex_normals = getattr(mesh, "vertex_normals", None)
    if vertex_normals is not None and len(vertex_normals) == len(mesh.vertices):
        kwargs["vertex_normals"] = np.asarray(vertex_normals, dtype=np.float32)[vmapping]
    return trimesh.Trimesh(**kwargs)


def _tripo_obj_material_from_pbr(material: Any) -> Any:
    """Map PBR factors onto explicit OBJ Phong constants for MTL export.

    trimesh's OBJ writer converts a PBRMaterial through ``to_simple()``, which
    only carries the diffuse factor and leaves ambient/specular at the
    library's 0.4 gray default, so ``Ka``/``Ks`` would darken or tint the
    baked albedo in Phong viewers. Build the SimpleMaterial explicitly
    instead: ambient and diffuse both carry the base color factor (identity
    for baked textures), specular follows metallicFactor (0.0 for baked
    albedo, so no synthetic sheen is added on top of photo-derived colors),
    and Ns inverts trimesh's ``roughness = (2 / (Ns + 2)) ** 0.25`` mapping.
    """
    import numpy as np
    from trimesh.visual.material import PBRMaterial, SimpleMaterial

    if not isinstance(material, PBRMaterial):
        return material
    base_color = material.baseColorFactor
    if base_color is None:
        base_rgba = np.array([255.0, 255.0, 255.0, 255.0], dtype=np.float64)
    else:
        raw = np.asarray(base_color).reshape(-1)[:4]
        base_rgba = raw.astype(np.float64)
        # Integer factors are already 0-255 (trimesh convention); float
        # factors in 0-1 follow the glTF convention and need rescaling.
        if raw.dtype.kind == "f" and float(base_rgba.max(initial=0.0)) <= 1.0:
            base_rgba = base_rgba * 255.0
        if base_rgba.size < 4:
            base_rgba = np.concatenate([base_rgba, np.full(4 - base_rgba.size, 255.0)])
    # glTF defaults are metallic=1.0 / roughness=1.0 when factors are absent.
    metallic = 1.0 if material.metallicFactor is None else float(material.metallicFactor)
    roughness = 1.0 if material.roughnessFactor is None else float(material.roughnessFactor)
    roughness = min(max(roughness, 1e-3), 1.0)
    glossiness = min(max(2.0 / roughness**4 - 2.0, 0.0), 1000.0)
    # Phong metals reflect through the specular color; dielectrics with baked
    # albedo carry no authored specular at all.
    specular = np.clip(base_rgba[:3] * metallic, 0.0, 255.0)
    return SimpleMaterial(
        image=material.baseColorTexture,
        ambient=base_rgba.astype(np.uint8),
        diffuse=base_rgba.astype(np.uint8),
        specular=np.concatenate([specular, [255.0]]).astype(np.uint8),
        glossiness=glossiness,
        name=material.name,
    )


def _tripo_export_obj_with_textures(
    mesh: Any, *, viewer_frame: bool = True
) -> tuple[bytes, Dict[str, bytes]]:
    from trimesh.exchange import obj as obj_exchange

    # Same viewer-orientation contract as the GLB export: OBJ has no scene
    # nodes, so the rotation is baked into the vertices (float-exact axis
    # permutation); UVs and texture bytes are untouched.
    export_source = mesh
    if viewer_frame:
        export_source = mesh.copy()
        export_source.apply_transform(_canonical_to_gltf_matrix())
    visual = getattr(export_source, "visual", None)
    original_material = getattr(visual, "material", None)
    export_material = _tripo_obj_material_from_pbr(original_material)
    if export_material is not original_material:
        visual.material = export_material
    try:
        payload, texture_files = obj_exchange.export_obj(
            export_source,
            include_texture=True,
            return_texture=True,
            write_texture=False,
            mtl_name="scene.mtl",
        )
    finally:
        if export_material is not original_material:
            visual.material = original_material
    sidecars: Dict[str, bytes] = {}
    for name, value in dict(texture_files or {}).items():
        if isinstance(value, bytes):
            sidecars[str(name)] = value
        elif isinstance(value, bytearray):
            sidecars[str(name)] = bytes(value)
        elif isinstance(value, str):
            sidecars[str(name)] = value.encode("utf-8")
    return payload.encode("utf-8"), sidecars


def _tripo_bake_textured_mesh(
    mesh: Any,
    *,
    model: Any,
    scene_code: Any,
    texture_resolution: int,
    texture_completion: str = "none",
    observed_views: Optional[Sequence[Mapping[str, Any]]] = None,
    observed_rgba: Optional[Any] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Bake the TripoSR texture through the shared projection pipeline.

    The generic bake (seam feathering, best-view-biased blending, reference
    exposure harmonization, camera-distance estimation) lives in
    `abstract3d.texturing`; TripoSR contributes its triplane color field as
    the prior for texels no view observed.
    """
    _require_texture_runtime_dependencies()

    from ..texturing import bake_projection_texture

    effective_views = list(observed_views or [])
    if not effective_views and observed_rgba is not None:
        effective_views.append(
            {
                "label": "front",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "rgba": observed_rgba,
                "role": "source",
            }
        )

    def _triplane_color_field(positions_texture: Any) -> Any:
        return _tripo_positions_to_colors(
            model,
            scene_code,
            positions_texture,
            texture_resolution=int(texture_resolution),
        )

    textured_mesh, stats = bake_projection_texture(
        mesh,
        observed_views=effective_views,
        texture_resolution=int(texture_resolution),
        texture_completion=str(texture_completion or "none"),
        base_color_fn=_triplane_color_field,
    )
    # Preserve the historical TripoSR mode vocabulary used by docs and tests.
    view_count = len(stats.get("observed_view_stats") or [])
    symmetry_applied = bool((stats.get("symmetry_completion") or {}).get("applied"))
    if view_count == 0:
        mode = "triplane_only"
    elif symmetry_applied:
        mode = (
            "hybrid_multiview_plus_symmetry_plus_triplane"
            if view_count > 1
            else "hybrid_observed_plus_symmetry_plus_triplane"
        )
    else:
        mode = "hybrid_multiview_plus_triplane" if view_count > 1 else "hybrid_observed_view_plus_triplane"
    stats["projection_mode"] = mode
    return textured_mesh, stats


# The pipeline's canonical object frame is Z-up with the subject facing +X.
# glTF mandates Y-up / front +Z, so viewers showed exports lying sideways.
# The cyclic axis permutation (x, y, z) -> (y, z, x) maps canonical to glTF
# exactly (a pure rotation; float-exact, no precision loss — verified by
# round-trip tests), and its inverse restores the canonical frame.
_CANONICAL_TO_GLTF = None


def _canonical_to_gltf_matrix() -> Any:
    global _CANONICAL_TO_GLTF
    if _CANONICAL_TO_GLTF is None:
        import numpy as np

        _CANONICAL_TO_GLTF = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return _CANONICAL_TO_GLTF


def _mesh_export_bytes(mesh: Any, *, file_type: str, viewer_frame: bool = True) -> bytes:
    """Serialize a mesh, presenting standard-viewer orientation by default.

    `viewer_frame=True` bakes the canonical->glTF rotation into the exported
    VERTICES (chosen over a glTF scene-node transform so the OBJ sidecar —
    which has no node concept — carries the same orientation, and so
    `trimesh.load(force="mesh")` round-trips identically for both formats).
    Texture bytes are untouched by construction. Callers that persist
    internal working state (e.g. `geometry.glb` consumed by the bake, whose
    math is canonical-frame) pass `viewer_frame=False`.
    """
    export_mesh = mesh
    if viewer_frame:
        export_mesh = mesh.copy()
        export_mesh.apply_transform(_canonical_to_gltf_matrix())
        # Persisted marker (survives GLB round-trips via glTF extras) so the
        # repo's own loaders/renderers can recognize viewer-frame files and
        # compensate back into canonical-frame math.
        export_mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    payload = export_mesh.export(file_type=file_type)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    raise Abstract3DError(f"Unsupported mesh export payload type for {file_type!r}: {type(payload)!r}")


def _write_bundle(
    *,
    root_dir: Path,
    primary_format: str,
    primary_bytes: bytes,
    obj_bytes: bytes,
    source_image: Any,
    prompt: str,
    metadata: Dict[str, Any],
    view_images: Sequence[Any],
) -> Dict[str, Path]:
    root_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    primary_path = root_dir / f"scene.{primary_format}"
    primary_path.write_bytes(primary_bytes)
    obj_path = root_dir / "scene.obj"
    obj_path.write_bytes(obj_bytes)
    source_path = root_dir / "input.png"
    source_image.save(source_path)
    preview_path = root_dir / "preview.png"
    if view_images:
        view_images[0].save(preview_path)
    stats_lines = [
        f"format: {primary_format}",
        f"verts: {metadata.get('vertex_count')}",
        f"faces: {metadata.get('face_count')}",
        f"device: {metadata.get('device')}",
        f"infer_s: {metadata.get('timings_s', {}).get('inference')}",
        f"mesh_s: {metadata.get('timings_s', {}).get('mesh')}",
    ]
    texture_mode = metadata.get("texture_mode")
    if texture_mode:
        stats_lines.append(f"texture: {texture_mode}")
    texture_resolution = metadata.get("texture_resolution")
    if texture_resolution:
        stats_lines.append(f"tex_res: {texture_resolution}")
    contact_sheet = build_case_contact_sheet(
        title=prompt or metadata.get("task", "scene3d"),
        source_image=source_image if isinstance(source_image, Image.Image) else Image.open(source_path),
        views=view_images,
        stats_lines=stats_lines,
    )
    contact_sheet_path = root_dir / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path)
    metadata_path = root_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "primary_path": primary_path,
        "obj_path": obj_path,
        "source_path": source_path,
        "preview_path": preview_path if preview_path.exists() else None,
        "contact_sheet_path": contact_sheet_path,
        "metadata_path": metadata_path,
    }


def _zip_bundle(root_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(root_dir)))
    return buf.getvalue()


_OPEN_FORM_WORDS = frozenset((
    # The user explicitly asked for an open/hollow form: never override.
    "convertible", "cabriolet", "roadster", "spyder", "spider", "targa",
    "open-top", "open", "topless", "cockpit", "interior", "hollow",
    "cutaway", "cross-section",
))

# Subjects whose t2i priors default to OPEN thin-shell forms that
# single-image shape models measurably mangle. "a red sports car" drew an
# open convertible: the shape field then invented the unseen cabin (50x
# the interior angle-defect energy of a closed coupe from the same seed
# and settings) and floated all four wheels off the body. A closed body
# is the same subject with strictly more generable geometry. Only classes
# with a MEASURED open-form failure belong here.
_OPEN_FORM_PRONE_WORDS = frozenset((
    "car", "supercar", "sportscar", "coupe", "sedan", "vehicle",
))

# Wording is position- and strength-sensitive (measured, seeds 11/2025):
# the weak trailing "closed body, no open top" still drew a convertible;
# this subject-level clause right after the user's text drew closed
# hardtops on every tested seed. Same lesson as the texture color anchor:
# mid-prompt subject claims move the model, suffix hints do not.
_CLOSED_FORM_CLAUSE = (
    "a hardtop with a fully closed solid roof, no convertible, "
    "no open cockpit"
)


def _default_text_to_image_prompt(prompt: str) -> str:
    import re

    suffix = (
        "single centered object, studio product photo, neutral light gray background, "
        "fully visible, no crop, no extra objects, realistic lighting"
    )
    base = str(prompt or "").strip()
    words = frozenset(re.findall(r"[a-zA-Z-]+", base.lower()))
    if (words & _OPEN_FORM_PRONE_WORDS) and not (words & _OPEN_FORM_WORDS):
        base = f"{base}, {_CLOSED_FORM_CLAUSE}" if base else base
    return f"{base}, {suffix}" if base else suffix


def _default_image_generator(owner: Any) -> Callable[..., bytes]:
    return default_image_generator(owner)


class TripoSRBackend:
    """Local TripoSR backend with composed text-to-3D support."""

    backend_id = "abstract3d:triposr"

    def __init__(self, owner: Any, *, image_generator: Optional[Callable[..., Any]] = None) -> None:
        self._owner = owner
        self._image_generator = image_generator
        self._resident_model: Optional[Any] = None
        self._resident_device: Optional[str] = None
        self._resident_model_id: Optional[str] = None
        self._resident_source_dir: Optional[str] = None
        self._resident_chunk_size: Optional[int] = None
        self._last_runtime_stats: Dict[str, Any] = {}

    def _model_id(self, requested: Optional[str] = None) -> str:
        model_id = str(requested or _owner_cfg(self._owner, "scene3d_model_id") or _env("ABSTRACT3D_MODEL_ID") or _DEFAULT_MODEL_ID).strip()
        return model_id or _DEFAULT_MODEL_ID

    def _chunk_size(self, requested: Optional[int] = None) -> int:
        if requested is not None:
            return int(requested)
        return _owner_cfg_int(self._owner, "scene3d_chunk_size", 2048)

    def _load_runtime(self, *, model_id: Optional[str] = None, device: Optional[str] = None, chunk_size: Optional[int] = None) -> Any:
        _require_runtime_dependencies()
        resolved_model = self._model_id(model_id)
        resolved_device = _select_device(self._owner, explicit=device)
        resolved_chunk = self._chunk_size(chunk_size)
        source_dir = _resolve_source_dir(self._owner)
        if (
            self._resident_model is not None
            and self._resident_device == resolved_device
            and self._resident_model_id == resolved_model
            and self._resident_chunk_size == resolved_chunk
            and self._resident_source_dir == str(source_dir)
        ):
            return self._resident_model

        started = time.perf_counter()
        model = _load_triposr_model(
            self._owner,
            model_id=resolved_model,
            device=resolved_device,
            chunk_size=resolved_chunk,
        )
        self._resident_model = model
        self._resident_device = resolved_device
        self._resident_model_id = resolved_model
        self._resident_source_dir = str(source_dir)
        self._resident_chunk_size = resolved_chunk
        self._last_runtime_stats = {
            "load_s": round(time.perf_counter() - started, 4),
            "device": resolved_device,
            "model_id": resolved_model,
            "source_dir": str(source_dir),
            "chunk_size": resolved_chunk,
        }
        return model

    def _clear_runtime(self) -> None:
        self._resident_model = None
        self._resident_device = None
        self._resident_model_id = None
        self._resident_source_dir = None
        self._resident_chunk_size = None
        self._last_runtime_stats = {}
        torch = importlib.import_module("torch")
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        installed = all(importlib.util.find_spec(name) is not None for name in _RUNTIME_IMPORTS)
        composition_ready = has_image_composer(self._owner)
        normalized_task = _TASK_ALIASES.get(str(task).strip().lower().replace("-", "_")) if task is not None else None
        if normalized_task == "text_to_scene3d":
            tasks = ["text_to_scene3d"]
            status = "available" if installed and composition_ready else "install_required"
        elif normalized_task == "image_to_scene3d":
            tasks = ["image_to_scene3d"]
            status = "available" if installed else "install_required"
        else:
            tasks = ["image_to_scene3d"]
            if composition_ready:
                tasks.append("text_to_scene3d")
            status = "available" if installed else "install_required"
        return [
            {
                "provider_id": "triposr",
                "display_name": "Local TripoSR",
                "tasks": tasks,
                "local": True,
                "remote": False,
                "status": status,
                "backend_id": self.backend_id,
                "installed": installed,
                "configured": bool(installed and (normalized_task != "text_to_scene3d" or composition_ready)),
                "selected": True,
                "metadata": {
                    "source_snapshot": _TRIPOSR_COMMIT,
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
        if selector and selector not in {"triposr", "abstract3d:triposr"}:
            return []
        return capability_model_records(task=task, validated_only=True)

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
                            "mc_resolution": {"type": "integer"},
                            "cleanup": {"type": "string", "enum": ["presentation", "none"]},
                            "texture_mode": {"type": "string", "enum": ["vertex_color", "baked_basecolor"]},
                            "texture_resolution": {"type": "integer"},
                            "texture_completion": {"type": "string", "enum": ["none", "mirror_symmetry", "auto"]},
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
                        "mc_resolution": {"type": "integer"},
                        "cleanup": {"type": "string", "enum": ["presentation", "none"]},
                        "texture_mode": {"type": "string", "enum": ["vertex_color", "baked_basecolor"]},
                        "texture_resolution": {"type": "integer"},
                        "remove_background": {"type": "boolean"},
                        "texture_reference_images": {"type": "array", "items": {"type": "string"}},
                        "texture_reference_angles": {"type": "array", "items": {"type": "string"}},
                        "texture_reference_remove_background": {"type": "boolean"},
                        "texture_completion": {"type": "string", "enum": ["none", "mirror_symmetry", "auto"]},
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
        already_loaded = self._resident_model is not None
        runtime = self._load_runtime(
            model_id=request.get("model"),
            device=request.get("device"),
            chunk_size=request.get("chunk_size"),
        )
        _ = runtime
        return {
            "task": str(request.get("task") or "scene3d_generation"),
            "provider": "triposr",
            "model": self._resident_model_id,
            "backend_id": self.backend_id,
            "state": "loaded",
            "loaded": True,
            "loaded_new": not already_loaded,
            "details": dict(self._last_runtime_stats),
        }

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        _ = filters
        if self._resident_model is None:
            return []
        return [
            {
                "task": "scene3d_generation",
                "provider": "triposr",
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
            "provider": "triposr",
            "model": model_id,
            "backend_id": self.backend_id,
            "state": "unloaded",
            "unloaded": True,
        }

    def _make_source_image(self, prompt: str, **kwargs: Any):
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
        foreground_ratio: float,
        mc_resolution: Optional[int],
        device: Optional[str],
        chunk_size: Optional[int],
        bundle: bool,
        model: Optional[str],
        **kwargs: Any,
    ):
        from PIL import Image
        import psutil
        import torch

        # Validate option names BEFORE loading the model or running inference:
        # the enforcement check at the end of this method (after every pop) is
        # authoritative, but reaching it costs minutes of model load + compute.
        # A typo or another backend's knob (e.g. `seed` — TripoSR is
        # feed-forward and takes none) must fail in milliseconds instead.
        from . import reject_unknown_options
        from ..image_composition import IMAGE_REQUEST_KEYS

        unknown_preflight = {
            key: None
            for key in kwargs
            if key not in _TRIPOSR_GENERATION_OPTION_KEYS and key not in IMAGE_REQUEST_KEYS
        }
        reject_unknown_options(self.backend_id, unknown_preflight)

        model_runtime = self._load_runtime(model_id=model, device=device, chunk_size=chunk_size)
        actual_task = _TASK_ALIASES.get(task, task)
        if actual_task not in {"text_to_scene3d", "image_to_scene3d"}:
            raise CapabilityNotSupportedError(f"Unsupported TripoSR task: {actual_task!r}")

        image_generation_s: Optional[float] = None
        if actual_task == "text_to_scene3d":
            from ..image_composition import pop_composition_kwargs

            image_started = time.perf_counter()
            image_bytes = self._make_source_image(prompt, **pop_composition_kwargs(kwargs))
            image_generation_s = round(time.perf_counter() - image_started, 4)
            image_input = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        else:
            if image is None:
                raise ValueError("image_to_scene3d requires an image input.")
            image_input = image

        preprocess_started = time.perf_counter()
        prepared = _prepare_triposr_image(
            image_input,
            remove_background=remove_background,
            foreground_ratio=float(foreground_ratio),
            artifact_store=artifact_store,
        )
        if len(prepared) == 3:
            source_preview, prepared_rgb, background_removed = prepared
            observed_rgba = None
        else:
            source_preview, prepared_rgb, background_removed, observed_rgba = prepared
        preprocess_s = round(time.perf_counter() - preprocess_started, 4)
        started = time.perf_counter()
        with torch.no_grad():
            scene_codes = model_runtime([prepared_rgb], device=self._resident_device or "cpu")
        inference_s = round(time.perf_counter() - started, 4)
        resolved_mc_resolution = _tripo_default_mc_resolution(self._owner, mc_resolution)
        cleanup_mode = _tripo_cleanup_mode(self._owner, kwargs.pop("cleanup", None))
        texture_mode = _tripo_texture_mode(self._owner, kwargs.pop("texture_mode", None))
        texture_requested = texture_mode == "baked_basecolor"
        texture_resolution = _tripo_texture_resolution(self._owner, kwargs.pop("texture_resolution", None)) if texture_requested else None
        texture_completion = _tripo_texture_completion_mode(self._owner, kwargs.pop("texture_completion", None)) if texture_requested else "none"
        raw_texture_reference_views = kwargs.pop("texture_reference_views", None)
        raw_texture_reference_images = kwargs.pop("texture_reference_images", None)
        raw_texture_reference_angles = kwargs.pop("texture_reference_angles", None)
        texture_reference_views = _tripo_normalize_texture_reference_views(
            raw_views=raw_texture_reference_views,
            raw_images=raw_texture_reference_images,
            raw_angles=raw_texture_reference_angles,
        )
        texture_reference_remove_background = _tripo_texture_reference_remove_background(
            self._owner,
            kwargs.pop("texture_reference_remove_background", None),
            fallback=remove_background,
        )
        # Every supported option has been consumed; anything left is a typo
        # or another backend's knob and must fail loudly, not silently.
        from . import reject_unknown_options

        reject_unknown_options(self.backend_id, kwargs)
        mesh_started = time.perf_counter()
        meshes = model_runtime.extract_mesh(scene_codes, not texture_requested, resolution=int(resolved_mc_resolution))
        geometry_mesh = meshes[0]
        geometry_mesh, postprocess_applied, postprocess_warnings, cleanup_details = _tripo_postprocess_mesh(
            geometry_mesh,
            cleanup_mode=cleanup_mode,
        )
        mesh_s = round(time.perf_counter() - mesh_started, 4)
        texture_s: Optional[float] = None
        texture_warnings: List[str] = []
        texture_artifacts: Dict[str, Any] = {}
        prepared_texture_reference_views: List[Dict[str, Any]] = []
        obj_texture_sidecars: Dict[str, bytes] = {}
        mesh = geometry_mesh
        if texture_requested:
            texture_started = time.perf_counter()
            try:
                prepared_texture_reference_views = _prepare_texture_reference_views(
                    source_observed_rgba=observed_rgba,
                    source_preview=source_preview,
                    texture_reference_views=texture_reference_views,
                    texture_reference_remove_background=texture_reference_remove_background,
                    foreground_ratio=float(foreground_ratio),
                    artifact_store=artifact_store,
                )
                mesh, texture_artifacts = _tripo_bake_textured_mesh(
                    geometry_mesh,
                    model=model_runtime,
                    scene_code=scene_codes[0],
                    texture_resolution=int(texture_resolution or _DEFAULT_TRIPOSR_TEXTURE_RESOLUTION),
                    texture_completion=texture_completion,
                    observed_views=prepared_texture_reference_views,
                )
            except Exception as exc:
                raise Abstract3DError(f"Failed to bake TripoSR texture atlas: {type(exc).__name__}: {exc}") from exc
            texture_s = round(time.perf_counter() - texture_started, 4)
        glb_bytes = _mesh_export_bytes(mesh, file_type="glb")
        if texture_requested:
            obj_bytes, obj_texture_sidecars = _tripo_export_obj_with_textures(mesh)
        else:
            obj_bytes = _mesh_export_bytes(mesh, file_type="obj")
        views = render_mesh_views(mesh)
        preview_renderer = get_last_render_backend()
        primary_format = str(format or "glb").strip().lower() or "glb"
        if primary_format not in {"glb", "obj", "zip"}:
            raise ValueError("scene3d format must be one of: glb, obj, zip")
        if texture_requested and primary_format == "obj" and not output_dir:
            raise ValueError("Baked TripoSR texture output requires output_dir when format=obj. Use format=glb or format=zip for single-payload results.")
        primary_bytes = glb_bytes if primary_format == "glb" else obj_bytes
        content_type = "model/gltf-binary" if primary_format == "glb" else "text/plain"
        if primary_format == "zip":
            content_type = "application/zip"
        total_s = round((image_generation_s or 0.0) + preprocess_s + inference_s + mesh_s + (texture_s or 0.0), 4)

        process = psutil.Process(os.getpid())
        runtime_meta: Dict[str, Any] = {
            "backend_id": self.backend_id,
            "provider": self.backend_id,
            "model_id": self._resident_model_id or self._model_id(model),
            "task": actual_task,
            "device": self._resident_device or "cpu",
            "format": primary_format,
            "content_type": content_type,
            "mc_resolution": int(resolved_mc_resolution),
            "chunk_size": self._resident_chunk_size or self._chunk_size(chunk_size),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "appearance_mode": "uv_basecolor" if texture_requested else "vertex_color",
            "texture_mode": texture_mode,
            "texture_resolution": int(texture_resolution) if texture_requested and texture_resolution is not None else None,
            "texture_completion": texture_completion if texture_requested else None,
            "uv_present": bool(texture_requested),
            "material_count": 1 if texture_requested else 0,
            "preview_renderer": preview_renderer,
            "timings_s": {
                "source_image_generation": image_generation_s,
                "preprocess": preprocess_s,
                "inference": inference_s,
                "mesh": mesh_s,
                "texture": texture_s,
                "load": self._last_runtime_stats.get("load_s"),
                "total": total_s,
            },
            "memory": {
                "rss_bytes": int(process.memory_info().rss),
                "mps_allocated_bytes": int(torch.mps.current_allocated_memory()) if (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()) else None,
            },
            "background_removed": bool(background_removed),
            "source_snapshot": _TRIPOSR_COMMIT,
            "cleanup_mode": cleanup_mode,
            "surface_cleanup": dict(cleanup_details.get("settings") or {}),
            "postprocess_cleanup": list(postprocess_applied),
            "postprocess_warnings": list(postprocess_warnings),
            "topology_before_cleanup": dict(cleanup_details.get("topology_before") or {}),
            "topology": dict(cleanup_details.get("topology_after") or {}),
            "texture_warnings": texture_warnings,
            "notes": [],
        }
        if cleanup_mode == "presentation":
            runtime_meta["notes"].append(
                "TripoSR applies a deterministic CPU postprocess after marching-cubes extraction: small-component pruning, marching-cube cleanup, light Taubin smoothing, hole repair, and normal repair."
            )
        if texture_requested:
            runtime_meta["texture_artifacts"] = {
                "geometry_glb_path": None,
                "texture_path": None,
                "uv_preview_path": None,
                "obj_sidecars": sorted(str(name) for name in obj_texture_sidecars.keys()),
                "texture_padding": texture_artifacts.get("texture_padding"),
                "projection_mode": texture_artifacts.get("projection_mode"),
                "observed_coverage_ratio": texture_artifacts.get("observed_coverage_ratio"),
                "observed_view_stats": list(texture_artifacts.get("observed_view_stats") or []),
                "texture_completion": texture_artifacts.get("texture_completion"),
                "symmetry_completion": dict(texture_artifacts.get("symmetry_completion") or {}),
                "reference_view_count": max(0, len(prepared_texture_reference_views) - 1),
                "uv_vertex_count": texture_artifacts.get("uv_vertex_count"),
                "vertex_mapping_count": texture_artifacts.get("vertex_mapping_count"),
                "reference_view_paths": [],
            }
            runtime_meta["notes"].append(
                "TripoSR baked_basecolor mode unwraps UVs, rasterizes a position atlas, projects one or more observed views onto front-facing texels, blends those projections, falls back to triplane color elsewhere, and exports a textured GLB."
            )
            if texture_completion == "mirror_symmetry":
                runtime_meta["notes"].append(
                    "TripoSR mirror_symmetry texture completion reflects front-view texels across the left-right object plane and fills only uncovered front-side regions."
                )
        runtime_meta["notes"].extend(postprocess_warnings)
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
            if texture_requested:
                geometry_glb_path = bundle_root / "geometry.glb"
                geometry_glb_path.write_bytes(_mesh_export_bytes(geometry_mesh, file_type="glb", viewer_frame=False))
                texture_path = bundle_root / "texture.png"
                texture_artifacts["texture_image"].save(texture_path)
                uv_preview_path = bundle_root / "uv_preview.png"
                texture_artifacts["uv_preview"].save(uv_preview_path)
                for sidecar_name, sidecar_bytes in obj_texture_sidecars.items():
                    (bundle_root / str(sidecar_name)).write_bytes(sidecar_bytes)
                reference_view_paths: List[str] = []
                for index, prepared_view in enumerate(prepared_texture_reference_views[1:], start=1):
                    label = re.sub(
                        r"[^a-z0-9_]+",
                        "_",
                        str(prepared_view.get("label") or f"reference_{index:02d}").strip().lower(),
                    ).strip("_")
                    label = label or f"reference_{index:02d}"
                    ref_path = bundle_root / f"texture_reference_{index:02d}_{label}.png"
                    prepared_view["source_preview"].save(ref_path)
                    reference_view_paths.append(str(ref_path))
                runtime_meta["texture_artifacts"] = {
                    "geometry_glb_path": str(geometry_glb_path),
                    "texture_path": str(texture_path),
                    "uv_preview_path": str(uv_preview_path),
                    "obj_sidecars": sorted(str(name) for name in obj_texture_sidecars.keys()),
                    "texture_padding": texture_artifacts.get("texture_padding"),
                    "projection_mode": texture_artifacts.get("projection_mode"),
                    "observed_coverage_ratio": texture_artifacts.get("observed_coverage_ratio"),
                    "observed_view_stats": list(texture_artifacts.get("observed_view_stats") or []),
                    "texture_completion": texture_artifacts.get("texture_completion"),
                    "symmetry_completion": dict(texture_artifacts.get("symmetry_completion") or {}),
                    "reference_view_count": max(0, len(prepared_texture_reference_views) - 1),
                    "uv_vertex_count": texture_artifacts.get("uv_vertex_count"),
                    "vertex_mapping_count": texture_artifacts.get("vertex_mapping_count"),
                    "reference_view_paths": reference_view_paths,
                }
            runtime_meta["bundle_dir"] = str(bundle_root)
            runtime_meta["preview_path"] = str(bundle_paths.get("preview_path")) if bundle_paths.get("preview_path") else None
            runtime_meta["contact_sheet_path"] = str(bundle_paths.get("contact_sheet_path")) if bundle_paths.get("contact_sheet_path") else None
            runtime_meta["metadata_path"] = str(bundle_paths.get("metadata_path")) if bundle_paths.get("metadata_path") else None
            runtime_meta["source_image_path"] = str(bundle_paths.get("source_path")) if bundle_paths.get("source_path") else None

        if primary_format == "zip":
            if bundle_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="abstract3d-bundle-"))
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
                if texture_requested:
                    geometry_glb_path = bundle_root / "geometry.glb"
                    geometry_glb_path.write_bytes(_mesh_export_bytes(geometry_mesh, file_type="glb", viewer_frame=False))
                    texture_path = bundle_root / "texture.png"
                    texture_artifacts["texture_image"].save(texture_path)
                    uv_preview_path = bundle_root / "uv_preview.png"
                    texture_artifacts["uv_preview"].save(uv_preview_path)
                    for sidecar_name, sidecar_bytes in obj_texture_sidecars.items():
                        (bundle_root / str(sidecar_name)).write_bytes(sidecar_bytes)
                    reference_view_paths = []
                    for index, prepared_view in enumerate(prepared_texture_reference_views[1:], start=1):
                        label = re.sub(
                            r"[^a-z0-9_]+",
                            "_",
                            str(prepared_view.get("label") or f"reference_{index:02d}").strip().lower(),
                        ).strip("_")
                        label = label or f"reference_{index:02d}"
                        ref_path = bundle_root / f"texture_reference_{index:02d}_{label}.png"
                        prepared_view["source_preview"].save(ref_path)
                        reference_view_paths.append(str(ref_path))
                    runtime_meta["texture_artifacts"] = {
                        "geometry_glb_path": str(geometry_glb_path),
                        "texture_path": str(texture_path),
                        "uv_preview_path": str(uv_preview_path),
                        "obj_sidecars": sorted(str(name) for name in obj_texture_sidecars.keys()),
                        "texture_padding": texture_artifacts.get("texture_padding"),
                        "projection_mode": texture_artifacts.get("projection_mode"),
                        "observed_coverage_ratio": texture_artifacts.get("observed_coverage_ratio"),
                        "observed_view_stats": list(texture_artifacts.get("observed_view_stats") or []),
                        "texture_completion": texture_artifacts.get("texture_completion"),
                        "symmetry_completion": dict(texture_artifacts.get("symmetry_completion") or {}),
                        "reference_view_count": max(0, len(prepared_texture_reference_views) - 1),
                        "uv_vertex_count": texture_artifacts.get("uv_vertex_count"),
                        "vertex_mapping_count": texture_artifacts.get("vertex_mapping_count"),
                        "reference_view_paths": reference_view_paths,
                    }
            primary_bytes = _zip_bundle(bundle_root)

        runtime_meta["output_bytes"] = len(primary_bytes)
        metadata_path = bundle_paths.get("metadata_path")
        if isinstance(metadata_path, Path):
            metadata_path.write_text(json.dumps(runtime_meta, indent=2, sort_keys=True), encoding="utf-8")

        artifact_id = stable_artifact_id(primary_bytes)
        stored = store_bytes(
            artifact_store,
            primary_bytes,
            content_type=content_type,
            run_id=run_id,
            tags=tags,
            artifact_id=artifact_id,
            metadata=runtime_meta,
        ) if artifact_store is not None else primary_bytes

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
            # The official TripoSR reference pipeline always segments the
            # subject and recenters it (resize_foreground) before inference;
            # the model was trained on segmented objects. Composed t23d images
            # come back with an opaque studio background, so background
            # removal must default to auto here or the reconstruction
            # degenerates into billboard-like sheets on thin subjects.
            remove_background=kwargs.pop("remove_background", None),
            foreground_ratio=float(kwargs.pop("foreground_ratio", 0.85) or 0.85),
            mc_resolution=kwargs.pop("mc_resolution", None),
            device=kwargs.pop("device", None),
            chunk_size=kwargs.pop("chunk_size", None),
            bundle=bool(kwargs.pop("bundle", False)),
            model=kwargs.pop("model", None),
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
            foreground_ratio=float(kwargs.pop("foreground_ratio", 0.85) or 0.85),
            mc_resolution=kwargs.pop("mc_resolution", None),
            device=kwargs.pop("device", None),
            chunk_size=kwargs.pop("chunk_size", None),
            bundle=bool(kwargs.pop("bundle", False)),
            model=kwargs.pop("model", None),
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
        mc_resolution: Optional[int] = None,
        device: Optional[str] = None,
        model: Optional[str] = None,
        model_subfolder: Optional[str] = None,
        remove_background: Optional[bool] = None,
        cleanup: Optional[str] = None,
        texture_mode: Optional[str] = None,
        texture_resolution: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        chunk_size: Optional[int] = None,
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
                cleanup=cleanup,
                texture_mode=texture_mode,
                texture_resolution=texture_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                chunk_size=chunk_size,
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
                model=model,
                remove_background=remove_background,
                cleanup=cleanup,
                texture_mode=texture_mode,
                texture_resolution=texture_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                chunk_size=chunk_size,
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
        return {
            "contact_sheet": str(sheet_path),
            "stats": str(stats_path),
            "rows": summary_rows,
        }
