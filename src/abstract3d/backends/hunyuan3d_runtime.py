"""Official Hunyuan3D-2.1 shape backend with an explicit license gate.

This backend wraps the official `tencent/Hunyuan3D-2.1` shape stage
(`Hunyuan3DDiTFlowMatchingPipeline`): a 3.3B flow-matching DiT plus a shape
VAE whose decoder is queried over an octree grid and surfaced with marching
cubes. The stage is fully self-contained (the DINOv2-L conditioner weights
ship inside the official checkpoint), so no gated companion model is needed.

License boundary: the Tencent Hunyuan 3D 2.1 Community License is
territory-restricted (it excludes the European Union, United Kingdom, and
South Korea) and caps large-scale commercial use. Because Abstract3D cannot
know where it is being run, this backend refuses to download or run official
weights until the operator explicitly acknowledges the license through
`scene3d_hunyuan_license_accepted` or `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`.

The official PaintPBR texture stage depends on CUDA-only rasterization
kernels and is intentionally out of scope here; textured output for this
backend goes through the shared projection-bake pipeline instead.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..artifacts import is_artifact_ref, stable_artifact_id, store_bytes
from ..errors import (
    Abstract3DError,
    CapabilityNotSupportedError,
    DependencyUnavailableError,
    InvalidRequestError,
    SourceBootstrapError,
)
from ..image_composition import COMPOSITION_INSTALL_HINT, has_image_composer, pop_image_generation_request
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
    _tripo_export_obj_with_textures,
    _tripo_normalize_texture_reference_views,
    _write_bundle,
    _zip_bundle,
)

_HUNYUAN_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
_HUNYUAN_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
_SOURCE_MANIFEST = ".abstract3d-source.json"
_OFFICIAL_MODEL_ID = "tencent/Hunyuan3D-2.1"
_DIT_SUBFOLDER = "hunyuan3d-dit-v2-1"
_MV_MODEL_ID = "tencent/Hunyuan3D-2mv"
_MV_DIT_SUBFOLDER = "hunyuan3d-dit-v2-mv"
_MV_SUPPORTED_SUBFOLDERS = (
    "hunyuan3d-dit-v2-mv",
    "hunyuan3d-dit-v2-mv-fast",
    "hunyuan3d-dit-v2-mv-turbo",
)
# Maps this repository's camera-azimuth convention (side_left = +90, camera
# on the subject's left) onto the MVImageProcessorV2 view tags, whose
# docstring defines left as "front clockwise 90" seen from above. Verified
# empirically on the checked face proof: the subject's-left profile photo
# conditions the mesh's left side when tagged "left".
_MV_TAG_BY_AZIMUTH = {0.0: "front", 90.0: "left", 180.0: "back", -90.0: "right"}
_MV_TAG_SNAP_TOLERANCE_DEG = 25.0
_LICENSE_NAME = "tencent-hunyuan-community"
_LICENSE_SUMMARY = (
    "Tencent Hunyuan 3D 2.1 Community License: territory-restricted "
    "(excludes the European Union, United Kingdom, and South Korea), "
    "with additional terms for large-scale commercial deployments."
)
_LICENSE_HINT = (
    "The Hunyuan3D-2.1 backend requires an explicit license acknowledgment because the official "
    f"weights are distributed under the {_LICENSE_SUMMARY} Review "
    "https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE and, if the terms apply to "
    "you, opt in with scene3d_hunyuan_license_accepted=true or ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1."
)
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
    "diffusers",
    "transformers",
    "huggingface_hub",
    "trimesh",
    "PIL",
    "skimage",
    "cv2",
    "pymeshlab",
    "einops",
    "yaml",
    "psutil",
)
_RUNTIME_LOCK = threading.Lock()

_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 5.0
# 512 measured strictly better than 384 at EQUAL decode time with the
# adaptive decoder (car: dihedral roughness 8.0% -> 6.7%, converged
# topology; 216s vs 228s). 256 is catastrophic on thin-shell subjects
# (84 extra spurious handles, euler -208 on the car). Guidance stays 5.0:
# 7.5 fused two wheels into the body (genus 17 -> 37).
_DEFAULT_OCTREE_RESOLUTION = 512
_DEFAULT_NUM_CHUNKS = 8000
_DEFAULT_MPS_NUM_CHUNKS = 32768
# Bounded by texture quality, not geometry fidelity: marching-cubes
# micro-detail above ~120k faces fragments the UV atlas into thousands of
# confetti charts (3315 charts at 200k vs 87 at 120k on the owl proof), which
# shows up as salt-and-pepper texel noise. Quadric decimation to this budget
# preserves the silhouette while keeping the atlas bakeable. The official
# pipeline textures at 40k faces for the same reason.
_DEFAULT_MAX_FACENUM = 120000
_DEFAULT_SEED = 2025


def _require_runtime_dependencies() -> None:
    missing = [name for name in _RUNTIME_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        raise DependencyUnavailableError(
            "Hunyuan3D-2.1 runtime dependencies are missing: "
            + ", ".join(sorted(missing))
            + '. Install with: pip install "abstract3d[hunyuan3d]"'
        )


def _license_accepted(owner: Any) -> bool:
    if _owner_cfg_bool(owner, "scene3d_hunyuan_license_accepted", False):
        return True
    raw = str(_env("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _require_license_acceptance(owner: Any) -> None:
    if not _license_accepted(owner):
        raise CapabilityNotSupportedError(_LICENSE_HINT)


def _select_device(owner: Any, explicit: Optional[str] = None) -> str:
    torch = importlib.import_module("torch")
    requested = str(explicit or _owner_cfg(owner, "scene3d_device") or _env("ABSTRACT3D_DEVICE") or "auto").strip().lower()
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    cuda_available = getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    if requested == "cpu":
        return "cpu"
    if requested in {"mps", "metal"} and mps_available:
        return "mps"
    if requested in {"cuda", "cuda:0"} and cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def _select_dtype(device: str, explicit: Optional[str] = None) -> str:
    requested = str(explicit or "").strip().lower()
    if requested in {"float16", "fp16", "half"}:
        return "float16"
    if requested in {"float32", "fp32"}:
        return "float32"
    # The official pipeline runs float16. CPU matmuls in float16 are both slow
    # and poorly supported, so CPU falls back to float32.
    return "float32" if device == "cpu" else "float16"


def _clone_repo(*, repo_url: str, commit: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="abstract3d-hunyuan3d-", dir=str(repo_dir.parent)))
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
        shutil.rmtree(tmp_root / "repo" / ".git", ignore_errors=True)
        (tmp_root / "repo" / _SOURCE_MANIFEST).write_text(
            json.dumps({"repo_url": repo_url, "commit": commit}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        shutil.move(str(tmp_root / "repo"), str(repo_dir))
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or exc.stdout or "").strip()
        raise SourceBootstrapError(f"Failed to prepare pinned Hunyuan3D-2.1 source snapshot: {stderr}") from exc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _resolve_source_dir(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_hunyuan_source_dir") or _env("ABSTRACT3D_HUNYUAN_SOURCE_DIR")
    if configured:
        path = Path(str(configured)).expanduser().resolve()
        if not (path / "hy3dshape" / "hy3dshape" / "pipelines.py").exists():
            raise SourceBootstrapError(
                f"Configured Hunyuan3D-2.1 source dir is missing hy3dshape/hy3dshape/pipelines.py: {path}"
            )
        return path
    repo_dir = _cache_root(owner) / "vendor" / "hunyuan3d21" / _HUNYUAN_COMMIT
    with _RUNTIME_LOCK:
        _clone_repo(repo_url=_HUNYUAN_REPO_URL, commit=_HUNYUAN_COMMIT, repo_dir=repo_dir)
    if not (repo_dir / "hy3dshape" / "hy3dshape" / "pipelines.py").exists():
        raise SourceBootstrapError(f"Pinned Hunyuan3D-2.1 snapshot is incomplete: {repo_dir}")
    return repo_dir


@contextmanager
def _sys_path(path: Path):
    entry = str(path)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def _resolve_model_selection(
    explicit_model: Optional[str],
    explicit_subfolder: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve `(repo_id, checkpoint_subfolder)` for the official families.

    Two official repositories are supported: the single-view 2.1 flagship
    and the multi-view 2.0-family `Hunyuan3D-2mv` (both under the same
    Tencent community license, so the same acknowledgment gate applies).
    Accepts bare repo ids, `repo/subfolder` forms (list_models round-trip),
    and an explicit subfolder for the mv speed variants.
    """
    requested = str(explicit_model or "").strip() or _OFFICIAL_MODEL_ID
    subfolder = str(explicit_subfolder or "").strip()

    if requested in {_OFFICIAL_MODEL_ID, f"{_OFFICIAL_MODEL_ID}/{_DIT_SUBFOLDER}"}:
        return _OFFICIAL_MODEL_ID, _DIT_SUBFOLDER
    if requested == _MV_MODEL_ID:
        selected = subfolder or _MV_DIT_SUBFOLDER
        if selected not in _MV_SUPPORTED_SUBFOLDERS:
            raise CapabilityNotSupportedError(
                f"Unsupported Hunyuan3D-2mv checkpoint subfolder {selected!r}. "
                f"Supported: {', '.join(_MV_SUPPORTED_SUBFOLDERS)}."
            )
        return _MV_MODEL_ID, selected
    for mv_subfolder in _MV_SUPPORTED_SUBFOLDERS:
        if requested == f"{_MV_MODEL_ID}/{mv_subfolder}":
            return _MV_MODEL_ID, mv_subfolder
    raise CapabilityNotSupportedError(
        "The Hunyuan3D backend accepts only the official model repositories "
        f"{_OFFICIAL_MODEL_ID!r} and {_MV_MODEL_ID!r}. Got {requested!r}."
    )


def _remap_mv_config(config_path: Path, cache_root: Path) -> Path:
    """Rewrite 2.0-family config targets onto the vendored 2.1 namespace.

    The `Hunyuan3D-2mv` checkpoint config targets the `hy3dgen.shapegen.*`
    package (the 2.0 codebase), which the pinned 2.1 source tree does not
    ship. Every referenced class exists 1:1 in the vendored `hy3dshape`
    package (`Hunyuan3DDiT`, `ShapeVAE`, `SingleImageEncoder`,
    `FlowMatchEulerDiscreteScheduler`, `MVImageProcessorV2`,
    `Hunyuan3DDiTFlowMatchingPipeline`), so a namespace rewrite is the whole
    compatibility layer. The patched copy lives in the abstract3d cache; the
    HF snapshot itself is never modified.
    """
    text = config_path.read_text(encoding="utf-8")
    if "hy3dgen.shapegen." not in text:
        return config_path
    patched = text.replace("hy3dgen.shapegen.", "hy3dshape.")
    target_dir = cache_root / "vendor" / "hunyuan3d-mv-config"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(patched.encode("utf-8")).hexdigest()[:16]
    target_path = target_dir / f"config-{digest}.yaml"
    if not target_path.exists():
        target_path.write_text(patched, encoding="utf-8")
    return target_path


def _download_official_weights(
    owner: Any,
    *,
    repo_id: str = _OFFICIAL_MODEL_ID,
    subfolder: str = _DIT_SUBFOLDER,
) -> Path:
    from huggingface_hub import snapshot_download

    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*"],
        )
    except Exception as exc:
        raise DependencyUnavailableError(
            f"Failed to download the official Hunyuan3D shape weights ({repo_id}, "
            f"subfolder {subfolder}): {type(exc).__name__}: {exc}"
        ) from exc
    subfolder_path = Path(snapshot_dir) / subfolder
    config_path = subfolder_path / "config.yaml"
    ckpt_path = subfolder_path / "model.fp16.ckpt"
    if not config_path.exists() or not ckpt_path.exists():
        raise DependencyUnavailableError(
            f"The Hunyuan3D snapshot at {subfolder_path} is missing config.yaml or model.fp16.ckpt."
        )
    return subfolder_path


class _AdaptiveVolumeDecoder:
    """Coarse-to-fine SDF grid decoder that is safe for thin structures.

    The upstream `HierarchicalVolumeDecoding` is arithmetically broken as
    shipped: it casts the float grid cell size to int (0 for any bounds
    within +-1.01 at octree >= 252), so every refinement query collapses to
    the cell corner and the decode crashes for EVERY subject on every
    device (reproduced on CPU and MPS; the earlier diagnosis here — thin
    geometry between coarse samples, MPS scatter — described real
    fragilities but not the actual failure, which fires before either).

    Correctness note (adversarially measured): on identical latents this
    decoder and upstream's dense `VanillaVolumeDecoder` produce the SAME
    field to fp32 precision — 0 sign disagreements across 57M grid
    vertices, bit-identical meshes, on both a smooth compact subject (owl)
    and a thin-shell/high-genus one (car) — at ~10x less decode time
    (228s vs 2362s at 384^3 on MPS). Mesh defects observed on hard
    subjects live in the DiT/VAE field itself, not in this surfacing.

    This decoder:
    - starts from a dense grid at the largest exact halving of the final
      resolution that does not exceed `coarse_resolution` (for the default
      384 final with coarse 128 this is 96; empirically fine enough to
      register thin parts that survive marching cubes at 256-512),
    - refines only cells within a conservative band around the coarse
      surface (sign changes dilated by two cells, plus |logit| below a band
      threshold), doubling resolution per level so fine vertex 2*i coincides
      exactly with coarse vertex i,
    - keeps all index bookkeeping in numpy on the host and only runs the
      cross-attention queries on the accelerator,
    - falls back to dense decoding when the coarse level finds no surface,
      and to a single dense level for odd final resolutions.
    """

    def __init__(self, *, coarse_resolution: int = 128, band: float = 0.95) -> None:
        self.coarse_resolution = int(coarse_resolution)
        self.band = float(band)

    @staticmethod
    def _query_points(
        points: Any,
        *,
        latents: Any,
        geo_decoder: Any,
        num_chunks: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        import numpy as np
        import torch

        outputs = []
        tensor_points = torch.from_numpy(np.ascontiguousarray(points)).to(device=device, dtype=dtype)
        for start in range(0, tensor_points.shape[0], int(num_chunks)):
            chunk = tensor_points[start : start + int(num_chunks)].unsqueeze(0)
            logits = geo_decoder(queries=chunk, latents=latents)
            outputs.append(logits[0, ..., 0].detach().to("cpu", dtype=torch.float32))
        if not outputs:
            import torch as _torch

            return _torch.zeros((0,), dtype=_torch.float32)
        import torch as _torch

        return _torch.cat(outputs, dim=0)

    def __call__(
        self,
        latents: Any,
        geo_decoder: Any,
        bounds: Any = 1.01,
        num_chunks: int = 8000,
        octree_resolution: Optional[int] = None,
        mc_level: float = 0.0,
        enable_pbar: bool = True,
        **kwargs: Any,
    ) -> Any:
        import numpy as np
        import torch
        from scipy import ndimage

        del enable_pbar, kwargs
        device = latents.device
        dtype = latents.dtype
        final_resolution = int(octree_resolution or 384)
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.asarray(bounds[0:3], dtype=np.float64)
        bbox_max = np.asarray(bounds[3:6], dtype=np.float64)

        # Build the level schedule as exact halvings of the final resolution
        # (e.g. 384 -> [96, 192, 384]). Exact doubling per level guarantees
        # that fine vertex 2*i coincides with coarse vertex i, which the
        # refinement's index mapping relies on; an inexact last step (e.g.
        # 256 -> 384) would refine misaligned cells and silently cap the
        # effective resolution.
        resolutions: List[int] = [final_resolution]
        while resolutions[0] > self.coarse_resolution and resolutions[0] % 2 == 0:
            resolutions.insert(0, resolutions[0] // 2)

        def grid_axes(resolution: int) -> tuple[Any, Any, Any]:
            xs = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float64)
            ys = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float64)
            zs = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float64)
            return xs, ys, zs

        # Dense pass at the coarse level.
        coarse_resolution = resolutions[0]
        xs, ys, zs = grid_axes(coarse_resolution)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        coarse_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)
        coarse_logits = self._query_points(
            coarse_points,
            latents=latents,
            geo_decoder=geo_decoder,
            num_chunks=num_chunks,
            device=device,
            dtype=dtype,
        )
        grid = coarse_logits.numpy().reshape(
            coarse_resolution + 1, coarse_resolution + 1, coarse_resolution + 1
        )

        if len(resolutions) == 1:
            return torch.from_numpy(grid[None]).float()

        if not (grid > float(mc_level)).any():
            # The coarse grid saw no interior at all: refining would propagate
            # nothing. Fall back to a dense decode at the final resolution so
            # genuinely small/thin objects still surface.
            xs, ys, zs = grid_axes(final_resolution)
            grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
            dense_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)
            dense_logits = self._query_points(
                dense_points,
                latents=latents,
                geo_decoder=geo_decoder,
                num_chunks=num_chunks,
                device=device,
                dtype=dtype,
            )
            return torch.from_numpy(
                dense_logits.numpy().reshape(
                    final_resolution + 1, final_resolution + 1, final_resolution + 1
                )[None]
            ).float()

        current_resolution = coarse_resolution
        for next_resolution in resolutions[1:]:
            level_value = float(mc_level)
            inside = grid > level_value
            # Surface cells: sign changes along any axis, dilated for safety.
            surface = np.zeros_like(inside)
            for axis in range(3):
                changed = np.diff(inside, axis=axis)
                pad_lo = [(0, 0)] * 3
                pad_hi = [(0, 0)] * 3
                pad_lo[axis] = (0, 1)
                pad_hi[axis] = (1, 0)
                surface |= np.pad(changed, pad_lo, mode="constant")
                surface |= np.pad(changed, pad_hi, mode="constant")
            # The |logit| band must scale with the FIELD, not carry a fixed
            # constant: unqueried cells inherit interpolated coarse values,
            # and on a hard-saturating field a thin gap whose straddling
            # samples all sit just outside a fixed band gets welded shut by
            # that fill (proven synthetically: 21k sign flips; the shipped
            # checkpoint clears the fixed 0.95 band by only 0.003 — one
            # retrain from silent failure). The sign-change shell measures
            # the near-surface logit scale directly; 1.5x that median keeps
            # every cell within ~1.5 cells of a crossing queried.
            band = self.band
            if surface.any():
                shell_scale = float(np.median(np.abs(grid - level_value)[surface]))
                band = max(self.band, 1.5 * shell_scale)
            surface |= np.abs(grid - level_value) < band
            surface = ndimage.binary_dilation(surface, iterations=2)

            scale = next_resolution // current_resolution
            fine_shape = (next_resolution + 1,) * 3
            # Mark fine cells whose parent coarse cell touches the surface.
            fine_mask = np.zeros(fine_shape, dtype=bool)
            coarse_idx = np.argwhere(surface)
            if len(coarse_idx) == 0:
                surface = np.ones_like(inside)
                coarse_idx = np.argwhere(surface)
            # Each coarse vertex spawns a (scale+1)^3 block of fine vertices.
            base = coarse_idx * scale
            block = np.stack(
                np.meshgrid(np.arange(scale + 1), np.arange(scale + 1), np.arange(scale + 1), indexing="ij"),
                axis=-1,
            ).reshape(-1, 3)
            fine_idx = (base[:, None, :] + block[None, :, :]).reshape(-1, 3)
            np.minimum(fine_idx, next_resolution, out=fine_idx)
            fine_mask[fine_idx[:, 0], fine_idx[:, 1], fine_idx[:, 2]] = True

            xs, ys, zs = grid_axes(next_resolution)
            refine_idx = np.argwhere(fine_mask)
            refine_points = np.stack(
                [xs[refine_idx[:, 0]], ys[refine_idx[:, 1]], zs[refine_idx[:, 2]]], axis=-1
            ).astype(np.float32)
            refine_logits = self._query_points(
                refine_points,
                latents=latents,
                geo_decoder=geo_decoder,
                num_chunks=num_chunks,
                device=device,
                dtype=dtype,
            ).numpy()

            # Unqueried cells take an interpolated estimate from the coarse
            # grid so marching cubes sees a smooth field far from the surface
            # instead of a sentinel value.
            zoom_factor = tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape))
            next_grid = ndimage.zoom(grid, zoom_factor, order=1, mode="nearest").astype(np.float32)
            next_grid[refine_idx[:, 0], refine_idx[:, 1], refine_idx[:, 2]] = refine_logits
            grid = next_grid
            current_resolution = next_resolution

        return torch.from_numpy(grid[None]).float()


def _resolve_generation_defaults(owner: Any, *, device: str) -> Dict[str, Any]:
    num_chunks_default = _DEFAULT_MPS_NUM_CHUNKS if device == "mps" else _DEFAULT_NUM_CHUNKS
    return {
        "num_inference_steps": _owner_cfg_int(owner, "scene3d_hunyuan_num_inference_steps", _DEFAULT_NUM_INFERENCE_STEPS),
        "guidance_scale": float(_owner_cfg(owner, "scene3d_hunyuan_guidance_scale") or _DEFAULT_GUIDANCE_SCALE),
        "octree_resolution": _owner_cfg_int(owner, "scene3d_hunyuan_octree_resolution", _DEFAULT_OCTREE_RESOLUTION),
        "num_chunks": _owner_cfg_int(owner, "scene3d_hunyuan_num_chunks", num_chunks_default),
        "max_facenum": _owner_cfg_int(owner, "scene3d_hunyuan_max_facenum", _DEFAULT_MAX_FACENUM),
    }


def _hunyuan_postprocess_mesh(
    mesh: Any,
    *,
    max_facenum: int,
) -> tuple[Any, List[str], List[str]]:
    """Official-style cleanup: floaters, degenerate faces, bounded face count.

    Hunyuan meshes come out much cleaner than marching-cubes-from-triplane
    meshes, so this intentionally avoids aggressive smoothing that would eat
    the sharp detail the model is good at.
    """
    applied: List[str] = []
    warnings: List[str] = []
    processed = mesh
    try:
        import trimesh

        if isinstance(processed, trimesh.Scene):
            geometries = [item for item in processed.geometry.values()]
            if geometries:
                processed = trimesh.util.concatenate(geometries)
                applied.append(f"concatenate_scene:{len(geometries)}")

        components = list(processed.split(only_watertight=False))
        if len(components) > 1:
            components = sorted(components, key=lambda item: (len(item.faces), float(item.area)), reverse=True)
            # Floater rule matched to upstream's (0.5% of TOTAL faces).
            # The previous 2%-of-largest was measured 3.2x more
            # aggressive: on an 844k-face car it would amputate genuine
            # detached parts in the 4k-13k face range (a side mirror is
            # exactly that size), which upstream keeps.
            total_faces = max(1, int(sum(len(item.faces) for item in components)))
            threshold = max(64, int(round(total_faces * 0.005)))
            kept = [item for item in components if len(item.faces) >= threshold]
            if not kept:
                kept = [components[0]]
            if len(kept) != len(components):
                applied.append(f"keep_significant_components:{len(components)}->{len(kept)}@{threshold}")
            processed = kept[0].copy() if len(kept) == 1 else trimesh.util.concatenate(kept)

        try:
            processed.update_faces(processed.nondegenerate_faces())
            applied.append("remove_degenerate_faces")
        except Exception as exc:
            warnings.append(f"Hunyuan3D degenerate-face removal skipped: {type(exc).__name__}: {exc}")

        if int(max_facenum) > 0 and len(processed.faces) > int(max_facenum):
            try:
                import pymeshlab

                ms = pymeshlab.MeshSet()
                ms.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=processed.vertices.astype("float64"),
                        face_matrix=processed.faces.astype("int32"),
                    ),
                    "hunyuan",
                )
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=int(max_facenum),
                    qualitythr=1.0,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                    autoclean=True,
                )
                decimated = ms.current_mesh()
                processed = trimesh.Trimesh(
                    vertices=decimated.vertex_matrix(),
                    faces=decimated.face_matrix(),
                    process=False,
                )
                applied.append(f"quadric_decimation:{int(max_facenum)}")
            except Exception as exc:
                warnings.append(f"Hunyuan3D face-count reduction skipped: {type(exc).__name__}: {exc}")

        try:
            processed.merge_vertices()
            applied.append("merge_vertices")
        except Exception as exc:
            warnings.append(f"Hunyuan3D merge_vertices skipped: {type(exc).__name__}: {exc}")

        try:
            trimesh.repair.fix_normals(processed)
            # Trimesh.invert preserves cached normals across its cache clear;
            # drop the cache so normals are recomputed from the fixed faces.
            try:
                processed._cache.clear()
            except Exception:
                pass
            _ = processed.face_normals
            _ = processed.vertex_normals
            applied.append("fix_normals")
        except Exception as exc:
            warnings.append(f"Hunyuan3D fix_normals skipped: {type(exc).__name__}: {exc}")

        if hasattr(processed, "remove_unreferenced_vertices"):
            processed.remove_unreferenced_vertices()
        return processed, applied, warnings
    except Exception as exc:
        warnings.append(f"Hunyuan3D postprocess skipped: {type(exc).__name__}: {exc}")
        return mesh, applied, warnings


def _hunyuan_canonicalize_axes(mesh: Any) -> tuple[Any, List[str]]:
    """Rotate the official Hunyuan output into this repo's render convention.

    Hunyuan3D-2.1 emits Y-up meshes with the subject facing +Z (glTF-style).
    The preview cameras and the texture projection math in this repository
    assume Z-up with the subject facing +X, matching the TripoSR backend, so
    exported and previewed meshes stay consistent across backends.
    """
    import numpy as np

    applied: List[str] = []
    try:
        rotation = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        mesh = mesh.copy()
        mesh.apply_transform(rotation)
        applied.append("yup_front_z_to_zup_front_x")
    except Exception:
        return mesh, applied
    return mesh, applied


def _mesh_topology(mesh: Any) -> Dict[str, Any]:
    topology: Dict[str, Any] = {}
    for key, attr in (("is_watertight", "is_watertight"), ("body_count", "body_count"), ("euler_number", "euler_number")):
        try:
            value = getattr(mesh, attr)
            topology[key] = bool(value) if key == "is_watertight" else int(value)
        except Exception:
            topology[key] = None
    return topology


class Hunyuan3DShapeBackend:
    """Official Hunyuan3D-2.1 shape backend (license-gated, geometry stage)."""

    backend_id = "abstract3d:hunyuan3d21-local"

    def __init__(self, owner: Any = None) -> None:
        self._owner = owner
        self._pipeline: Any = None
        self._resident_device: Optional[str] = None
        self._resident_dtype: Optional[str] = None
        self._last_runtime_stats: Dict[str, Any] = {}

    # -- capability surface -------------------------------------------------

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
        if not _license_accepted(self._owner):
            status = "license_acknowledgment_required"
        return [
            {
                "provider_id": "hunyuan3d21",
                "display_name": "Local Hunyuan3D-2.1 (license-gated)",
                "backend_id": self.backend_id,
                "tasks": tasks,
                "local": True,
                "status": status,
                "license": _LICENSE_NAME,
                "license_note": _LICENSE_SUMMARY,
            }
        ]

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from ..model_catalog import iter_model_specs

        selector = str(provider_id or provider or "").strip().lower()
        if selector and selector not in {"hunyuan3d21", "hunyuan3d", "hunyuan", self.backend_id}:
            return []
        rows: List[Dict[str, Any]] = []
        for spec in iter_model_specs(validated_only=False, task=task):
            if spec.provider_id != "hunyuan3d21":
                continue
            payload = spec.to_capability_model()
            payload["provider_id"] = "hunyuan3d21"
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
        del task
        parameter_schema = {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["glb", "obj", "zip"]},
                "num_inference_steps": {"type": "integer"},
                "guidance_scale": {"type": "number"},
                "octree_resolution": {"type": "integer"},
                "max_facenum": {"type": "integer"},
                "texture_mode": {"type": "string", "enum": ["baked_basecolor", "none"]},
                "texture_resolution": {"type": "integer"},
                "texture_completion": {"type": "string", "enum": ["none", "mirror_symmetry", "auto"]},
                "texture_reference_generation": {
                    "type": "string",
                    "enum": ["auto", "on", "off"],
                    "description": (
                        "Synthesize unseen-angle reference photos from the mesh's "
                        "clay renders when only one photo is provided. 'auto' "
                        "(default) fires only with an explicitly configured image "
                        "provider; generated views are plausible synthesis, not "
                        "ground truth. Person subjects are refused unless "
                        "texture_reference_allow_person is set."
                    ),
                },
                "texture_reference_allow_person": {
                    "type": "boolean",
                    "description": (
                        "Person-specific acknowledgment for reference generation: "
                        "no gate defends facial identity, so synthesizing views "
                        "of a person requires this explicit attestation (default "
                        "false; 'on' alone is not consent)."
                    ),
                },
                "device": {"type": "string"},
                "seed": {"type": "integer"},
            },
        }
        operations: List[Dict[str, Any]] = []
        if has_image_composer(self._owner):
            operations.append(
                {
                    "operation_id": "text_to_scene3d",
                    "task": "text_to_scene3d",
                    "input_modalities": ["text"],
                    "output_modalities": ["scene3d"],
                    "artifact_output": True,
                    "parameter_schema": parameter_schema,
                }
            )
        operations.append(
            {
                "operation_id": "image_to_scene3d",
                "task": "image_to_scene3d",
                "input_modalities": ["image"],
                "output_modalities": ["scene3d"],
                "artifact_output": True,
                "parameter_schema": parameter_schema,
            }
        )
        return operations

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        already_loaded = self._pipeline is not None
        self._load_runtime(
            model_id=request.get("model"),
            device=request.get("device"),
            dtype=request.get("dtype"),
            model_subfolder=request.get("subfolder"),
        )
        return {
            "task": str(request.get("task") or "scene3d_generation"),
            "provider": "hunyuan3d21",
            "model": f"{_OFFICIAL_MODEL_ID}/{_DIT_SUBFOLDER}",
            "backend_id": self.backend_id,
            "state": "loaded",
            "loaded": True,
            "loaded_new": not already_loaded,
            "details": dict(self._last_runtime_stats),
        }

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        del filters
        if self._pipeline is None:
            return []
        return [
            {
                "task": "scene3d_generation",
                "provider": "hunyuan3d21",
                "model": f"{_OFFICIAL_MODEL_ID}/{_DIT_SUBFOLDER}",
                "backend_id": self.backend_id,
                "state": "loaded",
                "loaded": True,
                "details": dict(self._last_runtime_stats),
            }
        ]

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        return self.list_loaded_models(filters)

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        self._clear_runtime()
        return {
            "task": "scene3d_generation",
            "provider": "hunyuan3d21",
            "model": f"{_OFFICIAL_MODEL_ID}/{_DIT_SUBFOLDER}",
            "backend_id": self.backend_id,
            "state": "unloaded",
            "unloaded": True,
        }

    # -- runtime ------------------------------------------------------------

    def _load_runtime(
        self,
        *,
        model_id: Optional[str],
        device: Optional[str],
        dtype: Optional[str],
        model_subfolder: Optional[str] = None,
    ) -> Any:
        # The legal gate outranks the dependency check: an operator who has not
        # acknowledged the license terms must see the license error first, even
        # on hosts where the optional runtime stack is not installed.
        _require_license_acceptance(self._owner)
        _require_runtime_dependencies()
        resolved_repo, resolved_subfolder = _resolve_model_selection(model_id, model_subfolder)
        resolved_device = _select_device(self._owner, device)
        resolved_dtype = _select_dtype(resolved_device, dtype)
        if (
            self._pipeline is not None
            and self._resident_device == resolved_device
            and self._resident_dtype == resolved_dtype
            and self._last_runtime_stats.get("model_id") == resolved_repo
            and self._last_runtime_stats.get("subfolder") == resolved_subfolder
        ):
            return self._pipeline

        torch = importlib.import_module("torch")
        source_dir = _resolve_source_dir(self._owner)
        weights_dir = _download_official_weights(
            self._owner, repo_id=resolved_repo, subfolder=resolved_subfolder
        )
        config_path = weights_dir / "config.yaml"
        if resolved_repo == _MV_MODEL_ID:
            config_path = _remap_mv_config(config_path, _cache_root(self._owner))
        load_started = time.perf_counter()
        with _sys_path(source_dir / "hy3dshape"):
            pipelines = importlib.import_module("hy3dshape.pipelines")
            pipeline = pipelines.Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                str(weights_dir / "model.fp16.ckpt"),
                str(config_path),
                device=resolved_device,
                dtype=getattr(torch, resolved_dtype),
                use_safetensors=False,
            )
        load_s = round(time.perf_counter() - load_started, 4)
        self._pipeline = pipeline
        self._resident_device = resolved_device
        self._resident_dtype = resolved_dtype
        self._last_runtime_stats = {
            "load_s": load_s,
            "model_id": resolved_repo,
            "subfolder": resolved_subfolder,
            "multiview_capable": resolved_repo == _MV_MODEL_ID,
            "source_dir": str(source_dir),
            "weights_dir": str(weights_dir),
            "device": resolved_device,
            "dtype": resolved_dtype,
        }
        return pipeline

    def _clear_runtime(self) -> None:
        self._pipeline = None
        self._resident_device = None
        self._resident_dtype = None
        try:
            import gc

            import torch

            gc.collect()
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _make_source_image(self, prompt: str, **kwargs: Any) -> bytes:
        if not has_image_composer(self._owner):
            raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT)
        generator = _default_image_generator(self._owner)
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

    # -- generation ---------------------------------------------------------

    def _run_generation(
        self,
        *,
        task: str,
        prompt: str,
        image: Optional[Any],
        format: str,
        artifact_store: Optional[Any],
        output_dir: Optional[str],
        remove_background: Optional[bool],
        device: Optional[str],
        model: Optional[str],
        **kwargs: Any,
    ):
        from PIL import Image
        import psutil
        import torch

        actual_task = _TASK_ALIASES.get(task, task)
        if actual_task not in {"text_to_scene3d", "image_to_scene3d"}:
            raise CapabilityNotSupportedError(f"Unsupported Hunyuan3D task: {actual_task!r}")
        _require_license_acceptance(self._owner)

        image_generation_s: Optional[float] = None
        if actual_task == "text_to_scene3d":
            from ..image_composition import pop_composition_kwargs

            image_started = time.perf_counter()
            image_bytes = self._make_source_image(prompt, **pop_composition_kwargs(kwargs))
            image_generation_s = round(time.perf_counter() - image_started, 4)
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            # Free MLX buffers before loading the 3.3B DiT on the same
            # unified-memory pool.
            from .step1x_runtime import _release_mlx_generation_cache

            _release_mlx_generation_cache()
        else:
            if image is None:
                raise ValueError("image_to_scene3d requires an image input.")
            source_image = _load_image_payload(image, artifact_store=artifact_store).convert("RGBA")

        dtype = kwargs.pop("dtype", None)
        model_subfolder = kwargs.pop("model_subfolder", None)
        pipeline = self._load_runtime(
            model_id=model, device=device, dtype=dtype, model_subfolder=model_subfolder
        )
        resolved_device = self._resident_device or "cpu"
        resolved_dtype = self._resident_dtype or "float32"
        multiview_capable = bool(self._last_runtime_stats.get("multiview_capable"))

        # The official preprocessing recenters on a white background using the
        # alpha channel. When the caller did not provide alpha, segment the
        # subject so the conditioner sees it and not the background. The
        # robust path prefers the isnet checkpoint and cleans the matte:
        # the default u2net checkpoint amputates low-contrast subject
        # regions (dark hair on light backgrounds), which corrupts the
        # canonical framing every downstream stage depends on.
        preprocess_started = time.perf_counter()
        original_preview = source_image.convert("RGB")
        alpha = source_image.getchannel("A")
        has_alpha = alpha.getextrema()[0] < 255
        background_removed = bool(has_alpha)
        from ..segmentation import clean_alpha_mask, remove_background_robust

        if has_alpha:
            source_image = clean_alpha_mask(source_image)
        elif remove_background is not False:
            try:
                source_image = remove_background_robust(source_image)
                background_removed = True
            except Exception as exc:
                if remove_background is True:
                    raise DependencyUnavailableError(
                        f"Background removal requested but rembg is unavailable: {type(exc).__name__}: {exc}"
                    ) from exc

        # Reference views are loaded up front because they serve two stages:
        # multi-view geometry conditioning (2mv checkpoints) and the texture
        # bake. Each opaque reference gets background removal so both
        # consumers see clean alpha.
        reference_views = _tripo_normalize_texture_reference_views(
            raw_views=kwargs.pop("texture_reference_views", None),
            raw_images=kwargs.pop("texture_reference_images", None),
            raw_angles=kwargs.pop("texture_reference_angles", None),
        )
        loaded_references: List[Dict[str, Any]] = []
        reference_warnings: List[str] = []
        for index, reference in enumerate(reference_views, start=1):
            try:
                loaded = _load_image_payload(reference["image"], artifact_store=artifact_store).convert("RGBA")
                if loaded.getchannel("A").getextrema()[0] >= 255:
                    try:
                        loaded = remove_background_robust(loaded)
                    except Exception:
                        pass
                else:
                    loaded = clean_alpha_mask(loaded)
                loaded_references.append(
                    {
                        "rgba": loaded,
                        "azimuth_deg": float(reference.get("azimuth_deg", 0.0)),
                        "elevation_deg": float(reference.get("elevation_deg", 0.0)),
                        "label": str(reference.get("label") or f"reference_{index:02d}"),
                        "role": "reference",
                    }
                )
            except Exception as exc:
                reference_warnings.append(
                    f"Hunyuan3D reference view {index} skipped: {type(exc).__name__}: {exc}"
                )

        # Multi-view geometry conditioning: map references whose azimuths
        # snap to the four canonical 2mv tags. The checkpoint was trained on
        # exactly these viewpoints; off-tag references still contribute
        # through the texture bake.
        geometry_condition: Any = source_image
        geometry_views_used: List[Dict[str, Any]] = [{"tag": "front", "label": "source"}]
        if multiview_capable and loaded_references:
            image_dict: Dict[str, Any] = {"front": source_image}
            for reference in loaded_references:
                azimuth = float(reference["azimuth_deg"])
                for tag_azimuth, tag in _MV_TAG_BY_AZIMUTH.items():
                    delta = abs(((azimuth - tag_azimuth) + 180.0) % 360.0 - 180.0)
                    if delta <= _MV_TAG_SNAP_TOLERANCE_DEG and tag not in image_dict:
                        image_dict[tag] = reference["rgba"]
                        geometry_views_used.append(
                            {
                                "tag": tag,
                                "label": reference["label"],
                                "declared_azimuth_deg": azimuth,
                                "snap_delta_deg": round(delta, 2),
                            }
                        )
                        break
            if len(image_dict) > 1:
                geometry_condition = image_dict
        preprocess_s = round(time.perf_counter() - preprocess_started, 4)

        defaults = _resolve_generation_defaults(self._owner, device=resolved_device)

        def _pop_number(key: str, default: Any, cast: Any) -> Any:
            # `None` means "use the default"; explicit falsy values such as
            # guidance_scale=0.0 or seed=0 are legitimate and must survive.
            value = kwargs.pop(key, None)
            return cast(default) if value is None else cast(value)

        num_inference_steps = max(1, _pop_number("num_inference_steps", defaults["num_inference_steps"], int))
        guidance_scale = _pop_number("guidance_scale", defaults["guidance_scale"], float)
        # The generic CLI exposes --mc-resolution; for this backend the grid
        # resolution knob is the octree resolution, so honor the alias when
        # no explicit octree_resolution was given (mirrors the Step1X CLI
        # contract).
        mc_resolution_alias = kwargs.pop("mc_resolution", None)
        if kwargs.get("octree_resolution") is None and mc_resolution_alias is not None:
            kwargs["octree_resolution"] = mc_resolution_alias
        octree_resolution = max(32, _pop_number("octree_resolution", defaults["octree_resolution"], int))
        num_chunks = max(1024, _pop_number("num_chunks", defaults["num_chunks"], int))
        max_facenum = _pop_number("max_facenum", defaults["max_facenum"], int)
        seed_value = kwargs.pop("seed", None)
        if seed_value is None:
            seed_value = kwargs.pop("image_seed", None)
        else:
            kwargs.pop("image_seed", None)
        seed = _DEFAULT_SEED if seed_value is None else int(seed_value)
        volume_decoder_mode = str(
            kwargs.pop("volume_decoder", None)
            or _owner_cfg(self._owner, "scene3d_hunyuan_volume_decoder")
            or _env("ABSTRACT3D_HUNYUAN_VOLUME_DECODER")
            or "hierarchical"
        ).strip().lower()
        # Texture options are consumed here (values are used post-inference)
        # so the strict unknown-option check can run BEFORE the multi-minute
        # diffusion stage: a typo must fail in milliseconds, not after 10
        # minutes of inference.
        texture_mode = str(
            kwargs.pop("texture_mode", None)
            or _owner_cfg(self._owner, "scene3d_hunyuan_texture_mode")
            or _env("ABSTRACT3D_HUNYUAN_TEXTURE_MODE")
            or "baked_basecolor"
        ).strip().lower()
        texture_resolution = int(kwargs.pop("texture_resolution", None) or 2048)
        # Default "auto": a single-photo bake observes a thin sliver of the
        # subject, and for measurably symmetric geometry (score-gated inside
        # the bake) the mirrored twin of that sliver is real content where
        # a propagated fill is a characterless wash. Explicit "none" or
        # "mirror_symmetry" continue to force their modes.
        texture_completion = str(kwargs.pop("texture_completion", None) or "auto").strip().lower()
        # Generated reference views: when the caller supplies only the one
        # photo, synthesize the unseen angles from the mesh's own clay
        # renders through the configured i2i provider ("auto" default;
        # see reference_generation.py). "off" disables; "on" makes a
        # missing image composer a hard error instead of a warning.
        texture_reference_generation = str(
            kwargs.pop("texture_reference_generation", None)
            or _owner_cfg(self._owner, "scene3d_texture_reference_generation")
            or _env("ABSTRACT3D_TEXTURE_REFERENCE_GENERATION")
            or "auto"
        ).strip().lower()
        if texture_reference_generation not in {"auto", "on", "off"}:
            raise InvalidRequestError(
                "texture_reference_generation must be one of: auto, on, off "
                f"(got {texture_reference_generation!r})"
            )
        from ..reference_generation import parse_generation_angles

        try:
            texture_reference_generation_angles = parse_generation_angles(
                kwargs.pop("texture_reference_generation_angles", None)
                or _owner_cfg(self._owner, "scene3d_texture_reference_generation_angles")
                or _env("ABSTRACT3D_TEXTURE_REFERENCE_GENERATION_ANGLES")
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        # Person-specific acknowledgment: "on" is texture-quality opt-in,
        # not identity-synthesis consent. Synthesizing views of a person
        # requires this separate, explicit attestation.
        texture_reference_allow_person = _as_bool(
            kwargs.pop("texture_reference_allow_person", None)
            if "texture_reference_allow_person" in kwargs
            else _owner_cfg(self._owner, "scene3d_texture_reference_allow_person")
            or _env("ABSTRACT3D_TEXTURE_REFERENCE_ALLOW_PERSON")
        )
        from . import reject_unknown_options

        reject_unknown_options(self.backend_id, kwargs)

        source_dir = Path(self._last_runtime_stats["source_dir"])
        inference_started = time.perf_counter()
        with _sys_path(source_dir / "hy3dshape"):
            volume_decoders = importlib.import_module("hy3dshape.models.autoencoders.volume_decoders")
            if volume_decoder_mode == "vanilla":
                pipeline.vae.volume_decoder = volume_decoders.VanillaVolumeDecoder()
            elif volume_decoder_mode == "hierarchical_upstream":
                pipeline.vae.volume_decoder = volume_decoders.HierarchicalVolumeDecoding()
            else:
                # Default: coarse-to-fine decoder with host-side bookkeeping.
                # The upstream hierarchical decoder starts too coarse for thin
                # structures and its scatter path is unreliable on MPS.
                volume_decoder_mode = "adaptive"
                pipeline.vae.volume_decoder = _AdaptiveVolumeDecoder()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                meshes = pipeline(
                    image=geometry_condition,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    mc_algo="mc",
                    generator=generator,
                    output_type="trimesh",
                    enable_pbar=False,
                )
        inference_s = round(time.perf_counter() - inference_started, 4)
        raw_mesh = meshes[0] if isinstance(meshes, list) else meshes
        if raw_mesh is None:
            raise Abstract3DError(
                "Hunyuan3D-2.1 produced no surface at the requested settings. "
                "Try more inference steps or a different seed."
            )

        mesh_started = time.perf_counter()
        mesh, postprocess_applied, postprocess_warnings = _hunyuan_postprocess_mesh(
            raw_mesh,
            max_facenum=max_facenum,
        )
        # Raw (pre-cleanup) topology: without it a surfacing regression is
        # invisible — the shipped numbers conflate decoder output with
        # floater removal and decimation (measured: raw euler -110 ->
        # post -124 on a car, fully explained by junk removal).
        topology_raw = _mesh_topology(raw_mesh)
        canonicalize = _owner_cfg_bool(self._owner, "scene3d_hunyuan_canonicalize_export_axes", True)
        axis_applied: List[str] = []
        if canonicalize:
            mesh, axis_applied = _hunyuan_canonicalize_axes(mesh)
        mesh_s = round(time.perf_counter() - mesh_started, 4)

        keep_resident = _owner_cfg_bool(self._owner, "scene3d_hunyuan_keep_resident", False)
        if not keep_resident and resolved_device == "mps":
            # Free ~7 GB of accelerator memory before texture bake and preview.
            self._clear_runtime()

        texture_requested = texture_mode == "baked_basecolor"
        texture_s: Optional[float] = None
        texture_stats: Dict[str, Any] = {}
        obj_texture_sidecars: Dict[str, bytes] = {}
        export_mesh = mesh
        quality_verdict: Dict[str, Any] = {"verdict": "healthy", "reasons": []}
        postprocess_warnings.extend(reference_warnings)
        if texture_requested:
            from ..texturing import bake_projection_texture

            texture_started = time.perf_counter()
            observed_views: List[Dict[str, Any]] = [
                {
                    "rgba": source_image.convert("RGBA"),
                    "azimuth_deg": 0.0,
                    "elevation_deg": 0.0,
                    "label": "front",
                    "role": "source",
                    # The un-matted photo: multi-view bakes build the identity
                    # correspondence against it (the matted rgba's tighter
                    # silhouette snaps the registration into a different
                    # basin — see the fringe-repair stage in texturing.py).
                    "identity_image": original_preview,
                }
            ]
            observed_views.extend(dict(reference) for reference in loaded_references)

            # Single-photo runs: synthesize the unseen angles from the mesh's
            # own clay renders (silhouette-locked i2i; IoU-gated; tone-matched)
            # so the bake witnesses the whole surface instead of surrendering
            # ~70% of it to mirror completion and fill.
            reference_generation_report: Optional[Dict[str, Any]] = None
            if (
                texture_reference_generation in {"auto", "on"}
                and not loaded_references
            ):
                from ..reference_generation import (
                    auto_generation_ready,
                    generate_reference_views,
                )

                subject_hint = str(prompt or "").strip() or None
                ready, readiness_reason = auto_generation_ready(self._owner)
                if texture_reference_generation == "auto" and not ready:
                    # "auto" is a default: it must never silently go remote.
                    # No subject prompt is needed — the pipeline captions the
                    # source photo itself.
                    postprocess_warnings.append(
                        "texture_reference_generation=auto skipped: "
                        f"{readiness_reason}. A single photo witnesses only part "
                        "of the surface; configure a local image provider "
                        "(scene3d_image_provider / ABSTRACT3D_IMAGE_PROVIDER) "
                        "or set texture_reference_generation=on to synthesize "
                        "the unseen angles."
                    )
                elif texture_reference_generation == "on" and not has_image_composer(self._owner):
                    raise DependencyUnavailableError(
                        "texture_reference_generation=on requires an image "
                        f"composer. {COMPOSITION_INSTALL_HINT}"
                    )
                else:
                    from ..image_composition import resolve_image_generation_request

                    try:
                        from ..reference_generation import DEFAULT_ANGLES

                        generated_views, reference_generation_report = generate_reference_views(
                            mesh,
                            source_image.convert("RGBA"),
                            owner=self._owner,
                            angles=texture_reference_generation_angles or DEFAULT_ANGLES,
                            subject_hint=subject_hint,
                            seed=seed,
                            image_request=resolve_image_generation_request(self._owner),
                            # Person subjects are refused in BOTH modes
                            # unless the person-specific acknowledgment is
                            # given: "on" is texture-quality opt-in, not
                            # identity-synthesis consent.
                            person_policy=(
                                "proceed"
                                if texture_reference_allow_person
                                else "skip"
                            ),
                        )
                        observed_views.extend(generated_views)
                        if not generated_views:
                            skip_reason = (
                                reference_generation_report or {}
                            ).get("skipped")
                            postprocess_warnings.append(
                                "texture_reference_generation skipped: "
                                f"{skip_reason}"
                                if skip_reason
                                else "texture_reference_generation produced no "
                                "accepted views (all angles rejected or "
                                "errored); continuing with the single photo. "
                                "See texture_artifacts.reference_generation "
                                "for per-angle reasons."
                            )
                        # Free the image model's accelerator pool before the
                        # 2 GB bake allocates its atlas stacks.
                        from .step1x_runtime import _release_mlx_generation_cache

                        _release_mlx_generation_cache()
                    except Exception as exc:
                        if texture_reference_generation == "on":
                            raise
                        postprocess_warnings.append(
                            "texture_reference_generation=auto failed, continuing "
                            f"with the single photo: {type(exc).__name__}: {exc}"
                        )
            try:
                bake_kwargs = dict(
                    texture_resolution=texture_resolution,
                    texture_completion=texture_completion,
                    # Hunyuan reconstructs in a canonical orthographic frame
                    # from a deterministically recentered conditioning image
                    # (ImageProcessorV2.recenter, border_ratio 0.15). The
                    # bake replicates that exact frame per view, which makes
                    # photo-to-mesh registration deterministic instead of an
                    # estimation problem.
                    projection_model="orthographic",
                    canonical_border_ratio=0.15,
                )
                # Snapshot real views before the candidate bake mutates them
                # in place (registration replaces view["rgba"]): the A/B
                # baseline must start from the same pristine inputs.
                generated_present = any(
                    view.get("generated") for view in observed_views)
                baseline_views = (
                    [dict(view) for view in observed_views
                     if not view.get("generated")]
                    if generated_present else None
                )
                export_mesh, texture_stats = bake_projection_texture(
                    mesh, observed_views=observed_views, **bake_kwargs)
                if generated_present and baseline_views:
                    # Whole-bake acceptance: per-view gates cannot see
                    # composition-level failure (handoff seams, overall
                    # darkening); ship the generated bake only if it does
                    # not regress the no-references baseline.
                    from ..bake_acceptance import evaluate_generated_bake

                    baseline_mesh_bake, baseline_stats = bake_projection_texture(
                        mesh.copy(), observed_views=baseline_views, **bake_kwargs)
                    # The gate resolves the fidelity pose from the
                    # baseline bake's own stats (a hardcoded (0,0) on a
                    # pose-estimated subject charges ~9 dE of pure pose
                    # error to both sides: measured +0.88 true regression
                    # read as +4.03 on the az-17.5 car). Declared-pose
                    # subjects are unaffected — (0,0) IS their recorded
                    # pose.
                    verdict = evaluate_generated_bake(
                        baseline_mesh_bake,
                        export_mesh,
                        source_rgba=baseline_views[0]["rgba"],
                        baseline_stats=baseline_stats,
                        candidate_stats=texture_stats,
                    )
                    if reference_generation_report is not None:
                        reference_generation_report["bake_acceptance"] = verdict
                    if not verdict["accepted"]:
                        export_mesh, texture_stats = (
                            baseline_mesh_bake, baseline_stats)
                        postprocess_warnings.append(
                            "texture_reference_generation: generated bake "
                            "regressed the no-references baseline and was "
                            "refused by the whole-bake acceptance gate; "
                            "shipping the baseline. Reasons: "
                            + "; ".join(verdict["reasons"])
                        )
                        # The refused-candidate branch ships the BASELINE
                        # — which is a single-photo bake that never met
                        # the A/B machinery and can be broken on its own
                        # (measured, integrator program: a pose-guard
                        # dead zone shipped a coverage-0.058 baseline
                        # here with verdict "healthy" while the sanity
                        # floors existed one branch below). Same
                        # self-healing contract as the no-references
                        # branch: mark it loudly.
                        from ..bake_acceptance import evaluate_single_view_bake

                        single_verdict = evaluate_single_view_bake(texture_stats)
                        texture_stats["single_view_sanity"] = single_verdict
                        if not single_verdict["accepted"]:
                            quality_verdict = {
                                "verdict": "degraded",
                                "reasons": single_verdict["reasons"],
                            }
                            postprocess_warnings.append(
                                "shipped baseline failed the single-view "
                                "sanity floors (quality_verdict=degraded): "
                                + "; ".join(single_verdict["reasons"])
                            )
                else:
                    # SINGLE-VIEW SANITY (self-healing contract): a bake
                    # with no accepted generated views ships ungated by
                    # the A/B machinery, and the measured car incident
                    # proved it can be broken on its own (mis-posed
                    # source, coverage 0.055, exit 0). Absolute floors
                    # calibrated on the healthy fleet mark it loudly.
                    from ..bake_acceptance import evaluate_single_view_bake

                    single_verdict = evaluate_single_view_bake(texture_stats)
                    texture_stats["single_view_sanity"] = single_verdict
                    if not single_verdict["accepted"]:
                        quality_verdict = {
                            "verdict": "degraded",
                            "reasons": single_verdict["reasons"],
                        }
                        postprocess_warnings.append(
                            "single-view bake failed the sanity floors "
                            "(shipping with quality_verdict=degraded): "
                            + "; ".join(single_verdict["reasons"])
                        )
            except Exception as exc:
                texture_requested = False
                texture_stats = {}
                quality_verdict = {
                    "verdict": "failed",
                    "reasons": [f"texture bake raised {type(exc).__name__}: {exc}"],
                }
                postprocess_warnings.append(
                    f"Hunyuan3D texture bake failed, exporting geometry only: {type(exc).__name__}: {exc}"
                )
            texture_s = round(time.perf_counter() - texture_started, 4)

        # Mesh-aware verdict: the measured v5 car shipped "healthy" with
        # four wheels floating off the body — the verdict only ever looked
        # at texture. Disconnected bodies on a single-subject generation
        # mean the field split parts across sub-resolution gaps (measured:
        # wheel-to-arch air channels of 1.4-1.9 grid cells); the mesh is
        # usable but not sound, so it ships as degraded with the reason.
        # Genus is deliberately NOT gated: high genus is legitimate for
        # many subjects (a spoked wheel alone is genus ~10; an accepted
        # face proof measures genus 210).
        mesh_topology = _mesh_topology(mesh)
        body_count = int(mesh_topology.get("body_count") or 1)
        if body_count > 1:
            reason = (
                f"mesh has {body_count} disconnected bodies after floater "
                "removal: the shape field split parts across sub-resolution "
                "gaps (single-subject generations should be one connected "
                "body; consider multi-view geometry conditioning or a "
                "closed-form input view)"
            )
            if quality_verdict.get("verdict") == "healthy":
                quality_verdict = {"verdict": "degraded", "reasons": [reason]}
            else:
                quality_verdict.setdefault("reasons", []).append(reason)
            postprocess_warnings.append(reason)

        glb_bytes = _mesh_export_bytes(export_mesh, file_type="glb")
        if texture_requested:
            obj_bytes, obj_texture_sidecars = _tripo_export_obj_with_textures(export_mesh)
        else:
            obj_bytes = _mesh_export_bytes(export_mesh, file_type="obj")
        views = render_mesh_views(export_mesh)
        from ..rendering import get_last_render_backend

        primary_format = str(format or "glb").strip().lower() or "glb"
        if primary_format not in {"glb", "obj", "zip"}:
            raise ValueError("scene3d format must be one of: glb, obj, zip")
        primary_bytes = glb_bytes if primary_format == "glb" else obj_bytes
        content_type = "model/gltf-binary" if primary_format == "glb" else "text/plain"
        if primary_format == "zip":
            content_type = "application/zip"
        total_s = round(
            (image_generation_s or 0.0) + preprocess_s + inference_s + mesh_s + (texture_s or 0.0),
            4,
        )
        process = psutil.Process(os.getpid())
        resolved_repo = str(self._last_runtime_stats.get("model_id") or _OFFICIAL_MODEL_ID)
        resolved_subfolder = str(self._last_runtime_stats.get("subfolder") or _DIT_SUBFOLDER)
        runtime_meta: Dict[str, Any] = {
            "backend_id": self.backend_id,
            "provider": self.backend_id,
            "model_id": f"{resolved_repo}/{resolved_subfolder}",
            "task": actual_task,
            "multiview_conditioning": bool(multiview_capable and len(geometry_views_used) > 1),
            "geometry_views": geometry_views_used,
            "device": resolved_device,
            "dtype": resolved_dtype,
            "format": primary_format,
            "content_type": content_type,
            "license": _LICENSE_NAME,
            "license_note": _LICENSE_SUMMARY,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution,
            "num_chunks": num_chunks,
            "volume_decoder": volume_decoder_mode,
            "seed": seed,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "appearance_mode": "uv_basecolor" if texture_requested else "geometry_only",
            "texture_mode": texture_mode if texture_requested else "none",
            "texture_resolution": int(texture_resolution) if texture_requested else None,
            "texture_completion": texture_completion if texture_requested else None,
            "uv_present": bool(texture_requested),
            "material_count": 1 if texture_requested else 0,
            "preview_renderer": get_last_render_backend(),
            "background_removed": background_removed,
            "source_snapshot": _HUNYUAN_COMMIT,
            "postprocess_cleanup": postprocess_applied,
            "postprocess_warnings": postprocess_warnings,
            # Machine-readable health: "healthy" / "degraded" / "failed"
            # with reasons. The measured car incident shipped a broken
            # texture with exit 0 and stdout-only warnings — unattended
            # callers need one field to check (the CLI exits 3 on it).
            "quality_verdict": quality_verdict,
            "export_axis_canonicalization": axis_applied,
            "topology": mesh_topology,
            "topology_raw": topology_raw,
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
                "mps_allocated_bytes": int(torch.mps.current_allocated_memory())
                if (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
                else None,
            },
            "notes": [
                "Hunyuan3D-2.1 official shape stage (flow-matching DiT + shape VAE + marching cubes).",
                _LICENSE_SUMMARY,
            ],
        }
        if texture_requested and texture_stats:
            runtime_meta["texture_artifacts"] = {
                "texture_padding": texture_stats.get("texture_padding"),
                "projection_mode": texture_stats.get("projection_mode"),
                "unseen_fill_mode": texture_stats.get("unseen_fill_mode"),
                "camera_distance": texture_stats.get("camera_distance"),
                "source_pose": dict(texture_stats.get("source_pose") or {}),
                "source_registration": dict(texture_stats.get("source_registration") or {}),
                "observed_coverage_ratio": texture_stats.get("observed_coverage_ratio"),
                "observed_view_stats": list(texture_stats.get("observed_view_stats") or []),
                "texture_completion": texture_stats.get("texture_completion"),
                "symmetry_completion": dict(texture_stats.get("symmetry_completion") or {}),
                "reference_view_count": max(0, len(reference_views)),
                "uv_vertex_count": texture_stats.get("uv_vertex_count"),
                "vertex_mapping_count": texture_stats.get("vertex_mapping_count"),
                "obj_sidecars": sorted(str(name) for name in obj_texture_sidecars.keys()),
            }
            if reference_generation_report is not None:
                # The report carries PIL objects under "rejected_images"
                # (persisted separately to rejected_refs/) — they must not
                # reach JSON serialization.
                runtime_meta["texture_artifacts"]["reference_generation"] = {
                    key: value
                    for key, value in reference_generation_report.items()
                    if key != "rejected_images"
                }

        # Keep the contact sheet's source panel on the original background so
        # reviewers compare against what the caller actually provided.
        source_preview = original_preview
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
            if texture_requested and texture_stats:
                geometry_glb_path = bundle_root / "geometry.glb"
                geometry_glb_path.write_bytes(_mesh_export_bytes(mesh, file_type="glb", viewer_frame=False))
                texture_path = bundle_root / "texture.png"
                texture_stats["texture_image"].save(texture_path)
                uv_preview_path = bundle_root / "uv_preview.png"
                texture_stats["uv_preview"].save(uv_preview_path)
                for sidecar_name, sidecar_bytes in obj_texture_sidecars.items():
                    (bundle_root / str(sidecar_name)).write_bytes(sidecar_bytes)
                runtime_meta["texture_artifacts"].update(
                    {
                        "geometry_glb_path": str(geometry_glb_path),
                        "texture_path": str(texture_path),
                        "uv_preview_path": str(uv_preview_path),
                    }
                )
                # Persist generated reference views (and the clay renders
                # that conditioned them) so the synthesis is auditable and
                # the bundle can be rebaked from the same witnesses.
                generated_paths: List[str] = []
                for view in observed_views:
                    if not view.get("generated"):
                        continue
                    label = str(view.get("label") or "generated")
                    generated_path = bundle_root / f"texture_reference_generated_{label}.png"
                    view["rgba"].save(generated_path)
                    generated_paths.append(str(generated_path))
                    clay = view.get("clay_render")
                    if clay is not None:
                        clay.save(bundle_root / f"texture_reference_generated_{label}_clay.png")
                if generated_paths:
                    runtime_meta["texture_artifacts"]["generated_reference_paths"] = generated_paths
                # Persist-for-diagnosis: downscaled copies of REJECTED
                # candidates (budget-capped) — a rejected class must be
                # diagnosable without a rerun.
                rejected_rows = (
                    (reference_generation_report or {}).pop("rejected_images", None))
                if rejected_rows:
                    rejected_dir = bundle_root / "rejected_refs"
                    rejected_dir.mkdir(exist_ok=True)
                    budget = 2 * 1024 * 1024
                    for row in rejected_rows:
                        if budget <= 0:
                            break
                        rejected_path = rejected_dir / f"{row['label']}_a{row['attempt']}.webp"
                        try:
                            row["image"].convert("RGB").save(
                                rejected_path, format="WEBP", quality=80)
                            budget -= rejected_path.stat().st_size
                        except Exception:
                            break
            runtime_meta["bundle_dir"] = str(bundle_root)
            for meta_key, bundle_key in (
                ("preview_path", "preview_path"),
                ("contact_sheet_path", "contact_sheet_path"),
                ("metadata_path", "metadata_path"),
                ("source_image_path", "source_path"),
            ):
                value = bundle_paths.get(bundle_key)
                runtime_meta[meta_key] = str(value) if value else None
        if primary_format == "zip":
            if bundle_root is None:
                bundle_root = Path(tempfile.mkdtemp(prefix="abstract3d-hunyuan3d-bundle-"))
                bundle_paths = _write_bundle(
                    root_dir=bundle_root,
                    primary_format="glb",
                    primary_bytes=glb_bytes,
                    obj_bytes=obj_bytes,
                    source_image=source_preview,
                    prompt=prompt,
                    metadata=runtime_meta,
                    view_images=views,
                )
                if texture_requested and texture_stats:
                    (bundle_root / "geometry.glb").write_bytes(_mesh_export_bytes(mesh, file_type="glb", viewer_frame=False))
                    texture_stats["texture_image"].save(bundle_root / "texture.png")
                    texture_stats["uv_preview"].save(bundle_root / "uv_preview.png")
                    for sidecar_name, sidecar_bytes in obj_texture_sidecars.items():
                        (bundle_root / str(sidecar_name)).write_bytes(sidecar_bytes)
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

    # -- public API ----------------------------------------------------------

    def t23d(self, prompt: str, **kwargs: Any):
        return self._run_generation(
            task="text_to_scene3d",
            prompt=str(prompt or ""),
            image=None,
            format=kwargs.pop("format", "glb"),
            artifact_store=kwargs.pop("artifact_store", None),
            output_dir=kwargs.pop("output_dir", None),
            remove_background=kwargs.pop("remove_background", None),
            device=kwargs.pop("device", None),
            model=kwargs.pop("model", None),
            **kwargs,
        )

    def i23d(self, image: Any, **kwargs: Any):
        return self._run_generation(
            task="image_to_scene3d",
            prompt=str(kwargs.pop("prompt", "") or ""),
            image=image,
            format=kwargs.pop("format", "glb"),
            artifact_store=kwargs.pop("artifact_store", None),
            output_dir=kwargs.pop("output_dir", None),
            remove_background=kwargs.pop("remove_background", None),
            device=kwargs.pop("device", None),
            model=kwargs.pop("model", None),
            **kwargs,
        )

    def generate(self, prompt: str = "", *, task: str = "text_to_scene3d", **kwargs: Any):
        actual_task = _TASK_ALIASES.get(task, task)
        if actual_task == "image_to_scene3d":
            image = kwargs.pop("image", None)
            return self.i23d(image, prompt=prompt, **kwargs)
        return self.t23d(prompt, **kwargs)

    def validate_suite(
        self,
        *,
        prompts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        output_dir: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        sheets: List[str] = []
        index = 0
        for prompt in list(prompts or []):
            index += 1
            case_dir = output_root / f"{index:02d}_t23d"
            result = self.t23d(prompt, output_dir=str(case_dir), **kwargs)
            metadata = dict(result.get("metadata") or {})
            rows.append({"case_id": case_dir.name, "mode": "t23d", "prompt": prompt, **metadata})
            if metadata.get("contact_sheet_path"):
                sheets.append(str(metadata["contact_sheet_path"]))
        for image_path in list(images or []):
            index += 1
            case_dir = output_root / f"{index:02d}_i23d"
            result = self.i23d(image_path, output_dir=str(case_dir), **kwargs)
            metadata = dict(result.get("metadata") or {})
            rows.append({"case_id": case_dir.name, "mode": "i23d", "image": image_path, **metadata})
            if metadata.get("contact_sheet_path"):
                sheets.append(str(metadata["contact_sheet_path"]))
        summary_dir = output_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        stats_path = summary_dir / "stats.json"
        stats_path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
        contact_sheet_path = summary_dir / "contact_sheet.png"
        try:
            from PIL import Image

            sheet_images = [Image.open(path) for path in sheets]
            if sheet_images:
                stack_contact_sheets(sheet_images).save(contact_sheet_path)
        except Exception:
            contact_sheet_path = Path(sheets[0]) if sheets else contact_sheet_path
        return {
            "cases": rows,
            "stats": str(stats_path),
            "contact_sheet": str(contact_sheet_path),
        }
