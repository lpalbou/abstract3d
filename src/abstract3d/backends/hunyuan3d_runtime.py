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
# HARD CAP on simultaneous 2mv conditioning views, with priority order.
# Measured (2026-07-14, /tmp/afix3/subsets): on identical conditioning
# images, seed and regime (384/30, adaptive decoder, mps fp16), every
# 1-3 view subset produced a healthy single-body car (raw euler -65..-174;
# 2-3 view subsets BETTER than single view) while the full 4-view dict
# shredded the field into film-shell debris TWICE independently (559 raw
# bodies / euler +693 at 384/30; 822 raw bodies / +887 at 512/50). Every
# proven upstream usage stays at <= 3 views (official snippet
# {front, left, back}; this repo's face proof {front, left, right}), so
# 4-view conditioning is outside the checkpoint's validated envelope and
# is refused: the LAST tag in priority order is dropped with a warning.
_MV_MAX_CONDITIONING_VIEWS = 3
_MV_TAG_PRIORITY = ("front", "back", "left", "right")
# -- multi-view geometry conditioning for single-photo flows ------------------
#
# Both mesh audits (2026-07: /tmp/mesh1 §8/§9, /tmp/mesh3 §6) convicted
# conditioning starvation as the primary cause of hallucinated geometry on
# self-occluding subjects (melted interiors, detached wheels) and ranked
# "feed the shape stage more views" as the first fix. The 2mv dict path
# below already works for callers who PASS reference views; single-photo
# t23d/i23d never had any. `geometry_conditioning` closes that gap by
# synthesizing the missing canonical views from the source photo (rotate
# i2i, meshless — no clay render exists yet) and gating them before the
# checkpoint sees them, because a WRONG conditioning view is worse than
# single-view.
_GEOMETRY_CONDITIONING_MODES = ("single", "multiview", "auto")
# The three canonical views the 2mv checkpoint accepts beyond the front:
# this repo's azimuth convention (side_left = +90, camera on the subject's
# left) snaps onto the MV tags through _MV_TAG_BY_AZIMUTH.
_GEOMETRY_VIEW_ANGLES = (
    ("back", 180.0),
    ("side_left", 90.0),
    ("side_right", -90.0),
)
_GEOMETRY_VIEW_PHRASES = {
    "back": "seen directly from behind",
    "side_left": "seen from its left side profile",
    "side_right": "seen from its right side profile",
}
# Two attempts per view: shape failure of the rotate i2i is stochastic
# (same finding as the texture-lane ladder), and each attempt costs one
# i2i generation (~15-40 s locally) against a 10-25 min shape stage.
_GEOMETRY_VIEW_ATTEMPTS = 2
# Seed plan: distinct from both the shape-candidate stride (base + 1000*i)
# and the texture-lane ladder (base + 1000*attempt), so no stage ever
# reuses another stage's draw. Reconstructible from the base seed.
_GEOMETRY_VIEW_SEED_OFFSET = 50_000
_GEOMETRY_VIEW_SEED_STRIDE = 1000
# Silhouette-plausibility floors, calibrated on real draws (2026-07-14,
# 24 mlx-klein generations across 4 spaced seeds on a frontal source (owl
# photo) and a three-quarter source (sportscar v5); normalized-mask
# convention of the shape ranker; full table in the A3 program report).
# In orthographic projection the BACK silhouette of any object is exactly
# the mirrored FRONT silhouette, and the LEFT/RIGHT silhouettes are exact
# mirrors of each other. Measured bands:
#   back vs mirrored source — healthy: owl 0.954-0.976, car 0.580-0.678
#     (the source's own off-axis angle, ~25-40 deg, is what lowers the
#     car band: the relation compares view@theta+180 with view@180);
#     wrong-subject swaps: 0.398-0.437. Floor 0.52 splits those bands
#     with >= 0.06 margin each way. It does NOT separate a side-view or
#     front-echo lie on COMPACT subjects (owl lies measure 0.86-0.87,
#     inside its healthy band) — by the same measurement, such a lie's
#     silhouette damage is bounded (the silhouettes barely differ);
#     content-level damage is the material gate's and the A/B's problem.
#   side pair mutual mirror — healthy same-seed pairs: owl 0.874-0.953,
#     car 0.764-0.788 (cross-seed independent draws: 0.729-0.939);
#     front-echo-as-one-side on the elongated class: 0.615-0.658.
#     Floor 0.68 keeps every measured healthy pair (margin 0.049) and
#     rejects the measured elongated-class lies (margin 0.022).
_GEOMETRY_BACK_MIRROR_IOU_MIN = 0.52
_GEOMETRY_SIDE_PAIR_MIRROR_IOU_MIN = 0.68
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
# The 2mv checkpoint is a 2.0-family model with its OWN validated regime:
# the official model-card snippet runs steps 30 / octree ~380, and this
# repo's checked face-2mv proof ran 384/30 (healthy mesh). Running 2mv at
# the flagship's 512/50 defaults was measured CATASTROPHIC (2026-07-14
# A/B: the same gated conditioning views produced 822 raw bodies /
# euler +887 / dihedral RMS 36.9 deg at 512/50 — a shredded film-shell
# field). Explicit caller options and configured fleet defaults still win.
_MV_DEFAULT_NUM_INFERENCE_STEPS = 30
_MV_DEFAULT_OCTREE_RESOLUTION = 384
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
        # Best-of-N shape selection; 1 = single draw, exactly the historical
        # behavior (no ranking, no extra renders). Each extra candidate adds
        # roughly one shape-stage time (~21-28 min measured on MPS at the
        # default 512 octree) and seconds of ranking.
        "shape_candidates": _owner_cfg_int(owner, "scene3d_hunyuan_shape_candidates", 1),
    }


# Ground-slab signature thresholds, measured on the 2026-07 fleet (slab car
# /tmp/fix3/car_final vs controls: slab-free car v7, owl on a legitimate
# carved base, chair legs, starship fins, face bust). The upstream defect
# (Hunyuan3D-2.1 issue #48) fuses a hallucinated ground plate under vehicle
# subjects; the plate is draw-dependent. Three independent conditions are
# ANDed — each alone is insufficient (the owl's carved base covers 49% of
# the footprint; a plate gate alone would amputate it):
#
#   plate:    bottom-anchored, near-planar down-facing skin whose area is a
#             large fraction of the whole-mesh footprint hull.
#             Measured: slab 0.97 vs owl 0.49, all others <= 0.04.
#   lamina:   an up-facing top skin within 5% of mesh height directly above
#             the plate (laterally inside the plate's own hull): the
#             plate is a THIN exposed sheet, not the underside of a solid
#             base. Measured: slab 0.61 (top skin ~2% H above the bottom)
#             vs 0.00 on owl/chair/starship/v7 (the owl base's carved top
#             sits at 10-11% H) and 0.06 on the face bust.
#   overhang: the plate's lateral hull extends beyond the convex footprint
#             of everything above it — ground extends past the subject.
#             Measured: slab 1.30 vs all controls <= 0.65 (a legitimate
#             base/legs always sit inside the subject's footprint).
_SLAB_NORMAL_COS_MIN = 0.9  # |n.up| for down/up skins (cone of ~25.8 deg)
_SLAB_BOTTOM_BAND_FRAC = 0.05  # plate search band above the mesh bottom
_SLAB_PLANE_BAND_FRAC = 0.01  # co-planarity band around the plate height
_SLAB_LAMINA_BAND_FRAC = 0.05  # paired top skin must sit within this above the plate
_SLAB_PLATE_FOOTPRINT_MIN = 0.30  # slab 0.97; strongest control (owl) 0.49
_SLAB_LAMINA_MIN = 0.25  # slab 0.61; strongest control (face) 0.06
_SLAB_OVERHANG_MIN = 1.05  # slab 1.30; strongest control (chair) 0.65
# Fail-closed cut budget: the subject must remain the majority of its own
# surface. The measured real slab cut removes 38% of the car's area (the
# plate is two full-footprint skins plus rim); anything beyond half the
# surface means "plate under a subject" is the wrong reading of the mesh,
# so detection is reported but the cut is refused.
_SLAB_MAX_CUT_AREA_FRAC = 0.50


def _lateral_hull_area(points_2d: Any) -> float:
    """Convex-hull area of 2D points; 0.0 for degenerate inputs."""
    import numpy as np

    points_2d = np.asarray(points_2d, dtype=np.float64)
    if len(points_2d) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        # For 2D inputs qhull's "volume" is the polygon area.
        return float(ConvexHull(points_2d).volume)
    except Exception:
        return 0.0


def _hunyuan_cut_ground_slab(
    mesh: Any,
    *,
    up_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """Detect and remove an extraneous ground plate fused under the subject.

    Runs in the frame the caller specifies via `up_axis` (the Hunyuan native
    output is Y-up, hence the default). Detection is the three-way measured
    signature documented above; removal is a planar cut just above the
    slab's top skin. The cut leaves an open rim (the pipeline tolerates
    non-watertight meshes: the bake projects onto visible surface, quadric
    decimation runs with preserveboundary, and `topology.is_watertight` is
    recorded, not gated). Returns `(mesh, report)`; `report` is None when no
    slab is detected, otherwise carries the measurements and the action
    ("removed", or "refused" when the fail-closed budget blocks the cut).
    """
    import numpy as np

    up = np.asarray(up_axis, dtype=np.float64)
    up = up / np.linalg.norm(up)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0 or len(vertices) == 0:
        return mesh, None
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    heights = vertices @ up
    bottom = float(heights.min())
    height = float(heights.max()) - bottom
    if height <= 0.0:
        return mesh, None
    face_heights = vertices[faces].mean(axis=1) @ up
    total_area = float(face_areas.sum())
    if total_area <= 0.0:
        return mesh, None

    # Lateral frame: the two world axes most orthogonal to up (exact for
    # the axis-aligned frames this backend works in).
    lateral_axes = [axis for axis in np.eye(3) if abs(float(axis @ up)) < 0.5][:2]
    lateral = np.column_stack([vertices @ lateral_axes[0], vertices @ lateral_axes[1]])
    face_lateral = np.column_stack(
        [vertices[faces].mean(axis=1) @ lateral_axes[0], vertices[faces].mean(axis=1) @ lateral_axes[1]]
    )
    footprint_hull = _lateral_hull_area(lateral)
    if footprint_hull <= 0.0:
        return mesh, None

    normal_dot_up = face_normals @ up
    down_band = (normal_dot_up <= -_SLAB_NORMAL_COS_MIN) & (
        face_heights < bottom + _SLAB_BOTTOM_BAND_FRAC * height
    )
    if int(down_band.sum()) < 3:
        return mesh, None
    # Plate height = area-weighted median of the down band: robust against
    # feet/wheel contact skins sharing the band with the plate.
    order = np.argsort(face_heights[down_band])
    cumulative = np.cumsum(face_areas[down_band][order]) / float(face_areas[down_band].sum())
    plate_height = float(face_heights[down_band][order][np.searchsorted(cumulative, 0.5)])
    plate = down_band & (np.abs(face_heights - plate_height) < _SLAB_PLANE_BAND_FRAC * height)
    plate_area = float(face_areas[plate].sum())
    plate_footprint_frac = plate_area / footprint_hull
    if plate_footprint_frac < _SLAB_PLATE_FOOTPRINT_MIN:
        return mesh, None

    # Paired thin top skin: up-facing area within the lamina band above the
    # plate, laterally restricted to the plate's own 2D convex hull (a
    # subject's horizontal surfaces elsewhere must not count as slab top).
    # The hull test is tessellation-independent — an occupancy-grid variant
    # sampled at face centroids was measured to under-pair coarsely
    # decimated plates whose triangles span many grid cells.
    plate_vertex_ids = np.unique(faces[plate].ravel())
    plate_xy = lateral[plate_vertex_ids]
    top_band = (
        (normal_dot_up >= _SLAB_NORMAL_COS_MIN)
        & (face_heights > plate_height)
        & (face_heights <= plate_height + _SLAB_LAMINA_BAND_FRAC * height)
    )
    top_idx = np.flatnonzero(top_band)
    if len(top_idx) and len(plate_xy) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(plate_xy)
            # Signed distances to the hull's facet lines (normals are unit
            # length); a small scale-free margin absorbs plate-rim jitter.
            margin = 0.02 * float(np.linalg.norm(plate_xy.max(axis=0) - plate_xy.min(axis=0)))
            homogeneous = np.column_stack([face_lateral[top_idx], np.ones(len(top_idx))])
            inside = (homogeneous @ hull.equations.T <= margin).all(axis=1)
            top_idx = top_idx[inside]
        except Exception:
            top_idx = top_idx[:0]
    top_area = float(face_areas[top_idx].sum()) if len(top_idx) else 0.0
    lamina_ratio = top_area / plate_area if plate_area > 0 else 0.0

    # Overhang: plate hull vs the convex footprint of everything above the
    # lamina band. An empty "above" region degenerates to infinite overhang
    # and is then caught by the fail-closed cut budget.
    plate_hull = _lateral_hull_area(plate_xy)
    above = heights > plate_height + _SLAB_LAMINA_BAND_FRAC * height
    subject_hull = _lateral_hull_area(lateral[above]) if int(above.sum()) >= 3 else 0.0
    overhang_ratio = plate_hull / subject_hull if subject_hull > 0 else float("inf")

    detected = lamina_ratio >= _SLAB_LAMINA_MIN and overhang_ratio >= _SLAB_OVERHANG_MIN
    if not detected:
        return mesh, None

    # Slab top level = area-weighted median of the paired top skin; the cut
    # clears it by a quarter of the slab thickness (floor: 0.5% of height)
    # so the top skin's own roughness cannot leave rim shards, while the
    # subject loses at most that sliver at its ground contacts.
    top_order = np.argsort(face_heights[top_idx])
    top_cumulative = np.cumsum(face_areas[top_idx][top_order]) / float(face_areas[top_idx].sum())
    slab_top = float(face_heights[top_idx][top_order][np.searchsorted(top_cumulative, 0.5)])
    slab_thickness = max(slab_top - plate_height, 0.0)
    cut_height = slab_top + max(0.005 * height, 0.25 * slab_thickness)
    cut_faces = face_heights < cut_height
    cut_area_frac = float(face_areas[cut_faces].sum() / total_area)

    report: Dict[str, Any] = {
        "plate_footprint_frac": round(plate_footprint_frac, 4),
        "lamina_ratio": round(lamina_ratio, 4),
        "overhang_ratio": round(overhang_ratio, 4) if np.isfinite(overhang_ratio) else None,
        "plate_area": round(plate_area, 6),
        "plate_height_rel": round((plate_height - bottom) / height, 4),
        "slab_thickness_rel": round(slab_thickness / height, 4),
        "cut_height_rel": round((cut_height - bottom) / height, 4),
        "cut_area_frac": round(cut_area_frac, 4),
        "faces_before": int(len(faces)),
    }
    if cut_area_frac > _SLAB_MAX_CUT_AREA_FRAC:
        report["action"] = "refused"
        report["refusal_reason"] = (
            f"cut would remove {cut_area_frac:.1%} of the surface "
            f"(budget {_SLAB_MAX_CUT_AREA_FRAC:.0%}): the subject must remain "
            "the majority of its own mesh"
        )
        return mesh, report

    processed = mesh.copy()
    processed.update_faces(~cut_faces)
    processed.remove_unreferenced_vertices()
    # The cut can strand slab shards (rim pieces bridged only through
    # removed faces). Sweep with the same 0.5%-of-total floater rule the
    # earlier cleanup uses, so genuine detached parts keep their guarantee.
    import trimesh

    components = list(processed.split(only_watertight=False))
    if len(components) > 1:
        total_faces = max(1, int(sum(len(item.faces) for item in components)))
        threshold = max(64, int(round(total_faces * 0.005)))
        kept = [item for item in components if len(item.faces) >= threshold]
        if not kept:
            kept = [max(components, key=lambda item: len(item.faces))]
        if len(kept) != len(components):
            report["orphans_dropped"] = len(components) - len(kept)
        processed = kept[0].copy() if len(kept) == 1 else trimesh.util.concatenate(kept)
    report["action"] = "removed"
    report["faces_after"] = int(len(processed.faces))
    return processed, report


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

        # Ground-slab removal runs BEFORE decimation (the face budget must
        # go to the subject, not the plate) and in the Hunyuan native frame
        # (Y-up; canonicalization happens after this postprocess). A
        # detected-but-refused cut is surfaced as a warning here and
        # demoted to a degraded verdict by the caller (fail-closed).
        ground_slab_report: Optional[Dict[str, Any]] = None
        try:
            processed, ground_slab_report = _hunyuan_cut_ground_slab(processed, up_axis=(0.0, 1.0, 0.0))
            if ground_slab_report is not None:
                measurements = (
                    f"plate={ground_slab_report['plate_footprint_frac']},"
                    f"lamina={ground_slab_report['lamina_ratio']},"
                    f"overhang={ground_slab_report['overhang_ratio']},"
                    f"cut_area_frac={ground_slab_report['cut_area_frac']}"
                )
                if ground_slab_report.get("action") == "removed":
                    applied.append(
                        "ground_slab_removed:"
                        f"{ground_slab_report['faces_before']}->{ground_slab_report['faces_after']}"
                        f"@{measurements}"
                    )
                else:
                    warnings.append(
                        "Hunyuan3D ground slab detected but not cut "
                        f"({ground_slab_report.get('refusal_reason')}); measurements: {measurements}"
                    )
        except Exception as exc:
            warnings.append(f"Hunyuan3D ground-slab check skipped: {type(exc).__name__}: {exc}")

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
        if ground_slab_report is not None:
            # Carried on the trimesh metadata dict (survives copy and
            # transform) so the caller can lift the measurements into the
            # run metadata; the caller pops it before any export so glTF
            # extras never grow a private field.
            try:
                processed.metadata["abstract3d_ground_slab"] = ground_slab_report
            except Exception:
                pass
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


# -- best-of-N shape candidate ranking ---------------------------------------
#
# Measured motivation (2026-07-13, /tmp/gfix3 car_a vs car_b: the same t23d
# request, same code and settings, different DiT draws): draw A came out
# euler -240 / not watertight / dihedral RMS 17.9 deg and baked to 28.6 dE
# baseline photo-fidelity; draw B came out euler -146 / watertight / RMS
# 15.1 and baked to 22.2 dE — visibly better everywhere downstream. Draw
# luck of the shape DiT is the dominant remaining quality factor, ranking
# takes seconds against ~21-28 min per draw, so drawing N shape candidates
# and keeping the best buys quality at a known, linear cost.
#
# Every constant below is calibrated on the persisted corpus (calibration
# table in CHANGELOG.md): car_a/car_b (same-subject pair with known
# ordering), sportscar_v7 (good), the pre-cutter car_final (ground-slab
# class), the owl/chair/starship/face fleet proofs, and adversarial probes
# built from car_b (60-iteration laplacian melt; its convex hull — the
# smoothest possible wrong-detail candidate).

# Spec'd stride between candidate seeds: far enough apart that consecutive
# candidates are independent draws, and reconstructible from the base seed
# (candidate i draws at seed + 1000*i; the texture stage keeps the BASE
# seed so reference generation is unchanged by candidate count).
_SHAPE_CANDIDATE_SEED_STRIDE = 1000
# Pose sweep for photo agreement. Hunyuan reconstructs the subject roughly
# facing the conditioning photo, so the photo's true pose is near the
# canonical front: every measured fleet pose estimate falls inside
# |azimuth| <= 40 deg, elevation 0..20 (car_a (25.9, 15), car_b (25, 8),
# v7 (17.5, 8), slab car rescue (40, 15), chair/owl/face at (<=10, 0),
# starship (30, 20)). 10-deg azimuth steps cost each candidate the same
# small quantization penalty (measured ~0.01 IoU), so comparisons stay
# fair; a FINER sweep was measured to hurt ranking (at 2.5-deg refinement
# the lumpy car_a refines ABOVE car_b, 0.9238 vs 0.9168 — each mesh finds
# a flattering angle and matte noise dominates).
_SHAPE_RANK_POSE_AZIMUTHS = tuple(float(a) for a in range(-40, 41, 10))
_SHAPE_RANK_POSE_ELEVATIONS = (0.0, 10.0, 20.0)
# 256 px renders / 128 px normalized masks: the corpus separations quoted
# below are measured at exactly these sizes; the full 27-pose sweep costs
# 1-4 s per candidate on this hardware (budget: 30 s).
_SHAPE_RANK_RENDER_SIZE = 256
_SHAPE_RANK_NORM_SIZE = 128
# Clay-render background (uint8). ModernGL clears to (0.95, 0.95, 0.93)
# -> (242, 242, 237); the matplotlib fallback paints #f3f2ee ->
# (243, 242, 238), within 1/255 of it. The brightest clay shade is
# 0.72 * 255 = 184 (>= 53 levels away), so one threshold classifies
# subject vs background under either renderer.
_SHAPE_RANK_RENDER_BG = (242, 242, 237)
_SHAPE_RANK_BG_TOLERANCE = 12
# Score weights. Justification (all values measured on the corpus at the
# sizes above; full table in CHANGELOG.md):
#   silhouette 1.0   — the reference scale. Wrong-subject candidates lose
#                      0.35-0.50 IoU (owl-mesh-vs-car-photo 0.41, chair
#                      0.53, car-vs-owl 0.56 against honest 0.90-0.93),
#                      more than every other term combined (0.65), so a
#                      wrong shape can never be bought back. The slab
#                      class loses 0.11 (0.80 vs 0.91).
#   concavity 0.35   — the anti-melt axis. Silhouette IoU alone CANNOT
#                      catch a smooth wrong-detail candidate on
#                      silhouette-convex subjects (measured: car_b's own
#                      convex hull scores 0.9115, ABOVE the true draw's
#                      0.9078), and dihedral smoothness actively rewards
#                      it. Concave detail — wheel arches, under-body gap —
#                      is what melting destroys: hull-minus-mask IoU
#                      collapses 0.235 -> 0.027 (8.7x) for the hull and
#                      0.057 for the slab car. 0.35 makes that collapse
#                      cost ~0.073, 3.2x the hull's maximum smoothness
#                      gain (0.023), with silhouette noise (+-0.005) far
#                      below it.
#   topology 0.20    — watertightness and single-body are the measured
#                      draw-quality separators from the motivating pair
#                      (car_a not watertight, car_b watertight; split
#                      bodies were the v5 floating-wheels incident). 0.20
#                      (0.10 per condition) decides same-subject ties
#                      where photo terms measure within noise (a vs b
#                      photo delta -0.004) without ever outranking a
#                      photo-term class gap (slab: 0.24).
#   smoothness 0.0   — dihedral RMS is RECORDED (diagnostic per candidate)
#                      but carries NO score weight. Adversarial validation
#                      (2026-07-13, laplacian melt ladder on the real car_b
#                      geometry, /tmp/hfix2) measured that any positive
#                      weight is a monotone reward for the melt direction:
#                      melting strictly improves RMS (15.09 -> 11.93/10.34/
#                      9.89 deg at 8/40/150 iterations) while quality
#                      strictly degrades, and the concavity axis does NOT
#                      collapse on realistic melts (true 0.2355 vs mild-melt
#                      0.2696, 60-it melt 0.2693 — only the convex-hull
#                      extreme collapses to 0.0272). At the original 0.10
#                      the 40-it melt outscored the true draw by +0.0101 and
#                      the 60-it calibration melt by +0.0181; at 0.0 every
#                      melt >= 40 iterations ranks at or below the true draw
#                      and the residual mild-melt delta (+0.0116) is
#                      symmetric concavity noise (good-draw concavity spread
#                      0.24-0.40), not a preference. Removing the weight
#                      costs nothing measured on legit pairs: car_b > car_a
#                      +0.102 -> +0.096 (decided by watertightness + photo
#                      terms), v7 > v4 unchanged, slab and wrong-subject
#                      margins unchanged, hull still rejected by -0.069.
_SHAPE_RANK_WEIGHT_SILHOUETTE = 1.0
_SHAPE_RANK_WEIGHT_CONCAVITY = 0.35
_SHAPE_RANK_WEIGHT_TOPOLOGY = 0.20
_SHAPE_RANK_WEIGHT_SMOOTHNESS = 0.0
# Normalizes dihedral RMS into [0, 1): ~1.6x the largest accepted-fleet
# value (face proof, 27.6 deg), so real meshes span (0.35, 1.0] linearly
# and the term never saturates or goes negative on plausible draws.
_SHAPE_RANK_SMOOTHNESS_SCALE_DEG = 45.0


def _dihedral_rms_deg(mesh: Any) -> Optional[float]:
    """RMS of face-adjacency dihedral angles, in degrees.

    A surface-noise measure: marching-cubes lumps show up as many small
    random bends between adjacent faces. Only comparable between meshes of
    the same subject at the same face budget (tessellation density scales
    the per-edge angles) — which is the candidate-ranking situation.
    """
    import numpy as np

    try:
        angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    except Exception:
        return None
    if angles.size == 0:
        return 0.0
    return float(np.degrees(np.sqrt(np.mean(np.square(angles)))))


def _normalize_mask_frame(mask: Any, *, size: int = _SHAPE_RANK_NORM_SIZE) -> Optional[Any]:
    """Crop a binary mask to its bbox, pad square, resize to a common grid.

    Same convention as the bake's normalized silhouette IoU (aspect ratio
    preserved — it is a real shape cue; scale and translation removed —
    camera distance is fitted later, in the bake, and must not leak into
    shape ranking). Returns None for empty masks.
    """
    import numpy as np
    from PIL import Image

    mask = np.asarray(mask, dtype=bool)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    side = max(cropped.shape)
    padded = np.zeros((side, side), dtype=bool)
    top = (side - cropped.shape[0]) // 2
    left = (side - cropped.shape[1]) // 2
    padded[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
    image = Image.fromarray((padded * 255).astype(np.uint8))
    return np.asarray(image.resize((int(size), int(size)), Image.BILINEAR)) > 127


def _mask_iou(mask_a: Any, mask_b: Any) -> float:
    import numpy as np

    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum()) / union


def _mask_convex_hull(mask: Any) -> Any:
    """Filled 2D convex hull of a raster mask (Delaunay point-in-hull).

    Degrades to the mask itself when the support is degenerate (<3 points
    or collinear), which makes the negative region empty — the safe
    reading for a mask with no measurable concavity.
    """
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    points = np.argwhere(mask)
    if len(points) < 3:
        return mask.copy()
    try:
        from scipy.spatial import Delaunay

        triangulation = Delaunay(points)
    except Exception:
        return mask.copy()
    grid = np.argwhere(np.ones_like(mask))
    inside = triangulation.find_simplex(grid) >= 0
    return inside.reshape(mask.shape)


def _clay_silhouette_mask(image: Any) -> Any:
    """Silhouette of a clay preview render via the known background color.

    Candidate meshes are untextured at ranking time, so the render
    background is a constant this module controls — no segmentation model
    is needed (running a salient-object matte on synthetic clay is its own
    failure mode).
    """
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    background = np.asarray(_SHAPE_RANK_RENDER_BG, dtype=np.int16)
    return (np.abs(array - background[None, None, :]) > _SHAPE_RANK_BG_TOLERANCE).any(axis=2)


def _photo_matte_mask(rgba_image: Any) -> Optional[Any]:
    """Photo matte as a boolean mask, or None when unusable for ranking.

    Unusable means: no alpha support at all, or alpha covering >95% of the
    frame (background removal did not run / failed, so the "matte" is the
    whole rectangle and silhouette agreement would measure nothing). The
    caller degrades to geometry-only ranking — symmetrically for every
    candidate — instead of ranking against a meaningless mask.
    """
    import numpy as np

    try:
        array = np.asarray(rgba_image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    alpha = array[:, :, 3]
    support = alpha > 12
    coverage = float(support.mean())
    if coverage <= 0.005 or coverage >= 0.95:
        return None
    return support


def evaluate_shape_candidate(
    mesh: Any,
    *,
    matte_mask: Any = None,
    azimuths: Any = _SHAPE_RANK_POSE_AZIMUTHS,
    elevations: Any = _SHAPE_RANK_POSE_ELEVATIONS,
    render_size: int = _SHAPE_RANK_RENDER_SIZE,
) -> Dict[str, Any]:
    """Measure one postprocessed candidate (canonical frame) for ranking.

    Geometry terms (always): watertightness, body count, euler number,
    dihedral RMS. Photo-agreement terms (when a usable matte is given):
    max normalized silhouette IoU over the pose sweep, and — at that same
    argmax pose, so the evidence stays consistent — the IoU of the
    NEGATIVE regions (convex hull minus mask on both sides). The negative
    region is what a melted/blob candidate destroys while keeping the
    outer silhouette; see the weight table above for the measured margins.
    """
    import numpy as np

    metrics: Dict[str, Any] = {}
    topology = _mesh_topology(mesh)
    metrics["watertight"] = bool(topology.get("is_watertight") or False)
    metrics["body_count"] = int(topology.get("body_count") or 0)
    metrics["euler_number"] = topology.get("euler_number")
    metrics["dihedral_rms_deg"] = _dihedral_rms_deg(mesh)
    rms = metrics["dihedral_rms_deg"]
    metrics["smoothness"] = (
        float(np.clip(1.0 - float(rms) / _SHAPE_RANK_SMOOTHNESS_SCALE_DEG, 0.0, 1.0))
        if rms is not None
        else 0.0
    )
    metrics["topology_score"] = 0.5 * float(metrics["watertight"]) + 0.5 * float(
        metrics["body_count"] == 1
    )
    metrics["photo_iou"] = None
    metrics["photo_concavity_iou"] = None
    metrics["photo_iou_pose"] = None

    if matte_mask is None:
        return metrics
    photo_norm = _normalize_mask_frame(matte_mask)
    if photo_norm is None:
        return metrics

    try:
        best_iou = -1.0
        best_pose: Optional[tuple[float, float]] = None
        best_render_norm: Any = None
        for elevation in elevations:
            views = render_mesh_views(
                mesh,
                size=int(render_size),
                azimuths=tuple(float(a) for a in azimuths),
                elevation=float(elevation),
            )
            for azimuth, view in zip(azimuths, views):
                render_norm = _normalize_mask_frame(_clay_silhouette_mask(view))
                if render_norm is None:
                    continue
                iou = _mask_iou(photo_norm, render_norm)
                if iou > best_iou:
                    best_iou = iou
                    best_pose = (float(azimuth), float(elevation))
                    best_render_norm = render_norm
        if best_pose is None:
            # No pose produced a silhouette: fail closed with zero photo
            # agreement (an unverifiable candidate must not win on
            # geometry terms that a blob can game).
            metrics["photo_iou"] = 0.0
            metrics["photo_concavity_iou"] = 0.0
            return metrics
        photo_negative = _mask_convex_hull(photo_norm) & ~photo_norm
        render_negative = _mask_convex_hull(best_render_norm) & ~best_render_norm
        metrics["photo_iou"] = round(float(best_iou), 4)
        metrics["photo_concavity_iou"] = round(_mask_iou(photo_negative, render_negative), 4)
        metrics["photo_iou_pose"] = {
            "azimuth_deg": best_pose[0],
            "elevation_deg": best_pose[1],
        }
    except Exception as exc:
        # Same fail-closed contract as the empty-render branch, with the
        # reason preserved for the metadata record.
        metrics["photo_iou"] = 0.0
        metrics["photo_concavity_iou"] = 0.0
        metrics["photo_agreement_error"] = f"{type(exc).__name__}: {exc}"
    return metrics


def score_shape_candidate(metrics: Mapping[str, Any]) -> float:
    """Composite ranking score; higher is better.

    Photo terms are omitted (not zeroed) when the matte was unusable — the
    caller guarantees that happens symmetrically for every candidate in a
    run, so scores stay comparable within the run.
    """
    score = (
        _SHAPE_RANK_WEIGHT_TOPOLOGY * float(metrics.get("topology_score") or 0.0)
        + _SHAPE_RANK_WEIGHT_SMOOTHNESS * float(metrics.get("smoothness") or 0.0)
    )
    if metrics.get("photo_iou") is not None:
        score += _SHAPE_RANK_WEIGHT_SILHOUETTE * float(metrics["photo_iou"])
        score += _SHAPE_RANK_WEIGHT_CONCAVITY * float(metrics.get("photo_concavity_iou") or 0.0)
    return float(score)


# -- pre-shape geometry-view synthesis ----------------------------------------


def _mv_snap_tag(azimuth_deg: float) -> Optional[tuple[str, float]]:
    """Snap a declared camera azimuth onto the nearest 2mv view tag.

    Returns `(tag, snap_delta_deg)` or None when no tag lies within the
    snap tolerance. Tags are 90 deg apart and the tolerance is 25 deg, so
    at most one tag can match.
    """
    best: Optional[tuple[str, float]] = None
    for tag_azimuth, tag in _MV_TAG_BY_AZIMUTH.items():
        delta = abs(((float(azimuth_deg) - tag_azimuth) + 180.0) % 360.0 - 180.0)
        if delta <= _MV_TAG_SNAP_TOLERANCE_DEG and (best is None or delta < best[1]):
            best = (tag, delta)
    return best


def _mv_cap_conditioning_views(
    mv_image_dict: Dict[str, Any],
    geometry_views_used: List[Dict[str, Any]],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Enforce the measured conditioning-view cap; returns dropped rows.

    Priority rationale (see _MV_MAX_CONDITIONING_VIEWS for the cliff
    measurement): the FRONT is the user's own photo — identity and
    registration anchor; the BACK is the audit-documented failure surface
    of single-view runs ("flat back / invented rear") and part of the
    official 3-view snippet; one SIDE constrains the profile; the second
    side is the most redundant view on a fleet measured ~97% bilaterally
    symmetric — and a dropped view still reaches the texture lane through
    the reference replay, so nothing synthesized is wasted.
    """
    dropped: List[Dict[str, Any]] = []
    if len(mv_image_dict) <= _MV_MAX_CONDITIONING_VIEWS:
        return dropped
    keep = sorted(mv_image_dict, key=_MV_TAG_PRIORITY.index)[:_MV_MAX_CONDITIONING_VIEWS]
    for tag in [t for t in mv_image_dict if t not in keep]:
        del mv_image_dict[tag]
        for row in list(geometry_views_used):
            if row.get("tag") == tag:
                geometry_views_used.remove(row)
                row["conditioning_dropped"] = "view_cap"
                dropped.append(row)
    warnings.append(
        "multiview conditioning capped at "
        f"{_MV_MAX_CONDITIONING_VIEWS} views (measured: 4-view 2mv "
        "conditioning shreds the field into film-shell debris — 559 raw "
        "bodies on the same images that produce a healthy mesh with any "
        "1-3 view subset); dropped: "
        + ", ".join(str(row.get("tag")) for row in dropped)
    )
    return dropped


def _geometry_view_prompt(label: str, subject_noun: Optional[str]) -> str:
    """Rotate-style synthesis instruction for a meshless conditioning view.

    Modeled on the texture lane's "rotate" conditioning variant (the only
    variant that works without a clay render — no mesh exists yet), with
    the material-free noun contract of `captioning.extract_subject_noun`:
    the template has no free-text slot, so no caption or user hint can
    inject a material claim. The whole-subject clause exists because a
    cropped or zoomed view corrupts the canonical recentring every 2mv
    view goes through (border_ratio framing assumes the full subject).
    """
    noun = (subject_noun or "").strip() or "object"
    phrase = _GEOMETRY_VIEW_PHRASES.get(label, f"seen from the {label.replace('_', ' ')} view")
    return (
        f"Rotate the camera to show this exact {noun} {phrase}. Keep the "
        "same object identity: the same proportions and shape, the same "
        "surface relief, colors and pattern. Do not change the material "
        "type. The whole subject stays fully visible and centered, at the "
        "same distance as the input photo. Plain dark background, soft "
        "diffuse even lighting."
    )


def _normalized_matte(rgba_image: Any) -> Optional[Any]:
    """Normalized (bbox-cropped, square, 128 px) boolean matte of an RGBA
    image, or None when the matte is unusable (no alpha support, or alpha
    covering nearly the whole frame — the same sanity line the shape
    ranker draws via `_photo_matte_mask`)."""
    mask = _photo_matte_mask(rgba_image)
    if mask is None:
        return None
    return _normalize_mask_frame(mask)


def _geometry_person_gate(
    source_rgba: Any,
    *,
    subject_hint: Optional[str],
    allow_person: bool,
) -> tuple[bool, Dict[str, Any]]:
    """Person doctrine for geometry-view synthesis; `(proceed, record)`.

    Identical doctrine to `reference_generation.generate_reference_views`
    (which cannot be reused directly here because it requires a mesh): no
    gate in the stack measures identity, so synthesizing views of a person
    requires the explicit person acknowledgment — and an UNAVAILABLE
    captioner is not a permission grant (fail closed). The acknowledgment
    is the same one the texture lane uses (`texture_reference_allow_person`)
    because it attests the same act: synthesizing unseen views of a person.
    """
    from ..captioning import caption_image
    from ..reference_generation import is_person_subject

    caption: Optional[str] = None
    hint = (subject_hint or "").strip()
    person_detected = is_person_subject(hint)
    if not person_detected:
        # A hint that doesn't name a person is not evidence of absence:
        # caption the photo itself before unattended synthesis.
        caption = caption_image(source_rgba)
        if caption is None and not allow_person:
            return False, {
                "person_detected": None,
                "caption": None,
                "refusal": (
                    "captioner unavailable: the person-subject check cannot "
                    "run, so geometry-view synthesis is refused (fail closed). "
                    "Install transformers/BLIP or pass the person "
                    "acknowledgment (texture_reference_allow_person)."
                ),
            }
        person_detected = is_person_subject(caption)
    record: Dict[str, Any] = {
        "person_detected": bool(person_detected),
        "caption": caption,
    }
    if person_detected and not allow_person:
        record["refusal"] = (
            "person subject detected: no gate defends facial identity, so "
            "synthesizing conditioning views of a person requires the "
            "explicit person acknowledgment (texture_reference_allow_person "
            "/ --texture-reference-allow-person)"
        )
        return False, record
    if person_detected:
        record["person_warning"] = (
            "person subject: synthesized conditioning views may not preserve "
            "facial identity (no identity gate exists); acknowledged "
            "explicitly, proceeding"
        )
    return True, record


def _harden_skimage_color_convert() -> None:
    """Route skimage's 3x3 color-matrix multiply through einsum on macOS.

    Documented host-class crash (KnowledgeBase: "Accelerate BLAS segfaults
    (exit 139) under threaded float64 GEMM"): skimage's `_convert` uses a
    float64 `@` that dispatches to Accelerate's cblas_dgemm, which
    segfaults on this host class — measured 2/2 on this feature's
    synthesis path (lab2rgb inside `match_tone_lab`) while the MLX image
    model is resident, EVEN WITH `VECLIB_MAXIMUM_THREADS=1`. The einsum
    form is the mitigation two prior audit programs shipped for harnesses;
    it is numerically identical (verified below to 1e-12 on a probe before
    the swap; on any mismatch the swap is refused). Called from the
    synthesis path only, so the default single-view flow never touches
    another library's internals.
    """
    if sys.platform != "darwin":
        return
    try:
        import numpy as np
        import skimage.color.colorconv as colorconv
    except Exception:
        return
    current = getattr(colorconv, "_convert", None)
    if current is None or getattr(current, "_abstract3d_einsum", False):
        return

    def _convert_einsum(matrix, arr):
        arr = colorconv._prepare_colorarray(arr)
        return np.einsum("...i,ji->...j", arr, matrix.astype(arr.dtype))

    try:
        probe = np.random.default_rng(1).random((5, 7, 3))
        matrix = np.asarray(
            [[3.24, -1.54, -0.50], [-0.97, 1.88, 0.04], [0.06, -0.20, 1.06]]
        )
        if float(np.abs(_convert_einsum(matrix, probe) - (probe @ matrix.T)).max()) > 1e-12:
            return
    except Exception:
        return
    _convert_einsum._abstract3d_einsum = True  # type: ignore[attr-defined]
    colorconv._convert = _convert_einsum


def _synthesize_geometry_views(
    owner: Any,
    source_rgba: Any,
    *,
    subject_noun: str,
    base_seed: int,
    labels: Any,
    attempts: int = _GEOMETRY_VIEW_ATTEMPTS,
    image_generator: Optional[Any] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Synthesize and gate canonical conditioning views from the source
    photo (no mesh exists yet). Returns
    `(accepted_views, view_records, rejected_images)` — rejected pixels
    travel back for bundle persistence (persist-for-diagnosis doctrine: a
    rejected class must be diagnosable without a rerun).

    Accepted views carry the matted, tone-matched RGBA the conditioner
    will see plus the RAW generation bytes (for the texture lane's replay
    through the full reference-acceptance machinery) and provenance.

    Gate battery, in evidence order (a wrong conditioning view is worse
    than single-view — the 2mv checkpoint TRUSTS its tags):
      1. matte sanity — the generation must segment to a subject that is
         neither empty nor the whole frame (`_photo_matte_mask` bounds);
      2. subject identity — `part_material_fidelity` floor line against
         the source photo (includes the chroma-collapse guard): a palette
         flip or monochrome collapse means the generator lost the subject,
         so its geometry cannot be trusted either;
      3. back-view plausibility — in orthographic projection the back
         silhouette of ANY object is exactly the mirrored front
         silhouette; normalized mirror-IoU against the source matte has a
         calibrated floor (approximate for perspective and off-axis
         sources, hence a floor rather than a precision gate);
      4. side-pair consistency (applied by the caller across views) —
         left/right orthographic silhouettes are exact mirrors of each
         other, so a surviving pair must mirror-agree or BOTH are dropped
         (blame between them is unattributable).
    """
    import hashlib
    import io as _io

    from PIL import Image

    from ..image_composition import resolve_image_generation_request
    from ..material_gates import part_material_fidelity
    from ..reference_generation import (
        DEFAULT_NEGATIVE_PROMPT,
        default_i2i_generator,
        match_tone_lab,
    )
    from ..segmentation import remove_background_robust

    # Host hardening BEFORE any LAB round-trip on this path (measured
    # segfault class with the MLX pool resident; see the helper).
    _harden_skimage_color_convert()

    generator = image_generator or default_i2i_generator(owner)
    request = resolve_image_generation_request(owner)
    request.pop("width", None)
    request.pop("height", None)
    request.pop("seed", None)
    request = {key: value for key, value in request.items() if value is not None}

    source_norm = _normalized_matte(source_rgba)

    buffer = _io.BytesIO()
    source_rgba.convert("RGB").save(buffer, format="PNG")
    conditioning_bytes = buffer.getvalue()

    accepted: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    rejected_images: List[Dict[str, Any]] = []

    def _record_rejected(label: str, attempt: int, image: Any) -> None:
        if len(rejected_images) >= 3 * len(tuple(labels)):
            return
        small = image.copy()
        small.thumbnail((512, 512))
        rejected_images.append({"label": label, "attempt": attempt, "image": small})

    for label_index, (label, azimuth) in enumerate(labels):
        record: Dict[str, Any] = {
            "label": label,
            "azimuth_deg": float(azimuth),
            "attempts": [],
            "accepted": False,
        }
        prompt = _geometry_view_prompt(label, subject_noun)
        record["prompt"] = prompt
        started = time.perf_counter()
        for attempt in range(int(attempts)):
            attempt_seed = (
                int(base_seed)
                + _GEOMETRY_VIEW_SEED_OFFSET
                + _GEOMETRY_VIEW_SEED_STRIDE * (attempt * len(tuple(labels)) + label_index)
            )
            # 8/12-step alternation mirrors the texture ladder's measured
            # finding: the distilled klein default (4 steps) is too few,
            # and alternation decorrelates the ladder from per-steps bias.
            call_kwargs = dict(request)
            call_kwargs["steps"] = 8 if attempt % 2 == 0 else 12
            call_kwargs["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
            attempt_row: Dict[str, Any] = {
                "seed": attempt_seed,
                "steps": call_kwargs["steps"],
            }
            try:
                payload = generator(prompt, conditioning_bytes, seed=attempt_seed, **call_kwargs)
                data = payload if isinstance(payload, (bytes, bytearray)) else None
                if data is None and isinstance(payload, Mapping):
                    for key in ("data", "bytes", "content"):
                        if isinstance(payload.get(key), (bytes, bytearray)):
                            data = payload[key]
                            break
                if data is None:
                    attempt_row["error"] = "generator returned no image bytes"
                    record["attempts"].append(attempt_row)
                    continue
                raw_bytes = bytes(data)
                generated = Image.open(_io.BytesIO(raw_bytes))
                matted = remove_background_robust(generated)
                # Tone match toward the source (capped LAB statistics
                # transfer): the DINOv2 conditioner reads tone as material
                # evidence, and the caps make it structurally unable to
                # whitewash a legitimately different unseen side. The
                # chroma-collapse guard below is dispersion-based, so a
                # capped mean shift cannot hide a collapse from it.
                try:
                    matted, tone_stats = match_tone_lab(matted, source_rgba)
                    attempt_row["tone_match"] = tone_stats
                except Exception as exc:
                    attempt_row["tone_match"] = {
                        "applied": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                view_norm = _normalized_matte(matted)
                if view_norm is None:
                    attempt_row["failure"] = "unusable matte (empty or whole-frame subject)"
                    record["attempts"].append(attempt_row)
                    _record_rejected(label, attempt, matted)
                    continue
                material = part_material_fidelity(matted, source_rgba)
                attempt_row["material"] = {
                    key: material.get(key)
                    for key in ("passed", "floor", "worst_part_delta_e", "reason",
                                "source_chroma_dispersion", "generated_chroma_dispersion")
                    if material.get(key) is not None
                }
                if not material.get("floor", material.get("passed")):
                    attempt_row["failure"] = (
                        "subject identity: "
                        + str(material.get("reason") or "part material floor exceeded")
                    )
                    record["attempts"].append(attempt_row)
                    _record_rejected(label, attempt, matted)
                    continue
                if label == "back" and source_norm is not None:
                    import numpy as np

                    mirror_iou = _mask_iou(np.fliplr(source_norm), view_norm)
                    attempt_row["back_mirror_iou"] = round(float(mirror_iou), 4)
                    if mirror_iou < _GEOMETRY_BACK_MIRROR_IOU_MIN:
                        attempt_row["failure"] = (
                            f"back-view silhouette implausible: mirror-IoU vs source "
                            f"{mirror_iou:.3f} < {_GEOMETRY_BACK_MIRROR_IOU_MIN}"
                        )
                        record["attempts"].append(attempt_row)
                        _record_rejected(label, attempt, matted)
                        continue
                record["attempts"].append(attempt_row)
                record["accepted"] = True
                record["seed"] = attempt_seed
                record["raw_payload_md5"] = hashlib.md5(raw_bytes).hexdigest()
                accepted.append(
                    {
                        "label": label,
                        "azimuth_deg": float(azimuth),
                        "elevation_deg": 0.0,
                        "rgba": matted,
                        "raw_bytes": raw_bytes,
                        "raw_payload_md5": record["raw_payload_md5"],
                        "seed": attempt_seed,
                        "norm_matte": view_norm,
                    }
                )
                break
            except Exception as exc:
                attempt_row["error"] = f"{type(exc).__name__}: {exc}"
                record["attempts"].append(attempt_row)
                continue
        record["seconds"] = round(time.perf_counter() - started, 1)
        records.append(record)

    # Side-pair consistency: left/right orthographic silhouettes of one
    # object are exact mirror images, so a surviving pair that disagrees
    # contains at least one lie — and nothing attributes the blame, so
    # both are dropped (conservative by design: the fallback is the
    # KNOWN-GOOD single-view path, not a broken run).
    by_label = {view["label"]: view for view in accepted}
    left = by_label.get("side_left")
    right = by_label.get("side_right")
    if left is not None and right is not None:
        import numpy as np

        pair_iou = _mask_iou(np.fliplr(left["norm_matte"]), right["norm_matte"])
        for record in records:
            if record["label"] in ("side_left", "side_right"):
                record["side_pair_mirror_iou"] = round(float(pair_iou), 4)
        if pair_iou < _GEOMETRY_SIDE_PAIR_MIRROR_IOU_MIN:
            for record in records:
                if record["label"] in ("side_left", "side_right"):
                    record["accepted"] = False
                    record["failure"] = (
                        f"side-pair mirror disagreement: IoU {pair_iou:.3f} < "
                        f"{_GEOMETRY_SIDE_PAIR_MIRROR_IOU_MIN} (both sides dropped: "
                        "blame between them is unattributable)"
                    )
            for dropped in (left, right):
                _record_rejected(dropped["label"], -1, dropped["rgba"])
            accepted = [v for v in accepted if v["label"] not in ("side_left", "side_right")]
    for view in accepted:
        view.pop("norm_matte", None)
    return accepted, records, rejected_images


def _persist_geometry_conditioning_artifacts(
    bundle_root: Path,
    *,
    accepted_views: List[Dict[str, Any]],
    rejected_images: List[Dict[str, Any]],
    record: Optional[Dict[str, Any]],
) -> None:
    """Persist the synthesized conditioning views into the bundle.

    Accepted views land as full PNGs (the exact pixels the 2mv conditioner
    saw); rejected candidates land downscaled and budget-capped under
    `rejected_geometry_views/` — the same persist-for-diagnosis contract
    as the texture lane's `rejected_refs/` (a rejected class must be
    diagnosable without a rerun).
    """
    if record is None:
        return
    accepted_paths: List[str] = []
    for view in accepted_views:
        try:
            path = bundle_root / f"geometry_view_synthesized_{view['label']}.png"
            view["rgba"].save(path)
            accepted_paths.append(str(path))
        except Exception:
            continue
    if accepted_paths:
        record["synthesized_view_paths"] = accepted_paths
    if rejected_images:
        rejected_dir = bundle_root / "rejected_geometry_views"
        rejected_dir.mkdir(exist_ok=True)
        budget = 2 * 1024 * 1024
        for row in rejected_images:
            if budget <= 0:
                break
            rejected_path = rejected_dir / f"{row['label']}_a{row['attempt']}.webp"
            try:
                row["image"].convert("RGB").save(rejected_path, format="WEBP", quality=80)
                budget -= rejected_path.stat().st_size
            except Exception:
                break


def _generate_references_with_replay(
    mesh: Any,
    source_rgba: Any,
    *,
    owner: Any,
    angles: Any,
    replay_sources: Mapping[str, bytes],
    **refgen_kwargs: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run `generate_reference_views` with a replay-first generator.

    For each requested angle whose label has a pre-shape synthesized view,
    the FIRST ladder attempt replays that view's raw bytes; every later
    attempt (and every angle without a replay) delegates to the real i2i
    generator. The point: the synthesized geometry views are OFFERED to
    the texture lane through the full, unmodified acceptance machinery
    (matting, clay-silhouette registration + IoU, texture/material/
    specular gates, whole-bake A/B) — nothing is bypassed, and a view the
    mesh diverged from simply fails the clay IoU gate and is regenerated
    by the normal ladder.
    """
    from ..reference_generation import default_i2i_generator, generate_reference_views

    delegate_cache: Dict[str, Any] = {}

    def _delegate(prompt: str, image: Any, **kwargs: Any) -> Any:
        if "generator" not in delegate_cache:
            delegate_cache["generator"] = default_i2i_generator(owner)
        return delegate_cache["generator"](prompt, image, **kwargs)

    views: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    pending: List[Any] = []
    for angle in angles:
        label = str(angle[0])
        raw = replay_sources.get(label)
        if raw is None:
            pending.append(angle)
            continue
        state = {"served": False}

        def _replay_first(prompt: str, image: Any, *, _raw=raw, _state=state, **kwargs: Any) -> Any:
            if not _state["served"]:
                _state["served"] = True
                return _raw
            return _delegate(prompt, image, **kwargs)

        angle_views, angle_report = generate_reference_views(
            mesh, source_rgba, owner=owner, angles=[angle],
            image_generator=_replay_first, **refgen_kwargs,
        )
        views.extend(angle_views)
        reports.append(angle_report)
    if pending:
        batch_views, batch_report = generate_reference_views(
            mesh, source_rgba, owner=owner, angles=pending, **refgen_kwargs,
        )
        views.extend(batch_views)
        reports.append(batch_report)

    if not reports:
        return views, {"angles": [], "accepted": 0, "rejected": 0}
    merged = dict(reports[0])
    merged["angles"] = [row for report in reports for row in (report.get("angles") or [])]
    merged["accepted"] = sum(int(report.get("accepted") or 0) for report in reports)
    merged["rejected"] = sum(int(report.get("rejected") or 0) for report in reports)
    rejected_images = [
        image for report in reports for image in (report.get("rejected_images") or [])
    ]
    if rejected_images:
        merged["rejected_images"] = rejected_images
    elif "rejected_images" in merged:
        del merged["rejected_images"]
    # Provenance: which angles started from a replayed pre-shape view (the
    # per-angle rows record the replayed payload's md5 on acceptance).
    merged["replayed_labels"] = sorted(
        str(angle[0]) for angle in angles if str(angle[0]) in replay_sources
    )
    return views, merged


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
                "shape_candidates": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Best-of-N shape selection: draw the shape stage N "
                        "times with spaced seeds (base + 1000*i), rank the "
                        "postprocessed candidates by photo agreement "
                        "(silhouette + concavity IoU) plus topology, and "
                        "keep the best (surface smoothness is recorded per "
                        "candidate but carries no score weight). Default 1 "
                        "(single draw, exactly the historical behavior). "
                        "Each extra candidate adds about one shape-stage "
                        "time (~21-28 min measured on MPS at octree 512); "
                        "ranking itself takes seconds. The texture stage "
                        "always uses the base seed."
                    ),
                },
                "geometry_conditioning": {
                    "type": "string",
                    "enum": ["single", "multiview", "auto"],
                    "description": (
                        "Shape-stage conditioning for single-photo flows. "
                        "'single' (default) is the historical one-view path. "
                        "'multiview' synthesizes the missing canonical views "
                        "(back, side_left, side_right) from the source photo "
                        "with the local i2i generator, gates them (subject "
                        "identity + silhouette plausibility; a wrong view is "
                        "worse than none), and conditions the Hunyuan3D-2mv "
                        "checkpoint on the survivors — falling back LOUDLY "
                        "to single-view when none survive. 'auto' does the "
                        "same only when an explicitly configured image "
                        "provider exists. Person subjects are refused "
                        "without texture_reference_allow_person. Views that "
                        "pass the texture-lane gates are also offered to "
                        "the texture bake."
                    ),
                },
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

        # Multi-view geometry conditioning mode. Validated here (cheap
        # string checks, before any model load) so a bad value fails in
        # milliseconds; the synthesis itself runs AFTER the strict
        # unknown-option check for the same reason.
        geometry_mode = str(
            kwargs.pop("geometry_conditioning", None)
            or _owner_cfg(self._owner, "scene3d_hunyuan_geometry_conditioning")
            or _env("ABSTRACT3D_HUNYUAN_GEOMETRY_CONDITIONING")
            or "single"
        ).strip().lower()
        if geometry_mode not in _GEOMETRY_CONDITIONING_MODES:
            raise InvalidRequestError(
                "geometry_conditioning must be one of: "
                f"{', '.join(_GEOMETRY_CONDITIONING_MODES)} (got {geometry_mode!r})"
            )
        geometry_conditioning_record: Optional[Dict[str, Any]] = None
        geometry_fallback_reason: Optional[str] = None
        geometry_warnings: List[str] = []
        multiview_active = geometry_mode in {"multiview", "auto"}
        if multiview_active:
            geometry_conditioning_record = {
                "requested": geometry_mode,
                "applied": "single_view",
                "fallback_reason": None,
                "synthesized_views": [],
            }
            # An explicit non-mv model contradicts an explicit multiview
            # request: fail loudly rather than silently overriding either.
            # "auto" is a preference, not a demand — it yields to the
            # explicit model choice with a recorded reason.
            explicit_repo: Optional[str] = None
            if model is not None:
                explicit_repo, _explicit_sub = _resolve_model_selection(model, model_subfolder)
            if explicit_repo == _OFFICIAL_MODEL_ID:
                if geometry_mode == "multiview":
                    raise InvalidRequestError(
                        "geometry_conditioning='multiview' requires the multi-view "
                        f"checkpoint {_MV_MODEL_ID!r}, but the model was explicitly "
                        f"set to {model!r}. Drop the explicit model (it is selected "
                        "automatically) or set geometry_conditioning='single'."
                    )
                multiview_active = False
                geometry_fallback_reason = (
                    "geometry_conditioning=auto yielded to the explicitly "
                    f"requested single-view model {model!r}"
                )
                geometry_warnings.append(
                    "geometry_conditioning fell back to single-view: "
                    + geometry_fallback_reason
                )
            elif geometry_mode == "multiview" and not has_image_composer(self._owner):
                # Explicit mode: missing tooling is a hard error, exactly
                # like texture_reference_generation="on".
                raise DependencyUnavailableError(
                    "geometry_conditioning='multiview' synthesizes conditioning "
                    f"views with the image composer. {COMPOSITION_INSTALL_HINT}"
                )
            elif geometry_mode == "auto":
                from ..reference_generation import auto_generation_ready

                ready, readiness_reason = auto_generation_ready(self._owner)
                if not ready:
                    # "auto" is a default-shaped option: it must never
                    # silently route the user's photo to a provider they
                    # never configured.
                    multiview_active = False
                    geometry_fallback_reason = (
                        f"geometry_conditioning=auto skipped: {readiness_reason}. "
                        "Configure a local image provider "
                        "(scene3d_image_provider / ABSTRACT3D_IMAGE_PROVIDER) "
                        "or set geometry_conditioning=multiview."
                    )
                    geometry_warnings.append(
                        "geometry_conditioning fell back to single-view: "
                        + geometry_fallback_reason
                    )

        if multiview_active:
            # Defer the DiT load until the conditioning views exist: the
            # i2i synthesis and the shape DiT otherwise co-reside on the
            # same unified-memory pool (the t23d composition stage frees
            # MLX before the load for exactly this reason). Device and
            # dtype resolve deterministically without loading.
            pipeline = None
            resolved_device = _select_device(self._owner, device)
            resolved_dtype = _select_dtype(resolved_device, dtype)
            multiview_capable = True
        else:
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
        # through the texture bake. Caller-provided references take the
        # tags first (real photos outrank derived synthesis); the
        # synthesized views (multiview mode, after the strict option
        # check) fill only the tags that remain open.
        geometry_condition: Any = source_image
        geometry_views_used: List[Dict[str, Any]] = [{"tag": "front", "label": "source"}]
        mv_image_dict: Dict[str, Any] = {"front": source_image}
        if multiview_capable and loaded_references:
            for reference in loaded_references:
                azimuth = float(reference["azimuth_deg"])
                snapped = _mv_snap_tag(azimuth)
                if snapped is None or snapped[0] in mv_image_dict:
                    continue
                tag, delta = snapped
                mv_image_dict[tag] = reference["rgba"]
                geometry_views_used.append(
                    {
                        "tag": tag,
                        "label": reference["label"],
                        "declared_azimuth_deg": azimuth,
                        "snap_delta_deg": round(delta, 2),
                    }
                )
            mv_dropped_views = _mv_cap_conditioning_views(
                mv_image_dict, geometry_views_used, geometry_warnings
            )
            if mv_dropped_views and geometry_conditioning_record is not None:
                geometry_conditioning_record["dropped_views"] = mv_dropped_views
            if len(mv_image_dict) > 1:
                geometry_condition = mv_image_dict
        preprocess_s = round(time.perf_counter() - preprocess_started, 4)

        defaults = _resolve_generation_defaults(self._owner, device=resolved_device)

        def _pop_number(key: str, default: Any, cast: Any) -> Any:
            # `None` means "use the default"; explicit falsy values such as
            # guidance_scale=0.0 or seed=0 are legitimate and must survive.
            value = kwargs.pop(key, None)
            return cast(default) if value is None else cast(value)

        # Model-family defaults: the built-in 512/50 regime is calibrated
        # for the 2.1 flagship; the 2mv family gets its own (see the
        # _MV_DEFAULT_* rationale). Explicit caller options and configured
        # owner defaults always win, so record explicitness BEFORE popping.
        steps_explicitly_set = (
            kwargs.get("num_inference_steps") is not None
            or _owner_cfg(self._owner, "scene3d_hunyuan_num_inference_steps") is not None
        )
        num_inference_steps = max(1, _pop_number("num_inference_steps", defaults["num_inference_steps"], int))
        guidance_scale = _pop_number("guidance_scale", defaults["guidance_scale"], float)
        # The generic CLI exposes --mc-resolution; for this backend the grid
        # resolution knob is the octree resolution, so honor the alias when
        # no explicit octree_resolution was given (mirrors the Step1X CLI
        # contract).
        mc_resolution_alias = kwargs.pop("mc_resolution", None)
        if kwargs.get("octree_resolution") is None and mc_resolution_alias is not None:
            kwargs["octree_resolution"] = mc_resolution_alias
        octree_explicitly_set = (
            kwargs.get("octree_resolution") is not None
            or _owner_cfg(self._owner, "scene3d_hunyuan_octree_resolution") is not None
        )
        octree_resolution = max(32, _pop_number("octree_resolution", defaults["octree_resolution"], int))
        if multiview_capable and not multiview_active:
            # An explicitly selected 2mv checkpoint (caller-reference path):
            # run the family's validated regime unless overridden. The
            # flagship regime was measured catastrophic on 2mv (822 raw
            # bodies at 512/50 vs a healthy mesh at 384/30 on the same
            # conditioning).
            if not steps_explicitly_set:
                num_inference_steps = _MV_DEFAULT_NUM_INFERENCE_STEPS
            if not octree_explicitly_set:
                octree_resolution = _MV_DEFAULT_OCTREE_RESOLUTION
        num_chunks = max(1024, _pop_number("num_chunks", defaults["num_chunks"], int))
        max_facenum = _pop_number("max_facenum", defaults["max_facenum"], int)
        seed_value = kwargs.pop("seed", None)
        if seed_value is None:
            seed_value = kwargs.pop("image_seed", None)
        else:
            kwargs.pop("image_seed", None)
        seed = _DEFAULT_SEED if seed_value is None else int(seed_value)
        # Best-of-N shape selection. Validated on the RESOLVED value so a
        # nonsensical config default fails as loudly as a bad option (0
        # draws is a request for nothing; the strict-contract doctrine
        # forbids silently clamping intent).
        try:
            shape_candidates = _pop_number("shape_candidates", defaults["shape_candidates"], int)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(
                f"shape_candidates must be an integer >= 1 (got {exc})"
            ) from exc
        if shape_candidates < 1:
            raise InvalidRequestError(
                f"shape_candidates must be an integer >= 1 (got {shape_candidates})"
            )
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

        # -- pre-shape geometry-view synthesis (multiview conditioning) ------
        # Runs AFTER the strict unknown-option check (a typo must fail in
        # milliseconds, not after minutes of synthesis) and BEFORE the DiT
        # load (deferred above), so the i2i pool and the DiT never co-reside.
        geometry_synthesis_s: Optional[float] = None
        synthesized_geometry_views: List[Dict[str, Any]] = []
        geometry_rejected_images: List[Dict[str, Any]] = []
        if multiview_active:
            synthesis_started = time.perf_counter()
            geometry_subject_hint = str(prompt or "").strip() or None
            # Tags still open after caller-provided references claimed
            # theirs; only these are synthesized — and only when the view
            # cap leaves conditioning room (a view that cannot condition
            # is synthesized later by the texture lane itself if needed).
            # When room exists, ALL open canonical tags are synthesized
            # even if the cap will drop one at merge: the side-pair gate
            # needs both sides to cross-check, and dropped views still
            # reach the texture bake through the replay.
            missing_views = []
            if len(mv_image_dict) < _MV_MAX_CONDITIONING_VIEWS:
                for view_label, view_azimuth in _GEOMETRY_VIEW_ANGLES:
                    snapped = _mv_snap_tag(view_azimuth)
                    if snapped is not None and snapped[0] not in mv_image_dict:
                        missing_views.append((view_label, view_azimuth))
            elif geometry_conditioning_record is not None:
                geometry_conditioning_record["synthesis_skipped"] = (
                    "conditioning view budget already filled by caller references"
                )
            proceed, person_record = _geometry_person_gate(
                source_image,
                subject_hint=geometry_subject_hint,
                allow_person=texture_reference_allow_person,
            )
            geometry_conditioning_record["person_check"] = person_record
            if not proceed:
                geometry_fallback_reason = str(person_record.get("refusal"))
            elif missing_views:
                from ..captioning import extract_subject_noun

                subject_noun = extract_subject_noun(
                    geometry_subject_hint or person_record.get("caption")
                )
                geometry_conditioning_record["subject_noun"] = subject_noun
                try:
                    (
                        synthesized_geometry_views,
                        synthesized_view_records,
                        geometry_rejected_images,
                    ) = _synthesize_geometry_views(
                        self._owner,
                        source_image,
                        subject_noun=subject_noun,
                        base_seed=seed,
                        labels=missing_views,
                    )
                    geometry_conditioning_record["synthesized_views"] = (
                        synthesized_view_records
                    )
                except Exception as exc:
                    if geometry_mode == "multiview":
                        # Explicit mode: a broken synthesis stack is the
                        # caller's problem to see, not to discover in the
                        # metadata after a single-view run they did not ask
                        # for (same contract as texture_reference_generation
                        # "on").
                        raise
                    geometry_fallback_reason = (
                        "geometry view synthesis failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                for view in synthesized_geometry_views:
                    snapped = _mv_snap_tag(view["azimuth_deg"])
                    if snapped is None or snapped[0] in mv_image_dict:
                        continue
                    mv_image_dict[snapped[0]] = view["rgba"]
                    geometry_views_used.append(
                        {
                            "tag": snapped[0],
                            "label": f"synthesized_{view['label']}",
                            "declared_azimuth_deg": float(view["azimuth_deg"]),
                            "snap_delta_deg": round(snapped[1], 2),
                            "synthesized": True,
                            "seed": int(view["seed"]),
                            "raw_payload_md5": view["raw_payload_md5"],
                        }
                    )
                if not synthesized_geometry_views and geometry_fallback_reason is None:
                    geometry_fallback_reason = (
                        "all synthesized conditioning views failed the "
                        "acceptance gates (a wrong conditioning view is worse "
                        "than single-view; see geometry_conditioning."
                        "synthesized_views for per-view reasons)"
                    )
                dropped_rows = _mv_cap_conditioning_views(
                    mv_image_dict, geometry_views_used, geometry_warnings
                )
                if dropped_rows:
                    geometry_conditioning_record["dropped_views"] = dropped_rows
            # Free the i2i pool before the DiT allocates on the same
            # unified-memory budget (mirrors the t23d composition stage).
            from .step1x_runtime import _release_mlx_generation_cache

            _release_mlx_generation_cache()
            if len(mv_image_dict) > 1:
                geometry_condition = mv_image_dict
                effective_model = model if model is not None else _MV_MODEL_ID
                # The 2mv family runs its own validated regime (see the
                # _MV_DEFAULT_* rationale; flagship 512/50 measured
                # catastrophic on this checkpoint). Explicit caller options
                # and configured owner defaults still win.
                if not steps_explicitly_set:
                    num_inference_steps = _MV_DEFAULT_NUM_INFERENCE_STEPS
                if not octree_explicitly_set:
                    octree_resolution = _MV_DEFAULT_OCTREE_RESOLUTION
            else:
                # LOUD single-view fallback: warning + machine-readable
                # record; the shape stage runs the exact known-good
                # single-view path (explicit model choices are respected —
                # an explicitly chosen 2mv checkpoint keeps its family
                # regime even single-view).
                geometry_condition = source_image
                effective_model = model
                if model is not None and _resolve_model_selection(model, model_subfolder)[0] == _MV_MODEL_ID:
                    if not steps_explicitly_set:
                        num_inference_steps = _MV_DEFAULT_NUM_INFERENCE_STEPS
                    if not octree_explicitly_set:
                        octree_resolution = _MV_DEFAULT_OCTREE_RESOLUTION
                geometry_warnings.append(
                    "geometry_conditioning fell back to single-view: "
                    + str(geometry_fallback_reason or "no conditioning views available")
                )
            pipeline = self._load_runtime(
                model_id=effective_model,
                device=device,
                dtype=dtype,
                model_subfolder=model_subfolder,
            )
            resolved_device = self._resident_device or "cpu"
            resolved_dtype = self._resident_dtype or "float32"
            multiview_capable = bool(self._last_runtime_stats.get("multiview_capable"))
            geometry_conditioning_record["applied"] = (
                "multiview"
                if multiview_capable and len(geometry_views_used) > 1
                else "single_view"
            )
            if geometry_conditioning_record["applied"] == "multiview":
                # Conditioning applied (caller references and/or synthesis):
                # any earlier refusal (e.g. the person gate skipping the
                # SYNTHESIS while real reference photos carry the tags)
                # stays visible in its own record, not as a fallback.
                geometry_fallback_reason = None
            geometry_synthesis_s = round(time.perf_counter() - synthesis_started, 4)

        source_dir = Path(self._last_runtime_stats["source_dir"])
        canonicalize = _owner_cfg_bool(self._owner, "scene3d_hunyuan_canonicalize_export_axes", True)
        # Best-of-N state. For the default N=1 nothing below the loop body
        # changes: no matte extraction, no ranking render, no extra
        # metadata — the single-draw path is the historical one.
        ranking_matte: Any = None
        shape_candidate_rows: List[Dict[str, Any]] = []
        candidate_stream_warnings: List[str] = []
        shape_selection_s = 0.0
        best_candidate: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        if shape_candidates > 1:
            ranking_matte = _photo_matte_mask(source_image)
            if ranking_matte is None:
                candidate_stream_warnings.append(
                    "shape candidate ranking ran without photo agreement "
                    "(source matte unusable: no alpha support or unsegmented "
                    "frame); candidates ranked on topology only"
                )
        inference_elapsed = 0.0
        mesh_elapsed = 0.0
        segment_started = time.perf_counter()
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
            inference_elapsed += time.perf_counter() - segment_started
            # Candidates run SEQUENTIALLY by design: on MPS the DiT + VAE
            # already fill most of the unified-memory pool, and one raw
            # marching-cubes mesh peaks at hundreds of MB — only the
            # current draw and the best-so-far survivor are held at once.
            for candidate_index in range(shape_candidates):
                candidate_seed = seed + _SHAPE_CANDIDATE_SEED_STRIDE * candidate_index
                draw_started = time.perf_counter()
                generator = torch.Generator(device="cpu").manual_seed(candidate_seed)
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
                candidate_inference_s = time.perf_counter() - draw_started
                inference_elapsed += candidate_inference_s
                raw_mesh = meshes[0] if isinstance(meshes, list) else meshes
                del meshes
                if raw_mesh is None:
                    if shape_candidates == 1:
                        raise Abstract3DError(
                            "Hunyuan3D-2.1 produced no surface at the requested settings. "
                            "Try more inference steps or a different seed."
                        )
                    # A failed draw is evidence, not an abort: record it in
                    # the candidates array (an unranked discard would be the
                    # same evidence destruction this project already made
                    # once with rejected reference views).
                    shape_candidate_rows.append(
                        {
                            "seed": candidate_seed,
                            "selected": False,
                            "status": "no_surface",
                            "score": None,
                            "inference_s": round(candidate_inference_s, 4),
                        }
                    )
                    candidate_stream_warnings.append(
                        f"shape candidate {candidate_index + 1}/{shape_candidates} "
                        f"(seed {candidate_seed}) produced no surface and was discarded"
                    )
                    continue

                candidate_mesh_started = time.perf_counter()
                candidate_mesh, candidate_applied, candidate_warnings = _hunyuan_postprocess_mesh(
                    raw_mesh,
                    max_facenum=max_facenum,
                )
                # Raw (pre-cleanup) topology: without it a surfacing
                # regression is invisible — the shipped numbers conflate
                # decoder output with floater removal and decimation
                # (measured: raw euler -110 -> post -124 on a car, fully
                # explained by junk removal).
                candidate_topology_raw = _mesh_topology(raw_mesh)
                del raw_mesh
                candidate_axis_applied: List[str] = []
                if canonicalize:
                    candidate_mesh, candidate_axis_applied = _hunyuan_canonicalize_axes(candidate_mesh)
                mesh_elapsed += time.perf_counter() - candidate_mesh_started
                # Ground-slab measurements travel on the trimesh metadata
                # dict; pop them here so no export path serializes a private
                # field into glTF extras, then surface them in the run
                # metadata below.
                try:
                    candidate_slab_report = candidate_mesh.metadata.pop("abstract3d_ground_slab", None)
                except Exception:
                    candidate_slab_report = None
                candidate_state: Dict[str, Any] = {
                    "mesh": candidate_mesh,
                    "applied": candidate_applied,
                    "warnings": candidate_warnings,
                    "topology_raw": candidate_topology_raw,
                    "axis_applied": candidate_axis_applied,
                    "ground_slab": candidate_slab_report,
                    "seed": candidate_seed,
                }
                if shape_candidates == 1:
                    best_candidate = candidate_state
                    continue

                ranking_started = time.perf_counter()
                # Ranking always measures the render-convention frame; when
                # export canonicalization is disabled the candidate stays in
                # the Hunyuan native frame and a rotated copy is ranked.
                ranking_mesh = candidate_mesh
                if not canonicalize:
                    ranking_mesh, _ = _hunyuan_canonicalize_axes(candidate_mesh)
                metrics = evaluate_shape_candidate(ranking_mesh, matte_mask=ranking_matte)
                candidate_score = score_shape_candidate(metrics)
                candidate_ranking_s = time.perf_counter() - ranking_started
                shape_selection_s += candidate_ranking_s
                row: Dict[str, Any] = {
                    "seed": candidate_seed,
                    "selected": False,
                    "status": "ranked",
                    "score": round(candidate_score, 4),
                    "inference_s": round(candidate_inference_s, 4),
                    "ranking_s": round(candidate_ranking_s, 4),
                    "face_count": int(len(candidate_mesh.faces)),
                    "vertex_count": int(len(candidate_mesh.vertices)),
                    "topology_raw": candidate_topology_raw,
                    "postprocess_cleanup": list(candidate_applied),
                    "postprocess_warnings": list(candidate_warnings),
                    "ground_slab": candidate_slab_report,
                }
                row.update(metrics)
                shape_candidate_rows.append(row)
                candidate_state["row"] = row
                # Strictly-greater comparison: on an exact tie the EARLIER
                # candidate (closest to the base seed) wins, so adding
                # candidates never changes a result it cannot improve.
                if best_score is None or candidate_score > best_score:
                    if best_candidate is not None:
                        best_candidate["mesh"] = None
                    best_score = candidate_score
                    best_candidate = candidate_state
                else:
                    candidate_state["mesh"] = None
                candidate_mesh = None
                ranking_mesh = None
                if resolved_device == "mps":
                    # Transient DiT/VAE buffers accumulate across draws on
                    # the unified-memory pool; the weights stay resident.
                    try:
                        import gc

                        gc.collect()
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        if best_candidate is None:
            raise Abstract3DError(
                f"Hunyuan3D-2.1 produced no surface in any of {shape_candidates} "
                "candidate draws at the requested settings. Try more inference "
                "steps or a different base seed."
            )
        mesh = best_candidate["mesh"]
        postprocess_applied = best_candidate["applied"]
        postprocess_warnings = best_candidate["warnings"]
        topology_raw = best_candidate["topology_raw"]
        axis_applied = best_candidate["axis_applied"]
        ground_slab_report: Optional[Dict[str, Any]] = best_candidate["ground_slab"]
        selected_shape_seed = int(best_candidate["seed"])
        selected_row = best_candidate.get("row")
        if selected_row is not None:
            selected_row["selected"] = True
        postprocess_warnings.extend(candidate_stream_warnings)
        inference_s = round(inference_elapsed, 4)
        mesh_s = round(mesh_elapsed, 4)

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
        postprocess_warnings.extend(geometry_warnings)
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

                        refgen_angles = texture_reference_generation_angles or DEFAULT_ANGLES
                        refgen_kwargs = dict(
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
                        # Pre-shape synthesized geometry views are OFFERED
                        # to the texture lane as the first ladder attempt
                        # of their angle — through the full acceptance
                        # machinery (clay IoU, material gates, whole-bake
                        # A/B), never around it. A view the mesh diverged
                        # from fails the clay gate and the ladder
                        # regenerates normally.
                        replay_sources = {
                            str(view["label"]): view["raw_bytes"]
                            for view in synthesized_geometry_views
                            if view.get("raw_bytes")
                        }
                        if replay_sources:
                            generated_views, reference_generation_report = (
                                _generate_references_with_replay(
                                    mesh,
                                    source_image.convert("RGBA"),
                                    owner=self._owner,
                                    angles=refgen_angles,
                                    replay_sources=replay_sources,
                                    **refgen_kwargs,
                                )
                            )
                        else:
                            generated_views, reference_generation_report = generate_reference_views(
                                mesh,
                                source_image.convert("RGBA"),
                                owner=self._owner,
                                angles=refgen_angles,
                                **refgen_kwargs,
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

        # Fail-closed ground-slab contract: a detected slab that the cut
        # budget refused to remove ships with the defect still fused to the
        # mesh — that must be visible in one machine-readable field, not
        # only in the warning stream. A cleanly cut slab demotes nothing.
        if ground_slab_report is not None and ground_slab_report.get("action") == "refused":
            slab_reason = (
                "ground slab detected but not removed: "
                f"{ground_slab_report.get('refusal_reason')}"
            )
            if quality_verdict.get("verdict") == "healthy":
                quality_verdict = {"verdict": "degraded", "reasons": [slab_reason]}
            else:
                quality_verdict.setdefault("reasons", []).append(slab_reason)

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
            (image_generation_s or 0.0)
            + preprocess_s
            + (geometry_synthesis_s or 0.0)
            + inference_s
            + mesh_s
            + shape_selection_s
            + (texture_s or 0.0),
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
            "ground_slab": ground_slab_report,
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
        if shape_candidate_rows:
            # Best-of-N evidence (N > 1 only — the default single-draw
            # metadata is unchanged): every candidate's seed, ranking
            # metrics, and postprocess record, selected flag included.
            # "seed" above stays the BASE seed (the texture stage and
            # reference generation keep using it); "shape_seed" is the
            # seed that actually drew the selected shape.
            runtime_meta["shape_candidates"] = shape_candidate_rows
            runtime_meta["shape_seed"] = selected_shape_seed
            runtime_meta["timings_s"]["shape_selection"] = round(shape_selection_s, 4)
        if geometry_conditioning_record is not None:
            # Multiview evidence (mode != "single" only — the default
            # metadata is unchanged): requested vs applied mode, the
            # person check, per-view synthesis attempts with gate
            # metrics, and the fallback reason when conditioning did not
            # apply.
            geometry_conditioning_record["fallback_reason"] = geometry_fallback_reason
            runtime_meta["geometry_conditioning"] = geometry_conditioning_record
            runtime_meta["timings_s"]["geometry_view_synthesis"] = geometry_synthesis_s
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
            _persist_geometry_conditioning_artifacts(
                bundle_root,
                accepted_views=synthesized_geometry_views,
                rejected_images=geometry_rejected_images,
                record=geometry_conditioning_record,
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
                _persist_geometry_conditioning_artifacts(
                    bundle_root,
                    accepted_views=synthesized_geometry_views,
                    rejected_images=geometry_rejected_images,
                    record=geometry_conditioning_record,
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
