"""Local TRELLIS.2 backend with Apple-safe sparse compatibility shims."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..artifacts import is_artifact_ref, stable_artifact_id, store_bytes
from ..errors import (
    Abstract3DError,
    CapabilityNotSupportedError,
    DependencyUnavailableError,
    SourceBootstrapError,
)
from ..image_composition import COMPOSITION_INSTALL_HINT, default_image_generator, has_image_composer, pop_image_generation_request
from ..model_catalog import capability_model_records
from ..rendering import build_case_contact_sheet, render_mesh_views, stack_contact_sheets


_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "abstract3d"
_TRELLIS2_REPO_URL = "https://github.com/microsoft/TRELLIS.2"
_TRELLIS2_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
_SOURCE_MANIFEST = ".abstract3d-source.json"
_OFFICIAL_MODEL_ID = "microsoft/TRELLIS.2-4B"
_DEFAULT_MODEL_ID = _OFFICIAL_MODEL_ID
_SPARSE_STRUCTURE_DECODER_ID = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
_DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_DINO_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
_REMBG_MODEL_ID = "briaai/RMBG-2.0"
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
    "transformers",
    "huggingface_hub",
    "safetensors",
    "torchvision",
    "trimesh",
    "PIL",
    "psutil",
)
_PIPELINE_CACHE: Optional[Dict[str, Any]] = None


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _owner_cfg(owner: Any, key: str) -> Optional[Any]:
    cfg = getattr(owner, "config", None)
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    return None


def _owner_cfg_int(owner: Any, key: str, default: int) -> int:
    raw = _owner_cfg(owner, key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _cache_root(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_cache_dir") or _env("ABSTRACT3D_CACHE_DIR")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return _DEFAULT_CACHE_ROOT


def _require_runtime_dependencies() -> None:
    missing = [name for name in _RUNTIME_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        raise DependencyUnavailableError(
            "TRELLIS.2 runtime dependencies are missing: "
            + ", ".join(sorted(missing))
            + '. Install with: pip install "abstract3d[trellis2]"'
        )


def _select_device(owner: Any, *, explicit: Optional[str] = None) -> str:
    import torch

    requested = str(explicit or _owner_cfg(owner, "scene3d_device") or _env("ABSTRACT3D_DEVICE") or "auto").strip().lower()
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    cuda_available = getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    if requested == "cpu":
        return "cpu"
    # Explicit accelerator requests fall back to auto-selection when the
    # requested accelerator is not available on this host, so the same
    # command line stays valid across macOS, Linux, and Windows hosts.
    if requested in {"mps", "metal"} and mps_available:
        return "mps"
    if requested == "cuda" and cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def _clone_repo(*, repo_url: str, commit: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="abstract3d-trellis2-", dir=str(repo_dir.parent)))
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(tmp_root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_root), "fetch", "--depth=1", "origin", commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_root), "checkout", "--detach", commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            (tmp_root / ".git").rmdir()
        except Exception:
            pass
        shutil.move(str(tmp_root), str(repo_dir))
        (repo_dir / _SOURCE_MANIFEST).write_text(
            json.dumps(
                {
                    "repo_url": _TRELLIS2_REPO_URL,
                    "commit": _TRELLIS2_COMMIT,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise SourceBootstrapError(exc.stderr or exc.stdout or str(exc)) from exc
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def _normalize_repo_url(repo_url: str) -> str:
    text = str(repo_url or "").strip().lower()
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def _verify_official_source_dir(owner: Any, source_dir: Path) -> Path:
    resolved = source_dir.expanduser().resolve()
    if not (resolved / "trellis2" / "__init__.py").exists():
        raise SourceBootstrapError(f"Configured TRELLIS.2 source dir does not contain a trellis2 package: {resolved}")

    manifest_path = resolved / _SOURCE_MANIFEST
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SourceBootstrapError(f"Could not read TRELLIS.2 source manifest: {manifest_path}") from exc
        repo_url = _normalize_repo_url(manifest.get("repo_url"))
        commit = str(manifest.get("commit") or "").strip()
        if repo_url != _normalize_repo_url(_TRELLIS2_REPO_URL) or commit != _TRELLIS2_COMMIT:
            raise SourceBootstrapError(
                "Configured TRELLIS.2 source dir is not an official pinned snapshot. "
                f"Expected repo {_TRELLIS2_REPO_URL!r} at commit {_TRELLIS2_COMMIT!r}, got repo {repo_url!r} commit {commit!r}."
            )
        return resolved

    git_dir = resolved / ".git"
    if git_dir.exists():
        try:
            repo_url = subprocess.run(
                ["git", "-C", str(resolved), "remote", "get-url", "origin"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "-C", str(resolved), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise SourceBootstrapError(exc.stderr or exc.stdout or str(exc)) from exc
        if _normalize_repo_url(repo_url) != _normalize_repo_url(_TRELLIS2_REPO_URL) or commit != _TRELLIS2_COMMIT:
            raise SourceBootstrapError(
                "Configured TRELLIS.2 source dir is not the official pinned Microsoft checkout. "
                f"Expected repo {_TRELLIS2_REPO_URL!r} at commit {_TRELLIS2_COMMIT!r}."
            )
        return resolved

    bootstrap_dir = (_cache_root(owner) / "vendor" / "trellis2" / _TRELLIS2_COMMIT).resolve()
    if resolved == bootstrap_dir:
        return resolved

    raise SourceBootstrapError(
        "Configured TRELLIS.2 source dir is not verifiable as an official pinned snapshot. "
        "Use the built-in bootstrap path or point scene3d_trellis2_source_dir at an official checkout "
        "with the pinned commit."
    )


def _resolve_source_dir(owner: Any) -> Path:
    configured = _owner_cfg(owner, "scene3d_trellis2_source_dir") or _env("ABSTRACT3D_TRELLIS2_SOURCE_DIR")
    if configured:
        return _verify_official_source_dir(owner, Path(str(configured)))
    repo_dir = _cache_root(owner) / "vendor" / "trellis2" / _TRELLIS2_COMMIT
    _clone_repo(repo_url=_TRELLIS2_REPO_URL, commit=_TRELLIS2_COMMIT, repo_dir=repo_dir)
    return _verify_official_source_dir(owner, repo_dir)


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


def _parse_aabb(aabb: Any, *, device: Any) -> Any:
    import torch
    import numpy as np

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    if not isinstance(aabb, torch.Tensor):
        raise TypeError(f"Unsupported aabb type: {type(aabb)!r}")
    return aabb.to(device=device, dtype=torch.float32)


def _parse_grid_inputs(*, aabb: Any, voxel_size: Any, grid_size: Any, device: Any) -> Tuple[Any, Any]:
    import numpy as np
    import torch

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
        voxel_size = voxel_size.to(device=device, dtype=torch.float32)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        return voxel_size, grid_size
    if grid_size is None:
        raise ValueError("Either voxel_size or grid_size must be provided.")
    if isinstance(grid_size, int):
        grid_size = [grid_size, grid_size, grid_size]
    if isinstance(grid_size, (list, tuple)):
        grid_size = np.array(grid_size)
    if isinstance(grid_size, np.ndarray):
        grid_size = torch.tensor(grid_size, dtype=torch.int32, device=device)
    grid_size = grid_size.to(device=device, dtype=torch.int32)
    voxel_size = (aabb[1] - aabb[0]) / grid_size
    return voxel_size, grid_size


def _local_flexible_dual_grid_to_mesh(
    coords: Any,
    dual_vertices: Any,
    intersected_flag: Any,
    split_weight: Optional[Any],
    aabb: Any,
    voxel_size: Any = None,
    grid_size: Any = None,
    train: bool = False,
):
    import torch

    if train:
        raise NotImplementedError("Training-mode flexible dual grid decode is not implemented in the local TRELLIS.2 shim.")

    device = coords.device
    coords = coords.to(dtype=torch.int32)
    dual_vertices = dual_vertices.to(dtype=torch.float32)
    flags = intersected_flag
    if flags.dtype != torch.bool:
        flags = flags > 0
    aabb_t = _parse_aabb(aabb, device=device)
    voxel_size_t, _ = _parse_grid_inputs(aabb=aabb_t, voxel_size=voxel_size, grid_size=grid_size, device=device)
    mesh_vertices = (coords.float() + dual_vertices) * voxel_size_t + aabb_t[0].reshape(1, 3)

    edge_offsets = (
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    )
    coord_map = {tuple(int(v) for v in row.tolist()): idx for idx, row in enumerate(coords.detach().cpu())}
    flags_cpu = flags.detach().cpu()
    coords_cpu = coords.detach().cpu()
    quads: List[List[int]] = []
    for row_index, (row, row_flags) in enumerate(zip(coords_cpu, flags_cpu)):
        base = tuple(int(v) for v in row.tolist())
        for axis in range(3):
            if not bool(row_flags[axis]):
                continue
            quad: List[int] = []
            for delta in edge_offsets[axis]:
                key = (base[0] + delta[0], base[1] + delta[1], base[2] + delta[2])
                neighbor_index = coord_map.get(key)
                if neighbor_index is None:
                    quad = []
                    break
                quad.append(int(neighbor_index))
            if len(quad) == 4:
                quads.append(quad)
    if not quads:
        return mesh_vertices, torch.empty((0, 3), dtype=torch.long, device=device)

    quad_indices = torch.tensor(quads, dtype=torch.long, device=device)
    if split_weight is None:
        attempt0 = quad_indices[:, [0, 1, 2, 0, 2, 3]]
        attempt1 = quad_indices[:, [0, 1, 3, 3, 1, 2]]
        normals0_a = torch.cross(
            mesh_vertices[attempt0[:, 1]] - mesh_vertices[attempt0[:, 0]],
            mesh_vertices[attempt0[:, 2]] - mesh_vertices[attempt0[:, 0]],
            dim=1,
        )
        normals0_b = torch.cross(
            mesh_vertices[attempt0[:, 2]] - mesh_vertices[attempt0[:, 1]],
            mesh_vertices[attempt0[:, 3]] - mesh_vertices[attempt0[:, 1]],
            dim=1,
        )
        normals1_a = torch.cross(
            mesh_vertices[attempt1[:, 1]] - mesh_vertices[attempt1[:, 0]],
            mesh_vertices[attempt1[:, 2]] - mesh_vertices[attempt1[:, 0]],
            dim=1,
        )
        normals1_b = torch.cross(
            mesh_vertices[attempt1[:, 2]] - mesh_vertices[attempt1[:, 1]],
            mesh_vertices[attempt1[:, 3]] - mesh_vertices[attempt1[:, 1]],
            dim=1,
        )
        align0 = (normals0_a * normals0_b).sum(dim=1).abs()
        align1 = (normals1_a * normals1_b).sum(dim=1).abs()
        choose0 = (align0 > align1).unsqueeze(1)
        triangles = torch.where(choose0, attempt0, attempt1).reshape(-1, 3)
    else:
        split_weight = split_weight.to(device=device, dtype=torch.float32).reshape(-1)
        quad_weights = split_weight[quad_indices]
        choose0 = (quad_weights[:, 0] * quad_weights[:, 2] > quad_weights[:, 1] * quad_weights[:, 3]).unsqueeze(1)
        split0 = quad_indices[:, [0, 1, 2, 0, 2, 3]]
        split1 = quad_indices[:, [0, 1, 3, 3, 1, 2]]
        triangles = torch.where(choose0, split0, split1).reshape(-1, 3)
    return mesh_vertices, triangles


def _build_conv_none_module() -> types.ModuleType:
    import torch
    import torch.nn as nn

    module = types.ModuleType("trellis2.modules.sparse.conv_none")

    def _offsets(kernel_size: Tuple[int, int, int], dilation: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        ranges = []
        for size, dil in zip(kernel_size, dilation):
            radius = size // 2
            ranges.append([index * dil for index in range(-radius, radius + 1)])
        return [(a, b, c) for a in ranges[0] for b in ranges[1] for c in ranges[2]]

    def _neighbor_map(x: Any, kernel_size: Tuple[int, int, int], dilation: Tuple[int, int, int]):
        cache_key = f"SubMConv3d_neighbor_cache_{kernel_size[2]}x{kernel_size[1]}x{kernel_size[0]}_dilation{dilation}"
        cached = x.get_spatial_cache(cache_key)
        if cached is not None:
            return cached
        coords_cpu = x.coords.detach().cpu().to(dtype=torch.int32)
        coord_map = {tuple(int(v) for v in row.tolist()): idx for idx, row in enumerate(coords_cpu)}
        kernel_offsets = _offsets(kernel_size, dilation)
        neighbors = torch.full((coords_cpu.shape[0], len(kernel_offsets)), -1, dtype=torch.long)
        for row_index, row in enumerate(coords_cpu):
            base = tuple(int(v) for v in row.tolist())
            for kernel_index, delta in enumerate(kernel_offsets):
                key = (base[0], base[1] + delta[0], base[2] + delta[1], base[3] + delta[2])
                neighbor_index = coord_map.get(key)
                if neighbor_index is not None:
                    neighbors[row_index, kernel_index] = int(neighbor_index)
        x.register_spatial_cache(cache_key, neighbors)
        return neighbors

    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        del indice_key
        if stride != 1 or padding is not None:
            raise NotImplementedError("The local TRELLIS.2 sparse shim only supports stride=1 submanifold convolutions.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (int(kernel_size),) * 3
        self.stride = (1, 1, 1)
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (int(dilation),) * 3
        self.weight = nn.Parameter(torch.empty((self.out_channels, self.in_channels, *self.kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in else 0.0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def sparse_conv3d_forward(self, x):
        neighbors = _neighbor_map(x, self.kernel_size, self.dilation).to(device=x.feats.device)
        padded = torch.cat(
            [torch.zeros((1, x.feats.shape[1]), dtype=x.feats.dtype, device=x.feats.device), x.feats],
            dim=0,
        )
        gathered = padded[(neighbors + 1).clamp(min=0)]
        weight = self.weight.permute(2, 3, 4, 1, 0).reshape(-1, self.in_channels, self.out_channels).to(dtype=x.feats.dtype)
        out = torch.einsum("tkc,kco->to", gathered, weight)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return x.replace(out)

    def sparse_inverse_conv3d_init(self, *args, **kwargs):
        raise NotImplementedError("SparseInverseConv3d is not required for the local TRELLIS.2 path.")

    def sparse_inverse_conv3d_forward(self, x):
        raise NotImplementedError("SparseInverseConv3d is not required for the local TRELLIS.2 path.")

    module.sparse_conv3d_init = sparse_conv3d_init
    module.sparse_conv3d_forward = sparse_conv3d_forward
    module.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
    module.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
    return module


def _build_representations_module() -> types.ModuleType:
    import torch

    module = types.ModuleType("trellis2.representations")

    class Mesh:
        def __init__(self, vertices, faces, vertex_attrs=None):
            self.vertices = vertices.float()
            self.faces = faces.int()
            self.vertex_attrs = vertex_attrs

        @property
        def device(self):
            return self.vertices.device

        def to(self, device, non_blocking: bool = False):
            attrs = None
            if self.vertex_attrs is not None:
                attrs = self.vertex_attrs.to(device, non_blocking=non_blocking)
            return Mesh(
                self.vertices.to(device, non_blocking=non_blocking),
                self.faces.to(device, non_blocking=non_blocking),
                attrs,
            )

        def cpu(self):
            return self.to("cpu")

        def cuda(self, non_blocking: bool = False):
            return self.to("cuda", non_blocking=non_blocking)

        def fill_holes(self, *args, **kwargs):
            del args, kwargs
            return None

        def simplify(self, *args, **kwargs):
            del args, kwargs
            return None

        def remove_faces(self, face_mask: torch.Tensor):
            keep = ~face_mask.to(dtype=torch.bool, device=self.faces.device)
            self.faces = self.faces[keep]

    class MeshWithVoxel(Mesh):
        pass

    module.Mesh = Mesh
    module.MeshWithVoxel = MeshWithVoxel
    module.MeshWithPbrMaterial = Mesh
    return module


def _install_trellis2_shims() -> None:
    if "easydict" not in sys.modules:
        module = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        module.EasyDict = EasyDict
        sys.modules["easydict"] = module
    if "trellis2.modules.sparse.conv_none" not in sys.modules:
        sys.modules["trellis2.modules.sparse.conv_none"] = _build_conv_none_module()
    if "trellis2.representations" not in sys.modules:
        sys.modules["trellis2.representations"] = _build_representations_module()
    if "o_voxel" not in sys.modules:
        sys.modules["o_voxel"] = types.ModuleType("o_voxel")
    convert_module = types.ModuleType("o_voxel.convert")
    convert_module.flexible_dual_grid_to_mesh = _local_flexible_dual_grid_to_mesh
    sys.modules["o_voxel.convert"] = convert_module
    setattr(sys.modules["o_voxel"], "convert", convert_module)


def _sparse_sdpa(*args, **kwargs):
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    arg_names = {
        1: ["qkv"],
        2: ["q", "kv"],
        3: ["q", "k", "v"],
    }
    total = len(args) + len(kwargs)
    if total not in arg_names:
        raise ValueError(f"Invalid sparse attention argument count: {total}")
    values = {}
    for index, name in enumerate(arg_names[total]):
        if index < len(args):
            values[name] = args[index]
        elif name in kwargs:
            values[name] = kwargs[name]
        else:
            raise ValueError(f"Missing sparse attention argument: {name}")

    q = values.get("q")
    kv = values.get("kv")
    k = values.get("k")
    v = values.get("v")
    qkv = values.get("qkv")

    sparse_q = None
    if qkv is not None:
        sparse_q = qkv
        q_list = [item[:, 0] for item in qkv.to_tensor_list()]
        k_list = [item[:, 1] for item in qkv.to_tensor_list()]
        v_list = [item[:, 2] for item in qkv.to_tensor_list()]
    elif kv is not None:
        if hasattr(q, "to_tensor_list"):
            sparse_q = q
            q_list = q.to_tensor_list()
        else:
            q_list = [tensor for tensor in q]
        if hasattr(kv, "to_tensor_list"):
            kv_list = kv.to_tensor_list()
            k_list = [item[:, 0] for item in kv_list]
            v_list = [item[:, 1] for item in kv_list]
        else:
            k_list = [item[:, 0] for item in kv]
            v_list = [item[:, 1] for item in kv]
    else:
        if hasattr(q, "to_tensor_list"):
            sparse_q = q
            q_list = q.to_tensor_list()
        else:
            q_list = [tensor for tensor in q]
        if hasattr(k, "to_tensor_list"):
            k_list = k.to_tensor_list()
            v_list = v.to_tensor_list()
        else:
            k_list = [tensor for tensor in k]
            v_list = [tensor for tensor in v]

    outputs = []
    for q_i, k_i, v_i in zip(q_list, k_list, v_list):
        out = sdpa(
            q_i.transpose(0, 1).unsqueeze(0),
            k_i.transpose(0, 1).unsqueeze(0),
            v_i.transpose(0, 1).unsqueeze(0),
        )[0].transpose(0, 1)
        outputs.append(out)
    if sparse_q is not None:
        return sparse_q.from_tensor_list(outputs)
    return torch.stack(outputs, dim=0)


def _patch_sparse_attention() -> None:
    full_attn = importlib.import_module("trellis2.modules.sparse.attention.full_attn")
    full_attn.sparse_scaled_dot_product_attention = _sparse_sdpa
    modules = importlib.import_module("trellis2.modules.sparse.attention.modules")
    modules.sparse_scaled_dot_product_attention = _sparse_sdpa
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.ATTN = "sdpa"
    sparse_config.CONV = "none"


def _import_trellis2(source_dir: Path):
    _require_runtime_dependencies()
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    with _sys_path(source_dir):
        _install_trellis2_shims()
        importlib.import_module("trellis2")
        _patch_sparse_attention()
        return {
            "models": importlib.import_module("trellis2.models"),
            "samplers": importlib.import_module("trellis2.pipelines.samplers"),
        }


def _variant_for_model(model_id: str) -> Dict[str, Any]:
    normalized = str(model_id or _DEFAULT_MODEL_ID).strip()
    if normalized == _OFFICIAL_MODEL_ID:
        return {
            "repo_id": normalized,
            "shape_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
            "shape_flow_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
            "sparse_structure_flow": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
            "display_name": "TRELLIS.2 official",
            "precision": "bf16/fp16",
        }
    raise CapabilityNotSupportedError(
        "The local TRELLIS.2 backend currently accepts only the official Microsoft repo: "
        f"{_OFFICIAL_MODEL_ID!r}. Got {normalized!r}."
    )


def _pipeline_args() -> Dict[str, Any]:
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE
    from huggingface_hub import hf_hub_download

    pipeline_path = hf_hub_download(_OFFICIAL_MODEL_ID, "pipeline.json")
    _PIPELINE_CACHE = json.loads(Path(pipeline_path).read_text(encoding="utf-8"))["args"]
    return _PIPELINE_CACHE


def _download_local_model(repo_id: str, relpath: str) -> Path:
    from huggingface_hub import hf_hub_download

    json_path = hf_hub_download(repo_id, f"{relpath}.json")
    model_path = hf_hub_download(repo_id, f"{relpath}.safetensors")
    _ = model_path
    return Path(json_path).with_suffix("")


def _download_local_model_ref(model_ref: str) -> Path:
    parts = str(model_ref).split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid model ref: {model_ref!r}")
    repo_id = "/".join(parts[:2])
    relpath = "/".join(parts[2:])
    return _download_local_model(repo_id, relpath)


def _validate_local_dino_snapshot(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.is_file():
        root = root.parent
    missing = [name for name in _DINO_REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise DependencyUnavailableError(
            "Configured TRELLIS.2 DINOv3 snapshot is incomplete. Missing files: "
            + ", ".join(sorted(missing))
            + f". Expected a local snapshot directory for {_DINO_MODEL_ID!r}, got: {root}"
        )
    return root


def _configured_dino_source(owner: Any) -> str:
    configured = _owner_cfg(owner, "scene3d_trellis2_dino_model") or _env("ABSTRACT3D_TRELLIS2_DINO_MODEL")
    if configured is None:
        return _DINO_MODEL_ID
    value = str(configured).strip()
    if not value:
        return _DINO_MODEL_ID
    local_path = Path(value).expanduser()
    if local_path.exists():
        return str(_validate_local_dino_snapshot(local_path))
    if value != _DINO_MODEL_ID:
        raise CapabilityNotSupportedError(
            "The local TRELLIS.2 backend accepts only the official DINOv3 companion model "
            f"{_DINO_MODEL_ID!r} or a local authorized snapshot directory. Got {value!r}."
        )
    return value


def _download_dino_model(owner: Any) -> Path:
    configured = _configured_dino_source(owner)
    local_path = Path(configured).expanduser()
    if local_path.exists():
        return _validate_local_dino_snapshot(local_path)

    from huggingface_hub import snapshot_download

    try:
        snapshot_dir = snapshot_download(
            repo_id=_DINO_MODEL_ID,
            allow_patterns=list(_DINO_REQUIRED_FILES),
        )
    except Exception as exc:
        raise DependencyUnavailableError(
            "Official TRELLIS.2 requires the official gated DINOv3 companion model "
            f"{_DINO_MODEL_ID!r}, distributed under Meta's DINOv3 License (commercial use permitted; "
            "'Built with DINOv3' attribution required when distributing; military/trade-control uses "
            "prohibited). To proceed: (1) sign in to Hugging Face, open the model page, review and "
            "accept the DINOv3 License (Meta reviews requests, typically within a few days), then "
            "authenticate this machine with `hf auth login` and retry; or (2) point "
            "scene3d_trellis2_dino_model / ABSTRACT3D_TRELLIS2_DINO_MODEL to a local authorized "
            "snapshot directory. Contributor mirrors and alternate encoders are not supported."
        ) from exc
    return _validate_local_dino_snapshot(Path(snapshot_dir))


def _image_conditioner(dino_snapshot: Path, device: str):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel.from_pretrained(str(dino_snapshot), local_files_only=True)
    model.eval()
    model.to(device)
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _extract(images: List[Image.Image], *, image_size: int = 512) -> Any:
        batch = []
        for image in images:
            resized = image.resize((image_size, image_size), Image.LANCZOS)
            array = np.array(resized.convert("RGB")).astype("float32") / 255.0
            batch.append(torch.from_numpy(array).permute(2, 0, 1).float())
        tensor = torch.stack(batch).to(device)
        tensor = transform(tensor)
        tensor = tensor.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(tensor, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(tensor)
        for layer_module in model.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    return model, _extract


def _prepare_image(image: Any, *, remove_background: Optional[bool], foreground_ratio: float):
    from PIL import Image
    from .triposr_runtime import _prepare_triposr_image

    source_preview, prepared_rgb, background_removed = _prepare_triposr_image(
        image,
        remove_background=remove_background,
        foreground_ratio=foreground_ratio,
    )
    if not isinstance(source_preview, Image.Image):
        source_preview = Image.open(io.BytesIO(source_preview)).convert("RGBA")
    return source_preview, prepared_rgb, background_removed


def _default_text_to_image_prompt(prompt: str) -> str:
    suffix = (
        "single centered object, studio product photo, neutral light gray background, "
        "fully visible, no crop, no extra objects, realistic lighting"
    )
    base = str(prompt or "").strip()
    return f"{base}, {suffix}" if base else suffix


def _default_image_generator(owner: Any) -> Callable[..., Any]:
    return default_image_generator(owner)


def _mesh_to_trimesh(mesh: Any):
    import numpy as np
    import trimesh

    vertices = np.asarray(mesh.vertices.detach().cpu().numpy(), dtype=np.float32)
    faces = np.asarray(mesh.faces.detach().cpu().numpy(), dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _mesh_export_bytes(mesh: Any, *, file_type: str) -> bytes:
    payload = _mesh_to_trimesh(mesh).export(file_type=file_type)
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
) -> Dict[str, Optional[Path]]:
    from PIL import Image

    root_dir.mkdir(parents=True, exist_ok=True)
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
        f"sample_s: {metadata.get('timings_s', {}).get('sampling')}",
        f"decode_s: {metadata.get('timings_s', {}).get('decode')}",
    ]
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
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(root_dir)))
    return buf.getvalue()


class Trellis2LocalBackend:
    """Local TRELLIS.2 backend using official Microsoft checkpoints only."""

    backend_id = "abstract3d:trellis2-local"

    def __init__(self, owner: Any, *, image_generator: Optional[Callable[..., Any]] = None) -> None:
        self._owner = owner
        self._image_generator = image_generator
        self._resident_bundle: Optional[Dict[str, Any]] = None
        self._resident_device: Optional[str] = None
        self._resident_model_id: Optional[str] = None
        self._resident_source_dir: Optional[str] = None
        self._resident_dino_source: Optional[str] = None
        self._last_runtime_stats: Dict[str, Any] = {}

    def _model_id(self, requested: Optional[str] = None) -> str:
        model_id = str(
            requested
            or _owner_cfg(self._owner, "scene3d_model_id")
            or _env("ABSTRACT3D_MODEL_ID")
            or _DEFAULT_MODEL_ID
        ).strip()
        return model_id or _DEFAULT_MODEL_ID

    def _load_runtime(self, *, model_id: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
        import torch

        resolved_model = self._model_id(model_id)
        variant = _variant_for_model(resolved_model)
        resolved_device = _select_device(self._owner, explicit=device)
        source_dir = _resolve_source_dir(self._owner)
        dino_source = _configured_dino_source(self._owner)
        if (
            self._resident_bundle is not None
            and self._resident_device == resolved_device
            and self._resident_model_id == resolved_model
            and self._resident_source_dir == str(source_dir)
            and self._resident_dino_source == dino_source
        ):
            return self._resident_bundle

        started = time.perf_counter()
        modules = _import_trellis2(source_dir)
        pipeline_args = _pipeline_args()
        sparse_structure_decoder_path = _download_local_model_ref(_SPARSE_STRUCTURE_DECODER_ID)
        sparse_structure_flow_path = _download_local_model(variant["repo_id"], variant["sparse_structure_flow"])
        shape_flow_512_path = _download_local_model(variant["repo_id"], variant["shape_flow_512"])
        shape_decoder_path = _download_local_model(variant["repo_id"], variant["shape_decoder"])

        models_mod = modules["models"]
        samplers_mod = modules["samplers"]
        sparse_structure_decoder = models_mod.from_pretrained(str(sparse_structure_decoder_path))
        sparse_structure_flow = models_mod.from_pretrained(str(sparse_structure_flow_path))
        shape_flow_512 = models_mod.from_pretrained(str(shape_flow_512_path))
        shape_decoder = models_mod.from_pretrained(str(shape_decoder_path), resolution=512)

        sparse_structure_sampler_cfg = pipeline_args["sparse_structure_sampler"]
        shape_slat_sampler_cfg = pipeline_args["shape_slat_sampler"]
        sparse_structure_sampler = getattr(samplers_mod, sparse_structure_sampler_cfg["name"])(**sparse_structure_sampler_cfg["args"])
        shape_slat_sampler = getattr(samplers_mod, shape_slat_sampler_cfg["name"])(**shape_slat_sampler_cfg["args"])
        dino_snapshot = _download_dino_model(self._owner)
        dino_model, dino_extract = _image_conditioner(dino_snapshot, resolved_device)

        bundle = {
            "variant": variant,
            "sparse_structure_decoder": sparse_structure_decoder.eval().to(resolved_device),
            "sparse_structure_flow": sparse_structure_flow.eval().to(resolved_device),
            "shape_flow_512": shape_flow_512.eval().to(resolved_device),
            "shape_decoder": shape_decoder.eval().to(resolved_device),
            "dino_model": dino_model,
            "dino_extract": dino_extract,
            "sparse_structure_sampler": sparse_structure_sampler,
            "shape_slat_sampler": shape_slat_sampler,
            "sparse_structure_sampler_params": dict(sparse_structure_sampler_cfg["params"]),
            "shape_slat_sampler_params": dict(shape_slat_sampler_cfg["params"]),
            "shape_slat_normalization": dict(pipeline_args["shape_slat_normalization"]),
        }
        self._resident_bundle = bundle
        self._resident_device = resolved_device
        self._resident_model_id = resolved_model
        self._resident_source_dir = str(source_dir)
        self._resident_dino_source = dino_source
        self._last_runtime_stats = {
            "load_s": round(time.perf_counter() - started, 4),
            "device": resolved_device,
            "model_id": resolved_model,
            "source_dir": str(source_dir),
            "dino_source": dino_source,
            "dino_snapshot": str(dino_snapshot),
            "precision": variant["precision"],
        }
        return bundle

    def _clear_runtime(self) -> None:
        self._resident_bundle = None
        self._resident_device = None
        self._resident_model_id = None
        self._resident_source_dir = None
        self._resident_dino_source = None
        self._last_runtime_stats = {}
        torch = importlib.import_module("torch")
        if getattr(torch.cuda, "is_available", lambda: False)():
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
                "provider_id": "trellis2",
                "display_name": "Local TRELLIS.2",
                "tasks": tasks,
                "local": True,
                "remote": False,
                "status": status,
                "backend_id": self.backend_id,
                "installed": installed,
                "configured": bool(installed and (normalized_task != "text_to_scene3d" or composition_ready)),
                "selected": False,
                "metadata": {
                    "official_only": True,
                    "source_snapshot": _TRELLIS2_COMMIT,
                    "dino_model_id": _DINO_MODEL_ID,
                    "text_mode": "abstractvision_composition",
                    "shape_only": True,
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
        if selector and selector not in {"trellis2", self.backend_id}:
            return []
        models = []
        for row in capability_model_records(task=task, validated_only=False):
            provider_name = str(row.get("provider_id") or "")
            if provider_name == "trellis2":
                models.append(row)
        return models

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
                            "pipeline_type": {"type": "string", "enum": ["512"]},
                            "remove_background": {"type": "boolean"},
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
                        "pipeline_type": {"type": "string", "enum": ["512"]},
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
        already_loaded = self._resident_bundle is not None
        self._load_runtime(model_id=request.get("model"), device=request.get("device"))
        return {
            "task": str(request.get("task") or "scene3d_generation"),
            "provider": "trellis2",
            "model": self._resident_model_id,
            "backend_id": self.backend_id,
            "state": "loaded",
            "loaded": True,
            "loaded_new": not already_loaded,
            "details": dict(self._last_runtime_stats),
        }

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        del filters
        if self._resident_bundle is None:
            return []
        return [
            {
                "task": "scene3d_generation",
                "provider": "trellis2",
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
        del request
        model_id = self._resident_model_id
        self._clear_runtime()
        return {
            "task": "scene3d_generation",
            "provider": "trellis2",
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

    def _get_cond(self, runtime: Dict[str, Any], source_image: Any) -> Dict[str, Any]:
        import torch

        features = runtime["dino_extract"]([source_image], image_size=512)
        return {
            "cond": features,
            "neg_cond": torch.zeros_like(features),
        }

    def _sample_sparse_structure(self, runtime: Dict[str, Any], cond: Dict[str, Any]) -> Any:
        import torch

        flow_model = runtime["sparse_structure_flow"]
        resolution = int(getattr(flow_model, "resolution", 16))
        in_channels = int(getattr(flow_model, "in_channels", 8))
        noise = torch.randn(1, in_channels, resolution, resolution, resolution, device=self._resident_device or "cpu")
        sample = runtime["sparse_structure_sampler"].sample(
            flow_model,
            noise,
            **cond,
            **runtime["sparse_structure_sampler_params"],
            verbose=False,
            tqdm_desc="Sampling sparse structure",
        ).samples
        decoded = runtime["sparse_structure_decoder"](sample) > 0
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def _sample_shape_slat(self, runtime: Dict[str, Any], cond: Dict[str, Any], coords: Any):
        import torch

        sparse_mod = importlib.import_module("trellis2.modules.sparse")
        flow_model = runtime["shape_flow_512"]
        noise = sparse_mod.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self._resident_device or "cpu"),
            coords=coords,
        )
        slat = runtime["shape_slat_sampler"].sample(
            flow_model,
            noise,
            **cond,
            **runtime["shape_slat_sampler_params"],
            verbose=False,
            tqdm_desc="Sampling shape SLat",
        ).samples
        std = torch.tensor(runtime["shape_slat_normalization"]["std"], device=slat.device).reshape(1, -1)
        mean = torch.tensor(runtime["shape_slat_normalization"]["mean"], device=slat.device).reshape(1, -1)
        return slat * std + mean

    def _decode_shape(self, runtime: Dict[str, Any], slat: Any) -> Any:
        decoder = runtime["shape_decoder"]
        if callable(getattr(decoder, "set_resolution", None)):
            decoder.set_resolution(512)
        result = decoder(slat, return_subs=True)
        meshes, _subs = result
        return meshes

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
        device: Optional[str],
        model: Optional[str],
        **kwargs: Any,
    ):
        from PIL import Image
        import psutil
        import torch

        runtime = self._load_runtime(model_id=model, device=device)
        actual_task = _TASK_ALIASES.get(task, task)
        if actual_task not in {"text_to_scene3d", "image_to_scene3d"}:
            raise CapabilityNotSupportedError(f"Unsupported TRELLIS.2 task: {actual_task!r}")

        image_generation_s: Optional[float] = None
        if actual_task == "text_to_scene3d":
            image_started = time.perf_counter()
            image_bytes = self._make_source_image(prompt, **kwargs)
            image_generation_s = round(time.perf_counter() - image_started, 4)
            image_input = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        else:
            if image is None:
                raise ValueError("image_to_scene3d requires an image input.")
            image_input = image

        preprocess_started = time.perf_counter()
        source_preview, prepared_rgb, background_removed = _prepare_image(
            image_input,
            remove_background=remove_background,
            foreground_ratio=float(foreground_ratio),
        )
        preprocess_s = round(time.perf_counter() - preprocess_started, 4)
        cond_started = time.perf_counter()
        cond = self._get_cond(runtime, prepared_rgb)
        conditioning_s = round(time.perf_counter() - cond_started, 4)
        sample_started = time.perf_counter()
        coords = self._sample_sparse_structure(runtime, cond)
        slat = self._sample_shape_slat(runtime, cond, coords)
        sampling_s = round(time.perf_counter() - sample_started, 4)
        decode_started = time.perf_counter()
        meshes = self._decode_shape(runtime, slat)
        decode_s = round(time.perf_counter() - decode_started, 4)
        mesh = meshes[0]
        glb_bytes = _mesh_export_bytes(mesh, file_type="glb")
        obj_bytes = _mesh_export_bytes(mesh, file_type="obj")
        mesh_for_preview = _mesh_to_trimesh(mesh)
        views = render_mesh_views(mesh_for_preview)
        primary_format = str(format or "glb").strip().lower() or "glb"
        if primary_format not in {"glb", "obj", "zip"}:
            raise ValueError("scene3d format must be one of: glb, obj, zip")
        primary_bytes = glb_bytes if primary_format == "glb" else obj_bytes
        content_type = "model/gltf-binary" if primary_format == "glb" else "text/plain"
        if primary_format == "zip":
            content_type = "application/zip"
        total_s = round((image_generation_s or 0.0) + preprocess_s + conditioning_s + sampling_s + decode_s, 4)

        process = psutil.Process(os.getpid())
        runtime_meta: Dict[str, Any] = {
            "backend_id": self.backend_id,
            "provider": self.backend_id,
            "model_id": self._resident_model_id or self._model_id(model),
            "task": actual_task,
            "device": self._resident_device or "cpu",
            "format": primary_format,
            "content_type": content_type,
            "pipeline_type": "512",
            "shape_only": True,
            "official_only": True,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "timings_s": {
                "source_image_generation": image_generation_s,
                "preprocess": preprocess_s,
                "conditioning": conditioning_s,
                "sampling": sampling_s,
                "decode": decode_s,
                "load": self._last_runtime_stats.get("load_s"),
                "total": total_s,
            },
            "memory": {
                "rss_bytes": int(process.memory_info().rss),
                "mps_allocated_bytes": int(torch.mps.current_allocated_memory()) if (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()) else None,
            },
            "background_removed": bool(background_removed),
            "source_snapshot": _TRELLIS2_COMMIT,
            "dino_model_id": _DINO_MODEL_ID,
            "weights_precision": runtime["variant"]["precision"],
        }
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
                temp_root = Path(tempfile.mkdtemp(prefix="abstract3d-trellis2-bundle-"))
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
            foreground_ratio=float(kwargs.pop("foreground_ratio", 0.85) or 0.85),
            device=kwargs.pop("device", None),
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
            device=kwargs.pop("device", None),
            model=kwargs.pop("model", None),
            **kwargs,
        )

    def generate(self, prompt: str = "", *, task: Optional[str] = None, image: Any = None, **kwargs: Any):
        normalized_task = _TASK_ALIASES.get(str(task or "text_to_scene3d").strip().lower().replace("-", "_"), "text_to_scene3d")
        if normalized_task == "image_to_scene3d":
            return self.i23d(image, prompt=prompt, **kwargs)
        return self.t23d(prompt, **kwargs)

    def validate_suite(
        self,
        *,
        prompts: Sequence[str],
        images: Sequence[str],
        output_dir: str,
        image_provider: Optional[str] = None,
        image_model: Optional[str] = None,
        device: str = "auto",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from PIL import Image

        del kwargs
        root_dir = Path(output_dir).expanduser().resolve()
        root_dir.mkdir(parents=True, exist_ok=True)
        cases: List[Dict[str, Any]] = []
        sheets = []
        generated_inputs_dir = root_dir / "inputs"
        generated_inputs_dir.mkdir(exist_ok=True)

        for index, prompt in enumerate(list(prompts or []), start=1):
            case_dir = root_dir / f"{index:02d}_t23d"
            result = self.t23d(
                prompt,
                output_dir=str(case_dir),
                format="glb",
                device=device,
                model=model,
                image_provider=image_provider,
                image_model=image_model,
            )
            meta = result.get("metadata") or {}
            if meta.get("contact_sheet_path"):
                sheets.append(Image.open(meta["contact_sheet_path"]).convert("RGB"))
            cases.append(meta)

        start_index = len(cases) + 1
        for offset, image_path in enumerate(list(images or []), start=start_index):
            case_dir = root_dir / f"{offset:02d}_i23d"
            result = self.i23d(
                image_path,
                output_dir=str(case_dir),
                format="glb",
                device=device,
                model=model,
            )
            meta = result.get("metadata") or {}
            if meta.get("contact_sheet_path"):
                sheets.append(Image.open(meta["contact_sheet_path"]).convert("RGB"))
            cases.append(meta)

        if not cases:
            raise ValueError("validate_suite requires at least one prompt or image.")
        contact_sheet = stack_contact_sheets(sheets, columns=1)
        contact_sheet_path = root_dir / "contact_sheet.png"
        contact_sheet.save(contact_sheet_path)
        summary = {
            "backend_id": self.backend_id,
            "model_id": self._model_id(model),
            "device": self._resident_device or _select_device(self._owner, explicit=device),
            "cases": cases,
            "contact_sheet_path": str(contact_sheet_path),
        }
        summary_path = root_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
