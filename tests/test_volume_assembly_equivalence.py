"""Independent equivalence harness: OLD vs NEW volume-assembly paths, bit-exact.

Adversarial audit #2 of the 2026-07-22 `_AdaptiveVolumeDecoder` performance
patch (per-offset scatter, np.nonzero enumeration, astype(copy=False),
progress logging). This module is deliberately INDEPENDENT of the patch
author's own tests in test_hunyuan3d_backend_unit.py:

- The OLD path is embedded below VERBATIM from the pre-patch source
  (`_FrozenOldAdaptiveVolumeDecoder`), extracted from git HEAD ade2ddd
  (class body lines 470-688 of src/abstract3d/backends/hunyuan3d_runtime.py)
  and verified byte-identical to the pre-patch working tree during the
  audit. It is the frozen semantic baseline: if any future edit to the
  production decoder changes one byte of the assembled field, these tests
  fail.
- The NEW path is the REAL production class, imported from the backend at
  test time (whatever its current implementation is).
- The vendored upstream decoders (`VanillaVolumeDecoder`, and the
  documented-broken `HierarchicalVolumeDecoding`) are imported from the
  pinned source snapshot the backend actually runs, and pinned against a
  frozen replica of their accumulation the same way.

Equivalence is asserted at the strictest level available:
- grid bytes (`tobytes()` equality — bit identity, NaN-safe),
- output dtype and shape contracts,
- the CHUNK TRACE: the exact sequence of (shape, dtype) the geo decoder
  was called with. Synthetic pointwise decoders cannot see a changed
  chunk partition (elementwise math is partition-independent), but the
  real cross-attention decoder CAN (batched matmul reduction order), so
  a silent partition change must fail the harness even when the grids
  happen to match.

Case matrix (from the audit brief): multiple scales incl. odd sizes
(8^3, 33^3, 65^3), float16/float32, non-divisible chunk sizes, the
empty-input edge, C/F memory order, and every branch of the adaptive
decoder (dense-single-level, no-interior dense fallback, empty-surface
ones fallback, multi-level refinement, mc_level != 0).
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

# Import torch before pymeshlab-backed helpers run (repo convention: both
# bundle an OpenMP runtime on macOS and initializing pymeshlab's first
# crashes torch later).
import torch  # noqa: F401

from abstract3d.backends import hunyuan3d_runtime as runtime

# --------------------------------------------------------------------------
# Provenance fingerprints (informational, not gating): sha256 of the class
# source as audited. OLD == git HEAD ade2ddd; PATCHED == the 2026-07-22
# assembly rewrite audited by this harness. A third value means the class
# drifted after the audit — the equivalence tests below remain the gate.
_OLD_CLASS_SHA = "09bb63210091d0d5"  # git HEAD ade2ddd, class lines 470-688
_AUDITED_PATCH_SHA = "ef2475350c2146d9"  # worktree 2026-07-22 04:12, class lines 497-791


def _current_decoder_sha() -> str:
    source = inspect.getsource(runtime._AdaptiveVolumeDecoder)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Deterministic recording decoders (pure elementwise functions of the query
# coordinates: bit-reproducible for a given input dtype regardless of chunk
# partition, so any grid difference between paths is an assembly difference).


class RecordingFieldDecoder:
    """Wrinkled-sphere logit field that records every call it serves.

    out_dtype:
      "query"   -> logits in the query dtype (the common fp16-on-MPS shape);
      "float32" -> fp32 logits regardless of query dtype (the autocast shape:
                   a naive patch that preallocates the accumulator in
                   latents.dtype TRUNCATES these and fails the harness).
    """

    def __init__(self, *, gradient: float = 7.0, radius: float = 0.62,
                 wrinkle: float = 0.05, out_dtype: str = "query") -> None:
        self.gradient = float(gradient)
        self.radius = float(radius)
        self.wrinkle = float(wrinkle)
        self.out_dtype = out_dtype
        self.calls: List[Tuple[Tuple[int, ...], str]] = []

    def __call__(self, *, queries: Any, latents: Any) -> Any:
        del latents
        self.calls.append((tuple(queries.shape), str(queries.dtype)))
        q = queries.to(torch.float32)
        r = torch.linalg.norm(q, dim=-1)
        bump = self.wrinkle * (
            torch.sin(9.0 * q[..., 0]) * torch.sin(11.0 * q[..., 1]) * torch.sin(13.0 * q[..., 2])
        )
        logits = self.gradient * (self.radius + bump - r)
        if self.out_dtype == "query":
            logits = logits.to(queries.dtype)
        # fp32-with-subnormal-tail values: exercise bits a fp16 round-trip
        # would destroy (upcast-preservation hazard).
        return logits.unsqueeze(-1)


class ConstantDecoder:
    """Constant field: sign selects the adaptive decoder's fallback branch."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.calls: List[Tuple[Tuple[int, ...], str]] = []

    def __call__(self, *, queries: Any, latents: Any) -> Any:
        del latents
        self.calls.append((tuple(queries.shape), str(queries.dtype)))
        return torch.full(
            queries.shape[:-1], self.value, dtype=queries.dtype
        ).unsqueeze(-1)


# --------------------------------------------------------------------------
# FROZEN OLD PATH — verbatim pre-patch semantics (git HEAD ade2ddd).
# Do not "improve" this class: it is the baseline the production code is
# measured against. Only the logging-free `_query_points` signature differs
# (the patch added a `label` kwarg the old code did not have).


class _FrozenOldAdaptiveVolumeDecoder:
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
        outputs = []
        tensor_points = torch.from_numpy(np.ascontiguousarray(points)).to(device=device, dtype=dtype)
        for start in range(0, tensor_points.shape[0], int(num_chunks)):
            chunk = tensor_points[start : start + int(num_chunks)].unsqueeze(0)
            logits = geo_decoder(queries=chunk, latents=latents)
            outputs.append(logits[0, ..., 0].detach().to("cpu", dtype=torch.float32))
        if not outputs:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

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
        from scipy import ndimage

        del enable_pbar, kwargs
        device = latents.device
        dtype = latents.dtype
        final_resolution = int(octree_resolution or 384)
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.asarray(bounds[0:3], dtype=np.float64)
        bbox_max = np.asarray(bounds[3:6], dtype=np.float64)

        resolutions: List[int] = [final_resolution]
        while resolutions[0] > self.coarse_resolution and resolutions[0] % 2 == 0:
            resolutions.insert(0, resolutions[0] // 2)

        def grid_axes(resolution: int) -> tuple[Any, Any, Any]:
            xs = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float64)
            ys = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float64)
            zs = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float64)
            return xs, ys, zs

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
            surface = np.zeros_like(inside)
            for axis in range(3):
                changed = np.diff(inside, axis=axis)
                pad_lo = [(0, 0)] * 3
                pad_hi = [(0, 0)] * 3
                pad_lo[axis] = (0, 1)
                pad_hi[axis] = (1, 0)
                surface |= np.pad(changed, pad_lo, mode="constant")
                surface |= np.pad(changed, pad_hi, mode="constant")
            band = self.band
            if surface.any():
                shell_scale = float(np.median(np.abs(grid - level_value)[surface]))
                band = max(self.band, 1.5 * shell_scale)
            surface |= np.abs(grid - level_value) < band
            surface = ndimage.binary_dilation(surface, iterations=2)

            scale = next_resolution // current_resolution
            fine_shape = (next_resolution + 1,) * 3
            fine_mask = np.zeros(fine_shape, dtype=bool)
            coarse_idx = np.argwhere(surface)
            if len(coarse_idx) == 0:
                surface = np.ones_like(inside)
                coarse_idx = np.argwhere(surface)
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

            zoom_factor = tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape))
            next_grid = ndimage.zoom(grid, zoom_factor, order=1, mode="nearest").astype(np.float32)
            next_grid[refine_idx[:, 0], refine_idx[:, 1], refine_idx[:, 2]] = refine_logits
            grid = next_grid
            current_resolution = next_resolution

        return torch.from_numpy(grid[None]).float()


# --------------------------------------------------------------------------
# Vendored upstream import (the REAL functions the pipeline runs).


def _vendored_volume_decoders():
    source_dir = (
        Path.home()
        / ".cache/abstract3d/vendor/hunyuan3d21"
        / runtime._HUNYUAN_COMMIT
        / "hy3dshape"
    )
    if not source_dir.exists():
        pytest.skip(f"pinned Hunyuan3D source snapshot not cached: {source_dir}")
    import sys

    entry = str(source_dir)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)
    try:
        import importlib

        return importlib.import_module("hy3dshape.models.autoencoders.volume_decoders")
    finally:
        if inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def _frozen_vanilla_decode(latents, geo_decoder, *, bounds, num_chunks, octree_resolution, module):
    """Verbatim replica of the vendored VanillaVolumeDecoder accumulation
    (list-append + one torch.cat along dim=1 + view + float): the baseline
    proving the REAL vendored decoder's assembly semantics are unchanged."""
    from einops import repeat

    device = latents.device
    dtype = latents.dtype
    batch_size = latents.shape[0]
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
    xyz_samples, grid_size, _ = module.generate_dense_grid_points(
        bbox_min=bbox_min, bbox_max=bbox_max, octree_resolution=octree_resolution, indexing="ij"
    )
    xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)
    batch_logits = []
    for start in range(0, xyz_samples.shape[0], num_chunks):
        chunk_queries = xyz_samples[start : start + num_chunks, :]
        chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
        logits = geo_decoder(queries=chunk_queries, latents=latents)
        batch_logits.append(logits)
    grid_logits = torch.cat(batch_logits, dim=1)
    return grid_logits.view((batch_size, *grid_size)).float()


# --------------------------------------------------------------------------
# Helpers.


def _grid_bytes(grid: Any) -> bytes:
    array = grid.numpy() if hasattr(grid, "numpy") else np.asarray(grid)
    return np.ascontiguousarray(array).tobytes()


def _run_kernel(kernel, points, decoder, *, num_chunks, dtype):
    return kernel(
        points,
        latents=torch.zeros((1, 4, 8), dtype=dtype),
        geo_decoder=decoder,
        num_chunks=num_chunks,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def _sphere_points(count: int, *, seed: int = 7, order: str = "C") -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.01, 1.01, size=(count, 3)).astype(np.float32)
    if order == "F":
        points = np.asfortranarray(points)
    return points


# ==========================================================================
# 0. Provenance report (never fails): which implementation is under audit.


def test_report_which_implementation_is_under_audit(capsys) -> None:
    sha = _current_decoder_sha()
    if sha == _AUDITED_PATCH_SHA:
        status = "AUDITED PATCH (2026-07-22)"
    elif sha == _OLD_CLASS_SHA:
        status = "PRE-PATCH BASELINE (patch not landed)"
    else:
        status = "POST-AUDIT DRIFT (equivalence tests below remain the gate)"
    print(f"_AdaptiveVolumeDecoder source sha256[:16] = {sha} -> {status}")
    assert True


# ==========================================================================
# 1. Accumulation kernel: real `_query_points` vs frozen old, all edges.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    ("count", "num_chunks"),
    [
        (0, 8000),        # empty input: zero chunks
        (1, 8000),        # single point, chunk larger than input
        (512, 512),       # exact multiple (8^3, one full chunk)
        (512, 100),       # non-divisible: 5 full + 1 partial chunk
        (512, 7),         # pathological small chunk, partial tail
        (35937, 8000),    # 33^3 odd cube, partial tail
        (274625, 32768),  # 65^3 at the PRODUCTION chunk size (8 full + 12481 tail)
    ],
    ids=["empty", "single", "exact-multiple", "partial-100", "partial-7", "cube33", "cube65-prod-chunk"],
)
def test_query_points_kernel_bit_identical(count: int, num_chunks: int, dtype) -> None:
    points = _sphere_points(count)
    old_decoder = RecordingFieldDecoder()
    new_decoder = RecordingFieldDecoder()

    old = _run_kernel(
        _FrozenOldAdaptiveVolumeDecoder._query_points, points, old_decoder,
        num_chunks=num_chunks, dtype=dtype,
    )
    new = _run_kernel(
        runtime._AdaptiveVolumeDecoder._query_points, points, new_decoder,
        num_chunks=num_chunks, dtype=dtype,
    )

    assert new.dtype == old.dtype == torch.float32
    assert new.shape == old.shape == (count,)
    assert _grid_bytes(new) == _grid_bytes(old)
    # Chunk-partition equality: same call sequence, same query dtypes. A
    # patch that repartitions chunks would pass the value check on this
    # pointwise decoder but diverge on the real cross-attention decoder.
    assert new_decoder.calls == old_decoder.calls


@pytest.mark.parametrize("order", ["C", "F"], ids=["c-order", "f-order"])
def test_query_points_kernel_memory_order_invariant(order: str) -> None:
    points = _sphere_points(4096, order=order)
    reference = _run_kernel(
        _FrozenOldAdaptiveVolumeDecoder._query_points,
        _sphere_points(4096, order="C"),
        RecordingFieldDecoder(),
        num_chunks=1000,
        dtype=torch.float32,
    )
    produced = _run_kernel(
        runtime._AdaptiveVolumeDecoder._query_points, points, RecordingFieldDecoder(),
        num_chunks=1000, dtype=torch.float32,
    )
    assert _grid_bytes(produced) == _grid_bytes(reference)


def test_query_points_kernel_non_contiguous_view() -> None:
    dense = _sphere_points(8192)
    view = dense[::2]  # non-contiguous stride-2 view
    assert not view.flags["C_CONTIGUOUS"]
    old = _run_kernel(
        _FrozenOldAdaptiveVolumeDecoder._query_points, view, RecordingFieldDecoder(),
        num_chunks=999, dtype=torch.float32,
    )
    new = _run_kernel(
        runtime._AdaptiveVolumeDecoder._query_points, view, RecordingFieldDecoder(),
        num_chunks=999, dtype=torch.float32,
    )
    assert _grid_bytes(new) == _grid_bytes(old)


def test_query_points_preserves_fp32_logits_under_fp16_queries() -> None:
    """Upcast-preservation hazard: the decoder may emit fp32 logits even for
    fp16 queries (autocast shape). The historical path converts each chunk
    to fp32 on CPU immediately, preserving those bits; a naive patch that
    preallocates the accumulator in latents.dtype (fp16) truncates them."""
    points = _sphere_points(4097)
    old_decoder = RecordingFieldDecoder(out_dtype="float32")
    new_decoder = RecordingFieldDecoder(out_dtype="float32")
    old = _run_kernel(
        _FrozenOldAdaptiveVolumeDecoder._query_points, points, old_decoder,
        num_chunks=512, dtype=torch.float16,
    )
    new = _run_kernel(
        runtime._AdaptiveVolumeDecoder._query_points, points, new_decoder,
        num_chunks=512, dtype=torch.float16,
    )
    assert _grid_bytes(new) == _grid_bytes(old)
    # The values must NOT all survive a fp16 round-trip (otherwise this
    # case could not catch a fp16-accumulator patch and would be vacuous).
    as_fp16_roundtrip = new.to(torch.float16).to(torch.float32)
    assert not torch.equal(new, as_fp16_roundtrip)


# ==========================================================================
# 2. Whole adaptive decoder: real patched class vs frozen old class,
#    through every branch.


def _compare_adaptive(
    *,
    octree_resolution: int,
    coarse_resolution: int,
    decoder_factory,
    latents_dtype=torch.float32,
    num_chunks: int = 97,
    mc_level: float = 0.0,
) -> None:
    latents = torch.zeros((1, 4, 8), dtype=latents_dtype)

    old_geo = decoder_factory()
    new_geo = decoder_factory()
    old = _FrozenOldAdaptiveVolumeDecoder(coarse_resolution=coarse_resolution)(
        latents, old_geo, bounds=1.01, num_chunks=num_chunks,
        octree_resolution=octree_resolution, mc_level=mc_level,
    )
    new = runtime._AdaptiveVolumeDecoder(coarse_resolution=coarse_resolution)(
        latents, new_geo, bounds=1.01, num_chunks=num_chunks,
        octree_resolution=octree_resolution, mc_level=mc_level,
    )

    side = octree_resolution + 1
    assert tuple(new.shape) == tuple(old.shape) == (1, side, side, side)
    assert new.dtype == old.dtype == torch.float32
    assert _grid_bytes(new) == _grid_bytes(old)
    assert new_geo.calls == old_geo.calls


def test_adaptive_two_refinement_levels_fp32() -> None:
    # 8 -> 16 -> 32; radius 0.98 pushes surface cells against the walls so
    # the fine-index clamp fires; num_chunks=97 forces partial tails.
    _compare_adaptive(
        octree_resolution=32,
        coarse_resolution=8,
        decoder_factory=lambda: RecordingFieldDecoder(gradient=165.0, radius=0.98, wrinkle=0.02),
    )


def test_adaptive_two_refinement_levels_fp16_latents() -> None:
    # fp16 latents drive fp16 queries end-to-end (the MPS production shape).
    _compare_adaptive(
        octree_resolution=64,
        coarse_resolution=16,
        decoder_factory=lambda: RecordingFieldDecoder(gradient=20.0),
        latents_dtype=torch.float16,
        num_chunks=32768,  # production chunk size: coarse level fits in one chunk
    )


def test_adaptive_two_refinement_levels_fp32_logits_fp16_queries() -> None:
    # Autocast shape through the WHOLE decoder (not just the kernel).
    _compare_adaptive(
        octree_resolution=32,
        coarse_resolution=8,
        decoder_factory=lambda: RecordingFieldDecoder(out_dtype="float32"),
        latents_dtype=torch.float16,
    )


def test_adaptive_nonzero_mc_level() -> None:
    _compare_adaptive(
        octree_resolution=32,
        coarse_resolution=8,
        decoder_factory=lambda: RecordingFieldDecoder(gradient=5.0),
        mc_level=0.15,
    )


def test_adaptive_odd_resolution_single_dense_level() -> None:
    # 33 is odd: the halving schedule stops immediately -> one dense level.
    _compare_adaptive(
        octree_resolution=33,
        coarse_resolution=128,
        decoder_factory=lambda: RecordingFieldDecoder(),
        num_chunks=5000,
    )


def test_adaptive_small_grid_dense_because_coarse_covers_it() -> None:
    # 8 <= coarse_resolution: single dense level through the same early return.
    _compare_adaptive(
        octree_resolution=8,
        coarse_resolution=128,
        decoder_factory=lambda: RecordingFieldDecoder(),
    )


def test_adaptive_all_outside_dense_fallback_branch() -> None:
    # Constant -5 field: coarse sees no interior -> dense final-resolution
    # fallback decode.
    _compare_adaptive(
        octree_resolution=16,
        coarse_resolution=8,
        decoder_factory=lambda: ConstantDecoder(-5.0),
        num_chunks=333,
    )


def test_adaptive_all_inside_ones_fallback_branch() -> None:
    # Constant +5 field: no sign changes and |logit| above the band ->
    # empty surface -> the ones_like fallback refines everything.
    _compare_adaptive(
        octree_resolution=16,
        coarse_resolution=8,
        decoder_factory=lambda: ConstantDecoder(5.0),
        num_chunks=333,
    )


# ==========================================================================
# 3. REAL vendored upstream decoders vs frozen replicas.


@pytest.mark.parametrize(
    ("octree_resolution", "num_chunks", "batch_size", "dtype"),
    [
        (8, 100, 1, torch.float32),     # non-divisible chunks, tiny grid
        (33, 7777, 1, torch.float16),   # odd size + fp16 + partial tail
        (16, 913, 2, torch.float32),    # batch>1: cat(dim=1) semantics
    ],
    ids=["r8-fp32", "r33-fp16-odd", "r16-batch2"],
)
def test_vendored_vanilla_decoder_matches_frozen_replica(
    octree_resolution: int, num_chunks: int, batch_size: int, dtype
) -> None:
    module = _vendored_volume_decoders()
    latents = torch.zeros((batch_size, 4, 8), dtype=dtype)

    real_geo = RecordingFieldDecoder()
    replica_geo = RecordingFieldDecoder()
    real = module.VanillaVolumeDecoder()(
        latents,
        real_geo,
        bounds=1.01,
        num_chunks=num_chunks,
        octree_resolution=octree_resolution,
        enable_pbar=False,
    )
    replica = _frozen_vanilla_decode(
        latents,
        replica_geo,
        bounds=1.01,
        num_chunks=num_chunks,
        octree_resolution=octree_resolution,
        module=module,
    )

    side = octree_resolution + 1
    assert tuple(real.shape) == tuple(replica.shape) == (batch_size, side, side, side)
    assert real.dtype == replica.dtype == torch.float32
    assert _grid_bytes(real) == _grid_bytes(replica)
    assert real_geo.calls == replica_geo.calls


@pytest.mark.xfail(
    strict=False,
    reason=(
        "canary: upstream HierarchicalVolumeDecoding is documented broken as "
        "shipped (float cell size cast collapses refinement queries; backend "
        "docstring, reproduced on CPU/MPS). If this starts passing, the vendor "
        "snapshot changed and the audit baseline must be revisited."
    ),
)
def test_vendored_hierarchical_decoder_matches_dense_reference() -> None:
    module = _vendored_volume_decoders()
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    geo = RecordingFieldDecoder(gradient=165.0, radius=0.62, wrinkle=0.02)

    hierarchical = module.HierarchicalVolumeDecoding()(
        latents,
        geo,
        bounds=1.01,
        num_chunks=100000,
        octree_resolution=128,
        min_resolution=63,
    )
    dense = module.VanillaVolumeDecoder()(
        latents,
        RecordingFieldDecoder(gradient=165.0, radius=0.62, wrinkle=0.02),
        bounds=1.01,
        num_chunks=100000,
        octree_resolution=128,
        enable_pbar=False,
    )
    inside_h = hierarchical[0].numpy() > 0.0
    inside_d = dense[0].numpy() > 0.0
    finite = np.isfinite(hierarchical[0].numpy())
    assert (inside_h[finite] == inside_d[finite]).all()


# ==========================================================================
# 4. Cross-decoder sanity: the adaptive decoder agrees with the vendored
#    dense decoder ON QUERIED CELLS (sign everywhere; exact values on the
#    refined band). Away from the surface the adaptive field is trilinear
#    fill BY DESIGN — asserting global bit-equality against dense would be
#    wrong, and any patch reviewer assuming it is a hazard this test names.


def test_adaptive_vs_vendored_dense_sign_agreement() -> None:
    module = _vendored_volume_decoders()
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)

    adaptive = runtime._AdaptiveVolumeDecoder(coarse_resolution=8)(
        latents,
        RecordingFieldDecoder(gradient=165.0, radius=0.62, wrinkle=0.02),
        bounds=1.01,
        num_chunks=100000,
        octree_resolution=32,
    )
    dense = module.VanillaVolumeDecoder()(
        latents,
        RecordingFieldDecoder(gradient=165.0, radius=0.62, wrinkle=0.02),
        bounds=1.01,
        num_chunks=100000,
        octree_resolution=32,
        enable_pbar=False,
    )
    a = adaptive[0].numpy()
    d = dense[0].numpy()
    assert a.shape == d.shape
    # Sign agreement everywhere (what marching cubes consumes)...
    assert ((a > 0.0) == (d > 0.0)).all()
    # ...and the near-surface band carries the exact dense values (refined,
    # not interpolated): |dense| below one coarse-cell logit scale.
    near = np.abs(d) < 165.0 * (2.02 / 32.0)
    assert near.any()
    assert np.array_equal(a[near], d[near])
