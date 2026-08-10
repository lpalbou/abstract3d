"""CPU micro-benchmark for `_AdaptiveVolumeDecoder` refinement bookkeeping.

Context (e22v2 forensics, 2026-07-22): a 3s `sample` of the live pass-2
decode at octree 512 showed the main thread 100% on-CPU inside the
refinement loop's numpy/scipy ops (NI_ZoomShift 34%, PyArray_Nonzero 19%,
mapiter_set 12%, reshape-copy 8.7%, binary erosion 8%, LONG_minimum 6.4%),
NOT in array_concatenate (0.8%). This script grounds the patch in numbers:

Part 1  replicates the refinement loop body op-by-op at true 512 scale
        (129^3 -> 257^3 -> 513^3) on synthetic fields bracketing the real
        checkpoint (hard-saturating shell vs softer field), cross-checked
        against the real decoder end-to-end (np.array_equal on the grid).
Part 2  measures the chunk-accumulation patterns at the real chunking
        (num_chunks=32768, dense 513^3 = 4122 chunks): the CURRENT pattern
        (list + one torch.cat), preallocated in-place writes, list + one
        np.concatenate, and the hypothesized repeated-per-chunk
        np.concatenate (measured small, extrapolated quadratically).
Part 3  measures candidate byte-identical replacements for the hot ops
        (nonzero-tuple vs argwhere+strided-columns, per-offset scatter vs
        27N-row index matrix, astype(copy=False), in-place pad-OR) and
        verifies equality on the spot.

Run niced; no GPU is touched (the fake geo decoder is analytic CPU torch):

    nice -n 10 .venv/bin/python scripts/experimental/bench_adaptive_decoder_cpu.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BOUNDS = 1.01
FINAL_RESOLUTION = 512
COARSE_RESOLUTION = 128
NUM_CHUNKS = 32768  # _DEFAULT_MPS_NUM_CHUNKS: the e22v2 run's actual batching


def _timer():
    return time.perf_counter()


class FieldSpec:
    """Analytic logit field: saturating linear SDF of a wrinkled sphere.

    gradient ~165/unit reproduces the shipped checkpoint's hard shell
    (docstring: median shell |logit| ~0.63 at 256-level cell spacing);
    gradient ~20 emulates a soft field where the |logit|<band criterion
    inflates the refinement set (the memory/CPU stress case).
    """

    def __init__(self, name: str, gradient: float, wrinkle: float) -> None:
        self.name = name
        self.gradient = gradient
        self.wrinkle = wrinkle

    def torch_decoder(self):
        import torch

        gradient = self.gradient
        wrinkle = self.wrinkle

        class _Decoder:
            def __call__(self, *, queries, latents):
                del latents
                q = queries
                r = torch.linalg.norm(q, dim=-1)
                bump = wrinkle * (
                    torch.sin(9.0 * q[..., 0]) * torch.sin(11.0 * q[..., 1]) * torch.sin(13.0 * q[..., 2])
                )
                logit = gradient * (0.62 + bump - r)
                return torch.clamp(logit, -10.0, 10.0).unsqueeze(-1)

        return _Decoder()

    def numpy_field(self, points: np.ndarray) -> np.ndarray:
        # Evaluated THROUGH the torch decoder (fp32 CPU) so the replica's
        # values are bit-identical to what the real decoder's query path
        # produces — numpy transcendentals (sin/norm) differ from torch's
        # in the last ulp and would break the grid equality cross-check.
        import torch

        decoder = self.torch_decoder()
        queries = torch.from_numpy(np.ascontiguousarray(points.astype(np.float32, copy=False)))
        return decoder(queries=queries, latents=None)[..., 0].numpy()


FIELDS = [
    FieldSpec("hard-shell (checkpoint-like)", gradient=165.0, wrinkle=0.03),
    FieldSpec("soft-field (band blow-up)", gradient=20.0, wrinkle=0.03),
]


def replicate_loop(field: FieldSpec) -> dict:
    """Replicates the LEGACY (pre-2026-07-22) refinement bookkeeping
    verbatim, timing each op. The analytic query is timed separately so
    'bookkeeping' excludes it. This is the before-patch baseline; the
    patched production decoder runs next to it for the after number and
    the grid equality cross-check."""
    from scipy import ndimage

    rows = []
    bbox_min = np.asarray([-BOUNDS] * 3, dtype=np.float64)
    bbox_max = np.asarray([BOUNDS] * 3, dtype=np.float64)

    resolutions = [FINAL_RESOLUTION]
    while resolutions[0] > COARSE_RESOLUTION and resolutions[0] % 2 == 0:
        resolutions.insert(0, resolutions[0] // 2)

    def grid_axes(resolution: int):
        xs = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float64)
        ys = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float64)
        zs = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float64)
        return xs, ys, zs

    def add(level: str, op: str, seconds: float, note: str = "") -> None:
        rows.append({"level": level, "op": op, "s": seconds, "note": note})

    query_seconds = [0.0]

    def query(points: np.ndarray) -> np.ndarray:
        t0 = _timer()
        out = field.numpy_field(points)
        query_seconds[0] += _timer() - t0
        return out

    total0 = _timer()
    coarse_resolution = resolutions[0]
    t0 = _timer()
    xs, ys, zs = grid_axes(coarse_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    coarse_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)
    add("coarse", "meshgrid+stack+astype", _timer() - t0, f"{coarse_resolution + 1}^3 pts")
    grid = query(coarse_points).reshape(coarse_resolution + 1, coarse_resolution + 1, coarse_resolution + 1)

    band_default = 0.95
    level_value = 0.0
    current_resolution = coarse_resolution
    for next_resolution in resolutions[1:]:
        tag = f"{current_resolution}->{next_resolution}"
        t0 = _timer()
        inside = grid > level_value
        add(tag, "inside = grid > level", _timer() - t0)

        t0 = _timer()
        surface = np.zeros_like(inside)
        for axis in range(3):
            changed = np.diff(inside, axis=axis)
            pad_lo = [(0, 0)] * 3
            pad_hi = [(0, 0)] * 3
            pad_lo[axis] = (0, 1)
            pad_hi[axis] = (1, 0)
            surface |= np.pad(changed, pad_lo, mode="constant")
            surface |= np.pad(changed, pad_hi, mode="constant")
        add(tag, "sign-change shell (diff+pad+or x3)", _timer() - t0)

        t0 = _timer()
        band = band_default
        if surface.any():
            shell_scale = float(np.median(np.abs(grid - level_value)[surface]))
            band = max(band_default, 1.5 * shell_scale)
        surface |= np.abs(grid - level_value) < band
        add(tag, "band widen (median+abs+or)", _timer() - t0, f"band={band:.3f}")

        t0 = _timer()
        surface = ndimage.binary_dilation(surface, iterations=2)
        add(tag, "binary_dilation x2", _timer() - t0)

        scale = next_resolution // current_resolution
        fine_shape = (next_resolution + 1,) * 3
        t0 = _timer()
        fine_mask = np.zeros(fine_shape, dtype=bool)
        coarse_idx = np.argwhere(surface)
        add(tag, "argwhere(surface)", _timer() - t0, f"N1={len(coarse_idx):,}")

        t0 = _timer()
        base = coarse_idx * scale
        block = np.stack(
            np.meshgrid(np.arange(scale + 1), np.arange(scale + 1), np.arange(scale + 1), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        fine_idx = (base[:, None, :] + block[None, :, :]).reshape(-1, 3)
        add(tag, "fine_idx = base+block (27N rows)", _timer() - t0, f"{len(fine_idx):,} rows")

        t0 = _timer()
        np.minimum(fine_idx, next_resolution, out=fine_idx)
        add(tag, "np.minimum clamp (27N x3)", _timer() - t0)

        t0 = _timer()
        fine_mask[fine_idx[:, 0], fine_idx[:, 1], fine_idx[:, 2]] = True
        add(tag, "scatter fine_mask[27N triples]", _timer() - t0)

        t0 = _timer()
        xs, ys, zs = grid_axes(next_resolution)
        refine_idx = np.argwhere(fine_mask)
        add(tag, "argwhere(fine_mask)", _timer() - t0, f"N2={len(refine_idx):,}")

        t0 = _timer()
        refine_points = np.stack(
            [xs[refine_idx[:, 0]], ys[refine_idx[:, 1]], zs[refine_idx[:, 2]]], axis=-1
        ).astype(np.float32)
        add(tag, "gather+stack+astype refine_points", _timer() - t0)

        refine_logits = query(refine_points)

        t0 = _timer()
        zoom_factor = tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape))
        next_grid = ndimage.zoom(grid, zoom_factor, order=1, mode="nearest").astype(np.float32)
        add(tag, "ndimage.zoom(order=1) + astype", _timer() - t0, f"-> {fine_shape[0]}^3")

        t0 = _timer()
        next_grid[refine_idx[:, 0], refine_idx[:, 1], refine_idx[:, 2]] = refine_logits
        add(tag, "scatter next_grid[N2] = logits", _timer() - t0)
        grid = next_grid
        current_resolution = next_resolution

    total = _timer() - total0
    return {
        "rows": rows,
        "grid": grid,
        "total_s": total,
        "query_s": query_seconds[0],
        "bookkeeping_s": total - query_seconds[0],
    }


def run_real_decoder(field: FieldSpec) -> dict:
    import torch

    from abstract3d.backends import hunyuan3d_runtime as runtime

    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=COARSE_RESOLUTION)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    t0 = _timer()
    grid = decoder(
        latents,
        field.torch_decoder(),
        bounds=BOUNDS,
        num_chunks=NUM_CHUNKS,
        octree_resolution=FINAL_RESOLUTION,
    )
    total = _timer() - t0
    return {"grid": grid[0].numpy(), "total_s": total}


def part1() -> None:
    print("=" * 88)
    print(f"PART 1: refinement-loop op decomposition at octree {FINAL_RESOLUTION} "
          f"({COARSE_RESOLUTION + 1}^3 -> {FINAL_RESOLUTION + 1}^3), num_chunks={NUM_CHUNKS}")
    print("=" * 88)
    for field in FIELDS:
        rep = replicate_loop(field)
        print(f"\nfield: {field.name}")
        print(f"{'level':>10}  {'op':<38} {'seconds':>9}  note")
        for row in rep["rows"]:
            print(f"{row['level']:>10}  {row['op']:<38} {row['s']:>9.3f}  {row['note']}")
        print(f"{'':>10}  {'TOTAL (incl. analytic queries)':<38} {rep['total_s']:>9.3f}")
        print(f"{'':>10}  {'analytic query time (excluded)':<38} {rep['query_s']:>9.3f}")
        print(f"{'':>10}  {'LEGACY BOOKKEEPING TOTAL':<38} {rep['bookkeeping_s']:>9.3f}")
        real = run_real_decoder(field)
        identical = np.array_equal(rep["grid"], real["grid"])
        print(f"{'':>10}  patched decoder end-to-end: {real['total_s']:.3f}s; "
              f"grid byte-identical to legacy: {identical}")
        if not identical:
            diff = np.abs(rep["grid"] - real["grid"])
            print(f"{'':>10}  !! max abs diff {diff.max():.3e} at {int((diff > 0).sum())} cells")


def part2() -> None:
    import torch

    total_points = (FINAL_RESOLUTION + 1) ** 3
    n_chunks = (total_points + NUM_CHUNKS - 1) // NUM_CHUNKS
    print("\n" + "=" * 88)
    print(f"PART 2: chunk accumulation at dense-512 scale: {total_points:,} float32 values, "
          f"{n_chunks:,} chunks of {NUM_CHUNKS}")
    print("=" * 88)

    chunk_np = np.random.default_rng(0).standard_normal(NUM_CHUNKS).astype(np.float32)
    chunk_t = torch.from_numpy(chunk_np.copy())

    # (a1) CURRENT pattern: python list of torch chunks + one torch.cat.
    t0 = _timer()
    outputs = []
    for _ in range(n_chunks):
        outputs.append(chunk_t)
    merged = torch.cat(outputs, dim=0)
    t_current = _timer() - t0
    del outputs, merged

    # (b) preallocated numpy buffer, in-place slice writes.
    t0 = _timer()
    out = np.empty(n_chunks * NUM_CHUNKS, dtype=np.float32)
    for i in range(n_chunks):
        out[i * NUM_CHUNKS : (i + 1) * NUM_CHUNKS] = chunk_np
    t_prealloc = _timer() - t0
    del out

    # (b') preallocated torch buffer (the shape the decoder code would use).
    t0 = _timer()
    out_t = torch.empty(n_chunks * NUM_CHUNKS, dtype=torch.float32)
    for i in range(n_chunks):
        out_t[i * NUM_CHUNKS : (i + 1) * NUM_CHUNKS] = chunk_t
    t_prealloc_t = _timer() - t0
    del out_t

    # (c) list append + single final np.concatenate.
    t0 = _timer()
    parts = []
    for _ in range(n_chunks):
        parts.append(chunk_np)
    merged_np = np.concatenate(parts, axis=0)
    t_list_concat = _timer() - t0
    del parts, merged_np

    # (a2) HYPOTHESIZED pattern: np.concatenate per chunk (quadratic).
    # Measured at reduced chunk counts, quadratic model extrapolated.
    quad_samples = []
    for n_small in (128, 256, 512):
        t0 = _timer()
        acc = np.empty(0, dtype=np.float32)
        for _ in range(n_small):
            acc = np.concatenate([acc, chunk_np])
        quad_samples.append((n_small, _timer() - t0))
        del acc
    # t(n) ~= k * n^2 (copy volume ~ n^2/2 * chunk); fit k on the largest.
    k = quad_samples[-1][1] / (quad_samples[-1][0] ** 2)
    t_repeated_extrapolated = k * n_chunks**2

    print(f"{'pattern':<58} {'seconds':>12}")
    print(f"{'(a1) CURRENT: list + one torch.cat':<58} {t_current:>12.3f}")
    print(f"{'(b)  preallocated numpy + in-place slice writes':<58} {t_prealloc:>12.3f}")
    print(f"{'(b2) preallocated torch + in-place slice writes':<58} {t_prealloc_t:>12.3f}")
    print(f"{'(c)  list + one final np.concatenate':<58} {t_list_concat:>12.3f}")
    for n_small, t_small in quad_samples:
        print(f"     (a2) repeated np.concatenate, {n_small:>5} chunks (measured) {t_small:>12.3f}")
    print(f"{'(a2) repeated np.concatenate @ 4122 chunks (EXTRAPOLATED)':<58} "
          f"{t_repeated_extrapolated:>12.1f}  (~{t_repeated_extrapolated / 3600:.1f} h)")


def part3() -> None:
    from scipy import ndimage

    print("\n" + "=" * 88)
    print("PART 3: byte-identical replacement candidates (checked for equality on the spot)")
    print("=" * 88)
    field = FIELDS[0]
    # Rebuild the final-level state (surface at 257^3) exactly as the loop does.
    res_in, res_out = FINAL_RESOLUTION // 2, FINAL_RESOLUTION
    xs = np.linspace(-BOUNDS, BOUNDS, res_in + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    grid = field.numpy_field(pts).reshape(res_in + 1, res_in + 1, res_in + 1)

    inside = grid > 0.0
    surface = np.zeros_like(inside)
    for axis in range(3):
        changed = np.diff(inside, axis=axis)
        pad_lo = [(0, 0)] * 3
        pad_hi = [(0, 0)] * 3
        pad_lo[axis] = (0, 1)
        pad_hi[axis] = (1, 0)
        surface |= np.pad(changed, pad_lo, mode="constant")
        surface |= np.pad(changed, pad_hi, mode="constant")
    shell_scale = float(np.median(np.abs(grid)[surface]))
    band = max(0.95, 1.5 * shell_scale)
    surface |= np.abs(grid) < band
    surface = ndimage.binary_dilation(surface, iterations=2)

    scale = res_out // res_in
    fine_shape = (res_out + 1,) * 3

    # -- candidate A: shell build with in-place slice ORs (no np.pad copies)
    t0 = _timer()
    shell_old = np.zeros_like(inside)
    for axis in range(3):
        changed = np.diff(inside, axis=axis)
        pad_lo = [(0, 0)] * 3
        pad_hi = [(0, 0)] * 3
        pad_lo[axis] = (0, 1)
        pad_hi[axis] = (1, 0)
        shell_old |= np.pad(changed, pad_lo, mode="constant")
        shell_old |= np.pad(changed, pad_hi, mode="constant")
    t_old_shell = _timer() - t0
    t0 = _timer()
    shell_new = np.zeros_like(inside)
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    for axis in range(3):
        changed = np.diff(inside, axis=axis)
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        shell_new[tuple(lo)] |= changed
        shell_new[tuple(hi)] |= changed
        lo[axis] = slice(None)
        hi[axis] = slice(None)
    t_new_shell = _timer() - t0
    print(f"shell old (diff+pad+or): {t_old_shell:.3f}s | slice-OR: {t_new_shell:.3f}s | "
          f"equal: {np.array_equal(shell_old, shell_new)}")

    # -- candidate B: fine_mask via 27-row index matrix (old) vs per-offset scatter (new)
    coarse_idx = np.argwhere(surface)
    t0 = _timer()
    base = coarse_idx * scale
    block = np.stack(
        np.meshgrid(np.arange(scale + 1), np.arange(scale + 1), np.arange(scale + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    fine_idx = (base[:, None, :] + block[None, :, :]).reshape(-1, 3)
    np.minimum(fine_idx, res_out, out=fine_idx)
    mask_old = np.zeros(fine_shape, dtype=bool)
    mask_old[fine_idx[:, 0], fine_idx[:, 1], fine_idx[:, 2]] = True
    t_mask_old = _timer() - t0
    peak_rows = len(fine_idx)
    del base, fine_idx

    t0 = _timer()
    mask_new = np.zeros(fine_shape, dtype=bool)
    b0 = coarse_idx[:, 0] * scale
    b1 = coarse_idx[:, 1] * scale
    b2 = coarse_idx[:, 2] * scale
    for d0 in range(scale + 1):
        i0 = np.minimum(b0 + d0, res_out)
        for d1 in range(scale + 1):
            i1 = np.minimum(b1 + d1, res_out)
            for d2 in range(scale + 1):
                i2 = np.minimum(b2 + d2, res_out)
                mask_new[i0, i1, i2] = True
    t_mask_new = _timer() - t0
    print(f"fine_mask old (27N matrix {peak_rows:,} rows): {t_mask_old:.3f}s | "
          f"per-offset scatter: {t_mask_new:.3f}s | equal: {np.array_equal(mask_old, mask_new)}")

    # -- candidate C: argwhere + strided columns (old) vs nonzero tuple (new)
    t0 = _timer()
    refine_idx = np.argwhere(mask_old)
    xs_f = np.linspace(-BOUNDS, BOUNDS, res_out + 1, dtype=np.float64)
    pts_old = np.stack(
        [xs_f[refine_idx[:, 0]], xs_f[refine_idx[:, 1]], xs_f[refine_idx[:, 2]]], axis=-1
    ).astype(np.float32)
    t_gather_old = _timer() - t0
    t0 = _timer()
    i0, i1, i2 = np.nonzero(mask_new)
    pts_new = np.stack([xs_f[i0], xs_f[i1], xs_f[i2]], axis=-1).astype(np.float32)
    t_gather_new = _timer() - t0
    print(f"refine gather old (argwhere+strided): {t_gather_old:.3f}s | nonzero tuple: {t_gather_new:.3f}s | "
          f"equal: {np.array_equal(pts_old, pts_new)}")

    # -- candidate D: scatter back, strided columns vs contiguous tuple
    logits = field.numpy_field(pts_old)
    target_old = np.zeros(fine_shape, dtype=np.float32)
    t0 = _timer()
    target_old[refine_idx[:, 0], refine_idx[:, 1], refine_idx[:, 2]] = logits
    t_scatter_old = _timer() - t0
    target_new = np.zeros(fine_shape, dtype=np.float32)
    t0 = _timer()
    target_new[i0, i1, i2] = logits
    t_scatter_new = _timer() - t0
    print(f"logit scatter old (strided cols): {t_scatter_old:.3f}s | contiguous tuple: {t_scatter_new:.3f}s | "
          f"equal: {np.array_equal(target_old, target_new)}")

    # -- candidate E: zoom astype copy elision
    t0 = _timer()
    z1 = ndimage.zoom(grid, tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape)), order=1, mode="nearest").astype(np.float32)
    t_zoom_copy = _timer() - t0
    t0 = _timer()
    z2 = ndimage.zoom(grid, tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape)), order=1, mode="nearest").astype(np.float32, copy=False)
    t_zoom_nocopy = _timer() - t0
    print(f"zoom+astype(copy=True): {t_zoom_copy:.3f}s | astype(copy=False): {t_zoom_nocopy:.3f}s | "
          f"equal: {np.array_equal(z1, z2)}")


if __name__ == "__main__":
    print(f"numpy {np.__version__}")
    part1()
    part2()
    part3()
