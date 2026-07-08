"""Tests for the public bundle API, the profiler, and the bit-exactness
contracts behind the performance work (balanced query, windowed commits)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from abstract3d import bundle as bundle_api
from abstract3d import texturing
from abstract3d.profiling import MemorySampler, StageProfiler


def make_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    mesh.export(bundle_dir / "geometry.glb")
    Image.new("RGBA", (64, 64), (120, 150, 200, 255)).save(bundle_dir / "input.png")
    (bundle_dir / "metadata.json").write_text(json.dumps({"texture_resolution": 64}))
    return bundle_dir


def test_load_bundle_exposes_geometry_and_metadata(tmp_path) -> None:
    bundle_dir = make_bundle(tmp_path)
    loaded = bundle_api.load_bundle(bundle_dir)
    assert loaded.metadata["texture_resolution"] == 64
    mesh = loaded.geometry_mesh()
    assert len(mesh.vertices) > 0


def test_load_bundle_without_geometry_raises(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    loaded = bundle_api.load_bundle(empty)
    with pytest.raises(FileNotFoundError, match="cannot be rebaked"):
        loaded.geometry_mesh()


def test_prepare_observed_views_source_and_reference_roles(tmp_path) -> None:
    photo = tmp_path / "ref.png"
    Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(photo)
    views = bundle_api.prepare_observed_views(
        Image.new("RGBA", (32, 32), (200, 100, 50, 255)),
        references=[{"image": photo, "angle": "side_left"}],
        remove_source_background=False,
    )
    assert views[0]["role"] == "source"
    assert views[0]["identity_image"] is not None
    assert views[1]["role"] == "reference"
    assert views[1]["azimuth_deg"] == 90.0
    assert views[1]["label"] == "side_left"


def test_rebake_bundle_writes_revision(tmp_path) -> None:
    bundle_dir = make_bundle(tmp_path)
    out_dir = tmp_path / "rebake"
    _mesh, stats = bundle_api.rebake_bundle(
        bundle_dir, output_dir=out_dir, texture_resolution=64)
    assert (out_dir / "scene.glb").exists()
    assert (out_dir / "texture.png").exists()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["schema_version"] == bundle_api.BUNDLE_SCHEMA_VERSION
    assert metadata["texture_resolution"] == 64
    assert metadata["texture_png_md5"]
    assert stats.get("texture_image") is not None


def test_rebake_bundle_is_deterministic(tmp_path) -> None:
    bundle_dir = make_bundle(tmp_path)
    hashes = []
    for name in ("a", "b"):
        out_dir = tmp_path / name
        bundle_api.rebake_bundle(bundle_dir, output_dir=out_dir, texture_resolution=64)
        metadata = json.loads((out_dir / "metadata.json").read_text())
        hashes.append(metadata["texture_png_md5"])
    assert hashes[0] == hashes[1]


def test_memory_sampler_records_rss() -> None:
    sampler = MemorySampler(interval_s=0.01, sample_mps=False).start()
    _ = [np.zeros((256, 256)) for _ in range(20)]
    sampler.stop()
    assert sampler.peak("rss") is not None
    assert len(sampler.samples) >= 2


def test_stage_profiler_wraps_and_restores() -> None:
    import types

    module = types.SimpleNamespace(fn=lambda x: x * 2)
    sampler = MemorySampler(interval_s=0.01, sample_mps=False).start()
    profiler = StageProfiler(sampler=sampler)
    profiler.wrap_module_functions(module, ["fn"])
    assert module.fn(21) == 42
    profiler.unwrap()
    sampler.stop()
    report = profiler.report()
    assert report["stages"][0]["name"] == "fn"
    assert report["stages"][0]["seconds"] >= 0.0
    # unwrap restored the original callable
    assert not hasattr(module.fn, "__wrapped__")


def test_balanced_query_matches_direct_query() -> None:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(3)
    tree_points = rng.normal(size=(2000, 3)).astype(np.float32)
    queries = rng.normal(size=(5000, 3)).astype(np.float32)
    tree = cKDTree(tree_points)
    d_direct, i_direct = tree.query(queries, k=4, workers=-1)
    d_balanced, i_balanced = texturing._balanced_query(tree, queries, k=4)
    assert (d_direct == d_balanced).all()
    assert (i_direct == i_balanced).all()


def test_mirror_fill_bounded_query_matches_reference_semantics() -> None:
    """The pruned+parallel mirror twin lookup must keep the acceptance set
    and colors identical to the unbounded exact-NN formulation."""

    rng = np.random.default_rng(7)
    size = 96
    positions = np.zeros((size, size, 4), np.float32)
    # a sphere-ish surface patch, symmetric across axis 1
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    positions[:, :, 0] = xs
    positions[:, :, 1] = ys
    positions[:, :, 2] = np.sqrt(np.clip(0.3 - xs**2 - ys**2, 0.0, None))
    positions[:, :, 3] = 1.0
    observed = np.zeros((size, size), bool)
    observed[:, : size // 2] = True  # observe one side only
    colors = rng.random((size, size, 3)).astype(np.float32)

    fill_rgb, fill_mask = texturing.mirror_fill_from_observed(
        positions_texture=positions,
        observed_mask=observed,
        colors_rgb=colors,
        axis=1,
    )
    # reference: brute-force exact mirror twins under the same threshold
    surface = positions[:, :, 3] > 0
    unseen = surface & ~observed
    obs_pts = positions[:, :, :3][observed]
    scale = float(np.linalg.norm(obs_pts.max(axis=0) - obs_pts.min(axis=0)))
    mirrored = positions[:, :, :3][unseen].copy()
    mirrored[:, 1] *= -1.0
    dists = np.linalg.norm(
        mirrored[:, None, :] - obs_pts[None, :, :], axis=2)
    expected_valid = dists.min(axis=1) <= 0.02 * scale
    assert int(fill_mask.sum()) == int(expected_valid.sum())
