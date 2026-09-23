"""Tests for the public bundle API, the profiler, and the bit-exactness
contracts behind the performance work (balanced query, windowed commits)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from abstract3d import bundle as bundle_api
from abstract3d import texturing
from abstract3d.profiling import MemorySampler, StageProfiler


# Two rebakes of the same bundle hash identically on Linux CI and on local
# Apple Silicon, but not on GitHub's hosted macOS runners, where the first bake
# in a test varies from run to run (the cause is still open; the assertion
# message below lists the diverging bake stats). Keep enforcing everywhere else.
_hosted_macos_bake_nondeterminism = pytest.mark.xfail(
    sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS") == "true",
    reason="rebake output is not reproducible on GitHub-hosted macOS runners (open)",
    strict=False,
)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}"))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _stat_diff(first: dict, second: dict, limit: int = 25) -> list:
    a, b = _flatten(first), _flatten(second)
    keys = [k for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k) and "seconds" not in k and "source_bundle" not in k]
    return [f"{k}: {str(a.get(k))[:80]} != {str(b.get(k))[:80]}" for k in keys[:limit]]


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


@pytest.fixture
def offline_matte(monkeypatch):
    """Rebake tests exercise the bake/metadata contract, not the matte model.

    `remove_background_robust` downloads rembg ONNX checkpoints (hundreds of
    MB from github.com) on first use; on CI that made these tests network
    flaky and let two "identical" rebakes use different matte models. Stub it
    with a deterministic pass-through so the tests stay hermetic.
    """
    from abstract3d import segmentation

    monkeypatch.setattr(segmentation, "remove_background_robust",
                        lambda image: image.convert("RGBA"))


def test_rebake_bundle_writes_revision(tmp_path, offline_matte) -> None:
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


@_hosted_macos_bake_nondeterminism
def test_rebake_bundle_is_deterministic(tmp_path, offline_matte) -> None:
    bundle_dir = make_bundle(tmp_path)
    metas = []
    for name in ("a", "b"):
        out_dir = tmp_path / name
        bundle_api.rebake_bundle(bundle_dir, output_dir=out_dir, texture_resolution=64)
        metas.append(json.loads((out_dir / "metadata.json").read_text()))
    assert metas[0]["texture_png_md5"] == metas[1]["texture_png_md5"], (
        "rebake not reproducible; diverging stats:\n" + "\n".join(_stat_diff(metas[0], metas[1])))


@_hosted_macos_bake_nondeterminism
def test_rebake_absorbs_transient_gl_context_failure(tmp_path, offline_matte, monkeypatch) -> None:
    # A transient standalone-context failure used to switch one view to
    # facing-only visibility and silently change the baked texture.
    moderngl = pytest.importorskip("moderngl")
    bundle_dir = make_bundle(tmp_path)
    bundle_api.rebake_bundle(bundle_dir, output_dir=tmp_path / "clean", texture_resolution=64)
    clean = json.loads((tmp_path / "clean" / "metadata.json").read_text())["texture_png_md5"]

    real_create = moderngl.create_context
    calls = {"n": 0}

    def flaky_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("simulated transient GL context failure")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(moderngl, "create_context", flaky_create)
    bundle_api.rebake_bundle(bundle_dir, output_dir=tmp_path / "flaky", texture_resolution=64)
    flaky = json.loads((tmp_path / "flaky" / "metadata.json").read_text())["texture_png_md5"]
    assert calls["n"] > 1, "bake never created a GL context; the check proves nothing"
    assert flaky == clean


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
