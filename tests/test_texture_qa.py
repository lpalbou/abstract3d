"""Unit tests for the close-range texture QA harness (scripts/texture_qa.py).

Detector tests run on synthetic images with known ground truth; material
gates run on GLBs assembled in-memory. No GPU context is required: the
UV-rasterization and render paths are exercised by the integration runs in
scripts/texture_qa.py itself, while these tests pin the pure-CPU logic the
verdicts depend on (parsing, detectors, calibration invariants).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

texture_qa = pytest.importorskip("texture_qa")
texture_qa_render = pytest.importorskip("texture_qa_render")


# ---------------------------------------------------------------------------
# synthetic image helpers
# ---------------------------------------------------------------------------

def voronoi_flat_image(size: int = 256, cells: int = 120, seed: int = 3) -> np.ndarray:
    """Piecewise-constant Voronoi cells: the nearest-vertex fill signature."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, size, size=(cells, 2))
    colors = rng.integers(60, 200, size=(cells, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    d = (yy[:, :, None] - seeds[None, None, :, 0]) ** 2 \
        + (xx[:, :, None] - seeds[None, None, :, 1]) ** 2
    label = np.argmin(d, axis=2)
    return colors[label].astype(np.uint8)


def noisy_photo_like(size: int = 256, seed: int = 5) -> np.ndarray:
    """Smooth gradient + sensor-like noise: legitimate photo texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 120 + 40 * np.sin(xx / 37.0) + 30 * (yy / size)
    noise = rng.normal(0, 3.0, (size, size))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=2)


# ---------------------------------------------------------------------------
# facet detector
# ---------------------------------------------------------------------------

def test_facet_detector_fires_on_voronoi_fill():
    thr = texture_qa.Thresholds()
    img = voronoi_flat_image()
    region = np.ones(img.shape[:2], bool)
    out = texture_qa.detect_facet_blocks(img, region, thr)
    assert out["facet_field"], "Voronoi flat cells must register as a facet field"
    assert out["cellular_fraction"] > 0.3


def test_facet_detector_passes_photo_like_texture():
    thr = texture_qa.Thresholds()
    img = noisy_photo_like()
    region = np.ones(img.shape[:2], bool)
    out = texture_qa.detect_facet_blocks(img, region, thr)
    assert not out["facet_field"], "noisy photo texture must not read as facets"


# ---------------------------------------------------------------------------
# seam detector
# ---------------------------------------------------------------------------

def test_seam_detector_measures_tone_step():
    rng = np.random.default_rng(11)
    img = np.full((200, 200, 3), 140, np.uint8)
    img[:, 100:] = 175  # 35-level tone step between the two halves
    img = np.clip(img.astype(np.int16)
                  + rng.normal(0, 2, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    side_a = np.zeros((200, 200), bool)
    side_b = np.zeros((200, 200), bool)
    side_a[:, :100] = True
    side_b[:, 100:] = True
    out = texture_qa.seam_steps(img, side_a, side_b)
    assert out is not None and out["p95"] > 15.0


def test_seam_detector_quiet_on_matched_texture():
    rng = np.random.default_rng(12)
    img = np.clip(150 + rng.normal(0, 4, (200, 200, 3)), 0, 255).astype(np.uint8)
    side_a = np.zeros((200, 200), bool)
    side_b = np.zeros((200, 200), bool)
    side_a[:, :100] = True
    side_b[:, 100:] = True
    out = texture_qa.seam_steps(img, side_a, side_b)
    assert out is not None and out["p95"] < 6.0


# ---------------------------------------------------------------------------
# dark smear detector
# ---------------------------------------------------------------------------

def test_dark_smears_fill_conditioned():
    img = np.full((256, 256, 3), 170, np.uint8)
    img[100:118, 100:130] = 12  # feature-dark fragment (within the area band)
    mask = np.ones((256, 256), bool)
    fill = np.zeros((256, 256), bool)
    fill[80:180, 80:200] = True
    hit = texture_qa.detect_dark_smears(img, mask, fill, texture_qa.Thresholds())
    assert len(hit["smears"]) == 1 and hit["smears"][0]["in_fill"]
    # same content with no fill mask (photo calibration mode): reports only
    quiet = texture_qa.detect_dark_smears(img, mask, None, texture_qa.Thresholds())
    assert quiet["smears"] == []
    # dark fragment OUTSIDE fill does not gate (observed-space is verdict1's job)
    off_fill = np.zeros((256, 256), bool)
    off_fill[200:250, 200:250] = True
    none = texture_qa.detect_dark_smears(img, mask, off_fill, texture_qa.Thresholds())
    assert none["smears"] == []
    # a dark area larger than the fragment band is shading, not a fragment
    big = np.full((256, 256, 3), 170, np.uint8)
    big[60:200, 60:200] = 12
    shading = texture_qa.detect_dark_smears(big, mask, fill, texture_qa.Thresholds())
    assert shading["smears"] == []


# ---------------------------------------------------------------------------
# fill character
# ---------------------------------------------------------------------------

def test_fill_character_flags_flat_fill():
    size = 256
    rng = np.random.default_rng(21)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, : size // 2] = np.clip(
        140 + rng.normal(0, 18, (size, size // 2, 1)), 0, 255).astype(np.uint8)
    img[:, size // 2:] = 128  # flat mush fill
    observed = np.zeros((size, size), bool)
    observed[:, : size // 2] = True
    fill = ~observed
    regions = texture_qa.Regions(
        observed=observed, symmetry=np.zeros_like(observed), fill=fill,
        per_view={"front": observed}, reconciliation=[])
    out = texture_qa.fill_character(img, regions)
    assert out["fill_to_observed_ratio"] < 0.2


# ---------------------------------------------------------------------------
# GLB material truth
# ---------------------------------------------------------------------------

def _glb_bytes(materials: list[dict]) -> bytes:
    gltf = {
        "asset": {"version": "2.0"},
        "materials": materials,
        "bufferViews": [],
        "images": [],
        "textures": [],
    }
    payload = json.dumps(gltf).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload))
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return header + chunk


def test_parse_glb_reads_raw_materials(tmp_path):
    factor = [0.4, 0.4, 0.4, 1.0]
    path = tmp_path / "scene.glb"
    path.write_bytes(_glb_bytes([{
        "pbrMetallicRoughness": {"baseColorFactor": factor,
                                 "roughnessFactor": 0.9},
    }]))
    gltf, _ = texture_qa_render.parse_glb(path)
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert pbr["baseColorFactor"] == factor
    assert "metallicFactor" not in pbr  # spec default 1.0 must stay visible


def test_material_gates_fail_dark_metal_and_pass_fixed(tmp_path):
    from PIL import Image as PILImage

    class FakeBundle:
        directory = tmp_path
        texture = PILImage.new("RGB", (1024, 1024), (128, 128, 128))

        @property
        def texture_array(self):
            return np.asarray(self.texture)

    thr = texture_qa.Thresholds()
    bad = FakeBundle()
    bad.gltf = {"materials": [{"pbrMetallicRoughness": {
        "baseColorFactor": [0.4, 0.4, 0.4, 1.0],
        "baseColorTexture": {"index": 0},
        "roughnessFactor": 0.9}}]}
    gates = {g.name: g.passed for g in texture_qa.material_gates(bad, thr)}
    assert not gates["material.glb.base_color_factor"]
    assert not gates["material.glb.metallic_factor"]

    good = FakeBundle()
    good.gltf = {"materials": [{"pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "baseColorTexture": {"index": 0},
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0}}]}
    gates = {g.name: g.passed for g in texture_qa.material_gates(good, thr)}
    assert gates["material.glb.base_color_factor"]
    assert gates["material.glb.metallic_factor"]
    assert gates["material.glb.roughness_factor"]


def test_mtl_gate(tmp_path):
    from PIL import Image as PILImage

    (tmp_path / "scene.mtl").write_text(
        "newmtl material_0\nKd 0.4 0.4 0.4\nKs 0.4 0.4 0.4\nmap_Kd material_0.png\n")

    class FakeBundle:
        directory = tmp_path
        texture = PILImage.new("RGB", (1024, 1024), (128, 128, 128))
        gltf = {"materials": [{"pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
            "baseColorTexture": {"index": 0}}}]}

        @property
        def texture_array(self):
            return np.asarray(self.texture)

    gates = {g.name: g.passed
             for g in texture_qa.material_gates(FakeBundle(), texture_qa.Thresholds())}
    assert not gates["material.mtl.kd"]
    assert not gates["material.mtl.ks"]
