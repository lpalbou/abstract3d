"""Tests for the whole-bake A/B acceptance gate.

The gate's job is monotone non-regression of the finished product: a
generated-references bake ships only if it does not regress the
no-references baseline on photo fidelity, brightness, and long seam
steps. Real-bundle calibration lives in the proof pack (the labeled
chair regression must reject; owl/spaceship/portrait must pass); these
tests pin the mechanics on synthetic textured meshes.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.texture import SimpleMaterial, TextureVisuals

from abstract3d.bake_acceptance import evaluate_generated_bake


def textured_sphere(texture: Image.Image) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    normals = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    u = 0.5 + np.arctan2(normals[:, 1], normals[:, 0]) / (2.0 * np.pi)
    v = 0.5 + np.arcsin(np.clip(normals[:, 2], -1.0, 1.0)) / np.pi
    mesh.visual = TextureVisuals(
        uv=np.column_stack([u, v]).astype(np.float32),
        image=texture,
        # PIL image (not encoded bytes) so the preview renderer samples it;
        # white diffuse so no 0.4-gray SimpleMaterial factor dims the render.
        material=SimpleMaterial(image=texture, diffuse=[255, 255, 255, 255]),
    )
    return mesh


def noise_texture(seed: int, *, size: int = 256, low: int = 150,
                  high: int = 200) -> Image.Image:
    rng = np.random.default_rng(seed)
    grain = rng.integers(low, high, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(grain, "RGB")


def source_photo_of(mesh: trimesh.Trimesh) -> Image.Image:
    """A stand-in 'source photo': the mesh's own front render, matted."""

    from abstract3d.reference_generation import clay_silhouette
    from abstract3d.rendering import render_mesh_views

    render = render_mesh_views(mesh, size=256, azimuths=[0.0], elevation=0.0)[0]
    rgba = np.asarray(render.convert("RGBA")).copy()
    rgba[:, :, 3] = np.where(clay_silhouette(render), 255, 0)
    return Image.fromarray(rgba, "RGBA")


@pytest.fixture(scope="module")
def baseline():
    mesh = textured_sphere(noise_texture(1))
    try:
        photo = source_photo_of(mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")
    return mesh, photo


def test_non_regressing_candidate_is_accepted(baseline) -> None:
    """A candidate that only revises the UNSEEN back (what completion-only
    generated views do) must pass: front content identical, no long seams,
    brightness stable."""

    base_mesh, photo = baseline
    revised = np.asarray(noise_texture(1), np.uint8).copy()
    back_band = np.asarray(noise_texture(2), np.uint8)
    # u < 0.1 and u > 0.9 map to the back of the sphere (front is u=0.5).
    revised[:, :26] = back_band[:, :26]
    revised[:, -26:] = back_band[:, -26:]
    candidate = textured_sphere(Image.fromarray(revised, "RGB"))
    verdict = evaluate_generated_bake(
        base_mesh, candidate, source_rgba=photo)
    assert verdict["accepted"] is True
    assert verdict["reasons"] == []
    assert set(verdict["metrics"]) == {
        "photo_fidelity_delta_e", "front_brightness_l", "seam_ratio"}


def test_darkened_candidate_is_rejected(baseline) -> None:
    base_mesh, photo = baseline
    dark = np.asarray(noise_texture(1), np.float32) * 0.55
    candidate = textured_sphere(Image.fromarray(dark.astype(np.uint8), "RGB"))
    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("brightness" in reason for reason in verdict["reasons"])


def test_long_seam_step_is_rejected_but_fine_detail_is_not(baseline) -> None:
    base_mesh, photo = baseline

    # Hard full-height/full-width tone bands wrap the sphere as long
    # meridian and equator step edges — the chair-class handoff failure.
    seam = np.asarray(noise_texture(1), np.uint8).copy()
    seam[:, 105:145] = 30
    seam[110:150, :] = 30
    seamed = textured_sphere(Image.fromarray(seam, "RGB"))
    verdict = evaluate_generated_bake(base_mesh, seamed, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("seam" in reason for reason in verdict["reasons"])

    # The same contrast as scattered short dashes is texture DETAIL: the
    # extent filter must not count it, and detail must not fail the gate.
    detail = np.asarray(noise_texture(1), np.uint8).copy()
    rng = np.random.default_rng(7)
    for _ in range(60):
        r = int(rng.integers(0, 250))
        c = int(rng.integers(0, 250))
        detail[r:r + 2, c:c + 6] = 30
    detailed = textured_sphere(Image.fromarray(detail, "RGB"))
    verdict = evaluate_generated_bake(base_mesh, detailed, source_rgba=photo)
    assert not any("seam" in reason for reason in verdict["reasons"])
