"""Tests for the whole-bake A/B acceptance gate.

The gate's job is monotone non-regression of the finished product: a
generated-references bake ships only if it does not regress the
no-references baseline on photo fidelity, brightness, and composition
tone damage. Real-bundle calibration lives in the fix-program fixtures
(labeled accepts: owl / portrait / starship / chair-clean / chair +12 L /
car @2048+1024; labeled refusals: +/-25 L mis-toned and content-shifted
chair backs — margins in CHANGELOG); these tests pin the gate MECHANICS
on synthetic textured meshes:

- revising unseen surface with same-tone content ships (completion),
- crisp new content with a bounded tone change ships (the car class the
  old long-edge seam axis wrongly refused),
- coherent tone displacement beyond the pipeline's sanctioned-adjustment
  floor refuses, in BOTH directions (the chair incident class, which
  gradient-domain compositing hides from any step detector),
- scattered high-contrast detail never refuses (mottle/grain immunity),
- the fidelity pose comes from the bake stats when the caller passes
  them (the measured (0,0)-pose bug class).
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


def back_banded(base_seed: int, band: Image.Image | np.ndarray) -> trimesh.Trimesh:
    """Candidate revising only the sphere's BACK (u < 0.1 and u > 0.9 map
    to the back; the front at u = 0.5 stays identical) — the shape of a
    completion-only generated-references bake."""

    revised = np.asarray(noise_texture(base_seed), np.uint8).copy()
    band = np.asarray(band, np.uint8)
    revised[:, :26] = band[:, :26]
    revised[:, -26:] = band[:, -26:]
    return textured_sphere(Image.fromarray(revised, "RGB"))


def test_non_regressing_candidate_is_accepted(baseline) -> None:
    """Same-tone back revision (what completion-only generated views do)
    must ship: front content identical, no tone displacement."""

    base_mesh, photo = baseline
    verdict = evaluate_generated_bake(
        base_mesh, back_banded(1, noise_texture(2)), source_rgba=photo)
    assert verdict["accepted"] is True
    assert verdict["reasons"] == []
    assert {"photo_fidelity_delta_e", "front_brightness_l",
            "composition_tone_damage", "seam_ratio",
            "source_pose"} <= set(verdict["metrics"])


def test_crisp_content_with_bounded_tone_change_is_accepted(baseline) -> None:
    """The car class: references replace blank fill with real content —
    crisp long contours, tone moved LESS than the sanctioned-adjustment
    floor (25 L). The retired long-edge axis refused exactly this
    (measured +0.029 long-edge delta on the labeled-good car); the tone
    axis must not."""

    base_mesh, photo = baseline
    # 150-200 noise is L ~ 65-75; 105-155 noise is L ~ 48-63: a ~12 L
    # low-band drop, inside the sanctioned floor, with hard step edges
    # against the untouched territory.
    candidate = back_banded(1, noise_texture(3, low=105, high=155))
    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    damage = verdict["metrics"]["composition_tone_damage"]
    assert damage["darken_worst"] <= damage["darken_max_allowed"]
    assert not any("tone damage" in reason for reason in verdict["reasons"])
    assert verdict["accepted"] is True


def test_darkened_candidate_is_rejected(baseline) -> None:
    base_mesh, photo = baseline
    dark = np.asarray(noise_texture(1), np.float32) * 0.55
    candidate = textured_sphere(Image.fromarray(dark.astype(np.uint8), "RGB"))
    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("brightness" in reason for reason in verdict["reasons"])


def test_coherent_tone_displacement_is_rejected_both_directions(baseline) -> None:
    """The chair incident class: a generated view lands at a wrong tone
    and the compositor smooths the handoff — no sharp step survives, but
    the displaced REGION does. Both directions must refuse (the measured
    incident darkened; the +25 L synthesis brightened)."""

    base_mesh, photo = baseline
    # ~40 L coherent drop on the back region only (30-60 gray vs 150-200).
    dark_back = back_banded(1, noise_texture(4, low=30, high=60))
    verdict = evaluate_generated_bake(base_mesh, dark_back, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("tone damage (darkening)" in r for r in verdict["reasons"])

    # Bright direction needs a dark baseline for the displacement to be
    # physically reachable (L saturates at 100; the brighten budget is
    # deliberately the looser one because legitimate fill-rescue
    # brightens): dark sphere (~L 30-48) whose back turns near-white.
    dark_sphere = textured_sphere(noise_texture(21, low=60, high=110))
    bright = np.asarray(noise_texture(21, low=60, high=110), np.uint8).copy()
    white_band = np.asarray(noise_texture(5, low=245, high=255), np.uint8)
    bright[:, :26] = white_band[:, :26]
    bright[:, -26:] = white_band[:, -26:]
    bright_back = textured_sphere(Image.fromarray(bright, "RGB"))
    verdict = evaluate_generated_bake(
        dark_sphere, bright_back, source_rgba=source_photo_of(dark_sphere))
    assert verdict["accepted"] is False
    assert any("tone damage (brightening)" in r for r in verdict["reasons"])


def test_scattered_detail_is_not_rejected(baseline) -> None:
    """High-contrast texture DETAIL (grain, mottle, panel lines) must
    never read as damage: scattered dark dashes move no low-band region
    beyond the floor."""

    base_mesh, photo = baseline
    detail = np.asarray(noise_texture(1), np.uint8).copy()
    rng = np.random.default_rng(7)
    for _ in range(60):
        r = int(rng.integers(0, 250))
        c = int(rng.integers(0, 250))
        detail[r:r + 2, c:c + 6] = 30
    detailed = textured_sphere(Image.fromarray(detail, "RGB"))
    verdict = evaluate_generated_bake(base_mesh, detailed, source_rgba=photo)
    assert not any("tone damage" in reason for reason in verdict["reasons"])


def red_noise_texture(seed: int, *, size: int = 256) -> Image.Image:
    """Saturated warm-red noise (chroma ~ 50): the anchor-class palette
    the hue axis is calibrated on; gray noise sits under the saturation
    floor and cannot exercise it."""

    rng = np.random.default_rng(seed)
    grain = np.zeros((size, size, 3), dtype=np.uint8)
    grain[:, :, 0] = rng.integers(150, 200, size=(size, size))
    grain[:, :, 1] = rng.integers(20, 60, size=(size, size))
    grain[:, :, 2] = rng.integers(20, 60, size=(size, size))
    return Image.fromarray(grain, "RGB")


def hue_rotate_array(rgb: np.ndarray, degrees: float) -> np.ndarray:
    from skimage import color as skcolor

    lab = skcolor.rgb2lab(rgb.astype(np.float32) / 255.0)
    theta = np.deg2rad(degrees)
    a, b = lab[:, :, 1].copy(), lab[:, :, 2].copy()
    lab[:, :, 1] = a * np.cos(theta) - b * np.sin(theta)
    lab[:, :, 2] = a * np.sin(theta) + b * np.cos(theta)
    return np.clip(skcolor.lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)


def test_constant_l_hue_rotation_is_rejected() -> None:
    """The measured L-axis hole (integrator program): a back region
    hue-rotated 30 deg at constant L passes fidelity/brightness/tone —
    only the chroma axis can refuse it."""

    base_mesh = textured_sphere(red_noise_texture(11))
    try:
        photo = source_photo_of(base_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")
    base = np.asarray(red_noise_texture(11), np.uint8).copy()
    rotated = hue_rotate_array(base, 30.0)
    # back band only (u < 0.1 / u > 0.9): the front the photo judges
    # stays identical, exactly like the fixture that measured the hole
    base[:, :26] = rotated[:, :26]
    base[:, -26:] = rotated[:, -26:]
    candidate = textured_sphere(Image.fromarray(base, "RGB"))

    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("hue damage" in reason for reason in verdict["reasons"])
    # the hole this closes: no OTHER axis may be carrying the refusal
    assert all("hue damage" in reason for reason in verdict["reasons"])


def test_scattered_hue_detail_is_not_rejected() -> None:
    """Hue-flipped DETAIL (specks, trim, badges) is content references
    legitimately add; only a coherent low-band rotation may refuse."""

    base_mesh = textured_sphere(red_noise_texture(11))
    try:
        photo = source_photo_of(base_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")
    detail = np.asarray(red_noise_texture(11), np.uint8).copy()
    rotated = hue_rotate_array(detail, 30.0)
    rng = np.random.default_rng(7)
    for _ in range(60):
        r = int(rng.integers(0, 250))
        c = int(rng.integers(0, 250))
        detail[r:r + 2, c:c + 6] = rotated[r:r + 2, c:c + 6]
    candidate = textured_sphere(Image.fromarray(detail, "RGB"))

    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert not any("hue damage" in reason for reason in verdict["reasons"])


def test_gate_pose_resolves_from_bake_stats(baseline) -> None:
    """The fidelity pose must come from the bake's own recorded estimate
    when the caller passes stats (measured bug class: gating a
    pose-estimated car at a hardcoded (0,0) charged ~9 dE of pose error
    to both sides and flipped the verdict); an explicit pose wins over
    stats (external capture fact)."""

    base_mesh, photo = baseline
    candidate = back_banded(1, noise_texture(2))
    stats = {"source_pose": {"azimuth_deg": 17.5, "elevation_deg": 8.0}}
    verdict = evaluate_generated_bake(
        base_mesh, candidate, source_rgba=photo, baseline_stats=stats)
    assert verdict["metrics"]["source_pose"] == {
        "azimuth_deg": 17.5, "elevation_deg": 8.0, "origin": "baseline_stats"}

    verdict = evaluate_generated_bake(
        base_mesh, candidate, source_rgba=photo, source_pose=(5.0, 0.0),
        baseline_stats=stats)
    assert verdict["metrics"]["source_pose"]["origin"] == "explicit"
    assert verdict["metrics"]["source_pose"]["azimuth_deg"] == 5.0

    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert verdict["metrics"]["source_pose"] == {
        "azimuth_deg": 0.0, "elevation_deg": 0.0, "origin": "default"}
