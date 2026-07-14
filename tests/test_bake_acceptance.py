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
  them (the measured (0,0)-pose bug class),
- the artifact battery refuses a candidate that ADDS a foreign pale
  blotch (the image-in-image stamp payload / white rim-splash class)
  and never punishes inherited or removed artifacts (A/B direction).
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
    assert verdict["warnings"] == []
    assert {"photo_fidelity_delta_e", "front_brightness_l",
            "composition_tone_damage", "seam_ratio",
            "source_pose", "artifact_battery"} <= set(verdict["metrics"])
    battery = verdict["metrics"]["artifact_battery"]
    assert battery["added_pale_blotch"]["votes"] is True
    assert battery["added_pale_wash"]["votes"] is False


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
    only the chroma axis can refuse it. The source-evidence veto must
    NOT silence it: rotated hue sits off the photo's own hue band, so
    the charge stands (this is the trap co-observed masking would have
    reopened — rotated references also land on baseline-fill surface)."""

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
    # and the veto must have left the charge standing (off-evidence)
    damage = verdict["metrics"]["composition_hue_damage"]
    assert damage["worst"] > damage["max_allowed"]
    assert damage["source_band"] is not None


def test_fill_hue_confound_matching_source_hue_is_accepted() -> None:
    """The measured false-refusal incident (hue program, /tmp/hue1): on
    never-witnessed surface the BASELINE carries propagated fill whose
    low-band hue drifts off the subject's, and the old single-population
    axis charged a CORRECT candidate for disagreeing with that fill
    (live pair: 1.869 vs budget 1.0). Fixture: the baseline's back band
    is hue-drifted 'fill mottle'; the candidate's back is fresh content
    matching the source photo's own hue. The drift charge is real
    (worst_raw over budget) but the source-evidence veto must clear it."""

    base = np.asarray(red_noise_texture(11), np.uint8).copy()
    drifted = hue_rotate_array(base, 20.0)
    base[:, :26] = drifted[:, :26]
    base[:, -26:] = drifted[:, -26:]
    base_mesh = textured_sphere(Image.fromarray(base, "RGB"))
    try:
        photo = source_photo_of(base_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")

    # candidate: same front, back replaced by NEW content in the
    # photo's own hue family (what a correct reference does to fill)
    revised = np.asarray(red_noise_texture(11), np.uint8).copy()
    fresh = np.asarray(red_noise_texture(12), np.uint8)
    revised[:, :26] = fresh[:, :26]
    revised[:, -26:] = fresh[:, -26:]
    candidate = textured_sphere(Image.fromarray(revised, "RGB"))

    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    damage = verdict["metrics"]["composition_hue_damage"]
    # the confound is present: the legacy charge alone would refuse
    assert damage["worst_raw"] > damage["max_allowed"]
    # ... but candidate hue matches the subject's evidence: no damage
    assert damage["worst"] <= damage["max_allowed"]
    assert not any("hue damage" in reason for reason in verdict["reasons"])
    assert verdict["accepted"] is True


def test_colorless_photo_keeps_legacy_hue_charge() -> None:
    """Fail-closed: a photo with no saturated hue mass yields no
    evidence band, and the axis must keep the legacy single-population
    charge (the veto may only fire on positive evidence — otherwise a
    colorless-photo bake would ship any palette rotation)."""

    base_mesh = textured_sphere(red_noise_texture(11))
    try:
        photo = source_photo_of(base_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")
    gray = np.asarray(photo.convert("RGBA")).copy()
    gray[:, :, :3] = 128
    gray_photo = Image.fromarray(gray, "RGBA")

    base = np.asarray(red_noise_texture(11), np.uint8).copy()
    rotated = hue_rotate_array(base, 30.0)
    base[:, :26] = rotated[:, :26]
    base[:, -26:] = rotated[:, -26:]
    candidate = textured_sphere(Image.fromarray(base, "RGB"))

    verdict = evaluate_generated_bake(
        base_mesh, candidate, source_rgba=gray_photo)
    damage = verdict["metrics"]["composition_hue_damage"]
    assert damage["source_band"] is None
    assert damage["worst"] == damage["worst_raw"]
    assert any("hue damage" in reason for reason in verdict["reasons"])


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


def blob_stamped(base_texture: np.ndarray) -> trimesh.Trimesh:
    """Candidate whose back band carries a compact pale desaturated blob
    - the render-space payload of the image-in-image stamp incident and
    of the white rim-blotch class (foreign matter on a saturated
    subject). Small enough that the tone axis's low-band integral stays
    inside its budgets; only the battery may refuse it."""

    stamped = base_texture.copy()
    stamped[118:158, :16] = (232, 229, 226)
    stamped[118:158, -4:] = (232, 229, 226)
    return textured_sphere(Image.fromarray(stamped, "RGB"))


def test_added_foreign_blotch_is_rejected_by_battery() -> None:
    """The stamp/rim-blotch class: fidelity only sees it when it lands
    on photo-witnessed surface at the gate pose, so a back-side stamp
    needs its own detector. The battery's added-blotch axis must carry
    the refusal alone."""

    base_mesh = textured_sphere(red_noise_texture(11))
    try:
        photo = source_photo_of(base_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")
    candidate = blob_stamped(np.asarray(red_noise_texture(11), np.uint8))

    verdict = evaluate_generated_bake(base_mesh, candidate, source_rgba=photo)
    assert verdict["accepted"] is False
    assert any("artifact battery (pale blotch)" in reason
               for reason in verdict["reasons"])
    # the hole this closes: no other axis may carry the refusal
    assert all("artifact battery" in reason for reason in verdict["reasons"])
    battery = verdict["metrics"]["artifact_battery"]
    assert (battery["added_pale_blotch"]["worst"]
            > battery["added_pale_blotch"]["max_allowed"])


def test_inherited_blotch_is_not_punished() -> None:
    """A/B direction: the same blob in the BASELINE (candidate merely
    inherits or even removes it) must never refuse - the battery
    punishes only ADDED artifact mass, mirroring the tone axes."""

    blob_mesh = blob_stamped(np.asarray(red_noise_texture(11), np.uint8))
    try:
        photo = source_photo_of(blob_mesh)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"offscreen renderer unavailable: {exc}")

    # candidate inherits the baseline's blob untouched: nothing added,
    # nothing else changed - the whole gate must accept
    inherited = blob_stamped(np.asarray(red_noise_texture(11), np.uint8))
    verdict = evaluate_generated_bake(blob_mesh, inherited, source_rgba=photo)
    assert verdict["accepted"] is True
    assert not any("artifact battery" in reason
                   for reason in verdict["reasons"])

    # candidate REMOVES the blob: the battery must measure zero added
    # mass (the tone axis may still judge the large low-band change on
    # its own calibrated terms - that is its verdict to make, not the
    # battery's)
    clean = textured_sphere(red_noise_texture(11))
    verdict = evaluate_generated_bake(blob_mesh, clean, source_rgba=photo)
    assert not any("artifact battery" in reason
                   for reason in verdict["reasons"])
    assert verdict["metrics"]["artifact_battery"][
        "added_pale_blotch"]["worst"] == 0.0


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
