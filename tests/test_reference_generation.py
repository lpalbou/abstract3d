"""Tests for generated reference views (synthesized unseen-angle photos).

The i2i provider is faked throughout: these tests pin the gating, tone
matching, report/provenance structure, and the wiring contracts — the
generative quality itself is validated by the proof bundles.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from abstract3d import reference_generation as refgen


@pytest.fixture(autouse=True)
def _stub_captioner(monkeypatch):
    """No test loads the real BLIP model; subject nouns come from hints or
    this stub (production caption behavior is covered by its own test)."""
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image", lambda image, **kw: "test object")


def sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=2, radius=0.5)


def solid_rgba(color, size=96, alpha=255):
    return Image.new("RGBA", (size, size), (*color, alpha))


def render_like_clay(mesh, azimuth):
    from abstract3d.rendering import render_mesh_views

    return render_mesh_views(mesh, size=96, azimuths=[azimuth], elevation=0.0)[0]


def make_fake_generator(images_by_call):
    calls = []

    def generator(prompt, image, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        payload = images_by_call[min(len(calls) - 1, len(images_by_call) - 1)]
        buffer = io.BytesIO()
        payload.save(buffer, format="PNG")
        return buffer.getvalue()

    generator.calls = calls
    return generator


def clay_matching_generation(mesh, azimuth, color=(180, 140, 100)):
    """A synthetic 'generation' whose silhouette matches the clay render."""

    clay = render_like_clay(mesh, azimuth)
    silhouette = refgen.clay_silhouette(clay)
    rgba = np.zeros((*silhouette.shape, 4), np.uint8)
    rgba[silhouette] = (*color, 255)
    return Image.fromarray(rgba, "RGBA")


def test_silhouette_iou_gate_accepts_matching_and_rejects_mismatched() -> None:
    mesh = sphere_mesh()
    clay = render_like_clay(mesh, 180.0)
    matching = clay_matching_generation(mesh, 180.0)
    assert refgen.silhouette_iou(matching, clay) > 0.95

    # a small off-center blob is a content-wrong generation
    wrong = Image.new("RGBA", clay.size, (0, 0, 0, 0))
    block = Image.new("RGBA", (20, 20), (200, 180, 160, 255))
    wrong.paste(block, (4, 4))
    assert refgen.silhouette_iou(wrong, clay) < 0.1


def test_generate_reference_views_accepts_and_reports(monkeypatch) -> None:
    mesh = sphere_mesh()
    matching = clay_matching_generation(mesh, 180.0)
    generator = make_fake_generator([matching])
    # bypass rembg: the fake generation already carries a clean alpha
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust", lambda img: img.convert("RGBA"))

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("back", 180.0, 0.0)],
        seed=7,
        render_size=96,
    )
    assert len(views) == 1
    view = views[0]
    assert view["role"] == "reference"
    assert view["generated"] is True
    assert view["azimuth_deg"] == 180.0
    assert report["accepted"] == 1
    assert report["angles"][0]["silhouette_iou"] >= 0.75
    assert report["angles"][0]["attempts"][0]["seed"] == 7
    assert "prompt" in report["angles"][0]


def test_generate_reference_views_rejects_low_iou_and_retries(monkeypatch) -> None:
    mesh = sphere_mesh()
    wrong = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    wrong.paste(Image.new("RGBA", (12, 12), (200, 180, 160, 255)), (2, 2))
    generator = make_fake_generator([wrong])
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust", lambda img: img.convert("RGBA"))

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("back", 180.0, 0.0)],
        max_attempts=2,
        render_size=96,
    )
    assert views == []
    assert report["accepted"] == 0
    assert report["rejected"] == 1
    assert len(report["angles"][0]["attempts"]) == 2  # both attempts tried
    # seeds advance between attempts so the retry is a different draw
    seeds = [a["seed"] for a in report["angles"][0]["attempts"]]
    assert seeds[0] != seeds[1]


def test_generator_errors_are_reported_not_raised(monkeypatch) -> None:
    mesh = sphere_mesh()

    def broken(prompt, image, **kwargs):
        raise RuntimeError("provider down")

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=broken,
        angles=[("back", 180.0, 0.0)],
        max_attempts=1,
        render_size=96,
    )
    assert views == []
    assert "provider down" in report["angles"][0]["attempts"][0]["error"]


def test_match_tone_lab_small_shift_reaches_source() -> None:
    source = solid_rgba((176, 140, 108))
    generated = solid_rgba((160, 128, 100))  # close: inside the cap
    matched, stats = refgen.match_tone_lab(generated, source)
    src = np.asarray(source.convert("RGB"), np.float32)
    out = np.asarray(matched.convert("RGB"), np.float32)
    assert stats["applied"] is True
    assert stats["clipped"] is False
    assert np.abs(out.mean(axis=(0, 1)) - src.mean(axis=(0, 1))).max() < 8.0


def test_match_tone_lab_caps_large_shifts() -> None:
    """A legitimately different unseen side must not be whitewashed into the
    source statistics: the mean shift is capped and the clip recorded."""

    source = solid_rgba((230, 225, 220))   # near-white front
    generated = solid_rgba((40, 45, 60))   # dark back
    matched, stats = refgen.match_tone_lab(generated, source)
    out = np.asarray(matched.convert("RGB"), np.float32)
    gen = np.asarray(generated.convert("RGB"), np.float32)
    assert stats["clipped"] is True
    # moved, but nowhere near the source: the L shift is bounded by 15
    moved = float(np.abs(out.mean(axis=(0, 1)) - gen.mean(axis=(0, 1))).max())
    assert 5.0 < moved < 80.0
    src = np.asarray(source.convert("RGB"), np.float32)
    assert float(np.abs(out.mean(axis=(0, 1)) - src.mean(axis=(0, 1))).max()) > 60.0


def test_suppress_specular_highlights_targets_pale_desaturated_blobs() -> None:
    base = np.zeros((128, 128, 4), np.uint8)
    base[:, :] = (150, 100, 60, 255)  # saturated warm body
    # a pale desaturated highlight blob
    base[40:60, 40:60, :3] = (240, 238, 235)
    image = Image.fromarray(base, "RGBA")
    out, fraction = refgen.suppress_specular_highlights(image)
    assert fraction > 0.0
    result = np.asarray(out.convert("RGB"), np.int16)
    # the blob moved toward the body color; the body itself is untouched
    assert result[50, 50, 0] < 235
    assert abs(int(result[10, 10, 0]) - 150) <= 2


def test_auto_generation_ready_gates(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_IMAGE_PROVIDER", raising=False)
    monkeypatch.delenv("ABSTRACT3D_IMAGE_MODEL", raising=False)
    ready, reason = refgen.auto_generation_ready(None)
    assert not ready and "provider" in reason

    # No subject hint required: the pipeline captions the source itself
    # (no-human-in-the-loop operation).
    monkeypatch.setenv("ABSTRACT3D_IMAGE_PROVIDER", "mlx-gen")
    ready, _reason = refgen.auto_generation_ready(None, None)
    assert ready


def test_subject_noun_extraction_bans_material_words() -> None:
    from abstract3d.captioning import extract_subject_noun

    assert extract_subject_noun("a wooden owl figurine with warm brown glaze") == "owl figurine"
    assert extract_subject_noun("a glazed ceramic owl") == "owl"
    assert extract_subject_noun("shiny metallic red sports car") == "sports car"
    assert extract_subject_noun("") == "object"
    assert extract_subject_noun(None) == "object"
    assert extract_subject_noun("golden polished marble") == "object"


def test_rebake_bundle_generates_when_single_view(tmp_path, monkeypatch) -> None:
    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")
    (bundle_dir / "metadata.json").write_text(json.dumps({"texture_resolution": 64}))

    matching = clay_matching_generation(mesh, 180.0)
    generator = make_fake_generator([matching, matching, matching])
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust", lambda img: img.convert("RGBA"))
    monkeypatch.setattr(
        "abstract3d.reference_generation.default_i2i_generator",
        lambda owner: generator,
    )

    out_dir = tmp_path / "rebake"
    _mesh, stats = bundle_api.rebake_bundle(
        bundle_dir,
        output_dir=out_dir,
        generate_references="on",
        generation_angles=[("back", 180.0, 0.0)],
        texture_resolution=64,
    )
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["generated_references"]["accepted"] == 1
    assert metadata["reference_count"] == 1
    assert (out_dir / "generated_back.png").exists()
    assert (out_dir / "generated_back_clay.png").exists()


def test_composite_conditioning_pairs_source_with_clay(monkeypatch) -> None:
    """The composite canvas must carry the SOURCE photo (left) and the clay
    render (right), and a model that echoes the full canvas back gets its
    right half taken as the generation."""

    mesh = sphere_mesh()
    seen = {}

    def echo_generator(prompt, image, **kwargs):
        seen["prompt"] = prompt
        seen["image_bytes"] = image
        # Echo the two-panel canvas back with the right panel "repainted"
        # in the source's material (the part-material gate is live in the
        # acceptance loop: an unpainted gray clay echo is now — correctly —
        # rejected as a material flip against the red source).
        canvas = Image.open(io.BytesIO(image)).convert("RGB")
        array = np.asarray(canvas).copy()
        right = array[:, canvas.width // 2:]
        clay_fg = right.std(axis=2) < 30  # clay is achromatic
        right[(right.sum(axis=2) > 90) & clay_fg] = (200, 40, 40)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()

    def fake_matte(img):
        # Key out the dark canvas background (production matting would):
        # with an all-opaque matte the material gate would rightly score
        # the background as a giant black "part" and reject.
        rgba = np.asarray(img.convert("RGBA")).copy()
        dark = rgba[:, :, :3].astype(int).sum(axis=2) < 90
        rgba[:, :, 3] = np.where(dark, 0, 255).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust", fake_matte)
    source = solid_rgba((200, 40, 40), size=96)  # distinct red source
    views, report = refgen.generate_reference_views(
        mesh, source,
        image_generator=echo_generator,
        angles=[("back", 180.0, 0.0)],
        subject_hint="a shiny red sphere ornament",
        conditioning="composite",
        render_size=96,
        silhouette_iou_min=0.0,  # the echoed clay panel passes trivially
    )
    assert report["conditioning"] == "composite"
    # HUMAN text is stripped structurally: the hint's finish word ("shiny")
    # must never reach the prompt. Color MAY appear — but only through the
    # measured pixel anchor (this source is saturated red, so the anchor
    # fires), never through the hint's own wording.
    assert report["subject_noun"] == "sphere ornament"
    assert "sphere ornament" in seen["prompt"]
    import re

    prompt_words = set(re.findall(r"[a-z]+", seen["prompt"].lower()))
    assert "shiny" not in prompt_words
    assert report["color_anchor"] == "saturated red"
    assert "The subject's main color is saturated red" in seen["prompt"]
    assert "material identity" in seen["prompt"]
    assert "Do not change any material type" in seen["prompt"]
    canvas = Image.open(io.BytesIO(seen["image_bytes"]))
    assert canvas.width == canvas.height * 2  # two panels
    left = np.asarray(canvas)[:, : canvas.width // 2]
    assert left[:, :, 0].mean() > 120  # source photo present on the left
    assert len(views) == 1
    # the accepted view is the right panel, not the full canvas
    assert views[0]["rgba"].width == views[0]["rgba"].height


def test_register_matte_to_clay_recovers_shift() -> None:
    mesh = sphere_mesh()
    clay = render_like_clay(mesh, 180.0)
    aligned = clay_matching_generation(mesh, 180.0)
    # shift the generation by 8% — below the gate without registration
    shifted = Image.new("RGBA", aligned.size, (0, 0, 0, 0))
    shifted.paste(aligned, (int(aligned.width * 0.08), 0))
    before = refgen.silhouette_iou(shifted, clay)
    registered, stats = refgen.register_matte_to_clay(shifted, clay)
    after = refgen.silhouette_iou(registered, clay)
    assert stats["applied"] is True
    assert after > before
    assert after > 0.9


def test_clay_and_projector_azimuth_conventions_agree() -> None:
    """The property that makes synthesis safe: a view generated from the
    mesh's az-A clay render projects back at exactly az A.

    Pins (a) the camera eye formula shared by the offscreen renderer and
    the bake projector, and (b) the projector painting the +Y hemisphere
    for an az +90 view (subject-left under the canonical frame, matching
    the geometry-conditioning tag convention)."""

    import math

    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position,
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    for azimuth, elevation in ((90.0, 0.0), (180.0, 0.0), (0.0, 55.0)):
        rad_az, rad_el = math.radians(azimuth), math.radians(elevation)
        renderer_eye = np.array([
            math.cos(rad_el) * math.cos(rad_az),
            math.cos(rad_el) * math.sin(rad_az),
            math.sin(rad_el),
        ]) * 3.0
        projector_eye = _tripo_camera_position(
            azimuth_deg=azimuth, elevation_deg=elevation, camera_distance=3.0)
        assert np.allclose(renderer_eye, projector_eye, atol=1e-5)

    # az +90 paints the +Y hemisphere
    mesh = sphere_mesh()
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=128, texture_padding=4)
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"], atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"], texture_resolution=128, texture_padding=4)
    positions_texture = _tripo_rasterize_position_atlas(mesh, **raster_kwargs)
    normals_texture = _tripo_rasterize_normal_atlas(mesh, **raster_kwargs)
    marker = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
    projection = _tripo_project_observed_texture(
        marker,
        mesh=mesh,
        positions_texture=positions_texture,
        normals_texture=normals_texture,
        azimuth_deg=90.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=0.8,
    )
    weight = np.asarray(projection["weight"], np.float32)
    positions = np.asarray(positions_texture, np.float32)
    painted = weight > 0.2
    assert painted.any()
    painted_y = positions[:, :, 1][painted]
    assert float(np.median(painted_y)) > 0.05


def test_rebake_bundle_auto_skips_gracefully_without_provider(tmp_path, monkeypatch) -> None:
    from abstract3d import bundle as bundle_api

    monkeypatch.delenv("ABSTRACT3D_IMAGE_PROVIDER", raising=False)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")

    out_dir = tmp_path / "rebake"
    _mesh, _stats = bundle_api.rebake_bundle(
        bundle_dir, output_dir=out_dir, generate_references="auto",
        texture_resolution=64)
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert "skipped" in metadata["generated_references"]
    assert metadata["reference_count"] == 0


def test_rebake_bundle_off_by_default(tmp_path, monkeypatch) -> None:
    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")

    called = {"n": 0}

    def spy(owner):
        called["n"] += 1
        raise AssertionError("generator must not be constructed when off")

    monkeypatch.setattr("abstract3d.reference_generation.default_i2i_generator", spy)
    out_dir = tmp_path / "rebake"
    bundle_api.rebake_bundle(bundle_dir, output_dir=out_dir, texture_resolution=64)
    assert called["n"] == 0


def test_protect_observed_texels_makes_generated_completion_only() -> None:
    """Generated weight is zeroed under credible real evidence, ramps in
    below the floor, and is untouched where nothing real sees the surface."""

    from abstract3d.texturing import protect_observed_texels

    real_weight = np.array([[0.8, 0.25], [0.1, 0.0]], np.float32)
    generated_weight = np.full((2, 2), 0.6, np.float32)
    real = {"label": "source", "weight": real_weight.copy()}
    generated = {"label": "back", "generated": True,
                 "weight": generated_weight.copy()}

    stats = protect_observed_texels([real, generated], protect_floor=0.25)

    out = np.asarray(generated["weight"], np.float32)
    assert out[0, 0] == 0.0                      # strong real evidence
    assert out[0, 1] == 0.0                      # exactly at the floor
    assert 0.0 < out[1, 0] < 0.6                 # grazing real: partial ramp
    assert out[1, 1] == pytest.approx(0.6)       # no real evidence: intact
    # real views are never modified
    assert np.array_equal(np.asarray(real["weight"]), real_weight)
    assert stats["applied"] is True
    assert stats["zeroed_by_view"]["back"] == 2

    # no generated views -> no-op
    assert protect_observed_texels([real])["applied"] is False


def test_floor_only_candidates_are_reported_not_baked(monkeypatch) -> None:
    """A candidate that only clears the gate FLOORS must not ship: the v2
    chair measured a floor-accepted view leaking stained fabric into the
    bake. The metrics stay in the report for diagnosis."""

    mesh = sphere_mesh()
    matching = clay_matching_generation(mesh, 180.0)
    generator = make_fake_generator([matching])
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    monkeypatch.setattr(
        "abstract3d.material_gates.texture_fidelity",
        lambda gen, src, **kw: {"passed": False, "floor": True,
                                "relief_ratio": 0.7, "flat_delta": 0.15,
                                "s50": 3.0, "selection_score": 0.5,
                                "reason": "floor-only stub"})

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=generator,
        angles=[("back", 180.0, 0.0)],
        max_attempts=1,
        render_size=96,
    )
    assert views == []
    assert report["accepted"] == 0
    assert report["rejected"] == 1
    entry = report["angles"][0]
    assert "floor" in entry["rejection_reason"]
    assert entry["attempts"][0]["texture"]["reason"] == "floor-only stub"


def test_person_subject_bypass_and_explicit_override(monkeypatch) -> None:
    """Unattended synthesis of a person is refused (no identity gate can
    defend a face); an explicit caller may proceed and gets the warning on
    the record."""

    mesh = sphere_mesh()
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image",
        lambda image, **kw: "a woman with long hair")
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))

    # default policy ("skip"): refused before any generator call
    def must_not_run(prompt, image, **kwargs):
        raise AssertionError("generator must not run for a person subject")

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((200, 170, 150)),
        image_generator=must_not_run,
        angles=[("back", 180.0, 0.0)],
        render_size=96,
    )
    assert views == []
    assert report["person_detected"] is True
    assert "person" in report["skipped"]

    # a non-person HINT is not trusted: the photo is captioned anyway
    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((200, 170, 150)),
        image_generator=must_not_run,
        angles=[("back", 180.0, 0.0)],
        subject_hint="figurine",
        render_size=96,
    )
    assert views == []
    assert report["person_detected"] is True

    # explicit "proceed": generation runs, warning goes on the record
    matching = clay_matching_generation(mesh, 180.0)
    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((200, 170, 150)),
        image_generator=make_fake_generator([matching]),
        angles=[("back", 180.0, 0.0)],
        person_policy="proceed",
        render_size=96,
    )
    assert report["person_detected"] is True
    assert "identity" in report["person_warning"]
    assert len(views) == 1


def test_person_gate_fails_closed_when_captioner_unavailable(monkeypatch) -> None:
    """A None caption means "person status unknown", not "not a person":
    an unavailable captioner must never become a permission grant for
    unattended synthesis of what might be someone's face."""

    mesh = sphere_mesh()
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image", lambda image, **kw: None)

    def must_not_run(prompt, image, **kwargs):
        raise AssertionError("generator must not run when the check cannot")

    # no hint: the caption was the only check, and it is unavailable
    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((150, 120, 90)),
        image_generator=must_not_run,
        angles=[("back", 180.0, 0.0)],
        render_size=96,
    )
    assert views == []
    assert "fail closed" in report["skipped"]
    assert report["person_detected"] is None

    # a non-person hint does not substitute for the photo check
    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((150, 120, 90)),
        image_generator=must_not_run,
        angles=[("back", 180.0, 0.0)],
        subject_hint="figurine",
        render_size=96,
    )
    assert views == []
    assert "fail closed" in report["skipped"]

    # explicit person acknowledgment overrides the unavailable check
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    matching = clay_matching_generation(mesh, 180.0)
    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((150, 120, 90)),
        image_generator=make_fake_generator([matching]),
        angles=[("back", 180.0, 0.0)],
        person_policy="proceed",
        render_size=96,
    )
    assert len(views) == 1


def test_is_person_subject_tokenization_and_recall() -> None:
    assert refgen.is_person_subject("a woman's face in profile")
    assert refgen.is_person_subject("a baby wearing a hat")
    assert refgen.is_person_subject("portrait of a human figure")
    assert not refgen.is_person_subject("a wooden owl figurine")
    assert not refgen.is_person_subject(None)


def test_reference_render_size_closes_letterbox_deficit() -> None:
    """Per-angle sizing (A4): the frame grows exactly enough that the
    subject's true pixels meet the canonical reference frame's demand
    (1024 * 0.85 = 870.4 px on the larger side), quantized up to 64."""

    # A subject filling 83% of its frame (the car_bo3 top-view measurement:
    # 637 px of 768) needs 870.4 / 0.8294 = 1049 -> 1088.
    mask = np.zeros((768, 768), dtype=bool)
    mask[65:702, 200:561] = True  # 637 x 361 bbox
    size, fill = refgen.reference_render_size(mask, base_size=768)
    assert fill == round(637 / 768, 4)
    assert size == 1088

    # A frame-filling subject (source-photo class) keeps the base size:
    # 870.4 / 0.9987 = 872 -> 896 exceeds base only when the cap allows;
    # with fill ~1.13x target the computed need quantizes to 896.
    full = np.zeros((768, 768), dtype=bool)
    full[1:767, 1:767] = True
    size_full, fill_full = refgen.reference_render_size(full, base_size=768)
    assert size_full == 896  # 870.4 / 0.9974 = 872.7 -> ceil to 896

    # The max cap bounds elongated subjects (tiny fill fractions).
    sliver = np.zeros((768, 768), dtype=bool)
    sliver[380:388, 200:500] = True
    size_sliver, _ = refgen.reference_render_size(sliver, base_size=768)
    assert size_sliver == 1280

    # Degenerate mask: base size, zero fill.
    empty = np.zeros((768, 768), dtype=bool)
    assert refgen.reference_render_size(empty, base_size=768) == (768, 0.0)


def test_generate_reference_views_auto_render_size_measures_each_angle(monkeypatch) -> None:
    """"auto" measures each angle's own clay silhouette and feeds the
    computed frame size through clay, guide and conditioning; an int keeps
    the historical single-size behavior byte-for-byte.

    With the CURRENT clay renderer the measured fill is ~0.85 for every
    angle — `render_mesh_views` frames adaptively per view (half extent =
    max camera-plane extent * 1.18) — so "auto" closes the same ~1.37x
    canonical-frame deficit at every angle (768 base -> 1088). The
    per-angle differentiation is the MECHANISM's contract (it measures,
    not assumes); if the renderer's framing ever changes, the sizes follow
    the measurement. Direct extent-variation coverage lives in
    `test_reference_render_size_closes_letterbox_deficit`.
    """

    mesh = sphere_mesh()
    mesh.apply_scale([1.0, 0.25, 0.25])  # prolate, car-like proportions
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))

    sizes_requested: list[int] = []
    from abstract3d import rendering as rendering_module

    real_render = rendering_module.render_mesh_views

    def spy_render(mesh_arg, *, size, **kwargs):
        sizes_requested.append(int(size))
        return real_render(mesh_arg, size=size, **kwargs)

    monkeypatch.setattr(
        "abstract3d.rendering.render_mesh_views", spy_render)

    def echo_generator(prompt, image, **kwargs):
        # echo the conditioning canvas: wider-than-tall composites are
        # cropped to the right panel, which matches the clay silhouette
        canvas = Image.open(io.BytesIO(image))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()

    views, report = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=echo_generator,
        angles=[("side_left", 90.0, 0.0), ("top", 0.0, 55.0)],
        max_attempts=1,
        render_size="auto",
        silhouette_iou_min=0.0,
        tone_match=False,
    )
    assert report["render_size_mode"] == "auto"
    for entry in report["angles"]:
        # adaptive clay framing: ~0.85 max-side fill at every angle
        assert 0.78 <= entry["subject_fill"] <= 0.92
        # deficit closure: 870.4 / 0.85 ~ 1024-1088, quantized to 64
        assert entry["render_size"] > 768
        assert entry["render_size"] % 64 == 0
        assert entry["render_size"] <= 1280
        # the per-angle clay re-render (and conditioning panel) used it
        assert entry["render_size"] in sizes_requested

    # int mode: unchanged single-size behavior, no auto fields
    views_int, report_int = refgen.generate_reference_views(
        mesh,
        solid_rgba((120, 90, 60)),
        image_generator=echo_generator,
        angles=[("side_left", 90.0, 0.0)],
        max_attempts=1,
        render_size=96,
        silhouette_iou_min=0.0,
        tone_match=False,
    )
    assert report_int["render_size_mode"] == 96
    assert report_int["angles"][0]["render_size"] == 96
    assert "subject_fill" not in report_int["angles"][0]


def test_generate_reference_views_rejects_unknown_render_size_mode() -> None:
    mesh = sphere_mesh()
    with pytest.raises(ValueError, match="render_size"):
        refgen.generate_reference_views(
            mesh,
            solid_rgba((120, 90, 60)),
            image_generator=lambda *a, **k: b"",
            angles=[("back", 180.0, 0.0)],
            render_size="huge",
        )


# --- coverage-driven adaptive angle planning --------------------------------


def test_plan_reference_angles_front_source_covers_the_complement() -> None:
    """A front source must plan the COMPLEMENT: the antipodal back first,
    only angles off the witnessed front hemisphere, and vertical (top or
    bottom band) coverage. On a featureless sphere the optimal complement
    under the bake's paint-weight law is back + elevated-rear compounds
    (they out-cover the equatorial sides against a locked front cap), so
    exact labels are asserted only where geometry forces them; the
    canonical-slot reproduction on real front-pose subjects is pinned by
    the recorded fleet table (owl/face plan back + both profiles)."""

    plan = refgen.plan_reference_angles(sphere_mesh(), (0.0, 0.0))
    selected = plan["selected"]
    assert selected[0]["label"] == "back"
    # no pick duplicates the source-facing front
    for row in selected:
        assert abs(row["azimuth_deg"]) >= 45.0 or abs(row["elevation_deg"]) >= 40.0
    assert "front" not in [row["label"] for row in selected]
    # vertical coverage: at least one elevated or depressed band pick
    assert any(abs(row["elevation_deg"]) >= 40.0 for row in selected)
    assert len(selected) <= plan["budget"]
    # gains are the greedy sequence: non-increasing, all above the floor
    gains = [row["predicted_gain"] for row in selected]
    assert gains == sorted(gains, reverse=True)
    assert all(g >= plan["min_gain"] for g in gains)
    # curve starts at the source coverage and increases with each pick
    curve = plan["coverage_curve"]
    assert curve[0] == plan["source_coverage"]
    assert all(b > a for a, b in zip(curve, curve[1:]))
    assert plan["predicted_coverage"] == curve[-1]
    # the plan must not lose to the static counterfactual it replaces
    assert plan["predicted_coverage"] >= plan["static_predicted_coverage"]


def test_plan_reference_angles_front_source_real_proportions() -> None:
    """On a subject with real-world proportions (wider than deep — the
    owl/face class), a front source plans the canonical profile slots:
    both sides witness large head-on area that the back cannot reach."""

    mesh = sphere_mesh()
    mesh.apply_scale([1.0, 0.6, 0.8])  # deep in x, narrow in y: side-heavy
    plan = refgen.plan_reference_angles(mesh, (0.0, 0.0))
    labels = [row["label"] for row in plan["selected"]]
    assert "side_left" in labels and "side_right" in labels
    assert "front" not in labels


def test_plan_reference_angles_top_source_plans_underside_drops_top() -> None:
    """The x-wing incident class: an elevated source witnesses the top, so
    the plan must select underside coverage and must NOT spend a slot on
    the (now redundant) static top."""

    plan = refgen.plan_reference_angles(sphere_mesh(), (0.0, 55.0))
    labels = [row["label"] for row in plan["selected"]]
    assert labels, "elevated source must still plan complementary views"
    # first pick is the antipode class: an underside-band angle
    assert plan["selected"][0]["elevation_deg"] <= -40.0
    assert "top" not in labels  # the incident's redundant slot
    assert any(row["elevation_deg"] <= -40.0 for row in plan["selected"])


def test_plan_reference_angles_respects_budget_and_min_gain() -> None:
    mesh = sphere_mesh()
    two = refgen.plan_reference_angles(mesh, (0.0, 0.0), budget=2)
    assert len(two["selected"]) <= 2

    # an impossible floor: nothing qualifies, the plan is legitimately empty
    nothing = refgen.plan_reference_angles(mesh, (0.0, 0.0), min_gain=1.0)
    assert nothing["angles"] == ()
    assert nothing["selected"] == []
    assert nothing["coverage_curve"] == [nothing["source_coverage"]]


def test_plan_reference_angles_deterministic_and_fast() -> None:
    mesh = sphere_mesh()
    photo = solid_rgba((150, 110, 80))
    first = refgen.plan_reference_angles(mesh, (10.0, 20.0), source_rgba=photo)
    second = refgen.plan_reference_angles(mesh, (10.0, 20.0), source_rgba=photo)
    for key in ("angles", "selected", "coverage_curve", "predicted_coverage",
                "static_predicted_coverage", "source_coverage",
                "unwitnessable_ratio"):
        assert first[key] == second[key], key


def test_plan_reference_angles_photo_lock_excludes_source_hemisphere() -> None:
    """Surface the photo witnesses is locked (the bake's
    protect_observed_texels forbids generated content there), so a
    candidate at the source pose itself must predict zero gain."""

    mesh = sphere_mesh()
    plan = refgen.plan_reference_angles(
        mesh, (0.0, 0.0), source_rgba=solid_rgba((150, 110, 80)),
        candidates=[("front", 0.0, 0.0), ("back", 180.0, 0.0)], budget=2,
        min_gain=0.0)
    by_label = {row["label"]: row for row in plan["selected"]}
    assert "back" in by_label
    assert by_label["back"]["predicted_gain"] > 0.1
    assert ("front" not in by_label
            or by_label["front"]["predicted_gain"] < 0.01)


def test_plan_reference_angles_report_carries_diagnosis_fields() -> None:
    plan = refgen.plan_reference_angles(sphere_mesh(), (0.0, 33.0))
    for key in ("source_pose", "source_coverage", "budget", "min_gain",
                "facing_min", "candidates_evaluated", "selected",
                "coverage_curve", "predicted_coverage",
                "static_predicted_coverage", "unwitnessable_ratio",
                "elapsed_s"):
        assert key in plan, key
    assert plan["source_pose"] == [0.0, 33.0]
    # a closed convex-ish mesh: nearly everything is witnessable
    assert plan["unwitnessable_ratio"] < 0.05
    # angles are bake-ready (label, azimuth, elevation) tuples
    for label, azimuth, elevation in plan["angles"]:
        assert isinstance(label, str)
        assert isinstance(azimuth, float) and isinstance(elevation, float)


def test_planner_lattice_labels_resolve_to_sensible_phrases() -> None:
    """Every lattice label must produce a human-sensible view phrase for
    the generation prompt (explicit entry or the generic fallback)."""

    for label, _azimuth, _elevation in refgen.DEFAULT_PLANNING_CANDIDATES:
        phrase = refgen._view_phrase(label)
        assert phrase.startswith("seen")
        assert "_" not in phrase
    assert "underside" in refgen._view_phrase("underside")
    assert "behind" in refgen._view_phrase("top_rear")


def test_rebake_bundle_adaptive_planning_generates_planned_angles(
        tmp_path, monkeypatch) -> None:
    """reference_angle_planning="adaptive": the generated views are exactly
    the planned angles, and the plan (with predicted gains) is persisted in
    the metadata."""

    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")
    (bundle_dir / "metadata.json").write_text(json.dumps({"texture_resolution": 64}))

    # The plan is deterministic: compute it up front to build one matching
    # generation per planned angle, in call order.
    plan = refgen.plan_reference_angles(
        mesh, (0.0, 0.0), source_rgba=solid_rgba((150, 120, 90), size=64))
    planned = list(plan["angles"])
    assert planned, "sphere front-source plan must select angles"

    def render_generation(azimuth, elevation):
        from abstract3d.rendering import render_mesh_views

        clay = render_mesh_views(
            mesh, size=96, azimuths=[azimuth], elevation=elevation)[0]
        silhouette = refgen.clay_silhouette(clay)
        rgba = np.zeros((*silhouette.shape, 4), np.uint8)
        rgba[silhouette] = (180, 140, 100, 255)
        return Image.fromarray(rgba, "RGBA")

    generator = make_fake_generator(
        [render_generation(az, el) for _label, az, el in planned])
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    monkeypatch.setattr(
        "abstract3d.reference_generation.default_i2i_generator",
        lambda owner: generator)

    out_dir = tmp_path / "rebake"
    _mesh, _stats = bundle_api.rebake_bundle(
        bundle_dir,
        output_dir=out_dir,
        generate_references="on",
        reference_angle_planning="adaptive",
        texture_resolution=64,
    )
    metadata = json.loads((out_dir / "metadata.json").read_text())
    report = metadata["generated_references"]
    assert report["angle_plan"]["angles_source"] == "adaptive"
    assert report["angle_plan"]["mode"] == "adaptive"
    generated_labels = [entry["label"] for entry in report["angles"]]
    assert generated_labels == [label for label, _az, _el in planned]
    selected = report["angle_plan"]["selected"]
    assert [row["label"] for row in selected] == generated_labels
    assert all(row["predicted_gain"] > 0 for row in selected)
    for label in generated_labels:
        assert (out_dir / f"generated_{label}.png").exists()


def test_rebake_bundle_auto_planning_stays_static_on_declared_pose(
        tmp_path, monkeypatch) -> None:
    """Default "auto" mode on a declared-pose subject keeps the static
    angle set (no fleet change) while still persisting the plan."""

    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")

    matching = clay_matching_generation(mesh, 180.0)
    generator = make_fake_generator([matching])
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    monkeypatch.setattr(
        "abstract3d.reference_generation.default_i2i_generator",
        lambda owner: generator)

    out_dir = tmp_path / "rebake"
    bundle_api.rebake_bundle(
        bundle_dir,
        output_dir=out_dir,
        generate_references="on",
        generation_angles=[("back", 180.0, 0.0)],
        texture_resolution=64,
    )
    metadata = json.loads((out_dir / "metadata.json").read_text())
    report = metadata["generated_references"]
    # explicit angles keep their historical precedence over any planning
    assert report["angle_plan"]["angles_source"] == "explicit"
    assert report["angle_plan"]["mode"] == "auto"
    # the plan itself is still recorded for diagnosis
    assert "selected" in report["angle_plan"]
    assert "predicted_coverage" in report["angle_plan"]
    assert "static_predicted_coverage" in report["angle_plan"]


def test_rebake_bundle_threads_estimated_pose_to_generation(
        tmp_path, monkeypatch) -> None:
    """The pose statement the bake uses (override here) must reach
    generation's source_pose — the historical flow hardcoded (0,0) for
    the runtime while the bake estimated, splitting the two consumers."""

    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    mesh = sphere_mesh()
    mesh.export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")

    seen = {}
    real_generate = refgen.generate_reference_views

    def spy_generate(mesh_arg, source, **kwargs):
        seen["source_pose"] = kwargs.get("source_pose")
        return real_generate(mesh_arg, source, **kwargs)

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views",
        spy_generate)
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    matching = clay_matching_generation(mesh, 180.0)
    monkeypatch.setattr(
        "abstract3d.reference_generation.default_i2i_generator",
        lambda owner: make_fake_generator([matching]))

    bundle_api.rebake_bundle(
        bundle_dir,
        output_dir=tmp_path / "rebake",
        generate_references="on",
        generation_angles=[("back", 180.0, 0.0)],
        source_pose_override=(10.0, 25.0),
        texture_resolution=64,
    )
    assert seen["source_pose"] == (10.0, 25.0)


def test_rebake_bundle_rejects_unknown_planning_mode(tmp_path) -> None:
    from abstract3d import bundle as bundle_api

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sphere_mesh().export(bundle_dir / "geometry.glb")
    solid_rgba((150, 120, 90), size=64).save(bundle_dir / "input.png")
    with pytest.raises(ValueError, match="reference_angle_planning"):
        bundle_api.rebake_bundle(
            bundle_dir, reference_angle_planning="clever",
            texture_resolution=64, write_outputs=False)
