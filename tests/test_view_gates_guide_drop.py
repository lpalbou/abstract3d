"""Guide-poisoning escape: after a matte rejection on a clay-guided
(identity-conditioning) draw, the remaining attempts for that angle drop
the clay reference and run photo-primary — measured on the e22v3 refusal:
the identity route reproduces a striated guide's surface on essentially
every draw, so redrawing with the same guide wastes the ladder."""

from __future__ import annotations

import io

import numpy as np
import pytest

import torch  # noqa: F401  (OpenMP init order on macOS)
import trimesh
from PIL import Image

from abstract3d import reference_generation as refgen


@pytest.fixture(autouse=True)
def _stub_captioner(monkeypatch):
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image", lambda image, **kw: "test object")


def sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=2, radius=0.5)


def clay_matching_generation(mesh, azimuth):
    from abstract3d.rendering import render_mesh_views

    clay = render_mesh_views(mesh, size=96, azimuths=[azimuth], elevation=0.0)[0]
    silhouette = refgen.clay_silhouette(clay)
    rgba = np.zeros((*silhouette.shape, 4), np.uint8)
    rgba[silhouette] = (180, 140, 100, 255)
    return Image.fromarray(rgba, "RGBA")


def test_matte_rejection_drops_clay_reference_for_remaining_attempts(
    monkeypatch,
) -> None:
    mesh = sphere_mesh()
    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust",
        lambda img: img.convert("RGBA"))
    image = clay_matching_generation(mesh, 90.0)
    ref_flags: list = []

    def generator(prompt, payload, **kwargs):
        ref_flags.append("reference_images" in kwargs)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    verdicts = [
        {"measured": True, "passed": False,
         "reason": "matte cleanliness: debris"},
        {"measured": True, "passed": True},
    ]
    calls: list = []

    def matte_gate(view_rgba, *, label, azimuth_deg, elevation_deg=0.0):
        calls.append(label)
        return dict(verdicts[min(len(calls) - 1, len(verdicts) - 1)])

    views, report = refgen.generate_reference_views(
        mesh,
        Image.new("RGBA", (96, 96), (120, 90, 60, 255)),
        image_generator=generator,
        angles=[("side_left", 90.0, 0.0)],
        conditioning="identity",
        max_attempts=3,
        render_size=96,
        matte_gate=matte_gate,
    )

    # attempt 1 carried the clay reference; the matte rejection dropped it
    # for attempts 2+ (photo-primary), and attempt 2's clean pass shipped.
    assert ref_flags[0] is True
    assert all(flag is False for flag in ref_flags[1:])
    assert len(views) == 1
    assert "clay_reference_dropped" in report
    entry = report["angles"][0]
    assert entry["accepted"] is True
    dropped_rows = [row for row in entry["attempts"]
                    if row.get("clay_reference_dropped")]
    assert dropped_rows, "the drop must be visible on the attempt rows"
