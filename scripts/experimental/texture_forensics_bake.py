#!/usr/bin/env python3
"""Isolated, instrumented re-run of the e20 projection bake (forensics).

Mirrors EXACTLY how `hunyuan3d_runtime` assembles `observed_views` +
`bake_projection_texture` kwargs (orthographic, canonical_border_ratio
0.15, texture_resolution 2048, texture_completion auto), on the shipped
`geometry.glb` (bake-frame mesh, verified round-trip identity), so the
bake's per-view registration/weight internals can be captured without a
30-minute shape stage.

Instrumentation is HARNESS-SIDE ONLY (monkey-patched capture wrappers that
never modify data): production texturing.py logic runs unpatched.

Outputs into --out:
  bake_stats.json      full bake stats (arrays stripped)
  registered_view_XX_<label>.png  canonical-frame registered image per view
  capture.npz          per-view final blend weights (f16), positions (f32),
                       normals (f16), per-view painted rgba (u8)
  texture.png          baked texture
  scene.glb            textured mesh (viewer frame, same convention as bundles)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

# Label defaults mirror _TEXTURE_REFERENCE_ANGLES for the four canonical tags.
LABEL_ANGLES = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "side_left": (90.0, 0.0),
    "side_right": (-90.0, 0.0),
}


def load_reference(path: Path, label: str, azimuth: float, elevation: float) -> dict:
    """Mirror hunyuan3d_runtime reference loading (lines 2565-2597)."""
    from abstract3d.segmentation import clean_alpha_mask, remove_background_robust

    loaded = Image.open(path).convert("RGBA")
    if loaded.getchannel("A").getextrema()[0] >= 255:
        loaded = remove_background_robust(loaded)
    else:
        loaded = clean_alpha_mask(loaded)
    return {
        "rgba": loaded,
        "azimuth_deg": float(azimuth),
        "elevation_deg": float(elevation),
        "label": label,
        "role": "reference",
        "generated": True,  # e20 passed texture_reference_synthesized
        "synthesized_source": "explicit_flag",
        "origin": "caller",
    }


def strip_arrays(obj):
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items() if not isinstance(v, np.ndarray)
                and not hasattr(v, "convert")}
    if isinstance(obj, (list, tuple)):
        return [strip_arrays(v) for v in obj if not isinstance(v, np.ndarray)]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="bake-frame geometry.glb")
    parser.add_argument("--source", type=Path, required=True, help="front photo (RGBA matte)")
    parser.add_argument(
        "--ref", action="append", default=[],
        help="label=path[@azimuth[,elevation]] in bake order", metavar="SPEC")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument(
        "--no-capture", action="store_true",
        help="run the bake with ZERO instrumentation (final deliverable bakes)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import trimesh

    from abstract3d import texturing
    from abstract3d.backends import triposr_runtime
    from abstract3d.segmentation import clean_alpha_mask

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    print(f"mesh: {len(mesh.vertices)} verts {len(mesh.faces)} faces", flush=True)

    # --- observed_views assembly, mirroring hunyuan3d_runtime 3152-3166 ---
    source_image = Image.open(args.source).convert("RGBA")
    original_preview = source_image.convert("RGB")  # identity_image (pre-matte-clean RGB)
    alpha_min = source_image.getchannel("A").getextrema()[0]
    if alpha_min < 255:
        source_image = clean_alpha_mask(source_image)
    else:
        raise SystemExit("source must carry an alpha matte (runtime would rembg here)")

    observed_views = [{
        "rgba": source_image.convert("RGBA"),
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "label": "front",
        "role": "source",
        "identity_image": original_preview,
    }]
    ref_specs = []
    for spec in args.ref:
        label, _, rest = spec.partition("=")
        path_text, _, angle_text = rest.partition("@")
        if angle_text:
            if "," in angle_text:
                az_text, el_text = angle_text.split(",", 1)
                azimuth, elevation = float(az_text), float(el_text)
            else:
                azimuth, elevation = float(angle_text), 0.0
        else:
            azimuth, elevation = LABEL_ANGLES[label]
        ref_specs.append((label, Path(path_text), azimuth, elevation))
    for label, path, azimuth, elevation in ref_specs:
        observed_views.append(load_reference(path, label, azimuth, elevation))
        print(f"ref {label} az={azimuth} el={elevation} <- {path}", flush=True)

    # --- capture instrumentation (never mutates data) ---
    capture: dict = {"registered": [], "positions": None, "normals": None,
                     "blend_weights": None, "blend_rgba": None, "blend_meta": None}

    real_project = triposr_runtime._tripo_project_observed_texture

    def project_capture(observed_rgba, **kwargs):
        capture["registered"].append(observed_rgba.convert("RGBA").copy())
        return real_project(observed_rgba, **kwargs)

    real_positions = triposr_runtime._tripo_rasterize_position_atlas
    real_normals = triposr_runtime._tripo_rasterize_normal_atlas

    def positions_capture(*a, **k):
        out = real_positions(*a, **k)
        capture["positions"] = np.asarray(out, dtype=np.float32).copy()
        return out

    def normals_capture(*a, **k):
        out = real_normals(*a, **k)
        capture["normals"] = np.asarray(out, dtype=np.float16).copy()
        return out

    real_blend = texturing.blend_projections

    def blend_capture(projections, **kwargs):
        capture["blend_weights"] = [
            np.asarray(p["weight"], dtype=np.float16).copy() for p in projections]
        capture["blend_rgba"] = [
            (np.clip(np.asarray(p["rgba"], dtype=np.float32)[:, :, :3], 0, 1) * 255)
            .astype(np.uint8) for p in projections]
        capture["blend_meta"] = [
            {"label": p.get("label"), "azimuth_deg": p.get("azimuth_deg"),
             "elevation_deg": p.get("elevation_deg"), "role": p.get("role"),
             "generated": bool(p.get("generated"))} for p in projections]
        return real_blend(projections, **kwargs)

    if not args.no_capture:
        triposr_runtime._tripo_project_observed_texture = project_capture
        triposr_runtime._tripo_rasterize_position_atlas = positions_capture
        triposr_runtime._tripo_rasterize_normal_atlas = normals_capture
        texturing.blend_projections = blend_capture
    try:
        textured_mesh, stats = texturing.bake_projection_texture(
            mesh,
            observed_views=observed_views,
            texture_resolution=int(args.resolution),
            texture_completion="auto",
            projection_model="orthographic",
            canonical_border_ratio=0.15,
        )
    finally:
        triposr_runtime._tripo_project_observed_texture = real_project
        triposr_runtime._tripo_rasterize_position_atlas = real_positions
        triposr_runtime._tripo_rasterize_normal_atlas = real_normals
        texturing.blend_projections = real_blend

    # --- persist ---
    for index, image in enumerate(capture["registered"], start=1):
        label = capture["blend_meta"][index - 1]["label"] if capture["blend_meta"] else f"v{index}"
        image.save(args.out / f"registered_view_{index:02d}_{label}.png")
    stats["texture_image"].save(args.out / "texture.png")
    stats["uv_preview"].save(args.out / "uv_preview.png")
    glb = triposr_runtime._mesh_export_bytes(textured_mesh, file_type="glb", viewer_frame=True)
    (args.out / "scene.glb").write_bytes(glb)

    if not args.no_capture:
        np.savez_compressed(
            args.out / "capture.npz",
            positions=capture["positions"],
            normals=capture["normals"],
            weights=np.stack(capture["blend_weights"]),
            view_rgba=np.stack(capture["blend_rgba"]),
            meta=json.dumps(capture["blend_meta"]),
        )
    clean = strip_arrays({k: v for k, v in stats.items()
                          if k not in ("texture_image", "uv_preview", "vmapping",
                                       "indices", "uvs")})
    clean["harness_refs"] = [
        {"label": label, "path": str(path), "azimuth_deg": azimuth,
         "elevation_deg": elevation}
        for label, path, azimuth, elevation in ref_specs]
    clean["harness_source"] = str(args.source)
    (args.out / "bake_stats.json").write_text(json.dumps(clean, indent=1))
    print(json.dumps({
        "observed_coverage_ratio": clean.get("observed_coverage_ratio"),
        "view_stats": clean.get("observed_view_stats"),
        "source_pose": clean.get("source_pose"),
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
