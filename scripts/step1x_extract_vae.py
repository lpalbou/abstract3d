#!/usr/bin/env python3
"""Run Step1X mesh extraction/postprocess in a fresh CPU-only helper process."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--geometry-subfolder", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--octree-resolution", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--near-surface-band", type=float, required=True)
    parser.add_argument("--max-facenum", type=int, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--do-remove-floater", action="store_true")
    parser.add_argument("--do-remove-degenerate-face", action="store_true")
    parser.add_argument("--do-reduce-face", action="store_true")
    parser.add_argument("--canonicalize-export", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    source_dir = Path(args.source_dir).expanduser().resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    import torch
    from PIL import Image
    from abstract3d.backends import step1x_runtime as runtime
    from abstract3d.rendering import render_mesh_views
    from step1x3d_geometry.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder

    latents = torch.load(str(Path(args.latents).expanduser().resolve()), map_location="cpu")
    vae = MichelangeloAutoencoder.from_pretrained(
        str(Path(args.snapshot_root).expanduser().resolve()),
        subfolder=f"{args.geometry_subfolder}/vae",
    ).to("cpu", dtype=torch.float32)
    decoded = vae.decode(latents.to("cpu", dtype=torch.float32))
    meshes = vae.extract_geometry(
        decoded,
        surface_extractor_type=None,
        bounds=1.05,
        mc_level=0.0,
        octree_resolution=int(args.octree_resolution),
        num_chunks=int(args.num_chunks),
        near_surface_band=float(args.near_surface_band),
        enable_pbar=False,
    )
    mesh = meshes[0] if isinstance(meshes, list) else meshes
    raw_vertex_count = int(mesh.verts.shape[0])
    raw_face_count = int(mesh.faces.shape[0])
    del decoded
    gc.collect()

    pipeline_utils = importlib.import_module("step1x3d_geometry.models.pipelines.pipeline_utils")

    def _ensure_trimesh(mesh_obj):
        if hasattr(mesh_obj, "vertices") and hasattr(mesh_obj, "faces"):
            return mesh_obj
        if hasattr(mesh_obj, "current_mesh") and hasattr(pipeline_utils, "pymeshlab2trimesh"):
            return pipeline_utils.pymeshlab2trimesh(mesh_obj)
        return runtime._step1x_trimesh_from_extract_result(mesh_obj)

    cleanup_flags = {
        "do_remove_floater": bool(args.do_remove_floater),
        "do_remove_degenerate_face": bool(args.do_remove_degenerate_face),
        "do_reduce_face": bool(args.do_reduce_face),
    }
    postprocess_applied: list[str] = []
    postprocess_warnings: list[str] = []
    processed_obj = mesh
    try:
        if cleanup_flags["do_reduce_face"] and int(args.max_facenum) > 0 and raw_face_count > int(args.max_facenum):
            processed_obj = pipeline_utils.reduce_face(processed_obj, int(args.max_facenum))
            postprocess_applied.append(f"reduce_face:{int(args.max_facenum)}")
            processed_obj = _ensure_trimesh(processed_obj)
        if cleanup_flags["do_remove_floater"]:
            processed_obj = pipeline_utils.remove_floater(processed_obj)
            postprocess_applied.append("remove_floater")
            processed_obj = _ensure_trimesh(processed_obj)
        if cleanup_flags["do_remove_degenerate_face"]:
            processed_obj = pipeline_utils.remove_degenerate_face(processed_obj)
            postprocess_applied.append("remove_degenerate_face")
            processed_obj = _ensure_trimesh(processed_obj)
    except Exception as exc:
        postprocess_warnings.append(f"Step1X helper pipeline-utils cleanup skipped: {type(exc).__name__}: {exc}")
    processed_obj = _ensure_trimesh(processed_obj)
    del mesh
    gc.collect()
    processed_mesh, final_component_applied, final_component_warnings = runtime._step1x_prune_components(processed_obj)
    postprocess_applied.extend(final_component_applied)
    postprocess_warnings.extend(final_component_warnings)
    processed_mesh, topology_applied, topology_warnings = runtime._step1x_repair_mesh_topology(processed_mesh)
    postprocess_applied.extend(topology_applied)
    postprocess_warnings.extend(topology_warnings)
    try:
        processed_mesh = processed_mesh.smooth_shaded
        postprocess_applied.append("shade_smooth")
    except Exception as exc:
        postprocess_warnings.append(f"Step1X helper smooth shading skipped: {type(exc).__name__}: {exc}")

    export_mesh = processed_mesh
    export_axis_applied = []
    export_axis_warnings = []
    export_axis_details = {}
    if args.canonicalize_export:
        export_mesh, export_axis_applied, export_axis_warnings, export_axis_details = runtime._step1x_canonicalize_mesh_axes(
            processed_mesh,
            prompt=args.prompt,
        )

    topology = runtime._step1x_mesh_topology(export_mesh)
    views = render_mesh_views(export_mesh)
    glb_bytes = runtime._mesh_export_bytes(export_mesh, file_type="glb")
    obj_bytes = runtime._mesh_export_bytes(export_mesh, file_type="obj")

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "mesh.glb").write_bytes(glb_bytes)
    (bundle_dir / "mesh.obj").write_bytes(obj_bytes)
    for index, image in enumerate(views):
        view_path = bundle_dir / f"view_{index:02d}.png"
        image.save(view_path)

    report = {
        "raw_vertex_count": raw_vertex_count,
        "raw_face_count": raw_face_count,
        "vertex_count": int(len(export_mesh.vertices)),
        "face_count": int(len(export_mesh.faces)),
        "postprocess_applied": list(postprocess_applied),
        "postprocess_warnings": list(postprocess_warnings),
        "topology": topology,
        "export_axis_canonicalization": {
            "applied": list(export_axis_applied),
            "warnings": list(export_axis_warnings),
            **export_axis_details,
        },
        "view_count": len(views),
    }
    (bundle_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    del views
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
