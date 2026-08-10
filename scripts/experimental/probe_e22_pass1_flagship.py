#!/usr/bin/env python3
"""e22 pass-1 candidate probe: can the FLAGSHIP single-view checkpoint
(tencent/Hunyuan3D-2.1, its validated 512/50 regime) reconstruct a usable
scaffold from the front photo alone?

Context: both front-only 2mv scaffold draws measured GARBAGE (512/50:
295-component debris, photo-vs-front-clay IoU 0.415; 384/30: 20 bodies,
IoU 0.226), and the experiment record contains NO validated front-only 2mv
draw — every validated bust mesh conditioned 2mv on >= 3 views. The
strategy names the single-view flagship run the known-good floor; this
probe measures whether it clears the scaffold health bar (photo IoU 0.60).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/e22_probe3")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import importlib

    import torch
    from PIL import Image

    from abstract3d.backends import hunyuan3d_runtime as rt
    from abstract3d.backends.triposr_runtime import _mesh_export_bytes
    from abstract3d.rendering import render_mesh_views
    from abstract3d.reference_generation import (
        register_matte_to_clay,
        silhouette_iou,
    )
    from abstract3d.segmentation import clean_alpha_mask

    source = clean_alpha_mask(
        Image.open("/tmp/laurent_front_clean4.png").convert("RGBA"))

    backend = rt.Hunyuan3DShapeBackend(owner=None)
    pipeline = backend._load_runtime(
        model_id=rt._OFFICIAL_MODEL_ID, device="mps", dtype=None,
        model_subfolder=None)
    source_dir = Path(backend._last_runtime_stats["source_dir"])
    started = time.perf_counter()
    with rt._sys_path(source_dir / "hy3dshape"):
        volume_decoders = importlib.import_module(
            "hy3dshape.models.autoencoders.volume_decoders")
        del volume_decoders  # adaptive decoder is the backend default below
        pipeline.vae.volume_decoder = rt._AdaptiveVolumeDecoder()
        generator = torch.Generator(device="cpu").manual_seed(2025)
        with torch.inference_mode():
            meshes = pipeline(
                image=source,
                num_inference_steps=50,
                guidance_scale=5.0,
                octree_resolution=512,
                num_chunks=32768,
                mc_algo="mc",
                generator=generator,
                output_type="trimesh",
                enable_pbar=False,
            )
    raw = meshes[0] if isinstance(meshes, list) else meshes
    mesh0, applied, warnings = rt._hunyuan_postprocess_mesh(
        raw, max_facenum=120000)
    mesh0, _ = rt._hunyuan_canonicalize_axes(mesh0)
    elapsed = round(time.perf_counter() - started, 1)
    backend._clear_runtime()

    (OUT / "flagship_pass1_mesh.glb").write_bytes(
        _mesh_export_bytes(mesh0, file_type="glb", viewer_frame=False))
    clay = render_mesh_views(mesh0, size=768, azimuths=[0.0],
                             elevation=0.0)[0].convert("RGBA")
    clay.save(OUT / "flagship_front_clay.png")
    registered, stats = register_matte_to_clay(source, clay)
    quality = {
        "photo_vs_front_clay_iou": round(silhouette_iou(registered, clay), 4),
        "registration": stats,
        "body_count": int(mesh0.body_count),
        "vertex_count": int(len(mesh0.vertices)),
        "extents": [round(float(v), 3) for v in mesh0.extents],
        "postprocess_cleanup": list(applied),
        "seconds": elapsed,
        "model": rt._OFFICIAL_MODEL_ID,
        "regime": "512/50 seed 2025",
    }
    print(json.dumps(quality, indent=1), flush=True)
    (OUT / "flagship_quality.json").write_text(json.dumps(quality, indent=1))
    for azimuth, name in ((90.0, "side_left"), (180.0, "back")):
        render_mesh_views(mesh0, size=768, azimuths=[azimuth],
                          elevation=0.0)[0].save(OUT / f"flagship_{name}_clay.png")


if __name__ == "__main__":
    main()
