#!/usr/bin/env python3
"""Re-bake a bundle mesh's texture from a chosen view set (cycle-2 lane).

Split-consumer law (measured on e18): the WINDOWED views that fix 2mv
geometry conditioning (common anatomical span -> no striated features)
STARVE the texture bake below the window cut. Geometry wants same-window
views; texture wants full-span views. This script re-bakes an existing
mesh with full-span views, leaving the (good) geometry untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "abstract3d" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="bundle with geometry.glb")
    parser.add_argument("--front", required=True, type=Path, help="full-span front photo (RGBA ok)")
    parser.add_argument("--view", action="append", nargs=2, metavar=("ANGLE", "PATH"),
                        required=True, help="e.g. --view back path.png (repeatable)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--texture-completion", default="auto")
    args = parser.parse_args()

    import numpy as np
    import trimesh
    from PIL import Image

    from abstract3d.segmentation import clean_alpha_mask, remove_background_robust
    from abstract3d.texturing import bake_projection_texture

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ANGLES = {
        "back": (180.0, 0.0),
        "side_left": (90.0, 0.0),
        "side_right": (-90.0, 0.0),
        "top": (0.0, 55.0),
    }

    glb = args.bundle / "geometry.glb"
    if not glb.exists():
        glb = args.bundle / "scene.glb"
    scene = trimesh.load(str(glb), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
    )
    # Exported GLBs are Y-up/front+Z; the bake works canonical Z-up/front+X.
    mesh.apply_transform(
        np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    )

    def load_rgba(path: Path) -> Image.Image:
        im = Image.open(path).convert("RGBA")
        if im.getchannel("A").getextrema()[0] >= 255:
            im = remove_background_robust(im)
        else:
            im = clean_alpha_mask(im)
        return im

    original = Image.open(args.front).convert("RGB")
    observed = [{
        "rgba": load_rgba(args.front),
        "azimuth_deg": 0.0, "elevation_deg": 0.0,
        "label": "front", "role": "source",
        "identity_image": original,
    }]
    for angle, path in args.view:
        if angle not in ANGLES:
            raise SystemExit(f"unknown angle {angle!r}; known: {sorted(ANGLES)}")
        az, el = ANGLES[angle]
        observed.append({
            "rgba": load_rgba(Path(path)),
            "azimuth_deg": az, "elevation_deg": el,
            "label": angle, "role": "reference",
            "generated": True,  # synthesized: protected + subordinated
            "origin": "caller",
        })

    textured, stats = bake_projection_texture(
        mesh,
        observed_views=observed,
        texture_resolution=2048,
        texture_completion=args.texture_completion,
        projection_model="orthographic",
        canonical_border_ratio=0.15,
    )

    # Back to glTF viewer frame for export.
    textured.apply_transform(
        np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    )
    out_glb = args.output_dir / "scene.glb"
    textured.export(str(out_glb))
    safe_stats = {}
    for key, value in (stats or {}).items():
        try:
            json.dumps(value)
            safe_stats[key] = value
        except TypeError:
            safe_stats[key] = str(value)[:400]
    (args.output_dir / "rebake_stats.json").write_text(json.dumps(safe_stats, indent=1))
    print("baked:", out_glb)
    print("observed_coverage:", safe_stats.get("observed_coverage_ratio"))


if __name__ == "__main__":
    main()
