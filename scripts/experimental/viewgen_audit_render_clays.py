#!/usr/bin/env python
"""Render the e11-mesh clay guides for the viewgen audit (side_left,
side_right, back at 768 px) — byte-faithful to what generate_reference_views
rendered for the loop_views_e11 run: same mesh load, same canonical-frame
rotation (regen_views_from_mesh.py), same render_mesh_views call."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/albou/tmp/abstractframework/abstract3d")
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import trimesh

from abstract3d.rendering import get_last_render_backend, render_mesh_views

BUNDLE = REPO / "out/laurent-bust-redo/e11_2mv_reg_hq"
OUT = Path("/tmp/viewgen_audit/e11_clays")
ANGLES = (("side_left", 90.0), ("side_right", -90.0), ("back", 180.0))


def load_e11_mesh() -> trimesh.Trimesh:
    scene = trimesh.load(str(BUNDLE / "scene.glb"), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                           faces=np.asarray(mesh.faces), process=False)
    mesh.apply_transform(np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    return mesh


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = load_e11_mesh()
    for label, azimuth in ANGLES:
        clay = render_mesh_views(mesh, size=768, azimuths=[azimuth],
                                 elevation=0.0)[0].convert("RGBA")
        backend = get_last_render_backend()
        if backend != "moderngl":
            raise RuntimeError(f"clay renderer is {backend!r}, need moderngl")
        path = OUT / f"e11_clay_{label}.png"
        clay.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
