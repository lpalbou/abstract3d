#!/usr/bin/env python3
"""Render high-resolution clay (untextured) face close-ups for bundles.

Verification lane for the double-mouth defect: 1024px geometry-only renders
at front/three-quarter/side angles, cropped to the face band, so topology
defects are visible (the incident shipped because 256px thumbnails hid it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def load_clay_mesh(glb_path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(glb_path), process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_mesh()
    else:
        mesh = scene
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
    )
    return mesh


def render_views(glb_path: Path, out_dir: Path, label: str, size: int = 1024) -> list[Path]:
    from abstract3d.rendering import render_mesh_views

    mesh = load_clay_mesh(glb_path)
    # GLB exports are Y-up with front at +Z; stamp the persisted export-frame
    # marker so the canonical-frame renderer rotates back (mesh_ops does the
    # same for .glb inputs).
    mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    views = [("front", 0.0), ("q_left", 40.0), ("side_left", 90.0), ("q_right", 320.0)]
    out_paths = []
    rendered = render_mesh_views(
        mesh,
        size=size,
        azimuths=[az for _, az in views],
        elevation=6.0,
    )
    for (name, _), img in zip(views, rendered):
        p = out_dir / f"{label}_clay_{name}.png"
        img.save(p)
        out_paths.append(p)
    return out_paths


def face_crop(image_path: Path, out_path: Path) -> None:
    """Crop the upper 58% of the subject, centered, and upscale to 1024."""
    im = Image.open(image_path).convert("RGBA")
    a = np.asarray(im)
    if a.shape[2] == 4 and a[..., 3].min() < 250:
        mask = a[..., 3] > 10
    else:
        lum = a[..., :3].mean(axis=2)
        mask = np.abs(lum - lum[0, 0]) > 8
    ys, xs = np.where(mask)
    if ys.size == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h = y1 - y0 + 1
    crop = im.crop((x0, y0, x1 + 1, y0 + int(h * 0.58)))
    crop = crop.resize((1024, int(1024 * crop.height / crop.width)), Image.LANCZOS)
    crop.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bundle in args.bundles:
        b = Path(bundle)
        glb = b / "scene.glb"
        if not glb.exists():
            glb = b if b.suffix == ".glb" else None
        if glb is None or not glb.exists():
            print(f"skip {bundle}: no glb")
            continue
        label = b.parent.name + "_" + b.stem if b.suffix == ".glb" else b.name
        paths = render_views(glb, args.output_dir, label)
        for p in paths:
            face_crop(p, p.with_name(p.stem + "_face.png"))
        print(f"rendered {label}")


if __name__ == "__main__":
    main()
