#!/usr/bin/env python3
"""Adversarial verifier renders: full orbit + face close-ups, clay + textured.

Independent of the project's metric scripts by design — this ONLY renders.
Judgments are made by a human/VLM reading the outputs.

Per model:
  az {0,30,...,330} x el 0, plus az {0,60,120,180,240,300} x el 20,
  clay + textured -> az{A}_el{E}_{clay|tex}.png
  face close-ups at az {0,30,330,60,300,90,270} el 0, clay + textured
  -> face_az{A}_{clay|tex}.png
  labeled contact sheets: sheet_clay_orbit / sheet_tex_orbit /
  sheet_face_tex / sheet_face_clay.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from abstract3d.rendering import render_mesh_views

ROOT = Path(__file__).resolve().parents[1] / "out" / "laurent-bust-redo"
OUT = ROOT / "review" / "verifier"

MODELS = [
    "e10_2mv_registered_refs",
    "e11_2mv_reg_hq",
    "e15_cleanfront",
    "e17_clean4",
    "e18_windowed",
    "e20_fixed_views",
    "e20_rebake_fixed",
    "e21_single_refs",
]

ORBIT_AZ = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
EL20_AZ = [0, 60, 120, 180, 240, 300]
FACE_AZ = [0, 30, 330, 60, 300, 90, 270]

ORBIT_SIZE = 1024
FACE_SIZE = 1600

BG = np.array([242, 242, 237], dtype=np.int32)  # renderer clear color ~0.95,0.95,0.93


def load_meshes(glb_path: Path):
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    mesh_tex = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh_tex.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    mesh_clay = trimesh.Trimesh(
        vertices=np.asarray(mesh_tex.vertices),
        faces=np.asarray(mesh_tex.faces),
        process=False,
    )
    mesh_clay.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    return mesh_clay, mesh_tex


def _font(size: int):
    for cand in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font(max(18, out.width // 24))
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle(
        (4, 4, 4 + bbox[2] - bbox[0] + 2 * pad, 4 + bbox[3] - bbox[1] + 2 * pad),
        fill="#1f2329",
    )
    draw.text((4 + pad, 4 + pad), text, fill="#ffe66d", font=font)
    return out


def face_crop(img: Image.Image) -> Image.Image:
    """Crop the head region: top ~52% of subject bbox, x-extent from those rows."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int32)
    mask = np.abs(arr - BG[None, None, :]).sum(axis=2) > 24
    ys, xs = np.where(mask)
    if ys.size == 0:
        return img
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    y_cut = y0 + int(h * 0.52)
    band = mask[y0:y_cut]
    bys, bxs = np.where(band)
    if bxs.size == 0:
        return img
    x0, x1 = bxs.min(), bxs.max()
    mx = int(0.05 * (x1 - x0 + 1))
    my = int(0.05 * (y_cut - y0))
    x0 = max(0, x0 - mx)
    x1 = min(arr.shape[1] - 1, x1 + mx)
    ya = max(0, y0 - my)
    yb = min(arr.shape[0] - 1, y_cut + my)
    return img.crop((int(x0), int(ya), int(x1) + 1, int(yb) + 1))


def sheet(tiles: list[tuple[str, Image.Image]], cols: int, tile_px: int, title: str) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    header = 54
    canvas = Image.new("RGB", (cols * tile_px, rows * tile_px + header), "#ddd7ca")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header), fill="#1f2329")
    draw.text((14, 12), title, fill="#f7f6f2", font=_font(30))
    for i, (name, img) in enumerate(tiles):
        t = label(img, name).copy()
        t.thumbnail((tile_px, tile_px))
        cell = Image.new("RGB", (tile_px, tile_px), "#ebe7df")
        cell.paste(t, ((tile_px - t.width) // 2, (tile_px - t.height) // 2))
        r, c = divmod(i, cols)
        canvas.paste(cell, (c * tile_px, header + r * tile_px))
    return canvas


def process(model: str) -> None:
    glb = ROOT / model / "scene.glb"
    out_dir = OUT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_clay, mesh_tex = load_meshes(glb)

    orbit: dict[str, dict[tuple[int, int], Image.Image]] = {"clay": {}, "tex": {}}
    for kind, mesh in (("clay", mesh_clay), ("tex", mesh_tex)):
        views_el0 = render_mesh_views(mesh, size=ORBIT_SIZE, azimuths=[float(a) for a in ORBIT_AZ], elevation=0.0)
        for a, img in zip(ORBIT_AZ, views_el0):
            orbit[kind][(a, 0)] = img
            img.save(out_dir / f"az{a}_el0_{kind}.png")
        views_el20 = render_mesh_views(mesh, size=ORBIT_SIZE, azimuths=[float(a) for a in EL20_AZ], elevation=20.0)
        for a, img in zip(EL20_AZ, views_el20):
            orbit[kind][(a, 20)] = img
            img.save(out_dir / f"az{a}_el20_{kind}.png")

    faces: dict[str, dict[int, Image.Image]] = {"clay": {}, "tex": {}}
    for kind, mesh in (("clay", mesh_clay), ("tex", mesh_tex)):
        views = render_mesh_views(mesh, size=FACE_SIZE, azimuths=[float(a) for a in FACE_AZ], elevation=0.0)
        for a, img in zip(FACE_AZ, views):
            crop = face_crop(img)
            faces[kind][a] = crop
            crop.save(out_dir / f"face_az{a}_{kind}.png")

    for kind in ("clay", "tex"):
        tiles = [(f"az{a} el0", orbit[kind][(a, 0)]) for a in ORBIT_AZ]
        tiles += [(f"az{a} el20", orbit[kind][(a, 20)]) for a in EL20_AZ]
        sheet(tiles, cols=6, tile_px=512, title=f"{model} — {kind} orbit").save(
            out_dir / f"sheet_{kind}_orbit.png"
        )
        ftiles = [(f"az{a}", faces[kind][a]) for a in FACE_AZ]
        sheet(ftiles, cols=4, tile_px=640, title=f"{model} — {kind} face close-ups").save(
            out_dir / f"sheet_face_{kind}.png"
        )
    print(f"done {model}", flush=True)


def main() -> None:
    targets = sys.argv[1:] or MODELS
    for model in targets:
        process(model)


if __name__ == "__main__":
    main()
