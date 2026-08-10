#!/usr/bin/env python3
"""Full-coverage turntable renderer: the evidence pass the old evaluation lacked.

Root failure this fixes (2026-07-21 operator incident): the automated
evaluation rendered ONLY front + one RIGHT oblique, so a completely wrong
LEFT side (e10: white/silver smear over neck/jaw/cheek) and an open-mouth
mesh (e18) shipped to manual review without any metric ever seeing them.
This script renders BOTH sides and the back, clay AND textured, at two
elevations, at >= 1024 px, with face close-ups for every front-hemisphere
azimuth, and one labeled contact sheet per model so a human (or a detector)
can audit every angle without opening a 3D viewer.

Conventions:
  azimuth 0 = front, 180 = back. Sign convention VERIFIED EMPIRICALLY with a
  synthetic colored-marker mesh (red sphere at glTF +X = subject's anatomical
  LEFT; the marker is visible at az +90 and absent at az -90): positive
  azimuth = camera on the subject's anatomical LEFT. The manifest records
  this so side-specific defect reports are unambiguous. Elevations: 0 and 15.

Outputs per model under <out-root>/<model>/:
  clay_e{EE}_az{+AAA}.png / tex_e{EE}_az{+AAA}.png    full views (>=1024 px)
  face_clay_az{+AAA}.png / face_tex_az{+AAA}.png      close-ups, elevation 0,
                                                      cropped from dedicated
                                                      hi-res renders
  contact_sheet.png                                   every view, labeled
  manifest.json                                       paths + conventions

Camera framing is automatic (render_mesh_views centers on the bbox and fits
an orthographic frustum); no human input anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw

# The full angle panel. Both sides ALWAYS rendered — single-side sampling is
# the exact hole the e10 left-smear escaped through.
AZIMUTHS: Tuple[float, ...] = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0)
ELEVATIONS: Tuple[float, ...] = (0.0, 15.0)
# Front hemisphere: every azimuth with a view of the face.
FACE_CROP_AZIMUTHS: Tuple[float, ...] = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)
DEFAULT_SIZE = 1536          # full views (floor per spec: 1024)
DEFAULT_FACE_RENDER_SIZE = 2304  # dedicated renders backing the face crops
# Face crop depth: top 66% of subject rows. bust_assessment used 0.55, which
# on these busts cuts at the upper lip — a mouth-state detector cannot see a
# mouth that was cropped out (verified on e18/e20 first renders).
FACE_FRACTION = 0.66
BACKGROUND_DIFF_THRESHOLD = 14.0


# --------------------------------------------------------------- mesh loading
def load_bundle_meshes(glb_path: Path):
    """Clay + textured mesh from one GLB, stamped with the viewer-frame marker.

    Same recipe as scripts/model_quality_sweep.py: trimesh Scene -> single
    mesh (keeps TextureVisuals for the textured pass), plus a visual-stripped
    copy for clay. The 'gltf_yup_front_pz' marker makes render_mesh_views
    apply its one un-rotation; without it every render lies sideways.
    """
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


def render_batch(mesh, azimuths: Sequence[float], elevation: float, size: int) -> List[Image.Image]:
    """One renderer call for a whole azimuth ring (one GL context per call)."""
    from abstract3d.rendering import render_mesh_views

    return render_mesh_views(mesh, size=int(size), azimuths=tuple(azimuths), elevation=float(elevation))


# --------------------------------------------------------------- subject bbox
def subject_mask(img: Image.Image) -> np.ndarray:
    """Foreground via border-median background differencing (renderer-agnostic)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    return np.abs(rgb - background).max(axis=2) > BACKGROUND_DIFF_THRESHOLD


def subject_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(row0, row1, col0, col1) of the foreground; raises if empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("No subject found in render (empty foreground mask).")
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def face_crop_box(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Square-ish crop of the top FACE_FRACTION of subject rows, centered on
    the subject columns — plain bbox math, works at every azimuth because a
    bust's head always occupies the top of the silhouette."""
    row0, row1, col0, col1 = bbox
    width, height = image_size
    crop_bottom = min(row0 + int(round(FACE_FRACTION * (row1 - row0))), height)
    crop_h = max(crop_bottom - row0, 1)
    center = (col0 + col1) / 2.0
    left = max(0, int(round(center - crop_h / 2.0)))
    right = min(width, int(round(center + crop_h / 2.0)))
    return (left, row0, right, crop_bottom)


# --------------------------------------------------------------- contact sheet
def _tile(img: Image.Image, size: int, label: str) -> Image.Image:
    tile = img.convert("RGB").copy()
    tile.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 18), "#ebe7df")
    canvas.paste(tile, ((size - tile.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size, size, size + 18), fill="#1f2329")
    draw.text((4, size + 3), label, fill="#f7f6f2")
    return canvas


def build_contact_sheet(
    model: str,
    fulls: Dict[Tuple[str, float, float], Path],
    faces: Dict[Tuple[str, float], Path],
    photo_path: Path | None,
    tile_px: int = 232,
) -> Image.Image:
    n_rows = len(AZIMUTHS)
    cols = [("clay", 0.0), ("clay", 15.0), ("tex", 0.0), ("tex", 15.0)]
    grid_w = tile_px * len(cols)
    face_w = tile_px * len(FACE_CROP_AZIMUTHS)
    width = max(grid_w, face_w) + 24
    header_h = 96
    row_h = tile_px + 18
    face_block_h = 2 * row_h + 28
    sheet = Image.new("RGB", (width, header_h + n_rows * row_h + face_block_h + 36), "#ddd7ca")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width, header_h), fill="#1f2329")
    draw.text((14, 12), f"TURNTABLE  {model}", fill="#f7f6f2")
    draw.text(
        (14, 34),
        "cols: clay e0 | clay e15 | tex e0 | tex e15   rows: az 0,+30,-30,+60,-60,+90,-90,+135,-135,180",
        fill="#d7dde5",
    )
    draw.text((14, 52), "positive az = subject's anatomical LEFT (verified: synthetic +X marker test)", fill="#d7dde5")
    if photo_path is not None and photo_path.exists():
        photo = Image.open(photo_path).convert("RGB")
        photo.thumbnail((header_h - 12, header_h - 12))
        sheet.paste(photo, (width - photo.width - 10, 6))
        draw.text((width - photo.width - 10, header_h - 14), "photo", fill="#f7f6f2")
    for r, az in enumerate(AZIMUTHS):
        y = header_h + r * row_h
        for c, (mode, elev) in enumerate(cols):
            path = fulls.get((mode, elev, az))
            if path is None:
                continue
            label = f"{mode} e{int(elev):02d} az{az:+.0f}"
            sheet.paste(_tile(Image.open(path), tile_px, label), (12 + c * tile_px, y))
    y_face = header_h + n_rows * row_h + 20
    draw.text((14, y_face - 14), "face close-ups (elevation 0, front hemisphere)", fill="#1f2329")
    for c, az in enumerate(FACE_CROP_AZIMUTHS):
        for r, mode in enumerate(("clay", "tex")):
            path = faces.get((mode, az))
            if path is None:
                continue
            sheet.paste(
                _tile(Image.open(path), tile_px, f"face {mode} az{az:+.0f}"),
                (12 + c * tile_px, y_face + r * row_h),
            )
    return sheet


# --------------------------------------------------------------- per-model run
def process_model(
    bundle_dir: Path,
    out_root: Path,
    *,
    size: int = DEFAULT_SIZE,
    face_render_size: int = DEFAULT_FACE_RENDER_SIZE,
    photo: Path | None = None,
) -> Path:
    glb = bundle_dir / "scene.glb"
    if not glb.is_file():
        raise FileNotFoundError(f"No scene.glb in {bundle_dir}")
    model = bundle_dir.name
    out_dir = out_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_clay, mesh_tex = load_bundle_meshes(glb)

    fulls: Dict[Tuple[str, float, float], Path] = {}
    for mode, mesh in (("clay", mesh_clay), ("tex", mesh_tex)):
        for elev in ELEVATIONS:
            views = render_batch(mesh, AZIMUTHS, elev, size)
            for az, img in zip(AZIMUTHS, views):
                path = out_dir / f"{mode}_e{int(elev):02d}_az{az:+04.0f}.png"
                img.save(path)
                fulls[(mode, elev, az)] = path

    # Face close-ups from dedicated hi-res renders (elevation 0). The crop
    # box comes from the CLAY silhouette and is shared with the textured
    # render (same camera math over identical geometry), so clay/tex crops
    # stay pixel-aligned.
    faces: Dict[Tuple[str, float], Path] = {}
    clay_hires = render_batch(mesh_clay, FACE_CROP_AZIMUTHS, 0.0, face_render_size)
    tex_hires = render_batch(mesh_tex, FACE_CROP_AZIMUTHS, 0.0, face_render_size)
    for az, clay_img, tex_img in zip(FACE_CROP_AZIMUTHS, clay_hires, tex_hires):
        box = face_crop_box(subject_bbox(subject_mask(clay_img)), clay_img.size)
        for mode, img in (("clay", clay_img), ("tex", tex_img)):
            path = out_dir / f"face_{mode}_az{az:+04.0f}.png"
            img.crop(box).save(path)
            faces[(mode, az)] = path

    sheet = build_contact_sheet(model, fulls, faces, photo)
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    manifest = {
        "model": model,
        "glb": str(glb),
        "render_size": int(size),
        "face_render_size": int(face_render_size),
        "azimuths": list(AZIMUTHS),
        "elevations": list(ELEVATIONS),
        "face_crop_azimuths": list(FACE_CROP_AZIMUTHS),
        "azimuth_convention": (
            "0=front, 180=back; positive azimuth = camera on the subject's "
            "anatomical LEFT (glTF +X; verified with a synthetic colored-"
            "marker mesh, red at +X visible only from az +90)."
        ),
        "face_fraction": FACE_FRACTION,
        "files": {
            "full": {f"{m}/e{int(e):02d}/az{a:+.0f}": str(p) for (m, e, a), p in fulls.items()},
            "face": {f"{m}/az{a:+.0f}": str(p) for (m, a), p in faces.items()},
            "contact_sheet": str(sheet_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", nargs="+", required=True, type=Path,
                        help="Bundle dirs (each containing scene.glb).")
    parser.add_argument("--out-root", type=Path,
                        default=Path("out/laurent-bust-redo/review/turntable"))
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--face-render-size", type=int, default=DEFAULT_FACE_RENDER_SIZE)
    parser.add_argument("--photo", type=Path, default=None,
                        help="Source photo for the contact-sheet reference tile.")
    args = parser.parse_args()

    failures = 0
    for bundle in args.bundle:
        try:
            out_dir = process_model(
                bundle, args.out_root, size=args.size,
                face_render_size=args.face_render_size, photo=args.photo,
            )
            print(f"OK   {bundle.name} -> {out_dir}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {bundle.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
