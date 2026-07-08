#!/usr/bin/env python3
"""Generate a high-resolution TripoSR comparison sheet across multiple MC resolutions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageDraw

from abstract3d import Scene3DManager
from abstract3d.rendering import render_mesh_views


DEFAULT_RESOLUTIONS = (128, 192, 256, 320)
DEFAULT_AZIMUTHS = (35.0, 125.0, 215.0, 305.0)
DEFAULT_ELEVATION = 20.0
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_VIEW_SIZE = 540


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TripoSR resolution study and build a large comparison sheet.")
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument("--output-dir", required=True, help="Artifact directory for the study bundle.")
    parser.add_argument("--publish-dir", default=None, help="Optional docs-facing directory to mirror final artifacts into.")
    parser.add_argument("--backend", default="triposr")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--format", default="glb")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--view-size", type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--azimuths", type=float, nargs="+", default=list(DEFAULT_AZIMUTHS))
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_metadata(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fit_square(image: Image.Image, *, size: int, background: str = "#ece8df") -> Image.Image:
    canvas = Image.new("RGB", (size, size), background)
    working = image.convert("RGB").copy()
    working.thumbnail((size - 28, size - 28))
    canvas.paste(working, ((size - working.width) // 2, (size - working.height) // 2))
    return canvas


def _annotate_tile(image: Image.Image, lines: Iterable[str]) -> Image.Image:
    tile = image.convert("RGB").copy()
    draw = ImageDraw.Draw(tile)
    text = "\n".join(str(line) for line in lines if str(line).strip())
    if not text:
        return tile
    x = 18
    y = 18
    bbox = draw.multiline_textbbox((x, y), text, spacing=4)
    draw.rounded_rectangle(
        (bbox[0] - 10, bbox[1] - 8, bbox[2] + 10, bbox[3] + 8),
        radius=12,
        fill="#1f2329",
    )
    draw.multiline_text((x, y), text, fill="#f7f6f2", spacing=4)
    return tile


def _compose_sheet(
    *,
    source_image: Image.Image,
    rows: Sequence[Mapping[str, Any]],
    tile_size: int,
    azimuths: Sequence[float],
    elevation: float,
) -> Image.Image:
    cols = 1 + len(azimuths)
    margin = 28
    gutter = 20
    header_h = 116
    row_footer_h = 64
    row_h = tile_size + row_footer_h
    width = margin * 2 + cols * tile_size + (cols - 1) * gutter
    height = margin * 2 + header_h + len(rows) * row_h + max(0, len(rows) - 1) * gutter
    sheet = Image.new("RGB", (width, height), "#ddd7ca")
    draw = ImageDraw.Draw(sheet)

    draw.rounded_rectangle((margin, margin, width - margin, margin + header_h), radius=18, fill="#1f2329")
    draw.text((margin + 20, margin + 16), "TripoSR resolution study", fill="#f7f6f2")
    draw.text(
        (margin + 20, margin + 46),
        f"Same input, same four viewpoints. Azimuths: {', '.join(str(int(v)) for v in azimuths)}. Elevation: {int(elevation)}.",
        fill="#d7dde5",
    )
    draw.text(
        (margin + 20, margin + 74),
        "Each rendered frame is labeled with marching-cubes resolution, view angle, vertices, and faces.",
        fill="#d7dde5",
    )

    source_tile_base = _fit_square(source_image, size=tile_size)
    top = margin + header_h + gutter
    for row in rows:
        row_y = top
        row_title = (
            f"mc={row['mc_resolution']} | verts={int(row['vertex_count']):,} | faces={int(row['face_count']):,}"
            f" | total={float(row['timings_s']['total']):.2f}s | mesh={float(row['timings_s']['mesh']):.2f}s"
        )
        draw.text((margin + 4, row_y - 18), row_title, fill="#1f2329")

        source_tile = _annotate_tile(
            source_tile_base,
            [
                "input",
                f"remove_bg={'yes' if row.get('background_removed') else 'no'}",
                f"chunk={row['chunk_size']}",
            ],
        )
        sheet.paste(source_tile, (margin, row_y))

        views = row["rendered_views"]
        for index, view in enumerate(views):
            tile = _fit_square(view, size=tile_size)
            tile = _annotate_tile(
                tile,
                [
                    f"mc={row['mc_resolution']} az={int(azimuths[index])}",
                    f"verts={int(row['vertex_count']):,}",
                    f"faces={int(row['face_count']):,}",
                ],
            )
            x = margin + (index + 1) * (tile_size + gutter)
            sheet.paste(tile, (x, row_y))

        footer_y = row_y + tile_size + 14
        footer = (
            f"bundle={row['bundle_dir']} | glb={row['scene_glb_path']} | "
            f"background_removed={'yes' if row.get('background_removed') else 'no'} | "
            f"inference={float(row['timings_s']['inference']):.2f}s"
        )
        draw.text((margin + 4, footer_y), footer, fill="#3e4349")
        top += row_h + gutter
    return sheet


def _write_summary_markdown(
    path: Path,
    *,
    image_path: Path,
    rows: Sequence[Mapping[str, Any]],
    azimuths: Sequence[float],
    elevation: float,
) -> None:
    lines = [
        "# TripoSR resolution study",
        "",
        f"- Source image: `{image_path}`",
        f"- View azimuths: `{', '.join(str(int(v)) for v in azimuths)}`",
        f"- View elevation: `{int(elevation)}`",
        "",
        "| MC | Vertices | Faces | Total (s) | Inference (s) | Mesh (s) | GLB |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['mc_resolution']} | {int(row['vertex_count']):,} | {int(row['face_count']):,} | "
            + f"{float(row['timings_s']['total']):.4f} | {float(row['timings_s']['inference']):.4f} | {float(row['timings_s']['mesh']):.4f} | "
            + f"`{row['scene_glb_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mirror_publish_dir(source_dir: Path, publish_dir: Path) -> None:
    publish_dir.parent.mkdir(parents=True, exist_ok=True)
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    shutil.copytree(source_dir, publish_dir)


def _run_case(
    *,
    manager: Scene3DManager,
    source_image: Image.Image,
    case_dir: Path,
    mc_resolution: int,
    device: str,
    chunk_size: int,
    output_format: str,
    remove_background: bool,
    azimuths: Sequence[float],
    elevation: float,
    view_size: int,
    resume: bool,
) -> Mapping[str, Any]:
    metadata_path = case_dir / "metadata.json"
    if resume and metadata_path.exists():
        metadata = _load_metadata(metadata_path)
    else:
        case_dir.mkdir(parents=True, exist_ok=True)
        result = manager.i23d(
            source_image.copy(),
            output_dir=str(case_dir),
            format=output_format,
            device=device,
            chunk_size=chunk_size,
            mc_resolution=mc_resolution,
            remove_background=remove_background,
        )
        metadata = dict(result.get("metadata") or _load_metadata(metadata_path))

    import trimesh

    scene_glb_path = case_dir / "scene.glb"
    mesh = trimesh.load(scene_glb_path, force="mesh")
    rendered_views = render_mesh_views(mesh, size=view_size, azimuths=azimuths, elevation=elevation)
    for index, view in enumerate(rendered_views, start=1):
        az = int(azimuths[index - 1])
        view.save(case_dir / f"highres_view_{index:02d}_az{az:03d}.png")

    payload = {
        "mc_resolution": int(metadata["mc_resolution"]),
        "chunk_size": int(metadata["chunk_size"]),
        "vertex_count": int(metadata["vertex_count"]),
        "face_count": int(metadata["face_count"]),
        "timings_s": dict(metadata["timings_s"]),
        "background_removed": bool(metadata.get("background_removed")),
        "bundle_dir": str(case_dir),
        "scene_glb_path": str(scene_glb_path),
        "scene_obj_path": str(case_dir / "scene.obj"),
        "metadata_path": str(case_dir / "metadata.json"),
        "contact_sheet_path": str(case_dir / "contact_sheet.png"),
        "rendered_views": rendered_views,
    }
    return payload


def main() -> int:
    args = _parser().parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    publish_dir = Path(args.publish_dir).expanduser().resolve() if args.publish_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image = Image.open(image_path).convert("RGBA")
    manager = Scene3DManager(backend_id=args.backend)

    rows = []
    for resolution in args.resolutions:
        case_dir = output_dir / f"mc-{int(resolution)}"
        print(f"[tripo-study] generating mc={int(resolution)} -> {case_dir}")
        row = _run_case(
            manager=manager,
            source_image=source_image,
            case_dir=case_dir,
            mc_resolution=int(resolution),
            device=str(args.device),
            chunk_size=int(args.chunk_size),
            output_format=str(args.format),
            remove_background=bool(args.remove_background),
            azimuths=tuple(float(v) for v in args.azimuths),
            elevation=float(args.elevation),
            view_size=int(args.view_size),
            resume=bool(args.resume),
        )
        rows.append(row)

    sample_view = rows[0]["rendered_views"][0]
    tile_size = int(sample_view.width)
    sheet = _compose_sheet(
        source_image=source_image.convert("RGB"),
        rows=rows,
        tile_size=tile_size,
        azimuths=tuple(float(v) for v in args.azimuths),
        elevation=float(args.elevation),
    )
    contact_sheet_path = output_dir / "contact_sheet.png"
    sheet.save(contact_sheet_path)

    serializable_rows = []
    for row in rows:
        serializable = dict(row)
        serializable.pop("rendered_views", None)
        serializable_rows.append(serializable)
    summary_json = {
        "source_image": str(image_path),
        "backend": str(args.backend),
        "device": str(args.device),
        "remove_background": bool(args.remove_background),
        "chunk_size": int(args.chunk_size),
        "azimuths": [float(v) for v in args.azimuths],
        "elevation": float(args.elevation),
        "contact_sheet_path": str(contact_sheet_path),
        "rows": serializable_rows,
    }
    summary_json_path = output_dir / "summary.json"
    _save_json(summary_json_path, summary_json)
    summary_md_path = output_dir / "summary.md"
    _write_summary_markdown(
        summary_md_path,
        image_path=image_path,
        rows=serializable_rows,
        azimuths=tuple(float(v) for v in args.azimuths),
        elevation=float(args.elevation),
    )

    if publish_dir is not None:
        _mirror_publish_dir(output_dir, publish_dir)

    print(json.dumps({"contact_sheet": str(contact_sheet_path), "summary": str(summary_md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
