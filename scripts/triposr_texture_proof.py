#!/usr/bin/env python3
"""Build a docs-ready comparison sheet for regular vs baked TripoSR texture output."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from abstract3d.rendering import get_last_render_backend, render_mesh_views


DEFAULT_CASES = (
    "i23d owl::artifacts/validation/triposr-texture-comparison/owl",
    "i23d rocket::artifacts/validation/triposr-texture-comparison/rocket",
    "t23d-derived chair::artifacts/validation/triposr-texture-comparison/chair_t23d_source",
)
DEFAULT_AZIMUTH = 35.0
DEFAULT_ELEVATION = 20.0
DEFAULT_FULL_SIZE = 1600
DEFAULT_TILE_SIZE = 300


@dataclass(frozen=True)
class ProofCase:
    label: str
    root_dir: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--publish-dir", default=None)
    parser.add_argument("--case", action="append", default=list(DEFAULT_CASES), help="Format: label::comparison_root_dir")
    parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--full-size", type=int, default=DEFAULT_FULL_SIZE)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    return parser


def _parse_case(spec: str) -> ProofCase:
    if "::" not in spec:
        raise ValueError(f"Invalid case spec {spec!r}; expected label::root_dir.")
    label, raw_path = spec.split("::", 1)
    return ProofCase(label=label.strip(), root_dir=(ROOT / raw_path.strip()).resolve())


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Any) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def _relativize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _relativize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return _repo_relative(value)
    return value


def _fit_square(image: Image.Image, *, size: int, background: str = "#ece8df") -> Image.Image:
    canvas = Image.new("RGB", (size, size), background)
    working = image.convert("RGB").copy()
    working.thumbnail((size - 24, size - 24))
    canvas.paste(working, ((size - working.width) // 2, (size - working.height) // 2))
    return canvas


def _annotate_tile(image: Image.Image, lines: Iterable[str]) -> Image.Image:
    tile = image.convert("RGB").copy()
    text = "\n".join(str(line) for line in lines if str(line).strip())
    if not text:
        return tile
    draw = ImageDraw.Draw(tile)
    x = 16
    y = 16
    bbox = draw.multiline_textbbox((x, y), text, spacing=4)
    draw.rounded_rectangle((bbox[0] - 10, bbox[1] - 8, bbox[2] + 10, bbox[3] + 8), radius=12, fill="#1f2329")
    draw.multiline_text((x, y), text, fill="#f7f6f2", spacing=4)
    return tile


def _subject_bbox(image: Image.Image, *, threshold: int = 18) -> tuple[int, int, int, int]:
    background = np.array([242, 241, 238], dtype=np.int16)
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(rgb - background).sum(axis=2) > threshold
    ys, xs = np.where(diff)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, image.width, image.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _subject_crop(image: Image.Image, *, size: int) -> Image.Image:
    x0, y0, x1, y1 = _subject_bbox(image)
    width = x1 - x0
    height = y1 - y0
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    crop_w = max(int(width * 0.44), 160)
    crop_h = max(int(height * 0.44), 160)
    left = max(0, center_x - crop_w // 2)
    top = max(0, center_y - crop_h // 2)
    right = min(image.width, left + crop_w)
    bottom = min(image.height, top + crop_h)
    crop = image.crop((left, top, right, bottom))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def _render_outputs_are_fresh(case_dir: Path, outputs: Sequence[Path]) -> bool:
    if not outputs or not all(path.exists() for path in outputs):
        return False
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        if not metadata.get("preview_renderer"):
            return False
    dependencies = [
        case_dir / "scene.glb",
        case_dir / "metadata.json",
        case_dir / "texture.png",
    ]
    newest_dependency = max(
        dependency.stat().st_mtime for dependency in dependencies if dependency.exists()
    )
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return oldest_output >= newest_dependency


def _render_artifact(case_dir: Path, *, azimuth: float, elevation: float, full_size: int) -> tuple[Path, Path, str | None]:
    full_path = case_dir / f"proof_render_{full_size}.png"
    crop_path = case_dir / f"proof_crop_{full_size}.png"
    if _render_outputs_are_fresh(case_dir, (full_path, crop_path)):
        metadata = _load_json(case_dir / "metadata.json")
        return full_path, crop_path, metadata.get("preview_renderer")
    mesh = trimesh.load(case_dir / "scene.glb", force="mesh")
    full = render_mesh_views(mesh, size=full_size, azimuths=(azimuth,), elevation=elevation)[0]
    renderer = get_last_render_backend()
    full.save(full_path)
    _subject_crop(full, size=480).save(crop_path)
    return full_path, crop_path, renderer


def _case_row(
    case: ProofCase,
    *,
    azimuth: float,
    elevation: float,
    full_size: int,
) -> Mapping[str, Any]:
    regular_dir = case.root_dir / "vertex_color"
    improved_dir = case.root_dir / "baked_2048"
    if not regular_dir.exists():
        raise FileNotFoundError(f"Missing regular comparison dir: {regular_dir}")
    if not improved_dir.exists():
        raise FileNotFoundError(f"Missing improved comparison dir: {improved_dir}")
    regular_meta = _load_json(regular_dir / "metadata.json")
    improved_meta = _load_json(improved_dir / "metadata.json")
    regular_full, regular_crop, regular_renderer = _render_artifact(regular_dir, azimuth=azimuth, elevation=elevation, full_size=full_size)
    improved_full, improved_crop, improved_renderer = _render_artifact(improved_dir, azimuth=azimuth, elevation=elevation, full_size=full_size)
    return {
        "label": case.label,
        "source_image": Path(regular_meta["source_image_path"]),
        "regular_meta": regular_meta,
        "improved_meta": improved_meta,
        "regular_preview_renderer": regular_renderer,
        "improved_preview_renderer": improved_renderer,
        "regular_full": regular_full,
        "regular_crop": regular_crop,
        "improved_full": improved_full,
        "improved_crop": improved_crop,
    }


def _compose_sheet(rows: Sequence[Mapping[str, Any]], *, tile_size: int, azimuth: float, elevation: float) -> Image.Image:
    columns = 5
    gutter = 18
    margin = 26
    header_h = 120
    label_h = 78
    width = margin * 2 + columns * tile_size + (columns - 1) * gutter
    height = margin * 2 + header_h + len(rows) * (tile_size + label_h) + max(0, len(rows) - 1) * gutter
    sheet = Image.new("RGB", (width, height), "#ddd7ca")
    draw = ImageDraw.Draw(sheet)
    draw.rounded_rectangle((margin, margin, width - margin, margin + header_h), radius=18, fill="#1f2329")
    draw.text((margin + 20, margin + 14), "TripoSR texture upgrade proof", fill="#f7f6f2")
    draw.text(
        (margin + 20, margin + 44),
        "Regular = cleaned vertex-color export. Improved = UV-baked basecolor texture at 2048.",
        fill="#d7dde5",
    )
    draw.text(
        (margin + 20, margin + 72),
        f"Same input, same viewpoint, same geometry cleanup. View azimuth {int(azimuth)}, elevation {int(elevation)}. Right-most panels are subject crops.",
        fill="#d7dde5",
    )
    top = margin + header_h + gutter
    column_labels = ["source", "regular full", "regular crop", "improved full", "improved crop"]
    for index, label in enumerate(column_labels):
        x = margin + index * (tile_size + gutter)
        draw.text((x + 4, top - 18), label, fill="#1f2329")

    for row in rows:
        y = top
        source_tile = _annotate_tile(_fit_square(Image.open(row["source_image"]), size=tile_size), [row["label"]])
        regular_full = _annotate_tile(
            _fit_square(Image.open(row["regular_full"]), size=tile_size),
            [
                "vertex_color",
                f"verts={int(row['regular_meta']['vertex_count']):,}",
                f"faces={int(row['regular_meta']['face_count']):,}",
            ],
        )
        regular_crop = _annotate_tile(
            _fit_square(Image.open(row["regular_crop"]), size=tile_size),
            [
                "vertex_color crop",
                f"total={float(row['regular_meta']['timings_s']['total']):.2f}s",
                f"preview={row['regular_preview_renderer'] or 'unknown'}",
            ],
        )
        improved_full = _annotate_tile(
            _fit_square(Image.open(row["improved_full"]), size=tile_size),
            [
                "baked_basecolor@2048",
                f"verts={int(row['improved_meta']['vertex_count']):,}",
                f"faces={int(row['improved_meta']['face_count']):,}",
            ],
        )
        improved_crop = _annotate_tile(
            _fit_square(Image.open(row["improved_crop"]), size=tile_size),
            [
                "baked crop",
                f"texture={float(row['improved_meta']['timings_s']['texture']):.2f}s",
                f"preview={row['improved_preview_renderer'] or 'unknown'}",
            ],
        )

        tiles = [source_tile, regular_full, regular_crop, improved_full, improved_crop]
        for index, tile in enumerate(tiles):
            x = margin + index * (tile_size + gutter)
            sheet.paste(tile, (x, y))

        footer = (
            f"{row['label']} | regular output {int(row['regular_meta']['output_bytes']):,} bytes | "
            f"improved output {int(row['improved_meta']['output_bytes']):,} bytes | "
            f"material_count={int(row['improved_meta']['material_count'])} | "
            f"preview={row['improved_preview_renderer'] or row['regular_preview_renderer'] or 'unknown'}"
        )
        draw.text((margin + 4, y + tile_size + 16), footer, fill="#3e4349")
        top += tile_size + label_h + gutter
    return sheet


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]], *, azimuth: float, elevation: float) -> None:
    lines = [
        "# TripoSR texture upgrade proof",
        "",
        f"- View azimuth: `{azimuth}`",
        f"- View elevation: `{elevation}`",
        "",
        "| Case | Source | Regular mode | Improved mode | Preview renderer | Regular total (s) | Improved total (s) | Regular bytes | Improved bytes |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        regular_meta = row["regular_meta"]
        improved_meta = row["improved_meta"]
        lines.append(
            "| "
            + f"{row['label']} | `{_repo_relative(row['source_image'])}` | "
            + f"`{regular_meta['texture_mode']}` | `{improved_meta['texture_mode']}@{improved_meta['texture_resolution']}` | "
            + f"`{row['improved_preview_renderer'] or row['regular_preview_renderer'] or 'unknown'}` | "
            + f"{float(regular_meta['timings_s']['total']):.4f} | {float(improved_meta['timings_s']['total']):.4f} | "
            + f"{int(regular_meta['output_bytes']):,} | {int(improved_meta['output_bytes']):,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mirror_publish_dir(source_dir: Path, publish_dir: Path) -> None:
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, publish_dir)


def main() -> int:
    args = _parser().parse_args()
    cases = [_parse_case(spec) for spec in args.case]
    rows = [_case_row(case, azimuth=args.azimuth, elevation=args.elevation, full_size=args.full_size) for case in cases]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = _compose_sheet(rows, tile_size=args.tile_size, azimuth=args.azimuth, elevation=args.elevation)
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    summary_json = {
        "azimuth": args.azimuth,
        "elevation": args.elevation,
        "rows": [
            {
                "label": row["label"],
                "source_image": _repo_relative(row["source_image"]),
                "regular_preview_renderer": row["regular_preview_renderer"],
                "improved_preview_renderer": row["improved_preview_renderer"],
                "regular": _relativize_paths(row["regular_meta"]),
                "improved": _relativize_paths(row["improved_meta"]),
            }
            for row in rows
        ],
        "contact_sheet": "contact_sheet.png",
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    summary_md_path = output_dir / "summary.md"
    _write_summary(summary_md_path, rows, azimuth=args.azimuth, elevation=args.elevation)

    if args.publish_dir:
        _mirror_publish_dir(output_dir, Path(args.publish_dir).resolve())

    print(json.dumps({"contact_sheet": str(sheet_path), "summary": str(summary_md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
