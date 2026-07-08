#!/usr/bin/env python3
"""Compose the cross-backend four-object proof sheet.

Reads the per-case bundles produced under `artifacts/validation/final-proof/`
(two backends x four checked objects) and writes:

- `summary/comparison_sheet.png`: one row per object, one column per backend,
  using each bundle's contact sheet
- `summary/summary.json`: per-case metrics pulled from bundle metadata

Usage:
    python scripts/final_proof_sheet.py \
        --proof-dir artifacts/validation/final-proof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

OBJECTS = ("owl", "chair", "starship", "face")
BACKENDS = ("triposr", "hunyuan")


def _load_case(proof_dir: Path, backend: str, obj: str) -> Optional[Dict]:
    case_dir = proof_dir / f"{backend}-{obj}"
    metadata_path = case_dir / "metadata.json"
    sheet_path = case_dir / "contact_sheet.png"
    if not metadata_path.exists() or not sheet_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    texture = metadata.get("texture_artifacts") or {}
    return {
        "backend": backend,
        "object": obj,
        "sheet_path": str(sheet_path),
        "vertex_count": metadata.get("vertex_count"),
        "face_count": metadata.get("face_count"),
        "watertight": (metadata.get("topology") or {}).get("is_watertight"),
        "body_count": (metadata.get("topology") or {}).get("body_count"),
        "total_s": (metadata.get("timings_s") or {}).get("total"),
        "observed_coverage_ratio": texture.get("observed_coverage_ratio"),
        "projection_mode": texture.get("projection_mode"),
        "source_pose": texture.get("source_pose"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", default="artifacts/validation/final-proof")
    args = parser.parse_args()

    proof_dir = Path(args.proof_dir).expanduser().resolve()
    summary_dir = proof_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    tile_width, label_height = 960, 42
    tiles: Dict[tuple, Image.Image] = {}
    for obj in OBJECTS:
        for backend in BACKENDS:
            case = _load_case(proof_dir, backend, obj)
            if case is None:
                continue
            rows.append(case)
            sheet = Image.open(case["sheet_path"]).convert("RGB")
            scale = tile_width / sheet.width
            sheet = sheet.resize((tile_width, int(sheet.height * scale)), Image.LANCZOS)
            tiles[(obj, backend)] = sheet

    if not tiles:
        print("No proof bundles found; nothing to compose.")
        return 1

    tile_height = max(image.height for image in tiles.values())
    grid_width = tile_width * len(BACKENDS)
    grid_height = (tile_height + label_height) * len(OBJECTS) + label_height
    canvas = Image.new("RGB", (grid_width, grid_height), "#16181d")
    draw = ImageDraw.Draw(canvas)
    for column, backend in enumerate(BACKENDS):
        draw.text((column * tile_width + 16, 12), backend, fill="#f0ede4")
    for row_index, obj in enumerate(OBJECTS):
        y = label_height + row_index * (tile_height + label_height)
        draw.text((16, y + 10), obj, fill="#f0ede4")
        for column, backend in enumerate(BACKENDS):
            image = tiles.get((obj, backend))
            if image is not None:
                canvas.paste(image, (column * tile_width, y + label_height))

    sheet_path = summary_dir / "comparison_sheet.png"
    canvas.save(sheet_path)
    (summary_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {sheet_path} and summary.json covering {len(rows)} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
