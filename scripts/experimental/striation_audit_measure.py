#!/usr/bin/env python
"""Striation audit — stage 1: reproduce the EXACT Hunyuan3D-2mv recenter for
every conditioning view of the e11/e15/e17 bust runs and emit (a) the
512x512 frames the DiT actually attended over, (b) ruler-annotated head
crops for feature-row reading, and (c) the recenter transform parameters
(bbox, scale) needed to map original rows to normalized DiT rows.

The recenter reproduced here is `hy3dshape.preprocessors.ImageProcessorV2
.recenter` at commit 82920d643c0dc2f7bfd7255f45f62d386edfe60c (the vendored
checkout under ~/.cache/abstract3d/vendor/hunyuan3d21), border_ratio=0.15,
followed by the 512x512 resize from `load_image`. Every integer truncation
is copied verbatim: mask = alpha > 0 (np.nonzero), h = x_max - x_min
(EXCLUSIVE of the last row: upstream's h is bbox span minus one pixel),
desired_size = int(size * 0.85), scale = desired_size / max(h, w),
h2 = int(h * scale), x2_min = (size - h2) // 2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path("/Users/albou/tmp/abstractframework/abstract3d")
REDO = REPO / "out" / "laurent-bust-redo"
OUT = Path("/tmp/striation_audit")
OUT.mkdir(exist_ok=True)

SETS = {
    "e11": {
        "front": "/tmp/audit_front_e11_matted.png",
        "back": str(REDO / "hunyuan_multiview/texture_reference_generated_back.png"),
        "side_left": str(REDO / "hunyuan_multiview/texture_reference_generated_side_left.png"),
        "side_right": str(REDO / "hunyuan_multiview/texture_reference_generated_side_right.png"),
    },
    "e15": {
        "front": "/tmp/laurent_front_clean.png",
        "back": str(REDO / "hunyuan_multiview/texture_reference_generated_back.png"),
        "side_left": str(REDO / "hunyuan_multiview/texture_reference_generated_side_left.png"),
        "side_right": str(REDO / "hunyuan_multiview/texture_reference_generated_side_right.png"),
    },
    "e17": {
        "front": "/tmp/laurent_front_clean4.png",
        "back": str(REDO / "loop_views_e11/loop_view_back.png"),
        "side_left": str(REDO / "loop_views_e11/loop_view_side_left.png"),
        "side_right": str(REDO / "loop_views_e11/loop_view_side_right.png"),
    },
}


def recenter_2mv(rgba: np.ndarray, border_ratio: float = 0.15, out_size: int = 512):
    """Byte-faithful ImageProcessorV2.recenter + the 512 resize.

    Returns (frame_512_rgba_uint8, params) where params carries everything
    needed to map an ORIGINAL image row y to the normalized DiT row:
        y_512 = (x2_min + (y - x_min) * scale) * (out_size / size)
        y_norm = y_512 / out_size
    """
    mask = rgba[..., 3]
    H, W = mask.shape
    size = max(H, W)
    coords = np.nonzero(mask)  # upstream: ANY nonzero alpha counts
    x_min, x_max = int(coords[0].min()), int(coords[0].max())
    y_min, y_max = int(coords[1].min()), int(coords[1].max())
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    y2_min = (size - w2) // 2
    crop = Image.fromarray(rgba[x_min:x_max, y_min:y_max])
    resized = np.asarray(crop.resize((w2, h2), Image.LANCZOS))
    result = np.zeros((size, size, 4), dtype=np.uint8)
    result[x2_min:x2_min + h2, y2_min:y2_min + w2] = resized
    # White-composite exactly like upstream (result[...,:3]*a + bg*(1-a)).
    a = result[..., 3:].astype(np.float32) / 255.0
    rgb = (result[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).clip(0, 255)
    frame = np.concatenate([rgb.astype(np.uint8), result[..., 3:]], axis=-1)
    frame_512 = np.asarray(
        Image.fromarray(frame).resize((out_size, out_size), Image.BICUBIC)
    )
    params = {
        "orig_size": [H, W],
        "bbox_rows": [x_min, x_max],
        "bbox_cols": [y_min, y_max],
        "bbox_h": h,
        "bbox_w": w,
        "square": size,
        "scale": scale,
        "x2_min": x2_min,
        "h2": h2,
        "out_size": out_size,
    }
    return frame_512, params


def orig_row_to_norm(y: float, p: dict) -> float:
    """Original-image row -> normalized row in the 512 DiT frame."""
    y_sq = p["x2_min"] + (y - p["bbox_rows"][0]) * p["scale"]
    return y_sq / p["square"]


def ruler_overlay(frame_512: np.ndarray, step: int = 16) -> Image.Image:
    im = Image.fromarray(frame_512[..., :3]).convert("RGB").resize((1024, 1024), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for y in range(0, 512, step):
        yy = y * 2
        major = (y % 64 == 0)
        color = (255, 60, 60) if major else (60, 160, 255)
        d.line([(0, yy), (1024 if major else 48, yy)], fill=color, width=1)
        if major:
            d.text((4, yy + 2), f"{y}", fill=(255, 60, 60))
    return im


def main() -> None:
    report = {}
    for set_name, views in SETS.items():
        report[set_name] = {}
        for view_name, path in views.items():
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            frame, params = recenter_2mv(rgba)
            base = f"{set_name}_{view_name}"
            Image.fromarray(frame).save(OUT / f"{base}_dit512.png")
            ruler_overlay(frame).save(OUT / f"{base}_ruler.png")
            params["source"] = path
            report[set_name][view_name] = params
            print(
                f"{set_name}/{view_name}: bbox rows {params['bbox_rows']} "
                f"(h={params['bbox_h']}px of {params['orig_size'][0]}), "
                f"scale={params['scale']:.4f}, x2_min={params['x2_min']}"
            )
    (OUT / "recenter_params.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
