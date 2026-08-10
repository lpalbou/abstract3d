#!/usr/bin/env python3
"""Feature-anchored row alignment of the FRONT source photo to the e20 mesh
(the viewgen-audit row law, applied to the source at bake-input time).

Measured problem (texture_forensics.md H4): the e20 mesh's face is
vertically stretched/displaced relative to the source photo's canonical
frame (glasses +39, nostril +42, mouth +36, chin +25 canonical px), and the
orthographic bake registers the source by recenter ONLY, so the photo's
mustache/lip rows land on the mesh's nose, glasses on the forehead, neck
shadow on the jaw.

Fix shape: piecewise-linear row displacement in RAW image space with FIXED
alpha-bbox endpoints (crown row 232, window cut row 975 stay put), so the
canonical recenter mapping (bbox -> 870 px frame) is bit-identical to the
production bake's and every interior feature lands on its mesh row.
Monotonicity is asserted (a fold would duplicate content). Anchors are
measured, not fitted: photo rows from the canonical-frame dark-band /
nostril profile scan, mesh rows from the dense leading-edge profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# (raw_row, displacement_raw_px): canonical-frame deltas / recenter scale
# crown + window-cut rows are alpha-bbox endpoints and MUST stay fixed.
RECENTER_SCALE = 870.0 / 916.0  # width-dominated bbox: 916 raw -> 870 canvas
ANCHORS_CANONICAL = [
    (159.0, 0.0),     # crown (bbox top -> fixed)
    (324.0, 39.5),    # glasses band center 324 -> mesh visor center 363.5
    (408.0, 42.0),    # nostril line 408 -> mesh nose base 450
    (444.0, 36.0),    # mouth center 444 -> mesh mouth 480
    (490.0, 25.0),    # chin 490 -> mesh chin 515
    (590.0, 0.0),     # collar: decay to identity
    (865.0, 0.0),     # window cut (bbox bottom -> fixed)
]


def raw_anchor_table() -> list[tuple[float, float]]:
    table = []
    for canonical_row, delta_canonical in ANCHORS_CANONICAL:
        raw_row = 603.5 + (canonical_row - 512.0) / RECENTER_SCALE
        table.append((raw_row, delta_canonical / RECENTER_SCALE))
    return table


def forward_map(height: int) -> np.ndarray:
    anchors = raw_anchor_table()
    rows = np.arange(height, dtype=np.float64)
    xs = [-1.0] + [a[0] for a in anchors] + [float(height)]
    ds = [0.0] + [a[1] for a in anchors] + [0.0]
    delta = np.interp(rows, xs, ds)
    mapped = rows + delta
    if not np.all(np.diff(mapped) > 0):
        raise SystemExit("row map is non-monotonic (would fold content)")
    return mapped


def warp_rows(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height = array.shape[0]
    mapped = forward_map(height)
    # output row j takes input row f^-1(j)
    inverse = np.interp(np.arange(height, dtype=np.float64), mapped,
                        np.arange(height, dtype=np.float64))
    low = np.clip(np.floor(inverse).astype(int), 0, height - 1)
    high = np.clip(low + 1, 0, height - 1)
    frac = (inverse - low)[:, None, None].astype(np.float32)
    out = array[low] * (1.0 - frac) + array[high] * frac
    return Image.fromarray(np.round(out).astype(np.uint8), "RGBA")


def main() -> None:
    src_path = REPO / "out/laurent-bust-redo/windowed_set_v2/win_laurent_front_clean4.png"
    out_path = Path("/tmp/texfor/source_rowlaw.png")
    image = Image.open(src_path).convert("RGBA")
    warped = warp_rows(image)
    # bbox invariance check (recenter mapping must stay identical)
    for name, im in (("orig", image), ("warped", warped)):
        alpha = np.asarray(im)[:, :, 3]
        rows = np.nonzero(alpha.max(axis=1) > 12)[0]
        cols = np.nonzero(alpha.max(axis=0) > 12)[0]
        print(name, "bbox rows", rows[0], rows[-1], "cols", cols[0], cols[-1])
    warped.save(out_path)
    print("->", out_path)

    # verification: canonical-frame feature rows after warp
    from abstract3d.segmentation import clean_alpha_mask
    from abstract3d.texturing import recenter_to_canonical_frame

    canon = np.asarray(recenter_to_canonical_frame(
        clean_alpha_mask(warped), border_ratio=0.15), dtype=np.float32)
    lum = canon[..., :3].mean(axis=2)
    amask = canon[..., 3] > 128
    cols = slice(430, 645)
    dark = ((lum[:, cols] < 60) & amask[:, cols]).sum(axis=1) / np.maximum(
        amask[:, cols].sum(axis=1), 1)
    band = [r for r in range(250, 440) if dark[r] > 0.5]
    print("warped canonical glasses band:", band[0], "-", band[-1],
          "center", (band[0] + band[-1]) / 2, "(target 363.5)")


if __name__ == "__main__":
    main()
