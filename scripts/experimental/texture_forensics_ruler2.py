#!/usr/bin/env python3
"""H2 ruler: audit-grade head-band IoU azimuth estimate vs the E20 mesh.

Reuses the viewgen-audit instrument verbatim (production
register_matte_to_clay on the FULL bust + head-band-only IoU, the
combination measured to expose pose error that whole-bust IoU hides), but
pointed at the e20 bake mesh (geometry.glb, canonical frame already).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

import trimesh

from viewgen_audit_analyze import estimate_azimuth  # noqa: E402


def main() -> None:
    mesh = trimesh.load(
        str(REPO / "out/laurent-bust-redo/e20_fixed_views/geometry.glb"),
        force="mesh", process=False)
    # geometry.glb is already canonical-frame (verified round-trip identity);
    # no marker rotation needed.
    jobs = [
        ("full_side_left", "out/laurent-bust-redo/viewgen_fixed/side_left.png", 90.0),
        ("full_side_right", "out/laurent-bust-redo/viewgen_fixed/side_right.png", -90.0),
        ("full_back", "out/laurent-bust-redo/viewgen_fixed/back.png", 180.0),
        ("win_side_left", "out/laurent-bust-redo/windowed_set_v2/win_side_left.png", 90.0),
        ("win_side_right", "out/laurent-bust-redo/windowed_set_v2/win_side_right.png", -90.0),
        ("win_back", "out/laurent-bust-redo/windowed_set_v2/win_back.png", 180.0),
        ("front", "/tmp/laurent_front_clean4.png", 0.0),
    ]
    report = {}
    for name, path, nominal in jobs:
        path_full = str(REPO / path) if not path.startswith("/") else path
        row = estimate_azimuth(path_full, mesh, nominal)
        report[name] = row
        print(name, "declared", nominal, "-> measured", row["estimated_deg"],
              f"iou@est {row['iou_at_estimate']} iou@declared {row['iou_at_nominal']}",
              flush=True)
    out = Path("/tmp/texfor/ruler_e20.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
