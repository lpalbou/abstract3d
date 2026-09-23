#!/usr/bin/env python
"""Re-measure existing ladder outputs (no i2i spend) with the current metric
and rewrite offsets/landmarks/pose in ladder_results.json in place."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT = Path("/tmp/viewgen_audit/ladder")
CLAY_PATH = Path("/tmp/viewgen_audit/e11_clays/e11_clay_side_left.png")


def main() -> None:
    from viewgen_audit_analyze import _load_mesh, estimate_azimuth, offsets
    from viewgen_audit_measure import measure

    results_file = OUT / (sys.argv[1] if len(sys.argv) > 1 else "ladder_results.json")
    rows = json.loads(results_file.read_text())
    clay_record = measure(str(CLAY_PATH), "clay", "side_left")
    mesh = _load_mesh(REPO / "out/laurent-bust-redo/e11_2mv_reg_hq")
    for row in rows:
        png = OUT / f"{row['id']}.png"
        if not png.exists():
            continue
        view_record = measure(str(png), "rgba", "side_left")
        row["offsets_vs_clay"] = offsets(view_record, clay_record, shared_frame=True)
        row["landmarks"] = {k: view_record.get(k) for k in
                            ("top", "glasses_row", "nose_tip", "nose_base",
                             "mouth", "chin", "shoulder")}
        row["nose_protrusion_px"] = view_record.get("nose_protrusion_px")
        if "pose" not in row or row.get("pose") is None:
            row["pose"] = estimate_azimuth(str(png), mesh, 90.0)
    results_file.write_text(json.dumps(rows, indent=1))
    clay_view = measure(str(CLAY_PATH), "clay", "side_left")
    print("clay nose_protrusion_px:", clay_view.get("nose_protrusion_px"))
    for row in rows:
        offs = row.get("offsets_vs_clay") or {}

        def fmt(name: str) -> str:
            value = (offs.get(name) or {}).get("registered_frac")
            return "n/a" if value is None else f"{100 * value:+.1f}%"

        print(f"{row['id']:30s} glasses {fmt('glasses_row'):>7s} ntip {fmt('nose_tip'):>7s} "
              f"nbase {fmt('nose_base'):>7s} mouth {fmt('mouth'):>7s} chin {fmt('chin'):>7s} "
              f"nose_px {row.get('nose_protrusion_px')}")


if __name__ == "__main__":
    main()
