#!/usr/bin/env python3
"""Worst-angle evaluation panel for bust models (red-team rebuild, 2026-07-21).

Scoring contract (operator-ordered after the e10-left/e18-mouth escape):
  - EVERY angle is evaluated on BOTH sides; a model's mesh score and texture
    score are each the MINIMUM over their per-angle scores. Never an average:
    one ruined angle is a user-visible defect, and averaging is how the old
    evaluation buried the left-side smear under six clean angles.
  - Per-detector margin: 1 - value/(2*threshold), clipped to [0,1]. Value 0
    -> 1.0, value == threshold -> 0.5, value >= 2*threshold -> 0.0. A score
    above 0.5 means every detector at that angle is under its calibrated
    threshold; at or below 0.5 means at least one fired.
  - Angle map (detectors from scripts/defect_detectors.py):
      mesh    az 0    open_mouth, double_mouth, mouth_striation, face_inflation
      mesh    az +90  profile_ripples (subject-left silhouette)
      mesh    az -90  profile_ripples (subject-right silhouette)
      texture az 0/+-30/+-60/+-90   side_smear
      texture az 0/+-30             skin_on_crown
      texture az +30                ghost_glasses (validated operating point)

Inputs: bundle dirs (scene.glb) + the turntable renders produced by
scripts/full_turntable.py. Outputs: JSON + a markdown ranking table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defect_detectors import geometry_report, texture_report  # noqa: E402


def margin(value: float, threshold: float) -> float:
    """1 at perfect, 0.5 at the calibrated threshold, 0 at 2x threshold."""
    if value is None:
        return 0.5  # unmeasurable: sit exactly on the gate, never hide it
    return max(0.0, min(1.0, 1.0 - float(value) / (2.0 * float(threshold))))


def evaluate_model(bundle: Path, turntable_root: Path) -> Dict:
    glb = bundle / "scene.glb"
    tdir = turntable_root / bundle.name
    geometry = geometry_report(glb)
    texture = texture_report(tdir, geometry["nose_row_frac"], glb)

    # ---- mesh: per-angle scores ------------------------------------------
    front_detectors = {
        "open_mouth": geometry["open_mouth"],
        "double_mouth": geometry["double_mouth"],
        "mouth_striation": geometry["mouth_striation"],
        "face_inflation": geometry["face_inflation"],
    }
    mesh_angles: Dict[str, float] = {
        "az+0": min(margin(d["value"], d["threshold"]) for d in front_detectors.values())
    }
    rip = geometry["profile_ripples"]
    for side in ("az+90", "az-90"):
        count = len(rip["per_side"].get(side, []))
        mesh_angles[side] = margin(count, rip["threshold"])

    # ---- texture: per-angle scores ---------------------------------------
    tex_angles: Dict[str, float] = {}
    for az_key, rec in texture["side_smear"].items():
        tex_angles[az_key] = margin(rec["value"], rec["threshold"])
    for az_key, rec in texture["skin_on_crown"].items():
        tex_angles[az_key] = min(tex_angles.get(az_key, 1.0), margin(rec["value"], rec["threshold"]))
    ghost = texture["ghost_glasses"]
    tex_angles["az+30"] = min(tex_angles.get("az+30", 1.0), margin(ghost["value"], ghost["threshold"]))

    mesh_worst = min(mesh_angles, key=mesh_angles.get)
    tex_worst = min(tex_angles, key=tex_angles.get)
    fired = {
        "mesh": [k for k, d in front_detectors.items() if d["fired"]] + (["profile_ripples"] if rip["fired"] else []),
        "texture": (
            [f"side_smear{k}" for k, d in texture["side_smear"].items() if d["fired"]]
            + [f"skin_on_crown{k}" for k, d in texture["skin_on_crown"].items() if d["fired"]]
            + (["ghost_glasses"] if ghost["fired"] else [])
        ),
    }
    return {
        "model": bundle.name,
        "mesh_score": round(min(mesh_angles.values()), 3),
        "mesh_worst_angle": mesh_worst,
        "mesh_angles": {k: round(v, 3) for k, v in mesh_angles.items()},
        "texture_score": round(min(tex_angles.values()), 3),
        "texture_worst_angle": tex_worst,
        "texture_angles": {k: round(v, 3) for k, v in tex_angles.items()},
        "fired": fired,
        "geometry": geometry,
        "texture": texture,
    }


def verdict(score: float) -> str:
    return "PASS" if score > 0.5 else "FAIL"


def build_table(results: List[Dict]) -> str:
    ordered = sorted(results, key=lambda r: (min(r["mesh_score"], r["texture_score"]),
                                             r["mesh_score"] + r["texture_score"]), reverse=True)
    lines = [
        "| rank | model | mesh | mesh verdict | mesh worst angle | texture | texture verdict | texture worst angle | fired detectors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(ordered, 1):
        fired = "; ".join(r["fired"]["mesh"] + r["fired"]["texture"]) or "none"
        lines.append(
            f"| {i} | {r['model']} | {r['mesh_score']:.3f} | {verdict(r['mesh_score'])} "
            f"| {r['mesh_worst_angle']} | {r['texture_score']:.3f} | {verdict(r['texture_score'])} "
            f"| {r['texture_worst_angle']} | {fired} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", nargs="+", required=True, type=Path)
    parser.add_argument("--turntable-root", type=Path,
                        default=Path("out/laurent-bust-redo/review/turntable"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for bundle in args.bundle:
        try:
            r = evaluate_model(bundle, args.turntable_root)
        except Exception as exc:
            r = {"model": bundle.name, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if "error" in r:
            print(f"{bundle.name:26s} ERROR {r['error']}")
        else:
            print(
                f"{bundle.name:26s} mesh={r['mesh_score']:.3f} ({verdict(r['mesh_score'])} @ {r['mesh_worst_angle']})  "
                f"tex={r['texture_score']:.3f} ({verdict(r['texture_score'])} @ {r['texture_worst_angle']})"
            )
    ok = [r for r in results if "error" not in r]
    table = build_table(ok)
    print()
    print(table)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=1))
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
