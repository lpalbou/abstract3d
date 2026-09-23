#!/usr/bin/env python
"""Viewgen audit — stage 2: azimuth pose ruler + feature-offset tables.

For every generated view of sets A and B:
  * estimate the EFFECTIVE HEAD AZIMUTH by sweeping the set's own mesh
    through candidate azimuths, registering the generated matte to each
    clay (production `register_matte_to_clay`), and scoring HEAD-BAND
    silhouette IoU (rows above the clay's shoulder flare; full-bust IoU is
    torso-dominated and pose-blind — measured: a 55 deg view scores 0.76
    full-bust IoU against a 90 deg clay, comfortably over the 0.75 gate);
  * compute per-feature row offsets vs the view's own clay guide (i2i
    infidelity) and vs the front photo (set inconsistency), in fractions
    of head height (top-of-head -> chin) and of subject bbox height.

Outputs /tmp/viewgen_audit/analysis.json + markdown tables on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT = Path("/tmp/viewgen_audit")
REDO = REPO / "out/laurent-bust-redo"

FEATURES = ("glasses_row", "nose_tip", "nose_base", "mouth", "chin")


def _load_mesh(bundle: Path):
    import trimesh

    scene = trimesh.load(str(bundle / "scene.glb"), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                           faces=np.asarray(mesh.faces), process=False)
    mesh.apply_transform(np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    return mesh


def head_band_iou(generated_path: str, mesh, azimuth: float) -> float:
    """Register the generated matte to the clay at `azimuth`, IoU over the
    clay's head rows only."""
    from PIL import Image

    from abstract3d.reference_generation import (
        clay_silhouette,
        register_matte_to_clay,
    )
    from abstract3d.rendering import render_mesh_views
    from viewgen_audit_measure import head_geometry

    clay = render_mesh_views(mesh, size=512, azimuths=[float(azimuth)],
                             elevation=0.0)[0].convert("RGBA")
    clay_mask = np.asarray(clay_silhouette(clay))
    geo = head_geometry(clay_mask)
    generated = Image.open(generated_path)
    registered, _stats = register_matte_to_clay(generated, clay)
    gen_mask = np.asarray(registered)[:, :, 3] > 128
    band = slice(geo["top"], geo["shoulder"])
    inter = int((gen_mask[band] & clay_mask[band]).sum())
    union = int((gen_mask[band] | clay_mask[band]).sum())
    return inter / union if union else 0.0


def estimate_azimuth(generated_path: str, mesh, nominal: float) -> Dict[str, Any]:
    """Sweep +/-55 deg around the nominal azimuth (7.5 deg steps, then a
    2.5 deg refine) and return the head-band IoU argmax."""
    sign = 1.0 if nominal >= 0 else -1.0
    if abs(nominal) == 180.0:
        candidates = [180.0 + d for d in range(-50, 55, 10)]
    else:
        candidates = [sign * a for a in np.arange(abs(nominal) - 55, abs(nominal) + 20, 7.5)]
    candidates.append(float(nominal))  # the nominal must be scored, never defaulted
    scores = {float(a): head_band_iou(generated_path, mesh, float(a)) for a in candidates}
    best = max(scores, key=scores.get)
    fine = [best - 5.0, best - 2.5, best + 2.5, best + 5.0]
    for a in fine:
        scores[float(a)] = head_band_iou(generated_path, mesh, float(a))
    best = max(scores, key=scores.get)
    return {
        "nominal_deg": nominal,
        "estimated_deg": best,
        "iou_at_estimate": round(scores[best], 4),
        "iou_at_nominal": round(scores.get(float(nominal), 0.0), 4),
        "sweep": {str(k): round(v, 4) for k, v in sorted(scores.items())},
    }


def offsets(view: Dict[str, Any], reference: Dict[str, Any],
            shared_frame: bool = False) -> Dict[str, Any]:
    """Per-feature (view - reference) offsets.

    `of_head` / `of_subject`: offsets of the normalized fractions (crop-
    invariant; right axis against the differently-cropped front photo).
    NOTE `of_head` is identically 0 for `chin` (chin defines the head
    normalization), so when both images share one registered frame
    (`shared_frame=True`, view registered onto its clay guide) the
    additional `registered_px` / `registered_frac` axes report the RAW row
    difference normalized by the reference head height — the axis that
    catches whole-face shifts and stretches the fraction axes cancel out.
    """
    out: Dict[str, Any] = {}
    ref_head = reference.get("head_height_px")
    for name in FEATURES:
        vf = view["fractions"].get(name, {})
        rf = reference["fractions"].get(name, {})
        row: Dict[str, Optional[float]] = {}
        for axis in ("of_head", "of_subject"):
            a, b = vf.get(axis), rf.get(axis)
            row[axis] = round(a - b, 4) if (a is not None and b is not None) else None
        if shared_frame:
            a, b = view.get(name), reference.get(name)
            if a is not None and b is not None and ref_head:
                row["registered_px"] = int(a - b)
                row["registered_frac"] = round((a - b) / ref_head, 4)
            else:
                row["registered_px"] = None
                row["registered_frac"] = None
        out[name] = row
    return out


def main() -> None:
    from viewgen_audit_measure import measure

    sets = {
        "A": {
            "mesh": REDO / "hunyuan_multiview",
            "views": {
                "side_left": REDO / "hunyuan_multiview/texture_reference_generated_side_left.png",
                "side_right": REDO / "hunyuan_multiview/texture_reference_generated_side_right.png",
                "back": REDO / "hunyuan_multiview/texture_reference_generated_back.png",
            },
            "clays": {
                "side_left": REDO / "hunyuan_multiview/texture_reference_generated_side_left_clay.png",
                "side_right": REDO / "hunyuan_multiview/texture_reference_generated_side_right_clay.png",
                "back": REDO / "hunyuan_multiview/texture_reference_generated_back_clay.png",
            },
        },
        "B": {
            "mesh": REDO / "e11_2mv_reg_hq",
            "views": {
                "side_left": REDO / "loop_views_e11/loop_view_side_left.png",
                "side_right": REDO / "loop_views_e11/loop_view_side_right.png",
                "back": REDO / "loop_views_e11/loop_view_back.png",
            },
            "clays": {
                "side_left": OUT / "e11_clays/e11_clay_side_left.png",
                "side_right": OUT / "e11_clays/e11_clay_side_right.png",
                "back": OUT / "e11_clays/e11_clay_back.png",
            },
        },
    }
    nominal = {"side_left": 90.0, "side_right": -90.0, "back": 180.0}

    front = measure("/tmp/laurent_bust_crop.png", "photo", "front")
    analysis: Dict[str, Any] = {"front": front, "sets": {}}
    for set_name, config in sets.items():
        mesh = _load_mesh(config["mesh"])
        rows: List[Dict[str, Any]] = []
        for label, view_path in config["views"].items():
            view = measure(str(view_path), "rgba", label)
            clay = measure(str(config["clays"][label]), "clay", label)
            pose = estimate_azimuth(str(view_path), mesh, nominal[label])
            rows.append({
                "label": label,
                "view": view,
                "clay": clay,
                "pose": pose,
                "offset_vs_clay": offsets(view, clay, shared_frame=True),
                "offset_vs_front": offsets(view, front),
            })
            print(f"[{set_name}/{label}] azimuth nominal {pose['nominal_deg']} -> "
                  f"estimated {pose['estimated_deg']} "
                  f"(IoU {pose['iou_at_estimate']} vs {pose['iou_at_nominal']} at nominal)",
                  flush=True)
        analysis["sets"][set_name] = rows
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # Markdown tables.
    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{100.0 * value:+.1f}%"

    for set_name, rows in analysis["sets"].items():
        print(f"\n### Set {set_name}: feature-row offsets")
        print("| view | ref (axis) | glasses | nose_tip | nose_base | mouth | chin | est. azimuth (nominal) |")
        print("|---|---|---|---|---|---|---|---|")
        for row in rows:
            pose = row["pose"]
            offs = row["offset_vs_clay"]
            cells = [fmt(offs[f].get("registered_frac")) for f in FEATURES]
            az = f"{pose['estimated_deg']:+.1f} ({pose['nominal_deg']:+.0f})"
            print(f"| {row['label']} | clay guide (registered rows / clay head height) | "
                  + " | ".join(cells) + f" | {az} |")
            offs = row["offset_vs_front"]
            cells = [fmt(offs[f]["of_head"]) for f in FEATURES]
            print(f"| {row['label']} | front photo (fraction of own head height) | "
                  + " | ".join(cells) + " |  |")
    print("\nfront photo fractions (of head):",
          {k: front["fractions"][k]["of_head"] for k in FEATURES})


if __name__ == "__main__":
    main()
