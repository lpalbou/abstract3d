#!/usr/bin/env python3
"""Iteration-loop driver: regenerate clay-guided conditioning views from an
EXISTING bundle's mesh (reconstruct -> re-render -> re-generate, the IM-3D
class loop). The output views are registered to the given mesh's clay
renders and are meant to condition the NEXT 2mv pass.

Uses the production refgen machinery (generate_reference_views) end to end:
clay render -> i2i (composite conditioning) -> register_matte_to_clay ->
silhouette/texture gates -> tone match. Nothing here forks pipeline logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="original photo")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--plan-angles",
        action="store_true",
        help=(
            "compute coverage-optimal angles with plan_reference_angles "
            "(greedy marginal quality-weighted coverage over the candidate "
            "view sphere) instead of the fixed back/left/right set"
        ),
    )
    parser.add_argument("--angle-budget", type=int, default=4)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, "src")
    import numpy as np
    import trimesh
    from PIL import Image

    from abstract3d.reference_generation import generate_reference_views
    from abstract3d.segmentation import remove_background_robust

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(str(args.bundle / "scene.glb"), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
    )
    # Exported GLBs are Y-up/front+Z; the reference machinery works in the
    # canonical frame (Z-up/front+X) — rotate back exactly like the
    # renderer's marker path does.
    mesh.apply_transform(
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    source = Image.open(args.source).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)

    if args.plan_angles:
        from abstract3d.reference_generation import plan_reference_angles

        plan = plan_reference_angles(
            mesh, (0.0, 0.0), source_rgba=source, budget=int(args.angle_budget)
        )
        angles = [tuple(row) for row in plan["angles"]]
        print(
            json.dumps(
                {
                    "planned_angles": [list(a) for a in angles],
                    "source_coverage": plan.get("source_coverage"),
                    "predicted_coverage": plan.get("predicted_coverage"),
                    "static_predicted_coverage": plan.get("static_predicted_coverage"),
                },
                indent=1,
            )
        )
    else:
        angles = [
            ("side_left", 90.0, 0.0),
            ("side_right", -90.0, 0.0),
            ("back", 180.0, 0.0),
        ]
    views, report = generate_reference_views(
        mesh,
        source,
        angles=angles,
        seed=args.seed,
        person_policy="proceed",
        source_pose=(0.0, 0.0),
    )

    saved = []
    for view in views:
        label = view["label"]
        path = args.output_dir / f"loop_view_{label}.png"
        view["rgba"].save(path)
        saved.append({"label": label, "path": str(path)})

    def scrub(obj):
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if k not in {"rgba", "image"}}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    (args.output_dir / "regen_report.json").write_text(
        json.dumps({"saved": saved, "report": scrub(report)}, indent=1)
    )
    print(json.dumps({"saved": saved, "accepted": len(views)}, indent=1))


if __name__ == "__main__":
    main()
