#!/usr/bin/env python3
"""Calibration runner: full measure matrix over the 8-model validation set.

Prints one row per (model, measure) with per-angle values so thresholds can
be placed inside the operator-BAD vs operator-OK gap. Writes JSON to
/tmp/eval_calibration/matrix.json and per-model projection debug images.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from abstract3d.model_evaluation import (  # noqa: E402
    EvaluationConfig,
    PhotoReference,
    evaluate_model,
)

MODELS = [
    "e10_2mv_registered_refs",
    "e11_2mv_reg_hq",
    "e15_cleanfront",
    "e17_clean4",
    "e18_windowed",
    "e20_fixed_views",
    "e20_rebake_fixed",
    "e21_single_refs",
    # legacy failure classes (double mouth, flagship slab) — must also fail
    "e2_2mv_explicit_refs",
    "e6_flagship_crop_refs",
]

ROOT = Path(__file__).resolve().parents[2] / "out" / "laurent-bust-redo"
OUT = Path("/tmp/eval_calibration")


def main() -> None:
    config = EvaluationConfig()
    photo = PhotoReference.load(Path("/tmp/laurent_front_clean4.png"), config)
    print(f"photo mask: {photo.mask_source}")
    results = {}
    for model in MODELS:
        glb = ROOT / model / "scene.glb"
        t0 = time.time()
        try:
            results[model] = evaluate_model(
                glb, photo, config, save_debug_dir=OUT / model
            )
        except Exception as exc:  # keep the sweep alive; report loudly
            results[model] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"FAIL {model}: {results[model]['error']}", file=sys.stderr)
            continue
        v = results[model]["verdict"]
        print(
            f"{model:28s} {time.time()-t0:5.1f}s mesh_ok={v['mesh_ok']} "
            f"tex_ok={v['texture_ok']} failing={v['failing_measures']}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps(results, indent=1))

    # compact matrix
    print("\n=== worst-angle matrix (rows=models) ===")
    header = None
    for model, res in results.items():
        if "error" in res:
            continue
        cells = {}
        for axis, group in res["measures"].items():
            for name, m in group.items():
                cells[f"{axis[:3]}.{name}"] = m["worst"]
        if header is None:
            header = list(cells)
            print(f"{'model':28s} " + " ".join(f"{h[:18]:>18s}" for h in header))
        print(
            f"{model:28s} "
            + " ".join(
                f"{cells[h]:>18.4f}" if cells[h] is not None else f"{'nan':>18s}"
                for h in header
            )
        )


if __name__ == "__main__":
    main()
