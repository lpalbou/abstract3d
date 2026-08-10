#!/usr/bin/env python3
"""Evaluate bust model(s) against the source photo — the general evaluation.

One command per model: renders the evaluation orbit fresh from the GLB
(seconds; no GPU generation, CPU/GL rasterization only), computes the
general comparisons, prints the per-angle score matrix, and emits a verdict
JSON an automated one-shot loop can iterate against:

  {"accept": bool, "mesh_ok": bool, "texture_ok": bool,
   "failing_measures": ["texture.palette_outliers@-90.0", ...],
   "worst_angle": {...per measure...}}

Design + calibration evidence: docs/research/evaluation_strategy_v2.md.
Measures (module docstring of abstract3d.model_evaluation):
  texture: source_structure / source_chroma (photo projected through the
           front camera, block-matched agreement) + palette_outliers
           (height-banded photo palette, full orbit)
  mesh:    mirror_symmetry + profile_articulation + cavity_mass +
           silhouette_floor
Aggregation is worst-angle (min), never mean.

Usage:
  .venv/bin/python scripts/evaluate_model.py \
      --photo /tmp/laurent_front_clean4.png \
      --glb out/laurent-bust-redo/e20_rebake_fixed/scene.glb \
      [--json-out /tmp/verdict.json] [--debug-dir /tmp/eval_debug]

Exit code: 0 when every evaluated model is accepted, 1 otherwise, 2 on
input errors — so a pipeline can gate on the exit code alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abstract3d.model_evaluation import (  # noqa: E402
    EvaluationConfig,
    PhotoReference,
    evaluate_model,
)


def print_matrix(result: dict) -> None:
    print(f"\n=== {result['model']} ===")
    print(f"photo mask: {result['photo_mask_source']}")
    print(
        "row alignment: scale=%(scale)s shift=%(shift)s ncc=%(ncc)s"
        % result["row_alignment"]
    )
    sil = result["silhouette_registration"]
    print(f"silhouette registration: iou={sil['iou']} at scale={sil['scale']}")
    for axis in ("texture", "mesh"):
        print(f"[{axis}]")
        for name, m in result["measures"][axis].items():
            angles = " ".join(
                f"{angle}:{value if value is not None else 'nan'}"
                for angle, value in m["per_angle"].items()
            )
            flag = "PASS" if m["pass"] else "FAIL"
            op = ">=" if m["direction"] == "min" else "<="
            print(
                f"  {name:20s} {flag}  worst={m['worst']} @ az {m['worst_angle']}"
                f"  (need {op} {m['threshold']})"
            )
            print(f"  {'':20s}       per-angle: {angles}")
    verdict = result["verdict"]
    print(
        f"verdict: accept={verdict['accept']} mesh_ok={verdict['mesh_ok']} "
        f"texture_ok={verdict['texture_ok']}"
    )
    if verdict["failing_measures"]:
        print(f"failing: {', '.join(verdict['failing_measures'])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--photo", required=True, type=Path,
                        help="Source photo (matted RGBA preferred; RGB degrades "
                             "to border-differencing with a #FALLBACK note).")
    parser.add_argument("--glb", nargs="+", required=True, type=Path)
    parser.add_argument("--json-out", type=Path,
                        help="Write full results (matrix + verdicts) as JSON.")
    parser.add_argument("--debug-dir", type=Path,
                        help="Save render-vs-projection side-by-sides here.")
    parser.add_argument("--render-size", type=int, default=None,
                        help="Override the evaluation render size (default 768).")
    args = parser.parse_args()

    if not args.photo.is_file():
        print(f"ERROR: photo not found: {args.photo}", file=sys.stderr)
        return 2
    config = EvaluationConfig()
    if args.render_size:
        config = dataclasses.replace(config, render_size=int(args.render_size))

    photo = PhotoReference.load(args.photo, config)
    if "#FALLBACK" in photo.mask_source:
        print(f"WARNING: {photo.mask_source}", file=sys.stderr)

    results = {}
    all_accepted = True
    for glb in args.glb:
        if not glb.is_file():
            print(f"ERROR: GLB not found: {glb}", file=sys.stderr)
            return 2
        label = glb.parent.name if glb.name == "scene.glb" else glb.stem
        started = time.time()
        debug_dir = (args.debug_dir / label) if args.debug_dir else None
        try:
            result = evaluate_model(glb, photo, config, save_debug_dir=debug_dir)
        except Exception as exc:
            print(f"ERROR evaluating {label}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
            all_accepted = False
            continue
        result["elapsed_s"] = round(time.time() - started, 1)
        results[label] = result
        print_matrix(result)
        print(f"elapsed: {result['elapsed_s']}s")
        all_accepted &= bool(result["verdict"]["accept"])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json_out}")

    return 0 if all_accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
