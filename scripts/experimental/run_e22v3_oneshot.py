#!/usr/bin/env python3
"""e22v3 one-shot launcher: the exact e22v2 recipe with an explicit seed.

The CLI does not expose the run seed, and the loop's refusal contract says
"Retry with a different seed" — pass-1 scaffold quality is the measured
dominant variance axis (photo-vs-front IoU 0.667-0.776 across same-seed
draws) and clean-machine runs reproduce the same scaffold (v3 runs 2/3
byte-identical). This launcher is `abstract3d i23d ... --geometry-
conditioning loop` through the same Scene3DManager path with `seed` (and
nothing else) added.

Usage:
  ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 ABSTRACT3D_IMAGE_PROVIDER=mlx-gen \
  ABSTRACT3D_IMAGE_MODEL=... ABSTRACT3D_IMAGE_LORA_ADAPTERS=... \
  python scripts/experimental/run_e22v3_oneshot.py \
      --output-dir out/bust/e22_oneshot_v3 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_SOURCE = "/tmp/laurent_front_clean4.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=512)
    args = parser.parse_args()

    from abstract3d.scene3d_manager import Scene3DManager

    manager = Scene3DManager(backend_id="hunyuan3d21")
    result = manager.i23d(
        args.source,
        format="glb",
        output_dir=args.output_dir,
        device="mps",
        geometry_conditioning="loop",
        texture_reference_allow_person=True,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=args.octree_resolution,
        seed=args.seed,
    )
    metadata = result.get("metadata") or {}
    print(json.dumps(metadata, indent=2, sort_keys=True))
    verdict = (metadata.get("quality_verdict") or {}).get("verdict", "healthy")
    print(f"quality_verdict: {verdict}", file=sys.stderr)


if __name__ == "__main__":
    main()
