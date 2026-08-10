#!/usr/bin/env python3
"""Control-experiment input prep (e22 view-quality isolation, 2026-07-22).

Separates "view quality" from "pipeline mechanics": e22_oneshot_v2's pass-2
carved catastrophic matte debris + a wrong-identity side view into the mesh
(out/bust/e22_oneshot_v2/geometry_view_windowed_side_left.png). The S2
bench produced PROVEN-CLEAN views of the same man at measured poses
(out/bust-viewbench/E/). This prep windows the bench views + the source
photo with the loop's OWN window law (loop_conditioning.window_view — the
exact helper e22v2's pass-2 used), so a pass-2 run conditioned on them is
byte-parallel to the failing run in everything but view quality.

Outputs into --out:
  windowed_front.png       source photo, matted + window law
  windowed_side_left.png   bench E profile_left (pose err 0 deg), window law
  windowed_back.png        bench E back (pose err 5 deg), window law
  fullspan_side_left.png   matted full-span copies for the later texture
  fullspan_back.png        bake (split-consumer law: bake wants full span)
  prep_report.json         window anchors + kept ratios per view
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_SOURCE = Path("/tmp/laurent_front_clean4.png")
BENCH = Path("out/bust-viewbench/E")


def load_matted(path: Path):
    """probe_e22_ladder.load_source: matte with the production helpers."""
    from PIL import Image

    from abstract3d.segmentation import clean_alpha_mask, remove_background_robust

    source = Image.open(path).convert("RGBA")
    if source.getchannel("A").getextrema()[0] < 255:
        return clean_alpha_mask(source)
    return remove_background_robust(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from abstract3d.loop_conditioning import window_view

    report = {}
    plan = [
        ("front", args.source, "windowed_front.png", None),
        ("side_left", args.bench / "profile_left.png",
         "windowed_side_left.png", "fullspan_side_left.png"),
        ("back", args.bench / "back.png", "windowed_back.png",
         "fullspan_back.png"),
    ]
    for label, src, win_name, full_name in plan:
        rgba = load_matted(src)
        if full_name:
            rgba.save(args.out / full_name)
        windowed, window_report = window_view(rgba)
        windowed.save(args.out / win_name)
        window_report["source"] = str(src)
        report[label] = window_report
        print(f"[{label}] {src} -> {win_name} anchors={window_report}")

    spreads = [r["kept_ratio_shoulder"] for r in report.values()]
    report["kept_ratio_spread"] = round(max(spreads) - min(spreads), 4)
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=1))
    print(f"kept_ratio_shoulder spread: {report['kept_ratio_spread']}")


if __name__ == "__main__":
    main()
