"""Aggregate generation statistics from every bundle metadata.json on disk.

Scans `artifacts/validation/**/metadata.json` (local archive included when
present) and reports per-backend/per-task medians and ranges for wall time,
stage times, and mesh density. Writes a markdown table and a JSON report so
docs/benchmarks.md can cite regenerable numbers.

Usage: python scripts/generation_stats.py [--json OUT] [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def collect(root: Path) -> list:
    rows = []
    for meta_path in root.rglob("metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not isinstance(meta, dict) or "backend_id" not in meta:
            continue
        timings = meta.get("timings_s") or {}
        if not timings.get("total"):
            continue
        rows.append({
            "bundle": str(meta_path.parent.relative_to(REPO)) if meta_path.is_relative_to(REPO) else str(meta_path.parent),
            "backend": str(meta.get("backend_id", "?")).replace("abstract3d:", ""),
            "task": {"image_to_scene3d": "i23d", "text_to_scene3d": "t23d"}.get(
                str(meta.get("task")), str(meta.get("task"))),
            "total_s": timings.get("total"),
            "inference_s": timings.get("inference"),
            "texture_s": timings.get("texture"),
            "image_gen_s": timings.get("source_image_generation"),
            "vertices": meta.get("vertex_count"),
            "faces": meta.get("face_count"),
        })
    return rows


def summarize(rows: list) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["task"])].append(row)

    def stat(entries, key):
        values = [e[key] for e in entries if e[key]]
        if not values:
            return None
        return {
            "median": round(statistics.median(values), 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "n": len(values),
        }

    summary = {}
    for (backend, task), entries in sorted(groups.items()):
        summary[f"{backend}/{task}"] = {
            "samples": len(entries),
            "total_s": stat(entries, "total_s"),
            "inference_s": stat(entries, "inference_s"),
            "texture_s": stat(entries, "texture_s"),
            "image_gen_s": stat(entries, "image_gen_s"),
            "vertices": stat(entries, "vertices"),
            "faces": stat(entries, "faces"),
        }
    return summary


def to_markdown(summary: dict) -> str:
    lines = [
        "| backend / task | n | total (median [min-max]) | mesh vertices (median) | mesh faces (median) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, s in summary.items():
        total = s["total_s"]
        verts = s["vertices"]
        faces = s["faces"]
        lines.append(
            f"| {key} | {s['samples']} | "
            f"{total['median']:.0f} s [{total['min']:.0f}-{total['max']:.0f}] | "
            f"{verts['median']:,.0f} | {faces['median']:,.0f} |"
            if total and verts and faces else f"| {key} | {s['samples']} | - | - | - |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO / "artifacts/validation"))
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    rows = collect(Path(args.root))
    summary = summarize(rows)
    print(to_markdown(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"rows": len(rows), "summary": summary}, indent=1))
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
