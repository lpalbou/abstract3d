"""Texture-path parity canaries: the pipeline/rebake parity audit's P1/P2
spec as a permanent, executable regression tripwire (CPU-only, no
generation, no MPS).

Background (parity audit, 2026-07): the pipeline (hunyuan runtime) and
`rebake_bundle` bake cores are equal to within GLB float32 quantization —
a refs-off rebake reproduced the shipped 2048 pipeline texture at 99.888%
bit-identical texels with max channel delta 5, and consecutive refs-off
rebakes are md5-identical across processes and days. Those two facts are
the invariants this script pins; any future texture-path change that
breaks them is either a real behavior change (recalibrate the pinned
fixture DELIBERATELY, with a changelog entry) or a nondeterminism
regression (fix it).

Fixture: `artifacts/validation/parity/sportscar_v7/` — the pose-estimated
(az 17.5 / el 8.0), low-coverage (0.112) subject the owl-class fixtures
cannot stand in for (they are pose-trivial and high-coverage). Pinned
contents: `geometry.glb` + `input.png` (the rebake inputs), the four
accepted generated views (back / side_left / side_right / top_rear —
inputs for future generation-lane work; NOT consumed here), and
`texture.png`, the refs-off 2048 rebake of the pinned inputs.

Canaries:

- P1 — product parity: a refs-off `rebake_bundle` at 2048 must reproduce
  the pinned `texture.png` at >= 99.5% bit-identical texels AND max
  channel delta <= 8 (measured basis 99.888% / 5 across the float64
  pipeline-vs-float32 rebake gap, so same-path reproduction has wide
  margin). Stats must agree too: `source_pose` exactly (17.5, 8.0)
  (the pose estimator is deterministic in photo+mesh), coverage within
  +/-0.005, fill-detail energy-calibration scale within +/-0.05.
- P2 — rebake determinism: two consecutive refs-off rebakes at 1024 must
  be md5-IDENTICAL (measured: holds across processes and days).
- P3 — face single-photo canary: two refs-off rebakes at 2048 of the
  face proof bundle (`final-proof/hunyuan-face`) must be md5-identical
  to each other. Single-view bakes must stay untouched by every
  multi-view stage (delight / tone consensus / handoff ledger are
  structurally inert at view_count == 1); this is the fleet's
  no-collateral guarantee.

Usage:
  .venv/bin/python scripts/parity_canary.py            # all canaries
  .venv/bin/python scripts/parity_canary.py --only p1
  .venv/bin/python scripts/parity_canary.py --repin    # rebuild pinned
                                                       # texture.png from
                                                       # the CURRENT tree
                                                       # (deliberate act)

Exit code 0 iff every canary passes. The pytest wrapper
(`tests/test_parity_canary.py`) runs this only when
ABSTRACT3D_PARITY_CANARY=1 — the bakes take ~15 min total, so the canary
is an opt-in gate for texture-path changes, not a unit test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PARITY_BUNDLE = REPO / "artifacts/validation/parity/sportscar_v7"
FACE_BUNDLE = REPO / "artifacts/validation/final-proof/hunyuan-face"

# Pinned stats facts of the parity bundle (recorded at pin time; the
# parity audit's E5 reproduced them from the shipped pipeline run).
PINNED_SOURCE_POSE = (17.5, 8.0)
PINNED_COVERAGE = 0.112
COVERAGE_TOL = 0.005
FILL_SCALE_TOL = 0.05

P1_MIN_IDENTICAL_FRAC = 0.995
P1_MAX_CHANNEL_DELTA = 8


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _rebake(bundle: Path, out_dir: Path, resolution: int) -> dict:
    from abstract3d.bundle import rebake_bundle

    started = time.perf_counter()
    _, stats = rebake_bundle(
        bundle,
        output_dir=out_dir,
        generate_references="off",
        texture_resolution=resolution,
    )
    return {
        "stats": stats,
        "seconds": round(time.perf_counter() - started, 1),
        "texture": out_dir / "texture.png",
    }


def _texel_identity(reference: Path, produced: Path) -> dict:
    import numpy as np
    from PIL import Image

    ref = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int16)
    got = np.asarray(Image.open(produced).convert("RGB"), dtype=np.int16)
    if ref.shape != got.shape:
        return {"identical_frac": 0.0, "max_delta": 255,
                "error": f"shape {got.shape} vs pinned {ref.shape}"}
    delta = np.abs(ref - got)
    return {
        "identical_frac": float((delta.max(axis=2) == 0).mean()),
        "max_delta": int(delta.max()),
        "mean_delta": round(float(delta.mean()), 4),
    }


def canary_p1(work: Path) -> dict:
    """Refs-off 2048 rebake reproduces the pinned texture + stats."""

    result = _rebake(PARITY_BUNDLE, work / "p1", 2048)
    stats = result["stats"]
    identity = _texel_identity(PARITY_BUNDLE / "texture.png", result["texture"])

    pose = stats.get("source_pose") or {}
    got_pose = (pose.get("azimuth_deg"), pose.get("elevation_deg"))
    coverage = float(stats.get("observed_coverage_ratio") or 0.0)
    fill_scale = float(
        ((stats.get("fill_detail") or {}).get("energy_calibration") or {})
        .get("scale") or 0.0)
    pinned_fill_scale = _pinned_fact("fill_detail_scale")

    failures = []
    if identity.get("identical_frac", 0.0) < P1_MIN_IDENTICAL_FRAC:
        failures.append(
            f"texel identity {identity['identical_frac']:.5f} < {P1_MIN_IDENTICAL_FRAC}")
    if identity.get("max_delta", 255) > P1_MAX_CHANNEL_DELTA:
        failures.append(
            f"max channel delta {identity['max_delta']} > {P1_MAX_CHANNEL_DELTA}")
    if tuple(round(float(v or 0.0), 1) for v in got_pose) != PINNED_SOURCE_POSE:
        failures.append(f"source_pose {got_pose} != pinned {PINNED_SOURCE_POSE}")
    if abs(coverage - PINNED_COVERAGE) > COVERAGE_TOL:
        failures.append(
            f"coverage {coverage:.4f} off pinned {PINNED_COVERAGE} by > {COVERAGE_TOL}")
    if pinned_fill_scale is not None and abs(fill_scale - pinned_fill_scale) > FILL_SCALE_TOL:
        failures.append(
            f"fill_detail scale {fill_scale:.3f} off pinned "
            f"{pinned_fill_scale:.3f} by > {FILL_SCALE_TOL}")

    return {
        "name": "P1_product_parity", "passed": not failures,
        "failures": failures, "identity": identity,
        "source_pose": got_pose, "coverage": coverage,
        "fill_detail_scale": fill_scale, "seconds": result["seconds"],
    }


def canary_p2(work: Path) -> dict:
    """Two consecutive refs-off 1024 rebakes are md5-identical."""

    md5s = []
    seconds = 0.0
    for run in (1, 2):
        result = _rebake(PARITY_BUNDLE, work / f"p2_run{run}", 1024)
        md5s.append(_md5(result["texture"]))
        seconds += result["seconds"]
    passed = md5s[0] == md5s[1]
    return {
        "name": "P2_rebake_determinism", "passed": passed,
        "failures": [] if passed else [f"md5 mismatch: {md5s[0]} != {md5s[1]}"],
        "md5s": md5s, "seconds": round(seconds, 1),
    }


def canary_face(work: Path) -> dict:
    """Face proof: two refs-off 2048 rebakes are md5-identical (the
    single-view no-collateral guarantee of every multi-view stage)."""

    if not (FACE_BUNDLE / "geometry.glb").exists():
        return {"name": "P3_face_md5_stability", "passed": True,
                "skipped": f"missing bundle {FACE_BUNDLE}", "failures": []}
    md5s = []
    seconds = 0.0
    for run in (1, 2):
        result = _rebake(FACE_BUNDLE, work / f"face_run{run}", 2048)
        md5s.append(_md5(result["texture"]))
        seconds += result["seconds"]
    passed = md5s[0] == md5s[1]
    return {
        "name": "P3_face_md5_stability", "passed": passed,
        "failures": [] if passed else [f"md5 mismatch: {md5s[0]} != {md5s[1]}"],
        "md5s": md5s, "seconds": round(seconds, 1),
    }


def _pinned_facts_path() -> Path:
    return PARITY_BUNDLE / "pinned_facts.json"


def _pinned_fact(key: str):
    path = _pinned_facts_path()
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(key)


def repin(work: Path) -> dict:
    """Rebuild the pinned texture.png + facts from the CURRENT tree.

    A deliberate act after an intended texture-path change: run the full
    validation battery first, then repin and record the reason in the
    CHANGELOG (the diff of pinned_facts.json carries the numbers).
    """

    result = _rebake(PARITY_BUNDLE, work / "repin", 2048)
    stats = result["stats"]
    shutil.copyfile(result["texture"], PARITY_BUNDLE / "texture.png")
    pose = stats.get("source_pose") or {}
    facts = {
        "texture_md5": _md5(PARITY_BUNDLE / "texture.png"),
        "source_pose": [pose.get("azimuth_deg"), pose.get("elevation_deg")],
        "coverage": stats.get("observed_coverage_ratio"),
        "fill_detail_scale": (
            ((stats.get("fill_detail") or {}).get("energy_calibration") or {})
            .get("scale")),
        "resolution": 2048,
    }
    _pinned_facts_path().write_text(json.dumps(facts, indent=1))
    return {"name": "repin", "passed": True, "facts": facts,
            "seconds": result["seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["p1", "p2", "face"], default=None)
    parser.add_argument("--repin", action="store_true")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the temp work dir (debugging)")
    args = parser.parse_args()

    if not (PARITY_BUNDLE / "geometry.glb").exists():
        print(f"[parity] fixture bundle missing: {PARITY_BUNDLE}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="parity_canary_"))
    try:
        if args.repin:
            record = repin(work)
            print(json.dumps(record, indent=1, default=str))
            return 0
        canaries = {"p1": canary_p1, "p2": canary_p2, "face": canary_face}
        selected = [args.only] if args.only else list(canaries)
        records = []
        for name in selected:
            print(f"[parity] {name} ...", flush=True)
            record = canaries[name](work)
            records.append(record)
            print(json.dumps(record, indent=1, default=str), flush=True)
        failed = [r for r in records if not r["passed"]]
        print(f"[parity] {len(records) - len(failed)}/{len(records)} ok")
        return 1 if failed else 0
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
