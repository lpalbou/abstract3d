"""Golden-bake regression harness: canonical rebakes must reproduce the
published texture hashes bit-exactly on the validated profile.

This is the executable form of the certification's determinism claim and the
safety net for every texture-pipeline change: run it BEFORE and AFTER a
change; a hash mismatch means the change altered certified output and must
either be reverted or re-certified through the adversarial battery.

Assets and canonical recipes (recovered from the cycle-7 verdict scripts):

- face  : iter3-multiview-fixed/face-2mv geometry + input.png (robust-matted)
          + left/right profile references at +/-90 deg, res 2048
- ship  : final-proof/hunyuan-starship geometry + input.png, single view,
          source_pose_override=(30, 15), res 2048
- owl   : final-proof/hunyuan-owl geometry + input.png, single view, res 2048

Usage:
  python scripts/golden_bake.py --asset ship [--out DIR] [--profile]
  python scripts/golden_bake.py --asset all --profile

Exit code 0 iff every baked hash matches its published hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FACE_BUNDLE = REPO / "artifacts/validation/iter3-multiview-fixed/face-2mv"
SHIP_BUNDLE = REPO / "artifacts/validation/final-proof/hunyuan-starship"
OWL_BUNDLE = REPO / "artifacts/validation/final-proof/hunyuan-owl"
FACE_PROTO = REPO / "artifacts/validation/face-multiview-prototype"

ASSETS = {
    "face": {
        "bundle": FACE_BUNDLE,
        "references": [
            {"image": FACE_PROTO / "left_profile_clean.png", "angle": "side_left"},
            {"image": FACE_PROTO / "right_profile_clean.png", "angle": "side_right"},
        ],
        "kwargs": {},
    },
    "ship": {
        "bundle": SHIP_BUNDLE,
        "references": [],
        "kwargs": {"source_pose_override": (30.0, 15.0)},
    },
    "owl": {
        "bundle": OWL_BUNDLE,
        "references": [],
        "kwargs": {},
    },
}

# The published (versioned) texture hashes are the goldens themselves: read
# them from disk so the harness can never drift from the repository state.
def published_md5(asset: str) -> str:
    return hashlib.md5((ASSETS[asset]["bundle"] / "texture.png").read_bytes()).hexdigest()


# Bake-stage functions instrumented under --profile. External wrapping only:
# the library code path is untouched, so profiled bakes stay bit-identical.
PROFILED_STAGES = [
    "estimate_pose_photometric",
    "register_view_2d",
    "register_view_by_width_profile",
    "register_reference_by_source_overlap",
    "refine_reference_poses",
    "blend_projections",
    "filter_projection_outliers",
    "resolve_projection_conflicts",
    "harmonize_and_gate_projection",
    "admit_scarce_witnesses",
    "mirror_fill_from_observed",
    "detect_mirror_rescue_discs",
    "mesh_graph_harmonic_fill",
    "synthesize_fill_detail",
    "texel_surface_smooth",
    "level_composed_seams",
    "tone_match_completion_components",
    "commit_trace_deposits",
    "commit_pale_chips",
    "tone_bottom_cap",
    "reconcile_shadow_aprons",
    "reconcile_specular_lobes",
    "world_space_voxel_graph_complex_clustering",
    "voxel_complex_clustering",
    "render_informed_speck_consolidation",
    "enforce_fill_luminance_floor",
    # helper-level probes (multiple calls each; useful for attribution)
    "_balanced_query",
    "_masked_gaussian_filter",
    "_fill_value_noise_3d",
    "_multigrid_orientation_field",
    "_voxel_ball_stats",
]


def run_asset(asset: str, out_root: Path, profile: bool, resolution: int) -> dict:
    from abstract3d import bundle as bundle_api

    spec = ASSETS[asset]
    references = [dict(r) for r in spec["references"]]
    for reference in references:
        if not Path(reference["image"]).exists():
            return {"asset": asset, "status": "skipped",
                    "reason": f"missing local reference photo {reference['image']}"}

    out_dir = out_root / asset
    profiler = None
    sampler = None
    if profile:
        from abstract3d import texturing
        from abstract3d.profiling import MemorySampler, StageProfiler

        sampler = MemorySampler().start()
        profiler = StageProfiler(sampler=sampler)
        profiler.wrap_module_functions(texturing, PROFILED_STAGES)
        try:
            import abstract3d.gradient_compositing as gradient_compositing

            profiler.wrap_module_functions(
                gradient_compositing, ["composite_gradient_domain"]
            )
        except Exception:
            pass
        try:
            import abstract3d.backends.triposr_runtime as triposr_runtime

            profiler.wrap_module_functions(
                triposr_runtime,
                ["_tripo_project_observed_texture", "_tripo_make_texture_atlas",
                 "_tripo_rasterize_position_atlas", "_tripo_render_camera_depth_map"],
            )
        except Exception:
            pass
        try:
            import abstract3d.feature_fringe_repair as feature_fringe_repair

            profiler.wrap_module_functions(
                feature_fringe_repair, ["repair_feature_fringes"])
        except Exception:
            pass
        try:
            import abstract3d.film_band_gradient as film_band_gradient
            import abstract3d.film_band as film_band

            profiler.wrap_module_functions(
                film_band_gradient, ["repaint_film_band"])
            profiler.wrap_module_functions(
                film_band, ["compute_view_film_maps"])
        except Exception:
            pass
        try:
            import abstract3d.reference_flow as reference_flow

            profiler.wrap_module_functions(
                reference_flow, ["estimate_reference_flow"])
        except Exception:
            pass

    started = time.perf_counter()
    try:
        _textured, stats = bundle_api.rebake_bundle(
            spec["bundle"],
            output_dir=out_dir,
            references=references,
            texture_resolution=resolution,
            **spec["kwargs"],
        )
    finally:
        if profiler is not None:
            profiler.unwrap()
        if sampler is not None:
            sampler.stop()
    elapsed = time.perf_counter() - started

    baked_md5 = hashlib.md5((out_dir / "texture.png").read_bytes()).hexdigest()
    expected = published_md5(asset) if resolution == 2048 else None
    record = {
        "asset": asset,
        "status": (
            "match" if expected == baked_md5
            else ("mismatch" if expected else "no-golden-at-this-resolution")
        ),
        "baked_md5": baked_md5,
        "expected_md5": expected,
        "seconds": round(elapsed, 1),
        "coverage": stats.get("observed_coverage_ratio"),
    }
    if profiler is not None:
        report = profiler.save(out_dir / "profile.json")
        record["rss_peak_gb"] = round((report["overall_rss_peak_bytes"] or 0) / 1e9, 2)
        top = sorted(report["stages"], key=lambda s: -s["seconds"])[:8]
        record["slowest_stages"] = [
            {"name": s["name"], "seconds": s["seconds"],
             "rss_peak_gb": round((s["rss_peak_bytes"] or 0) / 1e9, 2)}
            for s in top
        ]
        try:
            from abstract3d.profiling import plot_timeline

            full = json.loads((out_dir / "profile.json").read_text())
            plot_timeline(full, out_dir / "profile_timeline.png",
                          title=f"golden bake {asset} (res {resolution})")
        except Exception as exc:
            record["plot_error"] = str(exc)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="all",
                        choices=["all", *ASSETS.keys()])
    parser.add_argument("--out", default=str(REPO / "artifacts/validation/golden-bake"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--res", type=int, default=2048)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.asset == "all":
        # One subprocess per asset: RSS peaks stay attributable (a shared
        # process carries the previous asset's high-water mark forward).
        import subprocess

        records = []
        for asset in ASSETS:
            print(f"[golden-bake] {asset} (subprocess) ...", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--asset", asset, "--out", str(out_root),
                 "--res", str(args.res)] + (["--profile"] if args.profile else []),
                capture_output=True, text=True)
            sys.stdout.write(proc.stdout)
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
            entry = json.loads((out_root / "summary.json").read_text())
            records.extend(entry)
        (out_root / "summary.json").write_text(json.dumps(records, indent=1))
        failed = [r for r in records if r["status"] == "mismatch"]
        print(f"[golden-bake] {len(records) - len(failed)}/{len(records)} ok")
        return 1 if failed else 0

    records = []
    for asset in [args.asset]:
        print(f"[golden-bake] {asset} ...", flush=True)
        record = run_asset(asset, out_root, args.profile, args.res)
        records.append(record)
        print(json.dumps(record, indent=1), flush=True)

    (out_root / "summary.json").write_text(json.dumps(records, indent=1))
    failed = [r for r in records if r["status"] == "mismatch"]
    print(f"[golden-bake] {len(records) - len(failed)}/{len(records)} ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
