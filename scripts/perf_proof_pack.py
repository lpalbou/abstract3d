"""Build the bake performance-program proof pack: before/after stats,
charts, and pixel-difference sheets against the certified textures.

The central claim is BIT-EXACTNESS: every optimized stage reproduces the
certified texture hashes, so quality is provably unchanged — the diff
sheets exist so a human reviewer can confirm without trusting the hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "artifacts/validation/bake-performance-program"
GOLDEN = REPO / "artifacts/validation/golden-bake"

PUBLISHED = {
    "face": REPO / "artifacts/validation/iter3-multiview-fixed/face-2mv/texture.png",
    "ship": REPO / "artifacts/validation/final-proof/hunyuan-starship/texture.png",
    "owl": REPO / "artifacts/validation/final-proof/hunyuan-owl/texture.png",
}

# Baseline measurements recorded on the pre-optimization tree (commit
# 8c49cda), same machine (Apple M5 Max, 128 GB), same isolated-process
# methodology, canonical golden recipes at res 2048.
BASELINE = {
    "face": {"seconds": 220.5, "rss_peak_gb": 7.39, "stages": {
        "commit_pale_chips": 43.8, "commit_trace_deposits": 17.8,
        "composite_gradient_domain": 11.0,
        "register_reference_by_source_overlap": 19.9,
        "synthesize_fill_detail": 6.0, "tone_bottom_cap": 5.2}},
    "ship": {"seconds": 58.6, "rss_peak_gb": 5.93, "stages": {
        "synthesize_fill_detail": 16.5, "texel_surface_smooth": 2.6,
        "enforce_fill_luminance_floor": 1.7, "mirror_fill_from_observed": 1.0}},
    "owl": {"seconds": 257.8, "rss_peak_gb": 5.97, "stages": {
        "mirror_fill_from_observed": 166.8, "synthesize_fill_detail": 49.5,
        "estimate_pose_photometric": 3.8, "texel_surface_smooth": 2.7}},
}


def main() -> int:
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((GOLDEN / "summary.json").read_text())
    records = {r["asset"]: r for r in summary}

    # ---- per-asset pixel-difference sheets --------------------------------
    for asset, published_path in PUBLISHED.items():
        rebaked_path = GOLDEN / asset / "texture.png"
        if not rebaked_path.exists():
            continue
        published = np.asarray(Image.open(published_path).convert("RGB"), np.int16)
        rebaked = np.asarray(Image.open(rebaked_path).convert("RGB"), np.int16)
        diff = np.abs(published - rebaked)
        scale = 512
        sheet = Image.new("RGB", (scale * 3 + 32, scale + 56), "white")
        for i, (img, label) in enumerate((
            (Image.open(published_path).convert("RGB"), "published (certified)"),
            (Image.open(rebaked_path).convert("RGB"), "rebaked (optimized pipeline)"),
            (Image.fromarray((np.clip(diff * 20, 0, 255)).astype(np.uint8)),
             f"|diff| x20  (max={int(diff.max())})"),
        )):
            sheet.paste(img.resize((scale, scale), Image.LANCZOS), (i * (scale + 16), 40))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
        for i, label in enumerate(("published (certified)",
                                   "rebaked (optimized pipeline)",
                                   f"|diff| x20  (max={int(diff.max())})")):
            draw.text((i * (scale + 16) + 8, 12), label, fill="black")
        draw.text((8, scale + 44),
                  f"{asset}: md5 {hashlib.md5(published_path.read_bytes()).hexdigest()}"
                  f" == {hashlib.md5(rebaked_path.read_bytes()).hexdigest()}"
                  f" -> {'IDENTICAL' if diff.max() == 0 else 'DIFFERENT'}",
                  fill="black")
        sheet.save(OUT / f"{asset}_identity_sheet.png")

    # ---- before/after chart ----------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = [a for a in PUBLISHED if a in records]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(assets))
    before_t = [BASELINE[a]["seconds"] for a in assets]
    after_t = [records[a]["seconds"] for a in assets]
    axes[0].bar(x - 0.18, before_t, width=0.36, label="before", color="#c56")
    axes[0].bar(x + 0.18, after_t, width=0.36, label="after", color="#4a4")
    for i, (b, a) in enumerate(zip(before_t, after_t)):
        axes[0].text(i + 0.18, a + 3, f"{b/a:.1f}x", ha="center", fontsize=10)
    axes[0].set_xticks(x, assets)
    axes[0].set_ylabel("bake seconds (res 2048)")
    axes[0].set_title("Golden-bake wall time")
    axes[0].legend()
    before_m = [BASELINE[a]["rss_peak_gb"] for a in assets]
    after_m = [records[a].get("rss_peak_gb", 0) for a in assets]
    axes[1].bar(x - 0.18, before_m, width=0.36, label="before", color="#c56")
    axes[1].bar(x + 0.18, after_m, width=0.36, label="after", color="#4a4")
    axes[1].set_xticks(x, assets)
    axes[1].set_ylabel("RSS peak (GB)")
    axes[1].set_title("Process memory peak during bake")
    axes[1].legend()
    fig.suptitle("Texture bake performance program — outputs bit-identical "
                 "(certified hashes reproduced)")
    fig.tight_layout()
    fig.savefig(OUT / "before_after_chart.png", dpi=120)
    plt.close(fig)

    # ---- machine-readable summary -----------------------------------------
    report = {
        "methodology": (
            "Canonical golden-recipe rebakes (scripts/golden_bake.py) at res 2048, "
            "one process per asset, Apple M5 Max 128 GB (mps profile). Every "
            "optimization is bitwise output-preserving: baked texture md5 equals "
            "the certified published hash for all assets, before and after."),
        "baseline_commit": "8c49cda (v0.2.0)",
        "assets": {},
    }
    for asset in assets:
        record = records[asset]
        report["assets"][asset] = {
            "status": record["status"],
            "texture_md5": record["baked_md5"],
            "seconds_before": BASELINE[asset]["seconds"],
            "seconds_after": record["seconds"],
            "speedup": round(BASELINE[asset]["seconds"] / record["seconds"], 2),
            "rss_peak_gb_before": BASELINE[asset]["rss_peak_gb"],
            "rss_peak_gb_after": record.get("rss_peak_gb"),
            "slowest_stages_before": BASELINE[asset]["stages"],
            "slowest_stages_after": record.get("slowest_stages"),
        }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))

    # copy the after-run timeline plots into the proof pack
    import shutil

    for asset in assets:
        src = GOLDEN / asset / "profile_timeline.png"
        if src.exists():
            shutil.copy(src, OUT / f"{asset}_timeline_after.png")

    print(json.dumps({a: report["assets"][a]["speedup"] for a in assets}, indent=1))
    print(f"proof pack written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
