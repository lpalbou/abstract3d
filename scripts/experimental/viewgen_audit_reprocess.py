#!/usr/bin/env python
"""Viewgen audit — stage 4b: offline reprocess of SAVED raw draws.

The remote editor hit its billing hard limit mid-run, so no new draws are
possible; this reprocesses the raws persisted by stage 4 (round 3) through a
WIDENED registration search (the failed side_right draws are framed tighter
than the clay: production scale search stops at 0.88, close-up draws need
less) + the iterative piecewise row alignment + the same acceptance gate,
and merges the winner into viewgen_fixed/.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT_DIR = REPO / "out/laurent-bust-redo/viewgen_fixed"
WORK = Path("/tmp/viewgen_audit/regen")
SOURCE_PHOTO = Path("/tmp/laurent_bust_crop.png")
ACCEPT_FRAC = 0.02
LABEL = "side_right"
AZIMUTH = -90.0
WIDE_SCALES = (0.70, 0.76, 0.82, 0.88, 0.94, 1.0, 1.06, 1.12)


def main() -> None:
    from PIL import Image

    from abstract3d.reference_generation import (
        match_tone_lab,
        register_matte_to_clay,
        silhouette_iou,
        suppress_specular_highlights,
    )
    from abstract3d.segmentation import remove_background_robust
    from viewgen_audit_analyze import _load_mesh, estimate_azimuth, offsets
    from viewgen_audit_measure import measure
    from viewgen_audit_regen import ALIGN_PASSES, row_affine_align

    source = Image.open(SOURCE_PHOTO).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)
    clay = Image.open(WORK / f"clay_{LABEL}.png").convert("RGBA")
    clay_record = measure(str(WORK / f"clay_{LABEL}.png"), "clay", LABEL)
    mesh = _load_mesh(REPO / "out/laurent-bust-redo/e11_2mv_reg_hq")

    raws = sorted(WORK.glob(f"{LABEL}_seed*_raw.png"))
    if not raws:
        raise SystemExit(f"no saved raw draws for {LABEL} under {WORK}")
    print(f"reprocessing {len(raws)} saved draws for {LABEL}")

    entry: Dict[str, Any] = {"label": LABEL, "azimuth_deg": AZIMUTH,
                             "attempts": [], "accepted": False,
                             "reprocessed_from_saved_raws": True}
    best: Optional[Dict[str, Any]] = None
    for raw_path in raws:
        seed = raw_path.stem.split("seed")[1].split("_")[0]
        row: Dict[str, Any] = {"seed": int(seed), "raw_path": str(raw_path)}
        try:
            data = raw_path.read_bytes()
            row["raw_md5"] = hashlib.md5(data).hexdigest()
            generated = Image.open(io.BytesIO(data))
            if generated.width > generated.height:
                generated = generated.crop(
                    (generated.width - generated.height, 0,
                     generated.width, generated.height))
            matted = remove_background_robust(generated)
            matted, registration = register_matte_to_clay(
                matted, clay, scale_candidates=WIDE_SCALES, shift_range=0.14)
            row["registration"] = registration
            row["silhouette_iou"] = round(silhouette_iou(matted, clay), 4)
            aligns = []
            for align_pass in range(ALIGN_PASSES):
                pre_path = WORK / f"re_{LABEL}_{seed}_pre{align_pass}.png"
                matted.save(pre_path)
                pre_record = measure(str(pre_path), "rgba", LABEL)
                if align_pass == 0:
                    row["offsets_before_align"] = offsets(
                        pre_record, clay_record, shared_frame=True)
                warped, align = row_affine_align(matted, pre_record, clay_record)
                if align is None:
                    break
                matted = warped
                aligns.append(align)
                if abs(align["slope"] - 1.0) < 0.01 and abs(align["intercept"]) < 2.0:
                    break
            row["row_affine"] = aligns or None
            row["silhouette_iou_aligned"] = round(silhouette_iou(matted, clay), 4)
            matted, spec_fraction = suppress_specular_highlights(
                matted, source_rgba=source)
            row["specular_suppressed_fraction"] = round(spec_fraction, 4)
            matted, tone = match_tone_lab(matted, source)
            row["tone_match"] = tone
            candidate_path = WORK / f"re_{LABEL}_{seed}.png"
            matted.save(candidate_path)
            view_record = measure(str(candidate_path), "rgba", LABEL)
            offs = offsets(view_record, clay_record, shared_frame=True)
            row["offsets_vs_clay"] = offs
            measurable = {name: value.get("registered_frac")
                          for name, value in offs.items()
                          if value.get("registered_frac") is not None
                          and name != "glasses_row"}
            row["measured_features"] = measurable
            worst = max((abs(v) for v in measurable.values()), default=None)
            row["worst_offset_frac"] = worst
            required = ("nose_base", "mouth", "chin")
            iou_after = row.get("silhouette_iou_aligned", row["silhouette_iou"])
            row["passed"] = bool(all(name in measurable for name in required)
                                 and worst is not None and worst < ACCEPT_FRAC
                                 and iou_after >= 0.75)
            row["pixels"] = matted
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["passed"] = False
        entry["attempts"].append({k: v for k, v in row.items() if k != "pixels"})
        print(json.dumps({k: row.get(k) for k in
                          ("seed", "silhouette_iou", "silhouette_iou_aligned",
                           "worst_offset_frac", "passed", "error")}), flush=True)
        if row.get("passed") and "pixels" in row:
            best = row
            break
        if "pixels" in row and row.get("worst_offset_frac") is not None:
            def rank(candidate: Dict[str, Any]) -> tuple:
                iou = candidate.get("silhouette_iou_aligned") or 0.0
                return (iou >= 0.75, -candidate["worst_offset_frac"])

            if best is None or rank(row) > rank(best):
                best = row

    if best is not None and "pixels" in best:
        out_path = OUT_DIR / f"{LABEL}.png"
        best["pixels"].save(out_path)
        entry["accepted"] = bool(best.get("passed"))
        entry["shipped"] = {"path": str(out_path), "seed": best["seed"],
                            "raw_md5": best.get("raw_md5"),
                            "worst_offset_frac": best.get("worst_offset_frac"),
                            "passed": best.get("passed")}
        entry["pose"] = estimate_azimuth(str(out_path), mesh, AZIMUTH)
        print(f"[{LABEL}] shipped seed {best['seed']} worst "
              f"{best.get('worst_offset_frac')} pose {entry['pose']['estimated_deg']}")

    report_path = OUT_DIR / "viewgen_fixed_report.json"
    report = json.loads(report_path.read_text())
    report["angles"] = [e for e in report.get("angles", [])
                        if e.get("label") != LABEL] + [entry]
    report_path.write_text(json.dumps(report, indent=1))
    print("report:", report_path)


if __name__ == "__main__":
    main()
