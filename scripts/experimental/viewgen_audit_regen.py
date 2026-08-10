#!/usr/bin/env python
"""Viewgen audit — stage 4: regenerate the BEST-RECIPE view set.

The ladder verdict (docs/research/viewgen_audit.md): identity preservation
and clay-geometry adherence trade off across EVERY available route — the
local klein composite follows the clay to ~4% but paints a different person
(systematic across seeds); every identity-preserving route (rotate,
separate-refs, the remote default editor) places the face 6-32% of head
height away from the clay's rows after whole-bust registration. No raw
generation passes a <2% per-feature line, so the best RECIPE is:

  identity-preserving generation (the same default route the production
  loop run actually used) + production registration + FEATURE-ANCHORED
  row-affine alignment to the clay guide + the <2% acceptance gate,
  re-drawing on failure.

Renders fresh clay guides from the e11 mesh (same wiring as
regen_views_from_mesh.py), generates, aligns, measures, and accepts a view
only when every measurable facial feature lands within ACCEPT_FRAC of the
clay guide (back views: crown row + silhouette IoU — no facial features
exist from behind). The row-affine is part of the recipe and is recorded
per view (slope/intercept), never silent.

Output: out/laurent-bust-redo/viewgen_fixed/{side_left,side_right,back}.png
        + viewgen_fixed_report.json (offsets, pose estimates, provenance).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path("/Users/albou/tmp/abstractframework/abstract3d")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT_DIR = REPO / "out/laurent-bust-redo/viewgen_fixed"
WORK = Path("/tmp/viewgen_audit/regen")
SOURCE_PHOTO = Path("/tmp/laurent_bust_crop.png")
SUBJECT_NOUN = "man wearing blind face"
# The DEFAULT route — the same binding the production loop run (set B)
# resolved with its empty image_request: the configured OpenAI-compatible
# editor. Identity-preserving where every local-klein arm flipped the
# person (ladder L1-L10 vs L11/set-B, measured). Deliberate operator-key
# spend, same route the pipeline itself defaults to.
PROVIDER: Dict[str, Any] = {}
ACCEPT_FRAC = 0.02  # |offset| < 2% of clay head height, per feature
ANGLES = (("side_left", 90.0), ("side_right", -90.0), ("back", 180.0))

RECIPE: Dict[str, Any] = {"steps": 12}  # dropped by the remote route; keeps
                                        # a local fallback at the L3 winner
SEEDS = (11, 1011, 2011, 3011, 4011, 5011)  # remote editor ignores seeds: these
                                        # are DRAWS; variance is the bottleneck
IDENTITY_CLAUSE = (
    " Keep the person's exact facial proportions from the left photo: the "
    "same head size, and the eyebrows, nose base, mouth line and chin at "
    "exactly the same heights on the face as in the left photo and as in "
    "the right panel model. Do not enlarge or shrink any facial feature."
)
USE_IDENTITY_CLAUSE = True
ALIGN_FEATURES = ("nose_tip", "nose_base", "mouth", "chin")
# Wide but not unbounded: the remote editor reframes the face by up to
# ~1.3x run to run (measured round 2); beyond 1.35 the row stretch would
# visibly smear texture, so such draws are rejected instead of warped.
SLOPE_BOUNDS = (0.78, 1.35)
ALIGN_PASSES = 3  # warp -> re-measure -> residual warp (re-detection moves
                  # contour features a few px per pass; iterate to converge)


def _warp_rows(image_rgba: Any, source_rows: Any) -> Any:
    """Resample image rows: output row y takes input row source_rows[y]
    (linear interpolation, RGBA together; out-of-frame rows go transparent)."""
    import numpy as np
    from PIL import Image

    array = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32)
    height = array.shape[0]
    low = np.clip(np.floor(source_rows).astype(int), 0, height - 1)
    high = np.clip(low + 1, 0, height - 1)
    frac = np.clip(source_rows - low, 0.0, 1.0).astype(np.float32)
    out = (array[low] * (1.0 - frac)[:, None, None]
           + array[high] * frac[:, None, None])
    out[(source_rows < 0) | (source_rows > height - 1)] = 0.0
    return Image.fromarray((out + 0.5).astype(np.uint8), "RGBA")


def row_affine_align(image_rgba: Any, view_record: Dict[str, Any],
                     clay_record: Dict[str, Any]) -> tuple[Any, Optional[Dict[str, Any]]]:
    """Face-zone piecewise row alignment onto the clay guide.

    FACE ZONE (rows above the clay chin + margin): least-squares affine
    view_row = a*clay_row + b over the detected ALIGN_FEATURES (>= 3),
    slope clamped to SLOPE_BOUNDS — puts nose/mouth/chin on the clay's
    rows. BODY ZONE (below the clay shoulder): identity — the whole-bust
    registration already placed the torso, and a global affine measured
    -0.11 IoU dragging it off. Cosine blend between the zones (the neck
    absorbs the stretch; monotonicity asserted, fold -> refuse).

    Back views (no facial features): two-point exact fit on (crown,
    shoulder) rows instead, same piecewise body treatment.
    """
    import numpy as np

    if view_record.get("label") == "back" or clay_record.get("label") == "back":
        pairs = [(clay_record["top"], view_record["top"]),
                 (clay_record["shoulder"], view_record["shoulder"])]
        fit_features = ["top", "shoulder"]
    else:
        pairs = [(clay_record[name], view_record[name]) for name in ALIGN_FEATURES
                 if clay_record.get(name) is not None and view_record.get(name) is not None]
        fit_features = [name for name in ALIGN_FEATURES
                        if clay_record.get(name) is not None
                        and view_record.get(name) is not None]
        if len(pairs) < 3:
            return image_rgba, None
    clay_rows = np.asarray([p[0] for p in pairs], dtype=np.float64)
    view_rows = np.asarray([p[1] for p in pairs], dtype=np.float64)
    a, b = np.polyfit(clay_rows, view_rows, 1)
    if not (SLOPE_BOUNDS[0] <= a <= SLOPE_BOUNDS[1]):
        return image_rgba, None

    height = np.asarray(image_rgba.convert("RGBA")).shape[0]
    rows = np.arange(height, dtype=np.float64)
    face_map = a * rows + b
    body_map = rows
    chin = clay_record.get("chin") or clay_record["shoulder"]
    head_h = max(1, (clay_record.get("chin") or clay_record["shoulder"]) - clay_record["top"])
    face_end = float(chin + 0.10 * head_h)
    body_start = float(clay_record["shoulder"])
    if body_start <= face_end + 4:
        body_start = face_end + max(8.0, 0.05 * height)
    t = np.clip((rows - face_end) / (body_start - face_end), 0.0, 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * t)
    source_rows = face_map * (1.0 - blend) + body_map * blend
    if np.any(np.diff(source_rows) <= 0):
        return image_rgba, None  # fold: refuse rather than ship artifacts
    aligned = _warp_rows(image_rgba, source_rows)
    return aligned, {"slope": round(float(a), 4), "intercept": round(float(b), 1),
                     "fit_features": fit_features,
                     "face_end_row": int(face_end),
                     "body_start_row": int(body_start)}


def main() -> None:
    import numpy as np
    from PIL import Image

    from abstract3d.reference_generation import (
        DEFAULT_NEGATIVE_PROMPT,
        _view_prompt,
        clay_silhouette,
        default_i2i_generator,
        match_tone_lab,
        register_matte_to_clay,
        silhouette_iou,
        suppress_specular_highlights,
    )
    from abstract3d.rendering import get_last_render_backend, render_mesh_views
    from abstract3d.segmentation import remove_background_robust
    from viewgen_audit_analyze import _load_mesh, estimate_azimuth, offsets
    from viewgen_audit_ladder import build_composite, tone_stats
    from viewgen_audit_measure import measure

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE_PHOTO).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)
    mesh = _load_mesh(REPO / "out/laurent-bust-redo/e11_2mv_reg_hq")
    generator = default_i2i_generator(None)

    only = set(sys.argv[1:])  # optional angle filter: re-run one angle and
    report_path = OUT_DIR / "viewgen_fixed_report.json"  # merge into report
    if only and report_path.exists():
        report = json.loads(report_path.read_text())
        report["angles"] = [entry for entry in report.get("angles", [])
                            if entry.get("label") not in only]
    else:
        report = {
            "recipe": dict(RECIPE),
            "identity_clause": USE_IDENTITY_CLAUSE,
            "provider": dict(PROVIDER),
            "acceptance": f"per-feature |registered offset| < {ACCEPT_FRAC:.0%} of clay head height",
            "seeds": list(SEEDS),
            "angles": [],
        }
    for label, azimuth in ANGLES:
        if only and label not in only:
            continue
        clay = render_mesh_views(mesh, size=768, azimuths=[azimuth],
                                 elevation=0.0)[0].convert("RGBA")
        if get_last_render_backend() != "moderngl":
            raise RuntimeError("clay renderer must be moderngl")
        clay_path = WORK / f"clay_{label}.png"
        clay.save(clay_path)
        clay_record = measure(str(clay_path), "clay", label)
        composite = build_composite(source, clay, panel=768)
        prompt = _view_prompt(label, SUBJECT_NOUN, "composite")
        if USE_IDENTITY_CLAUSE:
            prompt += IDENTITY_CLAUSE

        entry: Dict[str, Any] = {"label": label, "azimuth_deg": azimuth,
                                 "attempts": [], "accepted": False}
        best: Optional[Dict[str, Any]] = None
        for seed in SEEDS:
            started = time.perf_counter()
            row: Dict[str, Any] = {"seed": seed}
            try:
                call_kwargs: Dict[str, Any] = dict(PROVIDER)
                call_kwargs["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
                call_kwargs.update(RECIPE)
                payload = generator(prompt, composite, seed=seed, **call_kwargs)
                data = bytes(payload) if isinstance(payload, (bytes, bytearray)) else None
                if data is None and isinstance(payload, dict):
                    for key in ("data", "bytes", "content"):
                        if isinstance(payload.get(key), (bytes, bytearray)):
                            data = bytes(payload[key])
                            break
                if data is None:
                    raise RuntimeError("generator returned no image bytes")
                row["raw_md5"] = hashlib.md5(data).hexdigest()
                # Persist every raw draw: the remote editor has no seed
                # control, so an unpersisted good draw is unrecoverable.
                (WORK / f"{label}_seed{seed}_raw.png").write_bytes(data)
                generated = Image.open(io.BytesIO(data))
                if generated.width > generated.height:
                    generated = generated.crop(
                        (generated.width - generated.height, 0,
                         generated.width, generated.height))
                matted = remove_background_robust(generated)
                matted, registration = register_matte_to_clay(matted, clay)
                row["registration"] = registration
                row["silhouette_iou"] = round(silhouette_iou(matted, clay), 4)
                # Feature-anchored row alignment (the recipe's second half):
                # measure the registered draw, fit rows onto the clay's rows,
                # warp, re-measure, repeat while a usable fit remains (the
                # contour re-detection shifts by a few px per pass).
                aligns = []
                for align_pass in range(ALIGN_PASSES):
                    pre_path = WORK / f"{label}_seed{seed}_pre{align_pass}.png"
                    matted.save(pre_path)
                    pre_record = measure(str(pre_path), "rgba", label)
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
                row["silhouette_iou_aligned"] = round(
                    silhouette_iou(matted, clay), 4)
                # Production post-processing order (specular suppression +
                # capped tone match) so the shipped pixels match what the
                # bake would consume.
                matted, spec_fraction = suppress_specular_highlights(
                    matted, source_rgba=source)
                row["specular_suppressed_fraction"] = round(spec_fraction, 4)
                matted, tone = match_tone_lab(matted, source)
                row["tone_match"] = tone
                candidate_path = WORK / f"{label}_seed{seed}.png"
                matted.save(candidate_path)
                view_record = measure(str(candidate_path), "rgba", label)
                offs = offsets(view_record, clay_record, shared_frame=True)
                row["offsets_vs_clay"] = offs
                row["tone"] = tone_stats(matted, source)
                measurable = {
                    name: value.get("registered_frac")
                    for name, value in offs.items()
                    if value.get("registered_frac") is not None
                    and name != "glasses_row"  # cross-extractor bias: dark band
                }                              # vs geometric shelf; report-only
                row["measured_features"] = measurable
                worst = max((abs(v) for v in measurable.values()), default=None)
                row["worst_offset_frac"] = worst
                if label == "back":
                    # No facial features from behind: hold the back to crown
                    # placement + silhouette agreement (post-alignment).
                    head_span = max(1, clay_record["shoulder"] - clay_record["top"])
                    top_off = abs(view_record["top"] - clay_record["top"]) / head_span
                    row["back_top_offset_frac"] = round(top_off, 4)
                    row["worst_offset_frac"] = round(top_off, 4)
                    iou_after = row.get("silhouette_iou_aligned",
                                        row["silhouette_iou"])
                    # 0.75 = the production bake gate; crown row is the one
                    # measurable "feature" a back view has.
                    passed = top_off < ACCEPT_FRAC and iou_after >= 0.75
                else:
                    required = ("nose_base", "mouth", "chin")
                    have_all = all(name in measurable for name in required)
                    iou_after = row.get("silhouette_iou_aligned",
                                        row["silhouette_iou"])
                    passed = bool(have_all and worst is not None
                                  and worst < ACCEPT_FRAC
                                  and iou_after >= 0.75)
                row["passed"] = bool(passed)
                row["candidate_path"] = str(candidate_path)
                row["pixels"] = matted
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["passed"] = False
            row["seconds"] = round(time.perf_counter() - started, 1)
            entry["attempts"].append(
                {k: v for k, v in row.items() if k != "pixels"})
            print(json.dumps({k: row.get(k) for k in
                              ("seed", "seconds", "silhouette_iou",
                               "worst_offset_frac", "passed", "error")}),
                  flush=True)
            if row.get("passed") and "pixels" in row:
                best = row
                break
            if "pixels" in row and row.get("worst_offset_frac") is not None:
                # Fallback ranking: a projectable silhouette (production
                # IoU line) outranks a lower feature offset — a 0.4-IoU
                # face close-up with perfect rows is still unusable.
                def rank(candidate: Dict[str, Any]) -> tuple:
                    iou = candidate.get("silhouette_iou_aligned") or 0.0
                    return (iou >= 0.75, -candidate["worst_offset_frac"])

                if best is None or rank(row) > rank(best):
                    best = row
        if best is not None and "pixels" in best:
            out_path = OUT_DIR / f"{label}.png"
            best["pixels"].save(out_path)
            entry["accepted"] = bool(best.get("passed"))
            entry["shipped"] = {
                "path": str(out_path), "seed": best["seed"],
                "raw_md5": best.get("raw_md5"),
                "worst_offset_frac": best.get("worst_offset_frac"),
                "passed": best.get("passed"),
            }
            pose = estimate_azimuth(str(out_path), mesh, azimuth)
            entry["pose"] = pose
            print(f"[{label}] shipped seed {best['seed']} "
                  f"worst_offset {best.get('worst_offset_frac')} "
                  f"pose {pose['estimated_deg']} (nominal {azimuth})", flush=True)
        report["angles"].append(entry)
        (OUT_DIR / "viewgen_fixed_report.json").write_text(
            json.dumps(report, indent=1))
    print("report:", OUT_DIR / "viewgen_fixed_report.json")


if __name__ == "__main__":
    main()
