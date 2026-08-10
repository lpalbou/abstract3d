#!/usr/bin/env python
"""Viewgen audit — stage 3b: supplemental recipe arms on side_left.

  L9   klein separate-reference edit: photo as the PRIMARY image, clay as a
       SECOND reference (multi_reference mode) — tests whether identity
       survives when the clay stops sharing the person's canvas.
  L10  klein composite, seed 1011 — is the composite identity flip seed
       luck or systematic?
  L11  default-route composite (owner=None, empty request = the exact
       set B / loop-regen call): confirms which backend the production
       loop run actually used and measures its anatomy.

Sequential; same measurement as the main ladder.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO = Path("/Users/albou/tmp/abstractframework/abstract3d")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT = Path("/tmp/viewgen_audit/ladder")
CLAY_PATH = Path("/tmp/viewgen_audit/e11_clays/e11_clay_side_left.png")
SOURCE_PHOTO = Path("/tmp/laurent_bust_crop.png")
SUBJECT_NOUN = "man wearing blind face"
SEED = 11
PROVIDER = {"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-9b-8bit"}

SEPARATE_REF_PROMPT = (
    "Two reference images: the first is a photo of a man wearing dark "
    "glasses; the second is an untextured gray model of the SAME man seen "
    "from his left side profile. Produce a real photograph of the man from "
    "the first image, in exactly the pose, framing and silhouette of the "
    "second image: seen from his left side profile. It is the SAME person: "
    "same age, same skin, same dense dark beard, same dark curly hair, same "
    "black t-shirt. Keep his facial proportions exactly: eyebrows, nose "
    "base, mouth and chin at the same heights as in the model. Plain dark "
    "background, soft diffuse lighting."
)


def main() -> None:
    from PIL import Image

    from abstract3d.reference_generation import (
        DEFAULT_NEGATIVE_PROMPT,
        _view_prompt,
        default_i2i_generator,
        register_matte_to_clay,
        silhouette_iou,
    )
    from abstract3d.segmentation import remove_background_robust
    from viewgen_audit_analyze import _load_mesh, estimate_azimuth, offsets
    from viewgen_audit_ladder import build_composite, tone_stats
    from viewgen_audit_measure import measure

    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE_PHOTO).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)
    clay = Image.open(CLAY_PATH).convert("RGBA")
    composite_bytes = build_composite(source, clay, panel=768)

    photo_buffer = io.BytesIO()
    source.convert("RGB").save(photo_buffer, format="PNG")
    photo_bytes = photo_buffer.getvalue()
    clay_buffer = io.BytesIO()
    clay.convert("RGB").save(clay_buffer, format="PNG")
    clay_bytes = clay_buffer.getvalue()

    composite_prompt = _view_prompt("side_left", SUBJECT_NOUN, "composite")

    arms = [
        {"id": "L9_separate_refs_steps12", "prompt": SEPARATE_REF_PROMPT,
         "image": photo_bytes,
         "kwargs": {"steps": 12, "reference_images": [clay_bytes], **PROVIDER}},
        {"id": "L10_composite_seed1011", "prompt": composite_prompt,
         "image": composite_bytes, "seed": 1011,
         "kwargs": {"steps": 12, **PROVIDER}},
        {"id": "L11_default_route_composite", "prompt": composite_prompt,
         "image": composite_bytes,
         "kwargs": {"steps": 8}},  # empty request: the exact loop-regen call
    ]

    generator = default_i2i_generator(None)
    mesh = _load_mesh(REPO / "out/laurent-bust-redo/e11_2mv_reg_hq")
    clay_record = measure(str(CLAY_PATH), "clay", "side_left")

    results_path = OUT / "ladder2_results.json"
    results = []
    for arm in arms:
        started = time.perf_counter()
        seed = int(arm.get("seed", SEED))
        call_kwargs: Dict[str, Any] = {"negative_prompt": DEFAULT_NEGATIVE_PROMPT}
        call_kwargs.update(arm["kwargs"])
        row: Dict[str, Any] = {"id": arm["id"], "seed": seed,
                               "kwargs": {k: ("<bytes>" if k == "reference_images" else v)
                                          for k, v in arm["kwargs"].items()}}
        print(f"=== {arm['id']}", flush=True)
        try:
            payload = generator(arm["prompt"], arm["image"], seed=seed, **call_kwargs)
            data = bytes(payload) if isinstance(payload, (bytes, bytearray)) else None
            if data is None and isinstance(payload, dict):
                for key in ("data", "bytes", "content"):
                    if isinstance(payload.get(key), (bytes, bytearray)):
                        data = bytes(payload[key])
                        break
            if data is None:
                raise RuntimeError(f"no image bytes (payload {type(payload)})")
            row["raw_md5"] = hashlib.md5(data).hexdigest()
            (OUT / f"{arm['id']}_raw.png").write_bytes(data)
            generated = Image.open(io.BytesIO(data))
            row["raw_size"] = list(generated.size)
            if generated.width > generated.height:
                generated = generated.crop(
                    (generated.width - generated.height, 0,
                     generated.width, generated.height))
            matted = remove_background_robust(generated)
            matted, registration = register_matte_to_clay(matted, clay)
            row["registration"] = registration
            row["silhouette_iou"] = round(silhouette_iou(matted, clay), 4)
            out_path = OUT / f"{arm['id']}.png"
            matted.save(out_path)
            view_record = measure(str(out_path), "rgba", "side_left")
            row["offsets_vs_clay"] = offsets(view_record, clay_record, shared_frame=True)
            row["landmarks"] = {k: view_record.get(k) for k in
                                ("top", "glasses_row", "nose_tip", "nose_base",
                                 "mouth", "chin", "shoulder")}
            row["pose"] = estimate_azimuth(str(out_path), mesh, 90.0)
            row["tone"] = tone_stats(matted, source)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = round(time.perf_counter() - started, 1)
        print(json.dumps({k: row.get(k) for k in
                          ("id", "seconds", "raw_md5", "raw_size",
                           "silhouette_iou", "error")}), flush=True)
        results.append(row)
        results_path.write_text(json.dumps(results, indent=1))
    print("done:", results_path)


if __name__ == "__main__":
    main()
