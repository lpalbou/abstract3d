#!/usr/bin/env python
"""Viewgen audit — stage 3: the recipe ladder on ONE angle (side_left).

Mirrors `generate_reference_views`' composite call byte-for-byte (same
letterboxed two-panel canvas, same prompt builder, same negative prompt,
same provider route: mlx-gen flux.2-klein-9b-8bit), then varies exactly one
knob per arm:

  L1  steps 8                      (production first-attempt baseline)
  L2  steps 8  + strength=0.5      (probe: verified silently dropped for the
                                    flux2 edit family -> must be IDENTICAL
                                    bytes to L1 or the drop claim is wrong)
  L3  steps 12                     (production alternation arm)
  L4  steps 16                     (does more denoise move anatomy?)
  L5  steps 12 + guidance 2.5      (flux2 edit allows guidance override;
                                    real CFG vs empty negative prompt)
  L6  steps 12 + guidance 4.0
  L7  rotate conditioning, steps 8 (set A lane recipe: photo only, no clay)
  L8  steps 12 + identity/proportion prompt clause (prompt-delta A/B)

Every arm: seed 11 (the loop run's base seed), sequential execution (MPS is
shared), per-arm PNG + feature-offset measurement vs the e11 clay guide +
foreground tone stats vs the source photo.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path("/Users/albou/tmp/abstractframework/abstract3d")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental"))

OUT = Path("/tmp/viewgen_audit/ladder")
CLAY_PATH = Path("/tmp/viewgen_audit/e11_clays/e11_clay_side_left.png")
SOURCE_PHOTO = Path("/tmp/laurent_bust_crop.png")
SUBJECT_NOUN = "man wearing blind face"  # the recorded production caption noun
SEED = 11
PROVIDER = {"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-9b-8bit"}

IDENTITY_CLAUSE = (
    " Keep the person's exact facial proportions from the left photo: the "
    "same head size, and the eyebrows, nose base, mouth line and chin at "
    "exactly the same heights on the face as in the left photo and as in "
    "the right panel model. Do not enlarge or shrink any facial feature."
)


def build_composite(source_rgba: Any, clay: Any, panel: int) -> bytes:
    """Byte-faithful copy of the composite construction in
    generate_reference_views (letterbox both panels onto 16,16,16)."""
    from PIL import Image

    from abstract3d.reference_generation import clay_silhouette

    def letterbox(image: Any) -> Any:
        rgb = image.convert("RGB")
        scale = panel / max(rgb.size)
        new_size = (max(1, int(round(rgb.width * scale))),
                    max(1, int(round(rgb.height * scale))))
        resized = rgb.resize(new_size, Image.LANCZOS)
        box = Image.new("RGB", (panel, panel), (16, 16, 16))
        box.paste(resized, ((panel - new_size[0]) // 2,
                            (panel - new_size[1]) // 2))
        return box

    guide_dark = Image.new("RGB", clay.size, (16, 16, 16))
    guide_dark.paste(clay.convert("RGB"), (0, 0),
                     Image.fromarray(
                         (np.asarray(clay_silhouette(clay)) * 255).astype("uint8")))
    canvas = Image.new("RGB", (panel * 2, panel), (16, 16, 16))
    canvas.paste(letterbox(source_rgba), (0, 0))
    canvas.paste(letterbox(guide_dark), (panel, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def tone_stats(image_rgba: Any, source_rgba: Any) -> Dict[str, Any]:
    """Foreground chroma + luminance vs the photo (clay-gray detector)."""
    import numpy as np
    from skimage import color as skcolor

    def stats(image: Any) -> Dict[str, float]:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)
        chroma = np.hypot(lab[:, :, 1][mask], lab[:, :, 2][mask])
        return {"mean_l": float(lab[:, :, 0][mask].mean()),
                "mean_chroma": float(chroma.mean()),
                "p90_chroma": float(np.percentile(chroma, 90))}

    generated, source = stats(image_rgba), stats(source_rgba)
    return {
        "generated": {k: round(v, 1) for k, v in generated.items()},
        "source": {k: round(v, 1) for k, v in source.items()},
        "chroma_ratio": round(generated["mean_chroma"] / max(source["mean_chroma"], 1e-6), 3),
    }


import numpy as np  # noqa: E402  (used by build_composite)


def main() -> None:
    import hashlib

    from PIL import Image

    from abstract3d.reference_generation import (
        DEFAULT_NEGATIVE_PROMPT,
        _view_prompt,
        default_i2i_generator,
        register_matte_to_clay,
        silhouette_iou,
    )
    from abstract3d.segmentation import remove_background_robust
    from viewgen_audit_measure import measure
    from viewgen_audit_analyze import _load_mesh, estimate_azimuth, offsets

    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE_PHOTO).convert("RGBA")
    if source.getchannel("A").getextrema()[0] == 255:
        source = remove_background_robust(source)
    clay = Image.open(CLAY_PATH).convert("RGBA")
    composite_bytes = build_composite(source, clay, panel=768)
    (OUT / "conditioning_composite.png").write_bytes(composite_bytes)

    rotate_buffer = io.BytesIO()
    source.convert("RGB").save(rotate_buffer, format="PNG")
    rotate_bytes = rotate_buffer.getvalue()

    composite_prompt = _view_prompt("side_left", SUBJECT_NOUN, "composite")
    rotate_prompt = _view_prompt("side_left", SUBJECT_NOUN, "rotate")

    arms = [
        {"id": "L1_steps8", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 8}},
        {"id": "L2_steps8_strength05", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 8, "strength": 0.5}},
        {"id": "L3_steps12", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 12}},
        {"id": "L4_steps16", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 16}},
        {"id": "L5_steps12_guidance25", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 12, "guidance_scale": 2.5}},
        {"id": "L6_steps12_guidance40", "prompt": composite_prompt, "image": composite_bytes,
         "kwargs": {"steps": 12, "guidance_scale": 4.0}},
        {"id": "L7_rotate_steps8", "prompt": rotate_prompt, "image": rotate_bytes,
         "kwargs": {"steps": 8}},
        {"id": "L8_steps12_identity_prompt",
         "prompt": composite_prompt + IDENTITY_CLAUSE, "image": composite_bytes,
         "kwargs": {"steps": 12}},
    ]

    generator = default_i2i_generator(None)
    mesh = _load_mesh(REPO / "out/laurent-bust-redo/e11_2mv_reg_hq")
    clay_record = measure(str(CLAY_PATH), "clay", "side_left")

    results = []
    for arm in arms:
        started = time.perf_counter()
        call_kwargs: Dict[str, Any] = dict(PROVIDER)
        call_kwargs["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
        call_kwargs.update(arm["kwargs"])
        row: Dict[str, Any] = {"id": arm["id"], "kwargs": dict(arm["kwargs"]),
                               "seed": SEED}
        print(f"=== {arm['id']} {arm['kwargs']}", flush=True)
        try:
            payload = generator(arm["prompt"], arm["image"], seed=SEED, **call_kwargs)
            data = bytes(payload) if isinstance(payload, (bytes, bytearray)) else None
            if data is None and isinstance(payload, dict):
                for key in ("data", "bytes", "content"):
                    if isinstance(payload.get(key), (bytes, bytearray)):
                        data = bytes(payload[key])
                        break
            if data is None:
                raise RuntimeError(f"no image bytes (payload type {type(payload)})")
            row["raw_md5"] = hashlib.md5(data).hexdigest()
            raw_path = OUT / f"{arm['id']}_raw.png"
            raw_path.write_bytes(data)
            generated = Image.open(io.BytesIO(data))
            row["raw_size"] = list(generated.size)
            if generated.width > generated.height:  # two-panel echo -> right panel
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
                          ("id", "seconds", "raw_md5", "silhouette_iou", "error")}),
              flush=True)
        results.append(row)
        (OUT / "ladder_results.json").write_text(json.dumps(results, indent=1))

    print("\nladder complete:", OUT / "ladder_results.json")


if __name__ == "__main__":
    main()
