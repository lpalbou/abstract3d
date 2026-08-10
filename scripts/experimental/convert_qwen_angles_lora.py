#!/usr/bin/env python
"""Rewrite the fal Qwen-Image-Edit-2511 multiple-angles LoRA into a key style
the installed mflux (0.17.5) Qwen LoRA mapping actually matches.

Integration gap (viewgen bench, 2026-07-21): the shipped file uses
diffusers-PEFT keys `transformer.transformer_blocks.{b}.<sub>.lora_A|B.weight`.
mflux's QwenLoRAMapping knows `diffusion_model.<...>.lora_A.weight`,
`transformer_blocks.<...>.lora_up|down.weight`, and the ComfyUI underscore
style — but NOT the bare `transformer.` prefix with `lora_A/B`; result:
"Applied to 0 layers (0/1680 keys matched)" and a silently LoRA-less arm.

This converter renames `transformer.` -> `diffusion_model.` (byte-identical
tensors). Coverage: 12 of the LoRA's 14 target submodules map; `img_mod.1`
and `txt_mod.1` (the modulation projections) have NO LoRATarget in mflux's
mapping and stay unapplied — recorded, not hidden (240 of 1680 tensors).
PEFT metadata says lora_alpha == r == 16, so alpha/rank scaling is 1.0 and
omitting alpha tensors preserves the trained effect scale exactly.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

SRC = Path.home() / "Library/Caches/mflux/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors"
DST = Path.home() / "Library/Caches/mflux/loras/qwen-image-edit-2511-multiple-angles-lora-mfluxkeys.safetensors"


def main() -> None:
    import mlx.core as mx

    # mlx handles bfloat16 natively (numpy does not; safetensors' numpy AND
    # mlx safe_open frameworks both route through numpy dtypes and choke).
    # mx.load reads safetensors directly and dtype-faithfully — mflux itself
    # loads LoRAs the same way.
    tensors_in = mx.load(str(SRC))
    tensors = {}
    for key, value in tensors_in.items():
        new_key = key
        if key.startswith("transformer."):
            new_key = "diffusion_model." + key[len("transformer."):]
        tensors[new_key] = value
    mod_keys = [k for k in tensors if ".img_mod." in k or ".txt_mod." in k]
    print(f"tensors: {len(tensors)}; modulation tensors mflux cannot map: "
          f"{len(mod_keys)} (kept in file; loader reports them unmatched)")
    mx.save_safetensors(str(DST), tensors, metadata={
        "converted_from": SRC.name,
        "conversion": "transformer.* -> diffusion_model.* (mflux 0.17.5 "
                      "QwenLoRAMapping key-style gap)",
        "note": "img_mod.1/txt_mod.1 submodules have no mflux LoRATarget",
    })
    print("wrote", DST)


if __name__ == "__main__":
    sys.exit(main())
