# Proposed: Hunyuan3D-2.1 flagship MPS shredding — diagnosis ladder

## Metadata

- Created: 2026-07-21
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none directly; touches the backend promotion criteria posture
- ADR impact: none expected (diagnosis + config fix, not a contract change)

## Context

The flagship `tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1` (3.3B) produced shredded
geometry in all 3 bust runs on this machine (e9: euler -17225, non-watertight,
octree 512 / 30 steps / mps / float16 / seed 2025), while `Hunyuan3D-2mv` (1.1B)
at the same device/dtype is healthy. The shredded-field signature (thousands of
film-shell bodies) matches a noise-banded SDF field, i.e. the DiT/VAE output is
corrupted — our adaptive volume decoder is exonerated (measured bit-identical to
upstream dense decoding; `hunyuan3d_runtime.py` `_AdaptiveVolumeDecoder` docstring).

## Evidence gathered (2026-07-21, docs/research/generative_process_2026.md axis 3)

1. The model itself runs correctly on MPS fp16: `VladimirTalyzin/hunyuan3d-2.1-mac`
   generates clean meshes on an M4 Pro (octree 256, 50 steps) with only
   device-string / `.ckpt`-loader / autocast-no-op patches. So our failure is
   environment- or config-specific, not inherent.
2. Our venv torch is 2.10.0 — inside a documented window of SILENT-GARBAGE MPS SDPA
   bugs: non-contiguous q/k/v on the post-2.8 fast path produce garbage
   (pytorch#163597; SDXL-on-MPS black/noise reported on 2.10/2.11, works on 2.9);
   2^32-element score-matrix overflow (pytorch#179352, fixed by #179592);
   2-pass kernel OOB fixed only in 2.11 (pytorch#174861). Both Hunyuan DiTs feed
   SDPA `rearrange`/`transpose`-produced (non-contiguous) tensors; the flagship
   differs from 2mv in head_dim (128 vs 64), MoE layers, and
   `torch.backends.cuda.sdp_kernel(enable_math=False)` wrappers.
3. The flagship DiT is attention-precision-sensitive by demonstration: ComfyUI had
   to force `low_precision_attention=False` because quantized attention produced
   all-NaN DiT output ("3D occupancy prediction requires higher precision at voxel
   boundaries", Comfy-Org/ComfyUI#12772). Upstream `hunyuandit.py` carries the
   comment "TODO: eps should be 1/65530 if using fp16" on its qk_norm.
4. The 2.1 scheduler is exonerated: `schedulers.py` already casts
   timesteps/sigmas to fp32 with an explicit "mps does not support float64" guard.

## Proposed direction (ordered, each step cheap and decisive; stop at first green)

1. **SDPA hygiene patch**: monkeypatch `F.scaled_dot_product_attention` inside the
   flagship pipeline to `.contiguous()` q/k/v and force the math backend; rerun the
   e9 configuration (same photo, seed 2025, 384/30 first — cheaper, family regime).
2. **Torch pin A/B**: same run in a scratch venv with torch 2.9.x (the reported-good
   version) and, if available, a 2.11+ wheel carrying the #179592/#174861 fixes.
3. **bfloat16 DiT**: same memory as fp16, fp32 range — kills the overflow class if
   step 1/2 point at precision rather than kernel dispatch.
4. **fp32 qk_norm/softmax only** (targeted mixed precision) if bf16 is not enough.
5. If all green paths require a torch downgrade we cannot ship: gate the flagship
   backend on torch version with an actionable error, and keep the contiguous+math
   SDPA patch as the vendored-source patch (`_patch_*_source` precedent from
   `step1x_runtime.py`).

## Why it might matter

A working flagship unlocks the strongest local single-view geometry (3.3B vs the
1.1B 2mv) and a better mesh clay for every downstream loop (view generation,
registration, texture). It may also explain other MPS fp16 fragilities in the fleet.

## Promotion criteria

- One reproducible green flagship run on this machine (single body, watertight or
  near, no film-shell debris) at 384/30, then at 512/50, seed-pinned.
- Root cause named with a minimal numerical repro (per repo rule: prove bugs with
  minimal ground-truth tests before fixing) OR an honest "torch-version bug, gated".

## Validation ideas

- A/B the exact e9 inputs; compare euler/body-count/watertightness via the existing
  shape-candidate topology metrics.
- Cross-check the same patch does not regress 2mv (its family regime 384/30).

## Non-goals

- No new backend; no scheduler rewrites; no upstream PRs required for promotion.
