# 0023 — Surface extraction performance: measured decomposition, refuted concatenate theory, Metal marching cubes later

## Status

Proposed (2026-07-22). Original claim from the live e22v2 one-shot run; **problem statement
corrected same day by the measured timing decomposition below** (auditor pass, 2026-07-22
early morning). The observability half of fix 1 shipped; the preallocation half is
**refuted as unnecessary** by measurement. Validation of the after-state is **queued**
(GPU busy at audit time — see Validation below).

## Problem (as measured — supersedes the original claim)

The e22v2 one-shot run (`--geometry-conditioning loop --num-inference-steps 50
--mc-resolution 512`, Apple Silicon MPS, pid 2726) spanned **3h44m11s wall**
(23:31:32.6 → 03:15:43.7 local) with ~2h of log silence in pass 2, and was read as a
volume→mesh stall. The measured decomposition says otherwise:

- **86m41s of the wall was the machine asleep, not the program running.** `pmset -g log`
  records Idle Sleep at 00:41:23 and seven sleep segments (27 + 301 + 981 + 1016 + 909 +
  951 + 1016 s = 5,201 s) with brief DarkWakes between (two re-sleeps citing "Dark Wake
  Thermal Emergency"), until a keyboard-activity full wake at 02:18:33. `time.perf_counter`
  on macOS excludes sleep, which is why the run's own `timings_s` (8,132.8 s summed stages)
  could not be reconciled with the wall span until the sleep log was consulted.
- **The pass-2 pipeline call (diffusion + volume decode + marching cubes) consumed
  2,433.4 s of awake time** (`timings_s.inference`) — 40m33s, not 2 h. Its *wall* span
  (00:35:36 → ~02:42:49) contains the sleep.
- **The claimed quadratic `np.concatenate` accumulation never existed in this decoder.**
  Git HEAD already accumulated `list + torch.cat`. The archived stack sample
  (`/tmp/e22v2_sample.txt`, taken 02:34:03, 2,252 samples) shows `array_concatenate` at
  **0.8%** of the busy window; the hot frames are the refinement loop's *bookkeeping*
  (`NI_ZoomShift` 34%, `PyArray_Nonzero` 19%, binary erosion/dilation 8%, `np.median`
  partition, reshape copies) at 45.7 GB RSS — the decoder working, not stalling. The
  original "~75% duty in array_concatenate" reading is not reproducible from that sample.
- Arithmetic bound (audited independently): even a pathological per-chunk
  `np.concatenate` over 513³ float32 in 4,122 chunks of 32,768 copies ~1.1 TB total
  ≈ **~23 s** at memory bandwidth — two orders of magnitude short of explaining 2 h.
  Bench confirmation: the shipped list+cat assembly costs **0.010 s** at that scale
  (`scripts/experimental/bench_adaptive_decoder_cpu.py`, Part 2).

## Measured stage decomposition (e22v2, BEFORE)

Sources: run log `/tmp/e22v2_run.log` (timestamps + the run's own `timings_s`), terminal
record (start 23:31:32.558, end 03:15:43.663, rc=3 = degraded-verdict exit — the run
**completed**, it was not killed), bundle mtimes (`loop_refgen_attempts.jsonl` 00:34:48;
all exports 03:15:32–36), stack sample 02:34:03, `pmset -g log`. Rows 5–8 wall boundaries
carry ±1 min (sleep-interval placement); awake seconds are exact timer values.

| # | Wall (local) | Stage | Awake time (perf_counter) |
|---|---|---|---|
| 0 | 23:31:33–23:31:57 | startup, imports, HF fetch | ~24 s (untimed) |
| 1 | 23:31:57–23:32:36 | pass-1 flagship DiT load | ~40 s (untimed slot, log-anchored) |
| 2 | 23:32:36–00:06:50 | pass-1 shape: 50-step diffusion + 512³ adaptive decode + MC + postprocess | 2,053.99 s (`pass1_inference`) |
| 3 | 00:06:50–00:34:49 | view synthesis: 3 Klein i2i draws + clays + gates (511.5/684.1/482.2 s per view) | 1,679.3 s (`geometry_view_synthesis`) |
| 4 | 00:34:53–00:35:35 | pass-2 2mv DiT load | 41.83 s (`load`) |
| 5 | 00:35:36–02:42:49± | pass-2 pipeline: 50-step diffusion + 512³ adaptive decode + MC. Wall window contains **all 5,201 s of machine sleep** (00:41:23→02:18:33) | 2,433.43 s (`inference`) |
| 6 | 02:42:49±–02:46:04± | pass-2 mesh postprocess (components, decimation, normals) | 195.30 s (`mesh`) |
| 7 | 02:46:04±–03:15:34± | texture: refgen (3 replayed + 1 generated top view, 463.2 s) + bake + whole-bake A/B | 1,770.28 s (`texture`) |
| 8 | 03:15:34±–03:15:43.7 | self-verification (4.3 s) + renders + exports + bundle writes | ~10–70 s (untimed) |

Consistency check: awake budget = wall 13,451 s − sleep 5,201 s = 8,250 s; timed stages
8,132.8 + 41.8 (load) + ~64 s untimed rows 0–1 ≈ 8,239 s — closes within ~11 s (DarkWake
execution slivers). The pass-1 chain (rows 1–3) closes against the log anchors to <5 s,
so the entire timing anomaly lived in the pass-2 wall window and is fully explained by
sleep.

**Diffusion-vs-decode split inside rows 2 and 5 is not recoverable from e22v2 artifacts**
(the decoder logged nothing then). Bounds for row 5: diffusion ≥ 348 s (the awake sliver
before first sleep is pre-decode), decode+MC ≤ ~2,085 s (≤ ~35 min). The new per-chunk
progress lines exist precisely to measure this split in the after-state.

## What actually shipped as "fix 1" (2026-07-22)

- **Observability, not preallocation**: `_AdaptiveVolumeDecoder` now logs the level
  schedule, per-level refinement sizes, and ~5% chunk-progress cadence at INFO
  (`volume decode [coarse 128³]: chunk N/M (P%)`), bounded at ≤21 lines per query batch.
  Two silent hours in a one-shot pipeline was an operability defect regardless of cause.
- **Assembly left as `list + torch.cat`** — measured optimal (0.010 s; preallocated
  in-place writes were not faster: 0.015 s torch / 0.009 s numpy). The originally
  proposed preallocation is dropped: it targets a cost that does not exist.
- Refinement bookkeeping measured at true 512³ scale by
  `scripts/experimental/bench_adaptive_decoder_cpu.py` (synthetic fields bracketing the
  shipped checkpoint, cross-checked `np.array_equal` against the real decoder): ~5 s
  total on a quiet machine. Note the 02:34:03 sample shows a full 2.25 s window inside
  this bookkeeping at 45.7 GB RSS on a freshly-woken machine — swap-in pressure can
  stretch it; the live after-run's progress-line timestamps are the arbiter.

## Validation (queued — GPU busy at audit time)

GPU check run 2026-07-22 ~04:20: `pgrep -f "abstract3d i23d"` → pid 32731 (live 512³
control run, `out/bust/control_cleanviews_512`, log `/tmp/control512_run.log`) plus a
BlackPixel `mlxgen` wan2.2 video generation (pid 34847) and two pytest suites. **Do not
launch GPU work under this load.** The in-flight control run itself exercises the 512³
decode on this hardware and its log/`timings_s` may serve as first after-evidence
(contended-GPU upper bound; it imported the module at 04:06, possibly predating the
final logging edit at 04:12).

Queued command (run only when `pgrep -f "abstract3d i23d"` is empty and no mlx/klein/
GPU pytest processes are live; `caffeinate` so sleep never contaminates wall time again):

```bash
cd /Users/albou/tmp/abstractframework/abstract3d && \
ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 \
caffeinate -dims nice -n 5 /Users/albou/tmp/abstractframework/.venv/bin/abstract3d i23d \
  out/bust/control_cleanviews/inputs/windowed_front.png \
  --output-dir out/bust/e23_decode_timing_512 \
  --backend hunyuan3d21 --model tencent/Hunyuan3D-2mv --device mps \
  --geometry-conditioning single \
  --texture-reference-image out/bust/control_cleanviews/inputs/windowed_side_left.png \
  --texture-reference-angle side_left \
  --texture-reference-image out/bust/control_cleanviews/inputs/windowed_back.png \
  --texture-reference-angle back \
  --num-inference-steps 50 --octree-resolution 512 \
  --texture-mode none --allow-degraded --seed 2025 \
  > /tmp/e23_decode_timing_512.log 2>&1
```

This is the minimal pass-2-only driver: 2mv, 3 conditioning views, 512³/50, texture
skipped — it isolates load + diffusion + decode + MC + postprocess.

Measurement checklist:

1. Progress lines present: `grep "volume decode" /tmp/e23_decode_timing_512.log` shows
   the level schedule, per-level sizes, and chunk cadence with timestamps.
2. Decode wall = last chunk line − "adaptive volume decode: levels" line (caffeinated,
   so wall == awake).
3. `timings_s.inference` / `timings_s.mesh` from the bundle metadata; decode share from
   step 2 completes the diffusion-vs-decode split the BEFORE table could not give.
4. Compare against BEFORE row 5 (2,433.4 s awake, decode ≤ ~35 min) on the same
   hardware; note GPU contention state of both runs.
5. Bit-exactness: covered CPU-side by the bench (grid `np.array_equal` against the real
   decoder); optional GPU A/B (vanilla vs adaptive at 384³) only if promoted.

## Fixes, ordered by leverage (updated)

1. ~~Preallocated grid assembly~~ **REFUTED — dropped.** Assembly measured at 0.010 s;
   see above. The shipped remainder of this item is the progress logging.
2. **Metal marching-cubes kernel in the torch lane (days).** Unchanged: the field tensor
   is already a torch tensor on MPS — a custom Metal compute op (two-pass: per-cell
   classification + prefix-sum allocation, then triangle emission) plugs in with zero
   device copies, as a `SurfaceExtractors`-style sibling of upstream's CUDA extractors.
   Home: `abstract3d` (NOT mlx-gen — that package is the image-diffusion provider;
   geometry kernels do not belong there). Re-scope AFTER the queued validation measures
   where the awake decode time actually goes (accelerator queries vs MC vs bookkeeping).
3. **MLX custom-kernel MC only with an MLX shape-lane adoption.** Unchanged: MLX's
   `mx.fast.metal_kernel` makes the kernel writable, but bolting MLX onto the torch lane
   costs a framework boundary crossing per run for one stage. If the MLX Hunyuan shape
   port (generative_process_2026.md, ZimengXiong lane) is ever adopted, its extraction
   belongs there; not before.
4. **Operational: long unattended runs must hold a sleep assertion** (`caffeinate -dims`
   wrapper or a power assertion in the CLI for runs expected >30 min). The measured 87
   minutes of the "stall" was idle sleep. Documented in `docs/api.md` and
   `docs/troubleshooting.md` guidance; a built-in assertion is a candidate follow-up.

## Stays CPU deliberately

Component filtering, quadric decimation, vertex merge, normals — graph algorithms over
the ~100k-face extracted mesh; seconds of work; not worth porting.

## Acceptance

- **BEFORE (measured, this document):** pass-2 pipeline awake time 2,433.4 s at 512³/50
  incl. diffusion; decode+MC ≤ ~35 min of it (exact split unmeasured — no progress lines
  existed). Wall-clock appearance of "hours" explained by 86m41s machine sleep.
- **AFTER (queued):** per-chunk progress lines visible in the run log; decode+MC wall
  (caffeinated) measured from those lines; target from the original item — volume→mesh
  under ~10 min at 512³ on this hardware — re-evaluated against the measured split, and
  this line replaced with the measured result when the queued run executes.
- Fix 2 (if promoted after validation): extraction output equivalent to the CPU path
  (same iso-level, vertex/face counts within tolerance) on the golden fixtures, measured
  speedup recorded.
