# Volume-assembly performance fix: independent adversarial audit (auditor #2)

Date: 2026-07-22 (night of the e22v2 incident). Scope: the `_AdaptiveVolumeDecoder`
assembly rewrite in `src/abstract3d/backends/hunyuan3d_runtime.py` (landed
2026-07-22 ~04:12, class sha256[:16] `ef2475350c2146d9`; pre-patch baseline =
git HEAD `ade2ddd`, class sha `09bb63210091d0d5`), its benchmark
(`scripts/experimental/bench_adaptive_decoder_cpu.py`), and the incident
attribution. Everything below was read or measured independently; nothing was
taken from the patch author's notes without reproduction.

## 1. The hypothesis under attack, and its math

**Hypothesis as briefed:** pass-2 geometry at `--mc-resolution 512` spent ~2 h
in volume→mesh because of per-chunk `np.concatenate` over a growing array
(quadratic copying at 512³).

### 1.1 The accumulation pattern actually in the code

Read directly, both pre- and post-patch, in three places:

- Backend `_AdaptiveVolumeDecoder._query_points`: `outputs.append(...)` +
  **one** `torch.cat(outputs, dim=0)` after the loop. Linear.
- Vendored `VanillaVolumeDecoder` (`~/.cache/abstract3d/vendor/hunyuan3d21/`
  `82920d…/hy3dshape/.../volume_decoders.py`, sha1 `798952d6…`, untouched by
  the patch): `batch_logits.append(...)` + one `torch.cat(batch_logits, dim=1)`.
  Linear.
- Vendored `HierarchicalVolumeDecoding` / `FlashVDMVolumeDecoding`: same
  list-append + single-cat shape. Linear.

**There is no concatenate-per-chunk-over-a-growing-array anywhere in the
decode path.** The hypothesis fails at the code-reading stage.

### 1.2 The closed form, computed anyway

If the hypothesized pattern existed at the run's real parameters
(`octree_resolution=512` → dense grid `513³ = 135,005,697` vertices;
`num_chunks = 32768` = `_DEFAULT_MPS_NUM_CHUNKS`, confirmed in the bundle
metadata; accumulated dtype float32 — each chunk is converted to fp32 on CPU
before accumulation):

- chunk count `n = ceil(513³ / 32768) = 4121` (the bench docstring's "4122"
  is an off-by-one typo; its code computes 4121),
- bytes copied by concat-per-chunk = `4 B × c × Σ_{k=1}^{n} k`
  `= 4 × 32768 × n(n+1)/2` `= 4 × 32768 × 8,493,381` **≈ 1.113 TB**.

Bandwidth, measured on this machine tonight (M5 Max, 128 GB, `nice -n 10`,
**system load average ≈ 111** — all absolute numbers below are
contention-degraded and stated as such):

| measurement | result |
|---|---|
| `np.copyto` 1 GiB fp32 | best 22.6 GB/s written, median 17.0 |
| `np.copyto` 128 KiB fp32 (cache) | best 22.6 GB/s |
| repeated `np.concatenate` growing pattern, n=512 chunks | 8.4 s → 2.0 GB/s effective |
| repeated concat, n=1024 | 27.3 s → 2.5 GB/s effective |
| repeated concat, n=2048 | 245.0 s → 1.1 GB/s effective (load spike visible) |

Plug-in: 1.113 TB / 22.6 GB/s ≈ **49 s**; at the *pessimal measured*
effective concat rate (1.1 GB/s, allocator churn + tonight's contention)
≈ **17 min**. Direct quadratic extrapolation from the measured n=2048 point:
245 s × (4121/2048)² ≈ 16.5 min.

**Verdict (deliverable a): the math cannot explain 2 hours.** Even granting
the hypothesized pattern its worst measured bandwidth, it predicts ~1–17
minutes — and the pattern is not in the code at all (the actual pattern
copies 513³ × 4 B = 0.54 GB once: sub-second).

### 1.3 What the stack sample actually shows

`/tmp/e22v2_sample.txt` (pid 2726, sampled 02:34:03, 2,252 × 1 ms samples,
main thread). My independent tally (top-of-chain attribution):

| symbol | samples | share |
|---|---|---|
| `NI_ZoomShift` (scipy `ndimage.zoom`) | 764 | 33.9 % |
| `PyArray_Nonzero` (`argwhere`/`nonzero`) | 424 | 18.8 % |
| `array_reshape` (copy of the 27N×3 index matrix) | 196 | 8.7 % |
| `NI_BinaryErosion(+2)` (`binary_dilation`) | 184 | 8.1 % |
| `mapiter_set` (fancy-index scatter) | 164 | 7.3 % |
| `LONG_minimum` (`np.minimum` int64 clamp) | 144 | 6.4 % |
| **`array_concatenate`** | **19** | **0.8 %** |

Every hot symbol maps one-to-one onto consecutive lines of the pre-patch
refine block (reshape of `base+block`, `np.minimum` clamp, `fine_mask[...]`
scatter, `argwhere`, `binary_dilation`, `ndimage.zoom`). The sample is the
refinement **bookkeeping**, not concatenation. (The patch author's bench
header quotes the same breakdown; the only figure I could not reproduce is
"mapiter_set 12 %" — I measure 7.3 %.)

A second sample (`/private/tmp/python3_2026-07-22_022008_sDiX.sample.txt`,
02:20:08) shows the main thread 88 % inside `scaled_dot_product_attention`
on MPS, blocked on Metal command-buffer creation (queue full) — i.e. the
GPU-side transformer work (DiT step or decode cross-attention query), plus
`pow`/`gelu` kernels. Zero numpy bookkeeping in that window. Together the
samples show a normal decode: mostly accelerator-bound queries, with a
CPU-visible bookkeeping slice between query phases.

### 1.4 The corrected attribution of the "2 hours": the machine slept

The wall-clock arithmetic of the run
(`out/bust/e22_oneshot_v2/metadata.json`, `/tmp/e22v2_run.log`, terminal
record, pmset):

- Process launched 23:31:32; bundle written 03:15:36 → wall 3 h 44 m.
- `timings_s` is a **sum of stage timers** (verified at line 4611:
  `total_s = preprocess + pass1 + views + inference + mesh + …`), not a
  span: 8,132.8 s ≈ 2 h 16 m. Timed stages + the separately-recorded load
  leave **≈ 5,269 s (~88 min) of wall time uncovered**.
- `time.perf_counter` on macOS is `CLOCK_UPTIME_RAW` — it **stops during
  system sleep**, so sleeping wall time is invisible to every stage timer.
- `pmset -g log` for the window: the machine entered **Deep Idle** shortly
  after the 2mv model load finished (00:35:35); DarkWakes at 00:41:50 (4 s),
  00:46:55 (45 s), 01:04:01 (119 s), 01:22:56 (95 s), 01:39:40 (197 s),
  01:58:48 (169 s); **full wake at 02:18:33 from keyboard UserActivity**.
  ≈ 97 min asleep minus ≈ 10.5 min of dark-wake slivers ≈ **87 min of true
  suspension — matching the 88-min timer hole to within a minute or two.**
- The reconstruction then closes exactly: pass-2 `pipeline()` started
  ≈ 00:35:36, ran ≈ 6 min before sleep, progressed only in dark-wake
  slivers, resumed at the 02:18:33 keyboard wake (the operator checking on
  it — which is also when the two samples were taken, 02:20 and 02:34),
  and finished ≈ 02:42:51 after its honest 2,433 s of awake time; mesh
  postprocess 195 s → 02:46:06; texture 1,770 s → 03:15:36 = the bundle
  write timestamp.

So "~2 h in volume→mesh" conflated: ≈ 87 min of **machine sleep** + ≈ 41 min
of real pass-2 (50 DiT steps + adaptive decode + marching cubes, exactly as
timed) that *looked* like one silent multi-hour decode because the decode
printed nothing (the vanilla decoder has a tqdm; the adaptive one had no
logging at all — fixed by the patch). The patch docstring's idle-sleep
attribution is **independently confirmed**. The run was `nice`d but not
`caffeinate`d; a GPU/CPU compute job takes no power assertion, so the
display-sleep → system-sleep chain proceeded normally once the operator
walked away.

Real but secondary contributors, in order: the accelerator cross-attention
queries themselves (the decode's actual cost, bounded inside the 2,433 s
pass-2 timer), the single-threaded host bookkeeping at 513³ (seconds per
level uncontended — see §3; minutes under that night's contention), and the
per-chunk `.to("cpu")` MPS sync (4,121 round-trips at dense-512 scale; not
measurable here under the no-GPU rule — flagged, not quantified).

## 2. Equivalence harness (deliverable b)

`tests/test_volume_assembly_equivalence.py` — independent of the patch
author's tests. The OLD path is embedded verbatim from git HEAD `ade2ddd`
(verified byte-identical to the pre-patch worktree before the patch landed);
the NEW path is the real production class imported at test time; the
vendored `VanillaVolumeDecoder`/`HierarchicalVolumeDecoding` are imported
from the pinned source snapshot the backend actually runs
(`~/.cache/abstract3d/vendor/hunyuan3d21/82920d64…`). Equality is asserted
on grid **bytes** (`tobytes()`, NaN-safe), dtype/shape contracts, and the
**chunk trace** — the exact `(shape, dtype)` sequence of geo-decoder calls,
because a changed chunk partition is invisible to a pointwise synthetic
decoder but changes real cross-attention numerics.

Results against the landed patch (2026-07-22, 31 passed / 1 xfailed, 29 s):

| case | result |
|---|---|
| kernel: empty input (0 points) | fp32 ✓ fp16 ✓ |
| kernel: single point, chunk > N | fp32 ✓ fp16 ✓ |
| kernel: exact chunk multiple (8³ = one chunk) | fp32 ✓ fp16 ✓ |
| kernel: non-divisible chunks (512/100, 512/7) | fp32 ✓ fp16 ✓ (×2) |
| kernel: 33³ odd cube, partial tail | fp32 ✓ fp16 ✓ |
| kernel: 65³ at the production chunk size 32768 | fp32 ✓ fp16 ✓ |
| kernel: C-order vs F-order points | ✓ |
| kernel: non-contiguous (stride-2 view) | ✓ |
| kernel: fp32 logits under fp16 queries preserved bit-exact | ✓ (with anti-vacuousness guard) |
| adaptive: two refinement levels, fp32, clamp-firing radius | ✓ |
| adaptive: two levels, fp16 latents, production chunk size | ✓ |
| adaptive: fp32-logit/fp16-query autocast shape, whole decoder | ✓ |
| adaptive: mc_level = 0.15 | ✓ |
| adaptive: odd 33 → single dense level branch | ✓ |
| adaptive: small grid ≤ coarse → dense branch | ✓ |
| adaptive: all-outside field → dense fallback branch | ✓ |
| adaptive: all-inside field → empty-surface ones fallback | ✓ |
| vendored Vanilla vs frozen replica: r8 fp32 non-divisible | ✓ |
| vendored Vanilla: r33 fp16 odd | ✓ |
| vendored Vanilla: r16 **batch=2** (cat dim=1 semantics) | ✓ |
| vendored Hierarchical vs dense reference | xfail as documented (upstream int-cast breakage confirmed live) |
| adaptive vs vendored dense: sign agreement + exact refined band | ✓ |

Every grid byte and every chunk trace matched. The patch author's own 7
decoder tests in `test_hunyuan3d_backend_unit.py` also pass (independently
executed).

## 3. Benchmark methodology audit (deliverable c)

Target: `scripts/experimental/bench_adaptive_decoder_cpu.py` (author:
adversary #1). Audited against the four briefed questions:

1. **Real chunk size?** Yes — `NUM_CHUNKS = 32768` matches
   `_DEFAULT_MPS_NUM_CHUNKS` and the bundle metadata. (Docstring says "4122
   chunks"; the code correctly computes 4121.)
2. **Dtype conversions and `.cpu()` transfer pattern included?** Partially.
   Part 2 accumulates CPU fp32 tensors — the fp16→fp32 per-chunk conversion
   and the MPS→CPU per-chunk sync are **not** in the measurement. Under the
   CPU-only audit rule they cannot be measured here either; flagged as the
   one cost this bench cannot see. Part 2 also appends the *same* tensor
   object per iteration (cache-hot source; production chunks are distinct
   allocations) — immaterial at the 0.01 s scale of the linear patterns, but
   worth naming.
3. **"Before" measured on the same allocation pattern?** Yes for the real
   comparison: (a1) list+cat-once *is* the shipped old pattern, and part 1
   replicates the OLD loop op-by-op and cross-checks the real decoder
   end-to-end with `np.array_equal`. The hypothesized quadratic (a2) is
   correctly kept separate.
4. **Extrapolation soundness?** The one real flaw: (a2) fits the quadratic
   constant on n ∈ {128, 256, 512} — working sets ≤ 67 MB, cache/SLC-
   adjacent — then extrapolates ×64 to n = 4121, where the running array is
   540 MB and DRAM/allocator-bound. My direct measurements at n = 1024/2048
   show the effective rate degrading 2.0 → 1.1 GB/s, so the small-n fit
   **understates** the quadratic cost roughly an order of magnitude (their
   printed "~23 s"; measured-slope extrapolation ≈ 9–17 min under tonight's
   load). This does not change any verdict — both figures are ≪ 2 h and ≫
   the actual 0.01 s linear pattern — but small-n quadratic fits should be
   labeled lower bounds.

Re-run of the bench on this machine (`nice -n 10`, load ≈ 111 — absolute
numbers are upper bounds; relative old/new comparisons remain meaningful as
both sides see the same contention):

- Part 1, hard-shell field (129³→257³→513³, refine sets 903,922 and
  3,626,194 points = 2.7 % of 135 M): legacy bookkeeping **12.1 s** total —
  `ndimage.zoom` alone 1.3 + 7.6 = 8.9 s, `argwhere` 1.2 s, the 27N-row
  matrix 1.0 s, dilation 0.4 s. Patched decoder end-to-end **9.5 s**, grid
  **byte-identical to the legacy replica: True**.
- Part 1, soft field (band blow-up, refine 10.05 M points at the 512
  level): legacy bookkeeping 10.1 s; patched end-to-end 10.5 s;
  **byte-identical: True**. The new progress logs render correctly
  (bounded, ≈ 20 cadence lines per query phase).
- Part 2 at dense-512 scale (135 M fp32, 4,121 chunks): current list+cat
  **0.025 s**; prealloc numpy 0.015 s; prealloc torch 0.030 s; one final
  `np.concatenate` 0.015 s — the linear patterns are all equivalent-cost
  noise. Hypothesized repeated concat measured 0.042/0.160/0.720 s at
  n = 128/256/512 → their extrapolation prints **46.7 s** (this run;
  their earlier run printed ~23 s). My independent direct measurement at
  n = 2048 (245 s) extrapolates to ≈ 17 min — the small-n fit understates
  ~20× (cache vs DRAM+allocator regime), exactly the flaw named above.
  Either bound is ≪ 2 h.
- Part 3 equality probes: all **equal: True** (slice-OR shell, per-offset
  scatter 0.127 s vs 27N matrix 0.337 s ≈ 2.7× — compatible with their
  2.2× claim; nonzero-tuple gather; contiguous-tuple scatter). The
  `astype(copy=False)` zoom variant measured *slower* in this run (10.2 vs
  8.6 s) — pure contention noise on a value-identical no-op elision; its
  benefit is the elided 540 MB copy, not wall time under load.

Bookkeeping verdict: the CPU bookkeeping at true 512 scale is **~10 s
class** (uncontended it would be less), against a pass-2 decode phase whose
timer-bounded cost is minutes of accelerator queries. The patch docstring's
"~5 s" is the right order of magnitude; the sampled 02:34 window caught the
513³ `zoom`/`argwhere` slice stretched by post-wake contention (`nice`
process, dozens of agent processes resuming), not a dominant cost.

## 4. Hazards that would make a naive preallocation patch WRONG (deliverable d)

None of these fire in the landed patch (the harness proves it), but each is
a real trap for the "obvious" rewrite, with the harness case that catches it:

1. **fp16 accumulator truncation.** The geo decoder can emit fp32 logits
   even for fp16 queries (autocast shape). The old path upcasts each chunk
   to fp32 *before* accumulating; preallocating the buffer in
   `latents.dtype` (fp16) truncates those bits.
   → `test_query_points_preserves_fp32_logits_under_fp16_queries` (the test
   also proves its own non-vacuousness: the values do NOT survive a fp16
   round-trip).
2. **Last-partial-chunk arithmetic.** `513³ mod 32768 = 1537`: a
   preallocation sized `n_chunks × chunk` instead of `N` leaves a garbage
   tail or breaks the reshape. → the `partial-*`, `cube33`, `cube65`
   kernel cases (odd cubes, non-divisor chunks).
3. **Empty-input edge.** Zero points → zero chunks; the old path returns
   `torch.zeros((0,), fp32)`. A prealloc-and-fill that assumes ≥1 chunk
   crashes or returns uninitialized memory. → `empty-fp32/fp16`.
4. **Batch dimension in the vendored vanilla decoder.** Accumulation is
   `cat(dim=1)` with batch as dim 0; a flat prealloc that appends along
   dim 0 and reshapes scrambles batch>1. → `r16-batch2`.
5. **Memory order.** The old path routes points through
   `np.ascontiguousarray`; dropping it changes what `torch.from_numpy`
   sees for F-ordered/strided inputs. → `f-order`, `non_contiguous_view`.
6. **Chunk-partition drift.** Any "optimization" that changes chunk
   boundaries produces bit-different logits from the *real* cross-attention
   decoder (batched matmul reduction order) even when a synthetic pointwise
   field matches. → every case asserts the recorded chunk trace, not just
   the grid.
7. **Adaptive-branch divergence.** The adaptive field is trilinear fill
   away from the surface BY DESIGN; asserting adaptive == dense globally is
   wrong, and a patch reviewer who "fixes" the fill to match dense changes
   shipped behavior. The right contract is sign-agreement everywhere plus
   exact values on the refined band. → `test_adaptive_vs_vendored_dense_sign_agreement`.
8. **Fallback branches.** The no-interior dense fallback and the
   empty-surface `ones_like` fallback are behavior, not dead code; a
   rewrite that restructures the loop can silently drop them.
   → `all_outside_dense_fallback`, `all_inside_ones_fallback`.

## 5. Verdict summary

- The quadratic-concatenate hypothesis is **disproven three independent
  ways** (code reading, sample tally at 0.8 %, closed-form ≈ 1–17 min ≪ 2 h).
- The ~2 h was **≈ 87 min of macOS idle sleep** (pmset-proven, arithmetic
  closes to the minute) on top of a legitimately-timed 41-min pass-2; the
  decode was additionally *silent*, which is what made it look hung.
- The landed patch (per-offset scatter, `np.nonzero`, `astype(copy=False)`,
  progress logging) is **bit-exact equivalent** to the old path across 31
  independent cases including every branch, both dtypes, odd sizes,
  non-divisible chunks, and the vendored decoders. It changes memory
  transients and observability, not values.
- The bench's one methodological flaw (small-n quadratic extrapolation
  understating the hypothetical cost ~10×) does not affect any conclusion.
