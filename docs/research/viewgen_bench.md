# Viewgen recipe bench — model + LoRA + prompt for same-subject angle views (2026-07-21)

Adversarial empirical bench of the image-to-image view-generation lane (S2).
Context: the operator ruled the freestanding i2i view generation the weak
input of the photo-to-3D pipeline (20–25° pose errors, identity flips, a
synthesized side view with parted lips the photo does not have, inherited by
the 3D model). Question benched: **which local model + LoRA + prompt recipe
produces views of the SAME subject at REQUESTED angles?**

Everything below is measured. Harness: `scripts/viewgen_bench.py` (reusable;
fixed seeds; provider pinned local per call). Artifacts:
`out/bust-viewbench/<arm>/<slot>[_raw].png`, `measurements.json` (full
per-view metrics, prompts, seeds, md5s, LoRA application reports, visual
verdicts), `contact_sheet_<arm>.png`.

PRIVACY: the subject is the operator's own face. Every generation ran on the
local mlx-gen route (`provider="mlx-gen"` + explicit model pinned on every
call); the harness raises before generating if binding resolution returns a
non-mlx backend and re-asserts `metadata.source == "mlx-gen"` after every
call. Zero remote calls were made by this bench.

## 0. Arms

Source photo `/tmp/laurent_bust_crop.png` (front, closed mouth, opaque dark
flat-top sunglasses, dense dark beard, black t-shirt). All arms 1024px, klein
steps 8 / qwen steps 4 (Lightning), guidance = model default (klein 1.0,
qwen 1.0 with Lightning). Slots: side65_left(+65), side65_right(−65),
profile_left(+90), profile_right(−90), back(180); seeds 11–15 fixed per slot.

| arm | model (mlx-gen) | LoRA (scale) | prompt |
|---|---|---|---|
| A | flux.2-klein-9b-8bit | — | production rotate template (`_geometry_view_prompt`, noun "man wearing blind face"; seeds 52025/53025/54025 = the production rotate-lane seeds). Vocabulary has **no 65° phrase** → 65-slots unsupported (finding, not omission) |
| B | flux.2-klein-9b-8bit | Flux2-Klein-9B-consistency-V2 (1.0) | angle phrase + explicit consistency clause ("IDENTICAL person: identical face, identical closed mouth … identical dark flat-top sunglasses, identical short dark curly hair, identical dense dark beard, identical black t-shirt") |
| C | flux.2-klein-9b-8bit | multiple-angles-flux-klein-9b (1.0) (= ModelScope `alding86/klein9b-Camera-Blocking`; card empty) | camera-command convention ("Rotate the camera N degrees to the left.") + consistency clause; convention chosen by probe (§3) |
| D | qwen-image-edit-2511-8bit | fal 2511 multiple-angles (1.0, **key-converted**, §4) + 2511-Lightning-4steps (1.0) | fal `<sks> [azimuth] [elevation] [distance]` tokens; 65° NOT expressible (45° lattice) — declared 45 recorded |
| E | flux.2-klein-9b-8bit | consistency V2 (1.0) | arm B prompt + clay clause; **photo as primary edit image + e20-mesh clay render at the target azimuth as a second `reference_images` entry** (the production "identity" conditioning layout) |

## 1. Instruments

* **Pose honesty** — `viewgen_audit_analyze.estimate_azimuth` verbatim
  (production `register_matte_to_clay` full-bust registration + head-band
  IoU, ±55° declared-centered sweep, 2.5° refine) against
  `out/laurent-bust-redo/e20_fixed_views/geometry.glb`. This is the exact
  instrument behind the operator's 65/−67.5/180 baseline; cross-validated
  this session on `viewgen_fixed`: **+65.0 / −67.5 / +170.0**. Bench
  additions: mirrored-declaration sweep (side-flip flag) and sweep-boundary
  censoring. Two "better" rulers (full-circle sweep; head-anchored
  neck-normalized IoU) were built, self-tested exact on known-pose clays,
  and **rejected on real generations**: hair-mass silhouette ambiguity
  produces spurious far-angle maxima (e.g. a visually clean right profile
  reading −30, a production back reading +227). Recorded as instrument
  dead-ends; the declared-window audit ruler stands.
* **Row consistency** — `view_consistency.row_profile` +
  `row_alignment_score` / `best_row_shift` (±18% window) vs the source
  photo, both in subject-bbox frames.
* **Subject fidelity (general)** — overlapping-region comparison on
  head-height-normalized bands (glasses / nose / mouth+chin) defined once in
  source head units and mapped through each image's own top/chin anchors:
  per-row edge-energy and dark-fraction profile correlations + LAB deltas.
  The parted-mouth check is ONE INSTANCE: `mouth_dark_excess` = interior
  dark-fraction peak(view) − peak(source) in the mouth band. Eyes remain the
  binding instrument: every view was inspected at full resolution and the
  verdict recorded (`visual` in measurements.json).
* **Matte** — background RGB std (raw), foreground fraction,
  semi-transparent fraction, `remove_background_robust` extractability.

## 2. The table

|err| = |declared − measured| deg (audit ruler). Fidelity: mouth = interior
mouth-band dark excess vs source (+ = darker gap than source has). chromaR =
foreground chroma ratio vs photo (1.0 = faithful).

| arm/slot | decl | meas | err | row shift | mouth Δdark | glasses | chromaR | bg std | sec | visual verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A/profile_left | +90 | +40.0 | 50.0 | 0.002 | −0.02 | yes | 0.93 | 1.5 | 128 | 3/4 view sold as profile; mirror-bright lens |
| A/profile_right | −90 | −55.0 | 35.0 | 0.045 | −0.09 | yes | 1.41 | 2.5 | 127 | **FAIL: parted lips** (source closed); strap on head |
| A/back | 180 | +182.5 | 2.5 | 0.016 | — | — | 0.77 | 1.5 | 142 | **FAIL: fabricated head-mount box + strap** |
| B/side65_left | +65 | +45.0 | 20.0 | 0.091 | −0.06 | yes | 1.09 | 1.6 | 180 | same man, closed lips; bg remnant patch |
| B/side65_right | −65 | **−65.0** | **0.0** | 0.127 | −0.10 | yes | 1.28 | 6.3 | 228 | same man, closed lips |
| B/profile_left | +90 | +70.0 | 20.0 | 0.106 | −0.09 | yes | 1.26 | 5.2 | 252 | same man, closed lips; grayer beard at zoom |
| B/profile_right | −90 | −75.0 | 15.0 | 0.180 | +0.01 | yes | 1.33 | 8.4 | 233 | closed lips; lens semi-transparent (drift); matte holes in curls |
| B/back | 180 | +190.0 | 10.0 | 0.065 | — | — | 0.66 | 6.2 | 228 | hallucinated earring; hair straighter |
| C/side65_left | +65 | +35.0 | 30.0 | −0.110 | +0.07 | yes | 0.97 | 0.6 | 156 | under-rotated |
| C/side65_right | −65 | −32.5 | 32.5 | 0.180 | −0.23 | yes | 1.04 | 1.6 | 169 | under-rotated |
| C/profile_left | +90 | +47.5 | 42.5 | −0.113 | +0.13 | yes | 0.79 | 2.6 | 261 | under-rotated; gray beard streaks |
| C/profile_right | −90 | −52.5 | 37.5 | 0.127 | −0.32 | yes | 1.12 | 16.8 | 312 | under-rotated; goggle-shaped glasses |
| C/back | 180 | +190.0 | 10.0 | 0.088 | — | — | 0.49 | 1.0 | 315 | strap band echo |
| D/side65_left | +45* | 0.0 | 45.0 | 0.111 | +0.13 | yes | 1.07 | 0.6 | 174 | **FAIL: identity drift** (younger, beard→stubble); near-frontal |
| D/side65_right | −45* | −12.5 | 32.5 | 0.084 | −0.04 | yes | 1.22 | 0.7 | 303 | drifted, near-frontal |
| D/profile_left | +90 | +30.0 | 60.0 | 0.180 | −0.14 | yes | 1.57 | 0.6 | 323 | under-rotated; hallucinated forearm tattoo |
| D/profile_right | −90 | −37.5 | 52.5 | 0.019 | 0.00 | yes | 0.82 | 0.6 | 341 | **FAIL: side flip** (mirror sweep wins) |
| D/back | 180 | +160.0 | 20.0 | 0.133 | — | — | 1.29 | 1.5 | 343 | **FAIL: full standing body fabricated** from bust source |
| E/side65_left | +65 | +60.0 | **5.0** | 0.168 | −0.29 | yes | 1.17 | 4.6 | 282 | same man, closed lips; skin bumps; clay fragment at edge |
| E/side65_right | −65 | −37.5† | 27.5† | 0.171 | −0.11 | yes | 1.34 | 4.4 | 338 | same man, closed lips; wrap-style glasses; head pitched down like clay |
| E/profile_left | +90 | **+90.0** | **0.0** | 0.180 | −0.19 | yes | 1.42 | 0.5 | 293 | same man, closed lips; skin bumps |
| E/profile_right | −90 | −80.0 | 10.0 | 0.178 | −0.07 | yes | 1.24 | 0.5 | 318 | same man, closed lips; wrap-style lens |
| E/back | 180 | +175.0 | **5.0** | 0.093 | — | — | 0.67 | 0.8 | 441 | cleanest back of the bench (no headset fabrication) |

\* D's `<sks>` vocabulary is a 45° lattice: 65° is not expressible; the
nearest token (front quarter, 45°) was declared and scored honestly.
† Instrument caveat: E/side65_right copied the clay's slight downward head
pitch; the elevation-0 sweep confounds pitch with azimuth (eye reads
~−65…−75, sweep top −37.5 by 0.014 IoU over −55). Kept as measured.

Per-arm aggregate (mean |err| over produced slots / hard fails / mean s per
1024px view on shared MPS):

| arm | mean pose err | worst | hard fails | mean sec |
|---|---|---|---|---|
| A baseline | 29.2° (3 slots; no 65° vocabulary) | 50.0° | 2 of 3 (parted lips; fabricated head mount) | 132 |
| B consistency LoRA | 13.0° | 20.0° | 0 | 224 |
| C klein angles LoRA | 30.5° | 42.5° | 0 (pose useless at 65/90) | 243 |
| D qwen 2511 + angles | 42.0° | 60.0° | 3 of 5 (identity drift; side flip; full-body fabrication) | 297 |
| **E = B + clay reference** | **9.5°** | 27.5° | **0** | 334 |

## 3. Arm C trigger discovery (the empty model card)

`multiple-angles-flux-klein-9b.safetensors` embeds ModelScope metadata
(`repoId: alding86/klein9b-Camera-Blocking`, "klein镜头调度"); the HF card
(Alexali/multiple-angles-flux2K) is empty. Probe on profile_left (seed 13,
steps 8, LoRA applied 288/288 keys): camera-command wording → +47.5;
fal-style `<sks> left side view eye-level shot medium shot` → +55.0. Neither
reaches profile; the command convention was kept for the arm run (closer to
the ModelScope family's documented usage). Conclusion: the LoRA biases
composition and keeps identity stable but does NOT deliver requested
azimuths on the edit route at scale 1.0.

## 4. Integration gaps found (each verified live)

1. **fal Qwen-2511 angles LoRA does not load in mflux 0.17.5** — the file
   ships diffusers-PEFT keys (`transformer.transformer_blocks.*.lora_A/B`);
   `QwenLoRAMapping` matches `diffusion_model.*`, `lora_up/down`, ComfyUI
   underscore styles, but not this one: **0/1680 keys applied, silently**
   (the bench's first D run generated LoRA-less pixels; only the per-load
   stdout report reveals it). Fix shipped:
   `scripts/experimental/convert_qwen_angles_lora.py` rewrites keys →
   1440/1680 apply (720 layers). Residual: `img_mod.1`/`txt_mod.1`
   modulation submodules have **no LoRATarget in mflux at all** (240
   tensors unapplied) — a possible cause of D's residual pose weakness, not
   fixable without an mflux mapping extension.
2. **72-poses LoRA is structurally incompatible with klein-9b** — it targets
   FLUX.2-dev (32B). mflux name-matches 256/276 keys, then **crashes
   mid-generation** on matmul shape (4096 vs 6144×16). mflux validates
   LoRA dimensions only at run time; never point this file at klein.
3. **LoRA request plumbing works end-to-end** (no code change needed):
   capability `i2i(..., lora_adapters=[{"source": <path>, "scale": 1.0}])`
   → `ImageEditRequest.lora_adapters` → `_ensure_model_impl(lora_paths=...,
   lora_scales=...)`. Verified by mflux's own load report (consistency V2:
   224/224 keys, 144 layers). `generate_reference_views` forwards
   `image_request` keys verbatim into the generator call, so production can
   carry LoRAs TODAY via `image_request`.
4. **Asset metadata is stripped by the capability i2i** (returns bare
   bytes): the LoRA application report never reaches the caller. The bench
   kept the `GeneratedAsset` (via the backend + `VisionManager` directly)
   to record `lora_paths`/`edit_mode`/seed evidence. Production acceptance
   reports would benefit from the same (P2).
5. **Production 65° vocabulary gap**: `_view_phrase` names only
   45/90/135/180-class angles; `parse_generation_angles` accepts
   `label:65,0` but the phrase falls through to a meaningless
   "seen from the <label> view". Arm A therefore cannot even ASK for the
   65° views the pipeline's own fixed set was measured to contain.

## 5. Failure notes per arm

* **A (baseline, production rotate lane)** reproduces the operator's
  complaint exactly: 35–50° under-rotation on profiles, a parted-lips
  profile_right (the fabricated-expression class, seed 53025), and a
  fabricated head-mounted box on the back view (echo of the source photo's
  headset cable). The back pose itself is honest (2.5°). Framing is the
  most source-faithful of all arms (row shifts ≤0.045) because the rotate
  template pins "same distance as the input photo".
* **B (consistency LoRA)** eliminates the identity flip the audit measured
  on every LoRA-less klein composite arm, keeps mouths closed on all five
  slots, halves the pose error (13° mean), and holds accessories except one
  semi-transparent lens on profile_right. Prompt-wording iteration
  (`B_prompt_iter`): anatomical-destination wording ("his right eye
  completely hidden behind his nose…") improved side65_left 20°→15° but
  WORSENED profile_left 20°→37.5° — wording effects are per-angle noise, not
  a lever; the durable pose lever is the clay reference (E).
* **C (klein multi-angle LoRA)** under-rotates everything (30–42.5° at
  side/profile slots); adds aging drift (gray beard/hair streaks) and one
  goggle-shaped glasses hallucination. Not competitive.
* **D (Qwen-2511 + angles)**: even after the key fix, near-frontal poses at
  45° tokens, one mirrored side, a full standing body fabricated from a
  bust crop (the `<sks>` distance token reframes hard), and consistent
  identity softening (younger, beard thinned). The 2511 base +
  Lightning-4step + angles stack is unusable for this subject class as
  configured. Untested variants that could rescue it: no-Lightning 20-step
  runs (5× cost), lower angles-LoRA scale, or an mflux mapping extension
  for the modulation layers.
* **E (B + clay identity-reference)** is the winner: mean pose error 9.5°
  (0.0 at profile_left, 5° at side65_left and back), zero hard fails, all
  mouths closed, best-in-bench back cleanliness. Costs: mesh surface noise
  leaks into skin as acne-like bumps (the e20 mesh's vertex roughness),
  glasses drift toward a wrap/sport shape on right-side views, occasional
  white clay fragments at frame edges (crop/matte them), and the clay's
  head pitch is copied (see † above). All are downstream-manageable
  (production tone match + matte + acceptance redraws); none is an
  identity/expression fail.

## 6. Winning recipe (exact)

```
model        mlx-gen : AbstractFramework/flux.2-klein-9b-8bit  (PIN EXPLICITLY — privacy)
lora         ~/Library/Caches/mflux/loras/Flux2-Klein-9B-consistency-V2.safetensors  scale 1.0
layout       photo = PRIMARY edit image; clay render of the current mesh at the
             target azimuth = second reference image (production "identity"
             conditioning; edit_mode=multi_reference)
steps        8       guidance  model default (1.0; do NOT add CFG — audit L5/L6)
size         1024    seeds     fixed per slot, recorded
prompt       "<angle phrase>. It is the SAME man as in the input photo, IDENTICAL
             person: identical face, identical closed mouth with lips together and
             a calm neutral expression, identical dark flat-top sunglasses,
             identical short dark curly hair, identical dense dark beard,
             identical black t-shirt. Real photograph, plain dark background,
             soft diffuse even lighting, sharp focus. A second reference image
             shows an untextured gray 3D model of the SAME man in exactly the
             requested pose: match that pose, framing and silhouette exactly,
             and keep his facial features at the same heights as in the model."
angle phrases  65°: "seen from his left/right side at about 65 degrees from
             frontal, a strong three-quarter view with most of the far half of
             his face hidden"
             90°: "seen in exact left/right side profile, a 90 degree side view,
             only the left/right side of his face visible"
             back: "seen directly from behind, the back of his head and his
             shoulders, no part of his face visible"
```

### Production integration (`reference_generation.py`) — how to call it

No code change required for the core recipe; it maps onto existing knobs:

```python
views, report = generate_reference_views(
    mesh, source_rgba,
    conditioning="identity",          # photo primary + clay reference (the E layout)
    person_policy="proceed",
    angles=[("side65_left", 65.0, 0.0), ("side65_right", -65.0, 0.0),
            ("side_left", 90.0, 0.0), ("side_right", -90.0, 0.0),
            ("back", 180.0, 0.0)],
    image_request={                   # forwarded verbatim into every i2i call
        "provider": "mlx-gen",        # REQUIRED: pins the local route (privacy)
        "model": "AbstractFramework/flux.2-klein-9b-8bit",
        "lora_adapters": [{"source": "<loras>/Flux2-Klein-9B-consistency-V2.safetensors",
                            "scale": 1.0}],
    },
    prompt_suffix=("It is the SAME man as in the input photo, IDENTICAL person: "
                   "identical face, identical closed mouth with lips together and a "
                   "calm neutral expression, identical dark flat-top sunglasses, "
                   "identical short dark curly hair, identical dense dark beard, "
                   "identical black t-shirt."),
    steps_schedule=(8, 8, 8), render_size=1024,
)
```

What SHOULD change in abstract3d (ranked):

1. **65°-class vocabulary** (P1): add named phrases for the ~65° views
   (`side65_left/right`) — or synthesize the phrase from the azimuth for
   `label:az,el` entries — so the pipeline can request the angles its own
   conditioning set actually uses. Today `_view_phrase` falls through to
   "seen from the <label> view".
2. **Identity-template expression pinning** (P1): the "identity" template's
   person clause pins age/skin/hair but NOT expression; append "identical
   closed mouth, calm neutral expression" (measured: closed mouths on 10/10
   B+E slots with this wording; A/profile_right without it fabricated
   parted lips — the exact incident class).
3. **Pose acceptance gate** (P1): gate each accepted view on
   `estimate_azimuth` |declared−measured| (suggest ≤15°, redraw otherwise)
   — today only silhouette-IoU gates pose, and it passes 40°-off views
   (audit §2, reconfirmed here). E passes 4/5 slots at ≤10° as-is.
4. **Mouth-band screen** (P2): the general overlapping-region agreement
   (mouth_dark_excess vs source) as a cheap parted-lips screen; visual
   confirmation stays the verdict (the numeric alone missed A/profile_right
   because the mustache shadow already saturates the band's dark fraction —
   recorded limitation).
5. **LoRA evidence surfacing** (P2): capability i2i strips asset metadata
   (gap 4) — production acceptance reports should record
   `lora_applied_file_count`/`edit_mode` per attempt so a silently-unloaded
   LoRA (gap 1's class) is visible in provenance.

## 7. Reproduce

```bash
cd /Users/albou/tmp/abstractframework/abstract3d
PY=/Users/albou/tmp/abstractframework/.venv/bin/python

nice -n 10 $PY scripts/viewgen_bench.py --smoke          # LoRA wiring proof (35 s)
nice -n 10 $PY scripts/viewgen_bench.py --probe-c        # arm-C trigger probe
$PY scripts/experimental/convert_qwen_angles_lora.py     # arm-D key conversion
nice -n 10 $PY scripts/viewgen_bench.py --arms A,B,C,D   # ~2.5 h shared MPS
nice -n 10 $PY scripts/viewgen_bench.py --make-e B       # winner + clay guidance
$PY scripts/viewgen_bench.py --arms A,B,C,D,E --measure-only   # re-measure, no spend
```

MPS discipline: all generations sequential and `nice`d; observed 1024px
walltimes: klein 8-step ≈ 128–252 s (no LoRA … LoRA + 2 refs), qwen-2511
4-step + 2 LoRAs ≈ 167–343 s. Every image's seed, prompt, md5 and LoRA
application report is in `measurements.json`.
