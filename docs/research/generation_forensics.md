# Generation forensics — three defect classes traced to causal inputs (2026-07-21)

Adversarial audit (Adversary H) of three operator-confirmed defects in the
`out/laurent-bust-redo/` bust family. No prior claim was trusted: every verdict
below is re-derived from the shipped artifacts — bundle metadata, the actual
conditioning/reference image bytes, isolated byte-level bake replays, and fresh
renders of the shipped GLBs. Evidence crops:
`out/laurent-bust-redo/review/forensics/` (sheets `D1_*`, `D2_*`, `D3_*`).

Method notes that matter for admissibility:

- **e10's bake was replayed exactly** with the instrumented harness
  (`scripts/experimental/texture_forensics_bake.py`, capture mode): all four
  per-view `coverage_ratio`/`capture_efficiency` rows and the full
  `source_pose` record match the shipped bundle to the 4th decimal; texture
  MAE vs shipped `texture.png` = 0.052/255 with 99.66% of texels within 2/255.
  The captured per-texel blend weights therefore ARE e10's bake decisions,
  and the replay simultaneously **pins the exact input bytes** (a wrong ref
  set cannot reproduce those numbers). This also independently validates the
  instrument the earlier e20 forensics used.
- Row-consistency numbers below come from the pipeline's own production
  instrument (`abstract3d.view_consistency.view_consistency_report`), pointed
  at the exact image pairs each run fed the DiT.
- Mesh renders are of the shipped `scene.glb`/`geometry.glb` (pyrender/
  moderngl only; no generation was run).

---

## Defect 1 — OPEN MOUTH in `e18_windowed` geometry

**Verdict: INPUT-CAUSED (conditioning side view hallucinated a parted-lips
expression; no gate looks at expression, and the caller-reference conditioning
lane runs no gates at all). The DiT fused faithfully; the bake is not involved.**

Evidence chain (`D1_e18_open_mouth_evidence.png`):

1. Ground truth: `/tmp/laurent_bust_crop.png` mouth is CLOSED
   (`d1_photo_mouth.png`).
2. What e18 actually conditioned on (pinned from bytes, not the report):
   - front tag = the FULL `/tmp/laurent_front_clean4.png` — `e18_windowed/
     input.png` still contains subject rows 976–999 that the windowed front
     (`windowed_set/win_laurent_front_clean4.png`, cut_row 975) removes. **The
     windowed front was never fed**; REPORT.md's "windowed set A" is wrong for
     the front view.
   - left tag = `windowed_set/win_texture_reference_generated_side_left.png`,
     whose "window" was a **no-op** (report `cut_row` 728 > subject bottom 704;
     verified RGB byte-identical to
     `hunyuan_multiview/texture_reference_generated_side_left.png`, 0 alpha
     px changed). e18's left conditioning is therefore exactly the set-A
     rotate-lane draw (klein i2i, seed 53025).
   - back tag = `windowed_set/win_texture_reference_generated_back.png`
     (the only genuinely windowed view used; 25,638 alpha px cut).
   - side_right dropped from conditioning by the 3-view cap (metadata
     `postprocess_warnings`), texture-only.
3. The mouth in each conditioning input, at high magnification:
   - front: closed (`d1_front_clean4_mouth.png`);
   - side_left: **lips parted** — separated upper/lower lip with a dark gap
     and a protruding lower lip (`d1_e18cond_sideleft_mouth_x10.png`); the
     parting already exists in the raw first-generation draw
     `hunyuan_multiview/geometry_view_synthesized_side_left.png`
     (`d1_rawA_sideleft_mouth_x5.png`) — origin artifact, 2026-07-20 03:13;
   - back: no mouth content;
   - (side_right, texture-only, is also parted:
     `d1_e18cond_sideright_mouth_x9.png`.)
4. The e18 mesh has parted lips: `d1_e18_mesh_front_mouth.png` (front, open
   slit between lips), `e18_mesh_az90_el0_mouthcrop.png` (profile).
5. Control: e20's regenerated side view (`viewgen_fixed/side_left.png`) has a
   closed mouth (`ctl_viewgenfixed_sideleft_mouth.png`) and e20's mesh drew no
   open-mouth complaint.

Why no gate caught it: the synthesis gate battery
(`hunyuan3d_runtime._synthesize_geometry_views`) checks matte sanity, part-
material ΔE, back mirror-IoU, row consistency, side-pair mirror — **nothing
examines expression/mouth state**, and the rotate prompt
(`_geometry_view_prompt`) contains no expression clause. Worse, e18 fed its
views through the **caller-reference path** (`--texture-reference-image`),
which maps refs onto DiT tags with **zero gating**
(`hunyuan3d_runtime.py` ~2856–2884: snap-to-tag only; compare e12's
metadata, where the auto lane records `row_consistency` per view — e10/e17/
e18/e20/e21 geometry_views records carry none). Run post-hoc, the production
row instrument scores e18's actual pairs `correctable, score 0.00, shift +118
rows (11.5%)` (front↔side_left) and `+108 rows` (front↔back) — the windowing
was defeated by feeding the un-windowed front.

**Minimal fix (implementable as-is):**

1. `_geometry_view_prompt` (hunyuan3d_runtime.py): for person subjects append
   an expression pin — "The face keeps exactly the same neutral expression as
   the input photo: mouth closed, lips together." (The template is slot-free;
   add the clause inside the person branch keyed on
   `is_person_subject`.)
2. Add a mouth-state acceptance check to the view gates for person subjects:
   crop the mouth region (below the detected glasses band / nose base, above
   chin) in source and candidate, and reject candidates whose dark
   inter-lip gap area exceeds the source's by a calibrated factor (the
   viewgen_fixed lane's redraw-until-accepted loop is the shipped precedent
   for per-feature gating; extend its feature set from rows to mouth state).
3. Route caller-provided geometry conditioning refs through the SAME gate
   battery as synthesized ones (one flag: treat `consumer != texture` caller
   refs as gate-subject; see cross-check section).

---

## Defect 2 — BULBOUS NOSE + DOUBLED-LIP PROFILE in `e17_clean4`

**Verdict: INPUT-CAUSED (the accepted side view carries a protruding lip mass
AND sits ~10–16% of subject height too low in the DiT's frame) +
GATE-BYPASS (two of the three conditioning views were consumed despite
recorded gate REJECTIONS). The DiT's double-carve of disagreeing rows is the
mixing mechanism, not the origin.**

Evidence chain (`D2_e17_profile_evidence.png`):

1. e17's conditioning (disclosed in `conditioning_inputs/conditioning_e17.png`):
   front = clean4; back/left tags = `loop_views_e11/loop_view_{back,side_left}.png`
   (set B, generated by `scripts/experimental/regen_views_from_mesh.py` which
   silently routed to the remote editor — gpt-image-1 — per the viewgen audit);
   side_right texture-only (3-view cap).
2. **Provenance violation, verified from mtimes + report content**:
   `loop_views_e11/regen_report.json` (16:23) records `side_left` accepted
   (IoU 0.7773) and **back REJECTED** (2× speculars — the 0019 dark-subject
   miscalibration — + 1 silhouette fail) and **side_right REJECTED** (3×
   silhouette ≤ 0.734). The `loop_view_back.png` (16:31) and
   `loop_view_side_right.png` (16:29) files on disk postdate the report by
   6–8 minutes: a gate-bypassed rerun that wrote NO report. e17 consumed
   them anyway.
3. The accepted side_left itself is defective in two independent ways:
   - **anatomy**: a protruding parted-lip mass
     (`d2_e17cond_loop_sideleft_mouth_x8.png`);
   - **rows**: in the DiT's bbox-recentered frame its mouth sits at
     normalized 0.387 vs the front's 0.299 (~8.8% of subject height low;
     `view_consistency_report` on the exact pair: `correctable, score 0.00,
     best shift +126 rows (12.3%)`; back pair +137 rows). Corroborates the
     viewgen audit's set-B numbers (face +9.5% low, chin +6.6% residual).
4. The e17 mesh profile at az +90 (`e17_mesh_az90_el0_mouthcrop.png`) shows
   the operator's defect: a rounded nose blob and **two stacked lip
   protrusions** with an open notch — geometry consistent with carving the
   front's mouth row AND the side view's lower mouth row (the documented 2mv
   double-carve mechanism), not with either input's silhouette alone. The
   conditioning side view's own nose profile is normal — the DiT did not copy
   a bulbous nose from anywhere; it fused two vertically disagreeing feature
   sets into one merged mass.
5. No gate ran on any of this at conditioning time: caller-reference lane
   (see defect 1); e17's `geometry_views` metadata records only
   `declared_azimuth_deg`/`snap_delta_deg`.

**Minimal fix:**

1. Gate the caller/loop conditioning lane with the existing instrument: run
   `view_consistency_report(source, ref)` on every caller ref that will take
   a DiT tag; reject `inconsistent`, row-align `correctable` (the code path
   exists in `_synthesize_geometry_views`; it needs to be called from the
   caller-reference branch of `generate()` too).
2. Make consumption of rejected views structurally impossible:
   `regen_views_from_mesh.py` must write its report atomically WITH the view
   files (one rerun = one report), and the consumer (the experiment runner /
   any future loop lane) must verify `accepted: true` per label before
   passing a file as a conditioning ref. Today nothing links
   `loop_view_back.png` to any acceptance record.
3. Land the 0019 speculars fix (absolute floor `L > max(median+60, ~90)`) so
   dark-subject back views stop failing for lit skin — the bypass pressure
   that created (2) disappears with the false rejections.

---

## Defect 3 — LEFT-SIDE WHITE SMEAR in `e10_2mv_registered_refs`

(The smear reads "left" in the operator's back/back-left renders; on the
subject it is the anatomical RIGHT side — canonical frame y<0 — of the
head/jaw/neck. 10,621 texels bright + desaturated in that zone.)

**Verdict: BAKE-CAUSED (completion stack invented pale content over a
witness-starved region); the starvation itself is INPUT-CAUSED (the
"side_right" reference is a ~−52° three-quarter view declared −90°, so its
registration collapsed and it painted almost nothing). NOT a projected white
background — that hypothesis is refuted below.**

Evidence chain (`D3_e10_white_smear_evidence.png`):

1. **What the bake actually consumed** (pinned by exact replay, see method
   note): source = `/tmp/laurent_bust_crop.png` matted by
   `remove_background_robust` (e10 `input.png` is byte-identical to the raw
   crop, md5 `8ea471c9…`); refs = `hunyuan_multiview/
   texture_reference_generated_{back,side_left,side_right}.png` at
   180/+90/−90, all synthesized-flagged by filename inference. The
   `aligned_views/` harmonized variants were NOT used (their headless/
   head-only crops don't reproduce the bundle numbers; several are
   catastrophically cropped — see cross-check).
2. **The bake decision for the smear region** (captured per-texel weights):

   | painter | smear texels won | color it painted | final texture there |
   |---|---|---|---|
   | NO VIEW (fill) | 4,289 (40%) | — | (214,192,181) pale |
   | back | 2,804 | (46,42,42) dark hair | (226,211,199) pale |
   | side_right | 2,554 | (106,114,124) dim blue-gray | (193,162,148) pale |
   | front | 974 | (139,114,102) cheek/beard edge | (207,171,160) pale |
   | side_left | 0 | — | — |

   Every painter delivered DARK or mid content; the shipped color is far
   brighter than ALL of them. The pale color was **created after painting**
   by the completion stack: `tone_consensus` applied gain e^0.494 = +64% to
   the back view (gain_log 0.4943 in the replayed stats); `delight`
   brightened; the 40% unpainted texels were mesh-harmonic-filled from
   bright neighbors (right-cheek skin, the matte's beige wall fragment);
   `fill_floor` then lifted fill texels by mean +26% / p99 +153%
   (`lifted_texels` 103,910) toward the observed-donor consensus, and
   `fill_detail` (gain 0.7) added the chaotic micro-structure. The
   gradient-domain solve only smoothed it: a full **legacy-compositing
   rebake reproduces the smear** (92% of the smear texels still bright +
   desaturated; `d3_e10_shipped_vs_legacy_uvcrop.png`) — so this is not a
   Poisson/compositing artifact.
3. **Why the region was starved** (replayed `view_registration`, since the
   bundle ships no bake_stats.json): side_right — the only view that can
   witness the subject's right flank — registered at silhouette IoU
   0.336→0.375 with photometric scale 0.57 / shift_y −0.19 / profile_error
   0.226; its capture efficiency is 0.176 and coverage 0.058 (bundle
   metadata; side_left by contrast: 0.468 / 0.152). Root cause: the image is
   a ~−52° view sold as −90° (set-A pose lie, corroborating the viewgen
   audit), with waist-up framing and an arm. The bake's only reference gate
   is overlap-DISAGREEMENT (attenuate >0.24, reject >0.4,
   `texturing.py` ~7016-7026); side_right's disagreement was 0.213 →
   `weight_scale 1.0`, fully accepted despite garbage registration. There is
   **no registration-IoU floor** anywhere in the bake.
4. **White-background hypothesis refuted**: all three refs are properly
   matted (background alpha = 0; near-white pixels inside the foreground:
   0.02–0.25%, the silver lens strip only), and projection samples gate on
   alpha > 32 — background cannot paint. The front matte DID leak a beige
   picture-frame fragment beside the head
   (`d3_e10_front_sample_scatter.png`, `/tmp/e10_front_matted.png`) but the
   974 front-won smear texels sample mostly cheek/beard-edge pixels
   (mean 139,114,102) — a minor warm anchor, not the author.

**Minimal fix:**

1. `texturing.bake_projection_texture`: add a per-reference registration
   floor — after the registration stages, a synthesized reference whose
   registered silhouette IoU stays below 0.5 is dropped from painting (keep
   it out of tone/delight statistics too). In e10 that removes exactly
   side_right (0.375) and keeps side_left (0.749); the starved texels then
   fall to mirror completion, which had usable donor content (geometry
   symmetry score 0.93 and the LEFT side textured correctly — dark
   hair/beard instead of invented pale fill).
2. Belt: bound `fill_floor` lift for texels whose donor consensus comes from
   across a hair/skin luminance boundary (e.g. cap `p99` lift at ~0.4), so
   starved dark-material regions can never be lifted to skin tones. Fix 1
   alone removes this defect; fix 2 removes the class.

---

## Also audited

### e21_single_refs "ghost glasses on cheeks" — prior attribution DOES NOT HOLD; corrected

The claimed causes (windowed views + duplicate angles + 12° pose error) are
two-thirds wrong for e21:

- **No windowed views and no duplicate angles**: e21 baked exactly 4 views
  (front + back/side_left/side_right singles; metadata
  `reference_view_count` 4 vs e20's 7). Ref bytes are unrecorded (P1
  provenance gap) but the full-span framing + arm content in the mesh are
  consistent with `viewgen_fixed/*.png`.
- **The pose error that matters is the FRONT's, and it's new**: e21 is the
  only e2x bundle where the source-pose estimator's silhouette veto did NOT
  fire — `source_pose: az +7.99, estimated: true, method gradient_ncc`
  (e17/e19/e20 on the same photo: `ncc_vetoed_by_silhouette` → declared 0,0;
  e18: accepted az −1.2 / el −19). The front view carries FULL paint
  authority, so an accepted +8° azimuth rotates the whole front projection —
  glasses-band and beard content slide sideways onto cheek texels
  (`aux_e21_l30_cheek.png`, `aux_e21_r30_cheek.png`: glasses band descending
  onto the left cheekbone; beard inside the right lens).
- The two real carried-over mechanisms: (a) the refs' own ~12° under-rotation
  (the viewgen audit's residual — that's where the "12°" belongs; e20's BAKE
  ruler measured 20–25° against its own mesh); (b) the row-law violation is
  BACK, because e21 fed full-span sides with the chest-up clean4 front:
  production row instrument on the exact pairs — side_left **inconsistent**
  (score 0.00, shift 138 rows / 18%), side_right correctable +100, back
  correctable +75. Ungated as always on the caller lane.
- Minimal fix for e21's class: make the source-pose veto symmetric (the
  silhouette guard must veto ACCEPTANCES it cannot positively confirm, not
  just reject bad NCC scores — a ±8° acceptance on a face needs a
  head-band-IoU margin, the same instrument the e20 forensics built), plus
  the caller-lane row gate from defect 1/2.

### e20_fixed_views texture displacement — prior attribution VERIFIED

The texture_forensics.md account holds up against the artifacts: the shipped
l30 render reproduces lips-on-nose-tip and the displaced visor
(`aux_e20_l30_cheek.png` / `out/assessment/oblique/e20_obl_*`); its
instrument (the same capture harness) reproduced e10's bundle numbers here to
the 4th decimal, which validates its e20 findings methodologically; and its
correction of the audit's 12° (generation-clay frame) to 20–25° (e20-mesh
frame) is the right frame distinction. H4 (source-frame row misregistration,
+25…+42 px) dominant; H2 pose; H1 twins minor — confirmed.

### Gates cross-check (rejections, substitutions, pose-vs-label)

1. **Caller-reference geometry conditioning is ungated** — the row/mirror/
   material battery exists only inside `_synthesize_geometry_views` (auto
   lane). Every defective bundle here (e10, e17, e18, e20, e21) conditioned
   the DiT through the ungated caller path; e12 (auto lane) is the only run
   whose metadata carries per-view `row_consistency`.
2. **Silent substitution, confirmed once**: `loop_views_e11` — recorded
   rejections (back: speculars ×2 (=0019) + silhouette; side_right:
   silhouette ×3) followed by gate-bypassed rerun files 6–8 min later, no
   report, consumed by e17. `rejected_refs/` persistence exists in the
   refgen lane (`hunyuan_refgen/rejected_refs/back_a{0,1,2}.webp`) but the
   experimental loop script has no such contract.
3. **Pose-vs-label disagreement is systemic and invisible in provenance**:
   every side reference in every set (A: ±50°; B: +80/−97°; viewgen_fixed:
   +77.5/−102.5°) is declared ±90° with `snap_delta_deg: 0.0` recorded; no
   bundle records a measured pose. The bake's reference pose refinement is
   hard-disabled on the orthographic path, so declared == projected, always.
4. **Source-pose veto is inconsistent across same-photo runs** (accepted:
   e18 az−1.2/el−19, e21 az+8; vetoed: e17/e19/e20) and each acceptance
   materially moves full-authority front paint. The veto's decision record
   (`score`, `score_at_declared`) is kept — good — but there is no
   head-band-IoU confirmation step.
5. **The windowed-set fix was partially fictional in practice**: e18 fed the
   un-windowed front (the window was computed for it and not used) and the
   side_left window was a no-op (`cut_row` beyond subject bottom) — the
   only actually-windowed conditioning view in e18 was the back.

### Provenance gaps (P1 — the pipeline must archive its inputs)

1. **No bundle records its caller reference images** — no copies, no md5s,
   no resolved paths (only `label` + declared angle). e10's inputs had to be
   identified by byte-level bake replay; e21's cannot be pinned at all.
   The auto lane already records `raw_payload_md5` per synthesized view —
   extend the same to caller refs and copy the refs into the bundle
   (`conditioning_inputs/` exists as a manual convention for e11/e15/e17
   only).
2. **No `bake_stats.json` in any shipped bundle** (only the manual
   `e20_rebake_fixed`). The bake computes and then discards
   `view_registration` (IoU/scales/flow), `view_consistency`,
   `delight`/`tone_consensus` gains, `fill_floor` lift stats — exactly the
   records this audit had to regenerate by replay. Persist `stats` (arrays
   stripped) next to `metadata.json`.
3. **`loop_views_e11/regen_report.json` describes a different run than the
   files beside it** (see substitution above): rerun scripts must overwrite
   the report atomically with the images.
4. `e10.log` contains only an HF fetch progress line — experiment logs
   should capture the CLI invocation (the `run_experiments.sh` pattern did
   this; the later ad-hoc runs did not).

---

## Defect genealogy (one page)

| # | defect (operator's words) | first artifact where visible | mechanism | minimal fix |
|---|---|---|---|---|
| 1 | Open mouth in e18 mesh | `hunyuan_multiview/geometry_view_synthesized_side_left.png` (klein rotate draw, seed 53025, Jul 20 03:13) — lips parted; photo closed | parted-lips side view → caller-ref conditioning lane has no gates and no gate checks expression → 2mv carves the profile mouth per its only side evidence (e18 consumed the same pixels via the no-op "windowed" copy) | expression pin in `_geometry_view_prompt` + mouth-state acceptance check for person subjects; run the gate battery on caller refs |
| 2 | Bulbous nose + doubled-lip profile in e17 | `loop_views_e11/loop_view_side_left.png` (gpt-image-1, Jul 20 16:23) — protruding lip mass, mouth rows ~10% low; plus rejected-then-bypassed `loop_view_back/side_right.png` (16:29–16:31) | row-disagreeing conditioning (front vs side: +126/+137 rows, score 0.00) fed ungated → DiT double-carves nose/lip rows into one merged mass; 2 of 3 views consumed despite recorded gate rejections | row-gate caller refs with `view_consistency_report`; consumer must verify `accepted:true` per view against a report written atomically with the files; fix 0019 so back views stop needing bypass |
| 3 | Left(-viewed) white smear on e10 cheek/jaw/neck | `e10_2mv_registered_refs/texture.png` (Jul 20 14:24); precondition visible in `hunyuan_multiview/texture_reference_generated_side_right.png` (~−52° view declared −90°) | side_right registration collapse (IoU 0.375, eff 0.176) + no registration floor → 40% of region unpainted, rest painted dark → tone_consensus +64% gain, harmonic fill from bright donors, fill_floor lift (p99 +153%) invent pale chaos; compositing-independent (legacy A/B) | drop synthesized refs registering under IoU 0.5 (falls back to mirror completion, which had correct dark donors); cap fill_floor lift across material boundaries |
| aux | e21 ghost glasses on cheeks | `e21_single_refs/texture.png` (Jul 21 12:08) | NOT twins/windowed views (4 single refs): front source-pose false-accept (az +8, only e2x run where the silhouette veto passed an estimate) rotates full-authority front paint onto cheeks; + refs' ~12° under-rotation; + full-span sides re-violate the row law (side_left `inconsistent`, 138 rows) | head-band-IoU confirmation before accepting a non-zero source pose; same caller-lane row gate |
| aux | e20 texture displacement | `e20_fixed_views/texture.png` (Jul 21 07:52) | VERIFIED as prior forensics: H4 source row misregistration (+25–42 px) dominant, H2 ref pose 20–25°, H1 duplicate-angle twins minor | already shipped: duplicate-angle guard + split-consumer flag + source row law (`e20_rebake_fixed`) |

### Evidence file index (`out/laurent-bust-redo/review/forensics/`)

- `D1_e18_open_mouth_evidence.png` (+ `d1_*` singles: photo/clean4/side-view
  mouth crops ×5–10 magnification, e18 mesh mouth renders)
- `D2_e17_profile_evidence.png` (+ `d2_*` singles, `e17_mesh_az90_el0*.png`,
  `d12_canonical_mouth_rows*.png` ruler sheets, `e1[78]_canonical_sil_az90.png`)
- `D3_e10_white_smear_evidence.png` (+ `d3_*`: textured renders, UV smear
  crop, per-texel winner map, shipped-vs-legacy A/B, front-matte scatter)
- `aux_e2[01]_*_cheek.png`, `e2[01]_tex_*.png` (e20/e21 verification renders)
- Replay outputs: `/tmp/e10_forensics/` (capture.npz, bake_stats.json,
  registered views), `/tmp/e10_legacy/texture.png` (compositing A/B)
