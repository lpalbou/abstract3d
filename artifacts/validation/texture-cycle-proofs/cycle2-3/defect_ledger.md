# CRITIC 1 — AUTHORITATIVE DEFECT LEDGER
Zero-defect texture cycle, 2026-07-05. Critic: owner's proxy. PASS requires
ZERO entries in status OPEN. "Better than before" is not a status.

> CYCLE-1 FINAL 2026-07-05 20:10: 31 entries, 31 OPEN, 0 FIXED. FAIL —
> `/tmp/critic1/RULING.md`.
>
> CYCLE-2 FINAL 2026-07-06 05:40: **12 FIXED, 5 PROVEN-LIMIT (closed), 14
> OPEN. Verdict FAIL** (zero-open bar; no new defects introduced by
> cycle-2 mechanisms). Per-entry statuses, my verifying evidence, the two
> harness rulings (owl matting ACCEPTED; shading compensation VALID as
> adjunct, raw gate authoritative), and cycle-3 orders:
> `/tmp/critic1/RULING_CYCLE2.md`. Evidence tree: `/tmp/critic1/c2/`,
> sheets `/tmp/critic1/evidence/sheets/c2_*.png`.

## Ground rules (solvers read this)

- Standard of proof: my eyes at 1000 px + 2x/4x nearest-neighbor crops, at
  16 azimuths x el {-20, 0, +15} plus the declared source pose. Harness
  gates (`/tmp/verdict1/qa.py`, `scripts/texture_qa.py`) are the FLOOR, not
  the bar. An entry closes only when I re-render and cannot see it.
- When you claim a fix: write the recipe in your REPORT.md. I rebake at
  2048 myself, rerun both harnesses, re-render every ledger region (not
  just yours), and mark FIXED / IMPROVED / UNCHANGED / REGRESSED. New
  defects you introduce become new entries attributed to you.
- Impossibility (P3): accepted only with a ceiling experiment (e.g. bake a
  synthetic perfect texture through the same path and show the same
  inspection still fails). Then the entry becomes PROVEN-LIMIT with the
  capture-side remedy documented. Anything less stays OPEN.
- Do not fit to the test bundles: fixes must be general pipeline logic.
  The owl is the regression canary; it gets rebaked and re-inspected too.

## Baseline state audited (T0 = 2026-07-05 15:40 CEST)

| bundle | texture.png md5 | scene.glb md5 | geometry.glb md5 |
|---|---|---|---|
| face-2mv (iter3-multiview-fixed) | 44587ff39d4b2076ead2676203eee574 | 0600146431ce264d46a4fc39b382821c | 95461cd02320681602d5e7547f5a5655 |
| hunyuan-starship (final-proof) | f724de143eff705910c0024f538aa797 | 79120202bc082c56ac5dbfd1111ffa48 | 0034ca4d2920d9937502492f24cb62a8 |
| hunyuan-owl (final-proof, canary) | f6dfb76dfde243780cc09bcf65787b18 | 6d5f45e01f306a554036997b09b55f59 | 103a7692290a72e5a09d1d30529bc122 |

Harness baselines at T0 (my runs, logs under /tmp/critic1/harness/):

- verdict1 qa.py face-2mv: **FAIL, 9 checks** — identity[front] ssim 0.613
  (<0.70) + mae 22.9 (>22.0); dark_debris az-22.5 el0 0.0041, az-35 el0
  0.0047, az-22.5 el10 0.0039, az-45 el10 0.0034 (gate 0.003); eye_count
  az-90 el0 & el10 = 0 (min 1); eye_count az+22.5 el10 = 3 (max 2).
- texture_qa face-2mv: PASS 0/13 (the close-zoom floor passes; my entries
  below are all above that floor).
- texture_qa hunyuan-starship: **FAIL 1/13** — close.dark_smears_4x = 4
  (probes: concavity_03 az-122 el+15, concavity_04 az-43 el-13,
  concavity_08 az-95 el-60, fill_03 az-89 el+60).
- texture_qa hunyuan-owl (canary): **FAIL 2/13** — viewer.brightness_ratio
  0.567 (floor 0.72); close.dark_smears_4x = 6.

Severity scale (owner's viewing): **blocking** = seen at arm's length in
the first ten seconds; **major** = plainly visible at 1x-2x or at specific
angles on a turntable; **minor** = 4x/grazing-angle only; **cosmetic** =
needs side-by-side comparison.

Lanes: S1 compositing seams/tone · S2 local registration/stretch ·
S3 delighting/confidence/fragments · S4 geometry-attributed.

Evidence tree: `/tmp/critic1/evidence/{face,ship,owl}/renders/` (full view
sets, `GRID_el*.png` overviews, `srcpose_*.png`), region zoom sheets under
`/tmp/critic1/evidence/sheets/` (tile label bottom-left of every tile),
harness evidence under `/tmp/critic1/harness/*/evidence/`. Crop recipe for
every entry: source render path + (cx, cy, zoom) into
`/tmp/critic1/make_sheet.py` conventions (crop side = 1000/zoom px
centered at cx,cy fractions).

---

## LEDGER — face-2mv

| ID | severity | lane | status | title |
|---|---|---|---|---|
| FACE-01 | blocking | S3 | OPEN | Beige film band w/ torn flake edges over both temples & hairline |
| FACE-02 | blocking | S3 | OPEN | Black debris hole/cracks at hair parting and az-90 hairline |
| FACE-03 | major | S3 | OPEN | Under-eye 3D flake clusters + gray dashes (both eyes) |
| FACE-04 | major | S3+S2 | OPEN | Mouth-corner beige/green smear, below-lip patch, chin flake cluster |
| FACE-05 | major | S1 | OPEN | Pale seam column: nose bridge -> philtrum -> chin |
| FACE-06 | major | S2+S1 | OPEN | Front identity: tone split + seam through subject-right eye; SSIM 0.613 |
| FACE-07 | major | S3 | OPEN | Ear texture debris: pale shards + dark strokes on both ears |
| FACE-08 | major | S4 | OPEN | Pointed "elf ear" silhouette, both ears |
| FACE-09 | major | S3 | OPEN | Rear hair mass: leopard-mottle fill, zero strand structure |
| FACE-10 | minor | S3+S1 | OPEN | Hair curtain inner-edge tan stripe + skin halo behind ears |
| FACE-11 | minor | S2+S3 | OPEN | Chest: white strap smears, white blob, lace patch |
| FACE-12 | minor | S4+S3 | OPEN | Bust cut disc: tan wash + dark radial rim streaks from below |
| FACE-13 | minor | S4+S3 | OPEN | Crown: ragged mesh flaps + mottle at el+15 |
| FACE-14 | major | S2+S1+S3 | OPEN | HARNESS identity[front] ssim 0.613 < 0.70, mae 22.9 > 22.0 |
| FACE-15 | minor | S2+S3 | OPEN (IMPROVED, mechanism verified on v14 base) | HARNESS eye_count az-90 = 0 (el0, el10): profile eye smudged |
| FACE-16 | major | S3 | OPEN | HARNESS eye_count az+22.5 el10 = 3: dark curl blob inside temple film |
| FACE-17 | blocking | S1+S2 | OPEN (cycle regression, candidate lane) | Pale nose ghost column + ghost lip stripe on current-tip bakes |
| FACE-18 | major | S2 | OPEN (cycle regression, candidate lane) | Pose drift +20->+12.5 erases profile eyes on fresh bakes |
| FACE-19 | major | S1 | OPEN (cycle regression, candidate lane) | delight_projections drives side_right MAE 24 -> 32-40 |

### FACE-01 — beige film band over temples/hairline [BLOCKING, S3, OPEN]
Translucent skin-beige sheets with torn, flake-fringed borders lie OVER the
hair mass at both temples and along the hairline; dark hair curls painted
inside the film; reads as a skin-colored shower-cap edge at arm's length.
Views: az -67.5..+67.5 at el {-20,0,+15} and at declared pose (-20,+8).
Worst: viewer-left temple az0 el0; viewer-right temple carries dangling
flake "fingers".
Evidence: `sheets/face_front_4x.png` tiles "az0 hairline-Lviewer 4x",
"az0 hairline-center 4x", "az0 hairline-Rviewer 4x", "srcpose hairline 4x";
renders `face/renders/az+0000.0_el+00.png` (cx .37 cy .33), srcpose file.
Root-cause note: the "witness gate" re-admits mottled film paint at temples
(verdict1 v14 addendum already called this wrong-direction).

### FACE-02 — black debris hole at parting + hairline cracks [BLOCKING, S3, OPEN]
Hard-black irregular hole (~40x15 px at 1000 px) at the parting above the
viewer-left brow, black crack strokes continuing along the film boundary;
at az-90 a long black crack streak runs down the face-hair boundary with
beige lace filaments beside it.
Views: az0/±22.5 (all els), az-90, az-67.5, el+15 parting.
Evidence: `sheets/face_front_4x.png` "az0 hairline-Lviewer 4x";
`sheets/face_side_4x.png` "az-90 hairline streak 4x";
`face/renders/az-0090.0_el+00.png` (cx .68 cy .33 z4).

### FACE-03 — under-eye flake clusters + gray dashes [MAJOR, S3, OPEN]
Beige, 3D-looking flake chips clustered under both eyes (worst
subject-right/viewer-left), with small gray/black dashes; tear-duct
fragments. Drives the four dark_debris harness failures (az-22.5 el0
0.0041, az-35 el0 0.0047, az-22.5 el10 0.0039, az-45 el10 0.0034).
Views: az0..±45 all els, srcpose.
Evidence: `sheets/face_front_4x.png` "az0 eye-R(subj) 4x", "srcpose eyes
4x"; `sheets/face_side_4x.png` "az-22.5 under-eye debris 4x";
harness crops `/tmp/critic1/harness/v1_face/evidence/az-022.5_el00_*.png`.

### FACE-04 — mouth-corner smear + below-lip patch + chin flakes [MAJOR, S3+S2, OPEN]
Beige smear with a greenish tint at the viewer-left mouth corner breaking
the lower-lip contour; pale patch below the lower lip; chin carries a
cluster of beige flakes with dark specks. Views: az0..±22.5 els, srcpose.
Evidence: `sheets/face_front_4x.png` "az0 mouth 4x", "az0 chin 4x",
"srcpose mouth+chin 4x"; `sheets/face_side_4x.png` "az-22.5 mouth/chin 4x".

### FACE-05 — pale seam column nose->chin [MAJOR, S1, OPEN]
A continuous pale, low-chroma seam column runs down the nose bridge,
through the philtrum, to the chin (mirror/projection boundary). Includes
the nose-tip/columella beige blob and smudged nostril rims.
Views: az0..±22.5, srcpose. Evidence: `sheets/face_front_4x.png`
"az0 nose 4x", "az0 mouth 4x" (philtrum), "srcpose eyes 4x" (bridge).

### FACE-06 — front identity: tone split + eye seam [MAJOR, S2+S1, OPEN]
The face is split by a vertical tone boundary left-of-nose: subject-right
of the seam is warmer/darker, subject-left paler; the seam passes through
the subject-right eye, whose iris tone differs across it. Identity harness
fails (ssim 0.613 vs 0.70, mae 22.9 vs 22.0 at declared pose az+20 el+8).
Views: az0..±22.5, srcpose. Evidence: `sheets/face_front_4x.png`
"az0 eye-R(subj) 4x", "srcpose(-20,8) face 2x";
`/tmp/critic1/harness/v1_face/evidence/identity_front_*.png`.

### FACE-07 — ear texture debris [MAJOR, S3, OPEN]
Pale flake shards and hard dark strokes baked onto both ears (antihelix,
canal, below-lobe "tape" strip). Views: az±45..±112.5 all els.
Evidence: `sheets/face_side_4x.png` "az+90 ear 4x", "az-90 ear 4x",
"az+45 ear silhouette 2x"; `sheets/face_rear_4x.png` "+112.5" tile.

### FACE-08 — elf-ear silhouette [MAJOR, S4, OPEN]
Both ears pointed (non-human) in silhouette at az0..±67.5. Geometry, not
texture (prior rulings agree: present in shipped mesh). Candidate for
PROVEN-LIMIT **only** with a P3 ceiling experiment (e.g. show the Hunyuan
mesh at this decimation cannot carry a round ear without regeneration, and
document the capture/geometry-side remedy). Until then OPEN.
Evidence: `sheets/face_front_4x.png` "az0 temple+ear-*" tiles;
`sheets/face_side_4x.png` "az+45 ear silhouette 2x".

### FACE-09 — rear hair leopard mottle [MAJOR, S3, OPEN]
Central-back hair is an irregular dark-blotch-on-brown mottle with no
strand direction; boundary to the profile-painted outer sheets visible as
texture-character seams. Views: az±112.5..180, all els; also the rear half
of both profiles. Evidence: `sheets/face_rear_4x.png` "az180 back center
2x/4x", "az180 nape 4x", "az±135/±157.5 4x";
`sheets/face_side_4x.png` "az+90 rear hair mass 4x", "az-90 rear hair mass 4x".

### FACE-10 — curtain inner-edge tan stripe + skin halo [MINOR, S3+S1, OPEN]
Tan/skin-tone vertical striping along the inner edge of the hair curtains
and a thin skin halo along the face-hair boundary behind the ears.
Views: az±112.5..±157.5. Evidence: `sheets/face_rear_4x.png` "az+135 hair
mass 4x", "az-135 hair mass 4x", "az+112.5 face-hair edge 4x".

### FACE-11 — chest strap smears + white blob + lace patch [MINOR, S2+S3, OPEN]
White strap fragments smeared diagonally on the chest, an isolated white
blob on the viewer-left chest, a lace-textured patch on the shoulder.
Views: az0/±22.5 el0 & el-20. Evidence: `sheets/face_front_4x.png`
"az0 chest/straps 2x".

### FACE-12 — bust cut disc from below [MINOR, S4+S3, OPEN]
The synthetic bust-cut disc shows a tan wash with dark radial rim streaks
and a ragged dark hair fringe; visible whenever the camera goes below
el ~-10. Views: el-20 all az. Evidence: `sheets/face_rear_4x.png`
"az0 el-20 bust disc 2x", "az180 el-20 bust under 2x".

### FACE-13 — crown mesh flaps + mottle [MINOR, S4+S3, OPEN]
Ragged mesh flaps break the crown silhouette; crown texture mottled at
el+15. Evidence: `sheets/face_rear_4x.png` "az180 crown 4x",
"az180 el15 crown/back 4x"; any el+15 render.

### FACE-14 — HARNESS identity[front] [MAJOR, S2+S1+S3, OPEN]
ssim 0.613 < 0.70; mae 22.9 > 22.0 at declared pose az+20 el+8. Aggregate
of FACE-01/03/05/06; closes when they close AND the numbers clear the bar.
Evidence: `/tmp/critic1/harness/v1_face.log`, results.json.

### FACE-15 — HARNESS eye_count az-90 = 0 [MINOR, S2+S3, OPEN]
Right-profile eye undercounted at el0/el10: iris sliver + tear-duct region
smudged by flakes (FACE-03/07 family). The eye is visually present; if a
solver claims detector-limit, that is a P3 claim needing the ceiling
experiment (photo-perfect profile bake still undercounted).
Evidence: `sheets/face_side_4x.png` "az-90 eye 4x".

### FACE-16 — HARNESS 3rd eye at az+22.5 el10 [MAJOR, S3, OPEN]
A dark hair curl painted INSIDE the temple film band is blob-counted as a
third eye. Same root cause as FACE-01 (film re-admits curl+film mix).
Evidence: `/tmp/critic1/harness/v1_face/evidence/az+022.5_el10_annotated.png`
and `_eye_*.png` crops.

---

## LEDGER — hunyuan-starship

| ID | severity | lane | status | title |
|---|---|---|---|---|
| SHIP-01 | blocking | S3 | OPEN | Starboard/fill side is cloudy mottle wash vs port's crisp panels |
| SHIP-02 | major | S3 | OPEN | HARNESS dark fill fragments at 4x (4 probes) |
| SHIP-03 | major | S2+S3 | OPEN | Bow/nose melt: dark streaked concavity + frayed wing leading edges |
| SHIP-04 | major | S3 | OPEN | Underside + engine nozzles: featureless soft wash |
| SHIP-05 | minor | S3 | OPEN | Bright white glow blob on rear-top starboard |
| SHIP-06 | cosmetic | S1 | OPEN | Global tone 25% darker than photo (ratio 0.752, floor 0.72) |
| SHIP-07 | major | S1+S3 | OPEN | Photo->fill patch seams interrupt panel lines mid-hull |
| SHIP-08 | major | S1+S3 | OPEN (cycle regression, candidate lane) | Current-tip rebake collapses fill energy 0.566 -> 0.277 (<0.5 gate) + facet_cellular 0.445 FAIL |

### SHIP-01 — fill side cloud mottle [BLOCKING, S3, OPEN]
Everything the photo (az+30 el+15) did not witness — starboard hull, most
of the rear, far wing — is a soft cloudy gray mottle: no panel lines, no
greebles, at 2x it reads as smoke. The port/photo side shows crisp panels;
rotating the turntable 90 degrees flips the asset from "model kit" to
"clay blank". Views: all az<0, especially -22.5..-135, all els.
Evidence: `sheets/ship_asym.png` "az-22.5 hull 2x" vs "az+22.5 hull 2x",
"az-135el15 rear top 2x" vs "az+135el15 rear top 2x";
`sheets/ship_regions.png` "az-22.5 starboard bow 4x", "az-157.5 rear-R 4x".
Content cannot be invented from one photo (P3 candidate for *content
truth*), but plausible panel-texture synthesis is the pipeline's own claim
(fill_energy gate >= 0.5) — the current fill is below that visual bar even
though the number passes. OPEN.

### SHIP-02 — HARNESS dark fill fragments at 4x [MAJOR, S3, OPEN]
texture_qa close.dark_smears_4x = 4: concavity_03 (az-122 el+15),
concavity_04 (az-43 el-13, fraction 0.0162), concavity_08 (az-95 el-60),
fill_03 (az-89 el+60, fraction 0.0225). My az-22.5 2x crop confirms two
dark pits near the starboard wing root. Evidence:
`/tmp/critic1/harness/tq_ship/evidence/concavity_04_z4_annotated.png`,
`fill_03_z4_annotated.png`; `sheets/ship_asym.png` "az-22.5 hull 2x".

### SHIP-03 — bow/nose melt [MAJOR, S2+S3, OPEN]
The nose concavity and cockpit surround at az0..±22.5 are covered by dark,
stretched, streaky texels (grazing-angle projection smear); wing leading
edges at az0 carry frayed light-gray "torn paper" mottle. Views: az0
el{-20,0,+15}, ±22.5. Evidence: `sheets/ship_regions.png` "az0 nose front
2x", "az0 nose front 4x", "az0 el-20 nose under 2x".

### SHIP-04 — underside/engines featureless [MAJOR, S3, OPEN]
The entire underside and the engine nozzles are a smooth soft-gradient
wash with faint blotches — no mechanical detail anywhere the photo could
not see (el-20 views; az180 stern). Views: el-20 all az; az180 el0.
Evidence: `sheets/ship_regions.png` "az180 el-20 underside 2x/4x",
"az180 engines 4x", "az0 el-20 nose under 2x"; `ship/renders/GRID_el-20.png`.
P3 candidate for content truth; plausible-fill bar still applies.

### SHIP-05 — bright glow blob rear-top starboard [MINOR, S3, OPEN]
A soft white glow blob sits in the mottle on the rear-top starboard
quarter (az-135 el+15 area), reading as a rendering light artifact baked
into the texture. Evidence: `sheets/ship_asym.png` "az-135el15 rear top
2x" (bright spot right of center).

### SHIP-06 — global tone [COSMETIC, S1, OPEN]
Viewer-truth brightness ratio 0.752 vs photo (floor 0.72) — passes but
visibly darker side-by-side; margin 0.03 is one regression away from
failing. Evidence: `/tmp/critic1/harness/tq_ship.log`;
`tq_ship/evidence/viewer_truth_vs_repo_render.png`.

### SHIP-07 — photo->fill patch seams on hull top [MAJOR, S1+S3, OPEN]
Where photo-projected panels end mid-hull, mottle patches begin with
visible texture-character boundaries cutting across panel lines (top hull,
az+157.5 el+15; az+135 el+15 transition zone; port-to-starboard crossover
on the spine). Evidence: `sheets/ship_regions.png` "az+157.5el15 rear top
4x", "az+22.5el15 hull top 4x" right edge; `sheets/ship_asym.png`
"az+135el15 rear top 2x".

---

## LEDGER — hunyuan-owl (regression canary)

| ID | severity | lane | status | title |
|---|---|---|---|---|
| OWL-01 | major | S1 | OPEN (IMPROVED on rebake: 0.709, still < 0.72) | HARNESS viewer brightness 0.567 (floor 0.72): render 44% darker than photo |
| OWL-02 | major | S3 | OPEN (mechanism FIXED on my fresh rebake: 0 smears; asset unchanged) | HARNESS 6 dark smear fragments at 4x in concavities |
| OWL-03 | major | S3 | OPEN | Rear/back fill: flat brown wash, carved-feather detail lost |
| OWL-04 | blocking | S2 | OPEN (cycle regression, candidate lane) | Pose lottery on canary: estimator turns frontal owl to az+32.5 (score 0.0043), wood smears off-face |

### OWL-01 — canary brightness [MAJOR, S1, OPEN]
The owl bundle (rebaked today 15:13) renders at 0.567 of the photo's
foreground luminance — visibly darker, worst regression class the cycle
claimed fixed. Evidence: `/tmp/critic1/harness/tq_owl.log`;
`tq_owl/evidence/viewer_truth_vs_repo_render.png`; `owl/renders/GRID_el+00.png`
vs `final-proof/hunyuan-owl/input.png`.

### OWL-02 — canary dark smears [MAJOR, S3, OPEN]
6 spurious dark fragments at 4x across concavity probes (az0.6 el35,
az-108 el31 x2, az23 el-4, az38 el10, az-12 el20). Evidence:
`/tmp/critic1/harness/tq_owl/evidence/concavity_*_z4_annotated.png`.

### OWL-03 — canary rear flat wash [MAJOR, S3, OPEN]
Back hemisphere (az±112.5..180) is a flat light-brown wash with sparse
mottle; the carving's feather relief texture is absent. Evidence:
`owl/renders/GRID_el+00.png` rows 2-3; `owl/renders/az+0180.0_el+00.png`.

---

## Counts at T0

| status | count |
|---|---|
| OPEN | 26 (16 face, 7 ship, 3 owl) |
| FIXED / PROVEN-LIMIT / anything else | 0 |

Blocking: FACE-01, FACE-02, SHIP-01. The owner sees these in the first ten
seconds at arm's length. No verdict better than FAIL is possible while any
entry is OPEN.

### VERIFICATION ROUND 3 — 20:00 CEST — ship + owl current-tip rebakes

My own rebakes on the end-of-cycle tip (ship pose PINNED to declared
+30/+15; owl estimator free):

- SHIP (`verify3/ship`): texture_qa **dark_smears_4x 4 -> 0** — solver3's
  fill floor genuinely eliminates the SHIP-02 fragment class (verified on
  my own bake, not their numbers). Brightness 0.752 -> 0.910 (SHIP-06
  would clear). BUT fill_gradient_energy_ratio **0.566 -> 0.277 FAIL**
  and facet_cellular **0.445 FAIL** (T0 passed both): the current stack
  strips the synthesized fill detail the ship gained last cycle — the
  fill side is now a SMOOTHER cloud than T0 (my crops
  `sheets/verify3_sheet.png` "V3 vs T0 az-22.5 / az-135el15"), pushing
  SHIP-01 in the WRONG direction. Logged as SHIP-08.
- OWL (`verify3/owl`): estimator chose **az+32.5 el0 (NCC 0.0043)** for a
  frontal statue photo — pose lottery on the canary; wood-grain content
  smears off the right flank and a white horn-shaped fill blob invades
  the crown (crop "V3 owl az0 face 2x"). Brightness 0.567 -> 0.709 (still
  < 0.72 floor); dark smears 6 -> 0; facet_fields 3 NEW FAIL. Logged as
  OWL-04. The canary says: fresh bakes through the current tip are NOT
  shippable without a pose gate.

## Update log

- T0 2026-07-05 15:55 CEST — ledger created; 26 entries OPEN.
- 17:25 CEST — 90-min mark. Reports filed: solver2 only (M1 analysis, no
  patch claim yet; attributes verdict2's "+9% stretch" to YuNet scale bias
  and confirms FACE-04's mouth-corner ghost is displaced right-profile lip
  content at winner weight 0.006-0.2 — attribution accepted, entry stays
  OPEN). Solvers 1/3/4 active, no REPORT.md. Extending watch; ruling will
  use whatever is verified by cutoff.

### VERIFICATION ROUND 1 — 19:05 CEST — CANDIDATE STACK REGRESSES

My own 2048 face rebake on the tip (texturing.py 75bb412b, triposr_runtime
c3d3a583, gradient_compositing b9a053ce; auto -> gradient_domain +
delight + fill floor + geometry confidence all active), estimator's own
pose: **az +12.5 el +8 (drifted from +20; NCC score 0.0052)**. Results:

- verdict1: **28 failed checks vs baseline 9.** identity[front] 0.582/26.2
  (was 0.613/22.9), localMin 0.018 FAIL; side_left localMin 0.049 FAIL;
  side_right MAE 40.0 FAIL; eye_count=0 at ±35..±90 EL BOTH on the minus
  side and at +90 (profile eyes now smudged away — NEW); chroma_seam FAIL
  at az0/−22.5/−35 (up to 63% vs gate 45% — NEW class); crown_flakes NEW
  at +45/+70/±135.
- My 4x crops (`sheets/verify1_key.png`): NEW pale nose ghost column
  (FACE-05 regressed hard), doubled lower lip (FACE-04 regressed to a
  full ghost pair), +90 eye covered by pale streak (NEW), −90 eye erased,
  hairline film band UNCHANGED (FACE-01/02 not improved), rear mottle
  UNCHANGED (FACE-09).
- texture_qa: PASS 0/13 — confirming the close-zoom floor is blind to all
  of this; my render battery is the standard.

Attribution unclear between the four concurrently-landed stages (pose
drift +20->+12.5 compounds it). Pinned +20/+8 bake running to separate
pose from stages. **Message to all solvers: the integrated tip is
currently a net regression on the face lane at owner conditions. Nothing
in the ledger moves to FIXED. New entries FACE-17/18 opened below.**

### FACE-17 — REGRESSION: pale nose ghost column + doubled lower lip at az0 [BLOCKING, S1+S2, OPEN]
Candidate-stack bake only (not in T0 baseline). Nose becomes a
desaturated pale column with displaced nose content beside it; a full
ghost lip pair sits below the true lips. Evidence:
`sheets/verify1_key.png` "V1 az0 nose 4x", "V1 az0 mouth 4x".
Owner of the regression: whichever stage(s) shipped it — solvers must
A/B their stage off to exonerate themselves; I will re-verify.

### FACE-18 — REGRESSION: profile eyes smudged/erased at ±90 [MAJOR, S3, OPEN]
Candidate-stack bake only. +90 eye overlaid by a pale streak; −90 eye
reduced to a dark smear; harness eyes=0 at +90 el0/el10 (baseline had 1)
and −35/−45/−70 minus-side undercounts. Evidence: `sheets/verify1_key.png`
"V1 az+90 eye 4x", "V1 az-90 eye 4x".

### VERIFICATION ROUND 1b — 19:30 CEST — POSE PIN ISOLATES THE DRIFT

Same tip, estimator pinned to az+20 el+8 (`verify1/face_pin20`):
verdict1 **10 failed checks** (unpinned 28, T0 baseline 9). Reading:

- The 28-check catastrophe was ~2/3 pose drift (+20 -> +12.5 at NCC score
  0.0052). FACE-18 (±90 eye erasure) is pose-drift-only: pinned bake's +90
  eye is intact. **The pose estimator's instability is now a ledger-scale
  hazard: any solver stage that changes the NCC landscape re-rolls the
  dice. A stability margin or low-score pin is required before ANY fresh
  bake can be trusted** (tex4 flagged this; it is now biting this cycle).
- Pinned-vs-T0 deltas: front SSIM 0.613 -> 0.620 (still FAIL), front MAE
  22.9 -> 23.1 (FAIL), side_right MAE 24.1 -> **32.1 NEW FAIL** (delight
  kept a side_right correction that moves the render AWAY from the
  reference photo at the declared pose), chroma_seam -22.5 el10 45% NEW
  marginal FAIL, dark_debris -22.5 el10 0.0039 -> 0.0051 worse, -70 el10
  eye NEW undercount; +22.5 el10 3-eye blob gone (FACE-16: the film-band
  mix moved, not cleaned — see crops, band still present).
- 4x crops (`sheets/pin20_key.png`): FACE-01 film band UNCHANGED (edges
  now harder, with dark rim segments); FACE-02 black hole UNCHANGED;
  FACE-03 under-eye flakes UNCHANGED; FACE-04/17: a dark-red ghost lip
  stripe under the lower lip + white streak crossing it — the ghost-lip
  class is IN the new stack even at pinned pose (weaker than unpinned);
  FACE-05 nose ghosting VISIBLY WORSE than T0 at az0; FACE-09 rear mottle
  UNCHANGED; chin dark dash heavier (drives the 0.0051 debris reading).

Status changes: **none to FIXED.** FACE-17 stays OPEN (present pinned,
worse unpinned). FACE-18 attributed to pose drift (S2 lane owns the
estimator gate). side_right tone shift logged as FACE-19.

### FACE-19 — REGRESSION: side_right identity MAE 24.1 -> 32.1 [MAJOR, S1, OPEN]
Pinned same-tip bake fails the side_right MAE gate (32.1 > 30) that the
T0 baseline passed. Suspect: delight correction "kept" on side_right
(solver3 reports overlap disagreement -26% on their 1024 runs) relights
the profile away from the reference photo; possibly compounded by the
fill floor lifting the hair mass. Solver3 must A/B delight-off at 2048.
Evidence: `/tmp/critic1/verify1/v1_pin20.log`.
Corroborated independently by solver4's bisect (§6 of their report):
same tip, delight OFF alone recovers 31 -> 9 fails and side_right MAE
39.5 -> 26.4. Attribution to `delight_projections` CONFIRMED from two
independent measurements (mine: pinned-pose MAE 32.1 with delight on).

### VERIFICATION ROUND 2 — 19:45 CEST — solver4 candidate (v14 base)

`/tmp/solve4/candidate_v14_solve4` (eyefix + earclamp on FROZEN v14, not
on the T0 ledger asset). My runs:

- verdict1: **5 fails** — ±90 eye_count ALL CLEARED (FACE-15 mechanism
  verified: mirror transplant produces a structured eye at -90, my crop
  `sheets/solve4_eye.png` "S4 az-90 eye 4x" vs "T0 az-90 eye 4x"; their
  A/B `g1_final_ab.png` consistent). New trade: az0 el0+el10 eyes=3 (v14's
  temple-band curl blob at BOTH els — T0 asset has it only at +22.5 el10)
  and az-45 el10 eyes=0 (transplant speculars fragment the blob;
  visually present). identity front 0.619/23.1 FAIL (v14 legacy).
- texture_qa: **FAIL 3 gates** — fill_gradient_energy 0.398 (<0.5),
  facet_cellular 0.575 (>0.377), facet_fields_4x = 3. The v14 texture
  family predates the fill-detail/facet fixes; at 4x its fill is
  honeycomb-faceted. My T0 asset passes all three.
- Net: **neither bundle dominates.** v14_solve4 wins the verdict1 count
  (5 vs 9) but ships close-zoom facet fields the T0 asset eliminated.
  Zero-defect requires the eyefix mechanism REBAKED onto the current
  fill stack (solver4's `mirror_rescue_disc` is in the repo — the path
  exists), not a return to the v14 texture.

Ledger deltas from round 2:
- FACE-15 -> **IMPROVED** (stays OPEN): mechanism verified on v14 base;
  not yet applied to a bundle that also passes texture_qa; new -45 el10
  trade must not survive integration.
- FACE-08 -> **IMPROVED** (stays OPEN), annotated: conservative geometry
  clamp verified silhouette-safe but visually marginal (my crop "S4
  az+45 ear clamp 2x" — apex still reads pointed under the film debris);
  solver4's quantified bound (apex prominence >=0.05 vs 0.02 cap) accepted
  as PROVEN-LIMIT for the CONSERVATIVE-CLAMP class only. The elf READ is
  half texture (skin painted on an apex the photos show hair-covered) —
  that half is OPEN and owned by the texture lanes (S3).
- FACE-09 -> annotated: solver4's G3 ceiling experiments (coherent
  multigrid orientation field + LIC still renders grain; stripe-carrier
  probe decoheres at strand wavelength) are ACCEPTED as PROVEN-LIMIT for
  "legible individual strands from these inputs by procedural fill".
  The entry stays OPEN for its actual bar: the rear must read as
  plausible hair MATERIAL (smooth sheet w/o leopard blotches), which no
  candidate achieves yet. Capture-side remedy documented: a rear/back
  reference photo (or generative hair prior) is required for strands.
- FACE-16 -> annotated: v14-based candidates make the curl-blob WORSE
  (az0 x2); the film-band cleanup (FACE-01) remains the root fix.
