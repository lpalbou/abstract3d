# Changelog

## Unreleased

### Changed (bake fusion — two-band detail ownership, research-grounded)

A three-track literature review (photogrammetry texturing practice,
generative texture pipelines, view-fusion theory) unanimously identified
the softmax-weighted view average as structurally wrong for detail:
averaging views with residual registration error is convolution with the
error kernel (two views offset by d texels null every feature finer than
~2d), and the softmax bias is a NO-OP exactly at weight ties — the whole
equal-facing ridge between adjacent view cones degenerates to a plain
average. Production tools (Metashape "mosaic", Baumberg two-band,
Hunyuan3D-Paint single-ownership) never average high frequencies.

- `blend_projections` now uses TWO-BAND fusion for multi-view bakes
  (`detail_fusion="two_band"`): the low band (tone, lighting) keeps the
  softmax average with its wide smooth transitions; the 2-8 px detail
  band is winner-take-all from the single best view per texel — argmax
  over weight maps smoothed at 8 texels (smooth the WEIGHTS, not the
  labels, or ownership dithers into slivers on the tie ridge), with a
  ~1.5-texel feather at handoffs (the band is only zero-mean BELOW the
  split scale; a zero-width switch still steps by local content).
  Single-view bakes are bit-identical; `detail_fusion="average"`
  restores the previous behavior.
- New HANDOFF-SEAM LEDGER in the blend stats (`handoff_seams`): tone
  disagreement (owners' low-band delta) measured exactly at
  detail-ownership boundaries, in texture space where the handoffs are
  known. Groundwork for the acceptance gate: a render-space seam metric
  cannot separate a genuine handoff seam from a crisp carved contour
  (measured: their side-tone distributions overlap completely).
- Measured and REVERTED, documented for re-landing: bicubic registration
  warps and cubic projection sampling recover 5-7% of the relief band
  each (owl back: 12.69 -> 13.54 texel-space band RMS), but their edge
  overshoot raises the acceptance gate's long-strong-edge statistic by
  the labeled chair-regression magnitude — real seams and restored
  carved contours become indistinguishable in the one metric that
  auto-protects unattended users. They return when the gate consumes the
  handoff ledger instead.
- Iteration protocol upgrade: every candidate bake in this program was
  screenshotted through MeshVault's headless viewer-truth endpoint
  (`GET /api/screenshot`) before judgment, and the full iteration ladder
  (10 bakes, 20 labeled contact sheets) ships in the review folder.
  Six-view angle densification (back_left/back_right at 135°) was
  generated, gate-checked, and parked: without cross-view consensus
  alignment the extra overlap washes tone (fidelity regression caught by
  the whole-bake gate), which is the Zhou-Koltun-style alignment stage
  on the roadmap.

### Changed (generated references — adversarial round 2: completion-only protection, strict-only baking, person bypass)

An independent adversarial review of the four-subject v2 bakes identified
three systemic failures the material gates could not see; each is now
closed at the layer that owns it:

- **Generated views complete, never revise** (`protect_observed_texels`,
  `texturing.py`): after the tone/consistency stages (which need overlap
  texels for their statistics), synthesized weight is zeroed wherever the
  strongest REAL view holds a credible claim (weight >= 0.25, the
  conflict-resolution priority floor), with a linear ramp below the floor
  so the real-rim -> generated-content handoff stays smooth. Root cause:
  weight subordination (x0.6) loses per-texel contests but the feathered
  blend still AVERAGES synthesis into photo-covered texels — measured as
  rust mottling on the chair's front fabric and skin blotches on the
  portrait's front face. A real photo is evidence; a generated view is
  plausible synthesis; synthesis must contribute nothing where evidence
  exists. A/B on the v2 reference sets: chair front-view contamination
  removed (10.4% of front pixels reverted to photo truth), zero regression
  on generated-exclusive surface.
- **Floor-accepts are reported, never baked** (`generate_reference_views`):
  selection now requires a STRICT pass on all three material oracles. The
  v2 chair measured why: a floor-accepted top view leaked stained fabric
  straight into the bake. A wrong texture on an unseen angle is a worse
  product defect than the featureless fill it displaces — fill is dull,
  wrong material is broken. Floor-only ladders surface in the report as
  `rejection_reason` with full per-attempt metrics.
- **Person subjects are refused unless explicitly acknowledged**
  (`person_policy`, default `"skip"`): the review measured generated side
  views drifting to a DIFFERENT person's face — different age, nose, skin
  — while every material gate strict-passed, because no gate in the stack
  measures facial identity. Both `auto` AND `on` refuse people: "on" is a
  texture-quality opt-in, not identity-synthesis consent. Synthesis of a
  person requires the person-specific acknowledgment
  (`allow_person_subjects` on `rebake_bundle`,
  `texture_reference_allow_person` on the backend,
  `--texture-reference-allow-person` on the CLI), which puts a
  `person_warning` on the record. The check FAILS CLOSED: the photo is
  captioned even when a non-person hint exists (a hint that doesn't name
  a person is not evidence of absence), and an unavailable captioner
  refuses instead of proceeding — an unavailable check is not a
  permission grant. Detection tokenizes alphabetic runs ("woman's"
  matches "woman") over a wide person-word list (incl. baby/human/bride);
  the robust upgrade path (face detector, identity-embedding floor) is
  tracked in the KnowledgeBase.
- **Whole-bake A/B acceptance gate** (`bake_acceptance.py`): per-view
  strict gates are structurally blind to COMPOSITION-level failure — on
  the chair, every shipped view strict-passed and the finished bake still
  regressed below the no-references baseline (a deltaE ~42 tone step
  where the generated top hands off to the protected front). When
  generated views enter a bake, the pipeline now also bakes the
  no-references baseline and ships the generated bake only if it does not
  regress three render-space axes: photo fidelity at the source pose,
  front brightness, and long coherent seam edges (extent-filtered so
  texture detail — grain, panel lines, plumage — never counts as a seam;
  the budget is ABSOLUTE, not a baseline multiple, because a multiplicative
  allowance lets a bad baseline launder a worse candidate). Calibrated on
  the labeled four-subject set: the chair auto-rejects (seam 0.138 vs
  ceiling 0.122) and ships its baseline with the verdict recorded in
  metadata; owl, spaceship, and portrait pass. Close-zoom triage of the
  owl's 13 dark-smear fragments localized all of them to the unwitnessed
  underside band (crevice shading speckle, worst per-view delta L -3.4 vs
  baseline); the close-range harness numbers stay on the record and the
  detector was left exactly as certified rather than recalibrated to
  flatter the feature.

### Changed (generated references v3 — zero-hint operation, material-identity gates)

- Reference generation is now fully autonomous: no subject hint is required
  anywhere in the API surface. When no user prompt exists, the source photo
  is captioned automatically (BLIP, `abstract3d.captioning`); when one does,
  it is used — and EITHER text is reduced to a material-free noun phrase
  (`extract_subject_noun`, a stoplist over material/finish/color vocabulary)
  before it may enter the generation prompt. Root cause, proven twice: any
  material claim in prompt text overrides the source photo's pixels (a
  hand-written "ceramic with glaze" hint regenerated a carved-wood owl as
  glazed pottery), while a correct claim adds nothing the photo doesn't
  carry. The prompt template has no free-text slot at all now; the source
  photo is the only material authority.
- New composite instruction (adversary-designed, subject-agnostic): leads
  with material-NEUTRAL relief vocabulary ("surface relief, carving depth,
  grooves, grain, cracks, fibers, micro-texture" — self-normalizing: for a
  smooth subject, copying its relief exactly yields smooth), names the
  output "a real photograph" (naming it a render biases CG-smooth output),
  maps materials PART BY PART (a wood-frame/fabric-seat chair must not
  spread one part's material onto the other), and forbids re-interpretation
  without naming any material class. A person clause (triggered by
  person-category caption words) anchors human subjects to "living person,
  real skin, real hair strands" — without it, i2i editors systematically
  render the clay panel as a sculpture.
- Acceptance is now a three-oracle gate stack run on the FINAL processed
  pixels (despecular and tone-match happen before gating, so the gate
  judges what the bake consumes), each catching a failure family the others
  are blind to, all calibrated on the critic-labeled v1+v2 result set:
  `texture_fidelity` (band-pass relief ratio + flat-fraction growth;
  catches wood→glaze smoothing), `part_material_fidelity` (k-means part
  palette, chroma-first distance with an L tolerance band for unseen-side
  shading; catches upholstery→camouflage flips the texture gate passes),
  and `gate_baked_speculars` (glossy highlight fields). Retry ladder:
  IoU failures re-roll the seed (stochastic), texture/material failures
  escalate the prompt (systematic bias); every IoU-passing candidate is
  scored, and only a STRICT pass may ship (see the adversarial round-2
  entry below — floor-only candidates are reported, never baked).
- Conditioning canvas fixes: both panels letterboxed (no anisotropic
  stretch of the material the model must copy), clay foreground composited
  onto the same dark background as the source panel (background mismatch
  read as "different photo sessions"), and the echo-crop heuristic now
  catches ANY wider-than-tall canvas echo (the old >=1.6 aspect test missed
  4:3 echoes and burned whole retry ladders).
- Despecular is relief-aware: pixels inside high band-pass-energy
  neighborhoods are exempt (carved-ridge micro-highlights satisfy the
  specular predicate; blending them toward the body estimate erased exactly
  the relief the transfer must preserve), and when the source photo itself
  flags a similar fraction under the same predicate, the correction blend
  is scaled down (measured 2% false-positive floor on matte carved wood).
- `auto` mode gate relaxed accordingly: it still requires an explicitly
  configured local image provider (never a silent remote route), but no
  longer requires a subject hint. Zero-hint four-subject validation
  (owl / chair / spaceship / portrait, FLUX.2-klein): 14/16 angles
  accepted with materials preserved; the two rejections are honest (a
  chair profile whose IoU never clears the gate, and a chair side whose
  camouflage-mottle material flip the part gate caught on every ladder
  attempt — that sector falls back to witnessed-texture fill, which
  cannot flip materials). Klein-9B resolves the portrait family (strict
  passes where 4B floor-accepts wet-look hair); documented as the
  recommended model for human subjects.
- KNOWN LIMIT (documented in KnowledgeBase): semantic re-rendering that
  preserves palette AND relief energy (v1's "sculpted goo" hair) is
  invisible to every foreground statistic tested; the countermeasure is
  generator quality (Klein-9B), not gating.

### Changed (generated references v2 — composite conditioning for source coherence)

- A coherence audit showed clay-only conditioning produced shape-correct
  but materially unfaithful views (the owl came back pale cream: LAB
  distance 28.4 from the source, hue correlation 0.44) — the i2i model
  never saw the source photo. The conditioning image is now a COMPOSITE:
  source photo (left panel) + clay render (right panel) with a
  texture-transfer instruction. Same model, coherence doubled: LAB
  distance 7.4, hue correlation 0.82; texture QA PASS on all three proof
  bundles including the previously-open fill-energy gate
  (`artifacts/validation/generated-reference-completion/`, v2).
- New `register_matte_to_clay` similarity registration (downsampled IoU
  search, winning transform applied at full resolution) absorbs the
  editor's small reframing before the acceptance gate, keeping the shape
  lock (raw composite IoU 0.74-0.83 -> registered 0.89-0.97).
- `conditioning` strategy parameter ("composite" default, "clay" and
  "rotate" available); provenance now records the strategy and per-attempt
  registration. Model notes: FLUX.2-klein-9B is HF-gated (stored token
  expired — refresh to enable); Qwen-Image-Edit-2511 8-bit downloads and
  registers but did not produce a first denoise step within 8 minutes on
  this host and is parked with the "rotate" strategy ready for it.

### Added (generated reference views — single-photo coverage completion)

- `abstract3d.reference_generation`: when a caller provides only ONE photo,
  the pipeline can synthesize the unseen angles and feed them into the
  certified bake as ordinary references: clay-render the reconstructed mesh
  from each target angle (silhouette lock; moderngl renderer REQUIRED — the
  matplotlib fallback's decimated silhouette would blind the gate), condition
  an `abstractvision` i2i generation on that render, gate acceptance on
  silhouette IoU >= 0.75 against the clay silhouette, suppress baked specular
  highlights (pale desaturated blobs vs the local diffuse body estimate),
  and cap-limited LAB tone matching toward the source photo (mean shift
  clamped per channel so a legitimately different unseen side is never
  whitewashed into the front photo's statistics; pre-match distance and the
  applied shift are recorded). Measured on the certified owl: observed
  coverage 0.30 -> 0.83 with four generated views (back/left/right/top),
  every acceptance first-attempt (IoU 0.92-0.98).
- Generated views are SUBORDINATED witnesses in the bake: their projection
  weights are attenuated (0.6) so they lose every per-texel contest against
  real photo content; the source view keeps its single-view facing semantics
  and scarcity-rescue stays off unless a REAL reference exists (generated
  views must not flip the certified single-photo regime). `observed_view_stats`
  rows and bundle metadata mark generated views explicitly.
- Hunyuan backend option `texture_reference_generation` (auto/on/off,
  default auto) with `texture_reference_generation_angles` (labels or
  `label:azimuth,elevation` entries — the validated starship underside is
  `bottom:0,-75`); CLI flags for both; `rebake_bundle` gains
  `generate_references` + `generation_angles` + `subject_hint`.
- Adversarially hardened before landing (1 controller agent, 15 findings):
  "auto" fires ONLY with an explicitly configured image provider (never the
  remote fallback route) AND a non-empty subject hint (the i2i model
  conditions on an untextured clay render; without subject knowledge it
  invents materials for exactly the default one-photo user) — otherwise it
  skips with an actionable warning; "on" expresses explicit intent. Full
  provenance is recorded per bundle: resolved provider/model, prompts,
  negative prompt, seeds, per-attempt IoU, accepted-image hashes, clay
  renderer, tone shifts; generated photos and their clay conditions are
  persisted as `texture_reference_generated_*.png`. The un-matted source
  photo now rides as `identity_image` on the backend's source view (the
  fringe-repair correspondence needs it). Honest scope, documented: a
  generated view is plausible synthesis, not ground truth — content on
  fully unobserved regions (a person's back of head) is invented, and the
  three side views are generated independently (no cross-view content
  consistency beyond tone).

### Added (executable golden-bake regression harness + public bundle API)

- `scripts/golden_bake.py` turns the certification's determinism claim into an
  executable gate: it rebakes the three certified proof assets through their
  canonical recipes and fails unless every baked `texture.png` reproduces the
  published hash bit-exactly. `--profile` adds per-stage wall time and RSS/MPS
  memory attribution (one process per asset so peaks stay attributable).
- `abstract3d.bundle` — the previously script-only rebake path is now a
  supported API: `load_bundle` / `prepare_observed_views` / `rebake_bundle`
  load a bundle's canonical `geometry.glb`, rebuild the observed-view list
  (source matting, reference angles, the identity-image contract), rebake,
  and write a versioned bundle revision (`schema_version`, texture md5,
  trimmed bake stats). Documented caveat: TripoSR bundles rebake without the
  resident triplane color prior; the certified Hunyuan bundles rebake with
  full fidelity.
- `abstract3d.profiling` — read-only stage/memory profiler (background RSS
  sampler + externally-wrapped stage functions), used by the harness;
  profiled runs stay bit-identical because nothing touches array state.

### Changed (strict generation-option contract)

- Backends now REJECT unknown generation options with the new typed
  `InvalidRequestError` instead of silently ignoring them (the CLI itself
  was sending diffusion knobs to the feed-forward TripoSR path with no
  effect and no warning). Each backend consumes its supported options and
  the leftovers fail loudly — before the expensive inference stage on the
  diffusion backends, so a typo costs milliseconds, not minutes. Envelope
  keys (`artifact_store`, `run_id`, `tags`, `metadata`) are exempt.
- The CLI forwards only explicitly-set flags (None-valued options are no
  longer sprayed at every backend) and exposes the mesh-density controls
  that were previously Python/config-only: `--octree-resolution` and
  `--max-facenum` (hunyuan3d21/step1x).
- Composed `t23d` image options are consumed through one helper
  (`pop_composition_kwargs`), so `image_provider/model/width/height/seed`
  are recognized composition keys on every backend and unknown `image_*`
  spellings fail like any other typo.

### Added (visual quality review protocol)

- `artifacts/validation/quality-review/`: reproducible per-backend quality
  scoring — two representative bundles per backend/task inspected through
  the headless MeshVault MCP server (structural `describe_scene` + three
  canonical renders each), with every render, the rubric, and per-group
  evidence versioned (`scores.json`). The benchmark table now carries
  mesh/texture quality scores and the model license per backend.

### Added (generation statistics + headless MeshVault verification)

- `scripts/generation_stats.py` aggregates wall time, stage times, and mesh
  density (vertices/faces) from every bundle `metadata.json`; the summary
  table and the time/density control matrix (what governs mesh size per
  backend, defaults, and Python-vs-CLI exposure) are published in
  `docs/benchmarks.md`. Known exposure gap recorded: `octree_resolution` /
  `max_facenum` are honored as Python kwargs and config keys but have no
  CLI flags yet.
- The rebaked assets were verified through the MeshVault MCP server driven
  headless over stdio JSON-RPC (`load_model` + `screenshot`), in addition
  to the interactive app check; proof render in
  `artifacts/validation/bake-performance-program/`.

### Changed (bake performance program — outputs bit-identical)

All optimizations below reproduce the certified texture hashes bit-exactly
(verified per-change on captured stage inputs AND end-to-end by the golden
harness; before/after evidence in
`artifacts/validation/bake-performance-program/`). Measured on the golden
recipes at res 2048 (Apple M5 Max): owl 258 s -> 88 s (2.9x), face
220 s -> 167 s (1.3x), ship 59 s -> 55 s (1.1x). Memory peaks -0.15 GB
(ship/owl); the peak-structure analysis is recorded in the profiles.

- `mirror_fill_from_observed`: the exact-NN mirror-twin lookup now runs
  parallel (`workers=-1`) and pruned at the acceptance threshold
  (`distance_upper_bound`) — most mirror twins land nowhere near an observed
  texel (1.6% acceptance measured on the owl), and unbounded exact-NN
  backtracking dominated the stage (167 s -> 1.1 s, 148x, bitwise-identical:
  pruned misses return inf and are dropped by the same `valid` mask).
- `synthesize_fill_detail`: donor k-NN queries go through `_balanced_query`,
  which randomizes query order before scipy's per-thread chunking (atlas-
  ordered queries give whole chunks of far-from-tree texels to one straggler
  thread) and undoes the permutation on return — exact same per-point
  results, 3.8x on the owl donor query. Full-atlas statistics intermediates
  (~0.5 GB) are released before the long query phase; the two observed
  quantiles they feed are computed ahead, unchanged.
- `commit_pale_chips`: per-blob work (masks, isolation dilation, gathers)
  now runs inside each blob's bounding window via `find_objects` (margin
  covers the dilation) with the loop-invariant plain-domain colors hoisted —
  474 committed / 2264 candidate blobs previously paid full-atlas ops each
  (42.8 s -> 0.8 s, 53x, bitwise-identical).
- `commit_trace_deposits`: eval units are stored as (window, local-mask)
  pairs; world-space ring/residue tests evaluate over precomputed flat
  domains (row-major extraction preserves reduction order bit-exactly);
  full-atlas masks are materialized only for units that actually commit
  (17.7 s -> 6.9 s on the face proof, bitwise-identical).
- Index maps for the flat domains use int32 (identical indexing behavior,
  half the footprint).

## 0.2.0 (2026-07-08)

First public release of the standalone repository (`github.com/lpalbou/abstract3d`).
Validated operating profile: Apple Silicon (`mps`), Python 3.12. See `README.md`
for the current backend/OS support matrix.

### Release engineering

- CI/CD on GitHub Actions (`.github/workflows/release.yml`): test matrix
  (ubuntu + macOS, CPU torch, headless GL), sdist/wheel build with twine check,
  GitHub release on `v*` tags, optional PyPI publication (skips without the
  `PYPI_API_TOKEN` secret), and a MkDocs Material doc site deployed to GitHub
  Pages (`https://lpalbou.github.io/abstract3d/`).
- Versioning policy for validation artifacts: only current-state experiments
  are versioned (certified bundles, certification record, generated-reference
  proofs); superseded experiment archives stay local (`.gitignore` allowlist).
- Cross-host portability fixes surfaced by CI: the Hunyuan license gate now
  fires before the optional-dependency check; Step1X seeding falls back to a
  CPU torch generator on builds without the resident backend; the
  gradient-domain determinism test encodes the portable contract (<= 1 LSB at
  quantization boundaries) while bit-identity remains the measured guarantee
  on the validated Apple-local profile.

### Cycle 8 — viewer-orientation export (EXP-01)

- Textured exports now present the glTF viewer frame: the pipeline's canonical object
  frame is Z-up / front +X while glTF mandates Y-up / front +Z, so every
  standards-compliant viewer displayed exports lying sideways. `_mesh_export_bytes` and
  the OBJ exporter bake the exact axis permutation (x, y, z) -> (y, z, x) into exported
  vertices (float-exact; texture bytes verified byte-identical) and stamp a persisted
  `abstract3d_export_frame` marker (glTF extras, survives round-trips). The repo
  renderer and the texture-QA harness detect the marker and rotate marked meshes back
  into canonical-frame math, so all gates measure identically (verified: face raw
  identity improved to SSIM 0.676 under the marker-compensated render; texture_qa
  13/13 on all three assets). Internal working files (`geometry.glb`, consumed by
  rebakes) keep the canonical frame via `viewer_frame=False`. All three certified
  bundles re-exported upright; MeshVault verification:
  `artifacts/validation/texture-cycle-proofs/upright_verification.png`.

### Cycle 7 — reference leverage

#### Hardened (MANDATORY first item of this pipeline change, per the certification contract)

- `feature_fringe_repair._render_structure_veto` — CUMULATIVE-BASELINE VETO
  (critic 2's cycle-5 recommendation, adopted as mandatory by the cycle-6
  certification): the advancing per-stamp baseline re-armed the +0.0003
  micro-island budget with every acceptance (measured: ~7 stamps produced
  +0.00096 at one view, triple the single-stamp budget, inside the letter of
  every per-stamp check). The veto now also refuses any candidate whose
  post-stamp micro fraction exceeds BOTH the view's ORIGINAL pre-repair
  fraction + 0.0003 AND the original battery-wide worst; the photo-truth
  exemption bound is pinned to the ORIGINAL battery worst for the same
  reason. Per-stamp advancing semantics unchanged. MD5-neutral on the
  certified face: the canonical-recipe 2048 bake reproduces
  `928705f3edfc9036348c12bf34435d9d` bit-exactly (predicted by critic 2's own
  measurement — the accepted creep stayed under the original battery worst).
  Test: `test_render_veto_cumulative_baseline_closes_rearm_creep`.

#### Added (reference-leverage ledger — permanent instrumentation)

- The project owner's standing critique ("the pipeline under-leverages the
  reference photos") is now measurable per bake: `bake_projection_texture`
  stats carry `leverage` — per view potential/painted/won texels with
  per-gate surrender attribution (facing gate / layered-zone gate /
  downstream kills / union drops), plus union ratios (photo-visible,
  direct-painted, leverage, surrendered-visible, unobservable) and the
  mirror-over-photo-visible watch (G4). The projector emits per-view
  diagnostic maps (`potential`, `zone`, `facing`, `geometry_factor`,
  `scarce_weight`, exact per-texel stretch). `scripts/texture_qa.py` prints
  the ledger as a non-gating reporting block and stores it in results.json.
  Instrumentation is md5-neutral (pinned-vs-current 1024 pair bit-identical).
- Measured on the certified face at 1024 (the honest inventory the critique
  asked for): photo-visible union 50.7% of surface, direct-painted 45.0%,
  leverage 88.8%, photo-visible-but-surrendered 5.8% (12,449 texels),
  unobservable 49.3%. The "sees 57% / paints 21%" reading compared the
  geometric-visibility union against the CONFIDENT-weight set (winner weight
  >= 0.35 = 25.0%); the painted-at-any-weight set was already 45%.

#### Added (G1: witness-scarcity admission — `admit_scarce_witnesses`)

- On texels NO view claims at its strict facing threshold (the certified
  bake surrendered 4.8% of the surface that at least one photo sees:
  jaw/cheek silhouette bands, under-chin, hairline/crown transitions), the
  bake now admits below-threshold witness claims bounded by the EXACT
  per-texel sampling stretch (<= 4.0, the texel->photo Jacobian — facing is
  a tilt proxy and cannot see collapsed mappings), above a grazing floor
  (facing > 0.05), still respecting first-surface visibility, photo alpha,
  the layered-zone surrender, and the stretch/concavity demotion.
  "Stretched content beats no content" — the single-view doctrine —
  generalized to per-texel witness scarcity; where ANY strict witness
  exists, the calibrated strict gates keep the texel and every scarce claim
  stays discarded.
- Admission guards (each measured load-bearing at 1024; unguarded admission
  lifted dark_debris 0.0022 -> 0.0038 vs the 0.003 gate): (1) contradiction
  of a color-consistent confident consensus (the mirror-copy guard's rule);
  (2) like-material support in BOTH directions (dark-on-bright is the
  debris/flake class, bright-on-dark is the FACE-07 pale-chip class —
  measured crown-flake failures 0 -> 0.0022 without it); (3) dark-mass
  adjacency (dark commitment requires the dark BODY; a nearby dark FEATURE
  licenses nothing); (4) a FEATURE MOAT (no admission within 0.044 x scale
  of a strong dark feature core — parallax-displaced feature adjacency was
  the measured debris source, and features are strictly witnessed by
  construction so the moat costs no leverage).
- Placement (two measured non-local failures): admission happens AFTER the
  global compositing solve as a strictly local paint (an early admission
  re-shaded photo-true content 20+ px away through the Poisson anchor set
  and flipped three knife-edge debris detectors; the fringe stage's
  pre-repair baseline inherited the drift and its exemption bound loosened)
  and BEFORE mirror completion (real observation beats symmetry guess —
  mirror no longer guesses surface a real witness paints). Rescued texels
  inherit the delight/harmonization tone corrections (the application masks
  now include scarce candidates; the fits never see them).
- `consolidate_unwitnessed_debris`: render-informed lift (the fringe lane's
  displaced-refill discipline) of isolated bright-ringed sub-feature dark
  islands whose first-surface texels are predominantly UNWITNESSED — fill
  pockets re-partitioned by the admission that the fill floor's
  anchor-tracking exemption correctly keeps but the absolute debris
  detectors count. Runs only when scarcity admission ran (bakes without it
  stay bit-identical); the island construction anchors its dark split to
  the light material's own median (the binding detectors' construction)
  and feature-class blobs are protected by the render battery's own
  footprint.
- `scarcity_rescue="auto"` enables the mechanism for multi-view bakes only;
  single-photo proof assets are pinned regression canaries: fresh 2048
  bakes with the change ON reproduce ship `b8e2b0d4...` and owl
  `ff746509...` bit-exactly. A measurement-only single-view ablation
  (ship, rescue forced on) admits 578 texels (+0.1 point leverage,
  texture_qa 13/13) — not worth re-certifying a frozen canary, so the
  auto scope stands.
- Face results (all gates green at both resolutions): at 1024,
  direct-painted 45.0% -> 45.4%, comp identity 0.705/14.7, detectors
  green, texture_qa 13/13, bit-deterministic. At 2048 (canonical recipe,
  determinism pair `2baf7408...`): compensated battery PASS 0 failures
  (front 0.70182/14.876), raw detectors green with the worst dark
  IMPROVED (0.0027 -> 0.00262) and comp MAE margin IMPROVED
  (0.09 -> 0.124), texture_qa 13/13, direct-painted 43.4% (+8,224
  photo-witnessed texels at the jaw/cheek/under-chin bands).
- PUBLICATION: the 2048 candidate is STAGED, NOT published. The comp SSIM
  knife-edge consumed 50.8% of its certified margin (0.7037 -> 0.70182,
  half-margin line 0.00185, consumed 0.00188) — per the certification's
  maintenance contract §5, more than half of a knife-edge margin requires
  a fresh critic battery, not just the harnesses. The certified bundle
  (`928705f3`) stays on disk; staged bytes + full harness evidence:
  `/tmp/c7/staging_face2048` and `/tmp/c7/REPORT.md`.

### Certified (zero-defect adversarial program, cycles 1-6)

- The three proof assets (multi-view face, single-view starship, single-view owl) are
  CERTIFIED at the zero-open-defect standard by the program's independent verdict agent:
  23 defect-ledger entries FIXED, 10 closed as PROVEN-LIMIT with the exact capture remedy
  documented per entry, 0 OPEN. Certification document (ledger state, maintenance
  contract, knife-edge watch thresholds, and the honest definition of zero-defect within
  the given inputs): `artifacts/validation/texture-cycle-proofs/CERTIFICATION.md`. Final
  face state: compensated identity 0.704/14.9 (anchored gate 0.70/15.0) with the full
  28-view compensated battery at zero failures, raw detectors green, texture_qa 13/13 on
  all three assets, and four independent canonical-recipe bakes sharing one texture hash
  (bit-deterministic pipeline). All six cycle rulings, both critics' mathematical
  reviews, and every solver report are preserved under
  `artifacts/validation/texture-cycle-proofs/`.
- Late-cycle mechanisms (each adversarially verified before certification):
  gradient-domain view compositing (screened Poisson over the texel surface graph),
  validated dense reference flow, film-band gradient repaint with off-pose displacement
  veto, feature-fringe repair driven by the identity gate's own correspondence,
  shadow-apron reconciliation (source cast-shadow truth vs reference lit tone),
  world-space voxel-graph feature-complex clustering, trace-deposit commit with rim
  feathering, field-support bounds on geodesic tone extrapolation, and completion tone
  matching — plus the publication checklist born from the FACE-21 incident (no bake
  ships without `identity_image` and pre-overwrite harness verification).
- Adopted forward-process governance from the certification: Critic 2's cumulative-veto
  hardening for the fringe lane's growth budget is the MANDATORY first item of any
  future pipeline change, and any texturing change re-proves determinism, re-runs the
  full gate set on staged bytes, and re-verifies the frozen canary hashes before
  publication.

### Fixed (FACE-22: region-boundary line-art on smooth skin — cycle 6)

Thin line-art contours on the neck/chest (a glyph-like cluster at az0 and
a large closed contour at az-22.5), pipeline-attributed (both the front
photo and the reference profiles are clean under contrast stretch at
those regions). Provenance established by a fully instrumented bake
(per-stage texture captures + per-mechanism ablation bakes + internal
mask captures at 2048; all difference maps in the cycle-6 evidence):
three mechanisms printed the marks, each fixed in its own vocabulary.

- `film_band_gradient.repaint_film_band` — FIELD SUPPORT BOUND
  (`FIELD_SUPPORT_TRANSITIONS`): the geodesic S field is a distance
  RATIO and takes mid-transition values arbitrarily far from the hair
  mass, so the repaint treated neck/chest skin at 9-24 pooled-profile
  transition lengths from the mass (S~0.66) where the measured falloff
  profile has no support: its envelope clamp printed the az0 glyph
  cluster (small clamp components, hp -0.058), its stamp borders and
  displaced-refill component borders printed contour segments and the
  az-22.5 closed contour. Treatment is now confined to within 6
  transition lengths of the mass (film strokes measured d_mass p5 8.8T
  vs the honest apron's p50 2.0T), feathered over the last transition;
  and authority stamps blend composite -> photo over
  `STAMP_BORDER_FEATHER_TEXELS` at treated-region borders (mass borders
  exempt — the stamp continues the mass content there), which also
  removes the support-cut chroma seams (measured 0.49-0.69 -> 0.13-0.23
  at az+22.5/+70).
- `texturing.commit_trace_deposits` — RIM FEATHER (`rim_feather_texels`):
  the deposit's antialiased border mixtures sit below `deviation_min` by
  construction (mixture deviation = coverage x deposit deviation), so
  the commit retoned the interior and left a 1-3 texel dark outline —
  the az-22.5 closed contour's crisp component. Rim texels carrying the
  same evidence class (direct, trace-weight, bright ball context,
  outside the film commit) now blend toward the ring-anchor tone,
  distance-decayed and ONE-SIDED (only darker-than-target texels move);
  the interpolation anchors exclude the feather band itself (a rim
  mixture bright enough to be an anchor otherwise pins its own darkness
  in place — measured: 483 -> 2678 feathered texels after exclusion).
- `texturing.tone_match_completion_components` (new, called from the
  mirror-completion block, multi-view bakes only): mirror completion
  copies the twin verbatim; on a lighting-asymmetric subject each copy
  lands at a tone offset from its destination (measured +16/255 on the
  chest) and its border prints as a contour. The legacy seam leveling
  reconciled mirror regions, but the gradient-domain solve runs BEFORE
  mirror completion — this is the missing handoff. Pure-bright copies
  against bright destination rings take a component-level log-median
  gain (clamped, detail verbatim); mixed-material copies and dark-ring
  components stay verbatim (measured: rescaling them re-classifies
  their own dark micro-content and mints dark_debris islands,
  0.0031-0.0036 vs the 0.003 gate).

Measured at 2048 (canonical recipe): the glyph cluster and the closed
contour are gone at the critic's crop framings; stroke-texel detector
inside the two probe boxes 4281 -> 3474 (glyph probe 36 -> 21, az-22.5
probe 2126 -> 1538; the remainder is soft pre-existing blend-content
boundaries, not line-art). Compensated battery PASS (identity[front]
0.7037/14.91 vs gate 0.70/15.0), raw battery: detectors all green, raw
front MAE 21.67 green (SSIM raw-diagnostic per the cycle-4 ruling),
texture_qa PASS 13/13. Single-photo canaries (ship/owl) bit-identical
(the new tone match is multi-view-gated; the film repaint and the
commit are structurally multi-view-only).

### Added (source-shadow apron reconciliation — the FACE-04/FACE-14 neck wash)

`gradient_compositing.reconcile_shadow_aprons` (+ `apply_shadow_apron_scale`),
wired into `composite_gradient_domain` beside the specular reconcile: the
DUAL of the FACE-05 mechanism. Where a REFERENCE view wins co-witnessed
surface with its own lit reading while the SOURCE photo (the identity
contract holder) validly samples the same surface substantially darker —
its cast shadow (chin/jaw onto the neck: measured -0.35 log vs a -0.08
pairwise gauge, source projection weight ~0 at the down-sloping neck) —
the composite carries the source's shading baseline there. The identity
gate at the source pose compares that surface against the source photo,
and the renderer's flat-biased headlight (~0.9 at the neck) cannot absorb
a real cast shadow, so only the albedo can carry it. Guards, each with a
measured counterexample: source-valid-only (no witness demotion where the
source has no evidence — the photo-curtain parallax band beside the wash
stays untreated), pairwise lighting gauge + margin (exposure differences
are not shadows), source-photo edge-density refusal (the curtain edge is
edge-dense; a shadow is smooth — refused components measured p85 2.8-5.8
vs the shadow's 0.5-1.3), world-ball fragment merge before the size floor
(the atlas cuts one apron into sub-floor UV fragments), one-sided
darkening only with per-consumer detail preserved verbatim (the correction
reduces to a smooth luminance scale; no reference chroma or detail is
imported, no brightening path exists). Measured at 2048 (canonical recipe,
paired A/B): compensated identity[front] +0.005 SSIM / -0.2 MAE, sides
within budgets, all detectors green; single-view bakes are structural
no-ops (no reference can win a texel) — ship/owl canaries bit-identical.

### Changed (feature-fringe repair: world-space complex formation + photo-truth exemption)

Three cycle-5 changes to `feature_fringe_repair`, each closing a measured
2048-resolution shortfall of the cycle-4 mechanism:

- Complex formation now clusters core texels in WORLD SPACE
  (`_cluster_core_texels_world`: voxel-graph connected components at a
  link cell of 0.006x the mesh diagonal, the rescue detector's
  construction) instead of atlas morphology. Atlas dilation counts
  TEXELS, so the same world gap spans twice the texels at 2048 and UV
  chart cuts fragment one physical feature into sub-floor pieces before
  any world merge can see them (measured: the mouth complex formed at
  r 0.045 vs the 1024 run's 0.11 and the lip-edge dash stayed
  half-covered; the chin complex never formed at all). World clustering
  is resolution-independent and chart-blind by construction.
- The render-space structure veto's micro-island growth budget carries a
  PHOTO-TRUTH EXEMPTION: a new sub-feature island whose pixels render
  from stamped texels the registered photo CONFIRMS is the photo's own
  anatomy (lip-corner line, lash fragments), not invented structure —
  measured: the chin/mouth-surround stamp that banks +0.006 compensated
  SSIM was refused for printing exactly what the photo prescribes. The
  exemption is BOUNDED by the battery's own pre-repair worst case (no
  view may become the new worst micro-island offender; measured
  unbounded, the eye complex's full re-registration pushed two views
  past the absolute debris detectors at 0.0030/0.0032 vs the 0.003
  gate) and feature-size new blobs stay banned unconditionally. The
  veto baseline now advances with each ACCEPTED stamp so a stamp's own
  exempted content is not counted as growth against later candidates.
- A final render-informed speck consolidation
  (`_consolidate_render_specks`) lifts repaired texels that render as
  NEW isolated sub-feature dark islands at any battery pose (relative
  to the pre-repair baseline) to just above the dark class under that
  view's own shading — the FACE-20 displaced-refill floor discipline at
  micro scale — with texels rendering inside any pre-existing
  feature-class blob's own pixels protected (measured: an unprotected
  lift brightened the az+90 profile eye's under-lash mass and
  eye_count dropped 1 -> 0). Whole-complex stamps also apply an
  in-stamp speck guard lifting stamp-made bright-ringed micro specks.

Measured end-to-end at 2048 (canonical recipe, all cycle-5 changes,
paired against the cycle-4 published baseline): compensated
identity[front] 0.688/14.42 -> 0.702/13.86 (comp gate 0.70/15.0 PASSES),
raw MAE 21.69 -> 21.45 (budget 22.0), full 28-view detector battery
green, texture_qa PASS 13/13, bit-deterministic across three bakes; at
1024 the full compensated battery stays PASS (0.705/14.8). Ship/owl
canaries bit-identical to the certified on-disk hashes.

### Added (feature-fringe repair — the protected-feature deposit class)

`feature_fringe_repair.repair_feature_fringes` (new module, wired into
`bake_projection_texture` after the fill floor, multi-view only): the
FACE-03/04 residue that `commit_trace_deposits` measurably cannot treat —
displaced-content chips and dashes INSIDE protected feature complexes
(tear-duct whites, lash-line dashes, the lip-edge dark-red dash), whose
surround consensus is feature-mixed by construction (cycle-3: ring votes
0.30-0.81 vs the 0.96 bar; committing them cost eye_count) — is repaired
with the photo's own content under the identity correspondence. The stage
rebuilds the identity gate's own registration in-bake (render at the
declared source pose, alpha-bbox map + NCC-refined similarity against the
caller-provided `identity_image`), z-buffers first-surface visibility,
and applies rescue-disc transplant semantics (tone match + feather +
whole patch) at two scales: whole-complex corrective stamps (mode ladder
full -> trace-only under a never-demote rule: non-source confident
content is never overwritten; source-confident content may be
re-registered to the gate correspondence) and deposit-scale patches.
Rescue-disc interiors are never photo-stamped (the disc fired because the
photo evidence there is bad); their fringe deposits re-copy through the
disc's own anchored correspondence and the disc is refreshed last so
healthy-side repairs propagate into the twin (the transplanted eye's
tear-duct chip was measured to be a COPY of the healthy side's chip).
Every stamp passes structure-preservation vetoes: a texel-space check
under the renderer's own shading model, then a render-space check with
the pipeline renderer at 15 views (no new/lost anatomical-feature-size
compact dark blob vs the pre-repair render; sub-feature micro-island
fraction budget +0.0003). Measured at 1024 (face proof, paired A/B):
compensated identity[front] 0.668/16.26 -> 0.708/14.7 (raw 0.643/21.6 ->
0.680/21.4), full 28-view battery PASS with zero detector regressions;
tear-duct/lash-line/lip-edge crops visibly repaired at 4x. Ship/owl
canaries bit-identical (single-view structural no-op, enforced by test).

### Added (off-pose displacement veto — the FACE-20 billboard strokes)

`film_band_gradient`: the hairline gradient repaint's source-authority
stage stamped hard BLACK stroke/arc artifacts across five-plus views (a
jagged crack down the left temple at az0, a feathered streak along the
temple silhouette at az-22.5, a ragged line at the az-90 hairline, black
arcs tracing the ear helix at az+90/+112.5). Provenance (replay-traced
per stroke): every stroke component was 80-99% source authority stamps
plus their gap-diffusion extension, carrying the front photo's own
CURTAIN-EDGE / EAR-SHADOW pixels (photo-space bins within ~1 transition
length of the dark-body boundary) billboarded onto surface the source ray
only grazes — and every component was ALREADY VETOED by another view's
base-material witness (veto consensus 0.7-1.0) while sitting fully
outside the feature moat, the only place the shipped mechanism consulted
the veto. The fix extends the veto by field position instead of by moat:
a connected would-be dark-stamp component whose texels are mostly vetoed
(>= 0.5) AND whose median S sits in the skin half of the transition
(>= 0.35, where the photos' own pooled falloff says the surface has left
the hair body) is rejected as parallax-displaced content
(`_displaced_stamp_components`). Near the mass (S below the gate) equally
vetoed dark stamps remain — they are the wisp/strand content whose global
veto was measured at -0.05 SSIM in cycle 3. Rejected sites refill AFTER
all guards: the local guard tone rescaled to the photo's luminance
pattern at 0.30 gain on a floor 1.02x the dark-material split — strictly
above the dark class, so the site cannot render as a dark stroke at any
pose by construction, while still paying part of the source-pose identity
contract. Measured (2048 face, same-tree A/B, fringe stage isolated): all
40 stroke-class components dead across the 48-view battery vs the C2
baseline with zero new flags introduced by the veto (residual flags are
shared 1:1 with the veto-off arm); front identity comp 0.674 vs 0.637
(veto-off, same tree) / raw 0.648 vs 0.612; side identities unchanged or
better; all 28-view detectors green; texture_qa PASS 13/13. Single-view
bakes structurally unreachable (ship/owl md5 pairs verified identical
with the veto forced on/off).

### Added (specular-lobe reconciliation in the gradient compositor — the pale seam column)

The pale desaturated column running inner-eye -> nose flank -> philtrum at
4x (three cycles open) was provenance-traced to the SOURCE PHOTO'S OWN
BAKED SPECULAR: under the estimated +20 deg head turn the nose-ridge
highlight projects onto the left nose flank (photo lum 218 vs 188 lateral
surround, saturation 40 vs 57 — the bright+desaturated signature), and the
screened-Poisson composite faithfully preserves it (the column exists in
the pre-solve blend; the solve moves it by <= 5/255; membrane/rail count
inside it is ZERO — neither a membrane path nor a selection boundary).
`gradient_compositing.reconcile_specular_lobes` (new, applied inside
`composite_gradient_domain` before gradient selection) reconciles the
source view's smooth bright+desaturated lobes against the cross-view
diffuse consensus: another view's valid sample reading the same surface
darker beyond the pairwise lighting gauge authorizes the lobe as
view-dependent light; the correction rebuilds those texels from the
source's OWN surround tone plus its own log-detail (no reference color is
ever imported), with saturation restored toward the surround. Feature
protection: edge-dense components (sclera/teeth class) refused by an
own-photo Scharr gate NORMALIZED to the reference resolution (a
fixed-world edge halves its per-texel response at 2048, and the
uncorrected 1024-calibrated bar stopped refusing eye-adjacent
components); dark-material context excluded; a DARK-CONTENT STANDOFF
feathers the correction to zero near texels substantially darker than
the surround (leveling the bright base right against dark micro-content
unmasked it into the debris counter — measured 0.0040-0.0054 at five
views without the standoff, all green with it under a frozen
downstream); reference-view lobes deliberately out of scope (measured
-0.005 side identity for no ledger gain). Single-view bakes no-op
structurally (no second witness). Measured: the az0 4x column is gone at
1024 (identity cost -0.003 raw SSIM); at 2048 the mechanism IMPROVES
identity[front] raw +0.006 SSIM / -0.1 MAE and comp +0.005 / -0.2 —
sides unchanged, texture_qa PASS, ship/owl bit-identical.

### Added (pale-chip commit — the dark-context dual of the trace-deposit commit)

`texturing.commit_pale_chips`: isolated PALE islands in DARK material
context (the FACE-07 ear-band class) — skin/mixture content displaced into
hair at trace witness weight, plus completion texels that copied those
anchors (measured population at both ears: 35-60% fill) — are vacated and
retoned from their validated dark ring anchors when every qualifying
witness reads the blob's plain 3D ring uniformly dark (>= 96% dark votes,
cover/single-cover gates as the bright-context commit). Guards: confident
witnesses never touched (trace w50 <= 0.30); chips 2-connected to a big
bright component are frontier slivers of real material and refused; area
cap 1.2e-3 of direct texels (without it a 700-texel rear blob committed
into a visibly flat gray wash — measured); film-commit and rescue
territories excluded; >= 2 projections required (single-view ring
consensus is vacuous), so single-photo canaries are structurally
untouchable. Measured at 1024: ear-band chips visibly reduced at
az+-90/112.5 4x, detectors within noise, identity unchanged.

### Added (synthetic cut-face toning — bust disc tone from its own rim)

`texturing.tone_bottom_cap`: the truncated bust's planar cut face is
synthetic geometry no photo witnessed, yet the global harmonic fill toned
it with a tan/taupe marble fed by rear-hair and neck anchors (FACE-12's
disc wash). The cut face is detected geometrically (down-facing planar
component >= 0.5% of surface, direct witness < 1%, thin slab) and toned by
inverse-distance interpolation of its OWN RIM's observed content (chest
skin at the front rim, hair curtain at the rear), smoothed at 24 texels,
keeping 60% of the cap's log-detail. Multi-view bakes only this cycle
(single-photo proof assets are pinned regression canaries — same scoping
precedent as the strand comb).

### Added (film-band gradient repaint — hairline apron tone from the photos' own falloff)

The committed film band (cycle-2 mechanism) still rendered as a smooth
putty-taupe stripe at 2x from the declared pose: the mesh fuses the wispy
hairline into a smooth APRON tens of texels wide, every photo compresses
its narrow (4-10 px) wisp-transition ribbon across that whole apron (the
front view's bins sit at median 1 px from its dark body across the apron),
and the commit's retone covered only ~8% of the visible band with an
attenuated pull. `film_band_gradient.repaint_film_band` (new) rebuilds the
apron:

- geodesic profile field on the texel surface graph: two multi-source
  Dijkstra fields (photo-confirmed dark mass; photo-space skin ring) and
  the photos' own skin-side falloff profiles pooled into S(u) give a tone
  target that is near-black at the hair-mass boundary and blends into the
  local skin tone at the face edge — the photo's own gradient;
- source authority: apron texels the source view images first-surface at
  solid alpha take the source photo's color verbatim (real strand layout;
  statistical tone alone measured 0.60-0.62 identity vs 0.65-0.69 with
  content) under measured guards: base-material witness veto inside the
  feature moat (the parallax-doubled-brow / third-eye class), standoff
  from reference-confident and reference-dominant territory (side
  identity contracts; side worst-window 0.116 -> 0.031 without it, crown
  flakes at az-70/-90), outermost-sheet depth corridor (inner curtain
  sheets sprayed skin shreds at 1024), feathered domain borders (hard
  edges printed dark crease lines at az-35);
- gap diffusion + envelope clamp: unreachable apron texels take
  graph-diffused stamp colors under a field-consistency gate; remaining
  over-envelope texels clamp one-sidedly (darkening only — brows, lashes
  and all legitimately dark content untouchable by construction);
- island guards on the final state: small treated dark components with no
  pre-existing dark-observed anchor revert; bright shell components
  disconnected from both the skin ring and protected blobs pull to the
  envelope;
- repainted texels are exempt from the fill-luminance floor (they carry
  the photos' falloff, not fill statistics; the floor re-lifted darkened
  curtain texels into pale shreds, measured at 1024);
- sampling floor: the mechanism requires the hairline transition to span
  >= 7 texels (measured working point 9.6 at 2048, failing point 4.8 at
  1024 on the face proof); below it the cycle-2 retone remains.

Face proof asset (2048, full verdict1 battery): failures 2 -> 1
(identity[front] MAE gate now passes at 21.5/22.0; SSIM 0.630 -> 0.651
against the 0.70 bar), all 28-view detectors green, side identities keep
their margins. Single-view bakes are untouched by construction
(starship texture bit-identical, mechanism-on vs off). Tests:
`tests/test_film_band_gradient.py`.

### Added (trace-deposit commit — multi-witness consensus retone for chip/dash debris)

The residual chip/dash class at close zoom (FACE-03/04/05 family: beige
flakes and gray dashes under the eyes, mouth-corner smears, chin flakes,
strap slivers): small deposits of DISPLACED view content that win texels
at TRACE witness weight on surface every confident witness reads as
uniform bright skin. Measured populations on the face proof at 1024:
chip blobs carry winner weight w50 0.02-0.29 while legitimate features
(lash lines, nostrils, lip borders) sit at w50 0.44-0.93 — weight
separates the classes where color-deviation thresholds measurably cannot
(cycle-2 negative results: flake deviation p50 0.12-0.26 vs legit
front-eye trace texels at 0.399).

`commit_trace_deposits` (new in texturing.py) retones such deposits from
their own validated surround, blob-by-blob, under film-band-style commit
semantics — every gate carries a measured counterexample from this cycle:

- blob-level trace gates (w50/w90): content ANY view confidently
  witnesses is never demoted;
- multi-witness bright consensus on the deposit's plain 3D ring (a
  world-space ball — atlas dilation crosses UV charts and picked up hair
  texels that veto valid commits); single-witness consensus requires
  dominant ring coverage, zero-witness consensus refuses (vacuous);
- BRIGHT deposits near a confident strong-contrast core (per-texel
  confident witness at high |contrast| — lash lines, sclera, lip
  borders) are refused: ambiguous with the feature's own fringe;
  committing them measurably washed the eye corner (eye_count 2->1 at
  az0 el10, 1->0 at ±90). Ball-mean witness cannot serve as the core
  signal: the ball mean around a trace chip is lifted by its confident
  surround (chin dash: ball weight 0.42 vs own w50 0.047);
- isolation: a dark deposit connected to a larger dark component (hair
  frontier whose dark side is unwitnessed fill, lip line) is never
  committed — committing frontier slivers painted pale streaks into the
  hair mass and dropped profile eye_count at ±90 el10;
- WHOLE-NEIGHBORHOOD rule: a blob commits only if every sub-threshold
  residue island inside its ring (mid-gray dashes, chip shadow edges at
  lum 0.45-0.60 on 0.73-median skin) is itself sweepable under the same
  consensus; partial cleanup UNMASKS the residue as new isolated dark
  islands on the cleaned surround (measured: dark_debris 0.0022 ->
  0.0037 at az0 without the rule, identical to control 0.0022 with it);
- retone from the validated BRIGHT ring anchors only (inverse-square 3D
  interpolation): membrane refill drags adjacent feature darkness across
  the hole (measured dark_debris 0.0024 -> 0.0044 at az-22.5), and the
  ring's own deviation filter admits feature-dark texels near
  boundaries;
- placement: runs late (after mirror completion, rescue, film retone;
  before detail synthesis) as a strictly local recolor — committing at
  the outlier stage cascaded through the Poisson anchors, rescue-disc
  localization and fill calibration (whole-face render diff mean 4.1/255,
  14% of pixels > 8/255) and flipped knife-edge detectors far from any
  chip; rescue-disc footprints are protected from both detection and
  retone (an unprotected retone erased the rescued -90 profile eye).

Multi-view bakes only (>= 2 projections): with one witness the ring
consensus collapses to the winner's own photo, exactly the
`commit_film_band` vacuity argument. Face proof A/B at 1024 (same pinned
tree, chips on vs off): dark_debris IDENTICAL to control at every gated
view, eye counts identical, identity[front] SSIM 0.648 vs 0.649 with MAE
21.0 vs 21.1, 50 blobs + 21 residue islands retoned; visible chip subset
(cheek/chin flakes, mouth-corner pale chips, bust-rim slivers, curtain
stripes) cleaned at 4x. Single-photo bakes are untouched by construction
(measured bit-identical ship/owl textures with the stage enabled vs
disabled).

### Added (strand-comb fill regime — combed low-contrast statistics for fiber material)

The rear hair fill read as leopard mottle (FACE-09): the blotch lives in
BOTH the coarse value-noise octaves of the fill detail pass (rosette
scale) and the harmonic membrane's tone wash (measured at 1000 px
renders: blotch statistic 4.6 for the raw membrane, 6.4 after the
default detail pass — the noise ADDS rosettes on hair-class fill).
`synthesize_fill_detail` gains an opt-in strand regime
(`strand_comb=True`; per-texel: donor anisotropy >= 0.40 AND base darker
than 0.55x the observed bright-half median):

- orientation from a MULTIGRID-propagated global field
  (`_multigrid_orientation_field`: anisotropy-weighted structure-tensor
  anchors pooled into coarse surface voxels, tensor diffusion over the
  voxel k-NN graph with seeded cells re-anchored) — donor-local
  orientation is noise deep inside the fill domain (solver-4 G3's
  measurement, |cos| p50 0.999 after propagation);
- carrier keeps only the finest octave, combed with extended LIC (48
  steps): the coarse octaves ARE the rosettes, and fine carriers buy
  more gradient energy per contrast unit, so the closed-loop energy
  calibration lands at visibly LOWER contrast for the same fill-energy
  gate;
- the BASE fill tone is advected along the same field (sparse
  index-doubling kernel, strides 1..2^8 LIC steps) so membrane tone
  blotches elongate into strand-parallel streams, and transferred
  amplitude is scaled 0.6x (elongated LOW-contrast statistics, the bar
  the owl's rear grain set).

Measured on the face proof at 1024: rear blotch 6.4 -> 5.0 (az180) and
7.9 -> 7.4 (az-135) against the 4.6 membrane floor, with fill/observed
Scharr energy 0.93 (gate >= 0.5). Enabled for multi-view bakes; single
photo proof bakes keep the default path (empty strand regime is
bit-identical by construction and by test), preserving the ship/owl
regression canaries.

### Fixed (SHIP-03 nose melt — projector-frame photo registration for ortho bakes)

At head-on views (az 0..30, el -20..+15) the starship's prow rendered as
"melted" smeared streaks. Root cause (measured, not the suspected
grazing-stretch demotion): the canonical recenter centers the PHOTO's
alpha-bbox at the frame center, while the orthographic projector centers
the WORLD ORIGIN — and away from the canonical front pose those two
centers diverge by the mesh bbox's projected offset (starship at
az+30/el+15: +54/-28 px at 1024; face at az+20/el+8: +16/+8 px; owl at
az0: ~1 px). Every photo sample therefore landed tens of pixels off the
surface that imaged it; at the prow (surface turning away, high content
gradient at the silhouette) the offset dragged dark under-hull and
background-adjacent content onto the nose and stretched rim content
across the concavity. A perfect-content synthetic-checker probe at the
same witness geometry measured the ceiling: with the offset, projected
content decorrelates from ground truth even at nominal sampling stretch
(binary checker agreement ~0.5 = chance); with the registration fixed,
agreement 0.72/0.71 at stretch 1.25-1.5/3-4 — the witness geometry
itself was never the limit at moderate stretch.

- `projected_frame_center_px` (new): the pixel where the mesh's
  camera-plane bbox center lands under the projector's own convention —
  deterministic, no content-based search.
- `recenter_to_canonical_frame` gains `center_px`; the ortho bake path
  registers views to the projector frame for OVERRIDDEN source poses
  (external capture facts the model never consumed; references keep
  their content-based residual registration on top). ESTIMATED poses
  (gradient_ncc) keep the legacy frame: the estimator searched az/el for
  the best gradient alignment of the legacy-centered photo, so pose and
  frame are co-adapted — re-centering one side alone was measured worse
  on the face proof (verdict1 failures 2 -> 10, front SSIM
  0.630 -> 0.598); registering the estimator itself to the projector
  frame is future work that belongs to the face lane. At the canonical
  front the two conventions agree to ~1 px by construction.
- Bake stats and bundle metadata record `source_registration`
  (`mesh_bbox_center`, dx/dy px); `scripts/texture_qa.py` reconstructs
  per-view visibility from the same frame so region attribution stays
  faithful (absent key = legacy behavior, old bundles unaffected).
- `synthesize_fill_detail`: transferred amplitude is FLOORED at the
  observed population's p25 raw-residual amplitude (per channel).
  Grazing-smeared donors carry artificially quiet statistics; fill
  anchored by them shipped as literal flat plateaus with straight
  chart-edge boundaries (an 11k-texel flat cell tripped
  texel.facet_cellular 0.092 vs 0.091 at 2048 after the registration
  fix exposed it). With the floor: facet_cellular 0.012, fill energy
  0.615 -> 0.620, sigma guard and granite test untouched.

Starship A/B (same tree, only the registration): source-pose render vs
photo MAE 45.5 -> 18.1, SSIM 0.092 -> 0.600; az0 4x nose crops go from
molten streaks to readable intake/grill structure; `texture_qa` PASS
13/13 at 1024 AND 2048 (dark smears 0 at both). The prescribed
alternative lever — steepening the Jacobian stretch demotion into a
coverage vacate — was prototyped and measured NOT better: cutoff 2.0
cleared residual streak anchors but surrendered 52% of witnessed
coverage (src-pose MAE 24.3, SSIM 0.436) and at cutoff >= 3 the melt
stayed; the negative result and numbers are in the cycle report. Owl
(estimated/declined pose -> legacy frame): bit-identical code path,
PASS 13/13 at both resolutions. Face (estimated pose -> legacy frame):
bit-identical code path, texture_qa PASS 13/13 at both resolutions,
verdict1 failure set unchanged vs the tree baseline (2 identity
failures, both pre-existing).

### Added (film-band commitment — multi-view material consensus for fused film bands)

The temple/hairline "film band" defect class (beige painted sheet, black
parting flecks, skin-flake mottle interleaved with dark curls): generated
meshes fuse wispy hair films INTO the head as one surface, so the
layered-density zone gate cannot see them (no second sheet => layered
density 0.02-0.05 << the 0.10 gate) while the photo pixels there are
bright skin+hair mixtures whose stamps win texels and read as painted
sheet; surrendered/unobserved remainders inherit the harmonic membrane's
mixed skin+hair tone (pale curtain).

`film_band.py` (new) adds a multi-view MATERIAL COMMITMENT on top of the
zone gate, all scale-free and subject-agnostic:

- per view (computed in the projector alongside the zone gate): the
  strong zone grows into connected weak evidence — any layered density at
  contrast, near the photo's dark-material main body, with substantial
  dark coverage of the window's foreground (the foreground normalization
  keeps silhouette rims meaningful); small components are dropped
  (membrane handles local ambiguity). Each view also carries a base
  WITNESS VETO map (imaged bins with no zone flag and < 0.25 dark
  coverage witness base material along their ray).
- commitment (`commit_film_band`) requires: some view's large-component
  extension flags the texel first-surface, NO view vetoes it, EVERY view
  imaging it first-surface flags it (flag consensus — a fused wisp
  floater aligns with the dark body from one pose only; committing it
  detaches under parallax into a floating dark blob, the "third-eye"
  class), and at least two imaging witnesses (single-witness consensus is
  vacuous; measured painting dark spots at ear-rim/crown silhouette skin).
- commit-coupled surrender: at committed texels whose local observed
  context is dark-dominated (voxel-ball dark/bright claim ratio at two
  scales), BRIGHT mixture claims of every view are vacated; dark claims
  are film-consistent content and stay (vacating them paled the
  rear-quarter temple ribbons over the crown-flake gate, az-135
  0.0006 -> 0.0027). Where we cannot commit, baseline claims stay —
  surrender-without-commitment leaves the membrane anchored by whatever
  survives nearby (measured: lash-dark anchors bled through a vacated
  eyelid rim as a floating dash).
- film retone (`retone_film_band`, after `texel_surface_smooth`, before
  detail synthesis): committed fill takes its tone from dark-material
  OBSERVED anchors only (octant-binned voxel-ball means at growing
  scales), scaled by photo wispiness and the same dark-dominance factor;
  mirror destinations inside the commit are removed; zero-weight rim
  coverage inside the film zone is demoted to fill
  (`demote_unwitnessed_rim`).

Face proof A/B (same tip, verdict1 harness): @1024 failed checks 4 -> 1
(the pre-existing front-identity SSIM; az-135 crown flakes x2 and az-45
dark debris all cleared), @2048 2 -> 2 (both pre-existing front-identity;
mean|RGB| 22.3 -> 22.2), az+22.5 el0 eye_count 1 -> 2 (correct), no
3-blob eye failures at any azimuth; hairline crops at az 0/±22.5 show
hair-toned fill where the beige sheet/pale curtain was (temple beige
remnants that survive are kept mixture claims under the witness veto —
committing them was measured strictly worse). `scripts/texture_qa.py`
PASS at both resolutions. Single-view bakes (starship/owl) are
bit-identical with the mechanism present vs disabled (md5-verified);
`commit_film_band` no-ops below two views by construction.

### Added (mirror twin rescue — general weak-twin feature transplant in the bake)

Mirror completion only writes UNOBSERVED texels, but on near-symmetric
subjects a feature region can be observed yet badly witnessed: every
covering view sees it at grazing incidence or through a misregistered
duplicate reference, so the texels carry a smear that no per-texel gate
downstream can repair (all covering witnesses agree on the wrong
content). Measured on the face proof at az -90: eye-disc ball witness
weight 0.16 vs the healthy twin's 0.55, harness `eye_count` 0, and the
resulting broken eye dragged the side_right identity registration 1.3%
off — the -0.132 worst-window "ghost" at the ear was a registration
artifact of the broken eye, not ear-texel damage (forcing the corrected
registration onto the unfixed render scores that window at +0.47).

`detect_mirror_rescue_discs` (new) finds such regions generally, with no
feature-class knowledge: strong-side discs that are confidently
witnessed, locally contrastful, and carry a coherent dark core, whose
mirror twin is observed but >= 2x weaker witnessed AND feature-empty
(pointwise blob response <= 0.5x the core's). Detected discs drive the
existing `mirror_rescue_disc` transplant (tone-matched, feathered) inside
`bake_projection_texture`, after mirror completion, under the same
geometry-symmetry gate (score >= 0.55); transplanted texels count as
completion, not photo truth. Gates that keep legitimately asymmetric
content untouched, each with a regression test: content well-witnessed
on both sides never triggers (twin-weight ratio); unobserved twins belong
to mirror completion (twin-coverage); a twin with its own comparable
structure is left alone (feature-emptiness, sampled pointwise because
ball averages dilute edge responses); discs straddling the symmetry
plane are refused entirely (a half-transplant guarantees a mid-feature
seam — measured painting a black dash on the face lane's front lips).
Two placement/tone refinements, both measured load-bearing: the
transplant is anchored along the mirror axis on the twin's own
evidence-weighted feature-dark centroid (capped at 0.4x the feature
radius — the pure geometric mirror position pulled the source-pose
identity registration 1.3% and its SSIM 0.632 -> 0.601; in-plane anchor
components are noise and re-rolled a bistable registration, so only the
axis component is used), and the tone-matching ring averages only
source-mask texels (in-bake the annulus contains not-yet-filled texels
whose zeros biased the offset ~0.02 dark, pushing transplanted skin
flecks across the dark-debris gate). Face proof A/B at 2048 (same tip,
verdict1 harness): failed checks 8 -> 2 (both remaining are the
pre-existing front-identity SSIM/MAE, 0.632/22.1 -> 0.629/22.3); az -90
eye_count 0/0 -> 1/1 at both elevations; identity[side_right] worst
window -0.132 -> +0.219 (gate 0.05) with SSIM 0.657 -> 0.682;
dark_debris at az -22.5/-35 all under the 0.003 gate (was
0.0030-0.0035); scripts/texture_qa.py stays PASS 13/13. Single-photo
bakes (starship/owl) fire zero discs (geometry scores 0.98 but the twin
side is unobserved) and their textures are bit-identical with the
detector disabled (md5-verified on the starship).

### Documented (identity-gate shading floor — measurement bias, not albedo signal)

Photo-vs-render identity metrics carry a perfect-texture penalty from the
preview renderer's own shading (`shade = 0.88 + 0.12*diffuse`): measured
SSIM 0.977 / mean|RGB| 11.45 for a PERFECT texture at the face lane's
declared pose. The term is texture-independent, so the correction belongs
in the measurement (photo multiplied by the white-texture shade field,
MAE budget re-tightened by the removed floor), never in the texture.
Full calibration data and the proposed harness patch:
`/tmp/c2d/REPORT.md` + `docs/KnowledgeBase.md` ("Identity gates that
compare shaded renders to photos carry a perfect-texture floor").
No pipeline code changed by this analysis lane.

### Fixed (fill-character restoration — closed-loop energy calibration in `synthesize_fill_detail`)

The fill-detail synthesis transferred observed log-residual SIGMA to the
fill, but the quality bar (`texture_qa` `texel.fill_gradient_energy_ratio`,
gate >= 0.5) judges LINEAR-luminance gradient energy — an open-loop proxy
that systematically undershoots. Measured decomposition on the starship
proof at 1024 (fill/observed energy 0.43 at gain 0.7, gate FAIL): donor
amplitude transfer 0.84x (color-similarity weighting favors donors
darker/quieter than the observed median), carrier frequency 0.69x (the
3-texel finest noise octave carries less per-sigma gradient than photo
micro-texture at 1-3 texels), base luminance 0.79x (multiplicative
log-detail on a darker fill base yields proportionally less linear
gradient). Two changes, both resolution-invariant:

- Finest carrier octave moved to ~2 texels (`wavelength_texels` 3 -> 2,
  `octaves` 2 -> 3, band now 2..8 texels): restores per-sigma spectral
  energy at every resolution.
- CLOSED-LOOP CALIBRATION: the pass provisionally applies the detail
  (clip + seam ramp included), measures the realized fill gradient energy
  with the same Scharr operator the QA uses, and solves (secant, 2-3
  evaluations) one global scale that lands the fill at `gain` x the
  observed energy. Bounds: never below 1 (already-rich fills — face hair
  streaks — are never dampened), never above `energy_calibration_max`
  (3.0), and never past a sigma guard that caps the fill's log-sigma at
  the observed population's band-matched residual sigma — gradient parity
  may not be bought with granite on edge-dominated subjects; any shortfall
  is reported in the bake stats (`fill_detail.energy_calibration`), not
  hidden.

Measured (fresh single-view bakes, current tree, `texture_qa`):
starship fill energy 0.39 -> 0.58 (1024) / 0.50 -> 0.63 (2048), owl
0.43 -> 0.58 (1024) / 0.60 -> 0.69 (2048), dark smears 0 and facet fields
0 at 4x throughout, seams within allowance; face (multi-view) stays PASS
with fill energy 1.06 -> 1.16 (its calibration correctly resolves to
scale 1.0 — the sigma guard binds). Tests:
`test_synthesize_fill_detail_energy_calibration_reaches_gate`,
`test_synthesize_fill_detail_calibration_never_injects_granite`.

### Fixed (texture QA photo reference — matte the photo like the bake does)

`scripts/texture_qa.py` derived every photo-side reference (viewer-truth
brightness, seam allowance, photo calibration, and the front view's
visibility alpha) from a "non-white" heuristic on RGB inputs. On unmatted
photos with non-white backdrops the heuristic measures the BACKGROUND:
on the owl proof photo (light-gray studio backdrop, ~205 median
luminance) it classified 100% of the frame as foreground, inflating the
brightness reference to 203 vs the subject's true 129 and failing
`viewer.brightness_ratio` at 0.567 on bakes whose subject tone was in
range — the gate measured backdrop bias, not albedo fidelity. The harness
now mattes RGB photos with the same `remove_background_robust` the bake
pipeline itself applies before projecting (RGBA photos keep their alpha;
if the matte model is unavailable or degenerate the old heuristic remains
as an explicit `heuristic_nonwhite` fallback recorded in results.json).
Same-bundle deltas (current-tip bakes): owl brightness 0.567 -> 0.891
(2048) / 0.562 -> 0.884 (1024), ship 0.752 -> 0.845, face 0.960
(unchanged; its photo is near-fully non-white so the heuristic was
accidentally right). The front-view visibility reconstruction also stops
counting background-ray texels as observed (ship coverage reconciliation
qa 0.261 vs bake 0.177 -> qa 0.211).

### Added (dense residual reference registration — strictly-local validated lattice flow)

Global similarity registration (width-profile matching + overlap similarity
search) cannot satisfy per-feature displacements on generated geometry: the
nose, mouth and eyes each want a DIFFERENT small 2D correction (measured on
the face lane: nose −10 px, mouth (−4,+4), eyes (+4,0) at 512), so
reference photos paint ghost lip/lash fragments next to the source's
features. New module `abstract3d/reference_flow.py`, wired into
`bake_projection_texture` directly after `register_reference_by_source_overlap`
(orthographic multi-view references only; single-view bakes verified
bit-identical):

- Energy: Charbonnier photometric residual of the gain-corrected reference
  against the SOURCE'S PAINTED TRUTH splatted into the reference's image
  plane through the shared surface (first-surface visibility, source
  confidence x reference-facing evidence weighting), regularized by a
  bending (thin-plate) energy on a coarse-to-fine control lattice
  (64/32/16 px), Gauss-Newton with a 2%-of-frame displacement cap. The
  photo is warped exactly once (flow upsampled to the native canvas).
- Safety architecture, each clause anchored to a measured failure: per-cell
  validation (>= 20% weighted-L1 improvement AND absolute post-warp error
  within 1.25x the median of improving cells), a one-ring evidence leash
  (adjacent cells keep flow only with substantive non-worsening own
  evidence), reference-facing evidence gating, and strictly-zero
  displacement everywhere else. Global extension of band-fit corrections
  was measured harmful twice (a residual affine collapsed side identity
  0.706 -> 0.587; even a pure translation moved the hair mass and tripped
  skin_in_hair at az +-135) — hence STRICTLY LOCAL.
- Validation: injected known warps (shift / rotation / barrel / local bump
  / combined) recovered to <= 0.7 px median inside the evidence band at the
  512 solve scale; acceptance additionally gated on a >= 2% overlap-error
  improvement with >= 3 validated cells, else the input photo is returned
  untouched.

Measured (face 3-view lane, same-tip A/B off -> on): 1024 harness failures
12 -> 10 (front identity SSIM/MAE/worst-window and side_left worst-window
failures cleared); 2048 failures 8 -> 8 with front SSIM +0.006 / MAE -0.8
and two dark_debris lines swapped at the 0.003 gate; ghost-lip fragments at
the mouth visibly reduced at both resolutions (crops in the cycle report).
Starship single-view lane: bit-identical texture with the stage on/off.
Tests: `tests/test_reference_flow.py` (injected-warp recovery, strict
locality, unreachable-content rejection, no-overlap identity).

### Added (photometric delighting of reference views — SH-in-normal-space shading removal)

Photos carry their own lighting, and two registered photos of the same
subject disagree on every shared surface point as a smooth function of the
surface NORMAL (each light shades each orientation differently). That
disagreement survived exposure harmonization (a scalar gain cannot express
a normal-dependent field), leaked into view-handoff tone steps, and gets
doubled by any viewer relight. New `texturing.delight_projections`, run on
the atlas projections before harmonization/gating:

- Model: Lambertian formation I_v = A * S_v(n); on OVERLAP texels the
  log-luminance ratio log Y_u - log Y_v = B(n) . (c_u - c_v) cancels the
  albedo EXACTLY. B is the order-2 real SH basis in the normal
  (Ramamoorthi & Hanrahan: >99% of distant-light irradiance energy), so
  genuine albedo detail — high-frequency in normal space — is outside the
  model span by construction.
- Estimation: joint weighted ridge LS over all overlapping view pairs
  with gauge c_source = 0 (references are relit to the SOURCE's light; the
  common lighting component is unobservable from ratios, and the source
  photo is the identity anchor everywhere else in the pipeline). Huber
  IRLS with MAD-adaptive threshold rejects content outliers
  (misregistration hair-over-skin) without rejecting legitimately strong
  shading ratios; the fitted field is clipped to the overlap's own
  [p1, p99] +- 0.1 (exclusive-region normals the fit never saw cannot
  receive extrapolated inventions) and capped at |log| <= 1.
- Application: luminance-only (chroma untouched), multiplied into the
  reference's covered texels; the existing per-channel exposure gain then
  handles only residual white balance (measured on the face lane: gains
  drop to ~1.02 after delighting).
- Overlap-proximity fade: the correction applies fully near the overlap
  surface (where seams form) and fades to zero deep inside the
  reference's EXCLUSIVE territory, where that photo is the only witness
  and per-view identity outranks consistency with a light no camera sees
  from there. An adversarial bisect measured the unfaded version
  relighting a profile's whole exclusive side (identity MAE vs its own
  photo 26.4 -> 39.5) and disabled the stage; the fade keeps the handoff
  fix with exclusive-side drift measured at 0.002 mean|RGB| (stats row
  `exclusive_mean_abs_delta`), and the stage is re-enabled.
- Revert-on-confound: kept per reference only when that reference's
  overlap mean|RGB| disagreement against the source DROPS by > 0.002 —
  the same statistic family as the exposure gate it generalizes (the DC
  gain is this model's order-0 term).

Measured (face 3-view lane, 1024): side_right overlap disagreement
0.085 -> 0.063 (-26%) with the correction kept; side_left reverts (its
overlap disagreement is content mismatch, exactly the confound the gate
exists for); synthetic two-light sphere proof x3.2 disagreement drop
capped (x70 uncapped), recovered albedos agree on overlap. Tests:
`test_delight_projections_recovers_agreeing_albedo_on_two_light_sphere`,
`test_delight_projections_fade_protects_exclusive_territory`,
`test_delight_projections_keeps_chroma_and_reverts_on_confound`.

### Added (geometric witness confidence — sampling-stretch and concavity terms in the projector)

Projection weight was `alpha * facing^2 * witness_factor`; facing measures
LOCAL TILT only, and the eye-socket class of defect rides through it: a
socket wall can face the camera acceptably while the composed texel->photo
mapping collapses, so one photo pixel smears down the whole wall.
`_tripo_projection_geometry_confidence` adds two exact terms:

- STRETCH: the texel->photo Jacobian J = [ds/dcol, ds/drow] by finite
  differences of the projector's own sample maps (exact for both camera
  models; chart-boundary pairs masked). sigma_min = smallest singular
  value = worst-direction sampling pitch; stretch = nominal / sigma_min
  with nominal = median sigma_min over well-facing texels (facing > 0.7).
  The nominal makes the statistic invariant to photo/atlas resolution AND
  to legitimate chart anisotropy (normalizing by sigma_max instead was
  built first and measurably mis-scored healthy texels on anisotropic
  charts — cylinder test). Weight *= 1/(1 + max(stretch-1, 0))^p with
  p = 2, measured by sweep on the face proof (adversarial harness, all
  else fixed): p=0 13 failures, p=1 13 (sub-threshold dark-debris
  improvements only), p=2 8 — three dark-debris views cleared, the az -70
  eye recovered, front identity MAE fail cleared — with texture_qa fully
  green and single-view assets within noise of p=1.
- CONCAVITY: mean curvature from the normal-field divergence over the
  surface (div n ~ (dn . dp)/|dp|^2 along both atlas axes), normalized to
  concavity = -0.5 div n * (0.02 * diagonal). Texels BOTH concave
  (> 0.35) and grazing (facing < 0.5) multiply by 0.25: concave interiors
  catch stretched/misplaced content exactly where the witness is weakest,
  while a well-facing concave eye keeps its claim (legitimate socket
  content and shading survive).

Per-projection stats key `geometry_confidence`. Synthetic proofs: rim
collapse demoted on an anisotropic-chart cylinder (front factor 1.0, rim
< 0.35); sharp-trench demotion strictly inside the concave interior with
a convex-ridge control undemoted. Tests:
`test_projection_geometry_confidence_stretch_demotes_collapsed_mapping`,
`test_projection_geometry_confidence_demotes_concave_grazing_only`.

### Added (synthesized-texel luminance floor — zero dark fill fragments at close zoom)

Provenance audit of the close-zoom "spurious dark fragment" failures on
the single-view proofs (starship 4, owl 6 at 4x): dark observed anchors
(intake interiors, panel shadows, occasional background-adjacent grazing
samples) seed the harmonic fill, whose maximum-principle solve freely
TRANSPORTS that darkness across hidden surface (measured: fill blobs at
luminance 14-26 whose nearest observed anchors sit at 61-114). Observed
texels carry photo evidence; fill texels carry none, so a context floor
is a legitimate prior. New `texturing.enforce_fill_luminance_floor`, run
as the bake's last color pass over FILL texels only (observed AND
mirror-completed texels bit-identical by construction — mirror copies
carry their twin's evidence, and an adversarial pupil-analog test showed
a local floor cannot tell a mirrored pupil from a defect):

- context floor: plain ball mean m1 at two world scales (R, 2R;
  R = 0.035 x diagonal) with a dark-minority gate per scale (smoothstep
  to zero as the ball's dark fraction passes 0.30 -> 0.45): a defect
  pocket is by definition a local anomaly, so regional darkness (hair
  mass, shaded hull side, hairline shadow bands) stands the floor down —
  an ungated floor measurably turned the face's hairline band into a
  pale film (verdict-harness pale_film 0.0055-0.0062 > 0.005 gate);
- donor-consensus floor ("donor validation"): the same ball statistics
  over direct-observed donors only, catching fill far darker than every
  donor around it;
- sheet-awareness: every ball statistic bins by dominant normal-axis
  direction (6 bins) and each texel reads every bin EXCEPT the opposite
  one — a Euclidean ball on thin-crust meshes otherwise judges a shaded
  underside against the sunlit topside millimeters away through the
  shell (measured: sheet-blind floor dropped the starship
  fill-gradient-energy gate 0.57 -> 0.48; a Hamming<=1 octant pooling
  variant let the face's rear hair read forward skin as context — the
  critic-measured "skin patches in rear hair" failure);
- dark-evidence exemption: a connected dark component (3D voxel
  connectivity at max(2 texel pitches, 0.003 x diagonal) — the WORLD
  floor prevents resolution-dependent fragmentation measured as an
  owl-only-at-2048 seam spike) containing >= 8 observed texels whose
  tone its fill TRACKS (fill mean <= 1.35 x evidence mean) is a
  witnessed feature continued into hidden surface (the owl's wing
  markings) and keeps its own tone; components failing the tracking
  test (starship engine-halo smears at 2-3.7x their cavity tone) get the
  full floor. Without this, lifting a legitimate marking manufactured a
  fresh observed|fill tone seam (owl p95 29 -> 52..60, gate 52.2);
- application: saturating per-pixel depth compression in log-luminance
  (remaining depth = residual_depth * (1 - exp(-d/residual_depth))):
  monotone (no posterization at the floor line), slope 1 at zero depth
  (no visible boundary), bounded remaining depth 0.10. A base/residual
  split was built first and measurably leaked dark bands wider than the
  base radius but narrower than the context ball (owl crease bands);
  compactness restrictions and boundary feathers were tried and
  reverted — they left pocket edges under the floor, re-detected as
  fresh smaller fragments. Pixels deeper than 1.2 below the floor blend
  toward the context consensus color (bright-half mean scaled to
  target: near-black 8-bit pixels carry no usable chroma to multiply);
- floor_ratio 0.65 vs the detector's 0.45: deliberate headroom because
  the render-window reference mixes brighter cross-sheet content than
  the sheet-aware ball (measured up to ~1.2x on the owl's wing creases).

Applied to the SHIPPED ledger bundles as a texture post-process (masks
reconstructed the same way the QA harness does): starship 4 -> 0 and owl
6 -> 0 dark fragments at 4x, with seam, facet-cellular, and
fill-gradient-energy gates all remaining green (ship fill energy 0.57 ->
0.55, fill Scharr edge energy +0.4% — hull panel lines survive). Fresh
bakes at 1024 and 2048 on all three proof assets: zero dark fragments
across every bundle. Stats key `fill_floor`. Tests:
`test_enforce_fill_luminance_floor_lifts_pockets_keeps_lines_and_dark_regions`,
`test_enforce_fill_luminance_floor_spares_mirror_features_and_opposite_sheets`,
`test_enforce_fill_luminance_floor_donor_anchor_catches_transported_darkness`.

### Added (mirror-consistency disc rescue for weakly-witnessed features)

`texturing.mirror_rescue_disc(colors_rgb, positions_texture=..., center=...,
radius=...)`: replaces a world-space feature disc's texels with their mirror
twins' content, tone-matched to the destination's surrounding annulus and
feathered at the edge. Complements `mirror_fill_from_observed`, which only
writes UNOBSERVED texels: a feature region can be observed yet badly
witnessed (all views at grazing incidence, or a mirrored duplicate reference
landing misregistered), leaving displaced content that no downstream gate
can repair because every covering view agrees on the wrong pixels. Measured
root cause on the face lane's right eye at az -90 (the "eye_count 0 at -90"
QA failure): the eye disc's best witnesses average blend weight 0.14 vs the
mirror twin's 0.50, and the painted iris band sits ~0.04 mesh units below
its mirror-correct position with a second stray band at the brow — the
detector sees two thin high-aspect fragments instead of one eye. Applying
the rescue to the frozen v14 ship candidate (disc centered on the twin of
the PASSING left eye, found by the QA detector itself) cleared all four
profile eye_count failures (az +/-90 x el 0/10) and improved the right
profile's worst 49-px identity window from 0.118 to 0.194; harness failures
6 -> 5 (the remaining trade: the az -45 el 10 eye blob fragments under the
transplanted specular highlights and undercounts, while visually reading as
a more structured eye). Geometry ceiling DISPROVEN for this defect: the
mesh's squinted lid aperture (0.20-0.29 vs the photo's 0.43) still renders
a machine-detectable and human-readable eye when correct content lands on
it — the defect was texture placement, not geometry. The function performs
only the geometry-driven transplant; callers decide WHERE (QA localization,
witness-quality maps). Test:
`test_mirror_rescue_disc_transplants_twin_feature_tone_matched` (synthetic
folded plane: twin feature transplants, tone offset cancelled, locality and
untouched-twin guarantees).

### Added (gradient-domain view compositing — one screened-Poisson composite replaces the seam-patch stack)

Multi-view composition previously fought tone seams with a stack of local
patches (softmax color blend, per-region seam-leveling offsets), and the
residuals were still visible: a mouth-crossing chroma seam at az 0, cheek
tone patchwork at close zoom, and dark-debris marginals at the
22.5–45-degree views. Root cause: every patch operated in the COLOR domain,
where exposure disagreement between witnesses is indistinguishable from
content. New module `gradient_compositing` composites the views in the
GRADIENT domain instead and solves one screened Poisson system over the
observed texel surface graph, running between outlier filtering and
completion so mirror/fill propagate equalized colors (the same relative
order the legacy path gives its leveling offsets). Energy: per-edge match
to a composited target gradient plus confidence-weighted soft anchors to
the blended colors; SPD normal equations; multigrid-preconditioned CG,
float32, deterministic.

- Graph: UV-grid 4-neighborhood within charts (3D jump guard) plus KD
  chart-stitch edges, with a normal-agreement gate so thin-shell sheets
  never stitch; the atlas solves as ONE closed surface (chart cuts proven
  invisible on a two-chart synthetic sphere, and broken when stitching is
  disabled).
- Target gradients: most confident common witness per edge (photo edges
  survive verbatim — no cross-view averaging); one-sided witnesses at
  winner-take-all handoffs; zero-gradient membrane only where no view
  sampled both endpoints.
- Witness-less (line) edges carry two measured safeguards: their weight
  scales with reference/resolution (boundary edges appear once per
  crossing row, so their energy per world length otherwise doubles per
  resolution octave — the mid-face chroma seam eliminated at 1024
  reappeared at 2048 through exactly this), and a material gate (the
  screened-Poisson analog of seam leveling's `boundary_cap`) releases
  edges whose color step exceeds ~0.18 so hair|skin and ear-fold borders
  are never tinted toward each other.
- Screening: lambda proportional to blend confidence with a 0.1 floor
  (rim texels otherwise drift freely and wash fragmented coverage),
  source-view claims boosted 4x above weight 0.4 (the photo's identity
  contract), completion left to inherit via fill, all rescaled by
  resolution^-2 so the equalization decay length is fixed in world units.
- Solver: geometric-aggregation multigrid V(1,1)-cycle as CG
  preconditioner (voxel-clustered levels, Galerkin operators, damped
  Jacobi, coarsest-level splu); float32 iteration with float64 scalar
  accumulation; converges in 21-141 iterations at ~1e-5 relative residual
  where plain Jacobi-CG needed ~1000; full compositor 1.8 s (face 1024) to
  9.2 s (face 2048) on this host.

Synthetic ground truth (two-chart sphere, two views, one corrupted):
additive exposure offset recovered exactly up to one global constant
(RMS < 2e-3), handoff discontinuity killed >20x, checkerboard edges keep
>= 95% contrast across the handoff, gain+vignette handled with a smooth
sub-visibility error field. `bake_projection_texture` gains
`compositing="auto"|"legacy"|"gradient_domain"` (auto = gradient_domain
for multi-view bakes, legacy for single-view where the solve measurably
only jitters threshold-marginal detectors; legacy remains selectable).
A/B on the proof assets (both QA harnesses, both resolutions, single
frozen tree, alignment-controlled identity): face adversarial failures
11 -> 10 (1024) and 8 -> 8 with strictly better defect magnitudes (2048:
top dark-debris 0.0045 -> 0.0033), dark-debris failing views 5 -> 1 at
1024; identity SSIM at controlled alignment equal or better on front and
side_left under either warp (front 0.619 -> 0.637, side_left
0.704 -> 0.707 under the baseline warp; the raw harness numbers are
warp-landing-confounded, verified at pixel level); texture_qa face and
ship PASS at parity or better medians.
Tests: `tests/test_gradient_compositing.py` (graph stitching/guards,
gradient selection rules incl. line classification, screened-Poisson
recovery/anchoring, end-to-end bake, determinism).

### Fixed (texture-cycle integration — pose stability, single-view outliers, acceptance harness)

- Hardened `estimate_pose_photometric` against two measured failure modes. First, the
  scorer compared photo and render gradients in mismatched frames: compact heads
  tolerated it, but an elongated subject's projected aspect swings with elevation and the
  correlation degraded into noise — the starship's true pose (az +30, el +15) was
  unrecoverable, and el +15 was not even in the elevation grid. Renders are now aligned
  into the photo's frame with crop-immune anchors (subject top row, silhouette centroid,
  mean width over common rows — full-bbox recentering was tested and rejected because it
  breaks on cropped photos), and the elevation grid spans +/-15 with local refinement on
  both axes. Ship observed coverage: 0.062 -> 0.20. Second, on bilaterally symmetric
  meshes the mirror pose (+az vs -az) is a structural near-tie for gradient-magnitude
  content, and 0.1% vertex jitter could flip the azimuth SIGN (an adversarial QA rebake
  drew a sign-flipped pose and produced a 65-failure bake). A chirality tie-break now
  correlates the horizontal ANTI-SYMMETRIC luminance components of photo and render —
  sign-opposite between mirror poses — and decides the sign with a margin symmetric
  content cannot dilute. Face pose is stable at +12.5..+20 across jitter trials.
- The surface outlier filter now also runs for single-view bakes: the foreign-view
  condition is vacuous with one witness, but the same-view color-extreme condition
  catches dark background-adjacent rim misprojections (measured on the starship: the
  surviving dark fragments at 4x zoom are exactly this class; 854 texels dropped).
- Regenerated the face-2mv, hunyuan-starship, and hunyuan-owl proof bundles end-to-end on
  the integrated pipeline. The face bundle passes all `scripts/texture_qa.py` gates
  (materials, viewer-truth brightness, fill character, facets, seams, close-zoom probes);
  the starship passes 12/13 (residual: 4 small dark fill fragments at 4x zoom across all
  probe crops, documented in ADR 0009); the owl's remaining brightness-gate failure is a
  harness calibration artifact (its input photo is unmatted white background, inflating
  the photo-side reference).
- Fixed `python -m abstract3d.cli` silently exiting 0 without running anything (missing
  `__main__` guard); the `abstract3d` console script was unaffected.

### Fixed (fifth adversarial cycle — observed-region close-range defects)

Owner-visible complaints at close zoom on the multi-view face: a vertical
tone seam down the nose/philtrum where front-photo content meets profile
content, and dark copy fragments on cheeks near the hairline. Provenance was
instrumented per texel (per-view weight/winner maps rendered from the bake's
own captures) before fixing; both fixes are general-purpose (no face logic):

- Added `level_composed_seams` (wired into multi-view bakes between mirror
  completion and fill): Ivanov/Lempitsky-style seam leveling on the mesh
  graph. Per-texel region = winning view or mirror fill; one additive
  low-frequency offset field per region cancels tone steps at region
  boundaries while high-frequency content is preserved. Two safeguards are
  load-bearing and covered by tests: boundary edges whose color step exceeds
  `boundary_cap` are genuine material borders (hair|skin) and are excluded
  (an uncapped solve tinted the ear region and dropped the right-profile
  identity's worst face window from +0.10 to -0.13 SSIM), and vertices whose
  winning witness is confident are pinned toward zero correction, so
  leveling only recolors the weak/contested bands between confident zones
  (each photo stays ground truth on surface it saw well). Face-2mv at 1024:
  adversarial harness failures 16 -> 14 (mid-face chroma-seam failures
  3 -> 1), identity SSIM front 0.614 -> 0.619, side_left 0.612 -> 0.703,
  side_right 0.688 -> 0.690; single-view bakes are structurally unaffected
  (verified bit-identical on the starship lane).
- Added a consensus guard to `mirror_fill_from_observed`: geometry is never
  perfectly symmetric (0.966 on the face), so twin lookups near material
  boundaries could copy hairline hair onto cheek skin (measured: half the
  dark defect pixels on the left cheek at close zoom were such copies). A
  copy is rejected only when the destination's observed 3D neighborhood is
  color-consistent AND the copy contradicts it; feature-rich destinations
  accept copies unchanged, rejected texels fall to the harmonic fill.
  Verified inert where twins are legitimate (starship mirror completion:
  guard on/off textures identical).

### Fixed (fourth adversarial cycle — hidden-surface fill quality)

Owner-visible complaint: texture on surface NOT visible in the input photos
read as flat "painted" mush (rear head, ship hull), with faceted flat-color
polygon blocks at close zoom and patchwork mottle on large single-view fills.
Root causes measured and fixed (metrics on the face-2mv and hunyuan-starship
proof assets at 1024, defect views + 4x zoom crops):

- Fixed the facet blocks: `mesh_graph_harmonic_fill` assigned every unseen
  texel its nearest VERTEX's solved color, so each vertex's Voronoi cell
  rendered as one flat polygon (~59k vertices serving ~4.2M texels at 2048).
  Texels now blend the 3 nearest vertices with inverse-distance weights
  (fill-region flat-plateau fraction 0.45 -> 0.21 on the face).
- Added `texel_surface_smooth`: a Jacobi relaxation of fill texels over the
  k-nearest-neighbor graph of texel 3D positions (observed texels fixed as
  Dirichlet anchors, normal-agreement-weighted edges). Runs after BOTH fill
  paths; it removes the residual vertex-cell seams of the harmonic fill and
  the donor-set patchwork of the KD fallback (fill Laplacian energy down
  30x / 8x respectively). Zoom crops show smooth material instead of blocks.
- Added `synthesize_fill_detail`: propagated fill is the correct average
  color but has ZERO micro-texture, which is exactly the "painted wash". The
  pass transfers observed texture STATISTICS — robust (L1) local residual
  amplitude and structure-tensor streak orientation, per material via
  normal + base-color matched surface-nearest donors — carried by
  deterministic multi-octave 3D value noise (seamless across UV charts),
  smeared along the transferred orientation (LIC) in proportion to donor
  anisotropy, applied multiplicatively in log domain, amplitude-capped at
  the observed p90 and feathered at the observed seam. Fill/observed
  local-variance ratio: 0.53 -> 0.83 (face rear hair, reads as combed
  streaks), 0.15 -> 0.59 (ship hull grain). Copying observed residual
  STRUCTURE (shift-map quilting) was prototyped and measured worse (chaotic
  panel fragments); statistics transfer makes hidden surface read as the
  same material, not the same content — a documented, honest limit.
  `bake_projection_texture(fill_detail_gain=...)` scales it; 0 disables.
- Added `texture_completion="auto"` (bake + both backends + CLI): apply
  mirror completion iff the mesh's own left-right symmetry score passes the
  existing >= 0.55 gate. The Hunyuan backend now defaults to "auto" — a
  single photo of the starship observes 6-9% of its texels while the
  geometry is 0.98 symmetric; the mirrored twin of the observed sliver is
  real panel content where any propagated fill is wash. Explicit "none" /
  "mirror_symmetry" behave exactly as before; `stats["texture_completion"]`
  reports the resolved mode and `texture_completion_requested` the request.
- Adversarial face-gate harness (28 views x 8 detectors + identity):
  failures did not increase (7 -> 6 at 1024, 6 -> 5 at 2048 on identical
  captures; the remaining failures are projection/pose classes untouched
  by the fill stage). All unit tests pass; fill passes are deterministic
  (fixed seed, hash-based noise).

### Fixed (third adversarial cycle — textured-export material factors)

- Fixed textured GLB/OBJ exports darkening ~60% and rendering as metal in
  spec-compliant viewers: the bake assembler built its `TextureVisuals` with a
  default trimesh `SimpleMaterial`, whose 0.4 gray diffuse became
  `baseColorFactor [0.4, 0.4, 0.4, 1.0]` on GLB export while `metallicFactor`
  was omitted entirely (the glTF default is 1.0, fully metallic). Baked-texture
  meshes now carry an explicit `PBRMaterial` (white base color, metallic 0.0,
  roughness 1.0) so the baked albedo is the authored surface color in any
  viewer. Affects TripoSR and Hunyuan3D textured exports; geometry-only
  exports are unchanged.
- Fixed the OBJ sidecar MTL carrying `Ka/Kd/Ks 0.4`: the OBJ exporter now maps
  the mesh's PBR factors onto explicit Phong constants (`Ka/Kd 1.0`, `Ks 0.0`
  for non-metallic baked albedo — a non-zero Ks would add a synthetic sheen on
  top of photo-derived colors in Phong viewers).
- Preview renders now multiply the sampled texture by the material's base
  color factor (both the ModernGL and matplotlib paths), matching what
  spec-compliant viewers show; previously previews sampled the raw texture and
  masked the darkening defect entirely.
- Added `scripts/check_export_materials.py`, a reusable GLB/MTL material
  factor auditor (`--strict` for CI-style gating), and
  `tests/test_export_materials.py` regression tests that export through the
  real helpers and assert the GLB JSON factors and MTL lines.
- Repaired the shipped `final-proof/hunyuan-starship` and
  `iter3-multiview-fixed/face-2mv` bundles in place (scene.glb materials JSON
  and scene.mtl only; geometry and texture bytes verified bit-identical).

### Added (second adversarial cycle — six-agent audit of the multi-view face bake)

- Added photometric source-pose estimation for canonical-frame bakes
  (`estimate_pose_photometric`): the ortho path no longer assumes the conditioning photo
  was taken from the canonical front. Multi-view reconstruction canonicalizes the OBJECT
  (symmetry plane onto the world axes), not the camera, so a photo of a subject whose head
  is turned sits 15-25 degrees away from the canonical front; five independent
  measurements put the checked face photo at azimuth +15..20 / elevation +8, and
  projecting it at 0 was the dominant cause of the doubled-face artifact. The estimator
  correlates signed gradient VECTOR fields (magnitude is bilaterally symmetric on faces
  and cannot tell a pose from its mirror) between the photo and untextured renders, with
  interior-distance weighting so pose-insensitive silhouette edges do not swamp the
  signal; the declared pose wins unless beaten by a real margin, and a genuinely frontal
  input returns "not estimated". See ADR 0008.
- Added overlap-photometric reference registration
  (`register_reference_by_source_overlap`): after the source view projects, each reference
  photo is aligned by minimizing source-weighted RGB disagreement over mutually observed
  texels. Silhouette registration aligns outlines — on heads, the hair contour — and left
  interior features displaced by ~6% of the frame (the profile's eye painted on the
  temple); registering interior content to the source's painted truth removed the doubled
  eye and cut the adversarial QA harness failures from 62 to 14 in one change.
- Added crop-immune width-profile registration (`register_view_by_width_profile`) as the
  coarse aligner for reference photos: row-wise silhouette widths below the subject's top
  form a scale-sensitive signature that survives cropping differences (area-IoU rewards
  degenerate blow-ups and edge-chamfer locks onto long edges on cropped photos).
- Added a layered-density witness gate to the projector (`layered_zone_gate`): photo
  regions where more than 10% of projected samples land a thin gap (3 epsilon to
  0.03 x diagonal) behind the first surface are imaging stacked film shells (hair wisps
  over a scalp); sub-pixel aim — not content — decides which sheet each pixel stamps, and
  the pixels are themselves material mixtures, so the view surrenders the whole region to
  better views or fill. Pixel-level gating was measured and rejected (survivors between
  layered pixels still anchor flakes). With mirror-source gating this cut hairline
  flake/debris failures by 77-80% (dark-debris view failures 22 -> 0).
- Mirror completion sources are now excluded from any view's contested layered band, and
  the confidence floor returns to 0.35 in the bake call: provenance tracing showed >90% of
  hairline flake islands were mirror/harmonic COPIES of a few low-confidence mixture
  anchors, not direct projections. Disabling mirror completion entirely measured worse
  (the far cheek degrades to harmonic mush) — gating, not removal.
- The source view stops painting beyond ~66 degrees off-axis in ortho multi-view bakes
  (per-role facing threshold 0.4): beyond that the source's samples are stretched rim
  content, and reference photos are the better witness. Single-view and perspective bakes
  keep the wide threshold (stretched content beats no content when nothing else covers).

### Fixed (second adversarial cycle)

- Fixed four numerically verified math defects found by a line-by-line audit (81 checks
  against analytic ground truth): the outlier filter's 2-hop consensus let every texel
  vote for itself (the diagonal of A@A equals vertex degree), so foreign-view misprojection
  islands self-certified and were never dropped — the consensus now excludes self-votes and
  binarizes path counts; the splat silhouette's dilation biased every edge-based
  registration toward +4% scale (a pixel-perfect canonical photo registered at 1.04) — an
  erosion pass restores the true rim; the strict z-buffer's scalar epsilon made smooth
  tilted surfaces occlude THEMSELVES (up to 40% of visible texels at 55-75 degree tilt
  demoted to milky fill) — the epsilon is now slope-aware (standard shadow-mapping
  practice), keeping the base tolerance for front-on sheets; horizontal registration
  shifts were converted back through photo WIDTH while fitted in a height-normalized
  frame, corrupting non-square photos on the perspective path.
- Removed the source view's residual silhouette registration in ortho mode: the canonical
  recenter IS the registration, and the residual scale/shift search chased reconstruction
  error in the geometry, displacing the photo's features (measured: it doubled the
  duplicate-feature count at three-quarter views).
- Retuned the conflict-resolution source-priority floor 0.45 -> 0.25 after ablation: 0.45
  handed contested cheek texels to stretched reference content; 0.25 still lets a head-on
  reference overrule truly grazing source rim samples.
- Guarded the mesh-graph harmonic fill against singular solves (fully unobserved
  disconnected components now fall back to the KD fill instead of painting black).
- The texture pipeline's evidence standard is now an adversarial QA harness (20 views x 5
  defect detectors + pose-aware identity gates, calibrated so the reference photos
  themselves pass): the rejected face bundle scored 66 failed checks; after this cycle the
  same bundle recipe scores 9-10, with the doubled-feature and ghost classes at zero.
  Remaining known limits are documented in ADR 0008 (eye-region geometry slivers, the
  front-vs-profile photo tone band, and hairline wispiness that a baked opaque texture
  cannot represent).

### Added

- Added `abstract3d.segmentation`: robust subject matting that prefers the
  `isnet-general-use` checkpoint and cleans the matte (dominant components kept, pinholes
  closed) before any geometric use. The default u2net checkpoint amputated 40% of the
  subject on the checked profile photo (dark hair against a light background), silently
  corrupting the alpha-driven framing of every downstream stage.
- Added canonical-frame orthographic projection to the shared texture bake
  (`projection_model="orthographic"`, `canonical_border_ratio`): the bake replicates the
  shape model's own image preprocessing (bounding-box recenter at the training border
  ratio) and projects with the orthographic half-extent that reproduces that exact frame,
  making source-photo registration deterministic. The Hunyuan backend uses this mode; the
  perspective model remains for TripoSR. See ADR 0007.

- Added an experimental, license-gated Hunyuan3D-2.1 shape backend
  (`abstract3d:hunyuan3d21-local`, aliases `hunyuan3d21`, `hunyuan3d`, `hunyuan`) wrapping the
  official `tencent/Hunyuan3D-2.1` flow-matching DiT and shape VAE. The backend refuses to
  download or run weights until the operator acknowledges the territory-restricted Tencent
  Hunyuan Community License (`scene3d_hunyuan_license_accepted` or
  `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`). Includes an adaptive coarse-to-fine volume decoder
  (host-side bookkeeping, exact-doubling level schedule) that replaces the upstream
  hierarchical decoder, which lost thin structures and misbehaved on Apple MPS. Ships with a
  new `abstract3d[hunyuan3d]` extra.
- Added `abstract3d.texturing`, a backend-agnostic projection texture bake shared by TripoSR
  and Hunyuan3D-2.1: silhouette-based source-pose estimation with an angular prior and IoU
  acceptance gate, 2D photo-to-silhouette registration plus interior-edge photometric
  refinement, contour alpha erosion, exposure harmonization for reference views, seam-feathered
  best-view-biased blending, geometry-verified mirror completion (3D observed-twin lookup),
  3D inverse-distance fill for unseen texels, and mesh-scale-aware camera distance
  estimation. TripoSR keeps its triplane color field as the unseen-texel prior.
- Added a CPU UV-atlas rasterizer fallback for the TripoSR texture bake so `baked_basecolor`
  works on hosts without OpenGL 3.3 / geometry-shader support (headless Linux, some Windows
  drivers). The ModernGL path remains the default when available.
- Textured preview renders now use a softer headlight shading model so contact sheets review
  the baked albedo rather than a single hard key light.
- Documented the TRELLIS.2 acceptance posture: the backend is maintained, and the gated
  DINOv3 companion is documented with a license summary (Meta DINOv3 License: commercial use
  permitted, "Built with DINOv3" attribution on distribution, acceptable-use restrictions,
  reviewed access), explicit unblock steps in `docs/models.md` and `docs/troubleshooting.md`,
  an upgraded runtime error message with the same guidance, and an acknowledgement entry in
  `ACKNOWLEDGEMENTS.md`.
- Added multi-view geometry conditioning through the official `tencent/Hunyuan3D-2mv`
  checkpoint (same license gate as 2.1, loaded from the pinned 2.1 source via a key-exact
  config namespace remap). Reference views whose angles snap to the trained front/left/back/
  right slots condition the shape itself; generation metadata records the views used. On the
  checked face proof, front + both profiles raised observed texture coverage from 0.19 to
  0.74 and replaced the hallucinated back of the head.
- Upgraded the multi-view texture bake through two adversarial review/empirical-attack
  cycles: per-reference pose solving in a window around the declared angle (real photo
  angles are routinely 10-20 degrees off their label; the refined pose is accepted only
  when it beats the declared pose's silhouette IoU by a clear margin), overlap-based
  per-channel color harmonization with a revert-on-confound rule (gains that fail to
  reconcile the overlap indicate content mismatch, not exposure), a reprojection-error QA
  gate evaluated against the union of previously accepted views (catches
  reference-vs-reference conflicts), per-texel best-witness conflict resolution (localized
  disagreement zeroes the weaker witness only on disputed texels instead of punishing the
  whole view), a mesh-scale-relative depth-occlusion tolerance (a fixed normalized
  tolerance let hair sheets bake through onto the face on large meshes), and a mesh-graph
  harmonic fill (Dirichlet Laplace solve over mesh edges, normal-weighted KD fallback) so
  hidden regions diffuse smooth colors along the surface instead of borrowing across
  space. The world-frame azimuth-sector masks in the projector now apply only when no
  rendered depth map is available: the depth test strictly dominates them, and empirically
  the sector mask discarded about half of the depth-validated texels on profile views
  (face proof coverage with front + both profiles: 0.50 -> 0.74).

### Fixed

- Integrated four adversarial audit findings (each proven on ground-truth harnesses before
  fixing): the pose-search grid never scored the DECLARED CENTER pose
  (`arange(-window, window, step)` excludes it unless window is a step multiple — rebuilt
  from symmetric integer offsets); `estimate_camera_distance` normalized column extents by
  width while the projector's NDC unit is height-based (up to +49% distance error on
  landscape frames) and used a one-pass linear correction biased 10-16% for deep subjects
  (now a 3-step fixed-point iteration); silhouette registration squashed photo masks
  anisotropically into the square comparison frame (now an aspect-preserving letterbox);
  and exposure harmonization gains are now gated on per-texel log-ratio spread (a true
  exposure shift is one tight multiplicative relation; content-mismatched overlap produced
  0.5-clamped gains that tinted whole views).
- Removed `refine_registration_photometric` from the default bake path: an adversarial
  ground-truth test recovered 0 of 15 injected known shifts, and the NCC objective proposed
  nearly the same warp regardless of the true offset (a constant attractor). The function
  remains available for callers.
- Mirror-symmetry completion now only copies from CONFIDENT observed texels
  (blend weight >= 0.35): 89% of unrestricted mirror sources on the checked face proof were
  grazing rim samples, and copying them fabricated a bright skin patch on the hidden crown.
- Fixed the ghosted "second face" on textured previews: the preview renderer's diffuse
  lighting re-drew the mesh's own geometric features (eye sockets, brow ridges, lip
  creases) over the photo albedo wherever geometry and texture disagree by a few percent.
  Textured meshes now render with a flat-biased headlight (12% diffuse cue); an
  independent CPU rasterization of the same textured mesh was used to prove the texture
  itself was correct. Also guarded a GLSL uniform that the shader change made eliminable.
- Fixed duplicate feature stamping onto hidden crust sheets: projector visibility is now a
  strict per-photo-pixel first-surface z-buffer built from the projected texels themselves
  (every surface texel occludes regardless of facing or photo alpha, 3x3 conservative
  widening, epsilon 0.25% of the surface diagonal). Replaces the GL depth-map tolerance
  test — any tolerance loose enough to survive depth-map interpolation stamped both sheets
  of the 0.005-0.02-unit hair films that generated meshes grow — and removes the
  world-frame azimuth sector masks entirely.
- Fixed reference-photo registration for differently-cropped photos: silhouette matching
  now scores symmetric edge-chamfer distance instead of region IoU (region IoU rewards
  degenerate blow-ups where the mismatch leaves the frame), and rows/columns touching the
  photo frame are treated as crop lines rather than shape boundaries.
- Added a mesh-surface outlier filter: a two-hop mesh-graph consensus iteratively erodes
  observed texels whose winning view AND color are foreign to their neighborhood (rim
  misprojections such as forehead pixels on hair-shell tips), demoting them to unobserved
  so the fill replaces them.
- Conflict resolution now gives the SOURCE photo priority on disputed texels wherever it
  faces the surface well (weight above 0.45): the user's actual photo outranks synthesized
  or auxiliary references on well-seen surface, while grazing rim content still defers to
  the best-facing witness.
- The mesh-graph harmonic fill now uses crease-aware edge conductance (normal-agreement
  squared), so hidden-region color diffuses along smooth sheets instead of leaking across
  shell fusion seams (skin bleeding up onto hair films).
- Fixed the pure-Python `torchmcubes` fallback so extracted meshes use the native torchmcubes
  `(x, y, z)` vertex convention and outward face winding. Without native torchmcubes installed
  (the default, since the package ships no wheels), every TripoSR mesh came out with swapped
  X/Z axes and inward-facing normals, which also silently collapsed observed-view texture
  coverage (facing weights saw inward normals). Environments with a locally compiled
  torchmcubes were unaffected, which is why earlier proof assets looked correct.
- Fixed a stale trimesh normal cache after `repair.fix_normals` in the TripoSR and Step1X
  cleanup passes: `Trimesh.invert` intentionally preserves cached normals across its cache
  clear, so vertex normals cached before the repair stayed inward even after the winding was
  corrected. The cleanup now drops the cache and recomputes normals from the repaired faces.
- Fixed TRELLIS.2 device selection to fall back to an available accelerator instead of
  returning `mps`/`cuda` unconditionally when explicitly requested on hosts without them.
- Made the validation harness process-guard portable: `start_new_session`/`os.killpg` are now
  used only on POSIX, with a psutil-based process-tree fallback for Windows.
- Changed the validation harness default device from `mps` to `auto`.

- Fixed composed `t23d` to default to automatic background segmentation instead of forcing
  `remove_background=False`. The official TripoSR pipeline always segments and recenters the
  subject before inference; feeding it opaque studio-background images degenerated thin
  subjects (e.g. chairs) into billboard-like sheets. The composed chair proof went from a
  collapsed 168-face sheet to a recognizable 77k-face chair with this fix alone.
- Fixed the observed-view projector to clip bilinear gather indices; texels projecting outside
  the image frame crashed the bake (newly reachable with estimated camera distances).
- Made the depth-occlusion renderer fall back to facing-only visibility when no standalone GL
  context exists instead of failing the whole texture bake.
- Fixed the validation harness memory guard to apply per heavy backend regardless of the
  requested device string; the previous `device == "mps"` check silently disabled the Step1X
  64 GiB guard once the default device became `auto`. Hunyuan3D runs now get the same default
  guard.
- Texture completion and inpaint gating now use the true projection coverage instead of the
  feathered blend weights, so seam-adjacent observed texels are no longer overwritten by
  mirror or inpaint fill.
- Unseen-texel fill now happens in 3D surface space (inverse-distance weighting over the
  nearest observed texels via a KD-tree) instead of UV space. UV-space fills bleed colors
  across unrelated xatlas charts and produced patchwork/speckle noise on hidden regions; the
  3D formulation lets thin parts (a chair back) borrow correctly from their opposite face.
  The upstream Hunyuan texture pipeline reaches the same conclusion with its mesh-graph
  inpainter.
- Projection weights are despeckled (small isolated coverage islands removed) and the
  position/normal atlases are no longer border-dilated, which used to let contaminated chart
  -gap texels pass the facing test and bake shadow pixels as speckle.
- Hunyuan3D meshes are decimated to a 120k-face budget before texture bake: above that,
  marching-cubes micro-detail fragments the UV atlas into thousands of confetti charts
  (3315 at 200k faces vs 87 at 120k on the owl proof) that show up as salt-and-pepper noise.
- Fixed the matplotlib preview fallback to honor the requested image size (it rendered at
  1.6x due to a figsize/dpi mismatch) and to survive GLB textures carried as raw encoded
  bytes, both of which broke previews on GL-less hosts.
- Fixed the pure-Python marching-cubes fallback to return an empty mesh for fields with no
  iso-crossing, matching native torchmcubes instead of raising.

## 0.1.0

- Added the first production-oriented `scene3d` plugin surface for AbstractCore.
- Added a validated local backend based on `stabilityai/TripoSR`.
- Added composed `text_to_scene3d` through `abstractvision` image generation plus TripoSR reconstruction.
- Added CLI commands for `catalog`, `i23d`, `t23d`, and `validate`.
- Added bundle outputs with previews, contact sheets, and per-case metadata.
- Added a reproducible local validation harness in [`scripts/validate_local.py`](scripts/validate_local.py).
- Added docs, ADRs, and benchmark assets for the validated Apple-local path.
