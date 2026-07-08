# CORRELATION AGENT B — CODE-FIRST ADVERSARIAL AUDIT
2026-07-07 · repo `/Users/albou/abstract3d` @ certified cycle-6 state · independent of agent A

Direction of this audit: **code → derivable defect classes → assets**. I read every
mechanism in the certified pipeline, constructed its violation space from first
principles, then checked (a) which violations the test suite pins, (b) which were HIT
during the six cycles (mapped to the critic-1 ledger), and (c) which are LATENT —
no test, no ledger entry, plausible on the next user input. MeshVault (MCP, headless
viewer) and offline geometric simulation were used to probe the latent ones on the
exact certified bytes.

## 0. Audit basis (what I verified myself, on the exact bytes)

| check | result |
|---|---|
| on-disk texture md5s vs certification table | **match**: face `928705f3…`, ship `b8e2b0d4…`, owl `ff746509…` |
| `texture_qa` face-2mv (my run) | **PASS 13/13** (brightness 0.954, fill energy 0.838, 0 dark smears) — reproduces the certified claim |
| `texture_qa` ship / owl (my runs) | **PASS 13/13 both** (0 failed gates; results in `/tmp/mvb/qa_ship/`, `/tmp/mvb/qa_owl/`) — all three certified `texture_qa` claims reproduce |
| geometry.glb vs scene.glb consistency, all 3 bundles | **consistent**: same face count; scene vertices = geometry vertices split at UV seams; max nearest-vertex distance 0.0; MeshVault `compare_models` (owl, 1500 samples) classifies `identical`, asymmetry 0.0 |
| MeshVault `describe_scene` issue detector | ship, owl: no issues. face: 1 non-manifold edge at (0.273, −0.476, 0.404) (informational; below any visual bar) |
| 48-view-class visual spot check in a real PBR viewer (IBL, spec materials) | face frontal/profile/rear, ship 6 views, owl 4 views — consistent with the certified reads; temple wisp band and ship fill-side softness visible but match their accepted/proven-limit states (screenshots in `artifacts/validation/texture-cycle-proofs/meshvault/agentB/`) |

Viewer note: the exported GLBs are Z-up/front-+X; MeshVault presets are Y-up. All
agent-B screenshots were taken after `rotate x −90`, with viewer azimuth = projector
azimuth + 90° (verified against the frontal face).

## 1. Per-mechanism audit table

Legend: **tested** = unit/regression tests exist for the stated violation class;
**hit** = ledgered during cycles 1–6 (ID); **latent** = derivable from the code, not
tested, not ledgered. Only load-bearing violations are listed; L# refers to the
latent-risk register in §2.

### 1.1 Projector — `backends/triposr_runtime.py`

| mechanism | intended invariant | violation space (code-derived) | tested | hit in cycles | latent |
|---|---|---|---|---|---|
| Ortho/perspective sampling + canonical recenter (`recenter_to_canonical_frame`, `projected_frame_center_px`, `canonical_ortho_half_extent`) | photo pixels land on exactly the surface that imaged them, deterministically, in the model's own conditioning frame | wrong camera model for the photo (true-perspective selfies under ortho); frame-center convention mismatch at off-front poses; alpha-bbox recenter corrupted by matting; frame/pose co-adaptation broken by re-centering one side | round-trip on sphere; frame-registration recovery; recenter convention; `projected_frame_center_px` convention (test_texturing 531, 551, 563, 1785, 1830, 1858) | SHIP-03 nose melt (54 px frame offset; FIXED c3); ADR-0007 §2 fovy-40 pinhole (pre-cycle) | **L2 perspective input photos**; soft-alpha framing (minor, unranked) |
| Strict z-buffer + slope-aware epsilon | a photo pixel paints only the first surface along its ray; no two-sheet ghosting; tilted surfaces don't self-reject | epsilon too small → self-rejection at tilt (was: 40% of visible texels demoted); too large → hidden-sheet leak where sheet gap < ε; bin resolution mismatch photo↔atlas; rim conservatism eats 1 px of true content | hidden-sheet rejection (test_texturing 635); out-of-frustum survival (379) | ADR-0008 math defect #3 (self-rejection, FIXED pre-cycle) | leak window between ε and layered-zone floor on grazing sheets is unmonitored (minor; bounded by facing gate) |
| **Layered-zone gate** (density > 0.10 in win 3% + local luminance std > 0.055) + contested dilation | mixture pixels over stacked film shells never stamp; same-material layering is spared (contrast condition); contested texels never seed mirror copies | (i) contrast is an **absolute** luminance-std threshold — dark-albedo/low-key photos evade it and mixture stamps return; (ii) chroma-only contrast (equal-luminance hue boundary) evades it; (iii) **layered/greebled real geometry** (hull plating, overhangs) triggers density and surrenders *witnessed* content wherever the photo has contrast; (iv) whole-thin objects (< 3% diag: fins, leaves, cards) read their own back face as a second sheet → whole-silhouette surrender | film-band surrender (test_texturing 813); thin-bright-sheen kept + dark-majority fail-safe (test_film_band_boundaries) | FACE-01/02/16 family = the gate's *absence* on fused films (FIXED via film_band); ADR-0008 notes coverage 0.57→0.40 cost | **L1 dark-albedo evasion** (measured: contrast-band pass rate 67%→26% at 0.35× exposure); **L4 witnessed-content surrender on layered geometry** (measured: ~14% of the ship's source-frame pixels meet density+contrast); **L5 thin geometry** |
| Facing weight (strength², per-role threshold 0.4/0.2) | stretched grazing content only paints when nothing better exists | per-role rule keys on `len(views)>1` only — a multi-view bake whose references never cover the far side leaves the 0.4 source cut-off with **no** replacement witness → hole becomes fill | implicit in end-to-end tests | rear-quarter losses absorbed into FACE-09/curtain rulings | unranked (minor; interacts with L4) |
| Geometry confidence (Jacobian σ_min stretch p=2; concavity×grazing ×0.25) | collapsed texel→photo mappings and concave-grazing texels can't win texels | nominal = median σ_min over facing>0.7: bimodal chart densities mis-normalize; whole-image grazing (no facing>0.7 population) falls back silently; concavity demote can starve legitimately-witnessed deep folds seen at 0.35<facing<0.5 | stretch demotion + concave-grazing-only (test_texturing 1510, 1541) | FACE-19-adjacent (delight interplay); eye-socket smear class (c2, FIXED) | mixed-density UV atlases (low; xatlas is uniform) |

### 1.2 Orchestration — `texturing.py`

| mechanism | intended invariant | violation space | tested | hit in cycles | latent |
|---|---|---|---|---|---|
| Photometric pose estimation (plateau centroid, chirality tie-break, relative margin +25%, absolute floor 0.008) | weak evidence never moves the projection; jitter cannot re-roll the pose; mirror pose can't win | subjects with flat interior-gradient landscapes score under the floor → declared pose wins even when wrong; ±40° window truncates large head-turns; elevation grid −15..+15 misses high/low captures; score floor calibrated on 3 subjects | injected-yaw recovery + frontal rejection (test_texturing 776); declared-center grid (511) | OWL-04 pose lottery (az+32.5 @ 0.0043) and FACE-18 drift (+20→+12.5 @ 0.0052) — both FIXED by the floor/margins; c1 verification round 1b | **L3: out-of-window/beyond-grid poses on next input**; floor generalization unproven beyond these subjects |
| Registration stack: width-profile → edge-chamfer residual → overlap-photometric → dense lattice flow (`reference_flow`) | references register interior content to the source's painted truth; unvalidated cells at identically zero flow | width-profile assumes subject TOP never cropped (hats/headroom crops break it); overlap fit needs ≥400 overlap texels at weight>0.25 (profile-only view pairs can drop under); flow validation median-of-improving-cells is self-referential when few cells improve | flow warp recovery, gate zeroing, unreachable rejection (test_reference_flow); overlap shift recovery (test_texturing 933) | FACE-04 mouth-corner ghost (displaced ref lip, w 0.006–0.2; FIXED); 58 px silhouette-registration nose error (pre-cycle, ADR-0008) | L11 cropped-top photos; L12 overlap starvation on sparse-view sets |
| `delight_projections` (SH ratio fit + overlap-proximity fade + revert gate) | shading differences removed only where cross-view consistency is measurable; exclusive territory bit-identical | unfaded version relights exclusive sides (was FACE-19); luminance-only assumption vs colored lighting (WB differences leak to `harmonize` stage); fade density thresholds on sparse overlap | two-light sphere recovery; fade protects exclusive; chroma kept + confound revert (test_texturing 1259, 1281, 1317) | FACE-19 side_right MAE 24→32-40 (delight kept unfaded; FIXED c2/c3 by fade) | low residual risk; the certified face bake ran with delight applied only where improving |
| `harmonize_and_gate` + `resolve_projection_conflicts` (source priority 0.25) | one multiplicative exposure relation per reference; local conflicts resolved per-texel with the source as ground truth where it sees well | gain fitted on content mismatch (IQR spread gate 0.7 mitigates); source-priority floor 0.25 hands contested texels to a stretched source at 0.25<w<0.45; union-gating order effects (reference 2 judged against attenuated reference 1) | exposure fix + disagreement gating (432); low overlap skip (468); best-witness + grazing-source deference (484, 688, 708) | FACE-06 tone split (S2+S1; FIXED via gradient compositing); v14 "+9% stretch" attribution (c1) | order-dependence unprobed for >3 views (next multi-photo input) |
| `filter_projection_outliers` (2-hop no-self-vote consensus, 3-pass erosion) | foreign islands can't survive on self-support; same-view extremes also drop | color_threshold 0.3 absolute: low-chroma subjects under-trigger; iterative erosion can eat thin genuine features whose view differs from surround (protected only by same-view rule) | planted island dropped incl. no-self-vote (728, 879) | ADR-0008 math defect #1 (self-votes; FIXED) | speckled-material rims (folded into L7) |
| Mirror machinery: `mirror_fill_from_observed` (source floor 0.35 + contested exclusion + consensus guard + graceful top-up), `tone_match_completion_components`, rescue discs | hidden texels get only confident, uncontested, consensus-compatible twins; completion tone can't print component borders; badly-witnessed twins of strong features get transplants, legitimately asymmetric content untouched | **geometry-only symmetry gate**: albedo asymmetry (text, logos, port/starboard markings) mirrors as false content; tone match is **bright-copies-on-bright-rings only** — dark-material completion borders unmatched; rescue placement anchored on weak witnesses (capped 0.4·r); disc transplant is whole-disc (measured trade, accepted) | consensus guard (1155); gate continuity + no-junk-sources (test_mirror_source_gate_robustness); tone-match scoping (2211); rescue fire/no-fire quartet (1601–1688); transplant tone match (1716) | FACE-15 (−90 eye; FIXED via rescue); FACE-22 owner #2 (mirror-copy border contours; FIXED via tone match); crown skin patch (c2, FIXED via source floor) | **L6 mirrored albedo on asymmetric-albedo subjects**; **L9 dark-side completion contours** |
| Fill stack: harmonic (crease conductance) → texel smooth → detail synthesis (strand comb, closed-loop energy, amplitude floor, sigma guard) → luminance floor v2 (octant balls, donor consensus, evidence exemption) | fill is C0, material-plausible, never darker than context/donor consensus, never noisier than the witnessed band; observed texels bit-identical | maximum-principle transport of dark anchors (the floor's reason to exist); bright-half medians assume bimodal luminance — uniformly dark subjects neutralize gates (minority gate correctly stands down, but then nothing bounds transported darkness); floor absolute 26/255 vs deep-black materials; strand comb multi-view-only; detail donors selected by color-similarity undershoot energy (closed loop compensates) | 15+ tests: harmonic IDW, smooth anchors, amplitude/energy/granite guard, floor lifts-vs-keeps, donor anchor, opposite sheets, mirror exemption, strand comb on/off/bit-identical | SHIP-02/OWL-02 dark smears (FIXED via floor); SHIP-08 fill-energy collapse (FIXED via calibration); FACE-09 leopard (FIXED-to-material-bar via strand comb); owl seam regression at 2048 (FIXED via world-cell connectivity) | bright-decal halos on dark subjects (unranked); strand comb untested on non-hair anisotropic materials (wood grain next multi-view input, part of L8) |
| `commit_trace_deposits` + `commit_pale_chips` (+rim feather, whole-neighborhood rule, frontier-sliver isolation) | trace-weight content contradicted by a multi-witness uniform ring is vacated and retoned; confident content never touched; single-view = structural no-op | ring-consensus semantics assume material uniformity around chips: **real speckled materials** (knots, rivets, stars) witnessed only at grazing/trace weight in a uniform ring carry the exact chip signature → erased on the next subject; bright_median bimodality assumption; absolute `deviation_min` 0.045 | retone + never-touch-confident + no-op single-view + feature-halo + rim feather + frontier slivers (2027–2253); pale-chip quartet (2614–2711) | FACE-03/04/05 residue chips (FIXED); FACE-22 owner #1 (trace rims; FIXED via rim feather); FACE-07 ear chips (pale commit) | **L7 grazing-witnessed real speckle erased** |
| `tone_bottom_cap` | synthetic −z cut faces get their rim's tone, geometry-detected, multi-view only | detection = −z planar component + <1% direct witness + thin slab: a *table top seen from above* / vehicle underside in a future multi-view bake qualifies (all three certified meshes pass the area gate: face 9.7%, ship 25%, owl 7.4% of −z-aligned area) → retone of legitimate (fill) undersides from rim colors; only −z axis handled | cap toned from rim + no-op without cap (2763, 2782) | FACE-12 disc (FIXED) | L13 multi-view flat-bottom subjects (bounded: it replaces fill with rim continuation, usually acceptable) |

### 1.3 Cycle modules

| mechanism | intended invariant | violation space | tested | hit in cycles | latent |
|---|---|---|---|---|---|
| `gradient_compositing` (graph + stitch gates, gradient selection with 1-sided/tier-2 witnesses, line down-weight, material gate, source anchor boost, screened-Poisson MG-CG) | tone seams vanish mathematically; witnessed edges survive verbatim; resolution-invariant; material borders not tinted | stitch normal gate 0.5 vs grid edges that carry **no normal check** (a chart wrapping a thin sheet could couple front/back — xatlas rarely does); f32/f64 singular-floor subtleties (guarded); material_step_cap 0.18 absolute; **docstring claims completion classes ride through the solve, but the only call site runs the solve before mirror/fill exists** (see D3) | graph stitch/opposed sheets; selection incl. one-sided; exposure-step removal; anchors pinned; end-to-end + determinism (test_gradient_compositing 89–320) | mid-face chroma seam at 2048 (line-weight imbalance; FIXED); ear/temple wash (material gate + confidence floor; FIXED); FACE-22 owner #3 (completion tone handoff missing — patched by `tone_match`, not by the solve) | D3 doc-vs-code; L9 (dark completion borders remain outside every tone-reconciliation path) |
| `reconcile_specular_lobes` | baked view-dependent highlights on the source can't ship as albedo; only cross-view-authorized, smooth, bright-base lobes level | genuine bright-desaturated albedo (white blaze/paint) reads as lobe **only if** a second view reads it darker beyond gauge — safe by construction; gauge from co-witnessed medians fails on <200-texel populations (falls to 0 silently); dark standoff assumes debris-class economics | flatten/keep/edge-refuse/single-view no-op (367–431) | FACE-05/17 pale column (FIXED c4, provenance overturned to baked specular) | iridescent/metallic multi-view subjects: every view disagrees beyond gauge → widespread leveling of real material response (next product-shot input; MED) |
| `reconcile_shadow_aprons` | where the source validly reads co-witnessed surface darker beyond gauge (its cast shadow), the composite carries the source's shading; never brightens; never touches source-invisible texels | the doctrine itself: a *badly lit source* (harsh flash) degrades well-lit references toward its shadows — accepted trade, but next-input-sensitive; smoothness/edge gates tuned on skin | apron carried/evidence-required/edge-refused/single-view no-op (488–564) + boundary tests (test_shadow_apron_boundaries: pure exposure no-fire, occluder not printed) | FACE-14 neck wash (C5 provenance corrected: reference lit tone over source's genuine shadow; FIXED) | doctrine trade-off on poorly-lit sources (MED-LOW) |
| `film_band` (zone extension, ≥2-witness flag consensus, veto, commit-coupled vacate, dark-dominance scaling, retone) | fused mixture bands surrender only where every first-surface witness agrees and no view positively witnesses base material; commit tone from dark anchors, wispiness-scaled | all thresholds relative to bright-half medians (good) **except** the inherited `min_contrast` 0.055 (absolute, L1); `dark_body_mask` needs a ≥2% dark component — dark-material subjects make *everything* body (fail-safe tested); bands only for ≥2 views | maps extension/veto, consensus, single-view no-op, retone direction (test_film_band); thin sheen kept, dark-majority fail-safe (boundaries) | FACE-01/02/16 (FIXED, c2–c3) | inherits L1; multi-view *non-face* subjects with large dark masses (black products) will run the full film machinery — validated on one subject class (L8) |
| `film_band_gradient` repaint (geodesic S field, source authority, moat + displaced-component veto + refill floor, support bound 6T, stamp feather, island guards, ≥7-texel transition bail) | the hairline apron carries the photos' own falloff and the source's own strands; strokes unprintable by construction; treatment confined to the profile's measured support | transition < 7 texels (sharp hairlines, small subjects in frame, 1024 bakes) → silent fallback to the weaker retone → putty apron class returns; S-field extrapolation beyond support (was FACE-22 owner; now bounded); moat/veto constants tuned on one subject | descending profile, feature split, apron replaced, standoff, displacement veto, **support bound refusal**, stamp feather, no-op contracts (test_film_band_gradient) | FACE-20 strokes (FIXED c4 via displacement veto); FACE-22 owner #1 (support bound, FIXED c6); 1024 skin-shred regression (FIXED via outermost-sheet + adequacy bail) | **L10 repaint silently off below its adequacy floor** — next low-res or sharp-hairline bake ships the weaker path with no gate flagging it |
| `feature_fringe_repair` (gate correspondence, complex stamps full→trace→skip, texel+render structure vetoes, photo-truth exemption bounded by battery worst, speck consolidation, disc refresh) | deposits inside feature complexes are replaced by the photo's own content under the identity gate's own correspondence; the repair can never create/destroy feature-scale blobs or grow micro-debris beyond baseline; single-view no-op | correspondence is a global bbox+NCC similarity — fails on subjects whose render-vs-photo NCC basin is ambiguous (returns garbage registration → `gate_ok` mostly false → mostly no-op, fail-safe); veto battery = 15 renders ×896 px per candidate (cost); per-acceptance re-arming of the +0.0003 growth budget was the critic-2 veto gap at audit start | fringe registration recovery, blob classifier, world clustering across UV cut, texel veto new-blob-only, no-op contracts (2409–2559); **cumulative-baseline rearm test (2559, landed mid-audit)** | FACE-03 partial (PROVEN-LIMIT: repair costs eye_count); FACE-21 recipe variance (identity_image matters; process-fixed); c6 knife-edge margins | **L15 — CLOSED mid-audit**: cumulative-baseline veto landed at 16:41 with wiring + test (see register) |
| `reference_flow` | strictly local, validated-cell-only warps; unreachable content never moves | leash/validation on 48 px cells: sub-cell features can ride a validated neighbor; median-of-improving-cells reference undefined when improving set is tiny (falls back to inf → everything within absolute factor) | recovery, gate zeroing, unreachable rejection, no-overlap return (test_reference_flow) | ghost lip/lash fragments (FIXED family) | LOW residual |

## 2. LATENT-RISK REGISTER — ranked by likelihood on the NEXT user input

The certification is scoped to "these inputs". This register is what bites when the
inputs change. Rank = probability × severity on a plausible next input.

| # | risk | mechanism(s) | why it fires next | evidence | severity |
|---|---|---|---|---|---|
| **L1** | **Dark-albedo / low-key photos evade every luminance-contrast and bright-half-median constant** — the layered-zone `min_contrast=0.055` (absolute std), and the film machinery seeded by it, stop surrendering mixture bands; the FACE-01 "beige film" class returns on a dark-skinned subject, dark clothing, or an under-exposed capture. Bright-half medians (`DARK_LUMINANCE_RATIO` splits) are scale-free, but the *contrast* condition is not. | projector zone gate; `film_band.compute_view_film_maps` (`WEAK` mask) | any subject/exposure darker than this studio portrait | measured on the real face photo: fraction of the hair-boundary band with windowed std > 0.055 falls 66.7% → 46.1% → 34.8% → 25.8% at exposure ×1.0/0.6/0.45/0.35 (`/tmp/mvb/numeric_probes.py`, P2) | **HIGH** — blocking-class defect family returns silently |
| **L2** | **True-perspective input photos** (phone selfies, short-lens product shots): the canonical path projects orthographically; near features magnify relative to far ones and nothing estimates or corrects focal length. Features can never land where the model built them — a milder, global cousin of the doubled-feature class, invisible to every current gate except identity SSIM. | canonical ortho projection (ADR-0007 §1) | most casual users shoot phone photos at <1 m | code-derived; the pipeline has no perspective-estimation hook for the canonical path (the perspective model exists but is only used for TripoSR) | **HIGH** for casual capture; low for studio/long-lens |
| **L3** | **Pose estimation beyond the searched envelope**: ±40° azimuth, elevations {−15…+15}+refine. A photo at 50° yaw or +30° elevation either scores under the floor (falls to declared az0 — *wrong* by construction) or locks a wrong local plateau. The certified fix (floors/margins/pins) makes bad estimates *refusable*, not *correct*; the owl only survives via the honest-decline path + pinned recipes. | `estimate_pose_photometric`; bake wiring | any "front" photo that isn't near-frontal | ledger (OWL-04, FACE-18, verification 1b: score 0.0043–0.0052 commits measured); window/grid constants in code | **HIGH** frequency, MED severity (declared-pose fallback misplaces every feature when the photo is genuinely off-front) |
| **L4** | **Layered-zone gate surrenders *witnessed* content on layered/greebled geometry**: on the certified ship, ~31% of source-frame pixels meet the second-surface density condition and **~14% ALSO meet the contrast condition** (wing 16%, body 13.8%) — i.e. the projector refused a measurable slice of exactly the panel-lined content the photo really witnessed; the fill stack then re-synthesizes what was real. Owl: 1.7% (benign). Face: 15.1% (intended — hairline/curtain mixtures). More detailed subjects surrender more, *because* they have contrast. | projector zone gate | any mechanically detailed subject (vehicles, machinery, buildings) | simulation of the exact density+contrast statistic in the source frame on the certified meshes+photos (`/tmp/mvb/probe_zone_ship.py`); MeshVault `ship_photo_p225.png` vs `ship_fill_m225.png` shows the crisp-vs-soft asymmetry | **MED-HIGH**; also see disagreement D1 |
| **L5** | **Whole-thin geometry** (< 3% of bbox diagonal front-to-back: fins, wings thinner than this ship's, leaves, cards, blades): the object's own back face lands inside the zone gap window `(3ε, 0.03·diag]`, the density condition holds across the whole silhouette, and wherever the photo has ≥0.055 contrast the entire witnessed surface is surrendered to fill. The certified ship's wing is *not* thin enough (extent p50 0.42 ≫ gap_max 0.069; only ~6% of its bins fall in-window), so this never fired in the cycles — no test covers a thin *object* (the layered-gate test uses a hovering sheet over a body). | projector zone gate | next flat/winged asset | measured extents on the certified ship (`/tmp/mvb` shell, wing/body extent stats); code window constants | **MED-HIGH** for the class, unbounded severity when it hits (whole-object fill) |
| **L6** | **Mirror completion fabricates mirrored albedo on geometrically-symmetric, albedo-asymmetric subjects**: the gate is geometry-only (score ≥ 0.55); text, logos, insignia, asymmetric paint mirror onto the hidden side — *mirrored text* is an owner-visible blocking defect on any labeled product. The consensus guard rejects copies contradicting *observed* neighborhoods, not reality; the hidden side has no observed neighborhood. | `texture_completion="auto"` + `mirror_fill_from_observed` | any product with printed text/logo photographed from one side | code-derived; all three certified subjects are label-free (why it never ledgered) | **HIGH** severity, MED frequency |
| **L7** | **Grazing-witnessed real speckle erased by the commit lanes**: a dark knot/rivet/marking in a uniformly bright ring (or pale chip in dark ring), witnessed only at trace weight (near rims/grazing), carries exactly the signature `commit_trace_deposits`/`commit_pale_chips` vacate. On speckled materials (wood knots, rivet lines, starfields) the "defect signature" IS the material. Multi-view only, so canaries never see it; tests cover "never touch confident" but not "trace-weight real content in uniform surround". | trace/pale commits | next multi-view speckled subject | code-derived (gate semantics); `deviation_min=0.045` absolute | **MED** |
| **L8** | **Multi-view non-face subjects run the whole face-derived stack unconditionally** (`len(projections) > 1` is the only gate): film bands (needs a ≥2% dark body — a black handbag qualifies), hairline repaint (skin ring = any bright ring), specular/shadow reconcile, trace/pale commits, bottom cap, fringe repair. Each has internal no-op contracts, but their *constants* (moat radii, S thresholds, ring brightness 0.96, halo 1.4×) were fitted on one human head. First multi-view product bake = first real-world execution of ~8 mechanisms at once. | all multi-view lanes | the very next two-photo bake of anything | code-derived; every "measured" constant in the docstrings cites the face proof | **MED-HIGH** aggregate |
| **L9** | **Dark-material completion borders have no tone reconciliation**: `tone_match_completion_components` is scoped to pure-bright copies on bright rings (measured to avoid minting dark debris). Mirror components landing in dark material (hair, dark hull) keep verbatim twin tone; their borders can print the FACE-22 contour class in dark regions where the pale-chip lane's area cap (1.2e-3) excludes extended components. Sub-visibility on this face (dark-on-dark); not guaranteed next time (lighting-asymmetric dark materials). | mirror completion + gradient path ordering | asymmetric lighting + dark material | code path analysis; FACE-22 fix scope is explicitly bright-only | **MED** |
| **L10** | **Hairline repaint silently disabled below its adequacy floor** (transition < 7 texels): sharp hairlines, small-in-frame heads, or 1024-resolution bakes fall back to the weak cycle-2 retone — the putty-apron class un-fixed with no stat surfacing it above `film_band.gradient_repaint = None`. The certified quality of the hairline is a 2048+, soft-hairline result. | `repaint_film_band` bail | crew cuts; default-resolution runs on smaller GPUs | code constants + docstring's own 4.8-texel failure measurement; face transition ≈15 px at the 512 input (`P4`) | **MED** |
| **L11** | **Cropped-at-top photos** break width-profile registration's core assumption ("the subject's TOP is almost never cropped") — hats, headroom-tight crops, truncated products; the fallbacks (edge-chamfer with crop-line exclusion) handle sides/bottom but the top anchor is single-point-of-failure for scale. | `register_view_by_width_profile` | portrait crops from social media | code-derived | **MED-LOW** |
| **L12** | **Reference-view sparse-overlap starvation**: overlap-photometric registration and delight both bail under ~400 texels at weight > 0.25; two profiles + no frontal (a plausible next capture set) leave references registered by silhouette only — the displaced-interior-feature class (58 px nose error, ADR-0008) returns for the un-anchored pair. | `register_reference_by_source_overlap`, `delight_projections` | non-standard reference sets | code-derived; min_overlap history (800→400 after a silent disable) shows the cliff is real | **MED-LOW** |
| **L13** | **`tone_bottom_cap` on legitimate flat undersides** in future multi-view bakes (vehicle/furniture): −z planar + <1% witnessed + thin slab retones from rim colors and smooths at σ24. All three certified meshes already pass the area gate (face 9.7% / ship 25% / owl 7.4% of −z-aligned area) — only the multi-view scoping kept it off the canaries. Usually benign (rim continuation ≈ right answer), but it will flatten synthesized detail tone under a subject whose underside should stay distinct. | `tone_bottom_cap` | multi-view vehicles | measured −z-aligned area fractions (`P3`) | **LOW-MED** |
| **L14** | **Bit-determinism is host-scoped**: CG/f32 solves, BLAS reductions, and KD-tree tie-breaking are deterministic per host but not across BLAS/architectures; the canary-md5 contract will "fail" on a different machine without any real regression, inviting a wrong re-certification loop. | maintenance contract §4 | first bake on new hardware | standard numeric reasoning; certification determinism evidence is single-host | **LOW** (process, not artifact) |
| **L15** | ~~The adopted cumulative-veto hardening is not yet code~~ — **CLOSED DURING THIS AUDIT.** At my first read of `feature_fringe_repair.py` the per-stamp +0.0003 budget re-armed per acceptance (critic-2's measured ~7-stamp → +0.00096 creep) with no original-baseline term. At 16:41 (mid-audit) a concurrent change landed the cumulative-baseline veto exactly per the contract wording (refuse when post > original + 0.0003 AND post > original battery worst), wired at the call site (`original_renders=pre_renders`), with a dedicated regression test (`test_render_veto_cumulative_baseline_closes_rearm_creep`) and a CHANGELOG entry. Residual risk drops to LOW; the certified canary md5s must be re-verified A/B per contract item 4 before the next publish (the shipped assets are unchanged — texture md5s still match). | `_render_structure_veto` | — | before/after file reads in this session; mtime 16:41:43; test + CHANGELOG grep | **LOW** (was HIGH at audit start) |

## 3. Explicit disagreements / challenges to the official record

**D1 — "SHIP-01/04/07 are pure single-photo content limits" is overbroad.**
The projector's layered-zone gate measurably refuses ~14% of the ship's *source-frame*
pixels (density + contrast satisfied — my simulation of the exact statistic on the
certified geometry + photo, `probe_zone_ship.py`; wing 16%, body 13.8%; precision
caveat: my photo alignment is the plain canonical recenter, the shipped bake used the
projector-frame recenter — a ~54 px offset at 1024 that shifts the contrast overlay
but not the geometry-only density condition, which alone reads 31%). That is
witnessed panel content the pipeline itself surrendered to fill, i.e. part of the
photo→fill transition structure on the *near* hull is pipeline-caused, not
photo-absent. The PROVEN-LIMIT grants (C2 ceiling experiment) are cited for the *far*
side and fill character; unless the ceiling experiment drove a synthetic perfect
texture through the **projector** (not just the fill), the register's "nothing the
pipeline could have prevented from these inputs" claim is not proven for the
surrendered near-side slice. Remedy unchanged (a second photo also fixes it), but the
attribution line "content limit" should read "content limit + measured zone-gate
surrender (~14% of frame)".

**D2 — The zero-defect scope quietly excludes the largest next-input hazards.**
Consistent with its own wording ("within THESE inputs"), but the certification's
maintenance contract watches *margins* (SSIM/MAE/debris) while none of the register's
top latent classes (L1 dark albedo, L2 perspective, L6 mirrored albedo) are visible to
those margins on the canary set — the canaries are all bright-albedo, long-lens,
label-free. A canary with dark albedo + printed text would cover 3 of my top 6 risks
at zero pipeline cost. Filed as a challenge to the contract's canary composition, not
to the verdict.

**D3 — `gradient_compositing` documentation misstates shipped behavior (doc-vs-code).**
`select_composite_gradients` documents mirror/fill classes (1/2) riding through the
solve, and the module docstring says completion content "rides into the same solve
unchanged" — but the only call site (`bake_projection_texture`) runs the solve
**before** mirror completion and fill exist, passing `class_map ∈ {0, −1}` only. The
class-1/2 path is dead code in production. This is exactly why FACE-22's third owner
(completion tone handoff) existed and why `tone_match_completion_components` had to be
bolted on afterwards. The cycle-6 provenance already concedes the behavior; the
disagreement is that the module still advertises the un-executed contract, which will
mislead the next solver who "fixes" completion tone by touching the solver.

**D4 — FACE-22's closure standard vs the code's own capability.**
The ruling closes FACE-22 on "no line-art read at 4x native contrast" while a 2–98%
stretch still traces fragments. I accept the visibility ruling (my MeshVault neck
views at native contrast show no line-art), but note the code now contains *four*
separate border-mixture feathers (trace-commit rim, stamp border, standoff, support
feather) — the recurring geometry (verbatim content meeting composite across a cut)
is a structural property of every stamp-class mechanism; each new stamp lane to date
has re-discovered it one cycle late (c3 strokes → c4 veto; c5 cleanup → c6 rims). Any
future stamp mechanism should inherit a shared border-feather contract instead of a
fifth bespoke fix; nothing in the maintenance contract encodes that.

**D5 — Owl "rear texture" and the fill-floor lift.**
The owl fill floor lifted 27.5k texels at mean 0.53 log (metadata) — the strongest
tone intervention on any canary — and the rear reads as clean-but-synthetic speckle
in MeshVault (`owl_rear.png`). OWL-03 is PROVEN-LIMIT (content), which I accept; I
flag only that the owl is the *regression* canary while carrying the largest
statistical repaint: a future floor regression would move the owl's rear silently
inside `texture_qa`'s gates (fill-character ratio has a wide 0.5–2.5 window).
A tighter owl-specific expectation (e.g. pinned rear-crop hash at the QA level) would
make the canary actually sensitive where it is most treated.

**Agreements worth stating** (they were verified, not assumed): the material-truth
gates (ADR-0009) hold in a real spec viewer — the assets read correctly lit in
MeshVault under IBL; `texture_qa` face PASS 13/13 reproduces; the geometry/scene
consistency claim holds bit-level (vertex sets identical up to UV-seam splitting);
FACE-22's glyph/contour is gone at owner conditions in an independent viewer; the ship
nose (SHIP-03) reads structured head-on (`ship_nose_front.png`); the certified md5s
are the on-disk bytes.

## 4. MeshVault evidence index (`artifacts/validation/texture-cycle-proofs/meshvault/agentB/`)

| file(s) | probe | reading |
|---|---|---|
| `face_az0_frontal.png`, `face_srcpose.png` | identity surfaces, viewer truth | frontal coherent; temple wisp band visible at native contrast (C5-accepted state); no film band, no black strokes, no ghost features |
| `face_neck_m22.png` | FACE-22 site at az −22.5 | no line-art/glyph read at native contrast — closure verified in an independent viewer |
| `face_rear.png`, `face_left_profile.png` | FACE-09 / curtain / ear | rear reads as combed dark hair material (no leopard); ear region carries the proven-limit debris class faintly |
| `face_disc_below.png` | FACE-12 bottom cap | disc toned as rim continuation; streaky marks visible from below (accepted state) |
| `face_xsec_sagittal_solid.png`, `face_xsec_axial_solid.png`, `face_xsec_axial_wire.png`, `face_xsec_coronal_tex.png` | film-shell geometry premise | axial cut shows the scalp ringed by *dozens of separate thin shell flaps* (the wire view resolves individual sheets) — direct visual confirmation of the layered-zone/film-band geometric premise and of why alpha-carrying wisps cannot be baked opaque |
| `ship_nose_front.png` | SHIP-03 | intake structured, no melt |
| `ship_photo_p225.png` vs `ship_fill_m225.png` | SHIP-01 asymmetry + L4 | crisp panels vs softer granular fill; the pair is the visual footprint of both the content limit *and* the measured zone surrender |
| `ship_underside.png` | SHIP-04 | plausible granular wash, no dark smears (floor working) |
| `ship_xsec_wing_solid.png` | L5 thin-geometry check | wing section thick relative to the zone gap window (matches the numeric finding: this ship is safe; the class isn't) |
| `owl_front.png`, `owl_rear.png` | canary state | clean; rear = synthetic speckle at fill-floor tone (see D5) |
| `face_base_*`, `ship_base_*`, `owl_base_*` | orientation/baseline walkarounds | includes the Y-up/Z-up preset mismatch documentation |

Numeric probe scripts and raw outputs: `/tmp/mvb/numeric_probes.py`,
`/tmp/mvb/probe_zone_ship.py`, `/tmp/mvb/mvlib.py` (MCP driver), `/tmp/mvb/qa_face/`.

## 5. Bottom line

The certified state is internally honest: every ledgered defect I could re-derive
from the code has either a shipped mechanism with tests, a proven-limit ruling I could
reproduce the geometry of, or an explicit disclosed residual. My independent gate runs
and viewer checks reproduce the certified claims on the exact bytes. The exposure is
almost entirely *forward*: the pipeline's constants and consensus semantics are fitted
to one bright-albedo studio face plus two bright single-photo objects, and the highest-
likelihood next inputs (darker albedo, phone perspective, off-envelope poses, printed
text, speckled/thin materials, any multi-view non-face) each walk through a gap that
no current test, canary, or margin watches. The maintenance contract's first item —
the cumulative-veto hardening (L15) — was still missing when this audit started and
landed in the working tree mid-session (16:41, with test + wiring); the remaining
top exposures are L1 (dark albedo defeats the absolute contrast constants), L6
(mirrored albedo/text), L2/L3 (capture geometry outside the calibrated envelope),
and L4/L5 (zone-gate surrender on detailed or thin geometry). None of these require
new subjects to fix pre-emptively: a dark-albedo + printed-text canary pair and a
relative (foreground-range-normalized) contrast constant would close the two highest
-ranked ones measurably.

## 6. Addendum

- Ship and owl `texture_qa`: **PASS 13/13 each** on my own runs (0 failed gates;
  `/tmp/mvb/qa_ship/results.json`, `/tmp/mvb/qa_owl/results.json`). Together with the
  face run in §0, all three certified harness claims reproduce on the exact bytes.
- **Concurrent-change disclosure**: this audit ran against a live working tree. At
  16:41 (mid-session) `feature_fringe_repair.py` gained the cumulative-baseline veto
  (contract item 1) with call-site wiring, a regression test, and a CHANGELOG entry;
  L15 was re-scored from HIGH to LOW/CLOSED accordingly. No shipped asset bytes
  changed during the session (texture md5s re-verified against the certification
  table at both ends).
