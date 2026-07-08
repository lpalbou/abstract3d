# Knowledge Base

Accumulated critical insights, lessons learned, and validated practices.
Entries here are never deleted; superseded items move to the DEPRECATED
section with an explanation.

## Critical insights

### Coverage-style critiques need per-texel accounting before per-gate surgery (cycle 7)

The owner's "the photos see 57% but the bake paints 21%" critique did not
reproduce on the certified bytes once measured per texel with explicit
definitions: the painted-at-any-weight union was already 45.0% of the
surface (88.8% of the photo-visible union), and "21%" was the
CONFIDENT-weight (>= 0.35) subset. The genuinely surrendered
photo-visible pool was 5.8 points of surface, attributable per gate
(facing thresholds 4.3, layered-zone orphans 0.9, kills/drops ~0.4,
mirror-over-visible 0.5). The permanent fix for this class of dispute is
the leverage LEDGER in the bake stats (per-view potential/painted/won +
per-gate surrender attribution, printed by texture_qa): measure the
inventory first, then attack gates in measured order — the largest
claimed number pointed at the wrong gate.

### Witness-scarcity admission: relax per-texel where nothing else paints, but only with like-material consensus and a feature moat (cycle 7, G1)

"Stretched content beats no content" (the single-view facing doctrine)
generalizes per texel: on surface NO view claims at its strict facing
threshold, a below-threshold witness bounded by the EXACT per-texel
sampling stretch (the projector's own Jacobian, <= 4x) is better than a
symmetry guess or fill — the certified face surrendered 4.8% of its
surface that a photo saw (jaw/cheek silhouette bands, under-chin, crown
transitions). Five measured lessons bound the working mechanism
(`admit_scarce_witnesses`; each violation shipped a detector regression
at 1024): (1) admission must be a strictly LOCAL paint AFTER the global
compositing solve — early admission re-shaded photo-true dark content
20+ px away through the Poisson anchors and flipped three knife-edge
debris detectors; (2) claims need LIKE-MATERIAL consensus support in
BOTH directions — dark-on-bright claims are the flake class
(dark_debris 0.0022 -> 0.0038), bright-on-dark claims are the pale-chip
class (crown flakes 0 -> 0.0022); (3) dark admission requires adjacency
to the confident dark MASS — a nearby dark FEATURE (lip slit) licenses
nothing; (4) NO admission inside a feature moat (0.044 x scale of
strong dark feature cores): parallax-displaced feature adjacency was
the load-bearing debris source, and the moat is free (features are
strictly witnessed by construction); (5) admission before mirror
completion, so symmetry never guesses surface a real witness paints.
Downstream, fill pockets re-partitioned by the changed observed set can
ship unwitnessed dark islands the fill floor's anchor-tracking
exemption correctly keeps — an isolated bright-ringed sub-feature dark
island that is predominantly UNWITNESSED is a defect by definition and
is lifted by the displaced-refill discipline
(`consolidate_unwitnessed_debris`), with the dark split anchored to the
LIGHT material's own median (a bright-half-median split measurably
fused the binding island into a feature-scale component).

### A distance-RATIO field claims surface far beyond its measured support; bound the domain by the mechanism's own absolute scale (FACE-22)

The film-band repaint's S field is u = d_base/(d_base + d_mass) run
through the photos' pooled falloff profile — a RATIO of geodesic
distances. Ratios are scale-free, so ANY surface topologically between
the hair mass and distant skin gets a mid-transition S: measured on the
face proof, neck/chest texels 9-24 pooled transition lengths from the
mass carried S~0.66 and were treated as "hairline apron". The falloff
profile the field interpolates was MEASURED within a few transition
lengths of the dark body; beyond that every derived product (tone
target, envelope, clamp) is extrapolation, and its component borders
print as line-art on smooth skin (the FACE-22 glyph = small envelope
clamp components at 13T). The general rule: a field built from a
normalized ratio needs an ABSOLUTE domain bound in the mechanism's own
measured scale (here: d_mass <= 6 transition lengths, feathered over
one transition). The same ratio-reach failure will reproduce in any
future mechanism that interpolates a locally-measured profile over a
globally-defined ratio field.

### Commits that retone blob interiors print their sub-threshold rims; border treatment belongs to the commit itself (FACE-22)

`commit_trace_deposits` selects candidates by color deviation >= a bar;
a deposit's antialiased border mixtures sit BELOW the bar by
construction (mixture deviation = coverage x deposit deviation), so the
interior retones and the border keeps the old tone — rendered, a closed
line-art outline around every large commit (the FACE-22 az-22.5 chest
contour). The fix is the commit's own border treatment (a one-sided,
distance-decayed rim feather toward the ring-anchor tone, gated by the
commit's own evidence class), NOT post-hoc paint. Two sub-lessons:
(a) the rim's target must interpolate from anchors OUTSIDE the feather
band — a rim mixture bright enough to qualify as a ring anchor
otherwise dominates its own inverse-square target (distance ~0) and
pins its darkness in place (measured: 483 vs 2678 feathered texels);
(b) one-sidedness (only lift darker-than-target) plus the witness gate
makes the feather structurally unable to eat feature edges.

### Completion copies need a tone handoff under gradient-domain compositing; match pure-bright components only (FACE-22)

The legacy compositor's seam leveling included mirror-completion
regions in its per-region offset solve; the gradient-domain path runs
its Poisson solve BEFORE mirror completion, so nothing reconciles a
copy's tone to its destination — on a lighting-asymmetric subject each
copy lands offset (measured +16/255 on the chest) and its border
prints as a contour at the copy's outline. The handoff
(`tone_match_completion_components`) must be scoped to PURE-BRIGHT
copies against BRIGHT destination rings: rescaling a mixed-material
copy re-classifies its own dark micro-content against the shifted
surround and mints dark_debris islands regardless of gain direction
(measured 0.0031-0.0036 vs the 0.003 gate at az-35/-22.5, both
directions) — the FACE-20 lesson (tone edits re-classify neighborhoods
at knife-edge detectors) applied to completion.

### Co-witnessed apron tone disagreements have TWO duals; the identity contract picks the winner

A smooth tone disagreement between the source photo and a reference on
co-witnessed surface has two mirror-image classes, and each one needs
the opposite correction: (a) the SOURCE carries baked light (the FACE-05
nose-ridge specular: source bright lobe, references read darker) — the
composite reconciles the source toward the cross-view diffuse consensus;
(b) a REFERENCE carries baked light where the source's own reading is a
cast shadow (the FACE-04 neck wash: side_right's lit tan won the neck at
w 0.17-0.42 while the source's valid samples read -0.35 log darker
against a -0.08 gauge, because the source's projection weight collapsed
at grazing facing) — the composite carries the source's shading baseline
because the identity contract holds at the source pose (the same
doctrine as the film band's source authority) and the renderer's
flat-biased headlight cannot absorb a real cast shadow into the
compensated gate. The cycle-4 gap analysis hypothesized class (a) for
the neck ("the photo's under-chin shadow baked into albedo"); texel
provenance measured the sign REVERSED — the reference's lit tone was
baked where the gate-pose photo shows shadow. Corollary: never pick the
correction direction from the render's appearance; measure which view's
reading the metric's own registration prescribes at the defect.

### Feature complexes must form in world space, not atlas space

Any morphology whose linking reach is a TEXEL count silently changes its
world-space semantics with texture resolution AND fragments physical
features across UV chart cuts: the 2048 mouth complex formed at r 0.045
vs the 1024 run's 0.11 (its atlas fragments fell below the size floor
BEFORE the world merge could see them), and the chin complex never
formed at all. Clustering the core texels by voxel-graph connected
components in world coordinates (link cell as a ratio of the mesh
diagonal — the rescue detector's construction) makes complex formation
resolution-independent and chart-blind; the measured 2048 stamp set then
matches the 1024 semantics and the chin/mouth stamps bank +0.010
compensated SSIM that atlas morphology structurally could not reach.

### Absolute detectors bind regardless of provenance; photo-truth exemptions need an absolute bound

A structure veto that compares only PRE vs POST (growth budgets) can be
correct about provenance and still ship a battery failure: photo-TRUE
micro content (lash fragments, lip-corner line tips) is real anatomy the
identity gate rewards, but the debris detectors count isolated dark
islands ABSOLUTELY — two views measured 0.0030/0.0032 against the 0.003
gate carrying only photo-confirmed fragments. The working rule: exempt
photo-confirmed sub-feature growth from the relative budget, but bound
the exemption by the battery's own worst pre-repair state (no view may
become the new worst offender), keep feature-size new blobs banned
unconditionally, and consolidate the surviving isolated specks at the
end (render-informed lift to just above the dark class under each
view's own shading — the displaced-refill floor discipline at micro
scale) with texels inside any pre-existing feature-class blob's pixel
footprint protected (atlas geometry cannot separate a profile eye's
under-lash mass from a brow tail; the render battery's own blob
detector can).

### Sequential stamp vetoes must advance their baseline with each acceptance

When a veto battery judges a LADDER of candidate repairs against one
shared pre-repair baseline, every accepted repair's own (legitimately
exempted) content is counted as growth against all later candidates:
one accepted eye stamp turned the stale baseline into a blanket
micro-island veto for every following complex including the mouth.
Re-measure the baseline after each acceptance so each candidate is
judged on ITS OWN structural delta; absolute end-state safety belongs
to the absolute bounds and the final battery, not to baseline
accumulation.

### Texture-level prototypes must A/B on the real export path

A texture patched from captured bake internals (colors_final +
edge-bleed reconstruction) measured -0.010 compensated SSIM for a
change the real export path banks at +0.003: the reconstruction's
global bleed/quantization differences swamp a local edit's signal.
Prototype deltas are only trustworthy when the control arm and the
candidate arm share the exact exported texture bytes path (patch the
bundle's own texture.png, or bake through the pipeline).

### Feature-fringe defects need the gate's own correspondence, rebuilt bit-faithfully

The displaced-content deposits INSIDE protected feature regions
(tear-duct chips, lash dashes, lip-edge dashes) are invisible to
surround-consensus machinery by construction (their rings are
feature-mixed) but fully visible to the identity correspondence: the
photo registered to the render at the declared source pose shows clean
material exactly where the bake shows the chip. Repairing against that
correspondence banks the identity gate directly (measured: comp
identity[front] 0.668 -> 0.708 at 1024) — but ONLY if the in-bake
registration reproduces the harness's construction bit-faithfully. Three
measured basin-flips from small implementation drift: (1) a matted
source image instead of the identity photo (residual dx 0.065 vs 0.025 —
different basin, banked nothing); (2) mean-channel grayscale + bilinear
downsampling instead of BT.601 + area-average (NCC 0.503 vs 0.905 at the
gate's own optimum); (3) symmetric-border closing instead of the gate's
flood-fill-from-corner hole rule (bbox 45 px short, whole map re-based).
Corollary: a repair that targets a metric's correspondence must import
or reproduce that metric's registration EXACTLY; "close" measurably
equals wrong.

### Photo re-registration preserves witnesses but can reshape detector-scale structure; ladder the domain and veto on structure

Re-stamping a feature region with the SAME witness's content under the
gate correspondence is not witness demotion (no witness loses to
another) — but it can reshape detector-scale structure: the re-registered
photo mouth rendered its slit as a compact 45 px dark blob where the old
content was a 101 px elongated one, and the az0 eye detector counted
THREE eyes (the photo itself calibrates clean; the renderer's shading
pushed the soft slit across the dark threshold). The working discipline
(measured across ~10 arms): (a) domain ladder per feature complex — full
re-registration first, trace-witness-only fallback, skip; (b) never
overwrite content a NON-source view confidently won; (c) structure
vetoes relative to the pre-repair state at feature scale, texel-space
first (with the renderer's own shading model — a shading-blind check
measurably missed the third eye), then render-space with the pipeline's
renderer; (d) the veto's blob floor must be the ANATOMICAL feature floor
(0.0009x the foreground bbox): a 22 px speck vetoing a whole-feature
repair inverts the mechanism's purpose, while sub-feature growth belongs
to a micro-island fraction budget (+0.0003 measured knife-edge safe).

### Repair stages that render their own evidence must run on the shipped colors

A repair lane that builds evidence by rendering the CURRENT texture
(and self-vetoes on renders of its candidates) must run after every
stage that repaints texels — detail synthesis and the fill floor placed
after it measurably repainted fill around the repairs into new isolated
micro-islands at six views (debris 0.0017 -> 0.0034), i.e. the vetoes
had judged a texture that never shipped. This inverts the "commit local
repairs late" placement one step further: LAST means after the last
color-touching stage, immediately before export, with repaired texels
marked completion so no downstream statistics re-lift them (nothing runs
after; the mark is for metadata truth and future stages).

### Co-witnessed surface is a zero-sum identity budget; repaint only up to the other contracts' margins

On fused thin-shell geometry (a hairline film apron), the same texels are
imaged first-surface by several photos whose contents disagree under
parallax: each photo's identity gate (rendered at its own declared pose)
demands ITS reading of those texels. Measured on the face proof: the
baseline satisfies the profiles at the front's cost (front SSIM 0.630);
billboarding the front photo onto the apron lifts the front to 0.686-0.719
while the side gates collapse (side worst-window 0.116 -> 0.031, crown and
skin-island detectors flip red); standing off reference-confident and
reference-dominant territory lands at front 0.651 with every other gate
green. No static texture satisfies all three contracts at once — the
metric-space counterfactual ("replace the band with the photo in the
comparison" => 0.73-0.78) is an upper bound no shippable texture can
realize, because it edits only ONE gate's comparison. Treat the front
identity shortfall above the green-frontier as geometry-bound (the same
class as the ear parallax ceiling) unless the mesh stops fusing the apron.

### Statistical tone cannot buy structural similarity; only witnessed content can

Replacing the putty apron with the CORRECT smooth hair-to-skin gradient
field (geodesic profile tone) made identity WORSE (0.630 -> 0.60-0.62):
the gate's SSIM wants the photo's local structure (strands, parting,
wisps), and smooth tone has none, while the putty accidentally correlates.
Only re-projecting the source photo's own sampled content (billboard
authority) moves SSIM up (+0.05). Corollary for all fill-quality work: a
tone-only fix can pass tone-sensitive detectors yet regress identity;
always A/B identity alongside the defect detector.

### Billboarded photo content is pose-local; guard every other pose's read

Stamping a photo's content onto fused-film texels reproduces that photo at
its own pose and prints parallax-displaced copies at every other pose.
Measured failure classes and their guards (film_band_gradient.py): doubled
brow/lid reads as a third eye at az0 (guard: base-material witness veto
inside the feature moat); skin sprayed through curtain interiors at low
texel density (guard: outermost-sheet depth corridor along the source
axis, resolution-independent); dark crease lines tracing the treatment
border (guard: feathered domain edges); isolated treated islands (guard:
final-state graph-connectivity checks, dark against the mass, bright
against the skin ring). Each guard was measured individually load-bearing;
the veto applied OUTSIDE the moat kills half the valid stamps (-0.05
SSIM), so scope matters as much as the rule.

### The witness veto binds by field position, not by feature proximity (FACE-20)

The moat-scoped veto above left a hole: dark stamps OUTSIDE the feature
moat landed unvetoed, and the photo's own curtain-edge/ear-shadow pixels
(content within ~1 transition length of the photo's dark-body boundary)
billboarded onto grazed surface printed hard black stroke/arc artifacts
across five-plus views — each stroke component measured at veto consensus
0.7-1.0 with moat fraction 0.0: the rejection evidence existed at bake
time and was consulted in the wrong region. The general scope rule is the
FIELD position S (the photos' own pooled hair-to-skin falloff): in the
skin half (S >= 0.35) a vetoed dark-stamp component is parallax-displaced
content (the stroke class measured S_med 0.35-0.66); near the mass the
equally vetoed dark stamps are the wisp/strand mass the identity gate
needs (S p50 0.23 — the population whose global veto costs -0.05 SSIM).
Component-level decisions, not texel-level: a stroke is one structure
with one provenance, and fragmenting it leaves speckles. Two corollaries,
both measured: (1) the displaced sites must refill ABOVE the dark class
(floor 1.02x the dark split) with the photo's luminance pattern at
reduced gain — flat guard tone leaves the site paying nothing at the
source-pose gate, verbatim photo tone re-prints the stroke; (2) removing
photo-true dark content COSTS the source-pose identity gate (the stamps
matched the photo exactly there: comp -0.002..-0.004 measured on the C3
tree) while paying the side gates and every other pose — the ruling's
assumption that stroke removal banks front-gate headroom had the sign
wrong; the strokes' cost lands on the 46 non-source poses and on
registration stability, not on the source-pose comparison.

### Repairs that encode photo evidence must be exempt from statistical floors

The fill-luminance floor treats synthesized texels as evidence-free and
lifts them toward donor consensus. Film-band repainted texels carry photo
evidence (source stamps, the photos' measured falloff): flooring them
re-lifted darkened curtain fill into pale shreds that read as skin islands
(measured at 1024). Every repair stage that writes photo-derived content
must flag its texels out of downstream statistical priors — the same
principle as the floor's existing mirror-texel exemption.

### Partial debris cleanup unmasks sub-threshold residue; commit whole neighborhoods or nothing

Detectors that count "isolated dark islands on clean material" (the
dark_debris gate, and human close-zoom reading alike) charge MORE for a
half-cleaned neighborhood than for an untouched one: removing the strong
chips isolates the mid-gray dashes and shadow edges that sat just under
every detection threshold, and on the cleaned surround they read as new
defects. Measured on the face proof (cycle 3): a trace-deposit commit
pass with stable identity and eye counts still lifted dark_debris from
0.0022 to 0.0037 at az0 purely by unmasking — the flagged islands sat
exactly beside the commits. Three gate iterations that tried to catch
the residue by loosening its own thresholds changed nothing measurable;
what worked was inverting the decision: a deposit commits ONLY if every
residue island inside its consensus ring is itself safely sweepable
(small, isolated, feature-clear), and then the whole neighborhood is
retoned in one pass (debris counts identical to control at every gated
view, 4th decimal). The general rule: cleanup mechanisms judged by
isolation-counting metrics must be all-or-nothing per neighborhood.

### Trace witness weight separates displaced-content debris from features; color deviation does not

The chip/dash debris class (displaced lash/lip/strap fragments) and
legitimate small features (moles, nostril rims, lash fringes) overlap in
every color statistic tried across two cycles (cycle-2: flake consensus
deviation p50 0.12-0.26 vs legit front-eye trace texels at 0.399 — the
features would demote FIRST). What separates the populations cleanly on
the same data is the RAW winner witness weight of the blob: chips are
won at w50 0.02-0.29 (grazing/trace claims landing where no confident
view painted), features at w50 0.44-0.93. Corollaries: (1) "never demote
what any view confidently witnesses" is implementable as a blob-level
w50/w90 gate; (2) ball-mean witness maps cannot express this — the ball
mean around a trace chip is lifted by its confident surround (measured:
chin dash ball-weight 0.42 vs its own w50 0.047), so per-texel raw
weights are the signal; (3) BRIGHT trace deposits near confident
strong-contrast cores remain genuinely ambiguous with the feature's own
fringe (sclera flecks, lid highlights) — committing them washed the eye
corner; a confident-core halo is required exactly there, and dark
deposits need no halo because the ring consensus already vetoes them
beside real feature darks.

### Global stages make early texel edits non-local; commit local repairs late

Any texel-level repair inserted before a global solve (screened-Poisson
compositing, rescue-disc detection, closed-loop fill calibration)
perturbs ALL of them: the anchors shift, detectors re-localize, the
calibration rescales every fill texel. Measured on the face proof: a
~4k-texel vacate at the outlier stage changed the whole-face render at
the gate pose by mean 4.1/255 with 14% of pixels above 8/255, flipping
knife-edge detectors (el10 profile eyes, 0.003-level debris margins) far
from any edited texel, and the identity gate swung 0.02+ SSIM between
mechanically equivalent variants through pure NCC-registration
bistability. The same repair applied AFTER the global stages, as a
strictly local recolor, left every unrelated detector bit-stable.
Corollary: repairs that only need LOCAL consistency (deposit retones,
spot fixes) belong after the last global stage; repairs that need global
propagation (tone, seams) belong before it — and a mechanism placed
early must re-verify every knife-edge gate, not just its own targets.

### Fiber-material fill needs combed low-contrast statistics; coarse noise octaves read as rosettes

On dark long-fiber material (hair), the default fill-detail recipe
(multi-octave value noise + anisotropy-scaled LIC) renders leopard
mottle, not material: the coarse octaves are blob-shaped at exactly the
rosette scale, and the closed-loop energy calibration then amplifies
them to meet the gradient target. The measured decomposition (1000 px
renders, face proof rear): raw membrane blotch 4.6, default detail 6.4 —
the pass ADDS the leopard. The working recipe for the strand regime
(anisotropic dark-donor fill): (1) orientation from a multigrid-
propagated global tensor field (donor-local orientation is noise deep in
the domain); (2) fine-octave-only carrier with extended LIC — fine
carriers buy more gradient energy per contrast unit, so the calibration
lands at LOWER visible contrast for the same fill-energy gate; (3) comb
the BASE fill tone along the same field (sparse index-doubling kernel)
so membrane tone washes elongate into strand-parallel streams; (4) scale
transferred amplitude down (elongated low-contrast statistics). Blotch
6.4 -> 5.0 with fill energy 0.93 vs the 0.5 gate. Strand legibility
itself stays out of reach for procedural fill (cycle-2 proven limit);
this recipe buys the MATERIAL read, not content.

### Synthesized-statistic transfers must close the loop on the gate's own metric

Transferring observed micro-texture SIGMA to fill regions (open loop)
systematically undershoots the statistic that actually judges the fill —
linear-luminance gradient energy, fill vs observed. The deficit is a
PRODUCT of independent factors, each innocuous alone (measured, starship
1024 at gain 0.7 = 0.43 vs gate 0.5): donor amplitude transfer 0.84x
(color-similarity weighting picks donors darker/quieter than the observed
median), carrier spectrum 0.69x (a 3-texel finest noise octave carries
less per-sigma gradient than 1-3-texel photo residual), base luminance
0.79x (log-detail on a darker base yields proportionally less linear
gradient). No open-loop parameter fixes all three across assets and
resolutions; the robust design measures the REALIZED fill energy (clip
and seam ramp included) with the same operator the gate uses and solves
one global scale for the target — bounded below by 1 (never dampen an
already-rich fill), above by a hard cap, and by a sigma guard at the
observed band-matched residual sigma so gradient parity is never bought
with granite on edge-dominated subjects (`synthesize_fill_detail`
energy calibration; fresh-bake evidence: ship 0.39->0.58 @1024,
0.50->0.63 @2048; owl 0.43->0.58, 0.60->0.68; dark smears and facet
fields stay 0). General lesson: when a downstream gate measures X, the
synthesis stage must control X itself, not a proxy of X.

### Harness photo references must pass through the same matte as the bake

Any photo-side reference statistic (brightness, seam allowance, detector
calibration, visibility alpha) computed on an UNMATTED photo measures the
background wherever the backdrop is not pure white. Measured on the owl
proof photo (light-gray studio backdrop, ~205 median luminance): the
"non-white" heuristic classified 100% of the frame as foreground, the
brightness reference became 203 instead of the subject's 129, and the
gate failed at 0.567 on bakes whose subject tone was in range — a harness
bug indistinguishable, from the score alone, from a real exposure
regression (it consumed a full cycle lane as OWL-01). The honest
reference is the subject the bake textured: matte RGB photos with the
SAME `remove_background_robust` the pipeline applies before projecting,
keep RGBA alphas, and record the method used (`photo_foreground` in
`scripts/texture_qa.py`); degraded fallbacks must be visible in
results.json, not silent.

### Landmark-detector ratios are face-scale-dependent; never compare across scales

YuNet's mouth-corner localization drifts with the face's PIXEL SIZE: the
same photo, isotropically rescaled, measures eye-to-mouth/interocular
1.05 -> 1.21 (a +11% swing with zero content change; eyes stay put, the
mouth estimate wanders vertically). A verdict metric that compared the
512 px input photo against 1000 px renders therefore reported a "+9%
vertical stretch of the painted face" that was ~9-12 points measurement
bias: forward-mapping the photo's landmarks through the exact projection
chain (recenter affine -> ortho rays -> first surface hit -> renderer
camera) proves the bake paints the photo faithfully (identity to 2.5-6 px
at all five landmarks at the bake pose; mediapipe, which is scale-stable
to +-0.15%, reads the painted face within ~1% of the photo). Any
landmark-ratio gate must use a scale-stable detector (mediapipe FaceMesh)
or strictly matched face-pixel scales. Evidence: /tmp/solve2/m1*,
m1_bias/report.json (scale sweep), m1_identity/ (pixel-level identity).

### Dense reference flow must be strictly local: corrections die outside validated evidence

Non-rigid reference-to-source residuals (nose/mouth/eyes each wanting a
different small shift) can be solved as a capped lattice flow against the
source's painted truth splatted into the reference's frame — but the
correction is only trustworthy where evidence VALIDATES it cell by cell.
Three measured failure modes define the design: (1) extrapolating a
band-fitted affine to the whole photo distorts the reference's exclusive
turf (side identity 0.706 -> 0.587); (2) even a pure translation moves
the hair mass (new skin_in_hair failures at az +-135); (3) leash
exemptions for "flat" cells drag dark hair strands (flat in the gained
reference) across the ear. The working contract: per-cell L1-improvement
validation with an absolute error ceiling, a one-ring leash requiring
substantive non-worsening own evidence, reference-facing weighting, zero
displacement everywhere else, plus injected-known-warp recovery as the
solver gate and full-bake A/B on both harnesses as the acceptance gate.
`abstract3d/reference_flow.py`; A/B evidence under /tmp/solve2/abtip*.

### Baked-in photo shading cancels in overlap ratios; only the DIFFERENCE is identifiable

Two registered photos of the same subject disagree on shared surface
points as a smooth function of the surface NORMAL (each photo's own
lighting), not as a scalar exposure. On overlap texels the log-luminance
ratio cancels albedo EXACTLY: `log Yu - log Yv = B(n).(cu - cv)` with B
the order-2 SH basis in the normal — freckles/hull markings are
high-frequency in normal space and cannot leak into the fit. Three hard
lessons: (1) the lighting component COMMON to all views is mathematically
unobservable from ratios (albedo/shading ambiguity) — pin the gauge to
the SOURCE view (identity contract) instead of pretending to reach flat
light; (2) robust estimation must adapt its outlier threshold to the fit
residual scale (MAD), because a fixed Huber delta treats legitimately
strong shading ratios as outliers and collapses the correction (measured:
fixed delta 0.18 recovered x1.5 of a possible x70 on the synthetic
two-light sphere); (3) a correction justified ON THE OVERLAP must not be
applied unfaded to a reference's EXCLUSIVE territory — the per-view
identity contract (render at that photo's pose matches THAT photo, under
its own light) outranks cross-view consistency there, and an adversarial
bisect measured the unfaded field relighting a profile's whole side
(identity MAE 26.4 -> 39.5). Fade by overlap density over the surface;
gate on measured improvement of the same statistic downstream consumers
use (overlap disagreement) and revert per view otherwise.
`texturing.delight_projections`.

### Localized specular lobes are outside the SH delight span; reconcile them by diffuse consensus in the compositor

The order-2 SH shading fit (`delight_projections`) is GLOBAL in normal
space: a localized specular lobe (the nose-ridge highlight) occupies a
narrow normal cone and the fit cannot represent it — measured on the face
proof, delight reverted (overlap disagreement 0.084 -> 0.085) while the
baked highlight shipped into the texture and rendered as a pale
desaturated COLUMN beside the ridge at az0 (the source pose's +20 deg head
turn projects the ridge highlight onto the nose flank; three cycles of
"compositing seam" hypotheses were wrong — the column was in the pre-solve
blend with zero membrane edges inside it). The identifiable signal is
LOCAL, not global: on texels where a second view validly samples the same
surface, the winner's smooth bright+desaturated deviation from its own
surround (specular signature) combined with the other view reading the
surface darker beyond the pairwise lighting gauge identifies the lobe as
view-dependent light. Rebuild from the winner's OWN surround tone + own
detail — never import the other view's color (misregistered reference
content must not leak through the reconciliation). Where both views read
the surface bright, consensus correctly refuses (shared shine or true
albedo). Feature protection is edge-density, not brightness: sclera and
teeth are bright+desaturated but edge-dense; shading lobes are smooth.
`gradient_compositing.reconcile_specular_lobes`.

### The dark-context chip dual: pale islands commit under dark ring consensus, with an area cap

The bright-context trace-deposit commit has an exact dual: pale
skin/mixture chips displaced into HAIR at trace weight (ear bands,
hairline), plus completion texels that copied those anchors (measured
35-60% of the visible ear-band chip population is FILL, not direct).
The same commit semantics transpose (trace witness, deviation vs dark
ball context, plain-ring multi-witness DARK consensus, isolation from
big BRIGHT components), with one new measured lesson: an AREA CAP is
load-bearing for the pale direction. The bright-context commit's
inverse-square ring retone stays textured on small blobs, but a
700-texel pale blob on the rear committed into a visibly FLAT GRAY WASH
(the anchors average out at that scale) — cap at ~1.2e-3 of direct
texels (the chip class measures 12-60 texels at 1024).
`texturing.commit_pale_chips`.

### Synthetic cut faces take their rim's tone, not the global fill's

A truncated bust's planar cut face is geometry no photo witnessed and no
material statement: the harmonic fill tones it with whatever the mesh
graph connects (tan marble from rear-hair/neck anchors — the FACE-12 disc
wash). The principled tone is the continuation of the cap's OWN RIM
(chest skin at the front arc, hair curtain at the rear), interpolated
across the plane — geometry-aware detection (down-facing planar thin-slab
component with near-zero direct witness), no asset-specific masks.
`texturing.tone_bottom_cap`.

### Facing measures tilt; sampling stretch measures the composed mapping

`facing^2` cannot see the eye-socket failure class: a concave wall can
face the camera acceptably while the composed texel->photo mapping
collapses (one photo pixel smeared along the wall). The exact signal is
the smallest singular value of the texel->photo Jacobian, computed by
finite differences OF THE PROJECTOR'S OWN SAMPLE MAPS (no model
approximation, valid for any camera). Normalize by the median sigma_MIN
of well-facing texels: normalizing by sigma_max mis-scores healthy texels
on legitimately anisotropic UV charts (measured on a cylinder atlas), and
any absolute normalization breaks resolution invariance and silently
re-weights every downstream absolute threshold (conflict priority floors,
mirror source gates). `_tripo_projection_geometry_confidence`.

### Fill texels need a sheet-aware, evidence-respecting luminance floor; witnessed texels are untouchable

The harmonic fill's maximum principle guarantees no NEW extremes but
freely TRANSPORTS observed darkness across hidden surface (measured: fill
blobs at luminance 14-26 whose nearest observed anchors are 61-114); at
close zoom these read as ink smears. The general fix is a floor on FILL
texels relative to (a) the local surface context and (b) the local
OBSERVED DONORS, each with a dark-minority gate so regional darkness
(hair mass, shaded hull, hairline bands) stands the floor down — an
ungated floor turns dark shading bands into pale films. Hard-won scope
rules, all measured: mirror-completed texels are EVIDENCE, not fill (a
local floor cannot tell a mirrored pupil from a defect — leave their
validation to the mirror gates); a connected dark component whose fill
TRACKS its own observed texels' tone (<= 1.35x) is a witnessed feature
continued into hidden surface and must keep its tone, else the floor
manufactures a fresh observed|fill tone seam exactly at the feature's
core (owl wing markings: seam p95 29 -> 52..60), while components whose
fill sits at 2-3.7x their evidence tone are transported smears and get
the full floor; component connectivity must be a WORLD scale, not texel
pitch, or the same marking fragments at 2048 that coheres at 1024. Ball
statistics on thin-crust meshes MUST be sheet-aware — bin by dominant
normal axis (6 bins) and exclude only the OPPOSITE bin: a Euclidean ball
judges a shaded underside against the sunlit topside through the shell,
while hemisphere pooling (Hamming<=1 octants) still fuses front skin
with rear hair. The lift must be saturating PER-PIXEL depth compression
(a base/residual split lets dark bands wider than the base radius hide
their deficit in the residual; compactness restrictions and boundary
feathers leave pocket edges under the floor which re-detect as fresh
smaller fragments); near-black pixels must blend toward a consensus
COLOR rather than be multiplied (no usable chroma, amplified
quantization noise). `texturing.enforce_fill_luminance_floor`: starship
4 -> 0 and owl 6 -> 0 close-zoom dark fragments (shipped bundles,
post-process) and zero fragments on all fresh 1024/2048 bakes, with fill
energy, seams, and edge preservation intact.

### Tone seams are a color-domain artifact; composite gradients instead

Exposure/white-balance disagreement between views is locally a constant
error field, and every color-domain remedy (blend weights, per-region
offset solves) must guess where the constant changes hands. Constants
vanish under the gradient operator: compositing per-view GRADIENTS
(most confident common witness per texel-graph edge) and solving one
screened Poisson system `(L + Λ)x = Λc + div g` over the texel surface
graph removes the step for free while preserving witnessed edges
verbatim — proven exact (up to one global constant) for additive
corruption on synthetic ground truth, and measured on the face lane
(chroma-seam max 0.50 -> 0.31, texture_qa seam p95 29.0 -> 22.3).
Two design decisions carry the result: screening must be proportional
to blend confidence (photos stay ground truth where they saw well;
`1/sqrt(lambda)` sets the equalization decay and must be rescaled by
resolution^-2 to stay fixed in world units), and completion regions
(mirror/fill) must contribute the gradients of their own completed
content INCLUDING at their borders — a zero-gradient membrane at those
borders flattens fragmented completion islands (measured: ear folds
washed out, side-profile identity SSIM 0.703 -> 0.606).
`gradient_compositing.py`, `compositing="gradient_domain"`.

### Screened surface-Laplacian solves need a multigrid preconditioner

The screened Poisson operator on a texel graph couples texel scale to
the `1/sqrt(lambda)` equalization scale (tens of texels), so plain
Jacobi-CG needs ~1000 iterations at 1024 and diverges from any runtime
budget at 2048. Geometric-aggregation multigrid (voxel-cluster the
texels' 3D positions, Galerkin `P^T A P`, damped-Jacobi V(1,1),
coarsest-level splu) used as a CG preconditioner converges the same
systems in 20-90 iterations. Implementation lessons that mattered:
precompute `P.T` as CSR (scipy converts on every application otherwise;
2.6x V-cycle cost), run float32 when screening dominates f32 rounding
of the diagonal (2x SpMV throughput; keep CG scalars in float64), and
solve all 3 color channels as one (N, 3) block against the shared
matrix.

### Exported material factors multiply the baked texture — they must be identity

A baked albedo atlas is only rendered as authored if every factor a
spec viewer multiplies on top of it is identity. trimesh's default
`SimpleMaterial` diffuse is 0.4 gray, and its GLB conversion
(`to_pbr()`) copies that into `baseColorFactor` while omitting
`metallicFactor` — and the glTF 2.0 default for an absent
`metallicFactor` is 1.0, fully metallic. The combination renders
textured exports ~60% darker, desaturated, and mirror-dark under
image-based lighting in any spec-compliant viewer (verified by parsing
the GLB JSON chunk: factor 0.4 + absent metallic on shipped bundles).
Textured meshes must carry an explicit `PBRMaterial` (white base color,
`metallicFactor=0.0`, `roughnessFactor=1.0`), and the OBJ/MTL sidecar
needs the same treatment separately: trimesh's `PBRMaterial.to_simple()`
only carries the diffuse factor, leaving `Ka`/`Ks` at the 0.4 default.
`Ks` must be 0 for baked albedo — Phong viewers evaluate
`Ks * (R·V)^Ns`, so `Ks 1.0` with a low `Ns` adds up to +100% white and
washes the asset out just as badly as the darkening it replaces.
`scripts/check_export_materials.py --strict` gates both formats.

### Preview renderers must apply material factors or they mask export defects

The in-repo preview shader sampled the raw texture and ignored the
material's base color factor, so previews looked correct while every
external viewer showed the 0.4-darkened asset — the defect shipped
because the only renderer anyone checked was the one renderer that
could not see it. Preview paths must apply the same factor arithmetic
as spec viewers (`texel * baseColorFactor`); previews may look worse
than reality, never better.

### Object canonicalization is not camera pose

Generative reconstruction models (Hunyuan family) canonicalize the
SUBJECT — its symmetry plane lands on the world axes — not the input
camera. A "front" conditioning photo of a subject whose head/body is
turned sits 15-25 degrees away from the canonical front. Assuming the
photo's camera is at azimuth 0 displaces every projected feature
laterally and produces doubled features at intermediate views. The photo
pose must be estimated (five independent methods agreed on the checked
face: YuNet nose-offset matching, mediapipe facial transform, two
render-sweep audits, signed-gradient correlation). ADR 0008.

### The recenter's frame and the projector's frame are different conventions; register to the projector

The canonical recenter centers the photo's ALPHA-BBOX at the frame
center; the orthographic projector centers the WORLD ORIGIN in its
sample map. The two agree only where the mesh's camera-plane bbox center
projects onto the camera axis — guaranteed at the canonical front (the
model itself recentered the conditioning image it reconstructed from),
false in general at any other source pose. Measured offsets: starship
az+30/el+15 = 54 px at 1024 (the SHIP-03 "nose melt": dark off-surface
content dragged onto the prow — src-pose fidelity MAE 45.5 -> 18.1 and
SSIM 0.092 -> 0.600 with the correction), face az+20/el+8 = 18 px, owl
az0 = 1 px. The correction (`projected_frame_center_px`) is a
deterministic function of mesh and pose — no content-based search — and
reduces to the plain recenter exactly where the old assumption held. A
synthetic-checker probe at the same witness geometry separates the two
failure families cleanly: registration error decorrelates content at ALL
stretch levels (binary agreement ~ chance), while true grazing-stretch
loss is gradual (0.72 at stretch 1.25-1.5, 0.71 at 3-4 after the fix).
Any harness that reconstructs per-view visibility from the input photo
must recenter with the SAME offset (recorded in bundle metadata as
`source_registration`), or region attribution shifts by exactly that
many pixels and every region-conditioned statistic drifts.

Scope boundary (measured, load-bearing): the correction applies to
OVERRIDDEN poses — external capture facts the model never consumed. For
ESTIMATED poses the estimator searched az/el to best align the
LEGACY-centered photo's gradients, so its pose and the legacy frame are
co-adapted; correcting the frame under the estimated pose breaks the
pair (face proof: verdict1 failures 2 -> 10, front SSIM 0.630 -> 0.598,
doubled features at az0 — the projection landed 18 px off the pose the
estimator had compensated for). Registering the ESTIMATOR itself to the
projector frame (estimate against projector-frame renders) would make
the correction universal; until then the correction and the estimator
must not be mixed.

### Grazing-smeared donors carry artificially quiet statistics; floor transferred amplitude

Statistics transfer (fill detail) inherits whatever the donors measured.
Donors imaged at extreme foreshortening are doubly damaged: their
CONTENT is smeared along the surface AND their residual-amplitude
statistic is erased by the same smear, so fill anchored by them ships as
a literal flat plateau bounded by straight chart edges (measured: an
11k-texel flat cell at the starship's under-hull, texel facet_cellular
0.092 vs the 0.091 allowance at 2048). The general rule: no fill region
may claim to be quieter than the observed population's own low quantile
— floor the transferred amplitude at the p25 RAW-residual amplitude (the
Gaussian-smoothed amplitude field spreads sparse line energy over flat
texels and would inject line-level noise; the raw quantile tracks the
flat majority's true stochastic level). The closed-loop energy
calibration and sigma guard above the floor keep the global level
honest (granite test unchanged).

### Signed gradient vectors, not magnitudes, for pose scoring on symmetric subjects

Gradient-magnitude correlation peaks equally at the true pose and its
mirror on bilaterally symmetric subjects (faces): the first integration
of the pose estimator picked -25 degrees for a +17-degree photo and made
the bake worse. Correlating the signed (gx, gy) vector field breaks the
tie — the mirror pose anti-correlates horizontally. Silhouette-edge
gradients must also be down-weighted (interior distance transform):
outlines are pose-insensitive on heads and swamp the interior signal.

### Register interior content to painted truth, not outlines to outlines

Silhouette-based photo registration aligns OUTLINES. On subjects whose
outline is dominated by one material (hair), interior features can stay
displaced by several percent of the frame while the outline fits
perfectly (measured: 58 px nose error at 1024; the profile's eye painted
on the temple). Once the source view has projected, its texels are
ground-truth colors at known surface points: minimizing photometric
disagreement on the mutual overlap registers the reference's interior
content directly. This single change cut adversarial QA failures 62 -> 14.

### Witness reliability is a region property, not a pixel property

Where thin film shells hover over a surface (generated hair wisps over a
scalp), each photo pixel's assignment between the sheets is decided by
sub-pixel aim, and the pixel colors are themselves material mixtures.
Gating individual "layered" pixels fails — the un-gated survivors between
them still anchor flakes, and fill inherits mixed anchors. The reliable
unit is the REGION: where the density of layered samples over a window
exceeds a threshold, the view must surrender the whole region. Use
per-sample fractions (resolution-invariant), not per-bin counts (sharpens
with texel density).

### Fused film bands need material commitment, not more geometric gating

The layered-zone gate detects hovering film shells by their second sheet.
Where the mesh FUSES the film into the head (single surface), no
geometric statistic can see it: layered density is ~0.02-0.05, plane-
removed relief and normal dispersion do not separate the band from eyes
or foreheads (measured — the whole upper head is rippled), and depth-gap
spectra show the fused hairline has literally zero second sheets. The
photo does carry the signal: the band sits on the dark-material main
body (scale-free two-mode luminance split), at high contrast, with high
dark coverage of the window's foreground. Growing the strong zone into
that evidence (hysteresis) finds the fused band; what to do with it is a
MATERIAL decision (commit to the film's tone) — not a further weighting
tweak.

### Material commitment requires multi-view consensus; parallax exposes single-pose consistency

Committing a texel to a material tone is only safe when every view that
images it first-surface agrees it is film (flag consensus), no view
positively witnesses base material along its ray (df < 0.25 at an
un-flagged imaged bin), at least two views image it (single-witness
consensus is vacuous), and the local observed context is dark-dominated.
Each condition was measured load-bearing on the face lane: a fused wisp
floater aligns with the dark hair from the front pose ONLY — committing
it painted a floating dark dash over the eyelid (the "third-eye" class);
single-witness commits painted dark spots on ear-rim/crown skin; and
committing against surviving bright context flipped coherent pale
ribbons into flake-island contrast (az-135 crown 0.0006 -> 0.0027).
Conversely, surrender WITHOUT commitment is also unsafe: vacating claims
where nothing dark takes over leaves the membrane anchored by whatever
survives nearby (a dark lash bled through a vacated eyelid rim as a
dash). Surrender and commitment must be one coupled decision.

### Completion stages propagate bad anchors

Mirror-symmetry and harmonic-diffusion completion COPY observed content.
Provenance tracing showed >90% of hairline flake islands were
mirror/harmonic copies of a few low-confidence mixture anchors — the
defect multiplies through completion, so source gating (confidence floor
+ contested-band exclusion) is worth more than post-hoc filtering.
Removing completion entirely is worse (hidden regions degrade to fill
mush): gate, don't amputate.

### Consensus filters must exclude self-votes

A neighborhood-consensus outlier filter where each element votes in its
own consensus is mathematically a no-op for its designed target: the
diagonal of A@A equals vertex degree, so any island at least one ring
wide dominates its own histogram and dilutes the deviation below
threshold. Zero the diagonal and binarize reach before voting.

### Tolerance constants interact with geometry slope

A scalar depth-epsilon in a z-buffer visibility test cannot serve both
front-on and tilted surfaces: tight enough to block thin-shell ghosting,
it makes smooth tilted surfaces occlude themselves (measured 40% of
visible texels wrongly rejected at 55-75 degrees, all demoted to milky
fill). Slope-aware bias (epsilon grows with tan(tilt) x pixel-world-size,
standard shadow-mapping practice) resolves the conflict.

### Morphological dilation biases geometric fitting

Any mask comparison where one side is dilated and the other is not biases
the fit by the dilation radius (measured: +4% scale on every registered
view from a 1.3 px splat dilation). Either dilate both sides or erode
back after closing.

### Evidence standards: thumbnails hide blocking defects

Two shipped "proofs" were rejected because quality claims rested on
420 px thumbnails and aggregate metrics (coverage ratios). Defect classes
that dominated user perception — doubled features, flake bands, milky
patches — are invisible below ~800 px per view. Every quality claim now
requires the adversarial QA harness (full-resolution renders across the
azimuth/elevation grid, per-view defect detectors calibrated so the
reference photos themselves pass, pose-aware identity gates) plus
full-resolution crop review of failures. Coverage is an accounting
number, not a quality metric: a 0.57 -> 0.40 drop accompanied a large
visual improvement because surrendered mixture bands render as clean fill
instead of flakes.

### Adversarial verification requires calibrated hostility

Self-declared success is worthless (two epic failures proved it), but
blanket negativity is equally useless. The working pattern: dedicated
verdict agents that (a) build repeatable, gate-based harnesses a fix
cannot game, (b) calibrate gates so known-good content passes (the
reference photos themselves), and (c) audit their own blind spots when
results contradict independent measurements (the az-0 identity gate
penalized the pose fix until measured at the photo's true viewpoint).
Fixer claims must carry instrumented before/after numbers on the shared
harness, and conflicting agent claims are settled by A/B on the same
stack — never by authority.

### Baked opaque textures cannot represent semi-transparent boundaries

At wispy material boundaries (hair over skin) the true appearance is a
semi-transparent mixture. A baked opaque texture must commit each texel
to one material; the best achievable is a clean commitment to the locally
dominant material (rendering as a slightly thinned hairline), not
faithful wispiness. Faithful reproduction requires alpha-carrying shell
geometry or generative hair-aware inpainting — outside projection-bake
scope. Known, documented limit (ADR 0008).

### Film-shell hair geometry is model-inherent and load-bearing

Hunyuan-2mv grows thin film shells (0.005-0.03 x diagonal) as the OUTER
hair surface near the hairline. They cannot be deleted: conservative
below-hairline flap removal is safe (face debris 4.5% -> 0%), but
removing the brow-band films opens ragged holes — they ARE the visible
surface. Texture logic must treat them as occluders and ambiguous
witnesses (layered-density gate), not try to remove them. Higher
inference steps, higher octree resolution, and dropping the mirrored
reference all failed to reduce them (statistical twins of the baseline).

### Vertex-resolution fill renders as polygon facets at texel resolution

Any fill that assigns texel colors from a coarser proxy re-introduces the
proxy's resolution as a visible artifact: nearest-vertex assignment after
the vertex-domain harmonic solve painted each vertex's Voronoi cell as one
flat polygon (~59k vertices serving ~4.2M texels at 2048 — 70 texels per
cell), and the KD fill's per-texel donor sets change abruptly between
neighbors (patchwork mottle). The fix is two-stage: interpolate the proxy
(IDW over nearest vertices) AND relax the fill at texel resolution over
the surface's own adjacency (KD graph of texel 3D positions, observed
texels as Dirichlet anchors). UV-space smoothing is NOT a substitute —
xatlas packs unrelated charts side by side, so image-space filters bleed
across charts (same reason the fill itself works in 3D).

### Fill detail: transfer texture statistics, never texture structure

Propagated fill (harmonic/KD) yields the correct average color with zero
micro-texture, which users read as "painted mush" next to observed
content. Two remedies measured on the proof assets: (a) copying observed
high-pass STRUCTURE (patch/shift-map quilting) — rejected, produces
chaotic misplaced fragments and level-set banding at any parameterization
tried; (b) transferring texture STATISTICS — robust local amplitude (L1,
not RMS: sparse panel-line edges inflate RMS and render as granite) and
structure-tensor streak orientation, carried by deterministic 3D value
noise with line-integral-convolution streaking — adopted, fill/observed
variance ratio 0.53 -> 0.83 (face) and 0.15 -> 0.59 (ship). Statistics
transfer makes hidden surface read as the same MATERIAL; it cannot invent
the specific content (a hairline whorl, a particular panel layout) — that
would require generative inpainting, out of projection-bake scope.

### Single-photo bakes of symmetric objects should mirror by default

A single photo observes a thin sliver of a closed object (starship: 6-9%
of texels; the shipped bundle's 24% came from an older, wrongly-posed
perspective bake). For geometry whose mirror-symmetry score passes the
standard gate (ship: 0.98), the mirrored twin of that sliver is REAL
content everywhere the fill would otherwise wash. `texture_completion=
"auto"` resolves to mirror completion through the same score gate that
protects explicit requests; the Hunyuan backend defaults to it. The
asymmetric-object risk is unchanged: the gate rejects meshes below 0.55.

### Multi-view tone seams are region phenomena; level them on the mesh graph

After global gain harmonization (gains 1.01-1.04) the face's mid-face
vertical seam remained: the residual is CONTENT-level (the photo's
baked-in directional shading vs evenly-lit synthetic profiles), localized
exactly at composition-region borders (per-texel winning view / mirror
fill). Screen-space provenance tracing measured the pale zone left of the
nose as winner-switching at weak weights (q50 0.13 vs 0.73 on the photo
side). The fix that measured best is per-region low-frequency offset
fields solved on the mesh graph (`level_composed_seams`, Ivanov/Lempitsky
seam leveling): boundary terms demand cross-region agreement, smoothness
keeps fields low-frequency, and two safeguards are load-bearing —
(1) boundary edges whose step exceeds ~0.18 mean|RGB| are real material
borders (hair|skin) and must NOT be leveled (an uncapped solve tinted the
ear region: right-profile worst-window SSIM +0.10 -> -0.13), and
(2) confident-witness vertices are pinned to zero correction so leveling
only bridges the weak bands BETWEEN confident zones. Alternatives measured
and rejected on this defect: per-texel winner-agreement attenuation (no
effect where only one view covers), lowering the conflict threshold to
0.18 (zeroed legitimate front content, mirror bands appeared), demoting
sub-0.05-weight evidence to fill (pale mirror bands replaced plausible
weak content), and a photo-gradient x stretch "smear gate" in the
projector (indistinguishable from the REAL eye at the same obliquity —
destroyed it).

### Mirror twins can cross material boundaries; guard with local consensus

Geometry symmetry 0.966 is not 1.0: near the hairline a twin lookup lands
just across the hair|skin border and copies hair onto cheek skin (half of
the dark close-zoom defect pixels on the face's left cheek were such
copies). Rejecting a copy ONLY when the destination's observed 3D
neighborhood is color-consistent (spread <= 0.09) AND the copy contradicts
it (deviation > 0.22) removes those without touching legitimate feature
completion — feature-rich neighborhoods (eyes/lips) have high spread and
accept everything (starship: guard on/off bit-identical).

### Export material factors are part of texture correctness

A baked albedo texture is only as correct as the material factors it
ships with: trimesh's default SimpleMaterial exported
`baseColorFactor 0.4` with no `metallicFactor` (glTF defaults to fully
metallic), so every spec-compliant viewer rendered assets ~60% darker
as tinted metal while the repo's factor-ignoring preview renderer
masked it completely. Two rules follow: exports must carry explicit
factors (white base, metallic 0, roughness ~1; MTL Kd 1.0, Ks 0.0), and
preview renderers must apply the same factors a real viewer applies —
a preview that can look better than reality is an overclaiming machine.

### Mirror poses are near-ties on symmetric subjects; break chirality explicitly

Gradient-magnitude content of a bilaterally symmetric subject is almost
identical at +az and -az, so any full-field correlation decides the
azimuth SIGN by a jitter-fragile margin (0.1% vertex noise flipped it).
The horizontal anti-symmetric luminance component isolates exactly the
chirality carriers and is sign-opposite between mirror poses —
correlating it decides the sign robustly. General pattern: when a
discrete symmetry makes candidates near-degenerate, score the
symmetry-odd component separately instead of hoping the full objective
resolves it.

### Estimator scorers must compare in a common frame

A correlation between a photo and candidate renders is only meaningful
after both are reduced to the same frame. Full-bbox recentering fails
on cropped photos (bbox scale disagrees between a chest-cropped
portrait and a full-bust render); no alignment fails on elongated
subjects whose projected aspect swings with pose. Crop-immune anchors —
subject top, silhouette centroid, mean width over commonly observed
rows — survive both. The same signature works for registration
(width-profile matching) and pose scoring.

### Fill must reproduce material statistics, not invent content

Propagated fill (harmonic/mirror/KD) yields the correct average color
and zero micro-texture — the "painted mush" a viewer notices instantly.
Transferring texture STATISTICS (residual amplitude + streak
orientation, carried by procedural noise) makes hidden surface read as
the same material; transferring literal patches (quilting) was measured
worse (chaotic fragments, banding). The honest ceiling: synthesis
renders material, never specific content (no invented panel layouts or
hair whorls) — and texel-resolution fill needs texel-domain smoothing,
because vertex-domain solutions render as per-vertex Voronoi facets at
close zoom.

### Observed-but-badly-witnessed regions need mirror arbitration, not fill

`mirror_fill_from_observed` writes only UNOBSERVED texels, so a region
every view covers at grazing incidence (or that a misregistered mirrored
duplicate reference paints wrongly) keeps its bad content: the right eye
at az -90 measured blend weight 0.14 vs its mirror twin's 0.50, with the
iris band displaced ~0.04 units — observed, wrong, and unreachable by any
unobserved-only completion. Rescue = transplant the strong twin's disc,
tone-matched to the destination annulus (composition legitimately shades
the two sides differently, so a raw copy reads as a pasted patch). Two
measured failure modes bound the design: per-texel confidence-ratio
demotion (blanket, no disagreement gate) demoted 21k texels and destroyed
valid front-view content; and the geometric aperture was NOT the limit —
a synthetic max-contrast almond on the actual lid band passes the eye
detector at 896 px, so squinted-lid geometry (aperture 0.20-0.29 vs photo
0.43) still supports a readable profile eye when content lands correctly.

### Weak-twin transplant detection: witness asymmetry + dark core + feature-empty twin

Generalizing the eye rescue above into the bake required gates measured
to separate "smeared twin of a real feature" from everything else on
real content (face/starship/owl proofs). The working per-texel basis is
voxel-ball statistics over DIRECT observed texels of surface-smoothed
luminance and RAW winner weights — the blend's feathered weights zero a
6-texel rim band and drag ball means below every calibrated threshold
(measured: the identical detector fires with raw weights and stays
silent with feathered ones). The gate stack, each with a no-fire test:
(1) strong side confidently witnessed (ball weight >= 0.35) and locally
contrastful (ball luminance std >= 0.05); (2) twin observed at >= 0.25
of the strong side's witness density — unobserved twins belong to mirror
COMPLETION; (3) twin ball weight <= 0.5x the strong side's — content
well-witnessed on both sides is legitimate asymmetry, never touched;
(4) a coherent DARK core (blob response <= -0.12 over >= 3e-4 of direct
texels) — transplanting bright speculars fragments neighboring features
(the v14-era az -45 trade); (5) the twin's own blob response at the
mirrored core <= 0.5x the core's, sampled POINTWISE at nearest direct
texels — ball averages dilute any blob wider than the ball to ~zero and
blind the gate to real twin structure; (6) the disc must lie strictly on
one side of the symmetry plane — a feature straddling the plane has no
clean twin side, and a half-transplant guarantees a mid-feature seam
(measured: a black dash + flake fringe on the front-view lips). The
identity-registration side effect is load-bearing for verification: a
broken high-contrast feature drags global NCC photo registration off by
~1%, and at the wrong alignment far-away high-contrast regions (ear/hair
boundaries) anti-correlate — a worst-window "ghost" can be pure
registration artifact of a defect elsewhere (side_right -0.132 at the
ear scored +0.47 under the corrected alignment with NO ear change).
Placement and tone must respect in-bake reality: (a) the geometric
mirror position is systematically off ALONG THE MIRROR AXIS (mesh
asymmetry + per-side registration), and the repaired feature then
becomes a second NCC attractor that pulls the SOURCE-pose registration
into a bistable flip (optima 1e-4 apart in NCC, 0.03 apart in SSIM);
anchoring the transplant on the twin's own evidence-weighted dark
centroid fixes it, but ONLY the axis component is signal — in-plane
components flipped signs across estimators and re-rolled the flip.
(b) Partial-keep transplants (excluding source-witnessed texels,
confidence ramps, dark-aware keeps, normal cuts) all fragment the
feature into doubled blobs at intermediate views — transplant whole
discs or not at all. (c) In-bake tone rings contain not-yet-filled
texels; average the ring over content-bearing (source-mask) texels or
the zeros bias the transplant dark.

### Identity gates that compare shaded renders to photos carry a perfect-texture floor

The preview renderer multiplies the baked albedo by its own shading field
(`shade = 0.88 + 0.12*diffuse`, world-fixed light). Any photo-vs-render
identity metric therefore penalizes even a PERFECT texture by
`mean(photo * (1 - shade))` — measured on the face lane at the declared
pose: shade p50 0.906 over the compared mask, perfect-texture score
SSIM 0.977 / mean|RGB| 11.45 (face hull 15.2), i.e. half of a 22.0 MAE
budget is consumed before the first texture defect is counted. The term
is texture-independent (multiplicative in the shader), so it shifts every
candidate equally: it is measurement bias, not albedo signal. Correction
belongs IN THE MEASUREMENT, never in the texture (brightening albedo to
"pass" an uncompensated gate ships a wrong texture to every other
viewer): render the same mesh with a white texture at the same pose (the
exact shade field through the identical resampling chain) and multiply
the registered photo by it before scoring, then re-tighten the MAE budget
by the removed floor. Measured populations keep their ordering under the
compensation (deltas < 0.005 SSIM) while systematic bias (-11 to -15
luminance on the face hull) cancels to ~+1.

### Bit-mirrored reference pairs should share one registration

When a reference photo is the horizontal mirror of another (the standard
stand-in for an unphotographed side), registering each side independently
lets estimator noise land them asymmetrically: the overlap-alignment stage
solved shift_y +0.08 for the genuine left profile but +0.04 for its
mirrored twin on the v14-era tip — a 0.04-unit relative displacement that
smeared the right eye into two thin fragments (the az -90 eye_count
failure). The mirror of the twin's registered result is better conditioned
than an independent solve because the weak side's overlap anchor data is
exactly the thin/stretched coverage that made it weak. (On the current tip
the width-profile rework happens to register both sides symmetrically, so
a unified-registration patch measured neutral — the guarantee matters
whenever the estimators disagree again.)

## Validated practices

- Prove bugs with minimal numerical ground-truth tests before fixing;
  keep the test as a regression harness.
- Bisect visual regressions by rebaking saved intermediate states before
  touching code; distinguish code effects from input (mesh) effects —
  the "terrible" jump attributed to code changes was ~90% a mesh swap.
- When two agents disagree (pose +4 vs +17), run the decisive A/B both
  claims share, on the current tip, with the shared harness.
- Scope aggressive thresholds to the path that measured them (the 0.4
  source facing threshold applies only to ortho multi-view; single-view
  bakes keep 0.2 because stretched content beats no content).
- Version only the validation experiments that reflect the current state
  of the code (certified bundles, the certification record, and active
  feature proofs); superseded experiment archives stay local. Rationale:
  the repository stays reviewable (~100 MB of proofs instead of ~2 GB),
  and stale evidence cannot be mistaken for current behavior. Enforced by
  the allowlist in `.gitignore`; user-facing docs may only link versioned
  paths (`docs/assets/` carries curated copies where needed).

## DEPRECATED

### Region-IoU silhouette registration (superseded)

Region IoU rewarded degenerate blow-ups on differently-cropped photos
(maximizing in-frame overlap). Replaced by edge-chamfer scoring, then by
width-profile + overlap-photometric registration (ADR 0008). Kept here
because the failure mode (objective rewards degenerate warps under
partial observation) recurs in registration design.

### Photometric NCC warp refinement, first formulation (superseded)

`refine_registration_photometric` recovered 0 of 15 injected shifts and
proposed nearly the same warp regardless of the true offset (constant
attractor). Removed from the default path. The lesson — validate any
estimator against injected known offsets before trusting it — directly
shaped the acceptance gates of `estimate_pose_photometric` and
`register_reference_by_source_overlap`, both of which pass
injected-offset recovery.

### Hard-pinned source pose in canonical frames (superseded by ADR 0008)

The original ortho-mode assumption "registration is deterministic, the
photo IS the canonical front" was disproved (see Critical insights). The
deterministic FRAMING remains valid; the POSE is estimated.
