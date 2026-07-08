# ADR 0008: Photometric Source Pose, Overlap Registration, and Witness Gating

Date: 2026-07-05
Status: Accepted

## Context

The first end-to-end multi-view face bundle produced under ADR 0007 was
rejected in review: three-quarter renders showed a second set of facial
features displaced onto the cheek, an ear-like patch mid-face, black flake
debris along the hairline, and milky fill patches. A six-agent adversarial
cycle (two results auditors, four error hunters) traced the failure to
five independent causes, each proven with instrumented rebakes before any
fix shipped:

1. The canonical-frame path assumed the conditioning photo is taken from
   the canonical front (azimuth 0). That assumption confuses OBJECT
   canonicalization with CAMERA pose: multi-view-conditioned reconstruction
   canonicalizes the subject's symmetry plane onto the world axes, so a
   photo of a subject whose head is turned sits 15-25 degrees away from
   the canonical front. Five independent measurements (YuNet nose-offset
   matching, mediapipe facial transform calibrated on known-azimuth
   renders, two verdict-agent render sweeps, gradient correlation) put the
   checked photo at azimuth +15..20, elevation ~+8. Projecting it at 0
   displaced every feature laterally — the dominant cause of the doubled
   face.
2. Silhouette registration aligns OUTLINES. On a head the outline is the
   hair contour, and aligning it leaves interior features (eye, nose,
   mouth) displaced by several percent of the frame (measured 58 px nose
   error at 1024): the profile photo painted its eye on the temple.
3. Thin film-shell hair geometry (sheets hovering 0.01-0.09 of the mesh
   diagonal over the scalp — the outer hair surface itself, not removable
   without holes) makes hairline photo pixels ambiguous witnesses: within
   the shell band, sub-pixel aim decides which sheet a pixel stamps, and
   the pixels are themselves hair-over-skin mixtures. Stamping the band
   produces salt-and-pepper mottle; mirror and harmonic completion then
   propagate the bad anchors (measured: >90% of detected flake islands
   were propagated copies, not direct projections).
4. Four math defects, each with a numerically verified fix: the outlier
   filter's 2-hop consensus let texels vote for themselves (a planted
   foreign island was never dropped); the splat silhouette's dilation
   biased every registration toward +4% scale; the strict z-buffer's
   scalar epsilon self-rejected up to 40% of visible texels on tilted
   surfaces (demoting them to milky fill); horizontal registration shifts
   converted back through width instead of height on non-square photos.
5. The evidence standard itself failed: 420 px thumbnails and coverage
   metrics declared success on renders whose defects were only visible at
   full resolution.

## Decision

- **Photometric source-pose estimation** (`estimate_pose_photometric`):
  the ortho bake estimates the source camera's pose by correlating the
  photo's signed gradient VECTOR field against untextured renders over a
  +/-40 degree azimuth grid. Two properties are load-bearing, both
  established against ground truth: signed vectors (gradient magnitude is
  bilaterally symmetric on faces, so a magnitude scorer cannot tell a pose
  from its mirror), and interior-distance weighting (silhouette-edge
  gradients are pose-insensitive on heads and swamp the interior signal).
  The declared pose wins unless a candidate beats it by a real margin, and
  an anti-correlated argmax is rejected outright; on a genuinely frontal
  input the estimator returns "not estimated".
- **Overlap-photometric reference registration**
  (`register_reference_by_source_overlap`): after the source view
  projects, each reference photo is aligned by minimizing source-weighted
  RGB disagreement on mutually observed texels — registering interior
  content to the source's painted truth rather than outlines to outlines.
  Silhouette methods remain as the coarse initializer (crop-immune
  width-profile matching plus a small residual search).
- **Layered-density witness gating** (projector `layered_zone_gate`):
  regions where more than 10% of a photo neighborhood's projected samples
  land a small gap (3 epsilon to 0.03 x diagonal) behind the first surface
  image stacked film shells; the view surrenders the whole region instead
  of stamping mixture content texel-by-texel. The statistic is a
  per-sample fraction (resolution-invariant) over a window of 3% of the
  photo's minimum dimension. Pixel-level gating was measured and rejected:
  the un-gated survivors between layered pixels still anchor flakes.
  Amendment (same date): surrender additionally requires local photo
  CONTENT CONTRAST (window luminance std above ~0.055). Ambiguity only
  matters when the mixed materials differ — layered hair-over-hair
  regions stamp the same material whichever sheet a pixel hits, and
  surrendering them removed real texture (a regression the final audit
  caught). Measured effect of the condition: the beige crown
  discoloration patches disappear (crown-flake checks 4 -> 0), the rear
  QUARTERS keep their profile-painted sheet, and the right-profile ear
  etch clears (the local-window identity gate passes for the first
  time). The CENTRAL rear remains smooth fill either way — no view
  witnesses it (the profiles see it edge-on), so it is a capture-set
  limit, not a gate tuning issue: "rear discoloration removed" is
  supportable, "rear texture restored" is not. Cost: mild temple mottle
  returns at the zone boundary and one painted curl reads as a spurious
  dark blob at the front view.
- **Gated mirror completion**: mirror sources must be confident
  (blend weight >= 0.35) and outside every view's dilated contested band.
  Disabling mirror completion entirely was measured worse (the far cheek
  degrades to harmonic mush); ungated it propagates hairline mixture
  anchors into flakes everywhere.
- **Per-role facing threshold** (ortho multi-view only): the source stops
  painting beyond ~66 degrees off-axis (threshold 0.4) where reference
  photos exist; single-view and perspective bakes keep the wide threshold
  because stretched content beats no content when nothing else covers.
- **Slope-aware z-buffer epsilon** (shadow-mapping practice): the strict
  first-surface visibility test adds a tilt-proportional bias so smooth
  tilted surfaces stop occluding themselves, while front-on sheets keep
  the base epsilon and the two-sheet ghosting protection.
- The four math fixes above ship as part of this decision, plus a
  singular-solve guard in the harmonic fill (fully unobserved disconnected
  components fall back to the KD fill instead of painting black).

## Consequences

- On the adversarial QA harness (20 views x 5 defect detectors +
  pose-aware identity gates, calibrated so the reference photos
  themselves pass), the checked face bundle improves from 66 failed
  checks (rejected bundle) to 9-10; dark-debris view failures 22 -> 0,
  doubled-feature classes -> 0. Front-photo identity SSIM at the
  estimated pose rises 0.516 -> 0.61-0.63.
- Observed coverage drops (0.57 -> ~0.40): surrendered mixture bands and
  gated mirror sources trade counted coverage for correctness; the
  surrendered regions render as smooth fill instead of flakes. Rear hair
  loses some strand energy (Laplacian 2.09 -> 1.72) — a texture-detail
  cost of refusing mixture witnesses.
- Estimating the source pose costs ~50 untextured renders (~10 s on the
  checked host); overlap registration adds one grid search per reference
  (~15 s at 1024).
- Remaining known limits, by attribution: eye-region geometry renders the
  side photos' eyes as thin slivers (geometry, not texture); the
  front-vs-profile photo tone difference leaves a pale band at the
  handoff (needs band-limited local harmonization, unimplemented); the
  hairline band renders clean but slightly hair-thinned because a baked
  opaque texture must commit each texel to one material — faithful
  wispiness requires alpha-carrying shells or generative inpainting,
  both outside projection-bake scope.
- The evidence standard is now enforced by process: no quality claim
  without the QA harness (full-resolution renders, per-view gates,
  pose-aware identity) plus full-resolution crop review. Contact-sheet
  thumbnails are presentation, not evidence.
