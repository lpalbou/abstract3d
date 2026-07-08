# FINAL VERDICT — adversarial texture cycle, 2026-07-05 (tex4, QA/verdict agent)

Scope: `/Users/albou/abstract3d`, assets `iter3-multiview-fixed/face-2mv`,
`final-proof/hunyuan-starship` (owner-named), `final-proof/hunyuan-owl`
(sanity third). Harnesses: `scripts/texture_qa.py` (new: material truth,
viewer-truth renders, close-zoom gates — built and photo-calibrated this
cycle, unit-tested, 159-test suite green) + `/tmp/verdict1/qa.py`
(mid-range face battery). All numbers below were measured by me from raw
exported bytes and fresh renders; no agent's self-reported number is used
un-reproduced. Evidence: `/tmp/tex4/runs/*/evidence/`,
`/tmp/tex4/evidence_pairs/`.

## RULING

| lane | ruling |
|---|---|
| Exported materials (all assets, fresh export path) | **PASS — defect eliminated** |
| Face re-texture on canonical geometry (integrated stack, pose pinned/stable) | **PASS with one marginal note** — shippable replacement exists |
| Face fresh end-to-end rebake (new geometry) | **FAIL — pose lottery, catastrophic when it misses** |
| Starship rebake | **FAIL (1 gate): residual dark fill fragments at 4x; everything else passes** |
| Owl (untouched by cycle) | **FAIL — still ships the pre-cycle defects; needs rebake through fixed pipeline** |
| Shipped bundles as they sit in artifacts/ TODAY | brightness/metal fixed in place; close-zoom texture defects still present (textures were not re-baked in place) |

Bottom line for the owner: the cycle genuinely fixed the export materials
(the "60% darker + metal-dark" class, verified at byte level and in
viewer-truth renders) and the pipeline now produces fill that passes every
close-zoom gate on the face and all but one on the ship. But what is in
`artifacts/` right now still carries the old textures with faceted fill; a
re-texture of the face on its existing geometry passes the full battery
and is ready to ship, while fully fresh face bakes are currently a
coin-flip because of an unowned source-pose instability.

## Gate table (final detector rev; every cell re-measured)

texture_qa (13 gates: 7 material, brightness, fill-energy, facet-cellular,
seams, facet-fields@4x, dark-smears@4x):

| bundle | failed gates | detail |
|---|---|---|
| face pre-cycle (backup) | 9/13 | factor 0.4 + metal default; brightness 0.392; fill energy 0.405; facet honeycomb; 4x fields; smears |
| ship pre-cycle (backup) | 9/13 | same classes; fill energy 0.224; smears |
| owl (live, untouched) | 9/13 | same classes; fill energy 0.136; brightness 0.232 |
| face shipped today (materials patched in place) | 3/13 | fill energy 0.405, facet cellular 0.577, facet fields 4x — texture unchanged |
| ship shipped today | 3/13 | fill energy 0.224, facet cellular, dark smears |
| **face re-texture, pose pinned +17.5 (full integrated stack: IDW+smooth fill, detail synthesis, mirror consensus guard, seam leveling)** | **0/11 applicable** | fill energy 0.925 (coarse 0.825), facets 0, smears 0, brightness 0.988, seams p95 32.2 (allow 72.3) |
| **agent 3 final 2048 re-texture (pose 20.0, pre-detail/leveling tip)** | **0/13** | fill energy in band, facets 0, smears 0, brightness pass |
| **ship fresh rebake 2048** | **1/13** | dark smears @4x: 9 fill-region fragments remain; fill energy 0.578, facets 0, brightness 0.779 |
| face fresh rebake (new geometry, pose -15) | 0 texture_qa / **65 verdict1** | close-range gates blind to doubling; mid-range battery catches it |

verdict1 (face lane; reference = shipped bundle post-material-fix, 6
failures; identity floor front 0.616/23.5, side_left 0.680/17.9,
side_right 0.679/24.7):

| candidate | failures | identity |
|---|---|---|
| face re-texture pinned +17.5 | **6 (= reference)** | side_left 0.687 (+0.007), side_right 0.682 (+0.002, MAE -2.4), front 0.603 (-0.013, MAE -0.1) |
| agent 3 REPO2048 (pose 20) | 9 | parity (±0.005); +4 marginal dark_debris (0.0034-0.0047 vs 0.003) at az -22.5..-45, +1 eye_count at az+22.5 el10; az-0 doubled eyes FIXED |
| fresh full rebake (pose -15) | **65** | front 0.566; doubled features at nearly every azimuth |

## Per-defect-class ruling

1. **Dark + metal exported materials — GONE.**
   Root cause (trimesh SimpleMaterial 0.4 defaults + omitted metallicFactor
   = glTF metal 1.0) fixed at the exporter; shipped face+ship GLB/MTL
   patched in place with BIN chunks sha256-identical. Verified: raw-byte
   gates on 5 bundles, `check_export_materials --strict`, fresh bakes
   through the real export path, viewer-truth brightness 0.779-0.988 vs
   input photos (was 0.23-0.39). Agent 1's report is accurate, including
   their justified Ks=0 deviation; they also removed the masking defect in
   the repo preview renderer (my independent renderer agrees pixel-exact,
   SSIM 1.000). Evidence: `evidence_pairs/face_viewer_truth_before_after.png`.

2. **Close-zoom faceted fill (Voronoi honeycomb) — GONE in rebakes,
   REMAINS in the shipped textures.**
   Root cause (nearest-vertex texel assignment in harmonic fill) fixed by
   agent 2 (IDW-3 + anchored texel-graph smoothing). My facet-field
   detector (photo-calibrated: photos 0, true honeycombs >= 0.81 flat
   tiling) finds 0 fields at 4x on the pinned face re-texture, REPO2048,
   and the fresh ship rebake; the shipped face texture still shows the
   honeycomb (it was never re-baked). Evidence:
   `evidence_pairs/face_chin_4x_before_after.png`.

3. **Fill character (flat mush) — face FIXED, ship IMPROVED to pass.**
   Gradient-energy ratio synthesized-vs-observed: face 0.405 -> 0.925
   (agent 2's statistics-transfer detail synthesis is real texture, coarse
   ratio 0.825 proves it is not injected pixel noise); ship 0.224 -> 0.578.
   Gate >= 0.5 passes on both.

4. **Eye-socket / dark-fragment class — face az-0 doubling FIXED in
   re-textures; displaced-fragment residue REMAINS.**
   Agent 3's provenance work root-caused the smears to displaced photo
   content + mirror copies; their consensus guard removed the mirror
   component (their negative results on smear gates are honest and
   verified consistent with my measurements). Residue: 4 marginal
   dark_debris views on REPO2048; az±90 profile eye undercount predates
   the cycle and remains (registration class). Ship: 9 fill-region dark
   fragments at 4x remain (patchwork residue) — the one ship gate still red.

5. **Tone seams — within photo-calibrated allowance throughout.**
   Texel-space seam steps: face worst p95 deltaE 32-34 (allowance 72.3),
   band-median 6.7-9.3 (allowance 17.0); ship/owl similar. Agent 3's seam
   leveling improves the mid-face chroma-seam counts at 1024 (16->14 their
   A/B) and does not regress identity at 2048; at 2048 its effect is
   admittedly subtle (their own honest limit #3).

6. **Ear crackle (cross-agent WIP regression I flagged at 13:20) —
   RESOLVED in finals.** side_right worst local window: 0.115-0.124 in
   final candidates (gate 0.05), vs -0.105/-0.117 in both agents' interim
   candidates.

7. **NEW, UNOWNED: front source-pose instability — the cycle's one
   blocking regression.**
   `estimate_pose_photometric` picks the front pose from a nearly flat
   NCC landscape (scores 0.009-0.05). On the canonical face geometry the
   current tip lands az +20 (near the shipped +17.5: acceptable). On a
   freshly generated geometry (MPS mesh jitter; same seed, 120000 vs
   119999 faces) it flipped to az **-15**, projecting the photo from the
   wrong side of the face: 65 verdict1 failures, doubled features
   everywhere. The ship's pose estimate also wandered across the day
   (az 35 el -15 at the 13:06 tip; az 27.5 el +15 — correct side — at the
   14:18 tip; agent 2 measured az 42.5 el -8 earlier). No agent claimed or
   fixed this; agent 2 explicitly deferred it. Until the estimator gains a
   stability margin (or bakes pin the pose when the score is under a
   floor), every fresh end-to-end bake is a pose lottery.

## What the owner will see in MeshVault now

- **Brightness/metal**: opening the CURRENT `scene.glb` of face or ship:
  correct full-brightness albedo, non-metal, roughness 1.0 — same
  appearance as the repo previews (which now honor the factor). The owl
  still opens dark+metallic until rebaked.
- **Close zoom TODAY (shipped artifacts)**: face/ship textures are still
  the old bakes — faceted under-chin/nape fill and ship patchwork are
  still there at 2-4x. The material patch could not and did not fix these.
- **Close zoom after re-texture (validated candidates:
  `/tmp/tex4/rebake/face_pinned`, `/tmp/tex3/bake_REPO2048`)**: smooth
  skin with plausible micro-texture under the chin/nape, no polygon
  blocks at 4x, no black smears in fill, tone within photo-natural
  variation; profile identity slightly better than shipped. Front view at
  ~0.6 SSIM vs photo remains below the 0.70 aspiration (projection/
  registration class, pre-existing).
- **Ship after rebake**: correct brightness, mottled-metal fill instead of
  billiard-flat wash; residual dark fragments visible at 4x in some fill
  areas; content placement depends on the estimated pose (correct side
  this time, elevation matched the old bake).

## Honest remaining limits

1. Front-pose instability (above) — blocks unconditional fresh bakes; the
   pinned/frozen-geometry re-texture lane is the shippable path today.
2. Front identity SSIM 0.60-0.62 vs the 0.70 gate on every candidate ever
   measured (incl. shipped): per-feature photo-geometry mismatch (nose
   -10px, mouth (-4,+4), eyes (+4,0) @512, non-rigid — agent 3's
   measurement). Requires validated local-flow registration (prototype
   exists, honestly rejected after A/B) or better geometry.
3. az±90 profile eye undercount (verdict1) predates and survives the cycle.
4. Ship: 9 dark fill fragments at 4x; pose-dependent content placement is
   ungated by any harness (no reference views exist for non-front ship
   surfaces) — a pose-truth gate for non-face assets is future QA work.
5. The photo's own baked-in shading gradient remains in the face texture
   (delighting is out of scope for a projection bake).
6. Owl untouched: rebake through the fixed pipeline before shipping it.
7. QA footgun found: `python -m abstract3d.cli ...` exits 0 doing NOTHING
   (no `__main__` guard); use the `abstract3d` console script. Cost me one
   silent no-op bake; will bite CI someday. (Pipeline fix not in my lane.)

## Process notes

- Agent 1 patched artifacts in place at 12:39 mid-baselining; pre-cycle
  numbers were regenerated from their byte-verified backups.
- The shared tree changed continuously (texturing.py landed in pieces
  13:06-14:18+); each bake below is tagged with its tip time. Bakes are
  bit-deterministic per tip (verified by agent 3, spot-consistent with my
  reruns).
- My facet-field detector was tightened mid-cycle (flat-tiling >= 0.60
  requirement) after the ship rebake exposed over-firing on mottled fill;
  ALL table rows above were re-measured with the final detector, and the
  input photos still pass calibration on all three assets.
- Full run inventory and adjudication trail: /tmp/tex4/BASELINE.md,
  /tmp/tex4/ADJUDICATION.md, /tmp/tex4/runs/.
