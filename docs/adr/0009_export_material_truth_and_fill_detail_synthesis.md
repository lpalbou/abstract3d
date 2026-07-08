# ADR 0009: Export Material Truth and Unseen-Area Detail Synthesis

Date: 2026-07-05
Status: Accepted

## Context

After the ADR 0008 cycle fixed projection and registration, the project
owner inspected exports in an external PBR viewer (MeshVault) and found
the textures still wrong: assets rendered dark and metallic, unseen
regions read as faceted flat-color mush (face crown, starship underside
— 76% of a single-view asset), the face showed a mid-face tone seam and
eye-socket smears at close zoom. A four-agent adversarial texture cycle
traced these to five distinct causes.

## Findings and decisions

1. **Export material truth.** Every baked-texture export carried
   trimesh's SimpleMaterial defaults: GLB `baseColorFactor [0.4,0.4,0.4]`
   with NO `metallicFactor` (glTF defaults to metal 1.0), MTL
   `Ka/Kd/Ks 0.4`. Spec-compliant viewers multiplied the albedo by 0.4
   and rendered it as rough metal — up to ~10x too dark under IBL. The
   repo's own preview renderer sampled the raw texture, masking the
   defect. DECISIONS: textured meshes are constructed with an explicit
   `PBRMaterial` (white base color, metallic 0.0, roughness 1.0); OBJ
   sidecars write `Ka/Kd 1.0, Ks 0.0` (Ks 1.0 measured +221% washout in
   Phong viewers); the preview renderer multiplies by the material's
   base color factor so previews can never look better than a real
   viewer; `scripts/check_export_materials.py` and material gates in
   `scripts/texture_qa.py` prevent regression.
2. **Unseen-area fill quality.** The vertex-domain harmonic fill
   assigned each texel its nearest vertex's color — every vertex Voronoi
   cell (~70 texels at 2048) rendered as one flat polygon facet. Large
   fill regions were correct-on-average but characterless. DECISIONS:
   the harmonic fill interpolates 3 nearest vertices (IDW); a
   `texel_surface_smooth` Jacobi pass over the texel 3D KD-graph runs
   after both fill paths (face flat-plateau fraction 0.45 -> 0.18); a
   deterministic `synthesize_fill_detail` pass transfers texture
   STATISTICS (robust L1 amplitude + structure-tensor streak orientation
   per material, carried by 3D value noise with line-integral-convolution
   streaking) from observed to fill regions — fill/observed gradient
   energy 0.53 -> 0.83 (face), 0.15 -> 0.59 (ship). Literal patch copying
   was tested and REJECTED (chaotic fragments, banding). Detail transfer
   renders the correct MATERIAL, not invented content.
3. **Single-view completion.** A single photo observes 6-30% of a
   symmetric object; the mirrored twin of the observed sliver is real
   content where any propagated fill is a wash. DECISION:
   `texture_completion="auto"` applies mirror completion whenever the
   geometry's measured symmetry passes the existing 0.55 gate; the
   Hunyuan backend defaults to auto.
4. **Observed-area close-range defects (face).** The mid-face tone seam
   is a winner handoff between views with genuinely different tone
   (front photo warmer than synthetic profiles; global gains cannot fix
   content differences). Half the dark cheek fragments were mirror-fill
   COPIES of hairline content; the ghost lip was the same family.
   DECISIONS: `level_composed_seams` — mesh-graph seam leveling
   (Ivanov/Lempitsky-style low-frequency offset field) with a material
   -boundary cap (hair|skin edges are never leveled; uncapped leveling
   tinted the ear) and confidence pinning (photos stay ground truth
   where they saw the surface well); a consensus guard in
   `mirror_fill_from_observed` rejects mirror copies contradicting a
   color-consistent observed neighborhood at the destination. Attempted
   and REVERTED with evidence: smear gating, mirror override, threshold
   lowering, local-flow registration (prototype kept, failed its A/B).
5. **Pose estimator hardening** (parent-agent work in the same cycle).
   The gradient scorer compared photo and render in mismatched frames —
   compact heads tolerated it; the elongated starship did not (its
   projected aspect swings with elevation), and the elevation grid
   lacked ±15. On bilaterally symmetric meshes the +az/-az mirror pose
   is a near-tie, and 0.1% vertex jitter could flip the argmax sign.
   DECISIONS: renders are aligned into the photo's frame with
   crop-immune anchors (subject top, silhouette centroid, mean width
   over common rows) before correlation; the elevation grid spans ±15
   with local refinement on both axes; a CHIRALITY TIE-BREAK correlates
   the horizontal anti-symmetric luminance components of photo and
   render — sign-opposite between mirror poses, so the sign is decided
   by a margin the symmetric content cannot dilute. Measured: ship pose
   recovered exactly (az +30, el +15; observed coverage 0.062 -> 0.20);
   face pose stable at +12.5..+20 under 0.1-0.3% vertex jitter (was
   sign-flipping); the same-view color-extreme condition of the outlier
   filter now also runs for single-view bakes.

## Consequences

- The adversarial texture harness (`scripts/texture_qa.py`: material
  gates, viewer-truth renders, photo-calibrated close-zoom facet/seam/
  fill-character/dark-fragment detectors) passes the face bundle fully;
  the starship passes 12/13 gates.
- Known limits, documented for honesty: single-view assets keep a few
  small dark fill fragments at 4x zoom (4-6 across all probe crops;
  down from whole-surface patchwork); fill detail is material-plausible
  texture, not invented content (no specific panel layouts or hair
  whorls); the harness's brightness gate requires a matted photo to be
  meaningful (an unmatted white background inflates the photo-side
  reference, seen on the owl bundle).
- The face bundle's front identity SSIM (0.61 vs the 0.70 gate
  calibrated on clean single-view bakes) remains open, attributed to
  the temple-band paint quality, hair variance, and renderer shading
  (ADR 0008).
- `python -m abstract3d.cli` now runs `main()` (missing `__main__`
  guard made it a silent no-op).
