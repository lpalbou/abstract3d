# ADR 0007: Canonical Orthographic Projection and First-Surface Visibility

- Status: accepted
- Date: 2026-07-05

## Context

The multi-view face proof looked wrong from every angle except the source
view, and an instrumented root-cause hunt (four adversarial agents plus a
ground-truth harness where every camera answer is known by construction)
found five independent, compounding causes rather than one bug:

1. The preview renderer re-lit textured meshes with a strong diffuse term.
   On generated geometry whose surface detail disagrees with the photo by a
   few percent, the mesh's own eye sockets, brow ridges, and lip creases
   were drawn in shading ON TOP of the photo albedo — a second "shading
   face" offset from the painted face, which read as ghosted duplicated
   features. A CPU reference rasterizer of the same textured mesh produced
   clean output, isolating the renderer.
2. The texture projector used a perspective pinhole (fovy 40) for photos
   that are effectively orthographic (Hunyuan reconstructs from a
   recentered conditioning image under an orthographic training camera;
   studio portraits are long-lens). Perspective magnifies near features
   relative to far ones, so features could never land exactly where the
   model built them.
3. Depth-map occlusion with a tolerance stamps BOTH sheets of the thin
   crust films that generated meshes grow (hair shells sit 0.005-0.02
   world units apart, below any tolerance that survives depth-map
   interpolation), duplicating photo features onto hidden geometry.
4. rembg's default u2net checkpoint amputated 40% of the subject on the
   checked profile photo (dark hair mass against a light background),
   corrupting the alpha-driven framing of every downstream stage.
5. Silhouette-IoU registration cannot align differently-cropped photos (a
   head-only profile against a head-plus-shoulders mesh silhouette) and
   rewards degenerate blow-ups where the mismatch leaves the frame.

## Decision

1. Canonical-frame orthographic projection for canonicalizing backends:
   the bake replicates the model's own preprocessing (bounding-box
   recenter at the model's border ratio — `ImageProcessorV2.recenter` for
   Hunyuan) and projects with the orthographic half-extent that reproduces
   that exact framing per view. Source-photo registration becomes
   deterministic; pose estimation and silhouette scale fitting are
   bypassed for the source view. `bake_projection_texture` gains
   `projection_model="orthographic"` and `canonical_border_ratio`.
2. Strict first-surface visibility everywhere: the projector builds a
   per-photo-pixel z-buffer from the projected surface texels themselves
   (every projectable texel occludes, regardless of facing or photo
   alpha), with a 3x3 conservative widening and an epsilon of 0.25% of the
   surface diagonal. The GL depth-map test and the world-frame azimuth
   sector masks are removed from the projector.
3. Flat-biased preview shading for textured meshes (12% diffuse cue): the
   preview must review the baked albedo, not re-light it.
4. Robust segmentation (`abstract3d.segmentation`): prefer the
   `isnet-general-use` checkpoint, then clean the matte (largest
   components, closed pinholes) before any geometric use of alpha.
5. Crop-aware edge-chamfer registration for reference photos: silhouette
   EDGES must coincide (symmetric chamfer distance), rows/columns touching
   the photo frame are treated as crop lines rather than shape boundaries,
   and reference views keep the full scale/shift search while the
   canonical source frame keeps only a residual search.
6. Surface-consensus outlier rejection and source-priority conflicts: a
   two-hop mesh-graph consensus iteratively erodes observed texels whose
   winning view AND color are foreign to their neighborhood (rim
   misprojections); on disputed texels the SOURCE photo wins wherever it
   faces the surface well (weight above 0.45), otherwise the best-facing
   witness wins.

## Consequences

- The checked face bake is clean at the source view and coherent from all
  azimuths; observed coverage on the corrected face mesh reaches ~0.89 at
  2048 with the remaining hidden surface filled by the anisotropic
  harmonic diffusion (crease-aware conductance so shell seams do not leak
  the underlying material's color).
- Perspective projection remains available (`projection_model=
  "perspective"`) for backends whose training cameras are perspective
  (TripoSR keeps its existing path).
- Known residual limitation: where the generated mesh grows hair-shell
  fringes whose hairline disagrees with the photo, the fringe tips
  legitimately receive skin pixels from the photo (the photo really shows
  skin along those rays). The outlier filter removes isolated patches;
  contiguous fringe bands shrink but do not vanish. The durable fix is
  geometry-side (fewer crust films), tracked with the volume-decoder
  experiments.
- The GL depth renderer (`_tripo_render_camera_depth_map`) stays available
  but is no longer part of projection visibility.

## Amendments From the Four-Agent Audit (same date)

Independent adversarial audits confirmed the convention layer is correct
(projector ground-truth error 0.00075; every hypothesized mirror/flip is
strictly worse; the Hunyuan 2mv view-tag map matches upstream examples) and
contributed four additional proven fixes now shipped: the pose-search grid
excluded the declared center pose; the camera-distance fit had a
width-vs-height NDC normalization error plus a one-pass depth bias (now
iterated); registration masks were squashed anisotropically (now
letterboxed); and harmonization gains are gated on per-texel log-ratio
spread. Two mechanisms were evaluated and REMOVED from the default path
with ground-truth evidence: the photometric NCC registration refiner
(recovered 0/15 injected shifts; constant-attractor behavior) and
unrestricted mirror-fill sources (89% were grazing rim samples; sources now
require blend weight >= 0.35). The audits also established that the
2026-07-04 `face-multiview-proof` artifacts were generated with
chirality-swapped reference photos and defective mattes; they are tainted
evidence, superseded by `artifacts/validation/iter3-multiview-fixed/`, and
the photo-set status is documented in
`artifacts/validation/face-multiview-prototype/README.md` (local validation
archive; superseded experiments are kept locally and not versioned).
