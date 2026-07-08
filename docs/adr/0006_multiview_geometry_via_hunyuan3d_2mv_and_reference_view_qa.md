# ADR 0006: Multi-View Geometry via Hunyuan3D-2mv and Reference-View QA

- Status: accepted
- Date: 2026-07-04

## Context

Single-image reconstruction cannot recover surfaces the photo never saw: on the checked
face proof, the back and sides of the head were hallucinated by the shape prior, and side
photos fed to the texture bake could paint those surfaces but never correct their shape.
An adversarial design review of the multi-view texture path additionally found that
reference views were second-class citizens: their declared angles ("side_left") were
trusted verbatim, no mechanism could detect that a reference made the texture worse, the
depth-occlusion tolerance did not scale with mesh size, and hidden-region fill borrowed
colors across empty space instead of along the surface.

Three geometry-side options were considered: iterative geometry re-runs conditioned on
baked textures (rejected: no such conditioning channel exists in single-image models),
visual-hull carving from reference silhouettes (rejected for the main path: pose errors
carve real geometry; accepted only as a possible gated experiment), and a natively
multi-view-conditioned checkpoint.

## Decision

1. Serve the official `tencent/Hunyuan3D-2mv` multi-view checkpoint (2.0 family, 1.1B
   DiT) through the existing Hunyuan backend rather than adding a new backend. The
   checkpoint config targets the 2.0 namespace (`hy3dgen.shapegen.*`); the loader rewrites
   it to the vendored 2.1 namespace (`hy3dshape.*`), where every referenced class exists
   1:1. The rewrite is verified key-exact against the checkpoint at load time (0 missing,
   0 unexpected, 0 shape-mismatched keys across VAE, DiT, and conditioner).
2. Reuse the existing reference-view inputs (`texture_reference_images` /
   `texture_reference_angles`) as the multi-view source. References whose angles snap to
   the trained `front`/`left`/`back`/`right` slots within 25 degrees join the geometry
   conditioning dictionary; all references continue to feed the texture bake. Metadata
   records `multiview_conditioning` and `geometry_views`.
3. Make reference views first-class in the texture bake: per-reference pose solving in a
   window around the declared angle (accepted only when it beats the declared pose's
   silhouette IoU by a clear margin — flat IoU landscapes otherwise drift as often as they
   correct), overlap-based color harmonization with a revert-on-confound rule (gains that
   fail to reconcile the overlap indicate content mismatch, not exposure), a
   reprojection-error QA gate against the union of previously accepted views (catches
   reference-vs-reference conflicts the source never observes), per-texel best-witness
   conflict resolution (localized disagreement zeroes the weaker witness only where
   disputed, instead of punishing the whole view), mesh-scale-relative depth occlusion
   tolerance, and a mesh-graph harmonic fill (Dirichlet Laplace solve over mesh edges) for
   unseen texels.
4. Restrict the world-frame azimuth-sector masks in the projector to the no-depth-map
   fallback path. When the rendered depth test is available it strictly dominates those
   heuristics; empirically the sector mask discarded about half of the depth-validated
   texels on profile views.

Two mechanisms were tried and reverted after adversarial empirical testing, and are
documented here so they are not reintroduced: a photometric-NCC tie-break over top
silhouette candidates for the source pose (the interior-edge NCC landscape on faces is
flat and structureless; it "decisively" preferred poses 15 degrees off frontal, and every
reference inherited the error), and a mirrored-pose retry for swapped left/right labels
(its acceptance signal — low disagreement on a small accidental overlap — also fired on
correctly labeled views and stole coverage from the true side).
4. Keep both official repositories behind the same explicit Tencent Hunyuan Community
   License acknowledgment (`ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`); both are
   territory-restricted (EU/UK/South Korea excluded).

## Consequences

- Multi-angle photos now constrain shape, not just paint. Checked face proof (front +
  both profiles): observed texture coverage 0.19 -> 0.74; the hallucinated bald back of
  the head is replaced by a hair-consistent shape; profiles match the reference photos.
- The 2mv checkpoint is a 2.0-family model: it lacks the 2.1 flagship's fine geometric
  micro-detail at equal settings, so single-image runs should stay on `Hunyuan3D-2.1`.
  The backend keeps 2.1 as its default model.
- The QA machinery makes bad references detectable and survivable: global attenuation,
  per-texel conflict zeroing, and pose acceptance decisions are recorded per view in
  `view_consistency` / `view_registration` stats. A deliberately mislabeled profile
  (declared 105 instead of ~90) no longer paints the back of the head.
- The pinned 2.1 source snapshot now serves two checkpoint families; upgrading it
  requires re-verifying the 2mv key-exact load.
