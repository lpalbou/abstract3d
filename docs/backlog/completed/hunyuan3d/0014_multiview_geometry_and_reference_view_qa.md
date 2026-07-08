# 0014: Multi-View Geometry Conditioning and Reference-View QA

- Track: hunyuan3d / texturing
- Status: completed
- Date: 2026-07-04
- Decision record: [ADR 0006](../../../adr/0006_multiview_geometry_via_hunyuan3d_2mv_and_reference_view_qa.md)

## Problem

Side photos could only paint the mesh, never correct its shape: the face proof kept a
hallucinated back of the head regardless of how many reference views were provided. The
texture path also trusted declared reference angles verbatim and had no way to detect a
reference that made the texture worse.

## What Shipped

- `tencent/Hunyuan3D-2mv` served by the Hunyuan backend (`model=` selection, same license
  gate, `-fast`/`-turbo` subfolders accepted). Config namespace remap onto the vendored
  2.1 source, verified key-exact at load.
- Reference angle -> trained view slot snapping (front/left/back/right, 25 degrees) with
  `multiview_conditioning` and `geometry_views` recorded in metadata.
- Texture bake upgrades: per-reference pose solving (window around the declared angle,
  accepted only on a clear IoU margin over the declared pose), overlap-based color
  harmonization with revert-on-confound, reprojection-error QA gate against the union of
  accepted views, per-texel best-witness conflict resolution, mesh-scale-relative depth
  occlusion tolerance, and a mesh-graph harmonic fill (normal-weighted KD-tree fallback)
  for unseen texels. The projector's azimuth-sector masks now apply only on the
  no-depth-map fallback path.
- Two mechanisms were tried and reverted after an adversarial empirical attack run
  (documented in ADR 0006): a photometric-NCC source-pose tie-break and a mirrored-pose
  retry for swapped left/right labels. Both regressed correctly labeled inputs.

## Checked Evidence

- Face proof (front + left/right profiles + three-quarter): observed texture coverage
  0.19 (single view) -> 0.74; back of head hair-consistent; profiles match reference
  photos; a deliberately mislabeled profile (declared 105 instead of ~90) is contained by
  the union QA gate plus per-texel conflict resolution instead of painting the back of
  the head.
- Unit suite: 123 tests green, including model selection, config remap, pose-center
  fallback, harmonization gains, QA gate attenuation, and best-witness conflict
  resolution.

## Follow-Up (2026-07-05, ADR 0007)

An adversarial root-cause hunt on the remaining turntable defects found five compounding
causes outside the multi-view logic itself (preview-renderer relighting, perspective
projection of orthographic photos, tolerance-based occlusion stamping double crust
sheets, u2net matte amputation, and IoU registration degeneracy on cropped photos). With
those fixed the checked proof reaches observed coverage 0.91 with a coherent turntable
(`artifacts/validation/iter3-multiview-fixed/face-2mv/`); 131 tests green.
