# Architecture Decision Records

This folder contains the durable policy decisions that govern Abstract3D.

## Accepted ADRs

- [ADR 0001: Scene3D is local-first and GLB-centric](0001_scene3d_local_first_glb_contract.md)
- [ADR 0002: The validated backend uses pinned TripoSR and composed `t23d`](0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [ADR 0003: TRELLIS.2 uses official upstream assets only](0003_trellis2_uses_official_upstream_assets_only.md)
- [ADR 0004: Step1X uses an official geometry-only backend with local compatibility patches](0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- [ADR 0005: Hunyuan3D-2.1 is a license-gated official shape backend](0005_hunyuan3d21_license_gated_official_shape_backend.md)
- [ADR 0006: Multi-view geometry via Hunyuan3D-2mv and reference-view QA](0006_multiview_geometry_via_hunyuan3d_2mv_and_reference_view_qa.md)
- [ADR 0007: Canonical orthographic projection and first-surface visibility](0007_canonical_orthographic_projection_and_first_surface_visibility.md)
- [ADR 0008: Photometric source pose, overlap registration, and witness gating](0008_photometric_pose_overlap_registration_and_witness_gating.md)
- [ADR 0009: Export material truth and unseen-area detail synthesis](0009_export_material_truth_and_fill_detail_synthesis.md)

## How To Use These ADRs

- update them when a durable decision boundary changes
- link them from user-facing docs when the policy affects setup or behavior
- do not use backlog notes or ad hoc comments as substitutes for these policies
