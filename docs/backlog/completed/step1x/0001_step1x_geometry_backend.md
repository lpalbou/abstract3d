# Completed: Official Step1X-3D geometry backend

## Metadata

- Created: 2026-06-22
- Status: Completed
- Completed: 2026-06-22

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md), [ADR 0003](../../../adr/0003_trellis2_uses_official_upstream_assets_only.md), [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: Resolved by ADR 0004

## Context

The repository already ships a validated local TripoSR backend and an experimental official-only TRELLIS.2 backend. The active user direction is to stop pursuing TRELLIS.2 and implement official Step1X-3D instead, with six-agent design/review support, a proper backlog, local-only execution, and production-grade proof artifacts.

## Current code reality

- `src/abstract3d/backends/triposr_runtime.py` is the validated local-first backend and already establishes patterns for source pinning, composed `t23d`, artifact bundles, and validation contact sheets.
- `src/abstract3d/backends/trellis2_runtime.py` establishes stricter provenance checks for official-only assets and backend-specific environment/config knobs.
- `src/abstract3d/backends/__init__.py`, `src/abstract3d/scene3d_manager.py`, and `src/abstract3d/integrations/abstractcore_plugin.py` already support multiple scene3d backends.
- `docs/models.md` currently lists `stepfun-ai/Step1X-3D` as research only, with outdated assumptions that the full stack is the only meaningful integration target.
- Upstream `stepfun-ai/Step1X-3D` officially publishes a geometry checkpoint (`Step1X-3D-Geometry-1300m`, about 7.25 GiB), a label-geometry checkpoint, and a texture adapter checkpoint (`Step1X-3D-Texture`, about 3.6 GiB). The upstream texture path also pulls `stabilityai/stable-diffusion-xl-base-1.0`, custom CUDA rasterization, and NVIDIA-centric dependencies.
- Upstream geometry checkpoint metadata shows `caption_encoder` and `label_encoder` are null for `Step1X-3D-Geometry-1300m`, so native text-to-3D is not part of the official geometry checkpoint shipped today.

## Problem

The repo does not yet provide a production-grade official Step1X backend, and the existing catalog/docs do not distinguish clearly enough between the permissive geometry path and the non-permissive plus CUDA-heavy texture path.

## What we want to do

Add an official local Step1X-3D geometry backend for `image_to_scene3d`, support composed `text_to_scene3d` through `abstractvision`, and publish validation evidence that proves the local geometry path works on this machine.

## Why

The user explicitly selected Step1X-3D as the new backend direction. A narrow geometry-first integration offers a higher-quality official model family than TripoSR without inheriting TRELLIS.2's DINOv3 licensing problem or silently shipping Step1X texture dependencies that violate the repo's current permissive/runtime bar.

## Requirements

- Use only official Step1X-3D upstream code and official Step1X geometry checkpoint assets.
- Keep runtime local-only with no remote inference fallback.
- Prefer bounded downloads: geometry subset only unless separate justification is recorded.
- Expose the backend cleanly through manager, CLI, plugin registration, and model catalog.
- Support composed `t23d` only if it is clearly documented as `text -> image -> geometry`.
- Produce proof artifacts, benchmark notes, and coredoc updates.
- Capture actual storage, memory, and timing observations from local validation.

## Suggested implementation

- Create a new backend runtime that follows the TripoSR/TRELLIS patterns for pinned source checkout, bounded Hugging Face downloads, and artifact packaging.
- Start with geometry-only output and explicitly disable or reject the texture stage.
- Add minimal Apple-safe compatibility work only where it is needed for the geometry path.
- Update packaging extras to install only the dependencies required for geometry inference and validation.
- Record the geometry-only Step1X policy in a new ADR and in model/docs pages.

## Scope

- Backend runtime
- backend registry and aliases
- packaging extras
- manager, CLI, and plugin registration updates
- tests
- docs, benchmarks, and proof artifacts
- backlog and ADR updates

## Non-goals

- Shipping the full upstream Step1X texture pipeline
- Relaxing the repository's local-first or provenance policy
- Claiming native text-to-3D where the official geometry checkpoint does not provide it
- Revalidating TRELLIS.2

## Dependencies and related tasks

- `docs/backlog/proposed/step1x/0002_step1x_texture_stage_research.md`
- upstream repo: `https://github.com/stepfun-ai/Step1X-3D`
- upstream model: `https://huggingface.co/stepfun-ai/Step1X-3D`
- `../abstractcore` scene3d backend routing tests

## Expected outcomes

- `abstract3d` ships an official Step1X geometry backend that can be selected locally.
- Docs and catalog distinguish validated, experimental, and research status accurately.
- The repo contains proof artifacts and benchmark notes for the Step1X geometry path.
- The Step1X texture stage is either explicitly out of scope or separately tracked, not implied by silence.

## Validation

- `pytest -q tests`
- relevant `../abstractcore` scene3d routing tests
- direct local Step1X geometry smoke run
- reproducible validation harness output and contact sheet publication
- documentation pass covering README, docs pages, ADR index, and llms files

## Progress checklist

- [x] Confirm local feasibility of the official geometry checkpoint on this machine
- [x] Define the backend boundary and record the ADR
- [x] Implement the runtime and backend registration
- [x] Add or update tests
- [x] Run local proof generation and benchmarks
- [x] Update docs and backlog closure metadata

## Guidance for the implementing agent

Re-check upstream license/runtime details before finalizing the backend. If the geometry path cannot be validated locally on this machine, stop and record the blocker explicitly instead of pretending the full Step1X stack is ready.

## Completion report

### Date

2026-06-22

### Summary

Completed a real local Step1X geometry-only backend for `abstract3d`, wired it into `abstractcore`, published checked proof assets, and documented the result honestly as experimental rather than validated-default.

### Files and symbols touched

- `src/abstract3d/backends/step1x_runtime.py`
- `src/abstract3d/backends/__init__.py`
- `src/abstract3d/integrations/abstractcore_plugin.py`
- `src/abstract3d/model_catalog.py`
- `src/abstract3d/cli.py`
- `scripts/validate_local.py`
- `tests/test_step1x_backend_unit.py`
- `tests/test_scene3d_manager.py`
- `tests/test_plugin_registration.py`
- `tests/test_model_catalog.py`
- `tests/test_packaging_metadata.py`
- `../abstractcore/abstractcore/capabilities/scene3d_selectors.py`
- `../abstractcore/tests/test_capabilities_registry.py`
- `../abstractcore/tests/test_multimodal_generate_output.py`
- public docs, ADRs, and LLM index files

### Tests

- `../.venv/bin/pytest -q tests`
- `../.venv/bin/pytest -q ../abstractcore/tests/test_capabilities_registry.py ../abstractcore/tests/test_multimodal_generate_output.py ../abstractcore/tests/capabilities/test_model_residency_facades.py ../abstractcore/tests/server/test_server_model_residency_control_plane.py`

### Validation

- checked historical Step1X baseline proof under `artifacts/validation/local-step1x/`
- current Apple-local Step1X reference lane under `artifacts/validation/local-step1x-label/`
- docs-ready Step1X proof assets under `docs/assets/validation/local-step1x-label/`
- docs-ready comparison assets under `docs/assets/validation/comparison-step1x-vs-triposr/`

### Docs updates

- synchronized README, getting-started, architecture, API, model strategy, benchmarks, AbstractCore integration, FAQ, troubleshooting, ADR index, and LLM indexes with the Step1X backend reality

### Behavior changes

- `abstract3d` now exposes `abstract3d:step1x-local` / `step1x` as an experimental local geometry backend
- the supported Step1X path is official-model-only, geometry-only, and composed for `t23d`
- the Apple-local runtime uses `float32` on `mps`

### Residual risk

- on the checked Apple-local benchmark cases, Step1X was much slower than TripoSR and produced less recognizable object shapes
- the texture stage remains intentionally out of scope

### Follow-ups

- [0002_step1x_texture_stage_research.md](../../proposed/step1x/0002_step1x_texture_stage_research.md)
- [0003_step1x_quality_tuning_on_apple_silicon.md](../../planned/step1x/0003_step1x_quality_tuning_on_apple_silicon.md)
