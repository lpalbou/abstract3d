# Completed: AbstractCore scene3d integration (capability plugin, mesh ops, AI tools, server endpoint)

## Metadata

- Created: 2026-07-19 (operator intake, laurent session directive)
- Status: Completed
- Completed: 2026-07-19
- Hub receipts: commons #3146 (intro + asks), #3169 (core rulings), #3179 (draft receipt),
  #3188/#3199 (core owner review: accepted), #3197 (adversarial round), #3198 (wave receipt)

## ADR status

- Governing ADRs: none new (integration follows AbstractCore's ruled contracts:
  explicit-import tools, /v1/audio/music extension-endpoint precedent,
  decision:domain-tool-classification-tags for the classification vocabulary)
- ADR impact: none

## Context

Operator directive: make abstract3d a first-class optional capability plugin of
AbstractCore (the abstractvision/abstractvoice model) so AbstractCore can generate,
modify, manipulate and analyze 3D objects/scenes — including over the server's
OpenAI-compatible surface — plus a tool set usable by AI/agents/entities. The
capability seam (Scene3dCapability protocol, facade, selectors, generate() output
dispatch, residency) already existed core-side; the real gaps were the server
generation endpoint, the tools story, and abstract3d's manipulate/analyze surface.

## What was delivered

- `abstract3d.mesh_ops` + `abstract3d.mesh_preview`: deterministic, byte-faithful
  mesh operations (analyze/transform/compose/convert/repair/preview) with the
  `[mesh]` extra (numpy, trimesh, scipy, Pillow, matplotlib).
- `abstract3d.tools`: eight LLM tools with AbstractCore ToolDefinitions, ruled
  explicit-import accessors (`abstract3d_tools()` / `abstract3d_tool_definitions()`
  / `abstract3d_tool_specs()`), and `SCENE3D_TOOL_CLASSIFICATION` (mutating /
  remote_write_capable / downloads_model_weights — semantics-passed).
- Hunyuan3D-2.1 registered with the capability plugin behind its unchanged license
  gate; typed `LicenseAcknowledgmentRequiredError` with stable `error_class`.
- TripoSR unknown-option preflight (millisecond rejection before model load).
- Drafted in abstractcore's tree per core's ruling (their review + merge):
  `server/scene3d_endpoints.py` (POST /v1/scene3d/generations + provider-scoped
  alias; 501/403/422/413 semantics; sync-def threadpool handlers), hunyuan aliases
  in `scene3d_selectors.py`, app.py include, 20 endpoint tests. Core owner review:
  ACCEPTED as shipped (2204 green in their tree).
- Proof harness `scripts/abstractcore_integration_proof.py`; definitive 4-stage run
  green (core.generate / ToolRegistry / HTTP / live-LLM tool call) with artifacts at
  `out/abstractcore-proof-final/`.

## Validation

- abstract3d suite: 442 passed / 3 skipped / 3 xfailed.
- Two adversarial subagent reviews (0 P0 / 9 P1 / 13 P2) — every P1 fixed same-day
  with regression tests; MeshVault independent verification PASS (its one real
  finding — compose baking transforms into float32 vertices — fixed and re-measured
  at parity).

## Follow-ups revealed

- Runtime/gateway toolset mounting of `abstract3d_tool_definitions()` (runtime's
  lane; zero core changes per core ruling c3169).
- Possible Agent Skill with the skill seat once the runtime lane lands.
- Camera↔abstract3d plugin review swap (offered both ways, pending camera's ask).
- Endpoint ships when core commits their tree (standing no-commit rule).
