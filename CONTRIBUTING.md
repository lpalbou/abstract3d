# Contributing

## Setup

Clone the repository and install the local contributor profile you need.

Runtime-only work:

```bash
pip install -e ".[triposr,test]"
```

Apple-local `t23d` work:

```bash
pip install -e ".[all-apple,test]"
```

If you are validating the AbstractCore plugin path from the sibling monorepo checkout, also install the sibling package in the same environment.

## Test

Package-local tests:

```bash
pytest -q tests
```

Host-side capability tests from the sibling AbstractCore checkout:

```bash
pytest -q \
  ../abstractcore/tests/test_capabilities_registry.py \
  ../abstractcore/tests/capabilities/test_model_residency_facades.py \
  ../abstractcore/tests/test_multimodal_generate_output.py \
  ../abstractcore/tests/server/test_server_model_residency_control_plane.py
```

## Validation

The repo includes a reproducible local proof harness:

```bash
python scripts/validate_local.py --device mps --mc-resolution 128
```

This writes:

- full case bundles under `artifacts/validation/local-triposr/`
- public benchmark assets under `docs/assets/validation/local-triposr/`

Update [`docs/benchmarks.md`](docs/benchmarks.md) if the validated profile, hardware baseline, or measured numbers change materially.

## Change Discipline

- Keep the validated path local-first.
- Do not silently switch to remote 3D services.
- Keep `glb` as the primary artifact contract unless the governing ADRs change.
- Update [`docs/models.md`](docs/models.md) when the validated backend or research catalog changes.
- Update [`docs/adr/README.md`](docs/adr/README.md) when a durable policy boundary changes.
