# Abstract3D Local Validation

- Backend: `step1x`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.12.13`
- Device: `mps`
- Marching cubes resolution: `128`
- Image model for composed/object inputs: `AbstractFramework/flux.2-klein-4b-8bit`

## Aggregate

- Cases: `4` (`t23d=2`, `i23d=2`)
- Average total time: `63.3743` s
- Average inference time: `24.9962` s
- Average mesh extraction time: `32.4387` s
- Average preprocessing time: `1.0976` s
- Average text-to-image composition time: `7.2627` s
- Average RSS: `19.5829` GiB
- Average MPS allocated: `0.0` GiB
- Average vertices: `56236.6667`
- Average faces: `108998.3333`

## Generated Inputs

- `01_input.png`: `768x768`, `437739` bytes
- `02_input.png`: `768x768`, `755486` bytes

## Per Case

| Case | Mode | Status | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_t23d | t23d | succeeded | 58.1184 | 6.2111 | 1.1239 | 22.3064 | 28.477 | 40377 | 79573 | 30.8584 | 0.0 |
| 02_t23d | t23d | succeeded | 85.9541 | 8.3143 | 1.1933 | 35.6316 | 40.8149 | 77938 | 148630 | 17.3461 | 0.0 |
| 03_i23d | i23d | failed | None | None | None | None | None | None | None | None | None |

- `03_i23d` failure: `Fetching 11 files:   0%|          | 0/11 [00:00<?, ?it/s]
Fetching 11 files: 100%|██████████| 11/11 [00:00<00:00, 15671.65it/s]
/Users/albou/tmp/abstractframework/.venv/lib/python3.12/site-packages/lightning_fabric/__init__.py:41: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
/Users/albou/tmp/abstractframework/.venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py:7: FutureWarning: Importing `BackboneConfigMixin` from `utils/backbone_utils.py` is deprecated and will be removed in Transformers v5.10. Import as `from transformers.backbone_utils import BackboneConfigMixin` instead.
  warnings.warn(
/Users/albou/tmp/abstractframework/.venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py:15: FutureWarning: Importing `BackboneMixin` from `utils/backbone_utils.py` is deprecated and will be removed in Transformers v5.10. Import as `from transformers.backbone_utils import BackboneMixin` instead.
  warnings.warn(

Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]
Loading pipeline components...:  83%|████████▎ | 5/6 [00:01<00:00,  3.95it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:01<00:00,  4.64it/s]
[transformers] `use_return_dict` is deprecated! Use `return_dict` instead!

  0%|          | 0/8 [00:00<?, ?it/s]
 12%|█▎        | 1/8 [00:04<00:29,  4.19s/it]
 25%|██▌       | 2/8 [00:06<00:16,  2.80s/it]
 38%|███▊      | 3/8 [00:09<00:16,  3.23s/it]
 50%|█████     | 4/8 [00:16<00:18,  4.69s/it]
 62%|██████▎   | 5/8 [00:18<00:11,  3.80s/it]
 75%|███████▌  | 6/8 [00:22<00:07,  3.76s/it]
 88%|████████▊ | 7/8 [00:26<00:03,  3.71s/it]
100%|██████████| 8/8 [00:31<00:00,  4.34s/it]
100%|██████████| 8/8 [00:31<00:00,  3.99s/it]
/Users/albou/.pyenv/versions/3.12.13/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '`
| 04_i23d | i23d | succeeded | 46.0504 | None | 0.9756 | 17.0505 | 28.0243 | 50395 | 98792 | 10.5441 | 0.0 |
