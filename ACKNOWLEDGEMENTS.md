# Acknowledgements

Abstract3D builds on a number of strong upstream projects and research artifacts.

## Runtime Lineage

- `stabilityai/TripoSR` and the upstream TripoSR repository for the validated local reconstruction path
- `stepfun-ai/Step1X-3D` for the experimental geometry backend (Apache-2.0)
- `tencent/Hunyuan3D-2.1` for the license-gated shape backend (Tencent Hunyuan 3D 2.1 Community License; territory-restricted)
- `microsoft/TRELLIS.2` for the experimental official-only backend (MIT), whose required companion model `facebook/dinov3-vitl16-pretrain-lvd1689m` is distributed under Meta's DINOv3 License (gated access; attribution and acceptable-use terms apply). Where a distributed configuration includes DINOv3-derived components: Built with DINOv3.
- `abstractvision` for the composed text-to-image stage used by the validated `t23d` path
- `trimesh`, `matplotlib`, and `torchmcubes` for export and preview generation
- `xatlas`, `moderngl`, `pymeshlab`, `scipy`, and `OpenCV` for UV unwrapping, rasterization, mesh cleanup, and texture processing
- `rembg` for optional background removal in image preprocessing

## Research Candidates

The model catalog in [`docs/models.md`](docs/models.md) tracks permissive open-source candidates that informed the initial design direction, including TRELLIS, TRELLIS.2, Step1X-3D, InstantMesh, LGM, and Shap-E.

## AbstractFramework

Abstract3D is designed to compose with:

- `abstractcore` for capability discovery and multimodal output routing
- `abstractvision` for local image generation and editing
- `abstractruntime` artifact stores when a host runtime is present
