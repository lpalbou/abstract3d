#!/usr/bin/env python3
"""e22 refusal forensics probe: drive the EXACT production loop-view ladder
(generate_reference_views, identity conditioning, e22's seeds/env/recipe)
for a small angle set against the regenerated e22 pass-1 scaffold, and
persist every per-attempt gate verdict + pixels.

The e22 one-shot loop run refused with "no eligible conditioning views"
after ~80 minutes and persisted nothing (defect fixed separately); this
probe answers "which gate killed the draws, with what numbers" without
paying a full loop run.

Stages:
  A. regenerate the e22 pass-1 scaffold (2mv, front photo alone, seed 2025,
     50 steps, octree 512 — the exact e22 settings) unless --mesh points at
     an existing glb (then that mesh is the ladder's clay/pose authority);
     the scaffold is cached to the output dir and reused on reruns.
  B. run generate_reference_views for the probe angles with pose_gate=None
     (isolation: material gates only decide) but measure pose POST-HOC on
     every surviving pixel, so one generation pass yields both the material
     verdicts and the pose deltas.
  C. post-hoc numbers: per-attempt pose vs the scaffold AND vs the e20
     reference mesh (scaffold-noise comparison for the acceptance-gate
     calibration), plus the source photo's own lightness statistics (the
     0019 speculars recalibration inputs).

PRIVACY: generation is pinned to the local mlx-gen route exactly like
scripts/viewgen_bench.py — the backend binding is asserted before any
generation and `metadata.source == "mlx-gen"` is asserted after every call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "src")

E20_MESH = Path("out/laurent-bust-redo/e20_fixed_views/geometry.glb")
DEFAULT_SOURCE = Path("/tmp/laurent_front_clean4.png")

# e22's exact knobs (run command + backend defaults on mps).
E22_SEED = 2025
E22_LOOP_SEED = E22_SEED + 70_000  # _LOOP_VIEW_SEED_OFFSET
E22_PASS1 = dict(
    num_inference_steps=50,
    guidance_scale=5.0,
    octree_resolution=512,
    num_chunks=32768,
    volume_decoder_mode="adaptive",
    max_facenum=120000,
)


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()
                if k not in {"rgba", "image", "raw_bytes"}}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def load_source(path: Path):
    from PIL import Image

    from abstract3d.segmentation import clean_alpha_mask, remove_background_robust

    source = Image.open(path).convert("RGBA")
    if source.getchannel("A").getextrema()[0] < 255:
        return clean_alpha_mask(source)
    return remove_background_robust(source)


def regenerate_pass1(source, out_dir: Path, device: str):
    """The e22 pass-1 scaffold via the production _run_loop_pass1 (exact
    settings, exact canonicalization); cached as pass1_mesh.glb."""
    import trimesh

    cached = out_dir / "pass1_mesh.glb"
    if cached.exists():
        print(f"[stage A] reusing cached scaffold {cached}", flush=True)
        return trimesh.load(str(cached), force="mesh", process=False)

    from abstract3d.backends.hunyuan3d_runtime import Hunyuan3DShapeBackend
    from abstract3d.backends.triposr_runtime import _mesh_export_bytes

    backend = Hunyuan3DShapeBackend(owner=None)
    started = time.perf_counter()
    # NOTE (post-forensics): _run_loop_pass1 now pins the FLAGSHIP
    # checkpoint internally (e22 fix); the original e22 reproduction of
    # the 2mv single-tag draw lives in the git history of this probe +
    # /tmp/e22_probe artifacts.
    mesh0, record, seconds = backend._run_loop_pass1(
        source_image=source,
        device=device,
        dtype=None,
        seed=E22_SEED,
        **E22_PASS1,
    )
    print(f"[stage A] pass-1 scaffold drawn in {seconds:.0f}s "
          f"({record['vertex_count']} verts)", flush=True)
    cached.write_bytes(_mesh_export_bytes(mesh0, file_type="glb", viewer_frame=False))
    (out_dir / "pass1_record.json").write_text(json.dumps(record, indent=1))
    del backend
    return mesh0


class PinnedGenerator:
    """viewgen_bench-style local generator with per-call provenance
    (metadata.source assert + LoRA application counts on every attempt)."""

    def __init__(self) -> None:
        from abstract3d.image_composition import _abstractvision_capability

        self._cap = _abstractvision_capability(None)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, prompt: str, image: bytes, *, seed: int, **kwargs: Any):
        from abstractvision.vision_manager import VisionManager

        model = kwargs.get("model")
        binding = self._cap._resolve_backend_binding(provider="mlx-gen", model=model)
        backend = self._cap._activate_request_backend(binding)
        backend_name = type(backend).__name__
        if "mflux" not in backend_name.lower() and "mlx" not in backend_name.lower():
            raise RuntimeError(
                f"PRIVACY P0: resolved backend {backend_name!r} is not the "
                "local mlx-gen backend — refusing to generate.")
        call_kwargs: Dict[str, Any] = {"seed": int(seed)}
        if kwargs.get("steps"):
            call_kwargs["steps"] = int(kwargs["steps"])
        if kwargs.get("negative_prompt"):
            call_kwargs["negative_prompt"] = str(kwargs["negative_prompt"])
        if kwargs.get("lora_adapters"):
            call_kwargs["lora_adapters"] = [dict(x) for x in kwargs["lora_adapters"]]
        refs = kwargs.get("reference_images") or ()
        if refs:
            call_kwargs["extra"] = {"reference_images": [bytes(b) for b in refs]}
        vm = VisionManager(backend=backend)
        started = time.perf_counter()
        asset = vm.edit_image(str(prompt), image=bytes(image), **call_kwargs)
        seconds = time.perf_counter() - started
        metadata = dict(getattr(asset, "metadata", {}) or {})
        if str(metadata.get("source") or "") != "mlx-gen":
            raise RuntimeError(
                f"PRIVACY P0: generation metadata source={metadata.get('source')!r} "
                "is not mlx-gen — the image may have left the machine.")
        keep = {k: metadata.get(k) for k in (
            "source", "model", "seed", "steps", "reference_image_count",
            "edit_mode", "lora_applied_file_count") if k in metadata}
        self.calls.append({"seed": int(seed), "seconds": round(seconds, 1), **keep})
        print(f"[draw] seed={seed} steps={call_kwargs.get('steps')} "
              f"{seconds:.0f}s meta={keep}", flush=True)
        return {"data": bytes(asset.data), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None,
                        help="existing glb to use instead of regenerating pass 1")
    parser.add_argument("--angles", default="side_left:90,0",
                        help="semicolon-separated label:az,el entries")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--skip-e20-pose", action="store_true")
    parser.add_argument("--pass1-steps", type=int, default=None,
                        help="override pass-1 num_inference_steps (default: e22's 50)")
    parser.add_argument("--pass1-octree", type=int, default=None,
                        help="override pass-1 octree_resolution (default: e22's 512)")
    parser.add_argument("--pass1-only", action="store_true",
                        help="stop after the scaffold + its quality numbers")
    args = parser.parse_args()
    if args.pass1_steps is not None:
        E22_PASS1["num_inference_steps"] = int(args.pass1_steps)
    if args.pass1_octree is not None:
        E22_PASS1["octree_resolution"] = int(args.pass1_octree)

    import trimesh

    from abstract3d.captioning import caption_image
    from abstract3d.image_composition import resolve_image_generation_request
    from abstract3d.loop_conditioning import measure_view_azimuth
    from abstract3d.reference_generation import (
        generate_reference_views,
        parse_generation_angles,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Provider pinning exactly like production loop mode resolves it.
    request = resolve_image_generation_request(None)
    if str(request.get("provider") or "") != "mlx-gen":
        raise RuntimeError(
            f"PRIVACY P0: resolved provider {request.get('provider')!r} is not "
            "mlx-gen; export ABSTRACT3D_IMAGE_PROVIDER/MODEL/LORA_ADAPTERS "
            "exactly like the e22 run command.")
    print(f"[env] image request: { {k: str(v) for k, v in request.items()} }",
          flush=True)

    source = load_source(args.source)
    angles = list(parse_generation_angles(args.angles) or ())
    print(f"[env] angles: {angles}", flush=True)

    # Stage A — the ladder's mesh authority.
    if args.mesh is not None:
        mesh0 = trimesh.load(str(args.mesh), force="mesh", process=False)
        print(f"[stage A] using provided mesh {args.mesh}", flush=True)
    else:
        mesh0 = regenerate_pass1(source, args.output_dir, args.device)

    # Scaffold quality vs the photo it was reconstructed from: a scaffold
    # whose own front clay cannot register onto the source is the e22
    # failure class (measured: shredded 512/50 scaffold at IoU 0.415 vs
    # the healthy e20 mesh at 0.883).
    from abstract3d.reference_generation import (
        register_matte_to_clay,
        silhouette_iou,
    )
    from abstract3d.rendering import render_mesh_views

    front_clay = render_mesh_views(mesh0, size=768, azimuths=[0.0],
                                   elevation=0.0)[0].convert("RGBA")
    front_clay.save(args.output_dir / "pass1_front_clay.png")
    registered, reg_stats = register_matte_to_clay(source, front_clay)
    scaffold_quality = {
        "photo_vs_front_clay_iou": round(silhouette_iou(registered, front_clay), 4),
        "registration": reg_stats,
        "body_count": int(mesh0.body_count),
        "extents": [round(float(v), 3) for v in mesh0.extents],
        "pass1_settings": dict(E22_PASS1),
    }
    print(f"[stage A] scaffold quality: {json.dumps(scaffold_quality)}", flush=True)
    (args.output_dir / "scaffold_quality.json").write_text(
        json.dumps(scaffold_quality, indent=1))
    if args.pass1_only:
        return

    # Free the DiT pool before the i2i pool loads (production ordering).
    try:
        from abstract3d.backends.step1x_runtime import _release_mlx_generation_cache

        _release_mlx_generation_cache()
    except Exception:
        pass

    # e22's subject hint: the person-gate caption (prompt was empty).
    caption = caption_image(source)
    print(f"[env] caption: {caption!r}", flush=True)

    # Stage B — the production ladder, pose_gate=None (isolation), with the
    # progressive attempt sink and the pinned instrumented generator.
    generator = PinnedGenerator()
    attempts_log = args.output_dir / "attempts.jsonl"

    def on_attempt(row: Dict[str, Any]) -> None:
        with attempts_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    views, report = generate_reference_views(
        mesh0,
        source,
        image_generator=generator,
        angles=angles,
        subject_hint=caption,
        seed=E22_LOOP_SEED,
        image_request=dict(request),
        conditioning="identity",
        person_policy="proceed",
        pose_gate=None,
        on_attempt=on_attempt,
    )

    # Persist accepted + rejected pixels.
    saved: List[str] = []
    for view in views:
        path = args.output_dir / f"accepted_{view['label']}.png"
        view["rgba"].save(path)
        saved.append(str(path))
    for row in report.get("rejected_images", []):
        path = args.output_dir / f"rejected_{row['label']}_a{row['attempt']}.png"
        row["image"].save(path)
        if row.get("raw_bytes"):
            (args.output_dir / f"rejected_{row['label']}_a{row['attempt']}_raw.png"
             ).write_bytes(row["raw_bytes"])
        saved.append(str(path))

    # Stage C — post-hoc pose of every surviving pixel vs scaffold and e20.
    e20_mesh = None
    if not args.skip_e20_pose and E20_MESH.exists():
        e20_mesh = trimesh.load(str(E20_MESH), force="mesh", process=False)
    pose_rows: List[Dict[str, Any]] = []
    by_angle = {str(a[0]): float(a[1]) for a in angles}

    def pose_of(tag: str, label: str, image: Any) -> None:
        nominal = by_angle.get(label)
        if nominal is None:
            return
        row: Dict[str, Any] = {"view": tag, "label": label, "nominal": nominal}
        try:
            row["vs_scaffold"] = {
                k: v for k, v in measure_view_azimuth(image, mesh0, nominal).items()
                if k != "sweep"}
        except Exception as exc:
            row["vs_scaffold"] = {"error": str(exc)}
        if e20_mesh is not None:
            try:
                row["vs_e20"] = {
                    k: v for k, v in measure_view_azimuth(image, e20_mesh, nominal).items()
                    if k != "sweep"}
            except Exception as exc:
                row["vs_e20"] = {"error": str(exc)}
        pose_rows.append(row)
        print(f"[pose] {tag}: {json.dumps(row, default=str)}", flush=True)

    for view in views:
        pose_of(f"accepted_{view['label']}", str(view["label"]), view["rgba"])
    for row in report.get("rejected_images", []):
        pose_of(f"rejected_{row['label']}_a{row['attempt']}", str(row["label"]),
                row["image"])

    # Source lightness statistics (0019 speculars recalibration inputs).
    import numpy as np
    from skimage import color as skcolor

    rgba = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    lightness = skcolor.rgb2lab(rgba[:, :, :3])[:, :, 0][mask]
    source_stats = {
        "median_l": round(float(np.median(lightness)), 1),
        "p90_l": round(float(np.percentile(lightness, 90)), 1),
        "p99_l": round(float(np.percentile(lightness, 99)), 1),
        "p995_l": round(float(np.percentile(lightness, 99.5)), 1),
        "max_l": round(float(lightness.max()), 1),
    }

    out = {
        "request": {k: str(v) for k, v in request.items()},
        "caption": caption,
        "angles": [list(a) for a in angles],
        "accepted": report.get("accepted"),
        "rejected": report.get("rejected"),
        "report": _scrub(report),
        "pose_posthoc": pose_rows,
        "source_lightness": source_stats,
        "generator_calls": generator.calls,
        "saved": saved,
    }
    (args.output_dir / "probe_report.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"[done] accepted={report.get('accepted')} rejected={report.get('rejected')} "
          f"-> {args.output_dir / 'probe_report.json'}", flush=True)


if __name__ == "__main__":
    main()
