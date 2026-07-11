"""Command-line interface for Abstract3D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model_catalog import catalog_rows
from .scene3d_manager import Scene3DManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstract3d", description="Abstract3D local-first 3D generation")
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="List validated and researched model candidates.")
    catalog.add_argument("--validated-only", action="store_true")
    catalog.add_argument("--json", action="store_true")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", default="glb", choices=["glb", "obj", "zip"])
    common.add_argument("--output-dir", required=True)
    common.add_argument("--device", default="auto")
    common.add_argument("--mc-resolution", type=int, default=None)
    common.add_argument("--cleanup", default=None, choices=["presentation", "none"])
    common.add_argument("--texture-mode", default=None, choices=["vertex_color", "baked_basecolor"])
    common.add_argument("--texture-resolution", type=int, default=None)
    common.add_argument("--texture-completion", default=None, choices=["none", "mirror_symmetry", "auto"])
    common.add_argument("--texture-reference-image", action="append", default=[])
    common.add_argument("--texture-reference-angle", action="append", default=[])
    common.add_argument("--texture-reference-remove-background", dest="texture_reference_remove_background", action="store_true")
    common.add_argument("--no-texture-reference-remove-background", dest="texture_reference_remove_background", action="store_false")
    common.set_defaults(texture_reference_remove_background=None)
    common.add_argument("--texture-reference-generation", default=None,
                        choices=["auto", "on", "off"],
                        help="Synthesize unseen-angle reference photos from the mesh's own "
                             "clay renders when only one image is provided (hunyuan3d21; "
                             "default auto).")
    common.add_argument("--texture-reference-generation-angles", default=None,
                        help="Angles to synthesize: labels (back, side_left, top, bottom, ...) "
                             "or explicit 'label:azimuth,elevation' entries separated by ';' "
                             "(e.g. 'bottom:0,-75; back:180,0').")
    common.add_argument("--texture-reference-allow-person", dest="texture_reference_allow_person",
                        action="store_true", default=None,
                        help="Person-specific acknowledgment for reference generation: no gate "
                             "defends facial identity, so synthesizing views of a person "
                             "requires this explicit attestation (people are refused otherwise, "
                             "even with --texture-reference-generation on).")
    common.add_argument("--allow-degraded", dest="allow_degraded", action="store_true",
                        help="Exit 0 even when quality_verdict is degraded/failed (the verdict "
                             "and reasons are still printed to stderr and recorded in metadata).")
    common.add_argument("--num-inference-steps", type=int, default=None)
    common.add_argument("--guidance-scale", type=float, default=None)
    common.add_argument("--octree-resolution", type=int, default=None,
                        help="Shape-VAE octree resolution (hunyuan3d21/step1x): higher = denser mesh.")
    common.add_argument("--max-facenum", type=int, default=None,
                        help="Post-process face-count cap (hunyuan3d21/step1x).")
    common.add_argument("--chunk-size", type=int, default=None)
    common.add_argument("--model", default=None)
    common.add_argument("--model-subfolder", default=None)
    common.add_argument("--backend", default="abstract3d:triposr")

    i23d = sub.add_parser("i23d", parents=[common], help="Generate a 3D object from one image.")
    i23d.add_argument("image")
    i23d.add_argument("--prompt", default="")
    i23d.add_argument("--remove-background", action="store_true")

    t23d = sub.add_parser("t23d", parents=[common], help="Generate a 3D object from text via AbstractVision plus the selected local backend.")
    t23d.add_argument("prompt")
    t23d.add_argument("--image-provider", default=None)
    t23d.add_argument("--image-model", default=None)
    t23d.add_argument("--image-width", type=int, default=768)
    t23d.add_argument("--image-height", type=int, default=768)
    t23d.add_argument("--image-seed", type=int, default=None)

    validate = sub.add_parser("validate", help="Run a small proof suite and emit a contact sheet.")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--device", default="auto")
    validate.add_argument("--mc-resolution", type=int, default=None)
    validate.add_argument("--cleanup", default=None, choices=["presentation", "none"])
    validate.add_argument("--texture-mode", default=None, choices=["vertex_color", "baked_basecolor"])
    validate.add_argument("--texture-resolution", type=int, default=None)
    validate.add_argument("--texture-completion", default=None, choices=["none", "mirror_symmetry", "auto"])
    validate.add_argument("--image-provider", default=None)
    validate.add_argument("--image-model", default=None)
    validate.add_argument("--backend", default="abstract3d:triposr")
    validate.add_argument("--model", default=None)
    validate.add_argument("--model-subfolder", default=None)
    validate.add_argument("--num-inference-steps", type=int, default=None)
    validate.add_argument("--guidance-scale", type=float, default=None)
    validate.add_argument("--prompt", action="append", default=[])
    validate.add_argument("--image", action="append", default=[])
    return parser


def _print_catalog(args) -> int:
    rows = catalog_rows(validated_only=bool(args.validated_only))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    for row in rows:
        print(f"{row['model_id']} [{row['status']}]")
        print(
            f"  tasks={','.join(row['tasks'])} license={row['license']} "
            f"apple={row['apple_silicon']} footprint_gb={row['footprint_gb']}"
        )
        print(f"  notes={row['notes']}")
        print(f"  source={row['source_url']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "catalog":
        return _print_catalog(args)

    manager = Scene3DManager(backend_id=args.backend)
    if args.command in ("i23d", "t23d"):
        # Only options the user explicitly set are forwarded: backends now
        # REJECT unknown options instead of ignoring them, so the CLI must
        # not spray every flag at every backend (e.g. --guidance-scale has
        # no meaning on the feed-forward TripoSR path).
        options = {
            "format": args.format,
            "output_dir": args.output_dir,
            "device": args.device,
            "mc_resolution": args.mc_resolution,
            "cleanup": args.cleanup,
            "texture_mode": args.texture_mode,
            "texture_resolution": args.texture_resolution,
            "texture_completion": args.texture_completion,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "octree_resolution": args.octree_resolution,
            "max_facenum": args.max_facenum,
            "chunk_size": args.chunk_size,
            "model": args.model,
            "model_subfolder": args.model_subfolder,
            "texture_reference_generation": args.texture_reference_generation,
            "texture_reference_generation_angles": args.texture_reference_generation_angles,
            "texture_reference_allow_person": args.texture_reference_allow_person,
        }
        if args.command == "i23d":
            options.update(
                prompt=args.prompt,
                remove_background=True if args.remove_background else None,
            )
            if args.texture_reference_image:
                options["texture_reference_images"] = list(args.texture_reference_image)
                options["texture_reference_angles"] = list(args.texture_reference_angle or [])
            options["texture_reference_remove_background"] = args.texture_reference_remove_background
        else:
            options.update(
                image_provider=args.image_provider,
                image_model=args.image_model,
                image_width=args.image_width,
                image_height=args.image_height,
                image_seed=args.image_seed,
            )
        options = {key: value for key, value in options.items() if value is not None}
        if args.command == "i23d":
            result = manager.i23d(args.image, **options)
        else:
            result = manager.t23d(args.prompt, **options)
        metadata = result.get("metadata") or {}
        print(json.dumps(metadata, indent=2, sort_keys=True))
        # LOUD health contract: the measured car incident shipped a broken
        # texture with exit 0 and warnings buried in stdout JSON. Verdict
        # and reasons go to stderr; a non-healthy verdict exits 3 (argparse
        # owns 2) unless the caller opts into degraded artifacts.
        verdict = (metadata.get("quality_verdict") or {})
        verdict_name = str(verdict.get("verdict") or "healthy")
        if verdict_name != "healthy":
            print(
                f"quality_verdict: {verdict_name}: "
                + "; ".join(verdict.get("reasons") or []),
                file=sys.stderr,
            )
            for warning in metadata.get("postprocess_warnings") or []:
                print(f"warning: {warning}", file=sys.stderr)
            if not getattr(args, "allow_degraded", False):
                return 3
        return 0
    if args.command == "validate":
        prompts = list(args.prompt or []) or [
            "a ceramic teapot with a curved spout and matte glaze",
            "a mid-century lounge chair with walnut legs and woven fabric",
        ]
        images = [str(Path(path).expanduser().resolve()) for path in list(args.image or [])]
        validate_kwargs = {
            "prompts": prompts,
            "images": images,
            "output_dir": args.output_dir,
            "image_provider": args.image_provider,
            "image_model": args.image_model,
            "mc_resolution": args.mc_resolution,
            "device": args.device,
            "model": args.model,
            "model_subfolder": args.model_subfolder,
            "cleanup": args.cleanup,
            "texture_mode": args.texture_mode,
            "texture_resolution": args.texture_resolution,
            "texture_completion": args.texture_completion,
        }
        if args.num_inference_steps is not None:
            validate_kwargs["num_inference_steps"] = args.num_inference_steps
        if args.guidance_scale is not None:
            validate_kwargs["guidance_scale"] = args.guidance_scale
        summary = manager.validate_suite(**validate_kwargs)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m abstract3d.cli`
    import sys

    sys.exit(main())
