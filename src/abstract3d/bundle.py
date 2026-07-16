"""Public bundle API: load generated bundles and rebake their textures.

Every generated bundle deliberately persists `geometry.glb` in the CANONICAL
frame (Z-up, front +X) precisely so the texture can be redone later with new
or additional reference views — the entire zero-defect certification program
and the generated-reference-views proofs were produced through this path.
Until now that path only existed as ad-hoc scripts; this module makes it a
supported API (and the golden-bake regression harness runs through it, so
the canonical recipe below is executable and hash-verified).

Canonical rebake recipe (matches the certified proof assets):

- mesh: `geometry.glb` loaded with `process=False` (topology must not change)
- source view: `input.png` through robust background removal, azimuth 0
- reference views: RGBA photos with angles relative to the source viewpoint
- bake: `bake_projection_texture(..., projection_model="orthographic",
  texture_completion="auto")` at the bundle's texture resolution
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

BUNDLE_SCHEMA_VERSION = 1


@dataclass
class LoadedBundle:
    """A generated bundle on disk, resolved to its rebake inputs."""

    bundle_dir: Path
    metadata: Dict[str, Any]

    @property
    def geometry_path(self) -> Path:
        return self.bundle_dir / "geometry.glb"

    @property
    def input_path(self) -> Path:
        return self.bundle_dir / "input.png"

    @property
    def texture_path(self) -> Path:
        return self.bundle_dir / "texture.png"

    def geometry_mesh(self) -> Any:
        """The canonical-frame geometry mesh (safe for rebakes)."""

        import trimesh

        if not self.geometry_path.exists():
            raise FileNotFoundError(
                f"{self.geometry_path} is missing: this bundle cannot be rebaked. "
                "Bundles produced before the geometry.glb contract must be regenerated."
            )
        return trimesh.load(str(self.geometry_path), force="mesh", process=False)


def load_bundle(bundle_dir: Union[str, Path]) -> LoadedBundle:
    bundle_dir = Path(bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    return LoadedBundle(bundle_dir=bundle_dir, metadata=metadata)


def _resolve_angle(angle: Union[str, float, Tuple[float, float]]) -> Tuple[float, float, str]:
    """Resolve an angle spec (label, degrees, or (az, el)) to (az, el, label)."""

    from .backends.triposr_runtime import _tripo_parse_texture_reference_angle

    return _tripo_parse_texture_reference_angle(angle)


def prepare_observed_views(
    source_image: Any,
    references: Sequence[Mapping[str, Any]] = (),
    *,
    remove_source_background: bool = True,
) -> List[Dict[str, Any]]:
    """Build the `observed_views` list for `bake_projection_texture`.

    `source_image` is a path or PIL image for the azimuth-0 source photo.
    Each reference mapping carries `image` (path or PIL) plus `angle`
    (a label like "side_left", degrees, or an (azimuth, elevation) pair) and
    optional `remove_background` / `label` keys. Reference angles are
    interpreted relative to the source viewpoint, matching the bake contract.
    """

    from PIL import Image

    from .segmentation import remove_background_robust

    def to_image(value: Any) -> Any:
        if isinstance(value, (str, Path)):
            return Image.open(value)
        return value

    original = to_image(source_image)
    if remove_source_background:
        source = remove_background_robust(original)
    else:
        source = original.convert("RGBA")
    views: List[Dict[str, Any]] = [
        # `identity_image` carries the un-matted photo: multi-view bakes build
        # the identity correspondence against it (matted silhouettes snap the
        # registration into a different basin — see the fringe-repair stage).
        {"rgba": source, "azimuth_deg": 0.0, "elevation_deg": 0.0,
         "label": "front", "role": "source", "identity_image": original}
    ]
    for index, reference in enumerate(references, start=1):
        image = to_image(reference["image"])
        if reference.get("remove_background"):
            rgba = remove_background_robust(image)
        else:
            rgba = image.convert("RGBA")
        azimuth, elevation, label = _resolve_angle(reference.get("angle", 0.0))
        views.append(
            {
                "rgba": rgba,
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "label": str(reference.get("label") or label or f"reference_{index:02d}"),
                "role": "reference",
            }
        )
    return views


def rebake_bundle(
    bundle_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    references: Sequence[Mapping[str, Any]] = (),
    generate_references: str = "off",
    generation_angles: Optional[Sequence[Tuple[str, float, float]]] = None,
    reference_angle_planning: str = "auto",
    subject_hint: Optional[str] = None,
    allow_person_subjects: bool = False,
    owner: Any = None,
    seed: Optional[int] = None,
    texture_resolution: Optional[int] = None,
    texture_completion: str = "auto",
    projection_model: str = "orthographic",
    source_pose_override: Optional[Tuple[float, float]] = None,
    scarcity_rescue: str = "auto",
    compositing: str = "auto",
    write_outputs: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebake a bundle's texture from its persisted canonical geometry.

    Returns `(textured_mesh, stats)`. With `write_outputs`, writes a bundle
    revision into `output_dir` (default: `<bundle>/rebake/`): viewer-frame
    `scene.glb`, `texture.png`, `uv_preview.png`, and `metadata.json` with
    the bake stats, input hashes, and schema version.

    `reference_angle_planning` selects the generation lane's angles when
    `generation_angles` is not given: "static" is the historical
    back/sides/top set, "adaptive" plans angles from the source pose and
    the mesh's coverage geometry (`plan_reference_angles`), and "auto"
    (default) is adaptive exactly when the pose is non-canonical (an
    estimated pose, or an explicit non-zero `source_pose_override`) —
    the measured class where the static set leaves coverage on the table.
    The plan and its predicted gains are recorded in the metadata either
    way (persist-for-diagnosis).

    Caveat documented from the review: TripoSR bundles rebake without the
    triplane color prior (`base_color_fn` requires the resident model), so
    unseen texels fall back to projection fill; Hunyuan bundles rebake with
    full fidelity. The certified proof assets are Hunyuan bundles.
    """

    from . import texturing
    from .backends.triposr_runtime import _mesh_export_bytes

    if reference_angle_planning not in ("auto", "adaptive", "static"):
        raise ValueError(
            "reference_angle_planning must be one of: auto, adaptive, "
            f"static (got {reference_angle_planning!r})"
        )
    loaded = load_bundle(bundle_dir)
    mesh = loaded.geometry_mesh()
    resolution = int(
        texture_resolution
        or loaded.metadata.get("texture_resolution")
        or 2048
    )
    views = prepare_observed_views(loaded.input_path, references)
    bake_kwargs = dict(
        texture_resolution=resolution,
        texture_completion=texture_completion,
        projection_model=projection_model,
        source_pose_override=source_pose_override,
        scarcity_rescue=scarcity_rescue,
        compositing=compositing,
    )

    generation_report: Optional[Dict[str, Any]] = None
    # Baseline-first bake state (generation lane only): the refs-off A/B
    # baseline is baked BEFORE generation so its own stats carry the ONE
    # source-pose statement (the guard's estimate, or the caller's
    # override) that is then threaded to angle planning and generation
    # conditioning. The generation flow already baked this exact baseline
    # (after the candidate); reordering costs zero extra bakes and removes
    # the generation-vs-bake pose split.
    baseline_textured: Optional[Any] = None
    baseline_stats: Optional[Dict[str, Any]] = None
    if generate_references in ("auto", "on") and len(views) == 1:
        from .reference_generation import (
            DEFAULT_ANGLES,
            auto_generation_ready,
            generate_reference_views,
            plan_reference_angles,
        )

        # Rebake/pipeline parity (measured on the sports-car incident):
        # rebakes previously regenerated references with a HARDCODED seed
        # while the pipeline threads the generation seed — same bundle,
        # guaranteed different candidates, and per-angle verdicts flipped
        # across the strict line (back 11.21 accepted vs 24.12 rejected).
        # Default both the seed and the subject hint from the bundle's own
        # record so a plain rebake reproduces the pipeline's inputs.
        if seed is None:
            recorded_seed = loaded.metadata.get("seed")
            seed = int(recorded_seed) if recorded_seed is not None else 11
        if subject_hint is None:
            recorded_caption = (
                (loaded.metadata.get("texture_artifacts") or {})
                .get("reference_generation") or {}
            ).get("caption")
            subject_hint = str(recorded_caption) if recorded_caption else None

        ready, readiness_reason = auto_generation_ready(owner, subject_hint)
        if generate_references == "auto" and not ready:
            generation_report = {"skipped": readiness_reason}
        else:
            try:
                baseline_textured, baseline_stats = texturing.bake_projection_texture(
                    loaded.geometry_mesh(),
                    observed_views=[dict(view) for view in views],
                    **bake_kwargs)
            except Exception:
                # The identical bake below will reproduce this failure on
                # the established path; generation proceeds with the
                # caller's pose statement (override or declared (0,0)),
                # exactly as it did before the reorder.
                baseline_textured = None
                baseline_stats = None
            source_pose_record = dict(
                (baseline_stats or {}).get("source_pose") or {})
            source_pose_tuple = (
                float(source_pose_record.get("azimuth_deg")
                      if source_pose_record.get("azimuth_deg") is not None
                      else (source_pose_override or (0.0, 0.0))[0]),
                float(source_pose_record.get("elevation_deg")
                      if source_pose_record.get("elevation_deg") is not None
                      else (source_pose_override or (0.0, 0.0))[1]),
            )
            pose_committed = bool(
                source_pose_record.get("estimated")
                or (source_pose_override is not None
                    and tuple(source_pose_override) != (0.0, 0.0))
            )

            angle_plan: Optional[Dict[str, Any]] = None
            plan_error: Optional[str] = None
            try:
                angle_plan = plan_reference_angles(
                    mesh, source_pose_tuple, source_rgba=views[0]["rgba"])
            except Exception as exc:
                plan_error = f"{type(exc).__name__}: {exc}"

            adaptive_active = bool(
                reference_angle_planning == "adaptive"
                or (reference_angle_planning == "auto" and pose_committed)
            )
            if generation_angles:
                resolved_angles: Sequence[Tuple[str, float, float]] = (
                    generation_angles)
                angles_source = "explicit"
            elif adaptive_active and angle_plan is not None:
                resolved_angles = tuple(angle_plan.get("angles") or ())
                angles_source = "adaptive"
            else:
                resolved_angles = DEFAULT_ANGLES
                angles_source = "static"
            angle_plan_record: Dict[str, Any] = {
                "mode": reference_angle_planning,
                "angles_source": angles_source,
            }
            if angle_plan is not None:
                angle_plan_record.update(angle_plan)
                angle_plan_record.pop("angles", None)
            if plan_error:
                angle_plan_record["error"] = plan_error
            try:
                if not resolved_angles:
                    # An adaptive plan may legitimately select ZERO angles
                    # (the source already witnesses every plannable region
                    # above the min-gain floor).
                    generation_report = {
                        "angles": [], "accepted": 0, "rejected": 0,
                        "skipped": (
                            "angle plan selected no angles: every "
                            "candidate's predicted coverage gain is below "
                            "the min-gain floor"
                        ),
                    }
                else:
                    generated_views, generation_report = generate_reference_views(
                        mesh,
                        views[0]["rgba"],
                        owner=owner,
                        angles=resolved_angles,
                        subject_hint=subject_hint,
                        seed=int(seed),
                        # The witnessed-consistency gate (and the optional
                        # tint projection) is conditioned on the same pose
                        # statement the bake uses: the caller's override
                        # when given, else the baseline bake's own guard
                        # estimate. (0,0) — the historical assumption —
                        # is exactly what a declared-pose subject gets.
                        source_pose=source_pose_tuple,
                        # Synthesizing a person's face is refused in BOTH
                        # modes unless the caller passes the person-specific
                        # acknowledgment: "on" is texture-quality opt-in, not
                        # identity-synthesis consent (no gate defends
                        # identity).
                        person_policy=(
                            "proceed" if allow_person_subjects else "skip"
                        ),
                    )
                    views.extend(generated_views)
                    try:
                        from .backends.step1x_runtime import _release_mlx_generation_cache

                        _release_mlx_generation_cache()
                    except Exception:
                        pass
                if generation_report is not None:
                    generation_report["angle_plan"] = angle_plan_record
            except Exception as exc:
                if generate_references == "on":
                    raise
                generation_report = {
                    "skipped": f"generation failed: {type(exc).__name__}: {exc}",
                    "angle_plan": angle_plan_record,
                }

    started = time.perf_counter()
    # Snapshot the real views BEFORE the candidate bake: the bake registers
    # views in place (view["rgba"] is replaced by its aligned version), and
    # the A/B baseline must start from the same pristine inputs. The
    # generated snapshots survive a bake-acceptance refusal — without them
    # a refused candidate left NO pixels behind (measured: the maxvis
    # rebake recorded 3 accepted views in metadata while persisting none).
    generated_present = any(view.get("generated") for view in views)
    baseline_views = (
        [dict(view) for view in views if not view.get("generated")]
        if generated_present else None
    )
    generated_view_snapshots = [dict(view) for view in views if view.get("generated")]
    if baseline_stats is not None and not generated_present:
        # The generation lane already baked the refs-off baseline for its
        # pose estimate and no generated views were accepted: that IS the
        # single-photo bake (deterministic inputs; the parity canaries pin
        # this determinism).
        textured, stats = baseline_textured, baseline_stats
    else:
        textured, stats = texturing.bake_projection_texture(
            mesh, observed_views=views, **bake_kwargs)

    if generated_present and baseline_views:
        # WHOLE-BAKE ACCEPTANCE (adversarial round 2): per-view gates are
        # blind to composition-level failure — on the measured chair case
        # every shipped view strict-passed and the finished bake still
        # regressed below the no-references baseline (a handoff seam no
        # single view contains). Bake the baseline too and ship the
        # generated bake only if it does not regress it (see
        # bake_acceptance.evaluate_generated_bake).
        from .bake_acceptance import evaluate_generated_bake

        if baseline_stats is None or baseline_textured is None:
            baseline_mesh = loaded.geometry_mesh()
            baseline_textured, baseline_stats = texturing.bake_projection_texture(
                baseline_mesh, observed_views=baseline_views, **bake_kwargs)
        # The gate resolves the fidelity pose from the baseline bake's own
        # stats (measuring at a hardcoded (0,0) on a pose-estimated
        # subject charges ~9 dE of pure pose error to BOTH sides:
        # measured +0.88 true regression read as +4.03 on the az-17.5
        # car). An explicit override remains the external capture fact.
        verdict = evaluate_generated_bake(
            baseline_textured,
            textured,
            source_rgba=baseline_views[0]["rgba"],
            source_pose=source_pose_override,
            baseline_stats=baseline_stats,
            candidate_stats=stats,
        )
        if generation_report is None:
            generation_report = {}
        generation_report["bake_acceptance"] = verdict
        # The gate computes both sides' single-view sanity floors and the
        # baseline regime once (metrics["baseline_regime"]); the shipped
        # side's verdict is threaded from there instead of recomputed.
        gate_regime = verdict["metrics"].get("baseline_regime") or {}
        from .bake_acceptance import evaluate_single_view_bake

        if not verdict["accepted"]:
            # The generated bake regressed the baseline: ship the baseline.
            # The generated images and the verdict stay in metadata as the
            # honest record of what was tried and why it was refused.
            textured, stats = baseline_textured, baseline_stats
            views = baseline_views
            # The shipped baseline is a single-photo bake that never met
            # the A/B machinery; record the same sanity floors the
            # no-references path records, so a broken-on-its-own baseline
            # (pose collapse: measured coverage 0.058 shipped "healthy")
            # is loud in the metadata instead of silent.
            stats["single_view_sanity"] = (
                gate_regime.get("baseline_sanity")
                or evaluate_single_view_bake(stats))
        elif gate_regime.get("regime") == "catastrophic":
            # Catastrophic-regime acceptance: the candidate replaced a
            # baseline whose source registration collapsed (the measured
            # x-wing incident class). Its own floors verdict — source-view
            # rows inherited from the broken registration until the pose
            # lane heals them — is the honest health record of what ships.
            stats["single_view_sanity"] = (
                gate_regime.get("candidate_sanity")
                or evaluate_single_view_bake(stats))
    bake_seconds = round(time.perf_counter() - started, 3)

    if write_outputs:
        out = Path(output_dir) if output_dir else (loaded.bundle_dir / "rebake")
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.glb").write_bytes(_mesh_export_bytes(textured, file_type="glb"))
        stats["texture_image"].save(out / "texture.png")
        uv_preview = stats.get("uv_preview")
        if uv_preview is not None:
            uv_preview.save(out / "uv_preview.png")
        texture_md5 = hashlib.md5((out / "texture.png").read_bytes()).hexdigest()
        clean_stats = {
            k: v
            for k, v in stats.items()
            if k not in ("texture_image", "uv_preview", "vmapping", "indices", "uvs")
        }
        # Copy the source photo so QA harnesses (scripts/texture_qa.py) can
        # reconstruct per-view visibility on the rebake output.
        if loaded.input_path.exists():
            shutil_src = loaded.input_path.read_bytes()
            (out / "input.png").write_bytes(shutil_src)
        metadata = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": "rebake",
            "source_bundle": str(loaded.bundle_dir),
            "texture_resolution": resolution,
            "texture_completion": texture_completion,
            "projection_model": projection_model,
            "source_pose_override": source_pose_override,
            "scarcity_rescue": scarcity_rescue,
            "compositing": compositing,
            "reference_count": len(views) - 1,
            "bake_seconds": bake_seconds,
            "texture_png_md5": texture_md5,
            "stats": _json_safe(clean_stats),
            # QA-harness compatibility: region reconstruction reads these
            # keys from `texture_artifacts` (or the metadata root).
            "texture_artifacts": _json_safe({
                "observed_view_stats": clean_stats.get("observed_view_stats"),
                "source_pose": clean_stats.get("source_pose"),
                "source_registration": clean_stats.get("source_registration"),
                "camera_distance": clean_stats.get("camera_distance"),
                "observed_coverage_ratio": clean_stats.get("observed_coverage_ratio"),
                "texture_completion": clean_stats.get("texture_completion"),
                "symmetry_completion": clean_stats.get("symmetry_completion"),
            }),
        }
        if generation_report is not None:
            rejected_images = generation_report.pop("rejected_images", None)
            metadata["generated_references"] = _json_safe(generation_report)
            # Persist from the pre-bake snapshots, not the shipped view
            # list: after a bake-acceptance refusal the shipped list is
            # the baseline and the candidate's pixels would vanish.
            for view in generated_view_snapshots:
                view["rgba"].save(out / f"generated_{view['label']}.png")
                clay = view.get("clay_render")
                if clay is not None:
                    clay.save(out / f"generated_{view['label']}_clay.png")
            if rejected_images:
                # Persist-for-diagnosis: the exact (downscaled) pixels the
                # gates rejected, budget-capped — a rejected class must be
                # diagnosable without a rerun (measured: the gray-car
                # class cost a full regeneration to even see).
                rejected_dir = out / "rejected_refs"
                rejected_dir.mkdir(exist_ok=True)
                budget = 2 * 1024 * 1024
                for row in rejected_images:
                    if budget <= 0:
                        break
                    path = rejected_dir / f"{row['label']}_a{row['attempt']}.webp"
                    try:
                        row["image"].convert("RGB").save(
                            path, format="WEBP", quality=80)
                        budget -= path.stat().st_size
                    except Exception:
                        break
        (out / "metadata.json").write_text(json.dumps(metadata, indent=1, default=str))
    return textured, stats


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of bake stats to JSON-serializable values."""

    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
