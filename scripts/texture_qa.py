#!/usr/bin/env python
"""Close-range texture QA: material truth, viewer-truth brightness, and
close-zoom texture-quality gates for an exported scene bundle.

Usage:
    python scripts/texture_qa.py BUNDLE_DIR [--out DIR] [--size 896]
                                 [--calibrate-photos]

Complements the mid-range render battery (per-view feature detectors) with
the acceptance surface a close-inspecting PBR viewer (MeshVault) actually
exercises:

  MATERIAL GATES   parse the raw GLB JSON and the OBJ MTL: the shipped
                   material must be full-brightness (baseColorFactor 1),
                   non-metal (metallicFactor 0 — the glTF DEFAULT IS 1.0,
                   so an absent factor fails), sanely rough, and carry the
                   texture. A 0.4 factor + default metallic renders ~60%
                   darker and metal-dark in any spec viewer regardless of
                   how good the baked texture is.
  VIEWER TRUTH     renders multiply the texture by the exported
                   baseColorFactor, so the harness sees what the viewer
                   shows, not what the factor-ignoring repo preview
                   renderer flatters. Gate: foreground luminance ratio
                   against the input photo.
  CLOSE-ZOOM GATES 2x/4x crops at auto-derived defect-prone locations
                   (highest-curvature concavities via the mesh shape
                   operator; largest synthesized-fill regions via the
                   observed-mask complement reconstructed from bundle
                   metadata): facet-block detector (near-constant polygonal
                   cells with straight texel-scale boundaries — the
                   nearest-vertex fill signature), spurious dark-smear
                   detector near concavities.
  TEXEL GATES      seam steps across view-boundary bands (masked-mean
                   Lab deltaE, allowance calibrated per bundle on the input
                   photo, which must pass), fill-character
                   (gradient-energy ratio fill vs observed, gate >= 0.5),
                   facet fraction of the fill region.
  ARTIFACT BATTERY the measured artifact-class detectors of
                   `abstract3d.artifact_gates` on the standard 8-view
                   turnaround: foreign pale blotches (image-in-image
                   stamps, white rim splashes), photo-background
                   contamination, broad desaturating wash (baked
                   speculars / clear-coat smears), translucent
                   patch-block rectangle cells, mid-band dark patchwork,
                   plus the stats-based fill-cap mottle-risk and
                   registration-floor checks. STANDALONE measurements
                   have no baseline to difference against, and their
                   good-corpus margins (1.14-1.8x) sit below the
                   project's zero-false-fire voting bar, so the battery
                   WARNS and records - it never fails the run (the
                   voting form lives in the whole-bake A/B gate).

Exit code 0 = all gates pass. Evidence crops land in OUT/evidence/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture_qa_render import (  # noqa: E402
    Bundle,
    Probe,
    TexelMaps,
    ViewSpec,
    ViewerTruthRenderer,
    build_texel_maps,
    concavity_probes,
    load_bundle,
    parse_glb,
    project_visibility,
)


@dataclass
class Gate:
    name: str
    passed: bool
    measured: str
    requirement: str
    evidence: Optional[str] = None


@dataclass
class Thresholds:
    """Gate values. Photo-independent values were chosen so the input photos
    themselves pass every detector (verified by --calibrate-photos); the
    seam allowance is re-derived per bundle from its own input photo."""

    metallic_max: float = 0.05
    base_color_factor_min: float = 0.99
    roughness_range: Tuple[float, float] = (0.4, 1.0)
    texture_min_px: int = 512
    mtl_kd_min: float = 0.99
    mtl_ks_max: float = 0.25
    # viewer-truth foreground luminance vs input photo foreground
    brightness_ratio_range: Tuple[float, float] = (0.72, 1.40)
    # fill-character: mean gradient energy of fill / observed (task gate),
    # with an upper bound so synthesized "detail" louder than the photo
    # texture (noise injection) cannot pass as material
    fill_energy_ratio_min: float = 0.50
    fill_energy_ratio_max: float = 2.50
    # facet cells: near-flat local std (uint8 levels), cell area band, and
    # the field requirement (>= field_min_cells straight-edged cells) that
    # separates a faceted fill from a legitimately smooth region
    facet_flat_std: float = 1.6
    facet_cell_min_px: int = 24
    facet_cell_max_frac: float = 0.03
    facet_field_min_cells: int = 4
    # a facet FIELD tiles the region with flat cells (measured: true
    # nearest-vertex honeycombs are >= 0.81 flat; photo foregrounds 0.24;
    # mottled/smoothed fills 0.05-0.44)
    facet_field_flat_min: float = 0.60
    # synthesized-region cellular fraction relative to the observed region
    # of the SAME texture (same photo source, same texel scale)
    cellular_ratio_max: float = 2.0
    cellular_floor: float = 0.08
    facet_fields_4x_max: int = 0
    # seam step allowance = max(photo p99 deltaE * margin, absolute floor),
    # measured between same-material-family sides only: side pairs whose
    # material clusters differ by more than seam_material_gap are content
    # edges (hair|skin), not candidate seams
    seam_margin: float = 1.35
    seam_floor: float = 10.0
    seam_material_gap: float = 35.0
    # dark smear detector (close zoom): feature-dark fragments inside
    # synthesized regions, judged against the local windowed context
    smear_dark_ratio: float = 0.45
    smear_min_area_frac: float = 0.0008
    smear_max_area_frac: float = 0.02
    smear_blocks_4x_max: int = 0


# ---------------------------------------------------------------------------
# material gates
# ---------------------------------------------------------------------------

def material_gates(bundle: Bundle, thr: Thresholds) -> List[Gate]:
    gates: List[Gate] = []
    materials = bundle.gltf.get("materials") or []
    if not materials:
        return [Gate("material.glb", False, "no materials", "PBR material present")]
    pbr = materials[0].get("pbrMetallicRoughness") or {}

    factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    ok = all(c >= thr.base_color_factor_min for c in factor[:3]) and factor[3] >= 0.99
    gates.append(Gate(
        "material.glb.base_color_factor", ok,
        f"{[round(float(c), 4) for c in factor]}",
        f"== [1,1,1,1] (viewer multiplies texture by this; "
        f"{factor[0]:.2f} ships {(1 - factor[0]) * 100:.0f}% darker)"))

    metallic = pbr.get("metallicFactor")
    measured = "ABSENT (glTF default 1.0 = full metal)" if metallic is None else f"{metallic:.4f}"
    gates.append(Gate(
        "material.glb.metallic_factor",
        metallic is not None and float(metallic) <= thr.metallic_max,
        measured, f"present and <= {thr.metallic_max} (non-metal diffuse)"))

    roughness = float(pbr.get("roughnessFactor", 1.0))
    lo, hi = thr.roughness_range
    gates.append(Gate("material.glb.roughness_factor", lo <= roughness <= hi,
                      f"{roughness:.4f}", f"in [{lo}, {hi}]"))

    has_texture = pbr.get("baseColorTexture") is not None
    size = bundle.texture.size
    gates.append(Gate(
        "material.glb.base_color_texture",
        has_texture and min(size) >= thr.texture_min_px,
        f"present={has_texture} size={size[0]}x{size[1]}",
        f"texture present, >= {thr.texture_min_px}px"))

    sidecar = bundle.directory / "texture.png"
    if sidecar.exists():
        side = np.asarray(Image.open(sidecar).convert("RGB"), dtype=np.int16)
        glb_tex = bundle.texture_array.astype(np.int16)
        if side.shape == glb_tex.shape:
            diff = float(np.abs(side - glb_tex).mean())
            gates.append(Gate("material.glb.texture_matches_sidecar", diff <= 2.0,
                              f"mean|diff|={diff:.2f}", "GLB texture == texture.png"))
        else:
            gates.append(Gate("material.glb.texture_matches_sidecar", False,
                              f"GLB {glb_tex.shape} vs sidecar {side.shape}",
                              "same resolution"))

    mtl_path = bundle.directory / "scene.mtl"
    if mtl_path.exists():
        values: Dict[str, List[float]] = {}
        has_map = False
        for line in mtl_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] in {"Kd", "Ks", "Ka"}:
                values[parts[0]] = [float(v) for v in parts[1:4]]
            elif parts[0] == "map_Kd":
                has_map = True
        kd = values.get("Kd", [1.0, 1.0, 1.0])
        gates.append(Gate(
            "material.mtl.kd", min(kd) >= thr.mtl_kd_min and has_map,
            f"Kd={kd} map_Kd={has_map}",
            f"Kd == 1 (OBJ viewers multiply map_Kd by Kd) and map_Kd present"))
        ks = values.get("Ks", [0.0, 0.0, 0.0])
        gates.append(Gate("material.mtl.ks", max(ks) <= thr.mtl_ks_max,
                          f"Ks={ks}", f"<= {thr.mtl_ks_max} (no gray specular sheen)"))
    return gates


# ---------------------------------------------------------------------------
# view specs from metadata + region split
# ---------------------------------------------------------------------------

def bundle_view_specs(bundle: Bundle) -> Tuple[List[ViewSpec], dict]:
    """Reconstruct the bake's projection cameras from bundle metadata.

    Projection model: bakes that estimated the source pose photometrically
    (gradient_ncc) or with camera_distance 3.0 ran the canonical
    orthographic path; older bundles carry a fitted perspective distance.
    The estimated source pose overrides the front view-stat azimuth when
    they disagree (the projector shot from the estimated pose)."""
    _ = bundle.mesh  # loaded lazily by trimesh; force materialization
    ta = bundle.metadata.get("texture_artifacts", bundle.metadata) or {}
    stats = ta.get("observed_view_stats") or []
    source_pose = ta.get("source_pose") or {}
    distance = float(ta.get("camera_distance", 1.9))
    ortho = (str(source_pose.get("method", "")) == "gradient_ncc"
             or abs(distance - 3.0) < 1e-6)

    front_alpha: Optional[np.ndarray] = None
    photo_path = bundle.directory / "input.png"
    if photo_path.exists():
        photo = Image.open(photo_path)
        alpha_mask: Optional[np.ndarray] = None
        if photo.mode == "RGBA":
            alpha_mask = np.asarray(photo)[:, :, 3] > 8
        else:
            # The bake mattes RGB photos before projecting
            # (`remove_background_robust`), so texels whose rays land on
            # background pixels were never painted. Reconstructing
            # visibility WITHOUT the matte counted those texels as
            # observed (measured on the starship: qa coverage 0.261 vs
            # bake truth 0.177) and misattributed synthesized fill to the
            # observed region in every downstream gate.
            mask, method = photo_foreground(photo, cache_key=str(photo_path))
            if method == "matte_robust":
                alpha_mask = mask
        if alpha_mask is not None:
            rgba = np.dstack([
                np.asarray(photo.convert("RGB")),
                np.where(alpha_mask, 255, 0).astype(np.uint8),
            ])
            photo = Image.fromarray(rgba, mode="RGBA")
            if ortho:  # the bake projected the canonically recentered photo
                from abstract3d.texturing import recenter_to_canonical_frame

                # Projector-frame registration (mesh_bbox_center bakes
                # paste the photo bbox at the mesh's projected bbox
                # center): reconstruct visibility from the same frame the
                # bake projected, or region attribution shifts by the
                # recorded offset (54 px on the starship source pose).
                registration = ta.get("source_registration") or {}
                center = None
                if registration.get("method") == "mesh_bbox_center":
                    frame_size = float(registration.get("frame_size", 1024))
                    center = (
                        frame_size / 2.0 + float(registration.get("frame_center_dx_px", 0.0)),
                        frame_size / 2.0 + float(registration.get("frame_center_dy_px", 0.0)),
                    )
                photo = recenter_to_canonical_frame(
                    photo, border_ratio=0.15, center_px=center)
            front_alpha = np.asarray(photo.convert("RGBA"))[:, :, 3] / 255.0

    specs: List[ViewSpec] = []
    for i, vs in enumerate(stats):
        azimuth = float(vs.get("azimuth_deg", 0.0))
        elevation = float(vs.get("elevation_deg", 0.0))
        if i == 0 and source_pose.get("estimated") and abs(azimuth) < 1e-6:
            azimuth = float(source_pose.get("azimuth_deg", azimuth))
            elevation = float(source_pose.get("elevation_deg", elevation))
        specs.append(ViewSpec(
            label=str(vs.get("label", f"view_{i}")),
            azimuth_deg=azimuth, elevation_deg=elevation,
            camera_distance=distance,
            projection_model="orthographic" if ortho else "perspective",
            alpha=front_alpha if i == 0 else None))
    if ortho and specs:
        from abstract3d.texturing import canonical_ortho_half_extent

        for spec in specs:
            spec.ortho_half_extent = canonical_ortho_half_extent(
                bundle.mesh, azimuth_deg=spec.azimuth_deg,
                elevation_deg=spec.elevation_deg, border_ratio=0.15)
    info = {
        "projection_model": "orthographic" if ortho else "perspective",
        "camera_distance": distance,
        "declared_view_stats": stats,
        "symmetry": ta.get("symmetry_completion") or {},
        "unseen_fill_mode": ta.get("unseen_fill_mode"),
        "observed_coverage_ratio": ta.get("observed_coverage_ratio"),
    }
    return specs, info


@dataclass
class Regions:
    """Texel-space region split derived from reconstructed visibility."""
    observed: np.ndarray           # union of per-view visibility
    symmetry: np.ndarray           # fill whose mirror lands on observed texels
    fill: np.ndarray               # synthesized (neither observed nor mirrored)
    per_view: Dict[str, np.ndarray]
    reconciliation: List[dict]     # per-view coverage vs metadata claim


def split_regions(maps: TexelMaps, specs: Sequence[ViewSpec], info: dict) -> Regions:
    per_view: Dict[str, np.ndarray] = {}
    reconciliation: List[dict] = []
    surface_count = max(int(maps.surface.sum()), 1)
    for spec, declared in zip(specs, info.get("declared_view_stats", [])):
        visible = project_visibility(maps, spec)
        per_view[spec.label] = visible
        claimed = float(declared.get("coverage_ratio", 0.0) or 0.0)
        reconciliation.append({
            "label": spec.label,
            "azimuth_deg": spec.azimuth_deg,
            "reconstructed_coverage": round(visible.sum() / surface_count, 4),
            "metadata_coverage": claimed,
        })
    observed = np.zeros_like(maps.surface)
    for visible in per_view.values():
        observed |= visible
    observed &= maps.surface

    unfilled = maps.surface & ~observed
    symmetry = np.zeros_like(observed)
    if (info.get("symmetry") or {}).get("applied") and unfilled.any() and observed.any():
        from scipy.spatial import cKDTree

        obs_points = maps.positions[observed]
        if len(obs_points) > 400_000:  # tolerance 1.5% of diagonal: subsample is safe
            obs_points = obs_points[:: len(obs_points) // 400_000 + 1]
        tree = cKDTree(obs_points)
        mirrored = maps.positions[unfilled].copy()
        mirrored[:, 1] *= -1.0  # bake mirror axis: world y (left-right)
        distances, _ = tree.query(mirrored, k=1, workers=-1)
        near = distances <= 0.015 * maps.diagonal
        rows, cols = np.nonzero(unfilled)
        symmetry[rows[near], cols[near]] = True
    fill = unfilled & ~symmetry
    return Regions(observed, symmetry, fill, per_view, reconciliation)


# ---------------------------------------------------------------------------
# photo-side foreground (the harness's "subject pixels")
# ---------------------------------------------------------------------------

_PHOTO_FOREGROUND_CACHE: Dict[str, Tuple[np.ndarray, str]] = {}


def photo_foreground(photo: Image.Image, *, cache_key: Optional[str] = None
                     ) -> Tuple[np.ndarray, str]:
    """Boolean subject mask for an input photo + the method that produced it.

    Every photo-side reference in this harness (brightness, seam allowance,
    photo calibration) means "the SUBJECT the bake textured", so the mask
    must match what the bake actually consumed:

    1. RGBA photos carry their matte: alpha > 8 (unchanged behavior).
    2. RGB photos are matted with the SAME `remove_background_robust` the
       bake pipeline applies before projecting. The previous "non-white"
       heuristic (any channel > 18 from 255) silently measured the
       BACKGROUND on unmatted photos with non-white backdrops — measured
       on the owl proof photo (light-gray studio backdrop ~205 luminance):
       the heuristic classified 100% of the frame as foreground and the
       brightness reference became the backdrop's 203 instead of the
       subject's 129, failing the gate at 0.567 on a bake whose subject
       tone was in range. The gate exists to measure albedo fidelity of
       the textured subject, not backdrop bias.
    3. If the matte model is unavailable (offline host), fall back to the
       non-white heuristic so the harness still runs; the method string
       ("heuristic_nonwhite") makes the degraded reference visible in
       results.json.
    """
    key = cache_key or getattr(photo, "filename", None) or str(id(photo))
    if key in _PHOTO_FOREGROUND_CACHE:
        return _PHOTO_FOREGROUND_CACHE[key]
    if photo.mode == "RGBA":
        mask = np.asarray(photo)[:, :, 3] > 8
        method = "alpha"
    else:
        mask = None
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from abstract3d.segmentation import remove_background_robust

            matted = remove_background_robust(photo)
            alpha = np.asarray(matted)[:, :, 3] > 8
            # A degenerate matte (empty or near-full frame) means the model
            # failed on this input; the heuristic is then the lesser evil.
            if 0.005 < float(alpha.mean()) < 0.98:
                mask, method = alpha, "matte_robust"
        except Exception:
            mask = None
        if mask is None:
            rgb = np.asarray(photo.convert("RGB"))
            mask = np.abs(rgb.astype(np.int16) - 255).max(axis=2) > 18
            method = "heuristic_nonwhite"
    _PHOTO_FOREGROUND_CACHE[key] = (mask, method)
    return mask, method


# ---------------------------------------------------------------------------
# texel-space detectors
# ---------------------------------------------------------------------------

def _luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1),
                                                    np.uint8)).astype(bool)


def fill_character(rgb: np.ndarray, regions: Regions) -> dict:
    """Gradient-energy ratio of SYNTHESIZED texels (mirror-completed +
    fill) vs observed: does completion carry material detail comparable to
    projected photo texture, or flat mush? The gate covers the whole
    unobserved set so improvements cannot be gamed by relabeling texels
    between mirror and fill; per-subset energies stay as diagnostics.
    Gradients are measured strictly inside each region (eroded 2px) so
    region borders cannot inflate the fill energy."""
    lum = _luminance(rgb)
    gx = cv2.Scharr(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lum, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy)

    def region_energy(mask: np.ndarray) -> Optional[float]:
        inner = _erode(mask, 2)
        if inner.sum() < 500:
            return None
        return float(magnitude[inner].mean())

    unobserved = regions.fill | regions.symmetry
    e_obs = region_energy(regions.observed)
    e_unobs = region_energy(unobserved)
    ratio = (e_unobs / e_obs) if (e_obs and e_unobs is not None and e_obs > 1e-6) else None

    # coarse-scale diagnostic: 4x downsample kills texel noise, so a fill
    # that "passes" only via injected high-frequency noise shows a low
    # coarse ratio while a real material completion keeps both scales up
    quarter = cv2.resize(rgb, (rgb.shape[1] // 4, rgb.shape[0] // 4),
                         interpolation=cv2.INTER_AREA)
    lum_q = _luminance(quarter)
    mag_q = np.sqrt(cv2.Scharr(lum_q, cv2.CV_32F, 1, 0) ** 2
                    + cv2.Scharr(lum_q, cv2.CV_32F, 0, 1) ** 2)

    def region_energy_coarse(mask: np.ndarray) -> Optional[float]:
        small = cv2.resize(mask.astype(np.uint8), (lum_q.shape[1], lum_q.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        inner = _erode(small, 1)
        if inner.sum() < 200:
            return None
        return float(mag_q[inner].mean())

    eq_obs = region_energy_coarse(regions.observed)
    eq_unobs = region_energy_coarse(unobserved)
    coarse_ratio = (eq_unobs / eq_obs) if (eq_obs and eq_unobs is not None
                                           and eq_obs > 1e-6) else None
    return {
        "observed_energy": e_obs,
        "unobserved_energy": e_unobs,
        "fill_energy": region_energy(regions.fill),
        "symmetry_energy": region_energy(regions.symmetry),
        "fill_to_observed_ratio": ratio,
        "coarse_ratio": coarse_ratio,
        "unobserved_texels": int(unobserved.sum()),
        "fill_texels": int(regions.fill.sum()),
        "observed_texels": int(regions.observed.sum()),
    }


def detect_facet_blocks(rgb: np.ndarray, region: np.ndarray, thr: Thresholds) -> dict:
    """Faceted-fill detector: a FIELD of small near-constant cells with
    straight polygonal boundaries — the signature of nearest-vertex /
    harmonic fill (Voronoi-flat patches). Discriminators, calibrated so the
    input photos never fire:

    * cell area band: facet cells are mesh-vertex-scale, not the large
      organic flat areas a smooth photo cheek produces;
    * straight boundaries: approxPolyDP fits the contour with few segments
      and high solidity;
    * multiplicity: a facet defect is a honeycomb, so a verdict requires
      >= `facet_field_min_cells` cells (isolated flat blobs are not fields);
    * tiling: a field verdict additionally requires the region to be mostly
      near-flat (`facet_field_flat_min`) — true honeycombs measure >= 0.81
      flat fraction while mottled or smoothed fills with incidental flat
      islands measure <= 0.44 (photos: 0.24);
    * components touching the image border are excluded (crop edges
      straighten organically shaped regions).

    Returns cells, `cellular_fraction` (cell px / region px, area band
    applied without the straightness requirement — the scale-free statistic
    used for the fill-vs-observed comparison), and the field verdict."""
    inner = _erode(region, 2)
    if inner.sum() < 500:
        return {"blocks": [], "cellular_fraction": 0.0, "flat_fraction": 0.0,
                "facet_field": False, "region_px": int(inner.sum())}
    h, w = inner.shape
    # CLAHE before the flatness analysis: in dark regions (hair underside)
    # facet cells differ by 1-2 gray levels and would otherwise merge into
    # one giant flat component that the cell-area band exempts. Contrast
    # normalization makes cell boundaries measurable exactly where a viewer
    # perceives them, and amplifies photo noise, which protects real photo
    # texture from reading as flat.
    lum = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).astype(np.float32)
    mean = cv2.boxFilter(lum, -1, (5, 5))
    sq_mean = cv2.boxFilter(lum * lum, -1, (5, 5))
    local_std = np.sqrt(np.clip(sq_mean - mean * mean, 0.0, None))
    flat = (local_std < thr.facet_flat_std) & inner
    area_cap = max(int(thr.facet_cell_max_frac * inner.sum()), thr.facet_cell_min_px * 4)

    blocks: List[dict] = []
    flat_px = 0
    cell_px = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        flat_px += area
        if not (thr.facet_cell_min_px <= area <= area_cap):
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if x0 <= 1 or y0 <= 1 or x0 + bw >= w - 1 or y0 + bh >= h - 1:
            continue
        cell_px += area
        if len(blocks) >= 400:
            continue
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * max(perimeter, 1.0), True)
        hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
        solidity = cv2.contourArea(contour) / hull_area
        if len(approx) <= 8 and solidity >= 0.55:
            blocks.append({"bbox": (x0, y0, x0 + bw, y0 + bh), "area": area,
                           "vertices": int(len(approx)),
                           "solidity": round(float(solidity), 3)})
    flat_fraction = float(flat_px) / float(inner.sum())
    return {
        "blocks": blocks,
        "cellular_fraction": float(cell_px) / float(inner.sum()),
        "flat_fraction": flat_fraction,
        "facet_field": (len(blocks) >= thr.facet_field_min_cells
                        and flat_fraction >= thr.facet_field_flat_min),
        "region_px": int(inner.sum()),
    }


def _masked_side_means(lab: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    m = mask.astype(np.float32)
    smoothed = [cv2.boxFilter(lab[:, :, c] * m, -1, (window, window)) for c in range(3)]
    weight = cv2.boxFilter(m, -1, (window, window))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.where(weight > 0.05, s / np.maximum(weight, 1e-6), np.nan)
                         for s in smoothed], axis=2)


def material_clusters(rgb: np.ndarray, mask: np.ndarray, k: int = 4) -> np.ndarray:
    """k-means Lab centers of the region's materials (skin/hair/hull shades).
    Seam measurement only compares boundary sides whose clusters are CLOSE
    (same material family): a skin|hair transition is content, a skin|skin
    tone step is a candidate seam."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    pixels = lab[mask]
    if len(pixels) < 10 * k:
        return np.zeros((1, 3), np.float32)
    if len(pixels) > 100_000:
        pixels = pixels[:: len(pixels) // 100_000 + 1]
    cv2.setRNGSeed(11)  # deterministic clustering, reproducible gates
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(np.ascontiguousarray(pixels), k, None,
                               criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers


def _classify(means: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-cluster label per pixel of a (H,W,3) mean map (NaN -> -1)."""
    flat = means.reshape(-1, 3)
    ok = ~np.isnan(flat[:, 0])
    labels = np.full(len(flat), -1, np.int32)
    if ok.any():
        d = np.linalg.norm(flat[ok][:, None, :] - centers[None, :, :], axis=2)
        labels[ok] = np.argmin(d, axis=1).astype(np.int32)
    return labels.reshape(means.shape[:2])


def seam_steps(rgb: np.ndarray, side_a: np.ndarray, side_b: np.ndarray,
               *, window: int = 15, band: int = 4,
               centers: Optional[np.ndarray] = None,
               material_gap: float = 35.0) -> Optional[dict]:
    """Lab deltaE between the two sides of a region boundary, sampled on the
    boundary band: a tone seam is a step that survives windowed averaging,
    unlike matched texture where the two side means agree. With material
    `centers`, band pixels whose two sides classify to clusters farther
    apart than `material_gap` are content edges (hair|skin) and excluded;
    same-family tone steps (lit skin | dark skin) stay measurable."""
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    near_a = cv2.dilate(side_a.astype(np.uint8), kernel).astype(bool)
    near_b = cv2.dilate(side_b.astype(np.uint8), kernel).astype(bool)
    boundary = near_a & near_b & (side_a | side_b)
    if boundary.sum() < 200:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    mean_a = _masked_side_means(lab, side_a & ~side_b, window)
    mean_b = _masked_side_means(lab, side_b & ~side_a, window)
    delta = np.sqrt(np.nansum((mean_a - mean_b) ** 2, axis=2))
    valid = boundary & ~np.isnan(mean_a[:, :, 0]) & ~np.isnan(mean_b[:, :, 0])
    if valid.sum() < 200:
        return None
    out: dict = {"band_px": int(valid.sum())}
    values_all = delta[valid]
    out["p95_all"] = float(np.percentile(values_all, 95))
    gated = valid
    if centers is not None and len(centers) > 1:
        label_a = _classify(mean_a, centers)
        label_b = _classify(mean_b, centers)
        pair_gap = np.linalg.norm(
            centers[np.clip(label_a, 0, None)] - centers[np.clip(label_b, 0, None)],
            axis=2)
        same_family = pair_gap <= material_gap
        if (valid & same_family).sum() >= 150:
            gated = valid & same_family
        else:
            out["same_material_px"] = int((valid & same_family).sum())
            return out  # boundary crosses materials only: nothing to gate
    values = delta[gated]
    ys, xs = np.nonzero(gated)
    worst = int(np.argmax(values))
    out.update({
        "same_material_px": int(gated.sum()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "worst_xy": (int(xs[worst]), int(ys[worst])),
    })
    return out


def photo_seam_allowance(photo: Image.Image, thr: Thresholds,
                         *, window: int = 15, bands: int = 60,
                         rng_seed: int = 7) -> dict:
    """Natural same-material masked-mean deltaE across random straight bands
    of the input photo foreground: the allowance a legitimate texture never
    exceeds. Material clusters keep hair|skin transitions out of the
    statistic, exactly as in the texture measurement."""
    rgb = np.asarray(photo.convert("RGB"))
    foreground, _ = photo_foreground(photo)
    centers = material_clusters(rgb, foreground)
    h, w = foreground.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(rng_seed)
    samples: List[float] = []
    medians: List[float] = []
    ys, xs = np.nonzero(foreground)
    for _ in range(bands):
        if len(ys) < 500:
            break
        i = rng.integers(len(ys))
        cy, cx = float(ys[i]), float(xs[i])
        theta = rng.uniform(0, np.pi)
        signed = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        step = seam_steps(rgb, foreground & (signed < 0), foreground & (signed >= 0),
                          window=window, band=3, centers=centers,
                          material_gap=thr.seam_material_gap)
        if step is not None and "p95" in step:
            samples.append(step["p95"])
            medians.append(step["p50"])
    if not samples:
        return {"allowance": thr.seam_floor * thr.seam_margin,
                "median_allowance": thr.seam_floor,
                "photo_p99": None, "bands": 0}
    p99 = float(np.percentile(samples, 99))
    med_p99 = float(np.percentile(medians, 99)) if medians else thr.seam_floor
    return {"allowance": max(p99 * thr.seam_margin, thr.seam_floor),
            "median_allowance": max(med_p99 * thr.seam_margin, 0.6 * thr.seam_floor),
            "photo_p99": p99, "photo_p50": float(np.percentile(samples, 50)),
            "photo_median_p99": med_p99,
            "bands": len(samples)}


def detect_dark_smears(rgb: np.ndarray, analysis_mask: np.ndarray,
                       fill_mask: Optional[np.ndarray], thr: Thresholds) -> dict:
    """Spurious dark fragments in SYNTHESIZED regions: feature-dark blobs
    (vs the local windowed context, so uniformly dark materials like a hair
    mass never flag) that lie in fill no view observed — the eye-socket
    black-smear class. Observed dark content (real lashes, irises, intakes,
    panel lines) never gates here: photo-legitimate dark shapes are not
    separable from defects without semantics, and the mid-range battery
    already polices observed-space debris. Without a fill mask the detector
    reports shape statistics only (photo calibration mode)."""
    if analysis_mask.sum() < 500:
        return {"smears": [], "dark_fraction": 0.0}
    lum = _luminance(rgb)
    m = analysis_mask.astype(np.float32)
    window = max(31, int(0.12 * lum.shape[0]) | 1)
    local_mean = cv2.boxFilter(lum * m, -1, (window, window))
    local_weight = cv2.boxFilter(m, -1, (window, window))
    with np.errstate(divide="ignore", invalid="ignore"):
        local_ref = np.where(local_weight > 0.05,
                             local_mean / np.maximum(local_weight, 1e-6), 0.0)
    dark = analysis_mask & (lum < np.maximum(22.0, thr.smear_dark_ratio * local_ref))
    min_area = max(int(thr.smear_min_area_frac * analysis_mask.sum()), 12)
    max_area = int(thr.smear_max_area_frac * analysis_mask.sum())
    smears: List[dict] = []
    dark_px = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        dark_px += area
        if fill_mask is None or area > max_area:
            continue  # whole-region darkness is shading, not a fragment
        component = labels == i
        if (fill_mask & component).sum() < 0.4 * area:
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 1.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
            solidity = cv2.contourArea(contour) / hull_area
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        smears.append({
            "bbox": (x0, y0, x0 + int(stats[i, cv2.CC_STAT_WIDTH]),
                     y0 + int(stats[i, cv2.CC_STAT_HEIGHT])),
            "area": area, "solidity": round(float(solidity), 3), "in_fill": True})
    return {"smears": smears,
            "dark_fraction": float(dark_px) / float(analysis_mask.sum())}


# ---------------------------------------------------------------------------
# close-zoom audit (render space)
# ---------------------------------------------------------------------------

def region_mask_texture(regions: Regions) -> Image.Image:
    """RGB mask texture: R=fill, G=observed, B=symmetry (255 each)."""
    h, w = regions.observed.shape
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0][regions.fill] = 255
    img[:, :, 1][regions.observed] = 255
    img[:, :, 2][regions.symmetry] = 255
    return Image.fromarray(img)


def fill_probes(maps: TexelMaps, regions: Regions, *, max_probes: int = 3) -> List[Probe]:
    """Largest synthesized-fill components, viewed along their mean normal."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        regions.fill.astype(np.uint8), 8)
    probes: List[Probe] = []
    if n <= 1:
        return probes
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order[:max_probes]:
        if stats[i, cv2.CC_STAT_AREA] < 2000:
            break
        component = labels == i
        pts = maps.positions[component]
        nrm = maps.normals[component].mean(axis=0)
        nrm /= max(float(np.linalg.norm(nrm)), 1e-8)
        probes.append(Probe("fill", pts.mean(axis=0), nrm,
                            float(stats[i, cv2.CC_STAT_AREA]),
                            label=f"fill_{len(probes) + 1:02d}"))
    return probes


def boundary_probes(maps: TexelMaps, regions: Regions, *, max_probes: int = 2) -> List[Probe]:
    """Largest view-boundary band clusters (where tone seams live)."""
    labels = list(regions.per_view)
    band = np.zeros_like(regions.observed)
    kernel = np.ones((9, 9), np.uint8)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = regions.per_view[labels[i]], regions.per_view[labels[j]]
            exclusive_a, exclusive_b = a & ~b, b & ~a
            band |= (cv2.dilate(exclusive_a.astype(np.uint8), kernel).astype(bool)
                     & exclusive_b)
    probes: List[Probe] = []
    n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    if n <= 1:
        return probes
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order[:max_probes]:
        if stats[i, cv2.CC_STAT_AREA] < 800:
            break
        component = comp_labels == i
        pts = maps.positions[component]
        nrm = maps.normals[component].mean(axis=0)
        nrm /= max(float(np.linalg.norm(nrm)), 1e-8)
        probes.append(Probe("view_boundary", pts.mean(axis=0), nrm,
                            float(stats[i, cv2.CC_STAT_AREA]),
                            label=f"boundary_{len(probes) + 1:02d}"))
    return probes


def close_zoom_audit(renderer: ViewerTruthRenderer, probes: Sequence[Probe],
                     thr: Thresholds, evidence_dir: Path,
                     *, size: int = 896) -> dict:
    """Render 2x/4x viewer-truth crops per probe plus a nearest-sampled
    region-mask pass; gate facet blocks and dark smears on the 4x crops."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report: List[dict] = []
    total_facets_4x = 0
    total_smears_4x = 0
    for probe in probes:
        azimuth, elevation = probe.view_angles()
        entry = {"probe": probe.label, "kind": probe.kind,
                 "azimuth": round(azimuth, 1), "elevation": round(elevation, 1),
                 "zooms": {}}
        for zoom in (2.0, 4.0):
            rgb = renderer.render(azimuth, elevation, size=size,
                                  target_world=probe.position, zoom=zoom)
            mask_rgb = renderer.render(azimuth, elevation, size=size,
                                       texture="regions", apply_factor=False,
                                       target_world=probe.position, zoom=zoom,
                                       flat=True, background=(0.0, 0.0, 0.0))
            fill_screen = mask_rgb[:, :, 0] > 127
            observed_screen = mask_rgb[:, :, 1] > 127
            on_surface = fill_screen | observed_screen | (mask_rgb[:, :, 2] > 127)
            tag = f"{probe.label}_z{int(zoom)}"
            Image.fromarray(rgb).save(evidence_dir / f"{tag}.png")

            facet = detect_facet_blocks(rgb, fill_screen, thr)
            smear = detect_dark_smears(rgb, on_surface, fill_screen, thr)
            entry["zooms"][f"{zoom:.0f}x"] = {
                "fill_screen_fraction": round(float(fill_screen.sum())
                                              / max(int(on_surface.sum()), 1), 4),
                "facet_cells": len(facet["blocks"]),
                "facet_field": bool(facet["facet_field"]),
                "cellular_fraction": round(facet["cellular_fraction"], 4),
                "dark_smears": len(smear["smears"]),
                "dark_fraction": round(smear["dark_fraction"], 4),
            }
            annotated = rgb.copy()
            for block in facet["blocks"][:20]:
                x0, y0, x1, y1 = block["bbox"]
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 0, 0), 2)
            for smear_item in smear["smears"][:10]:
                x0, y0, x1, y1 = smear_item["bbox"]
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 0, 255), 2)
            if facet["facet_field"] or smear["smears"]:
                Image.fromarray(annotated).save(evidence_dir / f"{tag}_annotated.png")
            if zoom == 4.0:
                total_facets_4x += 1 if facet["facet_field"] else 0
                total_smears_4x += len(smear["smears"])
        report.append(entry)
    return {"probes": report, "facet_fields_4x": total_facets_4x,
            "dark_smears_4x": total_smears_4x}


# ---------------------------------------------------------------------------
# viewer-truth brightness
# ---------------------------------------------------------------------------

def render_foreground(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    bg = np.median(border.reshape(-1, 3), axis=0)
    return np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 14


def brightness_gate(renderer: ViewerTruthRenderer, bundle: Bundle,
                    specs: Sequence[ViewSpec], thr: Thresholds,
                    evidence_dir: Path, *, size: int = 896) -> Tuple[List[Gate], dict]:
    front = specs[0] if specs else None
    azimuth = front.azimuth_deg if front else 0.0
    elevation = front.elevation_deg if front else 0.0
    truth = renderer.render(azimuth, elevation, size=size)
    flattered = renderer.render(azimuth, elevation, size=size, apply_factor=False)
    Image.fromarray(np.concatenate([truth, flattered], axis=1)).save(
        evidence_dir / "viewer_truth_vs_repo_render.png")

    photo_path = bundle.directory / "input.png"
    gates: List[Gate] = []
    detail: dict = {}
    truth_lum = float(np.median(_luminance(truth)[render_foreground(truth)]))
    flat_lum = float(np.median(_luminance(flattered)[render_foreground(flattered)]))
    detail["viewer_truth_median_lum"] = round(truth_lum, 1)
    detail["factor_ignoring_median_lum"] = round(flat_lum, 1)
    if photo_path.exists():
        photo = Image.open(photo_path)
        photo_rgb = np.asarray(photo.convert("RGB"))
        pfg, fg_method = photo_foreground(photo, cache_key=str(photo_path))
        photo_lum = float(np.median(_luminance(photo_rgb)[pfg]))
        ratio = truth_lum / max(photo_lum, 1e-6)
        detail["photo_median_lum"] = round(photo_lum, 1)
        detail["photo_foreground_method"] = fg_method
        detail["brightness_ratio"] = round(ratio, 3)
        lo, hi = thr.brightness_ratio_range
        gates.append(Gate(
            "viewer.brightness_ratio", lo <= ratio <= hi,
            f"{ratio:.3f} (render {truth_lum:.0f} vs photo {photo_lum:.0f} "
            f"[{fg_method}]; factor-ignoring renderer shows {flat_lum:.0f})",
            f"in [{lo}, {hi}] with baseColorFactor applied",
            evidence="viewer_truth_vs_repo_render.png"))
    return gates, detail


# ---------------------------------------------------------------------------
# artifact battery (standalone, warn-only)
# ---------------------------------------------------------------------------

def artifact_battery_report(bundle: Bundle, *, size: int = 512) -> dict:
    """Run the measured artifact-class battery on the bundle's own 8-view
    turnaround (the same views and renderer the whole-bake gate uses, so
    the good-corpus calibration transfers). Warn-only by doctrine: a
    single bake has no baseline to difference against, and the absolute
    good-corpus margins are below the zero-false-fire voting bar (see
    `abstract3d.artifact_gates`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import trimesh
    from abstract3d.artifact_gates import (
        evaluate_bundle_artifact_battery, photo_reference)
    from abstract3d.rendering import render_mesh_views

    # Reload the mesh untouched: render_mesh_views handles the export
    # frame marker itself (the calibration path); the harness's own
    # bundle.mesh is already counter-rotated and would double-rotate.
    mesh = trimesh.load(bundle.directory / "scene.glb", force="mesh",
                        process=False)
    views: List[Tuple[str, object]] = []
    for elevation in (10.0, 50.0):
        renders = render_mesh_views(
            mesh, size=size, azimuths=[0.0, 90.0, 180.0, -90.0],
            elevation=elevation)
        views.extend(
            (f"az{az}_el{int(elevation)}", render)
            for az, render in zip((0, 90, 180, -90), renders))

    photo_ref = None
    photo_path = bundle.directory / "input.png"
    if photo_path.exists():
        # The RAW photo, not the matte: the border-median background is
        # what the image-in-image payload check compares against (a
        # matted photo has its backdrop destroyed).
        photo_ref = photo_reference(Image.open(photo_path))

    # Stats live in texture_artifacts (pipeline bundles) and/or the
    # rebake "stats" block (fill_detail / leverage); merge for the
    # stats-based checks, which degrade gracefully when keys miss.
    ta = bundle.metadata.get("texture_artifacts", bundle.metadata) or {}
    stats = dict(bundle.metadata.get("stats") or {})
    for key in ("observed_view_stats", "fill_detail", "leverage"):
        stats.setdefault(key, ta.get(key))
    return evaluate_bundle_artifact_battery(
        views, photo_ref=photo_ref, stats=stats)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def save_texel_crop(rgb: np.ndarray, bbox: Tuple[int, int, int, int],
                    path: Path, margin: int = 48) -> None:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return
    scale = max(1, int(np.ceil(320 / max(crop.shape[:2]))))
    if scale > 1:
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(path)


def run(bundle_dir: Path, out_dir: Path, thr: Thresholds, *, size: int = 896) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = out_dir / "evidence"
    evidence.mkdir(exist_ok=True)

    bundle = load_bundle(bundle_dir)
    gates: List[Gate] = material_gates(bundle, thr)

    texture_rgb = bundle.texture_array
    maps = build_texel_maps(bundle.mesh, texture_rgb.shape[0])
    specs, info = bundle_view_specs(bundle)
    regions = split_regions(maps, specs, info)

    # region debug overlay: fill=red, symmetry=blue tint over the texture
    overlay = texture_rgb.copy()
    overlay[regions.fill] = (0.55 * overlay[regions.fill]
                             + np.array([115, 0, 0])).astype(np.uint8)
    overlay[regions.symmetry] = (0.55 * overlay[regions.symmetry]
                                 + np.array([0, 0, 115])).astype(np.uint8)
    Image.fromarray(overlay).save(evidence / "region_overlay.png")

    character = fill_character(texture_rgb, regions)
    ratio = character["fill_to_observed_ratio"]
    coarse = character.get("coarse_ratio")
    gates.append(Gate(
        "texel.fill_gradient_energy_ratio",
        ratio is not None
        and thr.fill_energy_ratio_min <= ratio <= thr.fill_energy_ratio_max,
        (f"{ratio:.3f} (observed {character['observed_energy']:.0f}, "
         f"synthesized {character['unobserved_energy']:.0f}, "
         f"coarse ratio {coarse:.3f})" if ratio is not None and coarse is not None
         else f"{ratio:.3f}" if ratio is not None
         else "n/a (no fill or no observed)"),
        f"in [{thr.fill_energy_ratio_min}, {thr.fill_energy_ratio_max}] "
        f"(synthesized detail vs projected photo detail; upper bound rejects noise)"))

    facet_fill = detect_facet_blocks(texture_rgb, regions.fill, thr)
    facet_observed = detect_facet_blocks(texture_rgb, regions.observed, thr)
    cellular_allowed = max(thr.cellular_ratio_max * facet_observed["cellular_fraction"],
                           thr.cellular_floor)
    gates.append(Gate(
        "texel.facet_cellular",
        facet_fill["cellular_fraction"] <= cellular_allowed,
        f"fill {facet_fill['cellular_fraction']:.3f} vs observed "
        f"{facet_observed['cellular_fraction']:.3f} "
        f"({len(facet_fill['blocks'])} straight-edged cells)",
        f"fill cellular fraction <= {cellular_allowed:.3f} "
        f"(max({thr.cellular_ratio_max} x observed, {thr.cellular_floor}))"))
    for k, block in enumerate(facet_fill["blocks"][:6]):
        save_texel_crop(texture_rgb, block["bbox"],
                        evidence / f"texel_facet_{k:02d}.png")

    photo_path = bundle.directory / "input.png"
    allowance_info = {"allowance": thr.seam_floor * thr.seam_margin, "photo_p99": None}
    if photo_path.exists():
        allowance_info = photo_seam_allowance(Image.open(photo_path), thr)
    allowance = allowance_info["allowance"]

    seam_report: Dict[str, dict] = {}
    centers = material_clusters(texture_rgb, regions.observed)
    labels = list(regions.per_view)
    pairs = [(labels[i], labels[j]) for i in range(len(labels))
             for j in range(i + 1, len(labels))]
    for a, b in pairs:
        step = seam_steps(texture_rgb, regions.per_view[a] & ~regions.per_view[b],
                          regions.per_view[b] & ~regions.per_view[a],
                          centers=centers, material_gap=thr.seam_material_gap)
        if step:
            seam_report[f"{a}|{b}"] = step
    step = seam_steps(texture_rgb, regions.observed, regions.fill | regions.symmetry,
                      centers=centers, material_gap=thr.seam_material_gap)
    if step:
        seam_report["observed|fill"] = step
    worst_pair, worst_p95 = None, 0.0
    worst_med_pair, worst_med = None, 0.0
    for pair_label, step in seam_report.items():
        if "p95" not in step:
            continue
        if step["p95"] > worst_p95:
            worst_pair, worst_p95 = pair_label, step["p95"]
        if step["p50"] > worst_med:
            worst_med_pair, worst_med = pair_label, step["p50"]
        x, y = step["worst_xy"]
        save_texel_crop(texture_rgb, (x - 40, y - 40, x + 40, y + 40),
                        evidence / f"seam_{pair_label.replace('|', '_vs_')}.png")
    median_allowance = allowance_info.get("median_allowance", thr.seam_floor)
    gates.append(Gate(
        "texel.seam_steps",
        worst_p95 <= allowance and worst_med <= median_allowance,
        (f"worst p95 deltaE {worst_p95:.1f} at {worst_pair}; "
         f"worst median {worst_med:.1f} at {worst_med_pair}") if worst_pair else "no bands",
        f"p95 <= {allowance:.1f} and median <= {median_allowance:.1f} "
        f"(photo-calibrated x {thr.seam_margin})"))

    factor = ((bundle.gltf.get("materials") or [{}])[0]
              .get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1]))
    renderer = ViewerTruthRenderer(bundle.mesh, bundle.texture, factor)
    try:
        renderer.set_texture("regions", region_mask_texture(regions), nearest=True)
        bright_gates, bright_detail = brightness_gate(
            renderer, bundle, specs, thr, evidence, size=size)
        gates.extend(bright_gates)

        uv = np.asarray(bundle.mesh.visual.uv, dtype=np.float32)
        tex_h, tex_w = texture_rgb.shape[:2]
        us = np.clip(np.rint(uv[:, 0] * (tex_w - 1)), 0, tex_w - 1).astype(np.int32)
        vs = np.clip(np.rint((1.0 - uv[:, 1]) * (tex_h - 1)), 0, tex_h - 1).astype(np.int32)
        vertex_darkness = 1.0 - _luminance(texture_rgb)[vs, us] / 255.0
        probes = (concavity_probes(bundle.mesh, max_probes=6,
                                   vertex_darkness=vertex_darkness)
                  + fill_probes(maps, regions)
                  + boundary_probes(maps, regions))
        close = close_zoom_audit(renderer, probes, thr, evidence, size=size)
        gates.append(Gate(
            "close.facet_fields_4x",
            close["facet_fields_4x"] <= thr.facet_fields_4x_max,
            f"{close['facet_fields_4x']} probe crop(s) show a faceted fill field at 4x",
            f"<= {thr.facet_fields_4x_max}"))
        gates.append(Gate(
            "close.dark_smears_4x",
            close["dark_smears_4x"] <= thr.smear_blocks_4x_max,
            f"{close['dark_smears_4x']} spurious dark fragments across 4x crops",
            f"<= {thr.smear_blocks_4x_max}"))
    finally:
        renderer.release()

    # Reference-leverage ledger (reporting only, never a gate): how much
    # of the photo-visible surface the bake painted directly vs
    # surrendered to completion/fill — recorded by the bake itself
    # (`texture_artifacts.leverage`) and surfaced here per the project
    # owner's visibility request.
    ta_meta = bundle.metadata.get("texture_artifacts", bundle.metadata) or {}
    leverage = ta_meta.get("leverage")

    # Artifact battery (warn-only; see artifact_battery_report). A GL or
    # load failure degrades to a recorded error, never a crash of the
    # material/texel gates above.
    try:
        battery = artifact_battery_report(bundle)
    except Exception as exc:  # pragma: no cover - environment specific
        battery = {"error": f"battery unavailable: {exc}", "warnings": []}

    result = {
        "bundle": str(bundle_dir),
        "verdict": "PASS" if all(g.passed for g in gates) else "FAIL",
        "failed": [g.name for g in gates if not g.passed],
        "gates": [asdict(g) for g in gates],
        "artifact_battery": battery,
        "projection": info,
        "leverage": leverage,
        "coverage_reconciliation": regions.reconciliation,
        "fill_character": character,
        "seam_allowance": allowance_info,
        "seams": seam_report,
        "facet_fill": {"cellular_fraction": facet_fill["cellular_fraction"],
                       "observed_cellular_fraction": facet_observed["cellular_fraction"],
                       "flat_fraction": facet_fill["flat_fraction"],
                       "blocks": facet_fill["blocks"][:20]},
        "close_zoom": close,
        "brightness": bright_detail,
        "thresholds": asdict(thr),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def calibrate_photos(photos: Sequence[Path], thr: Thresholds) -> dict:
    """The input photos must pass the photo-facing detectors: no faceted
    field (real texture has none), no fill-conditioned smears by
    construction, and a finite same-material seam allowance."""
    report = {}
    for path in photos:
        photo = Image.open(path)
        rgb = np.asarray(photo.convert("RGB"))
        foreground, _ = photo_foreground(photo, cache_key=str(path))
        facet = detect_facet_blocks(rgb, foreground, thr)
        allowance = photo_seam_allowance(photo, thr)
        report[str(path)] = {
            "facet_field": bool(facet["facet_field"]),
            "facet_cells": len(facet["blocks"]),
            "cellular_fraction": round(facet["cellular_fraction"], 4),
            "seam_allowance": allowance,
            "ok": not facet["facet_field"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--size", type=int, default=896)
    parser.add_argument("--calibrate-photos", action="store_true",
                        help="run photo-side detector calibration on the "
                             "bundle's input.png instead of the full audit")
    args = parser.parse_args()

    thr = Thresholds()
    bundle_dir = args.bundle.resolve()
    out = (args.out or Path("/tmp/texture_qa") / bundle_dir.name).resolve()

    if args.calibrate_photos:
        report = calibrate_photos([bundle_dir / "input.png"], thr)
        print(json.dumps(report, indent=2))
        sys.exit(0 if all(v["ok"] for v in report.values()) else 1)

    result = run(bundle_dir, out, thr, size=args.size)
    print(f"\n=== texture_qa: {bundle_dir.name}: {result['verdict']} "
          f"({len(result['failed'])} failed gates) ===")
    for gate in result["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {gate['name']}: {gate['measured']}"
              f"   (require {gate['requirement']})")
    battery = result.get("artifact_battery") or {}
    battery_warnings = battery.get("warnings") or []
    if battery.get("error"):
        print(f"\nartifact battery: {battery['error']}")
    elif battery_warnings:
        print("\nARTIFACT BATTERY WARNINGS (recorded, non-gating):")
        for warning in battery_warnings:
            print(f"  [WARN] {warning}")
    else:
        print("\nartifact battery: quiet (all detectors under their "
              "good-corpus warn lines)")
    print(f"\ncoverage reconciliation (reconstructed vs bake metadata):")
    for row in result["coverage_reconciliation"]:
        print(f"  {row['label']} az{row['azimuth_deg']:+.1f}: "
              f"qa={row['reconstructed_coverage']:.3f} "
              f"bake={row['metadata_coverage']:.3f}")
    leverage = result.get("leverage")
    if leverage and leverage.get("available"):
        print("\nreference leverage (bake ledger; reporting only):")
        print(f"  photo-visible {leverage['potential_union_ratio']:.1%} of surface | "
              f"direct-painted {leverage['direct_painted_ratio']:.1%} | "
              f"leverage {leverage['leverage_ratio']:.1%} | "
              f"surrendered-visible {leverage['surrendered_visible_ratio']:.1%} | "
              f"unobservable {leverage['unobservable_ratio']:.1%}")
        for row in leverage.get("views", []):
            print(f"  {row['label']}: potential {row['potential_ratio']:.1%} "
                  f"painted {row['painted_ratio']:.1%} won {row['won_ratio']:.1%} "
                  f"(surrendered: facing {row['surrendered_facing_gate']} "
                  f"zone {row['surrendered_zone_gate']} "
                  f"downstream {row['surrendered_downstream']} "
                  f"union {row['surrendered_union_drop']})")
    print(f"results: {out / 'results.json'}")
    print(f"evidence: {out / 'evidence'}")
    sys.exit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
