#!/usr/bin/env python3
"""Numeric attribution of e20 texture defects from an instrumented bake capture.

Consumes the dump of texture_forensics_bake.py and answers, with numbers:
  A. does the isolated re-bake reproduce the shipped e20 texture?
  B. which view WINS each defect region's texels (final blend weights)?
  C. twin (windowed vs full-span) content offset + co-paint disagreement (H1)
  D. where does each side view's lip content land in 3D vs the mesh's true
     mouth (effective pose error in degrees) (H2)
  E. what content painted the right-lens texels (H3)

Region boxes are given in each view's REGISTERED CANONICAL 1024 frame
(inspect the registered_view_*.png dumps first; boxes are recorded in the
output JSON so every number is reproducible).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from PIL import Image

CANVAS = 1024


def texel_view_coords(positions: np.ndarray, azimuth_deg: float, elevation_deg: float,
                      half_extent: float) -> dict:
    from abstract3d.reference_flow import first_surface_visible, project_texels_to_view

    proj = project_texels_to_view(
        positions, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
        camera_distance=3.0, ortho_half_extent=half_extent, canvas=CANVAS)
    proj["visible"] = first_surface_visible(proj, CANVAS)
    return proj


def region_mask_from_view(proj: dict, box: tuple[float, float, float, float],
                          require_visible: bool = True) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = (proj["sample_x"] >= x0) & (proj["sample_x"] <= x1) \
        & (proj["sample_y"] >= y0) & (proj["sample_y"] <= y1) & proj["in_frame"]
    if require_visible:
        mask &= proj["visible"]
    return mask


def is_skin(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (r > 105) & (r >= g + 8) & (g >= b - 4) & (r - b > 18)


def summarize_region(name: str, region: np.ndarray, weights: np.ndarray,
                     meta: list, winner: np.ndarray, painted_any: np.ndarray,
                     ship_tex: np.ndarray | None = None) -> dict:
    total = int(region.sum())
    rows = []
    region_painted = region & painted_any
    skin_mask = None
    if ship_tex is not None:
        skin_mask = is_skin(ship_tex) & region_painted
    for index, view_meta in enumerate(meta):
        w = weights[index]
        painted = (w > 0) & region
        wins = (winner == index) & region_painted
        row = {
            "view": f"{index + 1}:{view_meta['label']}",
            "azimuth_deg": view_meta["azimuth_deg"],
            "painted_texels": int(painted.sum()),
            "mean_weight": round(float(w[painted].mean()) if painted.any() else 0.0, 4),
            "winner_texels": int(wins.sum()),
            "winner_share": round(float(wins.sum()) / max(int(region_painted.sum()), 1), 4),
        }
        if skin_mask is not None and skin_mask.any():
            skin_wins = (winner == index) & skin_mask
            row["skin_winner_texels"] = int(skin_wins.sum())
            row["skin_winner_share"] = round(
                float(skin_wins.sum()) / max(int(skin_mask.sum()), 1), 4)
        rows.append(row)
    out = {"region": name, "texels": total,
           "painted_texels": int(region_painted.sum()), "views": rows}
    if ship_tex is not None and region_painted.any():
        out["shipped_mean_rgb"] = [
            round(float(v), 1) for v in ship_tex[region_painted].mean(axis=0)]
        out["shipped_skin_fraction"] = round(
            float(is_skin(ship_tex)[region_painted].mean()), 4)
    return out


def content_centroid(region: np.ndarray, positions: np.ndarray) -> dict:
    pts = positions[:, :, :3][region]
    if len(pts) == 0:
        return {"count": 0}
    c = pts.mean(axis=0)
    azimuth = float(np.degrees(np.arctan2(c[1], c[0])))
    return {"count": int(len(pts)),
            "centroid_xyz": [round(float(v), 4) for v in c],
            "centroid_azimuth_deg": round(azimuth, 2),
            "centroid_z": round(float(c[2]), 4)}


def vertical_content_offset(img_a: np.ndarray, img_b: np.ndarray,
                            row_range: tuple[int, int]) -> dict:
    """Best vertical shift aligning B to A by masked NCC over a row band."""
    lo, hi = row_range
    a = img_a[:, :, :3].astype(np.float32).mean(axis=2)
    b = img_b[:, :, :3].astype(np.float32).mean(axis=2)
    am = img_a[:, :, 3] > 64
    bm = img_b[:, :, 3] > 64
    best = {"shift_px": 0, "ncc": -2.0}
    for shift in range(-80, 81, 2):
        rb = np.roll(b, shift, axis=0)
        rm = np.roll(bm, shift, axis=0)
        m = am[lo:hi] & rm[lo:hi]
        if m.sum() < 500:
            continue
        va = a[lo:hi][m]
        vb = rb[lo:hi][m]
        va = va - va.mean()
        vb = vb - vb.mean()
        denominator = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
        if denominator < 1e-6:
            continue
        ncc = float((va * vb).sum()) / denominator
        if ncc > best["ncc"]:
            best = {"shift_px": shift, "ncc": round(ncc, 4)}
    # refine +-1
    for shift in (best["shift_px"] - 1, best["shift_px"] + 1):
        rb = np.roll(b, shift, axis=0)
        rm = np.roll(bm, shift, axis=0)
        m = am[lo:hi] & rm[lo:hi]
        if m.sum() < 500:
            continue
        va = a[lo:hi][m] - a[lo:hi][m].mean()
        vb = rb[lo:hi][m] - rb[lo:hi][m].mean()
        denominator = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
        if denominator < 1e-6:
            continue
        ncc = float((va * vb).sum()) / denominator
        if ncc > best["ncc"]:
            best = {"shift_px": shift, "ncc": round(ncc, 4)}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="repro dir")
    parser.add_argument("--bundle", type=Path, required=True, help="e20 bundle dir")
    parser.add_argument("--regions", type=Path, required=True,
                        help="JSON: per-view-label canonical-frame boxes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from abstract3d.texturing import canonical_ortho_half_extent

    import trimesh

    data = np.load(args.capture / "capture.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    weights = np.asarray(data["weights"], dtype=np.float32)
    view_rgba = np.asarray(data["view_rgba"])
    positions = np.asarray(data["positions"], dtype=np.float32)
    surface = positions[:, :, 3] > 0
    stats = json.loads((args.capture / "bake_stats.json").read_text())
    regions_config = json.loads(args.regions.read_text())
    mesh = trimesh.load(str(args.bundle / "geometry.glb"), force="mesh", process=False)

    report: dict = {"capture": str(args.capture), "meta": meta,
                    "regions_config": regions_config}

    # ---- A. reproduction check ----
    repro_tex = np.asarray(Image.open(args.capture / "texture.png").convert("RGB"),
                           dtype=np.int16)
    ship_tex = np.asarray(Image.open(args.bundle / "texture.png").convert("RGB"),
                          dtype=np.int16)
    if repro_tex.shape == ship_tex.shape:
        diff = np.abs(repro_tex - ship_tex).mean(axis=2)
        report["reproduction"] = {
            "texture_mae_surface": round(float(diff[surface].mean()), 3),
            "texture_mae_all": round(float(diff.mean()), 3),
            "frac_surface_within_2": round(float((diff[surface] <= 2).mean()), 4),
        }
    else:
        report["reproduction"] = {"error": f"shape {repro_tex.shape} vs {ship_tex.shape}"}

    # ---- winner map ----
    painted_any = (weights > 0).any(axis=0)
    winner = np.argmax(weights, axis=0)
    winner[~painted_any] = -1
    palette = np.array([
        [230, 60, 60], [60, 130, 230], [60, 200, 90], [230, 180, 50],
        [170, 90, 220], [80, 220, 210], [240, 120, 190], [150, 150, 150]],
        dtype=np.uint8)
    winner_rgb = np.zeros((*winner.shape, 3), dtype=np.uint8)
    for index in range(len(meta)):
        winner_rgb[winner == index] = palette[index % len(palette)]
    winner_rgb[~surface] = 30
    Image.fromarray(winner_rgb).save(args.out / "winner_map_atlas.png")
    report["winner_totals"] = [
        {"view": f"{i + 1}:{m['label']}",
         "winner_texels": int(((winner == i) & surface).sum()),
         "painted_texels": int(((weights[i] > 0) & surface).sum())}
        for i, m in enumerate(meta)]

    # ---- per-view projections (texel -> canonical view coords) ----
    projections = {}
    for index, view_meta in enumerate(meta):
        azimuth = float(view_meta["azimuth_deg"])
        elevation = float(view_meta["elevation_deg"])
        half = canonical_ortho_half_extent(
            mesh, azimuth_deg=azimuth, elevation_deg=elevation, border_ratio=0.15)
        projections[index] = texel_view_coords(positions, azimuth, elevation, half)

    def probe_projection(azimuth: float, elevation: float) -> dict:
        half = canonical_ortho_half_extent(
            mesh, azimuth_deg=azimuth, elevation_deg=elevation, border_ratio=0.15)
        return texel_view_coords(positions, azimuth, elevation, half)

    def splat_render(proj: dict, rgb_source: np.ndarray, path: Path) -> None:
        """Geometrically exact (same projector math) low-fi render for
        picking defect boxes in probe-view coordinates."""
        visible = proj["visible"]
        xs = np.clip(np.round(proj["sample_x"][visible]).astype(int), 0, CANVAS - 1)
        ys = np.clip(np.round(proj["sample_y"][visible]).astype(int), 0, CANVAS - 1)
        canvas_img = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
        canvas_img[ys, xs] = rgb_source[visible]
        from scipy.ndimage import grey_closing
        for c in range(3):
            canvas_img[:, :, c] = grey_closing(canvas_img[:, :, c], size=3)
        arr = canvas_img.copy()
        arr[::64, :] = [255, 0, 0]
        arr[:, ::64] = [255, 0, 0]
        Image.fromarray(arr).save(path)

    # ---- B/D/E: region attributions ----
    # Region color grading always reads the CAPTURED bake's own texture;
    # the bundle texture serves only the reproduction check above.
    ship_tex_u8 = np.asarray(Image.open(args.capture / "texture.png").convert("RGB"))
    for probe_name, (paz, pel) in {"probe_l30": (30.0, 12.0),
                                   "probe_r30": (-30.0, 12.0),
                                   "probe_front": (0.0, 0.0)}.items():
        splat_render(probe_projection(paz, pel), ship_tex_u8,
                     args.out / f"{probe_name}_shipped_splat.png")

    region_reports = []
    centroids = {}
    for region_name, spec in regions_config.items():
        if "pos_box" in spec:
            # 3D anatomical region in bake-frame world coordinates:
            # [xmin, xmax, ymin, ymax, zmin, zmax]
            xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in spec["pos_box"])
            xyz = positions[:, :, :3]
            source_region = (
                surface
                & (xyz[:, :, 0] >= xmin) & (xyz[:, :, 0] <= xmax)
                & (xyz[:, :, 1] >= ymin) & (xyz[:, :, 1] <= ymax)
                & (xyz[:, :, 2] >= zmin) & (xyz[:, :, 2] <= zmax))
            view_index = None
            proj = None
        else:
            box = tuple(float(v) for v in spec["box"])
            if "view_index" in spec:
                view_index = int(spec["view_index"])  # 0-based capture index
                proj = projections[view_index]
            else:
                view_index = None
                proj = probe_projection(float(spec["azimuth_deg"]),
                                        float(spec["elevation_deg"]))
            source_region = region_mask_from_view(proj, box)
        if spec.get("restrict_painted_by_view") and view_index is not None:
            source_region &= weights[view_index] > 0
        row = summarize_region(
            region_name, source_region, weights, meta, winner, painted_any,
            ship_tex=ship_tex_u8)
        # Which image rows each view SAMPLED for this region (content band):
        for index in range(len(meta)):
            painted = source_region & (weights[index] > 0)
            if painted.sum() >= 20:
                sy = projections[index]["sample_y"][painted]
                sx = projections[index]["sample_x"][painted]
                row["views"][index]["sample_row_mean"] = round(float(sy.mean()), 1)
                row["views"][index]["sample_row_p10_p90"] = [
                    round(float(np.percentile(sy, 10)), 1),
                    round(float(np.percentile(sy, 90)), 1)]
                row["views"][index]["sample_col_mean"] = round(float(sx.mean()), 1)
        region_reports.append(row)
        centroids[region_name] = content_centroid(source_region, positions)
        # visual: mark region on winner map
        overlay = winner_rgb.copy()
        overlay[source_region] = [255, 255, 255]
        Image.fromarray(overlay).save(args.out / f"region_{region_name}_atlas.png")
        # per-view content in the region, splatted into the probe frame
        if "azimuth_deg" in spec and proj is not None:
            xs = np.clip(np.round(proj["sample_x"][source_region]).astype(int),
                         0, CANVAS - 1)
            ys = np.clip(np.round(proj["sample_y"][source_region]).astype(int),
                         0, CANVAS - 1)
            x0v, y0v, x1v, y1v = (int(v) for v in box)
            tiles = []
            for index in range(len(meta)):
                canvas_img = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
                has_w = (weights[index] > 0)[source_region]
                canvas_img[ys[has_w], xs[has_w]] = view_rgba[index][source_region][has_w]
                tiles.append(canvas_img[y0v:y1v + 1, x0v:x1v + 1])
            strip = np.concatenate(tiles, axis=1)
            Image.fromarray(strip).save(args.out / f"region_{region_name}_perview.png")
    report["regions"] = region_reports
    report["region_centroids"] = centroids

    # ---- C. twin analysis (H1) ----
    labels = [m["label"] for m in meta]
    twins = []
    for label in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label]
        if len(indices) == 2:
            twins.append((label, indices[0], indices[1]))
    twin_rows = []
    registered_images = {}
    for index in range(len(meta)):
        path = args.capture / f"registered_view_{index + 1:02d}_{labels[index]}.png"
        registered_images[index] = np.asarray(Image.open(path).convert("RGBA"))
    for label, a, b in twins:
        co = (weights[a] > 0) & (weights[b] > 0)
        rgb_a = view_rgba[a].astype(np.int16)
        rgb_b = view_rgba[b].astype(np.int16)
        co_mae = float(np.abs(rgb_a[co] - rgb_b[co]).mean()) if co.any() else None
        half = canonical_ortho_half_extent(
            mesh, azimuth_deg=float(meta[a]["azimuth_deg"]),
            elevation_deg=0.0, border_ratio=0.15)
        px_world = 2.0 * half / CANVAS
        offset_face = vertical_content_offset(
            registered_images[a], registered_images[b], (120, 500))
        offset_full = vertical_content_offset(
            registered_images[a], registered_images[b], (100, 900))
        twin_rows.append({
            "label": label,
            "view_a": a + 1, "view_b": b + 1,
            "co_painted_texels": int(co.sum()),
            "co_painted_color_mae": round(co_mae, 2) if co_mae is not None else None,
            "content_offset_head_band_px": offset_face,
            "content_offset_full_px": offset_full,
            "px_world_units": round(px_world, 5),
            "head_band_offset_world": round(offset_face["shift_px"] * px_world, 4),
        })
    report["twins"] = twin_rows

    # registration table from bake stats
    report["view_registration"] = [
        {"label": row.get("label"),
         "width_profile": {k: row.get("photometric", {}).get(k)
                            for k in ("applied", "scale", "shift_x", "shift_y", "iou")},
         "reg2d": {k: row.get(k) for k in ("applied", "scale", "shift_x", "shift_y",
                                            "score", "improved")},
         "overlap": row.get("overlap_alignment"),
         "dense_flow": {k: (row.get("dense_flow") or {}).get(k)
                         for k in ("applied", "validated_cells", "mean_abs_px",
                                   "max_abs_px", "improvement")},
         "pose": row.get("pose")}
        for row in stats.get("view_registration", [])]

    (args.out / "analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("reproduction", "winner_totals", "twins")}, indent=1))
    print("regions ->", args.out / "analysis.json")


if __name__ == "__main__":
    main()
