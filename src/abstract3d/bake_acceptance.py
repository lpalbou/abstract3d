"""Whole-bake A/B acceptance gate for generated-reference bakes.

Per-view material gates are structurally blind to COMPOSITION-level
failure: on the measured chair case every shipped view strict-passed its
oracles, and the finished bake still regressed below the no-references
baseline (a deltaE ~42 tone step where the generated top view hands off
to the protected front, plus an overall brightness drop). No statistic of
a single view can see a defect that only exists where views meet.

This module judges the finished product instead: bake the texture twice —
once with the generated views (candidate) and once without (baseline) —
and ship the candidate only if it does NOT regress the baseline on three
comparative, render-space axes:

- photo fidelity: masked LAB distance between the source photo and the
  render at the source pose. Catches any synthesis contamination of the
  photo-observed surface (the failure protection cannot reach: texels the
  photo sees only at grazing incidence).
- brightness: foreground L mean of the front render. Catches the darkened
  bake the harness measured on the chair.
- seam steps: long, coherent strong-edge components across a turnaround
  render set. Catches view-handoff tone steps; texture DETAIL (grain,
  panel lines, plumage) forms short or thin edges and is not counted —
  only extended contours above a deltaE step threshold register.

Every check is A/B under an identical procedure, so registration and
renderer biases cancel; the contract is monotone non-regression, not an
absolute quality bar. Thresholds are calibrated on the four-subject
validation set (owl/chair/spaceship/portrait: the chair is the labeled
regression the gate must reject; the other three are labeled acceptances).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["evaluate_generated_bake", "evaluate_single_view_bake"]


def evaluate_single_view_bake(
    stats: Dict[str, Any],
    *,
    min_total_coverage: float = 0.12,
    min_source_coverage: float = 0.10,
    min_capture_efficiency: float = 0.30,
) -> Dict[str, Any]:
    """Sanity verdict for a bake that ships WITHOUT accepted generated views.

    The whole-bake A/B gate only runs when generated views exist — a bake
    whose generation rejected every angle (or never ran) shipped ungated,
    and the measured incident proved a single-photo bake can be broken on
    its own (source pose mis-estimated: coverage 0.055 vs healthy 0.18+,
    exit code 0). Three floors, all read from the bake's own stats:

    - total observed coverage >= `min_total_coverage` (healthy fleet min
      0.182, broken max 0.065),
    - the SOURCE view's own coverage >= `min_source_coverage` (robust when
      real references inflate the total),
    - capture efficiency >= `min_capture_efficiency`: source coverage over
      the facing fraction its pose could have painted (healthy min 0.40,
      broken max 0.26) — normalizes for subject shape, so an elongated
      car's honest ~0.16 coverage passes while a broken pose fails.

    Deliberately NO pose-score floor: measured healthy scores (0.015-0.053)
    overlap the broken commits (0.033-0.057) — the tempting signal is
    useless and gating on it would only manufacture false confidence.

    Returns the same report shape as `evaluate_generated_bake`.
    """

    metrics: Dict[str, Any] = {}
    reasons: List[str] = []

    total = stats.get("observed_coverage_ratio")
    metrics["observed_coverage_ratio"] = {
        "value": total, "min_allowed": min_total_coverage}
    if total is not None and float(total) < float(min_total_coverage):
        reasons.append(
            f"observed coverage {float(total):.3f} below floor "
            f"{min_total_coverage} (healthy fleet minimum 0.18)")

    view_rows = stats.get("observed_view_stats") or []
    source_row = next(
        (row for row in view_rows if row.get("index") == 1
         or row.get("label") in ("front", "source", "view_01")),
        view_rows[0] if view_rows else None)
    if source_row is not None:
        source_coverage = source_row.get("coverage_ratio")
        efficiency = source_row.get("capture_efficiency")
        metrics["source_view_coverage"] = {
            "value": source_coverage, "min_allowed": min_source_coverage}
        metrics["capture_efficiency"] = {
            "value": efficiency, "min_allowed": min_capture_efficiency,
            "facing_fraction": source_row.get("facing_fraction")}
        if source_coverage is not None and float(source_coverage) < float(min_source_coverage):
            reasons.append(
                f"source-view coverage {float(source_coverage):.3f} below "
                f"floor {min_source_coverage}")
        if efficiency is not None and float(efficiency) < float(min_capture_efficiency):
            reasons.append(
                f"capture efficiency {float(efficiency):.3f} below floor "
                f"{min_capture_efficiency}: the source pose painted far "
                "less than its viewpoint could see (wrong pose or broken "
                "registration)")

    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics}


def _render_foreground(render: Any) -> Any:
    """Foreground mask of an offscreen render (constant near-white bg)."""

    from .reference_generation import clay_silhouette

    return clay_silhouette(render)


def _lab(image: Any) -> Any:
    import numpy as np
    from skimage import color as skcolor

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return skcolor.rgb2lab(rgb).astype(np.float32)


def _photo_fidelity(mesh: Any, source_rgba: Any, pose: Tuple[float, float],
                    render_size: int) -> float:
    """Mean LAB deltaE between the source photo and the source-pose render.

    The photo matte is similarity-registered onto the render silhouette
    (crop/framing differ between the two) and compared on the foreground
    intersection. Absolute values carry registration residue; the GATE
    only compares candidate vs baseline under the identical procedure.
    """

    import numpy as np

    from .reference_generation import register_matte_to_clay
    from .rendering import render_mesh_views

    render = render_mesh_views(
        mesh, size=int(render_size), azimuths=[float(pose[0])],
        elevation=float(pose[1]))[0]
    registered, _ = register_matte_to_clay(source_rgba, render)
    photo = np.asarray(registered.convert("RGBA"), dtype=np.float32)
    photo_mask = photo[:, :, 3] > 128
    both = photo_mask & _render_foreground(render)
    if int(both.sum()) < 256:
        return float("nan")
    photo_lab = _lab(registered)
    render_lab = _lab(render)
    return float(np.linalg.norm(
        (photo_lab - render_lab)[both], axis=1).mean())


def _front_brightness(mesh: Any, pose: Tuple[float, float],
                      render_size: int) -> float:
    import numpy as np

    from .rendering import render_mesh_views

    render = render_mesh_views(
        mesh, size=int(render_size), azimuths=[float(pose[0])],
        elevation=float(pose[1]))[0]
    foreground = _render_foreground(render)
    if not foreground.any():
        return float("nan")
    return float(_lab(render)[:, :, 0][foreground].mean())


def _seam_ratio(render: Any, *, edge_delta_e: float,
                min_extent_ratio: float) -> float:
    """Fraction of foreground pixels on LONG coherent strong edges.

    Adjacent-pixel deltaE above `edge_delta_e` marks a step edge; connected
    components whose spatial extent exceeds `min_extent_ratio` of the
    foreground diagonal count as seams. Texture detail forms short or
    fragmented edges below the extent bar; a view-handoff tone step forms
    one long contour.
    """

    import numpy as np
    from scipy import ndimage

    lab = _lab(render)
    foreground = _render_foreground(render)
    if not foreground.any():
        return 0.0
    # Exclude the silhouette rim: the object/background step is not a seam.
    interior = ndimage.binary_erosion(foreground, iterations=3)
    dx = np.zeros(lab.shape[:2], dtype=np.float32)
    dy = np.zeros(lab.shape[:2], dtype=np.float32)
    dx[:, :-1] = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
    dy[:-1, :] = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
    edges = interior & (np.maximum(dx, dy) > float(edge_delta_e))
    if not edges.any():
        return 0.0
    labels, count = ndimage.label(edges, structure=np.ones((3, 3), bool))
    rows, cols = np.nonzero(foreground)
    diagonal = float(np.hypot(rows.max() - rows.min(),
                              cols.max() - cols.min())) or 1.0
    seam_pixels = 0
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        extent = float(np.hypot(sl[0].stop - sl[0].start,
                                sl[1].stop - sl[1].start))
        if extent >= float(min_extent_ratio) * diagonal:
            component = labels[sl]
            seam_pixels += int(np.count_nonzero(component))
    return float(seam_pixels) / float(np.count_nonzero(foreground))


def _max_seam_ratio(mesh: Any, render_size: int, *, edge_delta_e: float,
                    min_extent_ratio: float) -> Tuple[float, Optional[str]]:
    from .rendering import render_mesh_views

    worst = 0.0
    worst_label: Optional[str] = None
    for elevation in (10.0, 50.0):
        renders = render_mesh_views(
            mesh, size=int(render_size), azimuths=[0.0, 90.0, 180.0, -90.0],
            elevation=elevation)
        for azimuth, render in zip((0, 90, 180, -90), renders):
            ratio = _seam_ratio(render, edge_delta_e=edge_delta_e,
                                min_extent_ratio=min_extent_ratio)
            if ratio > worst:
                worst = ratio
                worst_label = f"az{azimuth}_el{int(elevation)}"
    return worst, worst_label


def evaluate_generated_bake(
    baseline_mesh: Any,
    candidate_mesh: Any,
    *,
    source_rgba: Any,
    source_pose: Tuple[float, float] = (0.0, 0.0),
    render_size: int = 512,
    fidelity_slack: float = 2.0,
    brightness_slack: float = 4.0,
    seam_budget: float = 0.02,
    edge_delta_e: float = 18.0,
    seam_min_extent_ratio: float = 0.12,
) -> Dict[str, Any]:
    """A/B non-regression verdict for a generated-references bake.

    Returns a report dict: `accepted` (bool), per-check `metrics`
    (baseline/candidate values), and human-readable `reasons` for every
    failed check. The caller ships the baseline when `accepted` is False.
    """

    import math

    metrics: Dict[str, Any] = {}
    reasons: List[str] = []

    base_fid = _photo_fidelity(baseline_mesh, source_rgba, source_pose, render_size)
    cand_fid = _photo_fidelity(candidate_mesh, source_rgba, source_pose, render_size)
    metrics["photo_fidelity_delta_e"] = {
        "baseline": round(base_fid, 2), "candidate": round(cand_fid, 2),
        "max_allowed": round(base_fid + float(fidelity_slack), 2)}
    if math.isfinite(base_fid) and math.isfinite(cand_fid):
        if cand_fid > base_fid + float(fidelity_slack):
            reasons.append(
                f"photo fidelity regressed: candidate deltaE {cand_fid:.2f} "
                f"vs baseline {base_fid:.2f} (+{fidelity_slack} allowed)")

    base_l = _front_brightness(baseline_mesh, source_pose, render_size)
    cand_l = _front_brightness(candidate_mesh, source_pose, render_size)
    metrics["front_brightness_l"] = {
        "baseline": round(base_l, 2), "candidate": round(cand_l, 2),
        "min_allowed": round(base_l - float(brightness_slack), 2)}
    if math.isfinite(base_l) and math.isfinite(cand_l):
        if cand_l < base_l - float(brightness_slack):
            reasons.append(
                f"front brightness regressed: candidate L {cand_l:.1f} vs "
                f"baseline {base_l:.1f} (-{brightness_slack} allowed)")

    base_seam, base_where = _max_seam_ratio(
        baseline_mesh, render_size, edge_delta_e=edge_delta_e,
        min_extent_ratio=seam_min_extent_ratio)
    cand_seam, cand_where = _max_seam_ratio(
        candidate_mesh, render_size, edge_delta_e=edge_delta_e,
        min_extent_ratio=seam_min_extent_ratio)
    # ABSOLUTE budget, not a multiple of the baseline: a multiplicative
    # allowance lets a bad baseline launder a worse candidate (measured on
    # the chair: baseline 0.102, candidate 0.138 passed a 1.5x rule while
    # being visibly seam-broken). Surface fraction newly covered by long
    # step edges is damage in absolute terms, whatever the starting point.
    seam_ceiling = base_seam + float(seam_budget)
    metrics["seam_ratio"] = {
        "baseline": round(base_seam, 5), "candidate": round(cand_seam, 5),
        "max_allowed": round(seam_ceiling, 5),
        "baseline_worst_view": base_where, "candidate_worst_view": cand_where}
    if cand_seam > seam_ceiling:
        reasons.append(
            f"seam steps regressed: candidate long-edge ratio {cand_seam:.4f} "
            f"({cand_where}) vs baseline {base_seam:.4f} "
            f"(ceiling {seam_ceiling:.4f})")

    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics}
