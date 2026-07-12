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
  photo sees only at grazing incidence). The pose is resolved from the
  bake's own stats (see `evaluate_generated_bake`): measuring at a
  hardcoded (0,0) on a pose-estimated subject charges ~9 dE of pure pose
  error to both sides and flipped a car verdict (+0.88 true regression
  read as +4.03).
- brightness: foreground L mean of the source-pose render. Catches the
  darkened bake the harness measured on the chair.
- composition tone damage: low-band |delta L| between candidate and
  baseline over a turnaround render set, counted only beyond the
  pipeline's own sanctioned tone-adjustment amplitude (see
  `_tone_field_damage`). Catches mis-toned and mis-registered view
  handoffs WHEREVER they face, including classes the previous long-edge
  step detector measurably missed (below).
- composition hue damage: the chroma analog (see `_chroma_field_damage`)
  — coherent low-band hue rotation on mutually saturated surface. The
  tone axis is L-based and measurably blind to constant-L chroma damage
  (a 30-deg hue-rotated back reference passed every other axis with
  fidelity IMPROVING; integrator program 2026-07, /tmp/fix3).

HISTORY — why the long-edge seam ratio no longer refuses (2026-07 fix
program, fixtures in /tmp/fix1): the original third axis counted
foreground pixels on long coherent strong-edge components (the v2-era
chair incident presented as a sharp deltaE ~42 step: baseline 0.102,
candidate 0.138). Measured on the current stack it mis-ranks BOTH sides:
(a) gradient-domain compositing smooths true tone damage below any step
threshold — a +25 L mis-toned chair back scored seam delta +0.013
(gradient) / +0.016 (legacy) and an 8%-content-shifted back scored
-0.005, ALL inside the +0.02 budget the axis refused candidates with;
(b) on low-coverage subjects the references legitimately add real long
contours to previously blank fill (the car's roof glass frame), which
the axis reads as damage — the labeled-good car candidate scored +0.029
and was refused while its fidelity and brightness both IMPROVED. The
axis is kept in `metrics` for observability but no longer votes; the
texture-space handoff ledger (`handoff_seams`) is likewise reported
only, because it is structurally blind to reference-to-fill frontiers
(measured: the chair candidate bake records boundary_texels == 0 while
carrying an obvious back-handoff regression — its front and back
observed sets never touch in UV).

Every check is A/B under an identical procedure, so registration and
renderer biases cancel; the contract is monotone non-regression, not an
absolute quality bar. Thresholds are calibrated on the labeled fixture
set (accepts: owl, portrait, starship, chair-with-clean-back, chair with
a +12 L back — inside the pipeline's own tone-match clamp — and the car
at 2048 and 1024; refusals: +/-25 L mis-toned chair backs, the +25 L
one under both compositors, and an 8%-content-shifted back) — margins
in CHANGELOG and /tmp/fix1/report.md.
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


def _turnaround_renders(mesh: Any, render_size: int) -> List[Tuple[str, Any]]:
    """The gate's fixed 8-view render set (labels carry az/el)."""

    from .rendering import render_mesh_views

    out: List[Tuple[str, Any]] = []
    for elevation in (10.0, 50.0):
        renders = render_mesh_views(
            mesh, size=int(render_size), azimuths=[0.0, 90.0, 180.0, -90.0],
            elevation=elevation)
        for azimuth, render in zip((0, 90, 180, -90), renders):
            out.append((f"az{azimuth}_el{int(elevation)}", render))
    return out


def _masked_smooth(field: Any, mask: Any, sigma: float) -> Any:
    """Gaussian low-pass that ignores non-mask pixels (normalized
    convolution), so background never bleeds into the foreground band."""

    import numpy as np
    from scipy import ndimage

    weights = mask.astype(np.float32)
    numerator = ndimage.gaussian_filter(field * weights, sigma)
    denominator = ndimage.gaussian_filter(weights, sigma)
    return numerator / np.maximum(denominator, 1e-6)


def _tone_field_damage(candidate_render: Any, baseline_render: Any, *,
                       floor_l: float, smoothing_sigma: float) -> Optional[Dict[str, float]]:
    """Low-band composition damage of one view, candidate vs baseline.

    Statistic: smooth each side's L channel over its own foreground
    (normalized convolution at `smoothing_sigma`), take the signed delta
    on the co-foreground interior, and integrate the excess above
    `floor_l` PER DIRECTION (mean of max(|dL| - floor, 0) per interior
    pixel, darkening and brightening reported separately).

    Why this shape (each choice measured on the /tmp/fix1 fixtures):

    - LOW BAND: gradient-domain compositing converts handoff damage from
      sharp steps into smooth region-scale tone displacement — a step
      detector sees nothing, but the displaced REGION is exactly what the
      user sees. Smoothing also cancels texture detail and specular
      structure, which references legitimately change (at sigma 8 the
      good car keeps 6% of its surface above the floor from crisp glass
      edges alone; at sigma 24 that collapses to 0.8% while true damage
      stays at 38%/4%).
    - FLOOR at the sanctioned-adjustment amplitude: every legitimate tone
      change of CORRECT content is bounded by the pipeline's own clamps —
      generation tone-matching caps at 15 L (`match_tone_lab
      max_shift_l`), and the in-bake harmonization/delight/leveling
      corrections operate within ~10 L. Their sum (25 L) is the largest
      low-band displacement correct content can legitimately acquire, so
      excess beyond it is alien content or broken composition, not
      re-toning.
    - DIRECTIONAL: the two directions have different legitimate
      amplitudes because baseline fill is dark-biased by construction
      (fill propagates a dark base under a luminance floor; the parity
      audit measured the 1024 car mottling 5.8x darker) — references
      legitimately BRIGHTEN large fill regions (measured up to +39 L on
      the portrait's neck-side fill, beyond-floor integral 0.22-0.23 on
      two labeled accepts), while legitimate DARKENING beyond the floor
      measures identically ZERO on all seven labeled accepts (dark true
      content, like the car's glass roof over dark-red mottle, stays
      within the floor). Alien-dark content (misprojected structure,
      handoff shadowing — the incident direction) is therefore refused
      at a far tighter budget than brightening.
    - INTEGRAL (area x excess), not area fraction or p95: violent-local
      and broad-moderate damage both register; a p95 form measured
      non-separating (good car@1024 21.8 vs shifted chair 23.3).

    KNOWN LIMIT (tracked, no fixture yet): a subject whose true unseen
    side is far DARKER than its front (black-backed white object) would
    have a too-bright baseline fill, and a correct generated back would
    read as legitimate darkening beyond the floor — the gate would ship
    the baseline (safe, but a missed rescue). Constants for that class
    wait for a measured fixture, per the no-untested-constants rule.

    Returns None when the co-foreground interior is too small to measure
    (a view neither bake covers carries no evidence either way).
    """

    import numpy as np
    from scipy import ndimage

    candidate_fg = _render_foreground(candidate_render)
    baseline_fg = _render_foreground(baseline_render)
    both = candidate_fg & baseline_fg
    interior = ndimage.binary_erosion(both, iterations=4)
    if int(interior.sum()) < 500:
        return None
    candidate_l = _lab(candidate_render)[:, :, 0]
    baseline_l = _lab(baseline_render)[:, :, 0]
    delta_l = (
        _masked_smooth(candidate_l, candidate_fg, smoothing_sigma)
        - _masked_smooth(baseline_l, baseline_fg, smoothing_sigma))[interior]
    darkening = np.maximum(-delta_l - float(floor_l), 0.0)
    brightening = np.maximum(delta_l - float(floor_l), 0.0)
    return {
        "darken_damage": float(darkening.mean()),
        "brighten_damage": float(brightening.mean()),
        "area_over_floor": float((np.abs(delta_l) > float(floor_l)).mean()),
        "p95_abs_delta_l": float(np.percentile(np.abs(delta_l), 95)),
    }


def _chroma_field_damage(candidate_render: Any, baseline_render: Any, *,
                         hue_floor_deg: float, sat_floor: float,
                         smoothing_sigma: float) -> Optional[Dict[str, float]]:
    """Low-band coherent HUE-ROTATION damage of one view (the chroma
    analog of `_tone_field_damage`, same directional-budget doctrine).

    Statistic: smooth each side's LAB a/b fields over its own foreground
    (same normalized convolution and sigma as the tone axis), and on the
    co-foreground interior where BOTH sides stay saturated (smoothed
    chroma >= `sat_floor`), integrate the ab-vector ANGLE beyond
    `hue_floor_deg`, weighted by the saturated-area fraction:
    `mean(max(angle - floor, 0)) * sat_frac`.

    Why the ANGLE and not ab displacement (measured, integrator program
    /tmp/fix3): legitimate reference work REPLACES content — dark fill
    mottle becomes saturated paint/glass — so the raw smoothed-ab
    displacement of the labeled-good car measures 4.1-5.4 while a
    hue-rotated back reference measures 5.2: non-separating (inverted on
    half the fixtures). The rotation signature is angular: on mutually
    saturated surface the good car's worst view measures 0.20 deg-mass,
    the whole labeled-accept fleet <= 0.46, while a 30-deg hue-rotated
    back reference measures 2.31, a 20-deg one 1.53, and 30-deg rotated
    sides 6.61.

    Constants:
    - `hue_floor_deg` 10: the sanctioned hue-drift amplitude — the
      generation tone-match clamps ab shifts at 10 (`match_tone_lab
      max_shift_ab`), which at the anchor-class chroma (~50) is an ~11
      deg rotation ceiling; in-bake harmonization stays inside it.
    - `sat_floor` 15 (smoothed chroma): below it hue is numerically
      meaningless (gray subjects: starship measures sat_frac 0.00,
      portrait 0.14 — the sat_frac weight makes the axis structurally
      quiet on low-chroma subjects instead of noisy).
    - budget 1.0 (caller kwarg): 2.2x above the worst labeled accept
      (chair_clean 0.460; rebuild noise measured <= 0.05 on the car
      recheck pair), 1.5x under the mildest synthesized damage (20 deg,
      2x the sanctioned drift), 2.3x under the 30-deg incident class.

    KNOWN LIMIT (tracked, mirror of the tone axis's dark-back limit): a
    subject whose true unseen side is a DIFFERENT hue family than its
    front (two-tone vehicles) would read as hue damage and ship the
    baseline — safe direction, missed rescue. No fixture of that class
    exists yet.

    Production role: generated references pass the per-view two-key
    gates (which refuse whole-view palette rotation: the 30-deg back
    measures consensus 31.1 / cloud 28.5, both far over their lines),
    but caller-provided references in `rebake_bundle` reach the bake
    UNGATED — this axis is the only chroma defense on that path.
    """

    import numpy as np
    from scipy import ndimage

    candidate_fg = _render_foreground(candidate_render)
    baseline_fg = _render_foreground(baseline_render)
    both = candidate_fg & baseline_fg
    interior = ndimage.binary_erosion(both, iterations=4)
    if int(interior.sum()) < 500:
        return None
    candidate_lab = _lab(candidate_render)
    baseline_lab = _lab(baseline_render)
    cand_a = _masked_smooth(candidate_lab[:, :, 1], candidate_fg,
                            smoothing_sigma)[interior]
    cand_b = _masked_smooth(candidate_lab[:, :, 2], candidate_fg,
                            smoothing_sigma)[interior]
    base_a = _masked_smooth(baseline_lab[:, :, 1], baseline_fg,
                            smoothing_sigma)[interior]
    base_b = _masked_smooth(baseline_lab[:, :, 2], baseline_fg,
                            smoothing_sigma)[interior]
    saturated = (np.hypot(cand_a, cand_b) >= float(sat_floor)) \
        & (np.hypot(base_a, base_b) >= float(sat_floor))
    sat_frac = float(saturated.mean())
    if not saturated.any():
        return {"hue_damage": 0.0, "hue_p95_deg": 0.0, "sat_frac": 0.0}
    dot = (cand_a * base_a + cand_b * base_b)[saturated]
    det = (base_a * cand_b - base_b * cand_a)[saturated]
    angle = np.degrees(np.abs(np.arctan2(det, dot)))
    return {
        "hue_damage": float(
            np.maximum(angle - float(hue_floor_deg), 0.0).mean() * sat_frac),
        "hue_p95_deg": float(np.percentile(angle, 95)),
        "sat_frac": sat_frac,
    }


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


def _max_seam_ratio(renders: Sequence[Tuple[str, Any]], *,
                    edge_delta_e: float,
                    min_extent_ratio: float) -> Tuple[float, Optional[str]]:
    worst = 0.0
    worst_label: Optional[str] = None
    for label, render in renders:
        ratio = _seam_ratio(render, edge_delta_e=edge_delta_e,
                            min_extent_ratio=min_extent_ratio)
        if ratio > worst:
            worst = ratio
            worst_label = label
    return worst, worst_label


def _resolve_gate_pose(
    source_pose: Optional[Tuple[float, float]],
    baseline_stats: Optional[Dict[str, Any]],
    candidate_stats: Optional[Dict[str, Any]],
) -> Tuple[Tuple[float, float], str]:
    """The pose fidelity/brightness are measured at, with its provenance.

    Explicit `source_pose` wins (external capture fact / reproduction
    pins). Otherwise the baseline bake's own recorded pose is the truth —
    the bake estimated it from the same photo and mesh the gate is about
    to render (measured identical between candidate and baseline bakes:
    the estimator is deterministic in photo+mesh). The candidate's stats
    are the fallback so a caller that only has one stats dict still gets
    the true pose. (0,0) only remains for stats-less callers, and the
    provenance string in the metrics makes that visible instead of
    silent — the measured failure mode was a diagnostic harness reading a
    (0,0) verdict without noticing the pose never came from the bake.
    """

    if source_pose is not None:
        return ((float(source_pose[0]), float(source_pose[1])), "explicit")
    for stats, origin in ((baseline_stats, "baseline_stats"),
                          (candidate_stats, "candidate_stats")):
        row = (stats or {}).get("source_pose") or {}
        if row:
            return ((float(row.get("azimuth_deg") or 0.0),
                     float(row.get("elevation_deg") or 0.0)), origin)
    return ((0.0, 0.0), "default")


def evaluate_generated_bake(
    baseline_mesh: Any,
    candidate_mesh: Any,
    *,
    source_rgba: Any,
    source_pose: Optional[Tuple[float, float]] = None,
    baseline_stats: Optional[Dict[str, Any]] = None,
    candidate_stats: Optional[Dict[str, Any]] = None,
    render_size: int = 512,
    fidelity_slack: float = 2.0,
    brightness_slack: float = 4.0,
    tone_damage_floor_l: float = 25.0,
    tone_darken_budget: float = 0.03,
    tone_brighten_budget: float = 0.7,
    tone_smoothing_px: float = 24.0,
    hue_floor_deg: float = 10.0,
    hue_sat_floor: float = 15.0,
    hue_damage_budget: float = 1.0,
    edge_delta_e: float = 18.0,
    seam_min_extent_ratio: float = 0.12,
) -> Dict[str, Any]:
    """A/B non-regression verdict for a generated-references bake.

    `baseline_stats` / `candidate_stats` are the bake stats dicts; when
    given they supply the fidelity pose (the bake's own estimated source
    pose) and the texture-space handoff ledger for the record. Passing an
    explicit `source_pose` overrides the stats (external capture fact).

    Constants, each carried by a measured separation (fixture program in
    /tmp/fix1/report.md; labeled accepts = owl / portrait / starship /
    chair-clean / chair +12 L (inside the pipeline's own tone-match
    clamp, so refusing it would refuse the pipeline's sanctioned
    behavior) / car @2048+1024; labeled refusals = +/-25 L mis-toned and
    8%-content-shifted chair backs, mis-tone under both compositors):

    - `tone_damage_floor_l` 25: the pipeline's sanctioned tone-adjustment
      budget (generation tone-match clamps at 15 L; in-bake
      harmonization/leveling within ~10 L) — legitimate re-toning of
      correct content cannot exceed their sum, so only the excess counts
      as damage.
    - `tone_darken_budget` 0.03: every labeled accept measures 0.000
      darkening damage (7/7 — legitimate reference work brightens the
      dark-biased fill, it does not darken beyond the floor), the
      weakest darkening refusal (content shift) measures 0.100 and the
      -25 L incident direction 0.33: the budget sits 3.3x under the
      weakest refusal and is pure noise headroom above the accepts.
    - `tone_brighten_budget` 0.7: legitimate brightening is structurally
      larger — baseline fill is dark-biased by construction (dark
      propagation under a luminance floor; the fill floor measures its
      own lift at 0.11-0.14), so true content replacing fill brightens
      far beyond any darkening (measured legit maxima: 0.224 portrait
      neck fill, 0.228 chair +12 L). Budget = 3x the worst labeled
      accept, 4x under the +25 L incident class (2.80-2.82).
    - `tone_smoothing_px` 24 at `render_size` 512 (scales with it):
      texture/specular detail that references legitimately change
      decorrelates below this scale, region-scale damage survives it
      (measured sweep sigma 8/16/24: accept-vs-refuse separation
      1.5x/2.4x/4.7x).
    - `hue_floor_deg` 10 / `hue_sat_floor` 15 / `hue_damage_budget` 1.0:
      the chroma analog (the L-based tone axis is measurably blind to
      constant-L hue rotation — a 30-deg rotated back reference passed
      every prior axis; integrator program, /tmp/fix3). See
      `_chroma_field_damage` for the measured calibration (accepts
      <= 0.46, 20-deg damage 1.53, 30-deg 2.31-6.61).

    Returns a report dict: `accepted` (bool), per-check `metrics`
    (baseline/candidate values), and human-readable `reasons` for every
    failed check. The caller ships the baseline when `accepted` is False.
    """

    import math

    metrics: Dict[str, Any] = {}
    reasons: List[str] = []

    gate_pose, pose_origin = _resolve_gate_pose(
        source_pose, baseline_stats, candidate_stats)
    metrics["source_pose"] = {
        "azimuth_deg": gate_pose[0], "elevation_deg": gate_pose[1],
        "origin": pose_origin}

    base_fid = _photo_fidelity(baseline_mesh, source_rgba, gate_pose, render_size)
    cand_fid = _photo_fidelity(candidate_mesh, source_rgba, gate_pose, render_size)
    metrics["photo_fidelity_delta_e"] = {
        "baseline": round(base_fid, 2), "candidate": round(cand_fid, 2),
        "max_allowed": round(base_fid + float(fidelity_slack), 2)}
    if math.isfinite(base_fid) and math.isfinite(cand_fid):
        if cand_fid > base_fid + float(fidelity_slack):
            reasons.append(
                f"photo fidelity regressed: candidate deltaE {cand_fid:.2f} "
                f"vs baseline {base_fid:.2f} (+{fidelity_slack} allowed)")

    base_l = _front_brightness(baseline_mesh, gate_pose, render_size)
    cand_l = _front_brightness(candidate_mesh, gate_pose, render_size)
    metrics["front_brightness_l"] = {
        "baseline": round(base_l, 2), "candidate": round(cand_l, 2),
        "min_allowed": round(base_l - float(brightness_slack), 2)}
    if math.isfinite(base_l) and math.isfinite(cand_l):
        if cand_l < base_l - float(brightness_slack):
            reasons.append(
                f"front brightness regressed: candidate L {cand_l:.1f} vs "
                f"baseline {base_l:.1f} (-{brightness_slack} allowed)")

    # One render pass feeds both the tone-damage vote and the
    # informational seam metric.
    baseline_renders = _turnaround_renders(baseline_mesh, render_size)
    candidate_renders = _turnaround_renders(candidate_mesh, render_size)

    sigma = float(tone_smoothing_px) * float(render_size) / 512.0
    worst = {"darken": (0.0, None), "brighten": (0.0, None)}
    worst_area = 0.0
    measured_views = 0
    worst_hue = (0.0, None)
    worst_hue_p95 = 0.0
    hue_views = 0
    for (label, candidate_render), (_, baseline_render) in zip(
            candidate_renders, baseline_renders):
        row = _tone_field_damage(
            candidate_render, baseline_render,
            floor_l=float(tone_damage_floor_l), smoothing_sigma=sigma)
        if row is None:
            continue
        measured_views += 1
        worst_area = max(worst_area, row["area_over_floor"])
        if row["darken_damage"] > worst["darken"][0]:
            worst["darken"] = (row["darken_damage"], label)
        if row["brighten_damage"] > worst["brighten"][0]:
            worst["brighten"] = (row["brighten_damage"], label)
        chroma_row = _chroma_field_damage(
            candidate_render, baseline_render,
            hue_floor_deg=float(hue_floor_deg),
            sat_floor=float(hue_sat_floor), smoothing_sigma=sigma)
        if chroma_row is not None:
            hue_views += 1
            worst_hue_p95 = max(worst_hue_p95, chroma_row["hue_p95_deg"])
            if chroma_row["hue_damage"] > worst_hue[0]:
                worst_hue = (chroma_row["hue_damage"], label)
    metrics["composition_tone_damage"] = {
        "darken_worst": round(worst["darken"][0], 4),
        "darken_worst_view": worst["darken"][1],
        "darken_max_allowed": float(tone_darken_budget),
        "brighten_worst": round(worst["brighten"][0], 4),
        "brighten_worst_view": worst["brighten"][1],
        "brighten_max_allowed": float(tone_brighten_budget),
        "floor_l": float(tone_damage_floor_l),
        "worst_area_over_floor": round(worst_area, 4),
        "measured_views": measured_views}
    if measured_views and worst["darken"][0] > float(tone_darken_budget):
        reasons.append(
            f"composition tone damage (darkening): candidate low-band L "
            f"drops {worst['darken'][0]:.3f} per-pixel mean beyond the "
            f"sanctioned-adjustment floor ({tone_damage_floor_l} L) at "
            f"{worst['darken'][1]}, budget {tone_darken_budget}")
    if measured_views and worst["brighten"][0] > float(tone_brighten_budget):
        reasons.append(
            f"composition tone damage (brightening): candidate low-band L "
            f"rises {worst['brighten'][0]:.3f} per-pixel mean beyond the "
            f"sanctioned-adjustment floor ({tone_damage_floor_l} L) at "
            f"{worst['brighten'][1]}, budget {tone_brighten_budget}")
    metrics["composition_hue_damage"] = {
        "worst": round(worst_hue[0], 4),
        "worst_view": worst_hue[1],
        "max_allowed": float(hue_damage_budget),
        "floor_deg": float(hue_floor_deg),
        "sat_floor": float(hue_sat_floor),
        "worst_p95_deg": round(worst_hue_p95, 2),
        "measured_views": hue_views}
    if hue_views and worst_hue[0] > float(hue_damage_budget):
        reasons.append(
            f"composition hue damage: candidate low-band hue rotates "
            f"{worst_hue[0]:.3f} deg-mass beyond the sanctioned drift "
            f"floor ({hue_floor_deg} deg) on saturated surface at "
            f"{worst_hue[1]}, budget {hue_damage_budget}")

    # Long-edge seam ratio: OBSERVABILITY ONLY (see module docstring for
    # the measured mis-ranking that retired its vote). Kept in the record
    # so fleet drift stays visible and future recalibrations have data.
    base_seam, base_where = _max_seam_ratio(
        baseline_renders, edge_delta_e=edge_delta_e,
        min_extent_ratio=seam_min_extent_ratio)
    cand_seam, cand_where = _max_seam_ratio(
        candidate_renders, edge_delta_e=edge_delta_e,
        min_extent_ratio=seam_min_extent_ratio)
    metrics["seam_ratio"] = {
        "baseline": round(base_seam, 5), "candidate": round(cand_seam, 5),
        "baseline_worst_view": base_where, "candidate_worst_view": cand_where,
        "votes": False}

    # Texture-space handoff ledger, for the record (no vote: measured
    # blind to reference-to-fill frontiers — chair candidate carries an
    # obvious back handoff at boundary_texels == 0).
    if candidate_stats is not None:
        metrics["handoff_seams"] = {
            "candidate": (candidate_stats or {}).get("handoff_seams"),
            "votes": False}

    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics}
