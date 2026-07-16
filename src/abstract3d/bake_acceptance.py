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
  fidelity IMPROVING; integrator program 2026-07, /tmp/fix3). The drift
  charge is judged as TWO populations via a source-evidence veto:
  candidate hue on the photo's own saturated hue band is subject
  evidence, never damage (the axis was measured false-refusing a
  correct reference by comparing it against baseline FILL hue on
  never-witnessed surface; hue program 2026-07, /tmp/hue1).

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

ARTIFACT BATTERY (2026-07-14 program, /tmp/afix2): the same turnaround
render set additionally feeds the measured artifact-class detectors of
`artifact_gates` under the same directional A/B doctrine — a candidate
must not ADD artifact mass the baseline does not carry. One axis votes
(added foreign pale-blotch component: the image-in-image stamp payload
and the white rim-blotch class, budget 3.5x above the worst legitimate
pair and 3.2x under the rebuilt stamp incident); the wash / patch-block
/ mottle detectors are recorded with loud warnings because the corpus
denies them zero-false-fire margins (full reasoning in
`artifact_gates`'s module docstring). The battery is proven
non-interfering on all pinned fixture pairs (fix1's 12, hue1's live
pairs, fix3's chroma rotations — verdicts unchanged).

CATASTROPHIC-BASELINE REGIME (2026-07-15 program, /tmp/xfix2): the A/B
doctrine PRESUPPOSES the baseline carries subject evidence. The
measured x-wing incident broke that assumption: the source pose failed
(coverage 0.0095 — 99% of the shipped baseline is propagated fill),
the baseline itself failed the single-view sanity floors, and the gate
still used it as the A/B reference — refusing a healthy candidate
(4 accepted references, IoU 0.81-0.94, battery quiet) on the tone
BRIGHTENING axis (0.814 vs budget 0.7), because brightening a
99%-dark-fill baseline is exactly what correct references DO. The
directional budgets were calibrated on baselines with witnessed
coverage 0.112-0.83 (fix1's labeled fleet); at witnessed collapse the
brighten statistic measures the fill deficit, not damage (correct
rescues measure 0.81-1.03, a +25 L mis-tone 10.85 — same sign, no
usable boundary).

The regime boundary is the REGISTRATION-COLLAPSE line already
calibrated in `artifact_gates.registration_floor_check` (source-view
coverage < 0.10 AND capture efficiency < 0.25, AND-ed because each
margin alone is thin): the healthy/pinned fleet measures source
coverage >= 0.1088 with the one sub-0.10 live bundle (car_final,
0.0572) rescued by its efficiency 0.3273, while the measured
catastrophes sit at 0.0096/0.090 (x-wing), 0.0498/0.166 (v4 ghost) and
0.013/0.030 (the broken-chair guards) — a >= 1.9x gap on the AND. The
single-view sanity verdicts of BOTH stats dicts (the floors signal the
runtime already records) are computed once here and recorded in
`metrics["baseline_regime"]` so callers thread them instead of
recomputing. Deliberately NOT the sanity-floors verdict itself as the
vote boundary: the pinned v7 car baseline fails the 0.12 total floor
at 0.112 while the fix1 program PROVED the A/B axes calibrated at that
coverage (its labeled-accept margins are the calibration table) — the
floors mark user-visible degradation, the collapse line marks where
A/B semantics measurably die.

Under the catastrophic regime (per-axis analysis measured in
/tmp/xfix2/report.md):

- RESCUE PRECONDITION: the candidate must measurably fix the collapse
  — its TOTAL observed coverage must pass the sanity floor (0.12) and
  exceed the baseline's (references exist to add witnessed surface; a
  candidate that does not is the both-broken case and ships the
  baseline + degraded exactly as today). Source-view floors are
  INHERITED (both bakes share the broken source registration) and
  recorded as unfixable-by-references; the pose lane owns them.
- photo fidelity A/B and front brightness A/B: KEEP VOTING unchanged —
  the photo is external truth and the baseline side only cancels
  registration/renderer residue; the witnessed subset must not
  regress (measured: x-wing +0.89 dE within the +2.0 slack).
- tone DARKENING A/B: KEEPS VOTING unchanged — baseline fill is
  dark-biased by construction, so content darker than fill beyond the
  25 L floor is alien in any regime (measured 0.000 on every
  catastrophic-lane accept).
- tone BRIGHTENING A/B: RECORDED ONLY (loud warning, no vote) — the
  poisoned axis (measured false-firing 0.81-1.03 on correct rescues).
  No absolute replacement exists: the photo's own L band cannot bound
  it (renders are headlight-lit and honestly-bright unseen surface is
  legitimate — the correct x-wing candidate measures 6.26 of
  above-band mass at floor 5 while a +25 L mis-tone measures 0.28:
  inverted), so the bright-side pure-L mis-tone on never-witnessed
  surface is a DOCUMENTED LIMIT of the lane (bounded by the floors,
  battery, hue and fidelity votes that still hold).
- composition hue: switches to the ABSOLUTE source-band form
  (`_band_distance_damage`): the A/B drift charge is fill-vs-content
  confound writ large at 98% fill (measured 1.04 on a CORRECT
  broken-chair rescue — over the 1.0 budget), while the candidate's
  own hue vs the photo band needs no baseline. Floor 10 deg (the same
  sanctioned drift), budget 0.15: catastrophic-lane accepts measure
  <= 0.035, a +25 L mis-tone's gamut-bend measures 0.284 (1.9x over)
  and a 30-deg rotated back that DOMINATES the bake measures 5.18
  (35x). No-band photos keep the legacy A/B charge (fail-closed).
- NEW mirror-pair consistency (`_mirror_pair_damage`, catastrophic
  only): geometry-symmetric subjects (score >= 0.95 from the bake's
  own `symmetry_completion` stats; fleet measures 0.966-0.985) whose
  TEXTURE breaks left-right low-band tone symmetry carry displaced
  content — candidate-internal evidence needing no baseline. az90 vs
  mirrored az-90 smoothed-L delta beyond a 15 L floor, budget 1.0:
  fleet accepts measure <= 0.516 (starship, honest texture
  asymmetry), the 8%-content-shifted back measures 1.496-1.499 in
  BOTH baseline regimes (lateral displacement breaks symmetry by
  construction). Catches the mis-registration class the collapse
  removed from the A/B darken axis's reach.
- artifact battery: unchanged A/B vote — measured on both broken
  baselines: fill carries 0.0000 blotch mass, so added == absolute in
  this lane by measurement, not by reinterpretation.
- tone-vs-photo band exceedance: RECORDED (non-voting) — measured
  non-separating in both directions (subject backs legitimately
  depart the photo band; the dark side overlaps the car-class's own
  sub-band glass/shadow mass), kept in metrics for fixture-driven
  recalibration.

KNOWN LIMITS of the lane (measured, documented): (a) the bright-side
pure-L mis-tone above; (b) a -25 L dark mis-tone under DEGRADED photo
evidence (the occluded-rim guard) is indistinguishable from the
honestly-dark-back accept class (portrait back ref measures -25.9 L
from its photo) on every absolute axis measured — it ships as a
floors-passing, battery-quiet, hue-consistent bake (the healthy lane
keeps refusing the class decisively via darken-A/B 0.33, and the
confounded axis that refused it before ALSO refused correct rescues).
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


def _source_hue_band(source_rgba: Any, *, sat_floor: float,
                     smoothing_sigma: float) -> Optional[Dict[str, float]]:
    """The source photo's own saturated hue evidence, measured the way
    the chroma axis measures the renders: normalized-convolution
    smoothing of the LAB a/b fields over the photo foreground, interior
    erosion, saturation floor, then circular [q2, q98] quantiles around
    the circular mean hue.

    Why SMOOTHED photo fields and not raw pixels (measured, /tmp/hue1):
    the axis compares low-band render fields, so the license must be the
    photo's low-band hue spread. Raw pixel quantiles are inflated by
    speckle/highlight noise (v7 car raw band 23 deg wide vs smoothed
    8.8) and let a 20-deg rotated reference collapse under the pinned
    refusal budget (vetoed mass 0.129 at margin 2 vs 1.312 smoothed).
    `smoothing_sigma` is the axis's render sigma (at 512 px); it is
    rescaled by the photo's own foreground extent so the photo low-band
    matches the render low-band regardless of photo resolution.

    Returns None when the photo carries no usable saturated mass
    (< 256 smoothed interior pixels, e.g. the gray starship): callers
    then keep the legacy single-population behavior — fail-closed, the
    chroma hole stays shut for colorless subjects.
    """

    import numpy as np
    from scipy import ndimage

    rgba = np.asarray(source_rgba.convert("RGBA"), dtype=np.uint8)
    foreground = rgba[:, :, 3] > 128
    if int(foreground.sum()) < 500:
        return None
    rows, cols = np.nonzero(foreground)
    extent = float(max(int(rows.max() - rows.min()),
                       int(cols.max() - cols.min()), 1))
    # The gate's turnaround subject spans ~85% of its 512 frame; scale
    # the photo sigma so both sides measure the same low band.
    sigma = float(smoothing_sigma) * extent / (0.85 * 512.0)
    lab = _lab(source_rgba)
    a = _masked_smooth(lab[:, :, 1], foreground, sigma)
    b = _masked_smooth(lab[:, :, 2], foreground, sigma)
    interior = ndimage.binary_erosion(foreground, iterations=4)
    saturated = interior & (np.hypot(a, b) >= float(sat_floor))
    if int(saturated.sum()) < 256:
        return None
    hue = np.degrees(np.arctan2(b[saturated], a[saturated]))
    rad = np.radians(hue)
    mu = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))
    centered = (hue - mu + 180.0) % 360.0 - 180.0
    lo, hi = np.percentile(centered, [2.0, 98.0])
    return {"mu_deg": round(mu, 2), "lo_deg": round(float(lo), 2),
            "hi_deg": round(float(hi), 2),
            "saturated_px": int(saturated.sum())}


def _chroma_field_damage(candidate_render: Any, baseline_render: Any, *,
                         hue_floor_deg: float, sat_floor: float,
                         smoothing_sigma: float,
                         source_band: Optional[Dict[str, float]] = None,
                         evidence_margin_deg: float = 3.0,
                         ) -> Optional[Dict[str, float]]:
    """Low-band coherent HUE-ROTATION damage of one view (the chroma
    analog of `_tone_field_damage`, same directional-budget doctrine),
    judged as TWO populations via a source-evidence veto.

    Statistic: smooth each side's LAB a/b fields over its own foreground
    (same normalized convolution and sigma as the tone axis), and on the
    co-foreground interior where BOTH sides stay saturated (smoothed
    chroma >= `sat_floor`), integrate the ab-vector ANGLE beyond
    `hue_floor_deg`, weighted by the saturated-area fraction — but a
    pixel's excess only counts when the candidate's own hue sits OFF the
    source photo's hue evidence (`source_band`, see `_source_hue_band`)
    by more than `evidence_margin_deg`:
    `mean(max(angle - floor, 0) * off_evidence) * sat_frac`.

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

    Why the SOURCE-EVIDENCE VETO (two-population principle; measured
    incident + calibration in /tmp/hue1): the drift angle is only
    meaningful where the BASELINE side carries subject content. On
    never-witnessed surface the baseline is propagated fill mottle, and
    the axis was measured charging a CORRECT red-car reference 1.869
    deg-mass (over the 1.0 budget) purely for disagreeing with baseline
    fill hue at az180 — a false refusal of the exact rescue this gate
    exists to permit. Geometric observed-vs-fill classification is not
    available to the gate (bake stats carry no texture-space masks), and
    a projector replica would drift from bake truth; instead each pixel
    is classified by EVIDENCE CONSISTENCY: candidate hue within the
    photo's own saturated hue band (+ margin) is the subject's evidence
    and can never be damage, whatever the baseline shows there; hue off
    the evidence keeps the full drift charge. A hue-rotated reference
    sits off-band by construction, so the veto cannot reopen the
    original chroma hole — measured: correct-refs pairs collapse 1.870
    -> 0.008 and 0.973 -> 0.000 while the pinned rotations hold at
    1.31 (20 deg), 1.98 (30-deg back), 6.51 (30-deg sides).

    Constants:
    - `hue_floor_deg` 10: the sanctioned hue-drift amplitude — the
      generation tone-match clamps ab shifts at 10 (`match_tone_lab
      max_shift_ab`), which at the anchor-class chroma (~50) is an ~11
      deg rotation ceiling; in-bake harmonization stays inside it.
    - `sat_floor` 15 (smoothed chroma): below it hue is numerically
      meaningless (gray subjects: starship measures sat_frac 0.00,
      portrait 0.14 — the sat_frac weight makes the axis structurally
      quiet on low-chroma subjects instead of noisy).
    - `evidence_margin_deg` 3 (caller kwarg): the measured saddle.
      Accept-population vetoed masses at margin 3: live car pair 0.008,
      fresh-draw pair 0.000, fleet accepts <= 0.157 (6.4x under budget);
      refusal population: 20-deg rotation 1.312 (1.3x over budget),
      30-deg 1.98-6.51. At margin 2 the live pair keeps 0.265 charged
      (photo-band edge noise); at margin 4 the 20-deg refusal thins to
      1.11x. A 15-deg rotation measures 0.623 — under budget but 4x
      above every accept: the refusal boundary sits between 1.5x and
      2x the sanctioned drift, exactly the class doctrine.
    - budget 1.0 (caller kwarg): with the veto, 6.4x above the worst
      labeled accept (car 0.157; rebuild noise measured <= 0.05 on the
      car recheck pair), 1.3x under the mildest synthesized damage
      (20 deg, 2x the sanctioned drift), 2x under the 30-deg incident
      class.

    KNOWN LIMITS (tracked): (a) mirror of the tone axis's dark-back
    limit — a subject whose true unseen side is a DIFFERENT hue family
    than its front reads as off-evidence hue damage and ships the
    baseline (safe direction, missed rescue; no fixture yet); (b) a
    rotation that lands INSIDE a broad multi-hue photo band is licensed
    by the veto — on such subjects the per-view two-key lane remains the
    defense (the 30-deg back measures consensus 31.1 > 16 there); (c) a
    colorless photo yields no band, so the legacy single-population
    charge applies unchanged (fail-closed).

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
        return {"hue_damage": 0.0, "hue_damage_raw": 0.0,
                "hue_p95_deg": 0.0, "sat_frac": 0.0}
    dot = (cand_a * base_a + cand_b * base_b)[saturated]
    det = (base_a * cand_b - base_b * cand_a)[saturated]
    angle = np.degrees(np.abs(np.arctan2(det, dot)))
    excess = np.maximum(angle - float(hue_floor_deg), 0.0)
    raw_damage = float(excess.mean() * sat_frac)
    damage = raw_damage
    if source_band is not None:
        cand_hue = np.degrees(np.arctan2(cand_b, cand_a))[saturated]
        centered = (cand_hue - float(source_band["mu_deg"])
                    + 180.0) % 360.0 - 180.0
        band_dist = np.maximum(
            np.maximum(centered - float(source_band["hi_deg"]),
                       float(source_band["lo_deg"]) - centered), 0.0)
        off_evidence = band_dist > float(evidence_margin_deg)
        damage = float((excess * off_evidence).mean() * sat_frac)
    return {
        "hue_damage": damage,
        "hue_damage_raw": raw_damage,
        "hue_p95_deg": float(np.percentile(angle, 95)),
        "sat_frac": sat_frac,
    }


def _source_tone_band(source_rgba: Any, *,
                      smoothing_sigma: float) -> Optional[Dict[str, float]]:
    """The L analog of `_source_hue_band`: smoothed-L [q2, q98] over the
    photo foreground (same normalized convolution, same extent-rescaled
    sigma), the subject's own low-band tone evidence.

    RECORD-ONLY consumer for now: measured on the fleet
    (/tmp/xfix2/measure_abs.json) the band cannot carry a vote in either
    direction — renders are headlight-lit and never-witnessed surface
    legitimately departs the photo span (the correct x-wing candidate
    carries 6.26 of above-band mass at floor 5 while a +25 L mis-tone
    carries 0.28; the dark side overlaps the car class's own sub-band
    glass/shadow mass at every floor). Kept in the catastrophic-regime
    metrics so future fixture classes can calibrate against real data.
    """

    import numpy as np
    from scipy import ndimage

    rgba = np.asarray(source_rgba.convert("RGBA"), dtype=np.uint8)
    foreground = rgba[:, :, 3] > 128
    if int(foreground.sum()) < 500:
        return None
    rows, cols = np.nonzero(foreground)
    extent = float(max(int(rows.max() - rows.min()),
                       int(cols.max() - cols.min()), 1))
    sigma = float(smoothing_sigma) * extent / (0.85 * 512.0)
    smooth_l = _masked_smooth(_lab(source_rgba)[:, :, 0], foreground, sigma)
    interior = ndimage.binary_erosion(foreground, iterations=4)
    if int(interior.sum()) < 256:
        return None
    values = smooth_l[interior]
    lo, hi = np.percentile(values, [2.0, 98.0])
    return {"lo_l": round(float(lo), 2), "hi_l": round(float(hi), 2),
            "median_l": round(float(np.median(values)), 2)}


def _band_distance_damage(render: Any, band: Dict[str, float], *,
                          floor_deg: float, sat_floor: float,
                          smoothing_sigma: float,
                          tone_band: Optional[Dict[str, float]] = None,
                          ) -> Optional[Dict[str, float]]:
    """ABSOLUTE hue judgment of one view against the source photo's own
    hue evidence — the catastrophic-regime replacement for the A/B drift
    charge, whose baseline side is propagated fill at witnessed
    collapse (measured charging a CORRECT broken-chair rescue 1.04 over
    the 1.0 budget purely for disagreeing with fill hue).

    Statistic: on the interior foreground where the candidate's own
    smoothed chroma clears `sat_floor`, integrate the circular distance
    OUTSIDE the photo band beyond `floor_deg`, weighted by the
    saturated-area fraction — `mean(max(band_dist - floor, 0)) *
    sat_frac`. Same smoothing, same saturation floor, same sanctioned
    drift floor (10 deg) as the A/B axis; only the reference changes
    (photo evidence instead of the untrustworthy baseline).

    Measured (/tmp/xfix2, catastrophic-lane populations): accepts
    <= 0.035 (correct broken-chair rescue; the x-wing candidate 0.000),
    +25 L mis-tone gamut-bend 0.284, a 30-deg hue-rotated back
    DOMINATING the bake 5.18. When `tone_band` is given, the L
    exceedance of the same interior is returned for the record
    (non-voting, see `_source_tone_band`).
    """

    import numpy as np
    from scipy import ndimage

    foreground = _render_foreground(render)
    interior = ndimage.binary_erosion(foreground, iterations=4)
    if int(interior.sum()) < 500:
        return None
    lab = _lab(render)
    a = _masked_smooth(lab[:, :, 1], foreground, smoothing_sigma)[interior]
    b = _masked_smooth(lab[:, :, 2], foreground, smoothing_sigma)[interior]
    out: Dict[str, float] = {}
    if tone_band is not None:
        smooth_l = _masked_smooth(lab[:, :, 0], foreground,
                                  smoothing_sigma)[interior]
        out["tone_below_band"] = float(np.maximum(
            float(tone_band["lo_l"]) - smooth_l, 0.0).mean())
        out["tone_above_band"] = float(np.maximum(
            smooth_l - float(tone_band["hi_l"]), 0.0).mean())
    saturated = np.hypot(a, b) >= float(sat_floor)
    sat_frac = float(saturated.mean())
    out["sat_frac"] = sat_frac
    if not saturated.any():
        out["hue_damage"] = 0.0
        return out
    hue = np.degrees(np.arctan2(b[saturated], a[saturated]))
    centered = (hue - float(band["mu_deg"]) + 180.0) % 360.0 - 180.0
    band_dist = np.maximum(
        np.maximum(centered - float(band["hi_deg"]),
                   float(band["lo_deg"]) - centered), 0.0)
    out["hue_damage"] = float(
        np.maximum(band_dist - float(floor_deg), 0.0).mean() * sat_frac)
    return out


def _mirror_pair_damage(render_a: Any, render_b: Any, *,
                        floor_l: float,
                        smoothing_sigma: float) -> Optional[float]:
    """Left-right texture-symmetry damage: smoothed-L delta between one
    side view and the MIRRORED opposite side view, integrated beyond
    `floor_l` on the co-foreground interior.

    A geometry-symmetric subject (score gated by the caller from the
    bake's own `symmetry_completion` stats) renders mirror-consistent
    silhouettes under the head-light rig; texture that breaks low-band
    left-right symmetry beyond the floor is displaced/foreign content —
    candidate-internal evidence that needs neither the baseline nor a
    photo view of the damaged surface. Measured (/tmp/xfix2): fleet
    accepts <= 0.516 at floor 15 (the starship's honest texture
    asymmetry; portrait hair 0.428; every car/owl/chair accept
    <= 0.30), the 8%-content-shifted chair back 1.496-1.499 under BOTH
    baseline regimes — the mis-registration class signature, structural
    because lateral displacement breaks symmetry by construction.
    """

    import numpy as np
    from scipy import ndimage

    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL is a hard dependency
        return None
    mirrored = render_b.transpose(Image.FLIP_LEFT_RIGHT)
    fg_a = _render_foreground(render_a)
    fg_b = _render_foreground(mirrored)
    interior = ndimage.binary_erosion(fg_a & fg_b, iterations=4)
    if int(interior.sum()) < 500:
        return None
    delta = np.abs(
        _masked_smooth(_lab(render_a)[:, :, 0], fg_a, smoothing_sigma)
        - _masked_smooth(_lab(mirrored)[:, :, 0], fg_b, smoothing_sigma)
    )[interior]
    return float(np.maximum(delta - float(floor_l), 0.0).mean())


def _source_view_row(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The SOURCE view's stats row (same resolution rule as
    `evaluate_single_view_bake`)."""

    view_rows = (stats or {}).get("observed_view_stats") or []
    return next(
        (row for row in view_rows if row.get("index") == 1
         or row.get("label") in ("front", "source", "view_01")),
        view_rows[0] if view_rows else None)


def _baseline_collapse(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Registration-collapse test of the BASELINE's source row — the
    catastrophic-regime boundary.

    Reuses the corpus-calibrated floors of
    `artifact_gates.registration_floor_check` (source coverage < 0.10
    AND capture efficiency < 0.25; AND-ed because each margin alone is
    thin): healthy/pinned baselines measure >= 0.1088 coverage (the one
    sub-0.10 live bundle, car_final at 0.0572, is rescued by its 0.3273
    efficiency), the measured catastrophes 0.0096/0.090 (x-wing),
    0.0498/0.166 (v4), 0.013/0.030 (broken-chair guards). Missing stats
    resolve to NOT collapsed — the status-quo healthy path, recorded as
    such.
    """

    from .artifact_gates import (REGISTRATION_EFFICIENCY_FLOOR,
                                 REGISTRATION_SOURCE_COVERAGE_FLOOR)

    row = _source_view_row(stats)
    coverage = None if row is None else row.get("coverage_ratio")
    efficiency = None if row is None else row.get("capture_efficiency")
    fired = (coverage is not None and efficiency is not None
             and float(coverage) < REGISTRATION_SOURCE_COVERAGE_FLOOR
             and float(efficiency) < REGISTRATION_EFFICIENCY_FLOOR)
    return {"fired": bool(fired), "source_coverage": coverage,
            "capture_efficiency": efficiency,
            "coverage_floor": REGISTRATION_SOURCE_COVERAGE_FLOOR,
            "efficiency_floor": REGISTRATION_EFFICIENCY_FLOOR}


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
    hue_evidence_margin_deg: float = 3.0,
    catastrophic_hue_budget: float = 0.15,
    mirror_floor_l: float = 15.0,
    mirror_damage_budget: float = 1.0,
    mirror_symmetry_floor: float = 0.95,
    edge_delta_e: float = 18.0,
    seam_min_extent_ratio: float = 0.12,
) -> Dict[str, Any]:
    """A/B non-regression verdict for a generated-references bake.

    `baseline_stats` / `candidate_stats` are the bake stats dicts; when
    given they supply the fidelity pose (the bake's own estimated source
    pose), the texture-space handoff ledger for the record, AND the
    baseline-regime signal: a baseline whose source registration
    collapsed (see `_baseline_collapse` and the module docstring's
    catastrophic-regime section) is not a trustworthy A/B reference, and
    the axes that presuppose one switch to absolute judgment. Passing an
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
    - `hue_evidence_margin_deg` 3: the source-evidence veto's margin —
      the drift charge only stands where the candidate's hue also sits
      off the source photo's own saturated hue band (two-population
      judgment; the axis was measured false-refusing a correct
      reference at 1.869 by comparing it against baseline FILL hue on
      never-witnessed surface). Measured saddle (/tmp/hue1): vetoed
      accepts <= 0.157 and the live false-refusal pair 0.008, while the
      pinned rotations keep 1.31 (20 deg) / 1.98 / 6.51 (30 deg). See
      `_chroma_field_damage`.

    Catastrophic-regime constants (all measured, /tmp/xfix2/report.md;
    only consulted when the baseline's source registration collapsed):

    - `catastrophic_hue_budget` 0.15: the absolute source-band hue
      vote (`_band_distance_damage`) — catastrophic-lane accepts
      measure <= 0.035 (4.3x under), the +25 L mis-tone gamut-bend
      0.284 (1.9x over), a bake-dominating 30-deg rotation 5.18 (35x).
    - `mirror_floor_l` 15 / `mirror_damage_budget` 1.0: left-right
      texture-symmetry damage (`_mirror_pair_damage`) — fleet accepts
      <= 0.516 (starship's honest asymmetry; 1.9x under), the
      8%-content-shifted back 1.496-1.499 (1.5x over) in BOTH regimes.
    - `mirror_symmetry_floor` 0.95: the axis only votes when the
      bake's own `symmetry_completion` geometry score clears it (fleet
      measures 0.966-0.985); asymmetric or unscored geometry abstains,
      recorded.

    Returns a report dict: `accepted` (bool), per-check `metrics`
    (baseline/candidate values, plus `baseline_regime` with both sanity
    verdicts and the collapse signal), and human-readable `reasons` for
    every failed check. The caller ships the baseline when `accepted`
    is False.
    """

    import math

    metrics: Dict[str, Any] = {}
    reasons: List[str] = []
    warnings: List[str] = []

    gate_pose, pose_origin = _resolve_gate_pose(
        source_pose, baseline_stats, candidate_stats)
    metrics["source_pose"] = {
        "azimuth_deg": gate_pose[0], "elevation_deg": gate_pose[1],
        "origin": pose_origin}

    # BASELINE REGIME (module docstring, catastrophic-regime section):
    # the sanity floors of BOTH stats dicts are computed once here (pure
    # stats math — callers reuse the recorded verdicts instead of
    # recomputing) and the collapse boundary decides which judgment the
    # relative axes get. Missing stats resolve to healthy: the status
    # quo, recorded as such.
    collapse = _baseline_collapse(baseline_stats)
    catastrophic = bool(collapse["fired"])
    regime = "catastrophic" if catastrophic else "healthy"

    def _sanity_of(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Only judge floors where coverage evidence exists — a stats
        # dict without it (older persisted subsets) records None, not a
        # hollow "accepted".
        if not stats or (stats.get("observed_coverage_ratio") is None
                         and not stats.get("observed_view_stats")):
            return None
        return evaluate_single_view_bake(stats)

    baseline_sanity = _sanity_of(baseline_stats)
    candidate_sanity = _sanity_of(candidate_stats)
    metrics["baseline_regime"] = {
        "regime": regime,
        "collapse": collapse,
        "baseline_sanity": baseline_sanity,
        "candidate_sanity": candidate_sanity,
    }
    if catastrophic:
        # RESCUE PRECONDITION: references exist to add witnessed
        # surface; a candidate that does not measurably fix the
        # coverage collapse is the both-broken case and ships the
        # baseline + degraded exactly as today. Source-view floors are
        # inherited by construction (both bakes share the broken source
        # registration) and stay recorded in the sanity verdicts above;
        # the pose lane owns their repair.
        base_total = (baseline_stats or {}).get("observed_coverage_ratio")
        cand_total = (candidate_stats or {}).get("observed_coverage_ratio")
        coverage_floor = 0.12  # evaluate_single_view_bake's total floor
        fixes = (cand_total is not None
                 and float(cand_total) >= coverage_floor
                 and (base_total is None
                      or float(cand_total) > float(base_total)))
        metrics["baseline_regime"]["candidate_fixes_collapse"] = {
            "value": bool(fixes), "candidate_total": cand_total,
            "baseline_total": base_total, "min_allowed": coverage_floor}
        if not fixes:
            reasons.append(
                "catastrophic-baseline regime: candidate does not fix "
                f"the witnessed-coverage collapse (total {cand_total} vs "
                f"floor {coverage_floor}, baseline {base_total}) — "
                "both bakes are broken, shipping the baseline")

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

    # The subject's own hue evidence, for the chroma axis's
    # two-population veto. None (colorless photo) keeps the legacy
    # single-population charge — fail-closed.
    try:
        source_band = _source_hue_band(
            source_rgba, sat_floor=float(hue_sat_floor),
            smoothing_sigma=float(tone_smoothing_px))
    except Exception:
        source_band = None

    sigma = float(tone_smoothing_px) * float(render_size) / 512.0
    worst = {"darken": (0.0, None), "brighten": (0.0, None)}
    worst_area = 0.0
    measured_views = 0
    worst_hue = (0.0, None)
    worst_hue_raw = (0.0, None)
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
            sat_floor=float(hue_sat_floor), smoothing_sigma=sigma,
            source_band=source_band,
            evidence_margin_deg=float(hue_evidence_margin_deg))
        if chroma_row is not None:
            hue_views += 1
            worst_hue_p95 = max(worst_hue_p95, chroma_row["hue_p95_deg"])
            if chroma_row["hue_damage"] > worst_hue[0]:
                worst_hue = (chroma_row["hue_damage"], label)
            if chroma_row["hue_damage_raw"] > worst_hue_raw[0]:
                worst_hue_raw = (chroma_row["hue_damage_raw"], label)
    metrics["composition_tone_damage"] = {
        "darken_worst": round(worst["darken"][0], 4),
        "darken_worst_view": worst["darken"][1],
        "darken_max_allowed": float(tone_darken_budget),
        "brighten_worst": round(worst["brighten"][0], 4),
        "brighten_worst_view": worst["brighten"][1],
        "brighten_max_allowed": float(tone_brighten_budget),
        "floor_l": float(tone_damage_floor_l),
        "worst_area_over_floor": round(worst_area, 4),
        "measured_views": measured_views,
        # Directional vote flags (regime honesty): darkening votes in
        # both regimes (fill is dark-biased by construction — darker
        # than fill beyond the floor is alien everywhere); brightening
        # presupposes the baseline carries subject tone and is
        # measurement-only when the baseline collapsed (measured
        # false-firing 0.81-1.03 on correct rescues of ~99%-fill
        # baselines: the x-wing incident).
        "darken_votes": True,
        "brighten_votes": not catastrophic}
    if measured_views and worst["darken"][0] > float(tone_darken_budget):
        reasons.append(
            f"composition tone damage (darkening): candidate low-band L "
            f"drops {worst['darken'][0]:.3f} per-pixel mean beyond the "
            f"sanctioned-adjustment floor ({tone_damage_floor_l} L) at "
            f"{worst['darken'][1]}, budget {tone_darken_budget}")
    if measured_views and worst["brighten"][0] > float(tone_brighten_budget):
        if not catastrophic:
            reasons.append(
                f"composition tone damage (brightening): candidate low-band L "
                f"rises {worst['brighten'][0]:.3f} per-pixel mean beyond the "
                f"sanctioned-adjustment floor ({tone_damage_floor_l} L) at "
                f"{worst['brighten'][1]}, budget {tone_brighten_budget}")
        else:
            warnings.append(
                f"catastrophic-baseline regime: brightening vs the "
                f"collapsed baseline measures {worst['brighten'][0]:.3f} "
                f"at {worst['brighten'][1]} (healthy-regime budget "
                f"{tone_brighten_budget}) — recorded, not voting: the "
                "baseline is ~all fill and brightening it is what "
                "correct references do")
    metrics["composition_hue_damage"] = {
        "worst": round(worst_hue[0], 4),
        "worst_view": worst_hue[1],
        "max_allowed": float(hue_damage_budget),
        "floor_deg": float(hue_floor_deg),
        "sat_floor": float(hue_sat_floor),
        "worst_p95_deg": round(worst_hue_p95, 2),
        "measured_views": hue_views,
        # Two-population observability: the pre-veto (legacy) mass and
        # the evidence band it was judged against. A large raw/vetoed
        # gap is the incident signature (baseline-fill hue confound).
        "worst_raw": round(worst_hue_raw[0], 4),
        "worst_raw_view": worst_hue_raw[1],
        "evidence_margin_deg": float(hue_evidence_margin_deg),
        "source_band": source_band,
        # In the catastrophic regime the A/B drift charge is the fill
        # confound writ large (98%+ of the co-foreground is fill) and
        # only the absolute source-band axis votes — except for
        # colorless photos (no band), where the legacy charge stays
        # fail-closed.
        "votes": (not catastrophic) or source_band is None}
    hue_ab_votes = (not catastrophic) or source_band is None
    if hue_views and worst_hue[0] > float(hue_damage_budget) and hue_ab_votes:
        reasons.append(
            f"composition hue damage: candidate low-band hue rotates "
            f"{worst_hue[0]:.3f} deg-mass beyond the sanctioned drift "
            f"floor ({hue_floor_deg} deg) on saturated surface off the "
            f"source photo's own hue evidence at "
            f"{worst_hue[1]}, budget {hue_damage_budget}")

    if catastrophic:
        # ABSOLUTE AXES (catastrophic regime only; module docstring +
        # /tmp/xfix2/report.md carry every margin). The candidate is
        # judged against the SOURCE PHOTO's own evidence and its own
        # internal symmetry — the collapsed baseline is not a reference.
        try:
            tone_band = _source_tone_band(
                source_rgba, smoothing_sigma=float(tone_smoothing_px))
        except Exception:
            tone_band = None
        worst_abs_hue = (0.0, None)
        worst_below = (0.0, None)
        worst_above = (0.0, None)
        abs_views = 0
        if source_band is not None:
            for label, candidate_render in candidate_renders:
                abs_row = _band_distance_damage(
                    candidate_render, source_band,
                    floor_deg=float(hue_floor_deg),
                    sat_floor=float(hue_sat_floor), smoothing_sigma=sigma,
                    tone_band=tone_band)
                if abs_row is None:
                    continue
                abs_views += 1
                if abs_row["hue_damage"] > worst_abs_hue[0]:
                    worst_abs_hue = (abs_row["hue_damage"], label)
                if abs_row.get("tone_below_band", 0.0) > worst_below[0]:
                    worst_below = (abs_row["tone_below_band"], label)
                if abs_row.get("tone_above_band", 0.0) > worst_above[0]:
                    worst_above = (abs_row["tone_above_band"], label)
        metrics["absolute_hue_damage"] = {
            "worst": round(worst_abs_hue[0], 4),
            "worst_view": worst_abs_hue[1],
            "max_allowed": float(catastrophic_hue_budget),
            "floor_deg": float(hue_floor_deg),
            "measured_views": abs_views,
            "votes": source_band is not None,
            "source_band": source_band}
        if abs_views and worst_abs_hue[0] > float(catastrophic_hue_budget):
            reasons.append(
                f"catastrophic-baseline regime: candidate hue sits "
                f"{worst_abs_hue[0]:.3f} deg-mass off the source photo's "
                f"own hue evidence beyond the sanctioned drift floor "
                f"({hue_floor_deg} deg) at {worst_abs_hue[1]}, budget "
                f"{catastrophic_hue_budget}")
        # Photo tone band exceedance: RECORDED ONLY (measured
        # non-separating in both directions — see _source_tone_band).
        metrics["absolute_tone_band"] = {
            "band": tone_band,
            "below_worst": round(worst_below[0], 4),
            "below_worst_view": worst_below[1],
            "above_worst": round(worst_above[0], 4),
            "above_worst_view": worst_above[1],
            "votes": False}

        # MIRROR-PAIR CONSISTENCY: candidate-internal evidence, gated
        # on the bake's own geometry-symmetry score.
        symmetry_score = ((candidate_stats or {}).get("symmetry_completion")
                          or {}).get("geometry_symmetry_score")
        mirror_applicable = (symmetry_score is not None and float(
            symmetry_score) >= float(mirror_symmetry_floor))
        worst_mirror = (0.0, None)
        mirror_views = 0
        if mirror_applicable:
            by_label = dict(candidate_renders)
            for elevation in ("el10", "el50"):
                left = by_label.get(f"az90_{elevation}")
                right = by_label.get(f"az-90_{elevation}")
                if left is None or right is None:
                    continue
                damage = _mirror_pair_damage(
                    left, right, floor_l=float(mirror_floor_l),
                    smoothing_sigma=sigma)
                if damage is None:
                    continue
                mirror_views += 1
                if damage > worst_mirror[0]:
                    worst_mirror = (damage, f"az90/az-90 {elevation}")
        metrics["mirror_consistency"] = {
            "worst": round(worst_mirror[0], 4),
            "worst_pair": worst_mirror[1],
            "max_allowed": float(mirror_damage_budget),
            "floor_l": float(mirror_floor_l),
            "geometry_symmetry_score": symmetry_score,
            "min_symmetry_score": float(mirror_symmetry_floor),
            "measured_pairs": mirror_views,
            "votes": mirror_applicable}
        if not mirror_applicable:
            warnings.append(
                "catastrophic-baseline regime: mirror-consistency axis "
                f"abstained (geometry symmetry score {symmetry_score} "
                f"below {mirror_symmetry_floor} or unrecorded)")
        if mirror_views and worst_mirror[0] > float(mirror_damage_budget):
            reasons.append(
                f"catastrophic-baseline regime: candidate texture breaks "
                f"left-right symmetry of its own geometry (score "
                f"{symmetry_score}) by {worst_mirror[0]:.3f} low-band L "
                f"beyond the {mirror_floor_l} L floor at "
                f"{worst_mirror[1]}, budget {mirror_damage_budget} — "
                "displaced/mis-registered content")

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

    # ARTIFACT BATTERY (measured detectors per shipped artifact class;
    # see artifact_gates for every constant's corpus margin). Same
    # renders, same A/B direction: only ADDED artifact mass can refuse.
    from .artifact_gates import evaluate_artifact_battery_ab, photo_reference

    try:
        photo_ref = photo_reference(source_rgba)
    except Exception:
        photo_ref = None
    battery = evaluate_artifact_battery_ab(
        candidate_renders, baseline_renders, photo_ref=photo_ref)
    metrics["artifact_battery"] = battery["metrics"]
    reasons.extend(battery["reasons"])
    warnings.extend(battery["warnings"])

    return {"accepted": not reasons, "reasons": reasons,
            "warnings": warnings, "metrics": metrics}
