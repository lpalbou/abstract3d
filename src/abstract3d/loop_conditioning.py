"""Loop geometry conditioning: the calibrated two-pass bust recipe as code.

Automates the reconstruct -> re-render -> re-generate loop validated on the
e18/e19/e20 experiment ladder (out/laurent-bust-redo/REPORT.md wave 3,
docs/research/viewgen_audit.md, 2026-07-21):

1. PASS 1 (caller: hunyuan3d_runtime) reconstructs a scaffold mesh from the
   front photo alone on the FLAGSHIP single-view checkpoint (e22 forensics
   2026-07-21: a single-front-tag 2mv draw shreds at every regime — the 2mv
   checkpoint fuses multiple tagged views; the flagship is the validated
   single-photo route and its scaffold registered onto the source photo at
   IoU 0.776 vs the 2mv single-tag draws' 0.415/0.226).
2. This module synthesizes the missing canonical views AGAINST that mesh's
   own clay renders through the IDENTITY conditioning route
   (`generate_reference_views(conditioning="identity")` — the composite
   two-panel canvas flips person identity on local editors, viewgen audit
   L1-L10) and gates each survivor: silhouette/material/specular (inside
   generate_reference_views); row consistency vs the view's OWN clay guide
   (MEASURED 2026-07-21: the front-photo-vs-profile axis INVERTS — the e20
   champion windowed set reads "inconsistent" while the striation set
   reads "consistent", because a pose-wrong three-quarter "profile" shares
   MORE frontal row structure with the front photo — so the clay is the
   row reference and the gate rejects the garbage class only); and POSE
   HONESTY, a head-band-IoU ruler measuring the view's TRUE azimuth
   (ported from scripts/experimental/viewgen_audit_{measure,analyze}.py —
   set A sold ~50 deg three-quarter views as 90 deg profiles; full-bust
   IoU is pose-blind on busts, 0.76+ at 40 deg off).
3. THE WINDOW LAW cuts every view (front included) to one anatomical span
   [head_top, shoulder + k*(shoulder - head_top)] so the 2mv
   bbox-recentring puts the same anatomy at the same normalized rows
   (scripts/experimental/harmonize_set.py; anchor spread 15-33% -> 3.4%).
   Windowed variants are CONDITIONING-ONLY: baking them starves the
   texture below the window (e18: sides at 6% coverage) — the
   split-consumer law the e18/e19/e20 ladder proved load-bearing.
4. Self-verification renders the surfaces the operator checks (oblique
   raking closeups; scripts/oblique_closeups.py) and runs the duplication
   autocorrelation flag (scripts/bust_assessment.py) on the front clay —
   the striated-mouth class shipped twice because front/45 verification
   angles hid it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Anatomical window (harmonize_set.py calibration, cycle-1 math): the cut
# row is shoulder + k*(shoulder - head_top). k=0.22 is the value the e18/
# e19/e20 cycles validated (shoulder_norm spread 0.034 across the set).
LOOP_WINDOW_K = 0.22
# Fraction of the max row width that counts as "shoulder reached" when
# anchoring the window (the shoulder flare is the widest structure of a
# bust at any azimuth — measurable in every view including the back).
WINDOW_SHOULDER_FRAC = 0.92
# Pose-honesty gate: |measured - declared| above this refuses the view
# from CONDITIONING (the 2mv tags are trained positions). Calibration
# (viewgen_audit.md sections 2/4): striation-class set A measured 22-40
# deg off (must refuse); accepted champion views 12.5 deg off (must
# pass). The view stays available to the texture bake at its MEASURED
# azimuth (projection is continuous).
POSE_MAX_DELTA_DEG = 20.0
# Pose-honesty ACCEPTANCE gate for freshly drawn views (viewgen bench
# 2026-07-21 section 6.3): a strict-passing candidate measured DECISIVELY
# more than this off its declared azimuth is rejected inside the draw
# ladder and the seed re-rolls — the winning recipe (arm E) passes 4/5
# slots at <= 10 deg, so a redraw is cheaper than shipping a lie.
# CALIBRATION = the RATIFIED two-key pose rule (strategy_v2 §stage
# contracts, stage E): refuse only when delta > 20 deg AND the measurement
# is decisive (iou_gap >= 0.10). An earlier 15-deg acceptance default was
# stricter than ratified and compounded with in-ladder measurement noise —
# the in-run ruler reads against the PASS-1 SCAFFOLD's clays (a rougher
# mesh than the e20-class instrument the bench calibrated on), so a
# stricter-than-ratified line rejects honest draws the conditioning fold
# below would have accepted. One rule, both judges (e22 forensics,
# 2026-07-21). Decisiveness (iou gap) still keys the gate — plateau
# argmaxes on bust backs are noise, not evidence (same two-key doctrine).
POSE_ACCEPT_MAX_DELTA_DEG = POSE_MAX_DELTA_DEG
# Refusal additionally requires the measured azimuth to BEAT the declared
# one decisively under head-band IoU: a plateau argmax is noise, not
# evidence. Measured on the shipped artifacts (2026-07-21): striation
# profiles gap 0.15-0.16 (refuse); champion profiles 0.048-0.076 (pass);
# bust backs 0.03-0.04 at 30-deg argmax offsets (plateau noise — e18/e19
# conditioned on such a back, healthy). 0.10 splits the bands.
POSE_GATE_MIN_IOU_GAP = 0.10
# Sweep-contrast floor for the RECORDED measurability note (profiles show
# ~0.15 contrast; near-flat sweeps carry little pose signal). Diagnostic
# only; the GATE key is the iou gap above.
POSE_SWEEP_CONTRAST_MIN = 0.08
# Duplication detector (bust_assessment.py calibration, 2026-07-20,
# moderngl 1024 px front clay renders, elevation 10 — the regime below):
# known-bad double-mouth mesh peaks at ratio 0.071, known-clean at 0.034;
# 0.05 splits them. A REVIEW signal calibrated on bust-class subjects.
DUPLICATION_PEAK_RATIO_THRESHOLD = 0.05
DUPLICATION_LAG_WINDOW = (0.03, 0.15)
FACE_REGION_FRACTION = 0.55
VERIFICATION_CLAY_SIZE = 1024
VERIFICATION_CLAY_ELEVATION_DEG = 10.0
# Oblique raking-light closeups: the verification surface the operator
# actually checks (oblique_closeups.py; elevation 12, 55 deg azimuths).
OBLIQUE_ANGLES = (("obl_l55", 55.0), ("obl_r55", 305.0))
OBLIQUE_RENDER_SIZE = 2048
OBLIQUE_ELEVATION_DEG = 12.0


def _subject_mask(image: Any) -> Any:
    """Boolean subject mask: alpha when it segments, else luminance distance
    from the top-left corner (harmonize_set.py's opaque-render contract)."""
    import numpy as np

    arr = np.asarray(image.convert("RGBA"), dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    if alpha.min() > 0.99:
        lum = arr[..., :3].mean(axis=2)
        return np.abs(lum - lum[0, 0]) > 12
    return alpha > 0.5


def window_anchors(image: Any) -> Dict[str, int]:
    """head_top / shoulder / bottom rows from the subject silhouette.
    shoulder = first row (scanning down) whose width reaches
    WINDOW_SHOULDER_FRAC of the row-wise max — where the shoulder flare
    has essentially arrived. Raises ValueError on empty subjects (loud)."""
    import numpy as np

    mask = _subject_mask(image)
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("no subject pixels: cannot anchor the window law")
    head_top = int(rows[0])
    widths = mask.sum(axis=1).astype(np.float64)
    candidates = np.flatnonzero(widths >= WINDOW_SHOULDER_FRAC * widths[rows].max())
    candidates = candidates[candidates > head_top]
    if candidates.size == 0:
        raise ValueError("no shoulder line found below the head top")
    return {"head_top": head_top, "shoulder": int(candidates[0]), "bottom": int(rows[-1])}


def window_view(image: Any, k: float = LOOP_WINDOW_K) -> Tuple[Any, Dict[str, Any]]:
    """Alpha-cut a view to the anatomical window [head_top, shoulder + k*span].

    The 2mv preprocessing recenters each view by its own alpha bbox; views
    cut at different torso depths put the SAME anatomy at DIFFERENT
    normalized rows and the shape DiT carves every hypothesis (the
    striated-mouth mechanism). Uniform scaling cannot fix a ratio; the
    shared window equalizes the ratio structure by construction. Raises
    ValueError on degenerate spans (loud, never a silent condition).
    """
    import numpy as np
    from PIL import Image

    anchors = window_anchors(image)
    span = anchors["shoulder"] - anchors["head_top"]
    if span <= 8:
        raise ValueError(f"degenerate head-to-shoulder span {span}")
    cut = int(round(anchors["shoulder"] + float(k) * span))
    arr = np.asarray(image.convert("RGBA")).copy()
    if cut + 1 < arr.shape[0]:
        arr[cut + 1 :, :, 3] = 0
    report = dict(anchors)
    report["cut_row"] = cut
    report["k"] = float(k)
    report["kept_ratio_shoulder"] = round(span / max(cut - anchors["head_top"], 1), 4)
    return Image.fromarray(arr, "RGBA"), report


def _head_band_rows(mask: Any) -> Tuple[int, int]:
    """(top, flare_row): the rows ABOVE the shoulder flare — the
    pose-bearing band. Uses the audit ruler's flare rule (first row whose
    smoothed width exceeds 1.45x the upper-head median), NOT the window
    anchor: 0.92-of-max sits at the flare's BOTTOM and would dilute the
    band with pose-blind torso mass."""
    import numpy as np

    widths = mask.sum(axis=1).astype(np.float64)
    rows = np.flatnonzero(widths > 0)
    if rows.size == 0:
        raise ValueError("empty mask: no head band")
    top, bottom = int(rows[0]), int(rows[-1])
    span = max(bottom - top, 1)
    sigma = max(2.0, span * 0.008)
    radius = max(1, int(round(3.0 * sigma)))
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    smoothed = np.convolve(widths, kernel / kernel.sum(), mode="same")
    band = smoothed[top + int(0.05 * span) : top + int(0.24 * span)]
    head_width = float(np.median(band)) if len(band) else float(smoothed[top])
    flare = bottom
    for row in range(top + int(0.20 * span), bottom):
        if smoothed[row] > 1.45 * head_width:
            flare = row
            break
    return top, int(flare)


def head_band_iou(view_rgba: Any, mesh: Any, azimuth_deg: float, *, render_size: int = 512) -> float:
    """Register the view onto the mesh's clay at `azimuth_deg`; IoU over
    the clay's HEAD BAND only (rows above the shoulder flare). Full-bust
    IoU is pose-blind on busts (a ~50 deg three-quarter view scores 0.76+
    against a 90 deg profile clay — the torso dominates); the head band is
    where azimuth changes the silhouette."""
    import numpy as np
    from PIL import Image

    from .reference_generation import clay_silhouette, register_matte_to_clay
    from .rendering import render_mesh_views

    view = view_rgba.convert("RGBA")
    if view.getchannel("A").getextrema()[0] >= 255:
        # No segmenting alpha (opaque render/photo): derive the matte —
        # a whole-frame "subject" scores every azimuth alike (noise).
        arr = np.asarray(view).copy()
        arr[..., 3] = np.where(_subject_mask(view), 255, 0).astype(np.uint8)
        view = Image.fromarray(arr, "RGBA")
    clay = render_mesh_views(
        mesh, size=int(render_size), azimuths=[float(azimuth_deg)], elevation=0.0
    )[0].convert("RGBA")
    clay_mask = np.asarray(clay_silhouette(clay))
    top, flare = _head_band_rows(clay_mask)
    registered, _stats = register_matte_to_clay(view, clay)
    view_mask = np.asarray(registered)[:, :, 3] > 128
    band = slice(top, max(flare, top + 1))
    union = int((view_mask[band] | clay_mask[band]).sum())
    if union == 0:
        return 0.0
    return float((view_mask[band] & clay_mask[band]).sum()) / union


def measure_view_azimuth(
    view_rgba: Any,
    mesh: Any,
    nominal_deg: float,
    *,
    render_size: int = 512,
) -> Dict[str, Any]:
    """Head-band-IoU pose ruler: sweep candidate azimuths around the
    nominal, refine around the argmax, report the measured azimuth. Sweep
    design (ported from viewgen_audit_analyze.estimate_azimuth): the coarse
    sweep reaches 55 deg INSIDE the nominal and 20 deg beyond it —
    under-rotation toward the front is the measured systematic failure
    direction. Only a `decisive` measurement (iou_gap floor) may act.
    """
    nominal = float(nominal_deg)
    sign = 1.0 if nominal >= 0 else -1.0
    if abs(abs(nominal) - 180.0) < 1e-6:
        candidates = [180.0 + delta for delta in range(-50, 55, 10)]
    else:
        magnitude = abs(nominal) - 55.0
        candidates = []
        while magnitude <= abs(nominal) + 20.0 + 1e-9:
            candidates.append(sign * magnitude)
            magnitude += 7.5
    candidates.append(nominal)  # the nominal must be scored, never defaulted
    scores: Dict[float, float] = {}
    for azimuth in candidates:
        scores[round(float(azimuth), 2)] = head_band_iou(
            view_rgba, mesh, float(azimuth), render_size=render_size
        )
    best = max(scores, key=lambda key: scores[key])
    for azimuth in (best - 5.0, best - 2.5, best + 2.5, best + 5.0):
        key = round(float(azimuth), 2)
        if key not in scores:
            scores[key] = head_band_iou(view_rgba, mesh, float(azimuth), render_size=render_size)
    best = max(scores, key=lambda key: scores[key])
    contrast = max(scores.values()) - min(scores.values())
    delta = abs(((best - nominal) + 180.0) % 360.0 - 180.0)
    gap = scores[best] - scores[round(nominal, 2)]
    return {
        "nominal_deg": nominal,
        "measured_deg": float(best),
        "delta_deg": round(delta, 2),
        "iou_at_measured": round(scores[best], 4),
        "iou_at_nominal": round(scores[round(nominal, 2)], 4),
        # The GATE key: how decisively the measured azimuth beats the
        # declared one (plateau argmaxes are noise; calibration above).
        "iou_gap": round(gap, 4),
        "decisive": bool(gap >= POSE_GATE_MIN_IOU_GAP),
        "sweep_contrast": round(contrast, 4),
        "measurable": bool(contrast >= POSE_SWEEP_CONTRAST_MIN),
        "sweep": {str(key): round(value, 4) for key, value in sorted(scores.items())},
    }


def build_pose_acceptance_gate(
    mesh: Any,
    *,
    max_delta_deg: float = POSE_ACCEPT_MAX_DELTA_DEG,
    render_size: int = 512,
) -> Optional[Any]:
    """Pose-honesty acceptance gate over `measure_view_azimuth` for the
    reference-generation draw ladder (`generate_reference_views(pose_gate=)`).

    Measures a candidate's TRUE azimuth against `mesh`'s own clay renders
    (the audit-grade head-band-IoU ruler — the instrument the viewgen bench
    cross-validated) and rejects it when the measurement is DECISIVE and
    more than `max_delta_deg` off the declared angle; the ladder then
    redraws. Two-key doctrine unchanged: plateau argmaxes (bust backs,
    rotation-symmetric subjects) never reject.

    No mesh, no gate: returns None so the caller records "pose_unmeasured"
    instead of gating blind. Elevated views abstain the same way — the
    ruler sweeps elevation-0 clays only.
    """

    if mesh is None:
        return None

    def pose_gate(view_rgba: Any, *, label: str, azimuth_deg: float,
                  elevation_deg: float = 0.0) -> Dict[str, Any]:
        del label  # part of the gate contract; the ruler is label-agnostic
        if abs(float(elevation_deg)) > 5.0:
            return {
                "measured": False, "passed": True,
                "note": ("pose_unmeasured: the azimuth ruler sweeps "
                         "elevation-0 clays only"),
            }
        try:
            pose = measure_view_azimuth(
                view_rgba, mesh, float(azimuth_deg),
                render_size=int(render_size))
        except Exception as exc:
            return {
                "measured": False, "passed": True,
                "note": f"pose_unmeasured: {type(exc).__name__}: {exc}",
            }
        verdict = dict(pose)
        verdict["measured"] = True
        verdict["max_delta_deg"] = float(max_delta_deg)
        rejected = bool(pose.get("decisive")) and (
            float(pose.get("delta_deg") or 0.0) > float(max_delta_deg))
        verdict["passed"] = not rejected
        if rejected:
            verdict["reason"] = (
                f"pose honesty: measured azimuth {pose['measured_deg']} deg "
                f"is {pose['delta_deg']} deg off the declared "
                f"{float(azimuth_deg)} deg (> {float(max_delta_deg)}, iou "
                f"gap {pose.get('iou_gap')} >= {POSE_GATE_MIN_IOU_GAP}); "
                "rejected — redrawing within the attempt ladder"
            )
        return verdict

    return pose_gate


# -- duplication-autocorrelation flag (bust_assessment.py port) --------------


def _subject_bbox(mask: Any) -> Tuple[int, int, int, int]:
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no subject pixels for the duplication detector")
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def duplication_flag(
    image: Any,
    *,
    face_fraction: float = FACE_REGION_FRACTION,
    lag_window: Tuple[float, float] = DUPLICATION_LAG_WINDOW,
    threshold: float = DUPLICATION_PEAK_RATIO_THRESHOLD,
) -> Dict[str, Any]:
    """Double-feature detector on a front clay render (general-purpose
    horizontal-feature duplication, not a mouth template): a duplicated
    horizontal feature shows as a strong non-zero-lag LOCAL autocorrelation
    peak of the face region's per-row edge-energy profile at the
    duplication distance; a healthy face decays monotonically there. A
    review flag, not a sole gate — legitimate paired structure
    (lips/glasses edges) can fire it (bust_assessment.py doctrine).
    """
    import numpy as np

    mask = _subject_mask(image)
    row0, row1, col0, col1 = _subject_bbox(mask)
    height = row1 - row0 + 1

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_mask = mask[:-1] & mask[1:]
    column_selector = np.zeros(mask.shape[1], dtype=bool)
    column_selector[col0:col1] = True
    face_row_end = min(row0 + int(round(face_fraction * height)), grad_y.shape[0])
    selected = edge_mask[row0:face_row_end] & column_selector[None, :]
    counts = selected.sum(axis=1)
    sums = (grad_y[row0:face_row_end] * selected).sum(axis=1)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).astype(np.float64)

    # Mean-removed autocorrelation normalized to lag 0 (DC otherwise
    # correlates every lag and hides the periodic signal).
    centered = profile - profile.mean()
    full = np.correlate(centered, centered, mode="full")
    autocorr = full[len(centered) - 1 :]
    autocorr = autocorr / autocorr[0] if len(autocorr) and autocorr[0] > 1e-12 else autocorr * 0.0

    low = max(2, int(round(lag_window[0] * height)))
    high = min(int(round(lag_window[1] * height)), len(autocorr) - 2)
    result: Dict[str, Any] = {
        "subject_height_px": int(height),
        "lag_window_px": [int(low), int(high)],
        "threshold": float(threshold),
        "peak_lag_px": None,
        "peak_lag_fraction_of_subject_height": None,
        "peak_ratio": None,
        "duplication_suspect": False,
    }
    if high < low:
        result["note"] = (
            "face-region profile too short for the lag window; subject too "
            "small at this render size for a duplication verdict"
        )
        return result
    # Local maxima only: a healthy profile decays monotonically through
    # the window, so its window max sits on the boundary and is no peak.
    peaks = [
        (float(autocorr[lag]), int(lag))
        for lag in range(low, high + 1)
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]
    ]
    if peaks:
        value, lag = max(peaks)
        result["peak_lag_px"] = lag
        result["peak_lag_fraction_of_subject_height"] = round(lag / height, 4)
        result["peak_ratio"] = round(value, 4)
        result["duplication_suspect"] = bool(value >= threshold)
    return result


# -- loop-view synthesis + gates (called by the hunyuan3d runtime) -----------

# Conditioning-clay smoothing (e22v3 refusal forensics, 2026-07-22): the
# identity i2i route conditions each draw on the scaffold's OWN clay
# render, and the generator faithfully REPRODUCES the clay's surface
# structure. A pass-1 scaffold's side/back clays carry heavy marching-
# cubes striation (photo-vs-front IoU 0.70-0.78 scaffolds still render
# sliced-bread sides), so every draw inherits paint-streak debris — the
# matte gate then honestly rejects 7/9 attempts and the loop refuses.
# The bench-E proof used a CLEAN mesh's clay (e20) and produced clean
# views at measured poses with the same generator/recipe. Taubin
# smoothing (volume-preserving, shrink-compensated) removes the streak
# texture while keeping the silhouette the gates and the pose ruler
# measure essentially unchanged. 30 iterations: the measured point where
# the streak texture is gone on the failing scaffold while lumps that
# carry real anatomy remain (0.1 s on a 59k-vertex mesh).
CONDITIONING_CLAY_TAUBIN_ITERATIONS = 30


def smooth_conditioning_mesh(mesh: Any, *, iterations: int = CONDITIONING_CLAY_TAUBIN_ITERATIONS) -> Any:
    """Taubin-smoothed COPY of the scaffold for clay-guided conditioning.

    Applied to the mesh the view-synthesis ladder renders its clay guides
    and rulers from — never to the mesh the caller keeps (pass 2 does not
    consume the scaffold's surface, only the synthesized views). Falls
    back to the raw mesh, loudly, if the smoother is unavailable."""
    import trimesh

    smoothed = trimesh.Trimesh(
        vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False
    )
    trimesh.smoothing.filter_taubin(
        smoothed, lamb=0.5, nu=-0.53, iterations=int(iterations)
    )
    return smoothed


def synthesize_loop_views(
    mesh0: Any,
    source_rgba: Any,
    *,
    owner: Any,
    angles: Sequence[Tuple[str, float, float]],
    seed: int,
    subject_hint: Optional[str],
    person_attested: bool,
    image_request: Mapping[str, Any],
    view_consistency: Tuple[Any, Any],
    image_generator: Optional[Any] = None,
    attempt_log_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Synthesize + gate the loop views against the pass-1 mesh.

    Returns a plan dict: `views` (per accepted view — full-span registered
    `rgba`, `raw_bytes` for the texture lane's replay, `windowed_rgba` for
    2mv conditioning only, declared + measured azimuths, `pose`,
    `row_consistency`, and two eligibility flags with refusal reasons:
    `conditioning_eligible` — pose-off or garbage-class views may not
    condition, the 2mv tags are trained positions — and `bake_eligible` —
    only the garbage class is refused from the bake; a pose-off view
    re-declares to its MEASURED azimuth there, where projection is
    continuous), `refgen_report`, `window` (per-view anchors + the
    front's), and `front_windowed`.

    Provider pinning: `image_request` must be the caller-resolved request
    for the REAL owner. An `owner=None` script bypass silently resolved a
    remote default editor once (measured incident: gate bypass + billing);
    the runtime resolves and threads the request explicitly so this
    surface never re-resolves with a missing owner.
    """
    from .reference_generation import generate_reference_views

    report_fn, _align_fn = view_consistency
    started = time.perf_counter()
    # Progressive attempt log (e22 forensics): every ladder attempt lands
    # as one JSON line the moment it completes, so a multi-minute-per-draw
    # synthesis is observable mid-run and a crash/refusal still leaves the
    # per-attempt reasons on disk. Append-per-event with an immediate
    # close: the writer must survive the process dying at any point.
    on_attempt = None
    if attempt_log_path is not None:
        import json
        from pathlib import Path

        log_path = Path(attempt_log_path)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_path = None
        if log_path is not None:
            def on_attempt(row: Mapping[str, Any], *, _path=log_path) -> None:
                with _path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(row), sort_keys=True,
                                            default=str) + "\n")

    # View-quality ACCEPTANCE gates (e22v2 forensics 2026-07-22): the
    # ladder accepted a wrong-identity, debris-covered side view — pose
    # honesty and row consistency are blind to matte cleanliness and to
    # WHO is in the picture. Both gates bind the SOURCE photo (the
    # subject's own baseline; no subject-specific rules) and reject
    # within the ladder so the seed re-rolls (view_gates.py carries the
    # calibration tables).
    from .view_gates import (
        build_matte_cleanliness_gate,
        build_subject_identity_gate,
    )

    # The ladder draws against the SMOOTHED scaffold's clays (see
    # smooth_conditioning_mesh): the raw pass-1 marching-cubes striation
    # is otherwise faithfully reproduced by the clay-guided generator and
    # the matte gate rejects every draw (e22v3 refusal, 2026-07-22). The
    # pose ruler + row consistency measure against the same smoothed
    # instrument (silhouettes essentially unchanged; one mesh, one frame).
    try:
        conditioning_mesh = smooth_conditioning_mesh(mesh0)
    except Exception:
        conditioning_mesh = mesh0  # loud in the report below, never fatal

    views, refgen_report = generate_reference_views(
        conditioning_mesh,
        source_rgba,
        owner=owner,
        angles=list(angles),
        subject_hint=subject_hint,
        seed=int(seed),
        image_request=dict(image_request),
        conditioning="identity",
        person_policy="proceed" if person_attested else "skip",
        image_generator=image_generator,
        # Pose-honesty ACCEPTANCE gate (viewgen bench 2026-07-21 section
        # 6.3): each draw is measured against the conditioning mesh's own
        # clays inside the ladder — a decisively pose-off candidate
        # redraws instead of ever becoming a view. The eligibility fold
        # below judges the accepted survivor under the SAME ratified
        # two-key rule (20 deg + decisive gap).
        pose_gate=build_pose_acceptance_gate(conditioning_mesh),
        matte_gate=build_matte_cleanliness_gate(source_rgba),
        identity_gate=build_subject_identity_gate(source_rgba),
        on_attempt=on_attempt,
    )
    refgen_report["conditioning_mesh"] = (
        f"taubin_smoothed_{CONDITIONING_CLAY_TAUBIN_ITERATIONS}"
        if conditioning_mesh is not mesh0
        else "#FALLBACK raw scaffold (smoothing unavailable)"
    )

    plan_views: List[Dict[str, Any]] = []
    window_reports: Dict[str, Any] = {}
    for view in views:
        label = str(view.get("label"))
        row: Dict[str, Any] = {
            "label": label,
            "azimuth_deg": float(view.get("azimuth_deg") or 0.0),
            "elevation_deg": float(view.get("elevation_deg") or 0.0),
            "rgba": view.get("rgba"),
            "raw_bytes": view.get("raw_bytes"),
            "clay_render": view.get("clay_render"),
            "seed": view.get("seed"),
            "raw_payload_md5": view.get("raw_payload_md5"),
            "conditioning_eligible": True,
            "bake_eligible": True,
        }
        # Matte/identity verdicts ride the accepted view (the ladder gates
        # already rejected failures — this is provenance for the bundle
        # record, plus a defensive belt: a riding FAILED verdict, which
        # only a bypassed ladder could produce, refuses both consumers).
        for gate_key, family in (("matte", "matte_debris"),
                                 ("identity", "subject_identity")):
            verdict = view.get(gate_key)
            if isinstance(verdict, Mapping):
                row[gate_key] = dict(verdict)
                if not verdict.get("passed", True):
                    row["conditioning_eligible"] = False
                    row["bake_eligible"] = False
                    row["conditioning_refusal"] = (
                        f"{family}: {verdict.get('reason') or 'gate failed'}"
                    )
        # Row consistency vs the view's OWN clay guide: garbage-class
        # rejection (scrambled/unrelated content measures <= ~0.40 under
        # the full remap search — the calibrated "inconsistent" band);
        # refused from BOTH consumers.
        clay = view.get("clay_render")
        if clay is not None:
            try:
                row_report = report_fn(clay, view.get("rgba"))
                row["row_consistency"] = row_report
                if str(row_report.get("verdict")) == "inconsistent":
                    row["conditioning_eligible"] = False
                    row["bake_eligible"] = False
                    row["conditioning_refusal"] = (
                        "row consistency vs the clay guide: verdict "
                        f"inconsistent (score {row_report.get('score')}, "
                        f"remap {row_report.get('score_at_best_remap')}) — "
                        "content the correction family cannot explain"
                    )
            except Exception as exc:  # loud in the record, fail closed
                row["conditioning_eligible"] = False
                row["bake_eligible"] = False
                row["conditioning_refusal"] = (
                    f"row consistency unverifiable: {type(exc).__name__}: {exc}"
                )
        # Pose honesty: TWO keys must agree before anything acts on the
        # measurement — the offset exceeds the gate AND the measured
        # azimuth beats the declared one decisively (see the constants'
        # calibration bands; plateau noise neither refuses nor re-declares).
        # The acceptance ladder's own measurement rides the view (same
        # ruler, same mesh, same pixels) — reuse it instead of paying the
        # ~20-clay-registration sweep twice per view.
        ladder_pose = view.get("pose")
        if isinstance(ladder_pose, Mapping) and ladder_pose.get("measured"):
            pose: Dict[str, Any] = dict(ladder_pose)
            pose["source"] = "acceptance_ladder"
        else:
            try:
                pose = measure_view_azimuth(
                    view.get("rgba"), conditioning_mesh, row["azimuth_deg"])
            except Exception as exc:
                pose = {"error": f"{type(exc).__name__}: {exc}", "measurable": False}
        row["pose"] = pose
        if pose.get("decisive"):
            row["measured_azimuth_deg"] = float(pose["measured_deg"])
            if float(pose.get("delta_deg") or 0.0) > POSE_MAX_DELTA_DEG and row["conditioning_eligible"]:
                row["conditioning_eligible"] = False
                row["conditioning_refusal"] = (
                    f"pose honesty: measured azimuth {pose['measured_deg']} deg is "
                    f"{pose['delta_deg']} deg off the declared {row['azimuth_deg']} deg "
                    f"(> {POSE_MAX_DELTA_DEG}, iou gap {pose.get('iou_gap')} >= "
                    f"{POSE_GATE_MIN_IOU_GAP}); the 2mv tags are trained positions — "
                    "the view is re-declared to its measured azimuth for the "
                    "texture bake instead"
                )
        else:
            # Plateau argmax (bust backs) or unmeasurable: noise — the
            # declared angle stands for both consumers, sweep on record.
            row["measured_azimuth_deg"] = row["azimuth_deg"]
        # The window law (conditioning variant only).
        if row["conditioning_eligible"]:
            try:
                windowed, window_report = window_view(view["rgba"])
                row["windowed_rgba"] = windowed
                window_reports[label] = window_report
            except Exception as exc:
                row["conditioning_eligible"] = False
                row["conditioning_refusal"] = (
                    f"window law failed on this view: {type(exc).__name__}: {exc}"
                )
        plan_views.append(row)

    front_windowed, front_window_report = window_view(source_rgba)
    window_reports["front"] = front_window_report
    spreads = [
        report["kept_ratio_shoulder"]
        for report in window_reports.values()
        if "kept_ratio_shoulder" in report
    ]
    return {
        "views": plan_views,
        "refgen_report": refgen_report,
        "front_windowed": front_windowed,
        "window": {
            "k": LOOP_WINDOW_K,
            "per_view": window_reports,
            "kept_ratio_spread": (
                round(max(spreads) - min(spreads), 4) if spreads else None
            ),
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


def self_verification(
    export_mesh: Any,
    *,
    textured: bool,
    clay_size: int = VERIFICATION_CLAY_SIZE,
    oblique_size: int = OBLIQUE_RENDER_SIZE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Recipe step 8: render the surfaces the operator checks and run the
    duplication flag on the calibrated instrument (front clay, 1024 px,
    elevation 10). Returns `(record, images)` — bundle-ready PIL images
    (oblique raking closeups + the front clay). Oblique verdicts are
    recorded but never gate (the threshold is calibrated on front renders
    only); size params exist for tests, production keeps the defaults."""
    import trimesh

    from .rendering import render_mesh_views

    started = time.perf_counter()
    clay_mesh = trimesh.Trimesh(
        vertices=export_mesh.vertices.copy(),
        faces=export_mesh.faces.copy(),
        process=False,
    )
    images: Dict[str, Any] = {}
    record: Dict[str, Any] = {"oblique": {}, "textured_rendered": bool(textured)}

    front_clay = render_mesh_views(
        clay_mesh,
        size=int(clay_size),
        azimuths=[0.0],
        elevation=VERIFICATION_CLAY_ELEVATION_DEG,
    )[0]
    images["verification_front_clay"] = front_clay
    try:
        record["duplication"] = duplication_flag(front_clay)
    except Exception as exc:
        record["duplication"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "duplication_suspect": False,
        }

    for name, azimuth in OBLIQUE_ANGLES:
        try:
            clay_view = render_mesh_views(
                clay_mesh, size=int(oblique_size), azimuths=[azimuth],
                elevation=OBLIQUE_ELEVATION_DEG,
            )[0]
            images[f"verification_{name}_clay"] = _face_crop(clay_view) or clay_view
            oblique_record: Dict[str, Any] = {}
            try:
                oblique_flag = duplication_flag(clay_view)
                oblique_record["duplication_recorded"] = {
                    key: oblique_flag.get(key)
                    for key in ("peak_ratio", "peak_lag_px", "duplication_suspect")
                }
                oblique_record["note"] = "recorded only; threshold calibrated on front renders"
            except Exception:
                pass
            if textured:
                tex_view = render_mesh_views(
                    export_mesh, size=int(oblique_size), azimuths=[azimuth],
                    elevation=OBLIQUE_ELEVATION_DEG,
                )[0]
                images[f"verification_{name}_tex"] = _face_crop(tex_view) or tex_view
            record["oblique"][name] = oblique_record
        except Exception as exc:
            record["oblique"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record, images


def _face_crop(image: Any, top_fraction: float = 0.62) -> Optional[Any]:
    """Upper-subject crop (oblique_closeups.py logic): top ~62% of the
    subject rows; wide crops downscale to 1200 px."""
    import numpy as np
    from PIL import Image

    mask = _subject_mask(image)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    height = rows[-1] - rows[0] + 1
    crop = image.crop(
        (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[0] + height * top_fraction))
    )
    if crop.width > 1200:
        crop = crop.resize((1200, int(1200 * crop.height / crop.width)), Image.LANCZOS)
    return crop


__all__ = [
    "DUPLICATION_PEAK_RATIO_THRESHOLD", "LOOP_WINDOW_K",
    "POSE_ACCEPT_MAX_DELTA_DEG", "POSE_GATE_MIN_IOU_GAP",
    "POSE_MAX_DELTA_DEG", "POSE_SWEEP_CONTRAST_MIN",
    "build_pose_acceptance_gate", "duplication_flag", "head_band_iou",
    "measure_view_azimuth", "self_verification", "synthesize_loop_views",
    "window_anchors", "window_view",
]
