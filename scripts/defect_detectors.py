#!/usr/bin/env python3
"""Defect detectors for bust models — geometry from depth maps, texture from
palette/structure statistics. Red-team rebuild after the 2026-07-21 escape.

Why depth maps: the failed evaluation measured SHADED clay pixels, which mix
geometry with lighting (its mesh metrics ranked a good mesh below a bad one).
Every geometry detector here reads an orthographic DEPTH map (protrusion in
units of the mesh's normalized radius) or the silhouette — pure geometry, no
light. Every face measurement is anchored on the NOSE TIP (global depth
maximum of the mid-sagittal band — verified unambiguous on all 9 calibration
models, glasses included) and scaled by the FACE WIDTH at the nose row (Wf),
so windows track the subject, not the framing.

Detectors (calibration numbers in docstrings; validated both ways on the
2026-07-21 operator-verdict models):

  open_mouth          front depth, mid-sagittal median profile: deepest
                      valley prominence in the lip window nose+[0.08,0.45]Wf.
                      Parted lips carve a real notch between two lip ridges;
                      a closed deep lip seam does not reach it.
  profile_shape       (a) face inflation: (z_nose - z_neck) / Wf — a bulbous
                      blown-up face profile inflates protrusion relative to
                      width; (b) side-silhouette ripple count below the nose
                      at both +/-90 (doubled/striated lips are extra convex
                      ridges the photo's single lip pair cannot produce).
  side_smear          textured side views: connected bright low-saturation
                      blob fraction over the face band. The subject's palette
                      (dark hair, mid skin, black shirt, dark glasses) has no
                      large bright-desaturated region; white/silver bake
                      garbage does.
  skin_on_crown       textured front hemisphere: skin-colored fraction of the
                      crown zone (rows above the eyewear band). The photo's
                      crown is hair; face patches printed on hair light up.
  ghost_glasses       textured front hemisphere: smooth very-dark mass in the
                      eyewear strip that is DISJOINT from the primary
                      (largest) glasses component — echo frames printed on
                      cheeks. Smoothness separates lens material from beard.

All detectors return raw value + threshold + fired flag so a panel can score
margins, not just booleans.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

GLTF_TO_CANONICAL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# ---------------------------------------------------------------- thresholds
# Calibrated on the 2026-07-21 operator-verdict set (see evaluation_redteam.md
# for the full tables). Each lists (worst passing, best failing) evidence.
OPEN_MOUTH_THRESHOLD = 0.0105          # e18 fails at 0.0125; closest pass e11 0.0086
DOUBLE_MOUTH_SEAM_FIRE = 2             # e2 (double mouth) = 2 seams; all 8 candidates <= 1
DOUBLE_MOUTH_SEAM_PROMINENCE = 0.003   # seam depth floor (mesh-radius units)
MOUTH_STRIATION_FIRE = 3               # e15/e17 (striated) = 3 fine ridges; e20/e18/e21 = 2
MOUTH_STRIATION_PROMINENCE = 0.0008    # fine-ridge floor after detrending
FACE_INFLATION_THRESHOLD = 0.650       # e17 fails at 0.687; closest pass e2 0.609
SIDE_RIPPLE_PROMINENCE = 0.0045        # silhouette ripple min prominence (units of Wf)
SIDE_RIPPLE_FIRE_COUNT = 2             # measured: e17/e18 = 2 ripples, all clean models = 0
SIDE_SMEAR_THRESHOLD = 0.020           # e10 left fails at 0.081; e20_rebake worst side 0.009
SKIN_ON_CROWN_THRESHOLD = 0.12         # e10/-30 fails at 0.21; worst clean e17 0.081
GHOST_EXTRA_BANDS_FIRE = 2             # e21 fails at 3 bands; e20_rebake 0 (az+30, elev 12)
BACKGROUND_DIFF_THRESHOLD = 14.0


# =============================================================== depth render
def load_canonical_mesh(glb_path: Path):
    """scene.glb -> bare trimesh in the canonical render frame (Z-up, +X front)."""
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
    )
    mesh.apply_transform(GLTF_TO_CANONICAL)
    return mesh


def render_depth(mesh, azimuth_deg: float, elevation_deg: float, size: int = 768):
    """Orthographic depth (protrusion toward the camera, mesh-radius units).

    Same camera math as abstract3d.rendering.render_mesh_views (bbox-centered,
    radius-normalized, orthographic fit at 1.18x), so depth-map coordinates
    correspond to the shaded renders. Returns (depth, mask); depth is NaN
    outside geometry, HIGHER = closer to the camera.
    """
    import moderngl

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    centered = (vertices - center)
    radius = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    centered = centered / radius

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    eye = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float32,
    ) * 3.2
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    upv = np.cross(side, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = side
    view[1, :3] = upv
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye

    cam_space = centered @ view[:3, :3].T + view[:3, 3]
    half_extent = float(np.max(np.abs(cam_space[:, :2]))) * 1.18
    near, far = 0.1, 16.0
    proj = np.eye(4, dtype=np.float32)
    proj[0, 0] = 1.0 / max(half_extent, 1e-3)
    proj[1, 1] = 1.0 / max(half_extent, 1e-3)
    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)
    mvp = proj @ view

    units_per_px = (2.0 * half_extent) / float(size)

    ctx = moderngl.create_context(standalone=True)
    try:
        prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                out float v_depth;
                uniform mat4 u_mvp;
                uniform mat4 u_view;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_depth = -(u_view * vec4(in_pos, 1.0)).z;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_depth;
                out vec4 f_out;
                void main() { f_out = vec4(v_depth, 1.0, 0.0, 1.0); }
            """,
        )
        tri = centered[faces].reshape(-1, 3).astype(np.float32)
        vbo = ctx.buffer(tri.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f", "in_pos")])
        fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((size, size), 4, dtype="f4")],
            depth_attachment=ctx.depth_renderbuffer((size, size)),
        )
        fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        prog["u_mvp"].write(mvp.astype(np.float32).T.tobytes())
        prog["u_view"].write(view.astype(np.float32).T.tobytes())
        vao.render()
        raw = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32)
        raw = raw.reshape(size, size, 4)[::-1]
        covered = raw[..., 1] > 0.5
        depth = np.where(covered, 3.2 - raw[..., 0], np.nan)
        for obj in (vao, vbo, prog, fbo):
            obj.release()
    finally:
        ctx.release()
    return depth, covered, units_per_px


# ============================================================ face anchoring
class FaceFrame:
    """Nose-anchored face coordinate frame from a FRONT depth map."""

    def __init__(self, depth: np.ndarray, mask: np.ndarray, units_per_px: float):
        self.units_per_px = float(units_per_px)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:
            raise ValueError("Empty depth map: no subject.")
        self.r0, self.r1 = int(rows[0]), int(rows[-1] + 1)
        self.c0, self.c1 = int(cols[0]), int(cols[-1] + 1)
        self.H = self.r1 - self.r0
        self.cc = (self.c0 + self.c1) // 2
        dn = np.where(mask, depth, np.nan)
        self.depth = dn
        self.mask = mask
        hw = max(3, int(0.02 * (self.c1 - self.c0)))
        with np.errstate(all="ignore"):
            pmid = np.nanmax(dn[:, self.cc - hw : self.cc + hw], axis=1)
        self.mid_profile = pmid
        # Nose tip: global max of the mid-sagittal band in rows 20-55% of the
        # subject. Verified the frontmost mid-face point is the nose on every
        # calibration model (the glasses sit BEHIND the nose tip on all).
        na, nb = self.r0 + int(0.20 * self.H), self.r0 + int(0.55 * self.H)
        seg = np.where(np.isfinite(pmid[na:nb]), pmid[na:nb], -np.inf)
        self.nose_row = na + int(np.argmax(seg))
        self.z_nose = float(pmid[self.nose_row])
        wrow = np.flatnonzero(mask[self.nose_row])
        self.face_width = int(wrow[-1] - wrow[0] + 1)  # Wf, px

    def median_profile(self, col_frac: float = 0.22) -> np.ndarray:
        cw = int(col_frac * self.face_width)
        with np.errstate(all="ignore"):
            return np.nanmedian(self.depth[:, self.cc - cw : self.cc + cw], axis=1)


def _interp_finite(z: np.ndarray) -> np.ndarray:
    v = np.flatnonzero(np.isfinite(z))
    if v.size == 0:
        return np.zeros_like(z)
    return np.interp(np.arange(len(z)), v, z[v])


# ========================================================== geometry detectors
def detect_open_mouth(frame: FaceFrame) -> Dict:
    """Deepest valley prominence in the lip window of the mid-sagittal median
    profile. Parted lips (open mouth) = two lip ridges with a real notch
    between them; a closed mouth's lip seam stays below the threshold.

    Calibration (front depth 768px):
      e18_windowed (operator: OPEN)   0.0125
      e11_2mv_reg_hq (closed)         0.0086   <- worst passing
      e20_fixed_views (closed)        0.0068
      e10/e2/e21/e17/e15              0.0025-0.0050
    """
    prof = frame.median_profile()
    wf = frame.face_width
    z0 = frame.nose_row
    z = _interp_finite(prof[z0 : min(frame.r1 - 1, z0 + int(0.60 * wf))].copy())
    zs = ndimage.gaussian_filter1d(z, max(1.5, 0.012 * wf))
    band = slice(int(0.08 * wf), max(int(0.08 * wf) + 2, min(len(zs) - 1, int(0.45 * wf))))
    zb = zs[band]
    valleys, props = find_peaks(-zb, prominence=0.0012)
    deepest = float(props["prominences"].max()) if len(valleys) else 0.0
    return {
        "metric": "mouth_gap_prominence",
        "value": round(deepest, 4),
        "threshold": OPEN_MOUTH_THRESHOLD,
        "fired": bool(deepest >= OPEN_MOUTH_THRESHOLD),
    }


def detect_double_mouth(frame: FaceFrame) -> Dict:
    """Duplicated mouth seams: count of DISTINCT valleys (prominence >=
    0.003 radius units) in the seam window nose+[0.14,0.36]Wf of the
    mid-sagittal median profile. A single mouth carves ONE seam (the lip line
    or the mentolabial fold dominates); a double mouth carves one per mouth.

    Replaces the old shading-autocorrelation duplication detector, which
    re-measured on today's renders ranks e10 (single mouth, operator-proven)
    ABOVE e2 (double mouth): 0.093 vs 0.089 front — non-discriminative.

    Calibration (front depth 768px, seam window nose+[0.14,0.36]Wf):
      e2_2mv_explicit_refs (operator: double mouth)  2 seams (0.0031, 0.0050)
      all 8 current candidates                       0-1 seams
    """
    prof = frame.median_profile()
    wf = frame.face_width
    z0 = frame.nose_row
    z = _interp_finite(prof[z0 : min(frame.r1 - 1, z0 + int(0.60 * wf))].copy())
    zs = ndimage.gaussian_filter1d(z, max(1.5, 0.012 * wf))
    valleys, props = find_peaks(-zs, prominence=DOUBLE_MOUTH_SEAM_PROMINENCE)
    seams = [
        (round(q / wf, 3), round(float(props["prominences"][k]), 4))
        for k, q in enumerate(valleys)
        if 0.14 <= q / wf <= 0.36
    ]
    return {
        "metric": "mouth_seam_count",
        "value": len(seams),
        "seams": seams,
        "threshold": DOUBLE_MOUTH_SEAM_FIRE,
        "fired": bool(len(seams) >= DOUBLE_MOUTH_SEAM_FIRE),
    }


def detect_mouth_striation(glb_path: Path) -> Dict:
    """Striated / multi-ridge mouth: fine parallel ridges in the lip zone.

    The raking-light striations the operator saw on e15/e17 are shallow
    (~0.1-0.4% of the mesh radius) parallel ridges. At 1536px front depth,
    detrend the mid-sagittal median profile (subtract a 0.06 Wf gaussian),
    smooth the residual at 0.004 Wf, and count ridges with prominence >=
    0.0008 in nose+[0.03,0.45]Wf.

    Calibration: striated per operator — e15 3 (0.0041/0.0044/0.0027),
    e17 3, e10 3, e11 3, e2 3; clean — e18 2, e20 2, e20_rebake 2, e21 2.
    Fires at >= 3. (e10/e11 firing matches the operator's 'mesh wrong'
    verdict for the e10 class.)
    """
    mesh = load_canonical_mesh(glb_path)
    depth, mask, upp = render_depth(mesh, 0.0, 0.0, size=1536)
    frame = FaceFrame(depth, mask, upp)
    prof = frame.median_profile()
    wf = frame.face_width
    z0 = frame.nose_row
    z = _interp_finite(prof[z0 : min(frame.r1 - 1, z0 + int(0.55 * wf))].copy())
    heavy = ndimage.gaussian_filter1d(z, 0.06 * wf)
    fine = ndimage.gaussian_filter1d(z - heavy, max(1.0, 0.004 * wf))
    band = fine[int(0.03 * wf) : int(0.45 * wf)]
    ridges, props = find_peaks(band, prominence=MOUTH_STRIATION_PROMINENCE)
    return {
        "metric": "fine_ridge_count",
        "value": int(len(ridges)),
        "prominences": [round(float(p), 4) for p in props["prominences"]],
        "threshold": MOUTH_STRIATION_FIRE,
        "fired": bool(len(ridges) >= MOUTH_STRIATION_FIRE),
    }


def detect_face_inflation(frame: FaceFrame) -> Dict:
    """Bulbous/blown-up face profile: protrusion of the nose above the neck
    trough, normalized by face width (both in mesh-radius units via the
    render's own orthographic scale).

    Calibration (front depth 768px): e17 0.687 (operator: bulbous profile)
    | e2 0.609 | e18 0.597 | e10 0.595 | e11 0.588 | e15 0.581 | e21 0.555
    | e20 0.551. Threshold 0.650 sits between e17 and the rest.
    """
    prof = frame.median_profile()
    wf = frame.face_width
    rn = frame.nose_row
    npk = np.nanmax(prof[max(frame.r0, rn - int(0.10 * wf)) : rn + int(0.10 * wf)])
    za, zb = rn + int(0.60 * wf), min(frame.r1 - 1, rn + int(1.40 * wf))
    if zb <= za + 2:
        return {"metric": "face_inflation", "value": None, "threshold": FACE_INFLATION_THRESHOLD, "fired": False,
                "note": "no neck rows in frame"}
    zneck = np.nanmin(prof[za:zb])
    # Depth values are mesh-radius units; convert the face width from px to
    # the same units with the render's own orthographic scale so the ratio
    # is unit-consistent and framing-independent.
    wf_units = frame.face_width * frame.units_per_px
    ratio = float((npk - zneck) / max(wf_units, 1e-9))
    return {
        "metric": "face_inflation",
        "value": round(ratio, 3),
        "threshold": FACE_INFLATION_THRESHOLD,
        "fired": bool(ratio >= FACE_INFLATION_THRESHOLD),
    }


def _side_face_contour(mask: np.ndarray, r0: int, r1: int) -> Tuple[np.ndarray, int]:
    """Face-front silhouette x(row) from a +/-90 depth mask; +x toward the
    face. Face side = the silhouette side whose upper-half extreme sticks out
    most relative to its own median (the nose)."""
    h, w = mask.shape
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    for r in range(r0, r1):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            left[r] = w - 1 - cols[0]
            right[r] = cols[-1]
    up = slice(r0 + int(0.25 * (r1 - r0)), r0 + int(0.60 * (r1 - r0)))
    def excess(c):
        seg = c[up]
        seg = seg[np.isfinite(seg)]
        return (np.max(seg) - np.median(seg)) if seg.size else -1.0
    return (right, +1) if excess(right) >= excess(left) else (left, -1)


def detect_profile_ripples(mesh, size: int = 768) -> Dict:
    """Side-silhouette convexity ripples below the nose at both +/-90.

    On this subject (closed mouth, beard), a healthy profile renders the
    lips/chin as one smooth convex mass — ZERO ripples above the prominence
    floor. Doubled lips (e17) and parted lips (e18) each add distinct convex
    ridges. Count = max over the two sides.

    Calibration (768px silhouettes, prominence floor 0.0045 Wf): e17 = 2
    (0.002/0.002 in H units), e18 = 2 (0.0038/0.0045); every clean model
    (e2/e10/e11/e15/e20/e20_rebake/e21) = 0. Fires at >= 2.
    """
    counts = {}
    details = {}
    for az in (90.0, -90.0):
        depth, mask, _upp = render_depth(mesh, az, 0.0, size=size)
        rows = np.flatnonzero(mask.any(axis=1))
        r0, r1 = int(rows[0]), int(rows[-1] + 1)
        H = r1 - r0
        contour, _sign = _side_face_contour(mask, r0, r1)
        c = ndimage.median_filter(_interp_finite(contour), 5)
        # nose on the side view: max contour x in rows 25-55%H
        na, nb = r0 + int(0.25 * H), r0 + int(0.55 * H)
        rn = na + int(np.argmax(c[na:nb]))
        # face width proxy on the side view: head depth at nose row is not
        # available from the silhouette; use the front-calibrated 0.30*H
        # (Wf/H is 0.29-0.33 on every calibration model).
        wf = 0.30 * H
        zone = c[rn + int(0.10 * wf) : rn + int(0.75 * wf)] / H
        zs = ndimage.gaussian_filter1d(zone, max(1.5, 0.010 * wf))
        # SIDE_RIPPLE_PROMINENCE is expressed in units of Wf; the contour is
        # normalized by H, so convert via wf = 0.30*H -> p_H = p_Wf * 0.30.
        pk, props = find_peaks(zs, prominence=SIDE_RIPPLE_PROMINENCE * 0.30)
        counts[az] = int(len(pk))
        details[f"az{az:+.0f}"] = [round(float(p), 4) for p in props["prominences"]]
    worst = max(counts.values())
    return {
        "metric": "profile_ripple_count",
        "value": worst,
        "per_side": details,
        "threshold": SIDE_RIPPLE_FIRE_COUNT,
        "fired": bool(worst >= SIDE_RIPPLE_FIRE_COUNT),
    }


# =========================================================== texture detectors
def subject_mask_rgb(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    return np.abs(rgb - background).max(axis=2) > BACKGROUND_DIFF_THRESHOLD


def _face_zone(mask: np.ndarray, top_frac: float = 0.66) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    r0, r1 = rows[0], rows[-1] + 1
    zone = np.zeros_like(mask)
    zone[r0 : r0 + int(top_frac * (r1 - r0))] = True
    return zone & mask


def detect_side_smear(img: Image.Image) -> Dict:
    """Bright low-saturation blob fraction over the face band of a textured
    view. The subject's palette (dark hair / mid skin / black shirt / dark
    glasses) produces almost none; white-silver bake garbage produces a lot.

    Calibration (photo front baseline 0.013): e10 left -60/-90 = 0.081/0.085
    (operator: chaotic smear); e20_rebake worst side 0.009; threshold 0.030.
    """
    small = img.resize((512, 512), Image.LANCZOS)
    mask = subject_mask_rgb(small)
    fz = _face_zone(mask)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float64)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    cand = (mx > 150) & (sat < 0.25) & fz
    labels, n = ndimage.label(cand)
    keep = np.zeros_like(cand)
    min_area = 0.002 * fz.sum()   # blobs, not speckle: >=0.2% of the face zone
    for k in range(1, n + 1):
        comp = labels == k
        if comp.sum() >= min_area:
            keep |= comp
    frac = float(keep.sum() / max(fz.sum(), 1))
    return {
        "metric": "bright_blob_fraction",
        "value": round(frac, 4),
        "threshold": SIDE_SMEAR_THRESHOLD,
        "fired": bool(frac >= SIDE_SMEAR_THRESHOLD),
    }


def detect_skin_on_crown(img: Image.Image, nose_row_frac: float) -> Dict:
    """Skin-colored blob fraction of the crown (rows well above the eyewear
    line). This subject's crown is hair in the photo, so contiguous skin
    printed on hair lights up — the phantom-second-face class (e10/e11/e21
    paint a whole extra face onto the side hair, visible at az -30).
    Blob filtering (components >= 0.5% of the crown zone) rejects warm
    hair-highlight speckle.

    The crown's lower bound is anchored GEOMETRICALLY: the nose-tip row from
    the mesh's front depth map (nose_row_frac of subject height) minus 0.12H.
    A texture-derived anchor would move with the defect being measured.

    Calibration (max over az 0/+30/-30): e10 0.213 | e11 0.177 | e21 0.163
    | e15 0.125 || e17 0.081 | e18 0.075 | e20_rebake 0.074 | e20 0.071.
    Threshold 0.12. KNOWN MISS (documented): e18's smaller skin patches sit
    BELOW the crown line (between hairline and glasses top) among legitimate
    temple skin — not separable by this zone; e18 is gated by its mesh
    defects and the ghost detector instead.
    """
    small = img.resize((640, 640), Image.LANCZOS)
    mask = subject_mask_rgb(small)
    rgb = np.asarray(small.convert("RGB"), dtype=np.float64)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    rows = np.flatnonzero(mask.any(axis=1))
    r0, r1 = int(rows[0]), int(rows[-1] + 1)
    H = r1 - r0
    nose_row = r0 + int(nose_row_frac * H)
    crown_lo = r0 + int(0.03 * H)
    crown_hi = max(crown_lo + 2, nose_row - int(0.12 * H))
    zone = np.zeros_like(mask)
    zone[crown_lo:crown_hi] = True
    zone &= mask
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    skin = (
        (lum > 95)
        & (r > g) & (g > b)                     # warm hue ordering
        & ((r - b) > 18)
        & (sat > 0.12) & (sat < 0.62)
        & zone
    )
    labels, n = ndimage.label(skin, structure=np.ones((3, 3)))
    keep = np.zeros_like(skin)
    min_area = 0.005 * max(zone.sum(), 1)
    for k in range(1, n + 1):
        comp = labels == k
        if comp.sum() >= min_area:
            keep |= comp
    frac = float(keep.sum() / max(zone.sum(), 1)) if zone.any() else 0.0
    return {
        "metric": "skin_on_crown_fraction",
        "value": round(frac, 4),
        "threshold": SKIN_ON_CROWN_THRESHOLD,
        "fired": bool(frac >= SKIN_ON_CROWN_THRESHOLD),
    }


# --- ghost glasses: PORT of the validated model_quality_sweep.tex_oblique_ghost
# at its validated operating point (textured render, azimuth +30 RIGHT
# oblique, elevation 12). The metric counts distinct wide dark row-bands in
# the face band beyond the widest one (one pair of glasses = 1 band) plus a
# saturation-speckle term. Re-validated 2026-07-21 at the original operating
# point: e21 = 3.00 (operator: ghost glasses on cheeks -> FAIL), e17 = 6.00,
# e20_fixed = 2.00 (operator: texture wrong -> FAIL), e18 = 2.00,
# e20_rebake = 0.00 (PASS), e10 = 0.00, e11 = 1.00, e15 = 0.00.
# HONEST LIMIT (measured): at azimuth -30 this metric returns 3.0-13.0 for
# EVERY model including the cleanest — the left beard/jaw shadow reads as
# extra dark bands on this subject — so the LEFT oblique value is reported
# as advisory and excluded from gating; left-side texture garbage is gated
# by side_smear and skin_on_crown instead.
def _sweep_subject_mask(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] == 4 and arr[..., 3].min() < 250:
        return arr[..., 3] > 10
    lum = arr[..., :3].mean(axis=2)
    return np.abs(lum - lum[0, 0]) > 8


def _sweep_face_band(arr: np.ndarray, mask: np.ndarray, top_frac: float = 0.62):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    y_cut = y0 + int((y1 - y0 + 1) * top_frac)
    return arr[y0:y_cut, x0 : x1 + 1], mask[y0:y_cut, x0 : x1 + 1]


def ghost_band_score(tex_obl: Image.Image) -> float:
    """Verbatim port of model_quality_sweep.tex_oblique_ghost (2026-07-21)."""
    arr = np.asarray(tex_obl.convert("RGBA"), dtype=np.float64)
    band, bm = _sweep_face_band(arr, _sweep_subject_mask(arr))
    lum = band[..., :3] @ np.array([0.299, 0.587, 0.114])
    dark = (lum < 60) & bm
    width = bm.sum(axis=1)
    frac = np.divide(dark.sum(axis=1), np.maximum(width, 1))
    rows = (frac > 0.30) & (width > 0)
    runs, in_run = 0, False
    for r in rows:
        if r and not in_run:
            runs += 1
            in_run = True
        elif not r and in_run:
            in_run = False
    extra_bands = max(0, runs - 1)
    mx = band[..., :3].max(axis=2)
    mn = band[..., :3].min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    speckle = ((sat > 0.55) & (mx > 120) & bm).sum() / max(bm.sum(), 1)
    return round(extra_bands + float(speckle) * 20.0, 3)


def detect_ghost_glasses(glb_path: Path) -> Dict:
    """Ghost-glasses gate: ghost_band_score on a fresh textured render at the
    VALIDATED operating point (az +30, elevation 12, 1536 px). The left
    oblique (az -30) is measured and reported as advisory (see port note)."""
    import trimesh

    from abstract3d.rendering import render_mesh_views

    scene = trimesh.load(str(glb_path), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    right, left = render_mesh_views(mesh, size=1536, azimuths=(30.0, -30.0), elevation=12.0)
    value_right = ghost_band_score(right)
    value_left = ghost_band_score(left)
    return {
        "metric": "ghost_band_score",
        "value": value_right,
        "advisory_left": value_left,
        "threshold": GHOST_EXTRA_BANDS_FIRE,
        "fired": bool(value_right >= GHOST_EXTRA_BANDS_FIRE),
    }


# ================================================================ orchestration
def geometry_report(glb_path: Path, size: int = 768) -> Dict:
    mesh = load_canonical_mesh(glb_path)
    depth, mask, units_per_px = render_depth(mesh, 0.0, 0.0, size=size)
    frame = FaceFrame(depth, mask, units_per_px)
    return {
        "open_mouth": detect_open_mouth(frame),
        "double_mouth": detect_double_mouth(frame),
        "mouth_striation": detect_mouth_striation(glb_path),
        "face_inflation": detect_face_inflation(frame),
        "profile_ripples": detect_profile_ripples(mesh, size=size),
        "nose_row_frac": round((frame.nose_row - frame.r0) / frame.H, 4),
    }


def texture_report(turntable_dir: Path, nose_row_frac: float, glb_path: Path) -> Dict:
    """Texture detectors over the turntable renders (full_turntable.py output).

    side_smear runs at every azimuth (both sides always — the e10 escape);
    skin-on-crown runs on the near-front views; ghost_glasses renders its own
    validated operating point (az +30, elev 12) from the GLB.

    nose_row_frac: nose-tip row as a fraction of subject height, measured on
    the front DEPTH map (FaceFrame) — the geometric anchor the crown detector
    needs so a displaced texture cannot move its own search zone.
    """
    out: Dict[str, Dict] = {"side_smear": {}, "skin_on_crown": {}}
    for az in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
        img = Image.open(turntable_dir / f"tex_e00_az{az:+04.0f}.png")
        out["side_smear"][f"az{az:+.0f}"] = detect_side_smear(img)
        if az in (0.0, 30.0, -30.0):
            out["skin_on_crown"][f"az{az:+.0f}"] = detect_skin_on_crown(img, nose_row_frac)
    out["ghost_glasses"] = detect_ghost_glasses(glb_path)
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--turntable-dir", type=Path, default=None,
                        help="Model dir under review/turntable for texture checks.")
    args = parser.parse_args()
    geometry = geometry_report(args.glb)
    report = {"geometry": geometry}
    if args.turntable_dir:
        report["texture"] = texture_report(args.turntable_dir, geometry["nose_row_frac"], args.glb)
    print(json.dumps(report, indent=1))
