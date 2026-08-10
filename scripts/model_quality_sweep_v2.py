#!/usr/bin/env python3
"""Four-metric model evaluation v2: mesh and texture, each at two+ angles.

Operator-ruled criteria (2026-07-21): mesh quality and texture quality are
INDEPENDENT axes measured at >=2 angles each. v1's mesh metrics failed:
clay-vs-photo edge-profile NCC was swamped by lighting mismatch, and the
striation autocorrelation ran on headlight clay renders whose flat shading
hides the ridges the operator sees under Blender-quality (raking) light.

v2 evaluates MESH quality on NORMAL-MAP renders — surface normals encoded
as RGB are a lighting-independent geometry visualization (the standard
geometry-inspection surface in 3D-generation evaluation, e.g. Eval3D and
GPT-4V-as-evaluator both feed normal renders precisely because RGB shading
hides geometry defects). Regions are anchored on YuNet face landmarks
detected on the front normal render and lifted to 3D through the position
buffer, so every view measures the SAME anatomical band.

Headline panel (raw, directional, no opaque composite):
  mesh_front_defect    lower=better. Coherent-ridge energy of the mouth
                       region on the FRONT normal render (striation /
                       multi-ridge / lumpy-carve detector) + a duplication
                       surcharge when a second crease in the mouth band
                       rivals the first (double-mouth class, e2).
  mesh_oblique_defect  lower=better. Same ridge energy averaged over four
                       oblique views (az +-30, +-55; the raking-light
                       analog where striated mouths shipped twice).
  tex_front_dE         lower=better. CIE76 delta-E of the textured front
                       render vs the photo over the shared face band
                       (unchanged from v1 — reproduces operator verdicts).
  tex_oblique_ghost    lower=better. Ghost-structure score on the
                       30-degree textured render (unchanged from v1).

Optional extras (loud notes when unavailable; never rank drivers):
  sface_identity       cosine similarity photo vs textured front render
                       (OpenCV SFace, 37 MB ONNX). Catches semantic
                       texture misplacement (lips painted on nose class);
                       measured blind to ghost-glasses class — see
                       docs/research/evaluation_methods_2026.md.
  lpips_face           LPIPS (squeeze) on landmark-anchored face crops,
                       photo vs textured front.

Diagnostics (reported, non-ranking): landmark_row_delta (silhouette-
normalized nose/mouth row offsets vs photo — measured NON-discriminating
on the calibration set because upstream row gates already pin feature
heights; kept as a red flag for gross misplacement), detection
confidences, per-view ridge energies.

Calibration provenance: all constants below were calibrated on the
laurent-bust-redo 9-bundle set (2026-07-21) against the operator's visual
verdicts and must be re-checked on the next subject; they encode ridge
geometry (mouth-region extent in nose-to-mouth spans), not bundle
identities.

Face models (one-time local downloads, official opencv_zoo):
  ~/.cache/abstract3d_eval/face_detection_yunet_2023mar.onnx   (232 KB)
  ~/.cache/abstract3d_eval/face_recognition_sface_2021dec.onnx (37 MB)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
from scipy import ndimage, signal

# ---------------------------------------------------------------- constants
MODEL_DIR = Path.home() / ".cache" / "abstract3d_eval"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET_URL = f"{ZOO}/face_detection_yunet/{YUNET_FILE}"
SFACE_URL = f"{ZOO}/face_recognition_sface/{SFACE_FILE}"

# Mouth region in landmark units: z in nose-to-mouth spans around the mouth
# line, y in mouth-width margins around the corners. Calibrated so the band
# covers upper lip to chin without swallowing the nose (whose legitimate
# ridge would dominate the energy).
MOUTH_Z_BELOW = 1.35   # spans below the mouth line (chin side)
MOUTH_Z_ABOVE = 0.72   # spans above (stops under the nose base)
MOUTH_Y_MARGIN = 0.35  # fraction of mouth width beyond each corner
# Duplication band: narrow z window for crease counting.
DUP_Z_HALF = 0.75      # spans on each side of the mouth line
DUP_Y_MARGIN = 0.85    # fraction of mouth width
# Ridge extraction: patch is resampled to fixed rows so energy is
# scale-invariant; the x-smoothing window keeps horizontally COHERENT
# ridges (striations) and cancels isotropic texture (stubble, noise).
PATCH_ROWS = 128
RIDGE_SIGMA = 1.0
RIDGE_XWIN = 17
MASK_EROSION = 4       # px off the silhouette (normals turn fast there)
CORE_EROSION = 2       # px off the resampled patch border
# Duplication surcharge: a second mouth-band crease with prominence beyond
# DUP_FREE_PROMINENCE of a *deep* first crease signals a duplicated mouth.
DUP_FREE_PROMINENCE = 0.35
DUP_GAIN = 60.0

OBLIQUE_AZIMUTHS = (30.0, -30.0, 55.0, -55.0)
ELEVATION = 12.0

# Exported scene.glb sits in the glTF viewer frame (Y-up / front +Z); the
# canonical frame here is Z-up / front +X (matches abstract3d.rendering).
GLTF_TO_CANON = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def note(msg: str) -> None:
    print(f"  NOTE: {msg}", file=sys.stderr)


# ---------------------------------------------------------------- models
def ensure_model(filename: str, url: str, allow_download: bool) -> Path | None:
    path = MODEL_DIR / filename
    if path.exists():
        return path
    if not allow_download:
        note(f"{filename} missing and --no-download set ({url})")
        return None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    note(f"downloading {filename} from opencv_zoo (one-time, local cache)")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as exc:
        note(f"download failed: {exc}")
        return None


# ---------------------------------------------------------------- geometry render
class GeoRenderer:
    """Offscreen renderer producing view-space normals, world positions and
    a coverage mask per view (orthographic, matches abstract3d.rendering's
    camera framing so face crops stay comparable with clay renders).

    Backend: ModernGL when a GL context is available; otherwise a pure-numpy
    painter's-splat rasterizer (approximate but adequate for band statistics
    — Hunyuan meshes are dense enough that vertex splats tile the surface).
    `self.backend` records which one ran.
    """

    def __init__(self, mesh, size: int = 1024):
        self.size = size
        m = mesh.copy()
        m.apply_transform(GLTF_TO_CANON)
        verts = np.asarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.faces, dtype=np.int32)
        center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
        radius = float(np.max(np.linalg.norm(verts - center, axis=1))) or 1.0
        self.center, self.radius = center, radius
        self.verts_c = (verts - center) / radius
        self.normals = np.asarray(m.vertex_normals, dtype=np.float32)
        self.backend = "moderngl"
        try:
            self._init_gl(faces)
        except Exception as exc:
            note(f"moderngl unavailable ({type(exc).__name__}: {exc}); numpy splat fallback (approximate)")
            self.backend = "splat"
            self.ctx = None

    def _init_gl(self, faces: np.ndarray) -> None:
        import moderngl
        import time

        self._gl = moderngl
        packed = np.concatenate(
            [self.verts_c[faces].reshape(-1, 3), self.normals[faces].reshape(-1, 3)], axis=1
        ).astype(np.float32)
        # macOS CGL context creation can fail transiently under GL churn
        # (same mitigation as abstract3d.rendering._create_standalone_context).
        last = None
        for attempt in range(3):
            try:
                ctx = moderngl.create_context(standalone=True)
                break
            except Exception as exc:
                last = exc
                time.sleep(0.2 * (attempt + 1))
        else:
            raise last
        self.ctx = ctx
        size = self.size
        self.prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos; in vec3 in_nrm;
                out vec3 v_nrm; out vec3 v_wpos;
                uniform mat4 u_mvp;
                void main(){ gl_Position=u_mvp*vec4(in_pos,1.0); v_nrm=in_nrm; v_wpos=in_pos; }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_nrm; in vec3 v_wpos;
                layout(location=0) out vec4 f_nrm;
                layout(location=1) out vec4 f_pos;
                uniform mat3 u_view3;
                void main(){
                    vec3 n = normalize(u_view3 * normalize(v_nrm));
                    if (n.z < 0.0) n = -n;  // double-sided orientation
                    f_nrm = vec4(n*0.5+0.5, 1.0);
                    f_pos = vec4(v_wpos, 1.0);
                }
            """,
        )
        self.vbo = ctx.buffer(packed.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 3f", "in_pos", "in_nrm")])
        self.tex_nrm = ctx.texture((size, size), 4, dtype="f4")
        self.tex_pos = ctx.texture((size, size), 4, dtype="f4")
        self.depth = ctx.depth_renderbuffer((size, size))
        self.fbo = ctx.framebuffer([self.tex_nrm, self.tex_pos], self.depth)

    def _render_splat(self, view: np.ndarray, half: float) -> dict:
        """Vertex z-buffer splat at vertex-density-matched resolution, then
        upsampled: with ~1 vertex per pixel the splats tile the surface and
        pinhole fill cannot bleed back-surface normals through. Approximate
        silhouettes; band statistics (the metric surface) survive."""
        s = int(np.clip(math.sqrt(len(self.verts_c) * 2.2), 192, self.size))
        cam = self.verts_c @ view[:3, :3].T + view[:3, 3]
        px = ((cam[:, 0] / half) * 0.5 + 0.5) * (s - 1)
        py = (0.5 - (cam[:, 1] / half) * 0.5) * (s - 1)
        depth = -cam[:, 2]
        order = np.argsort(depth)[::-1]  # far first; near overwrites
        xi = np.clip(np.rint(px[order]).astype(np.int64), 0, s - 1)
        yi = np.clip(np.rint(py[order]).astype(np.int64), 0, s - 1)
        vn = (self.normals @ view[:3, :3].T)[order]
        flip = vn[:, 2] < 0
        vn[flip] = -vn[flip]
        world = self.verts_c[order] * self.radius + self.center
        nrm_img = np.zeros((s, s, 3), dtype=np.float32)
        pos_img = np.zeros((s, s, 3), dtype=np.float32)
        mask = np.zeros((s, s), dtype=bool)
        nrm_img[yi, xi] = vn
        pos_img[yi, xi] = world
        mask[yi, xi] = True
        holes = ndimage.binary_dilation(mask, iterations=2) & ~mask
        if holes.any():
            idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            nrm_img[holes] = nrm_img[idx[0][holes], idx[1][holes]]
            pos_img[holes] = pos_img[idx[0][holes], idx[1][holes]]
            mask |= holes
        # drop isolated fill-islands outside the true surface
        mask = ndimage.binary_opening(mask, iterations=1)
        zoom = self.size / s
        nrm_up = ndimage.zoom(nrm_img, (zoom, zoom, 1), order=1)[: self.size, : self.size]
        pos_up = ndimage.zoom(pos_img, (zoom, zoom, 1), order=1)[: self.size, : self.size]
        mask_up = ndimage.zoom(mask.astype(np.float32), zoom, order=1)[: self.size, : self.size] > 0.5
        lens = np.linalg.norm(nrm_up, axis=2, keepdims=True)
        nrm_up = np.where(mask_up[..., None], nrm_up / np.maximum(lens, 1e-6), 0.0)
        return {"normals": nrm_up.astype(np.float32), "mask": mask_up, "world": pos_up.astype(np.float32)}

    @staticmethod
    def _look_at(eye, target, up):
        fwd = target - eye
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-8)
        side = np.cross(fwd, up / max(float(np.linalg.norm(up)), 1e-8))
        side = side / max(float(np.linalg.norm(side)), 1e-8)
        up2 = np.cross(side, fwd)
        view = np.eye(4, dtype=np.float32)
        view[0, :3], view[1, :3], view[2, :3] = side, up2, -fwd
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    def render(self, azimuth: float, elevation: float) -> dict:
        az, el = math.radians(azimuth), math.radians(elevation)
        eye = 3.2 * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float32,
        )
        view = self._look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
        cam = self.verts_c @ view[:3, :3].T + view[:3, 3]
        half = max(float(np.max(np.abs(cam[:, :2]))) * 1.18, 1e-3)
        if self.backend == "splat":
            return self._render_splat(view, half)
        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = proj[1, 1] = 1.0 / half
        proj[2, 2], proj[2, 3] = -2.0 / 15.9, -16.1 / 15.9  # near .1 / far 16
        self.prog["u_mvp"].write((proj @ view).astype(np.float32).T.tobytes())
        self.prog["u_view3"].write(view[:3, :3].astype(np.float32).T.tobytes())
        self.fbo.use()
        self.ctx.enable(self._gl.DEPTH_TEST)
        self.ctx.disable(self._gl.CULL_FACE)
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.vao.render()
        s = self.size
        nrm = np.frombuffer(self.tex_nrm.read(), dtype=np.float32).reshape(s, s, 4)[::-1]
        pos = np.frombuffer(self.tex_pos.read(), dtype=np.float32).reshape(s, s, 4)[::-1]
        return {
            "normals": nrm[..., :3] * 2.0 - 1.0,
            "mask": nrm[..., 3] > 0.5,
            "world": pos[..., :3] * self.radius + self.center,
        }

    def release(self) -> None:
        if self.ctx is None:
            return
        for r in (self.vao, self.vbo, self.prog, self.tex_nrm, self.tex_pos, self.depth, self.fbo, self.ctx):
            try:
                r.release()
            except Exception:
                pass


def normal_image(buf: dict) -> Image.Image:
    rgb = ((buf["normals"] * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb[~buf["mask"]] = 30
    return Image.fromarray(rgb)


# ---------------------------------------------------------------- detection
def pil_to_bgr(img: Image.Image, bg=(200, 200, 200)) -> np.ndarray:
    import cv2

    if img.mode == "RGBA":
        base = Image.new("RGBA", img.size, bg + (255,))
        img = Image.alpha_composite(base, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def detect_face(img_bgr: np.ndarray, yunet_path: Path, thresholds=(0.6, 0.4, 0.25, 0.12)):
    """Best YuNet face; descending threshold ladder because clay/normal
    renders are out-of-distribution and legitimately score lower."""
    import cv2

    h, w = img_bgr.shape[:2]
    for t in thresholds:
        det = cv2.FaceDetectorYN.create(str(yunet_path), "", (w, h), float(t), 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(img_bgr)
        if faces is not None and len(faces):
            f = max(faces, key=lambda r: r[14])
            return {
                "box": [float(v) for v in f[:4]],
                "landmarks": f[4:14].reshape(5, 2).astype(float).tolist(),
                "conf": float(f[14]),
                "threshold": t,
            }
    return None


def world_at(buf: dict, px: float, py: float):
    """Median world position in a small neighborhood of a pixel."""
    h, w = buf["mask"].shape
    x0, x1 = max(0, int(px) - 3), min(w, int(px) + 4)
    y0, y1 = max(0, int(py) - 3), min(h, int(py) + 4)
    m = buf["mask"][y0:y1, x0:x1]
    if m.sum() < 3:
        return None
    return np.median(buf["world"][y0:y1, x0:x1][m], axis=0)


# ---------------------------------------------------------------- mesh metrics
def mouth_anchors(buf_front: dict, det: dict):
    """Lift front-view mouth landmarks to canonical 3D."""
    lm = np.array(det["landmarks"], dtype=np.float64)
    nose = world_at(buf_front, *lm[2])
    ml = world_at(buf_front, *lm[3])
    mr = world_at(buf_front, *lm[4])
    if nose is None or ml is None or mr is None:
        return None
    mouth_z = 0.5 * (ml[2] + mr[2])
    return {
        "mouth_z": float(mouth_z),
        "span": float(max(nose[2] - mouth_z, 1e-6)),  # nose-to-mouth vertical span
        "y_c": float(0.5 * (ml[1] + mr[1])),
        "mouth_w": float(max(abs(mr[1] - ml[1]), 1e-6)),
    }


def fallback_anchors(buf_front: dict):
    """No face detected: approximate the mouth band from silhouette z
    quantiles (head occupies the top of a bust; mouth sits ~2/3 down the
    head). Loudly less precise — used only so the sweep still reports."""
    zs = buf_front["world"][..., 2][buf_front["mask"]]
    z_top, z_bot = float(np.quantile(zs, 0.995)), float(np.quantile(zs, 0.30))
    head_h = z_top - z_bot
    mouth_z = z_top - 0.62 * head_h
    ys = buf_front["world"][..., 1][buf_front["mask"]]
    y_c = float(np.median(ys))
    return {"mouth_z": mouth_z, "span": 0.10 * head_h, "y_c": y_c, "mouth_w": 0.22 * head_h}


def region_mask(buf: dict, anchors: dict, z_lo_k: float, z_hi_k: float, y_margin: float) -> np.ndarray:
    z_lo = anchors["mouth_z"] - z_lo_k * anchors["span"]
    z_hi = anchors["mouth_z"] + z_hi_k * anchors["span"]
    y_half = (0.5 + y_margin) * anchors["mouth_w"]
    w = buf["world"]
    sel = (
        buf["mask"]
        & (w[..., 2] >= z_lo) & (w[..., 2] <= z_hi)
        & (w[..., 1] >= anchors["y_c"] - y_half) & (w[..., 1] <= anchors["y_c"] + y_half)
    )
    return sel & ndimage.binary_erosion(buf["mask"], iterations=MASK_EROSION)


def ridge_energy(buf: dict, sel: np.ndarray) -> float:
    """Coherent-ridge energy: vertical gradient of the vertical view-space
    normal component, smoothed ALONG x so isotropic micro-texture (stubble,
    marching-cubes noise) cancels while horizontally coherent ridges
    (striations, duplicated creases, lumpy carving) survive. The patch is
    resampled to a fixed row count so the number is scale-invariant."""
    if sel.sum() < 400:
        return float("nan")
    ny = buf["normals"][..., 1]
    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    filled = np.where(sel, ny, 0.0)[y0:y1, x0:x1]
    weight = sel[y0:y1, x0:x1].astype(float)
    zoom = (PATCH_ROWS / filled.shape[0], PATCH_ROWS / filled.shape[1])
    f_r = ndimage.zoom(filled, zoom, order=1)
    w_r = ndimage.zoom(weight, zoom, order=1)
    m_r = w_r > 0.5
    ny_r = np.where(m_r, f_r / np.maximum(w_r, 1e-6), 0.0)
    gy = np.gradient(ndimage.gaussian_filter(ny_r, RIDGE_SIGMA), axis=0)
    gy_x = ndimage.uniform_filter1d(gy, size=RIDGE_XWIN, axis=1)
    core = ndimage.binary_erosion(m_r, iterations=CORE_EROSION)
    if core.sum() < 100:
        return float("nan")
    return float(np.sqrt((gy_x[core] ** 2).mean())) * 1000.0


def dup_second_crease(buf: dict, anchors: dict) -> float:
    """Prominence of the SECOND-deepest crease in a narrow band around the
    mouth line, binned by world z (view-independent rows). One mouth = one
    dominant crease (chin crease stays mild); a duplicated mouth carves a
    rival crease of comparable depth."""
    z_lo = anchors["mouth_z"] - DUP_Z_HALF * anchors["span"]
    z_hi = anchors["mouth_z"] + DUP_Z_HALF * anchors["span"]
    y_half = DUP_Y_MARGIN * anchors["mouth_w"]
    w = buf["world"]
    sel = (
        buf["mask"]
        & (w[..., 2] >= z_lo) & (w[..., 2] <= z_hi)
        & (w[..., 1] >= anchors["y_c"] - y_half) & (w[..., 1] <= anchors["y_c"] + y_half)
    )
    sel = sel & ndimage.binary_erosion(buf["mask"], iterations=MASK_EROSION)
    if sel.sum() < 300:
        return float("nan")
    rows = 160
    z, v = w[..., 2][sel], buf["normals"][..., 1][sel]
    zi = np.clip(((z - z_lo) / max(z_hi - z_lo, 1e-9) * rows).astype(int), 0, rows - 1)
    prof, cnt = np.zeros(rows), np.zeros(rows)
    np.add.at(prof, zi, v)
    np.add.at(cnt, zi, 1.0)
    ok = cnt > 4
    prof = np.where(ok, prof / np.maximum(cnt, 1), np.nan)
    prof = prof[~np.isnan(prof)]
    if len(prof) < 60:
        return float("nan")
    p = ndimage.gaussian_filter1d(prof, 2.0)
    _, props = signal.find_peaks(-p, prominence=0.04, distance=5)
    proms = sorted(props["prominences"], reverse=True)
    return float(proms[1]) if len(proms) >= 2 else 0.0


# ---------------------------------------------------------------- texture metrics (v1, unchanged)
def subject_mask(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] == 4 and arr[..., 3].min() < 250:
        return arr[..., 3] > 10
    lum = arr[..., :3].mean(axis=2)
    return np.abs(lum - lum[0, 0]) > 8


def face_band(arr: np.ndarray, mask: np.ndarray, top_frac: float = 0.62):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    y_cut = y0 + int((y1 - y0 + 1) * top_frac)
    return arr[y0:y_cut, x0 : x1 + 1], mask[y0:y_cut, x0 : x1 + 1]


def tex_front_delta(tex_front: Image.Image, photo: Image.Image) -> float:
    from skimage import color as skcolor

    pair = []
    for img in (tex_front, photo):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
        band, bm = face_band(arr, subject_mask(arr))
        rgb = Image.fromarray(band[..., :3].astype(np.uint8)).resize((384, 384), Image.LANCZOS)
        mm = Image.fromarray((bm * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST)
        pair.append((np.asarray(rgb, dtype=np.float64), np.asarray(mm) > 127))
    (ra, rm), (pa, pm) = pair
    shared = rm & pm
    if not shared.any():
        return float("nan")
    la, lb = skcolor.rgb2lab(ra / 255.0), skcolor.rgb2lab(pa / 255.0)
    return round(float(np.sqrt(((la - lb) ** 2).sum(axis=2))[shared].mean()), 2)


def tex_oblique_ghost(tex_obl: Image.Image) -> float:
    arr = np.asarray(tex_obl.convert("RGBA"), dtype=np.float64)
    band, bm = face_band(arr, subject_mask(arr))
    lum = band[..., :3] @ np.array([0.299, 0.587, 0.114])
    dark = (lum < 60) & bm
    width = bm.sum(axis=1)
    frac = np.divide(dark.sum(axis=1), np.maximum(width, 1))
    rows = (frac > 0.30) & (width > 0)
    runs, in_run = 0, False
    for r in rows:
        if r and not in_run:
            runs, in_run = runs + 1, True
        elif not r and in_run:
            in_run = False
    extra_bands = max(0, runs - 1)
    mx = band[..., :3].max(axis=2)
    mn = band[..., :3].min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    speckle = ((sat > 0.55) & (mx > 120) & bm).sum() / max(bm.sum(), 1)
    return round(extra_bands + float(speckle) * 20.0, 3)


# ---------------------------------------------------------------- optional extras
class Extras:
    """SFace identity + LPIPS face similarity, loaded lazily and loudly
    degraded when unavailable (never rank drivers)."""

    def __init__(self, sface_path: Path | None):
        self.sface_path = sface_path
        self.lpips_fn = None
        try:
            import lpips
            import torch  # noqa: F401

            self.lpips_fn = lpips.LPIPS(net="squeeze", verbose=False).eval()
        except Exception as exc:
            note(f"lpips unavailable ({exc}); lpips_face skipped (pip install lpips)")

    def sface_embed(self, img_bgr: np.ndarray, det: dict):
        if self.sface_path is None:
            return None
        import cv2

        rec = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
        row = np.array(
            det["box"] + [c for pt in det["landmarks"] for c in pt] + [det["conf"]],
            dtype=np.float32,
        )
        aligned = rec.alignCrop(img_bgr, row)
        return rec.feature(aligned).flatten()

    def lpips_face(self, photo: Image.Image, render: Image.Image, det_p: dict, det_r: dict):
        if self.lpips_fn is None:
            return None
        import torch

        def crop(img, det, size=384):
            lm = np.array(det["landmarks"], dtype=np.float64)
            eye_mid, mouth_mid = 0.5 * (lm[0] + lm[1]), 0.5 * (lm[3] + lm[4])
            s = float(np.linalg.norm(mouth_mid - eye_mid))
            cx, cy = 0.5 * (eye_mid + mouth_mid)
            half = s * 1.55
            box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            return img.convert("RGB").crop(box).resize((size, size), Image.LANCZOS)

        def to_t(im):
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return float(self.lpips_fn(to_t(crop(photo, det_p)), to_t(crop(render, det_r))))


# ---------------------------------------------------------------- diagnostics
def landmark_row_delta(det_photo: dict, photo_mask: np.ndarray, det_render: dict, render_mask: np.ndarray) -> float:
    """|Δrow| of nose+mouth, each image normalized by its own bust
    silhouette (top row -> height). Red flag above ~0.2 for gross feature
    misplacement; measured within +-0.05 noise on the calibration set."""
    def rows(det, mask):
        ys = np.where(mask.any(axis=1))[0]
        top, h = float(ys.min()), float(max(ys.max() - ys.min(), 1))
        lm = np.array(det["landmarks"], dtype=np.float64)
        return (lm[2][1] - top) / h, (0.5 * (lm[3][1] + lm[4][1]) - top) / h

    pn, pm = rows(det_photo, photo_mask)
    rn, rm = rows(det_render, render_mask)
    return round(abs(pn - rn) + abs(pm - rm), 4)


# ---------------------------------------------------------------- evaluation
def load_meshes(glb_path: Path):
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    mesh_tex = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh_tex.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    mesh_clay = trimesh.Trimesh(
        vertices=np.asarray(mesh_tex.vertices), faces=np.asarray(mesh_tex.faces), process=False
    )
    mesh_clay.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    return mesh_clay, mesh_tex


def evaluate(
    glb: Path,
    photo: Image.Image,
    photo_ctx: dict,
    yunet: Path | None,
    extras: Extras,
    render_dir: Path | None,
) -> dict:
    from abstract3d.rendering import render_mesh_views

    mesh_clay, mesh_tex = load_meshes(glb)

    # --- geometry buffers (normal-map evaluation surface)
    rend = GeoRenderer(mesh_clay, size=1024)
    buf_front = rend.render(0.0, ELEVATION)
    buf_obl = {az: rend.render(az, ELEVATION) for az in OBLIQUE_AZIMUTHS}
    rend.release()

    det_n = None
    if yunet is not None:
        det_n = detect_face(pil_to_bgr(normal_image(buf_front)), yunet)
    if det_n is None:
        note(f"{glb.parent.name}: no face on front normal render — z-quantile fallback band (less precise)")
        anchors = fallback_anchors(buf_front)
    else:
        anchors = mouth_anchors(buf_front, det_n) or fallback_anchors(buf_front)

    # --- mesh metrics
    e_front = ridge_energy(buf_front, region_mask(buf_front, anchors, MOUTH_Z_BELOW, MOUTH_Z_ABOVE, MOUTH_Y_MARGIN))
    if np.isnan(e_front) and det_n is not None:
        # a misplaced detection can anchor the band off-surface; the
        # quantile band is coarser but always lands on the head
        note(f"{glb.parent.name}: landmark band empty (bad detection?) — retrying with z-quantile band")
        det_n = None
        anchors = fallback_anchors(buf_front)
        e_front = ridge_energy(buf_front, region_mask(buf_front, anchors, MOUTH_Z_BELOW, MOUTH_Z_ABOVE, MOUTH_Y_MARGIN))
    e_obl = {}
    for az, buf in buf_obl.items():
        e_obl[str(az)] = ridge_energy(buf, region_mask(buf, anchors, MOUTH_Z_BELOW, MOUTH_Z_ABOVE, MOUTH_Y_MARGIN))
    obl_vals = [v for v in e_obl.values() if not np.isnan(v)]
    mesh_oblique = float(np.mean(obl_vals)) if obl_vals else float("nan")

    dup_views = [dup_second_crease(b, anchors) for b in (buf_front, buf_obl[30.0], buf_obl[-30.0])]
    dup_views = [v for v in dup_views if not np.isnan(v)]
    dup_p2 = float(np.median(dup_views)) if dup_views else float("nan")
    surcharge = max(0.0, (dup_p2 - DUP_FREE_PROMINENCE)) * DUP_GAIN if not np.isnan(dup_p2) else 0.0
    mesh_front = e_front + surcharge if not np.isnan(e_front) else float("nan")

    # --- texture metrics (v1 renderer: flat-lit textured views)
    tex_front = render_mesh_views(mesh_tex, size=1536, azimuths=[0.0], elevation=0.0)[0]
    tex_obl = render_mesh_views(mesh_tex, size=1536, azimuths=[30.0], elevation=ELEVATION)[0]
    tex_de = tex_front_delta(tex_front, photo)
    tex_ghost = tex_oblique_ghost(tex_obl)

    result = {
        "mesh_front_defect": round(mesh_front, 2),
        "mesh_oblique_defect": round(mesh_oblique, 2),
        "tex_front_dE": tex_de,
        "tex_oblique_ghost": tex_ghost,
        "detail": {
            "ridge_front": round(e_front, 2) if not np.isnan(e_front) else None,
            "ridge_oblique": {k: (round(v, 2) if not np.isnan(v) else None) for k, v in e_obl.items()},
            "dup_second_crease": round(dup_p2, 3) if not np.isnan(dup_p2) else None,
            "dup_surcharge": round(surcharge, 2),
            "normal_det_conf": round(det_n["conf"], 3) if det_n else None,
        },
    }

    # --- extras + diagnostics (photo-referenced; need detections)
    det_photo, photo_mask = photo_ctx["det"], photo_ctx["mask"]
    if det_photo is not None and det_n is not None:
        result["detail"]["landmark_row_delta"] = landmark_row_delta(
            det_photo, photo_mask, det_n, buf_front["mask"]
        )
    if det_photo is not None and yunet is not None:
        tex_bgr = pil_to_bgr(tex_front)
        det_t = detect_face(tex_bgr, yunet)
        # Textured renders are in-domain for YuNet (all calibration bundles
        # scored >=0.86); a low-threshold hit here is a false positive on a
        # non-face subject and would feed garbage into the extras.
        if det_t is not None and det_t["conf"] < 0.5:
            note(f"{glb.parent.name}: textured-render face conf {det_t['conf']:.2f} < 0.5 — extras skipped")
            det_t = None
        if det_t is not None:
            emb_r = extras.sface_embed(tex_bgr, det_t)
            emb_p = photo_ctx.get("emb")
            if emb_r is not None and emb_p is not None:
                cos = float(np.dot(emb_p, emb_r) / (np.linalg.norm(emb_p) * np.linalg.norm(emb_r) + 1e-9))
                result["sface_identity"] = round(cos, 4)
            lp = extras.lpips_face(photo, tex_front, det_photo, det_t)
            if lp is not None:
                result["lpips_face"] = round(lp, 4)

    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        label = glb.parent.name if glb.name == "scene.glb" else glb.stem
        strip = Image.new("RGB", (5 * 512, 512))
        for i, buf in enumerate([buf_front] + [buf_obl[az] for az in OBLIQUE_AZIMUTHS]):
            strip.paste(normal_image(buf).resize((512, 512), Image.LANCZOS), (i * 512, 0))
        strip.save(render_dir / f"{label}_normals.png")

    return result


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--glb", nargs="+", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--render-dir", type=Path, help="save normal-map strips here for human verification")
    parser.add_argument("--no-download", action="store_true", help="never download face models")
    args = parser.parse_args()

    yunet = ensure_model(YUNET_FILE, YUNET_URL, not args.no_download)
    sface = ensure_model(SFACE_FILE, SFACE_URL, not args.no_download)
    if yunet is None:
        note("YuNet unavailable: regions fall back to z-quantile bands; landmark/identity metrics skipped")
    extras = Extras(sface)

    photo = Image.open(args.photo)
    photo_bgr = pil_to_bgr(photo)
    det_photo = detect_face(photo_bgr, yunet) if yunet is not None else None
    if det_photo is None:
        note("no face detected on the photo — photo-referenced extras disabled")
    photo_ctx = {
        "det": det_photo,
        "mask": subject_mask(np.asarray(photo.convert("RGBA"), dtype=np.float64)),
        "emb": extras.sface_embed(photo_bgr, det_photo) if det_photo is not None else None,
    }

    results = {}
    for glb in args.glb:
        label = glb.parent.name if glb.name == "scene.glb" else glb.stem
        try:
            results[label] = evaluate(glb, photo, photo_ctx, yunet, extras, args.render_dir)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
        headline = {k: v for k, v in results[label].items() if k != "detail"}
        print(f"{label:28s}", json.dumps(headline))

    # ranking table: mesh axis sorted by front+oblique defect sum
    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        print("\nMesh ranking (best -> worst, mesh_front_defect + mesh_oblique_defect):")
        print(f"{'bundle':28s} {'mesh_front':>10s} {'mesh_obl':>9s} {'mesh_sum':>9s} {'tex_dE':>7s} {'ghost':>6s}")
        def mesh_sum(v):
            a, b = v.get("mesh_front_defect"), v.get("mesh_oblique_defect")
            return (a if a == a else 1e9) + (b if b == b else 1e9)
        for label in sorted(ok, key=lambda k: mesh_sum(ok[k])):
            v = ok[label]
            print(
                f"{label:28s} {v['mesh_front_defect']:10.2f} {v['mesh_oblique_defect']:9.2f} "
                f"{mesh_sum(v):9.2f} {v['tex_front_dE']:7.2f} {v['tex_oblique_ghost']:6.2f}"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
