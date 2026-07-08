# FIXER AGENT 4 — Mathematical audit of the texture bake path

Date: 2026-07-05. Repo: `/Users/albou/abstract3d` (read-only audit; no repository file modified).
Audited state: snapshot at `/tmp/fixer4/snapshot/` — `texturing.py` md5 `6e836589…` (2464 lines),
`triposr_runtime.py` md5 `8cc9c293…` (2770 lines), `rendering.py` md5 `02f592bc…`. The workspace
files were being edited by other agents during the audit; the snapshot was re-verified identical
to the workspace immediately before this report was written.

Method: for every function, (1) the intended math was restated, (2) a minimal numerical test with
an analytically known answer was built (unit spheres, unit quads, synthetic masks/atlases),
(3) run against the live code, (4) PASS with a measured error bound or FAIL with the wrong term
identified and a corrected formula verified by the same test. All raw results:
`/tmp/fixer4/results.jsonl`; all harnesses: `/tmp/fixer4/audit_*.py` (rerunnable with
`source .venv/bin/activate && python /tmp/fixer4/audit_X.py` from the repo root).

Environment: numpy 2.4.6, scipy 1.18.0, PIL 12.3.0, trimesh 4.12.2, xatlas 0.0.11,
moderngl 5.12.0 (standalone GL context available: Apple M4 Max), torch 2.12.1.

---

## 1. Executive summary — ranked contribution to the observed defects

The shipped artifact (`artifacts/validation/iter3-multiview-fixed/face-2mv`, baked 04:54) shows a
face whose features are painted rotated/displaced ("three-quarter face distortion"), foreign
hair-colored patches stamped on skin and vice versa, and smooth pale ("milky") regions.
Note on timeline: the code has changed since that bake (the metadata's `source_pose.estimated:
false` proves the new photometric pose stage did not run for it). The ranking below separates
what corrupts bakes in the CURRENT code state from what corrupted the SHIPPED one.

| Rank | Finding | State | Verdict | Likely contribution |
|---|---|---|---|---|
| 1 | `estimate_pose_photometric` estimates the shipped face's KNOWN-FRONTAL source photo at **azimuth 25°** (matted; 22.5° raw). The bake applies this offset to the source view unconditionally in the orthographic branch. A 25° pose error paints the entire face rotated a quarter-turn-ish around the head — exactly a "three-quarter" distortion — and creates massive front-vs-profile overlap conflicts → weight zeroing → fill → milky patches. Its acceptance margin gate (0.002) is 5× smaller than the measured false-positive margin (0.0105). | CURRENT (added after the shipped bake) | FAIL (G3/G3b) | **Primary for all future bakes.** The perspective branch's own comment records that a photometric NCC pose stage was previously tried and reverted for exactly this failure mode ("decisively preferred poses 15 degrees off frontal"). |
| 2 | `filter_projection_outliers` is mathematically a no-op for its designed target: the 2-hop consensus `A + A@A` includes each texel's own vote with multiplicity ≈ vertex degree (6), so a misprojected island dominates both the view vote and the color consensus of its own members. A planted 3×3 foreign island (color deviation 0.55, foreign winning view) survives all 3 iterations untouched. The shipped contact sheet shows precisely the artifacts this filter exists to remove. Fix (self-vote exclusion + binarization) verified: island fully dropped, genuine detail kept, agreement case untouched. | SHIPPED + CURRENT | FAIL (C5, C6) → fix verified (C5-fix, C6-fix) | **High for the shipped patches** (skin stamped on hair shells, hair fringe on skin persisting into the final texture). |
| 3 | Registration scale bias: `_splat_silhouette` dilates the mesh mask by ~1.3 px (dilation 1 + closing 2) but the photo mask is not dilated, so the chamfer optimum systematically **enlarges every photo by ~3–4%** (the residual grid offers 1.04 and picks it even for an exactly canonical photo — verified). At 1024 px canonical frames that displaces features radially by up to ~17 px per view, differently per view → cross-view feature disagreement on three-quarter surfaces → ghosting/conflict churn. Erosion-compensated splat restores identity (verified). | SHIPPED + CURRENT | FAIL (G1, also visible in A6a: scale 1.05) | **High for feature displacement + cross-view disagreement.** |
| 4 | Strict z-buffer self-rejection band: scalar epsilon `0.0025·diag` vs per-bin depth spread `tan(tilt)·pixel` means smooth, genuinely visible surfaces tilted above a resolution-dependent onset are wrongly zeroed (39.6% of facing>0.25 texels at 55–75° for H=512 frames; onset ~72° at the shipped 1024/0.94 scales, 33% at 75°). Wrongly zeroed texels are demoted to unobserved and handed to the harmonic fill → **milky streaks on strongly tilted surfaces** (nose flanks, jaw undersides, cheek turn). Slope-aware epsilon (support 2.5 px) verified: 0% wrong rejection ≤70°, 5.1% residual at 75° (weights there <0.01), two-sheet occlusion and sphere untouched. | SHIPPED + CURRENT | FAIL (B5-final) → fix verified | **Medium** at shipped scales (the band sits at weights <0.02); larger for lower-res photos/frames. |
| 5 | `register_view_2d` / `refine_registration_photometric` convert the fitted x-shift back with `shift_x * width` although the fit frame is height-normalized on BOTH axes. On non-square photos the horizontal shift is scaled by W/H (verified: 40 px injected error reproduced exactly on a W=2H photo; height-conversion patch lands within 0.5 px). All ortho-branch photos are square after the canonical recenter, so this does NOT affect the Hunyuan face path — it silently corrupts the TripoSR/perspective path for landscape/portrait photos. | CURRENT (perspective path) | FAIL (A6b/A6c) → fix verified | **None for this artifact; latent for TripoSR.** |
| 6 | Coverage arithmetic of the shipped bake: 3 views × ~20% each ≈ 57% observed union → 43% of the surface is `mesh_harmonic` fill. The fill itself is verified exact (matches an independent dense Dirichlet solve to 3e-8) and maximum-principle-bounded — the milky look is the CORRECT rendering of insufficient observed coverage, whose causes are ranks 1–4 plus genuine hair-shell occlusion. | both | context | — |

Everything else audited passed with tight error bounds (tables below), including the whole
V-flip chain (delta-texel round trips through CPU sampling AND the moderngl renderer), both
depth-map projection matrices, both atlas rasterizers' half-pixel conventions, the ortho/
perspective sampling formulas (dot→texel identity within 1 photo pixel), the facing-weight law
(2.7e-7), the harmonize gain/spread/attenuation laws, conflict priority logic, speckle/feather
laws, blend softmax algebra (7e-9), mirror twin lookup (exact), and the canonical recenter parity
with the upstream Hunyuan `ImageProcessorV2.recenter` (≤1 px @512).

---

## 2. Per-function verdict table

Error bounds are measured maxima from the harnesses (units in parentheses).

### texturing.py

| Function | Verdict | Measured bound | Notes |
|---|---|---|---|
| `recenter_to_canonical_frame` | PASS | ≤1 px @512 vs upstream cv2 implementation (bbox size and center, hard+soft alpha) | Upstream uses `mask.nonzero()` (any alpha>0) vs ours >12; upstream bbox is `max-min` (off by one vs ours `+1`); measured net ≤1 px. A1 |
| `canonical_ortho_half_extent` | PASS | rel. err ≤ 8.5e-8 | `max(extent)/2(1-border)` exact on analytic boxes incl. general az/el. A2 |
| `_splat_silhouette` (ortho + persp) | PASS with caveat | center 0.0 px; radius +1.3 px | Geometry exact; the +1.3 px silhouette growth from `dilation(1)+closing(2)` is the root of Finding 3. A3 |
| `estimate_view_pose` (grid) | PASS | exact | Center pose always scored; grid symmetric; window respected (5 configs incl. non-multiples). A4 |
| `estimate_camera_distance` | PASS / edge-case WARN | ≤0.06% rel. err in-frame (3 distances) | Fixed-point converges; height normalization correct. WARN: subject touching frame edges biases +7% (measured) — no crop-awareness. A5 |
| `estimate_pose_photometric` | **FAIL** | 25° az error on ground-truth-frontal shipped case | Margin gate 0.002 vs measured false margin 0.0105. Finding 1. G3/G3b |
| `register_view_by_width_profile` | PASS | scale 1.01 (true 1.0), top err 6.8 px @1024 on cropped-photo case | Contains dead term `(1.0-scale)*letter_offset*0.0`; height-based back-conversion correct. G2 |
| `register_view_2d` | **FAIL** ×2 | (a) scale bias +4% on canonical input; (b) x-shift off by W/H on non-square photos (40.5 px reproduced) | (a) = Finding 3 (splat dilation); (b) = Finding 5 (PIL affine uses `shift_x*width`, fit is height-normalized). Both fixes verified. A6, G1 |
| `harmonize_and_gate_projection` | PASS | gains ≤0.02 abs err; spread discriminates 0.0 vs 1.61; attenuation linear ≤2e-4 | Log-ratio IQR gate 0.7 ≈ ratio 2.0 = the gain clamp — coherent. Degenerate inputs safe. C1–C3 |
| `resolve_projection_conflicts` | PASS | exact | Priority floor honored (source>0.45 wins; grazing source loses); no-conflict case untouched. C4 |
| `filter_projection_outliers` | **FAIL** | planted foreign island: 0/9 dropped; strip: 8/12 | Self-vote in `A+A@A` (diag = degree) + path-count weighting dilute the consensus. Drop-index bookkeeping itself is correct (no aliasing: boolean `drop[texel_sel] = ~alive` maps in construction order). Fix verified. C5/C6 |
| `mesh_graph_harmonic_fill` | PASS | 2.95e-8 vs independent dense solve; constant boundary 2.4e-8 | Conductance floor 0.05²=2.5e-3 keeps L_uu an irreducibly dominant M-matrix while boundary is reachable; near-zero-conductance solve stable (D4). WARN: unobserved DISCONNECTED component ⇒ singular block, fills black (measured) instead of falling back — guard with `np.isfinite` check. Uniform graph weights (not cotangent) give O(1) discretization error on anisotropic triangulations — bounded by max principle, acceptable for color. D1–D4, D2i |
| `mirror_fill_from_observed` | PASS | exact twin colors (0.0000); 0 visible-side writes; coverage 0.905 | Weight gate verified both directions (fallback <500 confident, restriction ≥500). Threshold basis = observed-bbox diagonal (see thresholds). D5/D6 |
| `inpaint_unseen_texels` | PASS | constant field 1.7e-7; empty-observation → 0.5 gray | D7 |
| `remove_speckle_weights` | PASS | exact | Small∧weak removed; small∧strong kept; big kept. C7 |
| `feather_projection_weight` | PASS | 2e-8 vs `w·clip(EDT/f,0,1)` | Never zeroes a covered texel. C7b |
| `blend_projections` | PASS | softmax algebra 7.2e-9; coverage semantics correct | Raw coverage preserved under feather; rim reclaim bounded by feather+1. C8 |
| `bake_projection_texture` (ortho orchestration) | PASS (frame math) / FAIL (pose stage) | E2E centered sphere: mean RGB err 0.0046 (≈1% of world); off-center mesh: residual search recovers 0.12/-0.08 offsets → 0.0045 | Recenter → half-extent → projector → atlas → PNG round-trip is mutually consistent (F1/F2). The shipped face mesh is nearly centered (2–6 px offsets @1024, F3) so the canonical-translation assumption holds. The pose stage is Finding 1. |

### backends/triposr_runtime.py

| Function | Verdict | Measured bound | Notes |
|---|---|---|---|
| `_tripo_project_observed_texture` ortho sampling | PASS | dot→texel identity ≤1 photo px (B1: 0.005 world = 1/3 px) | `sample = 0.5·H/half·x_cam + W/2 − 0.5` maps world→pixel-center indices exactly; −0.5 convention proven by the dot test. |
| — perspective sampling | PASS | ≤0.0073 world (1 px = 0.0114) on W≠H frame | Height-based focal for both axes confirmed. B2r |
| — facing weight law | PASS | 2.7e-7 vs `alpha·((f−0.2)/0.8)²` | B3 |
| — strict z-buffer (two sheets) | PASS | 0 leaked texels for gap>ε; leaks (correctly) for gap<ε | Occluder set includes all in-window surface texels regardless of alpha/facing ✓; bins `round(sample)` consistent with the −0.5 center convention ✓. B4r |
| — z-buffer smooth-surface self-rejection | **FAIL** | 39.6% of facing>0.25 texels zeroed at 55–75° tilt (H=512); onset 72° at shipped scales | Scalar ε ignores per-bin depth spread `tan(θ)·px·(3×3 support)`. Slope-aware ε verified (0% ≤70°). Finding 4. B5-final |
| — sparse-vs-dense sampling regimes | PASS | sphere at 39k verts: 0 wrong rejections @512/1024 | When texels are sparser than photo pixels each bin self-compares; the band only appears in the dense regime (2048 atlas + 1024 photo = ~6 texels/bin). B6 |
| `_tripo_render_camera_depth_map` (ortho + persp) | PASS | depth exact to <1e-3·far; footprint ≤0.4 px; orientation (flipud) exact | Ortho matrix rows match GL convention (`P22=2/(n−f)`, `P23=(f+n)/(n−f)`, `P33=1`); perspective `P32=−1` branch consistent; both upload `.T` (column-major) identically. E3/E3b |
| `_tripo_texture_image` / flip chain | PASS | exact | Atlas row0 = v≈0 (proven by rasterizer E5); single FLIP_TOP_BOTTOM; sampled back with `(u, 1−v)`: delta-texel identity CPU (E1), trimesh sampler (E2a), moderngl shader `texture(u_tex, vec2(u, 1−v))` (E2b). |
| `_tripo_edge_bleed_texture` | PASS | covered texels intact (≤0.002, uint8 rounding); gaps = nearest color | E4 |
| `_tripo_build_textured_mesh` | PASS | via E2a/E2b round trips | UVs passed through untouched; vmapping-expanded vertices/normals. |
| `_tripo_rasterize_vec3_atlas_cpu` / `_moderngl` | PASS | texel-center value error 0.0000 (≤half-texel bound); CPU≡GL on common coverage (0.0) | Both use pixel-center = (i+0.5)/R convention. E5 |
| dead parameter | WARN | — | `depth_tolerance=0.02` in the projector signature is unused since the strict z-buffer replaced the GL depth test. |

### rendering.py (V-flip witnesses)

| Function | Verdict | Notes |
|---|---|---|
| `_render_mesh_views_moderngl` texture path | PASS | Uploads PIL bytes (row0=top), samples `(u,1−v)` — E2b proves a v-top band renders at image top. |
| `_sample_texture_vertex_colors` | PASS (preview-only caveat) | Uses `(1−v)·(H−1)` nearest lookup — up to half-texel bias at chart edges; only used for matplotlib preview fallback. |
| `_orthographic_projection` | PASS | Standard GL ortho (differs in sign layout from the depth-map one but algebraically identical). |

---

## 3. FAIL details with verified patches

### 3.1 Finding 1 — `estimate_pose_photometric` rotates the source pose (texturing.py:440–574, applied at 2053–2079)

Intended math: maximize interior-weighted signed-gradient correlation between the recentered
photo and untextured renders over ±40° azimuth; accept only if the best beats the declared pose
by `min_margin=0.002`.

Measured: on the shipped `face-2mv` geometry + its own conditioning photo (ground truth az=0,
because the 2mv geometry was conditioned on exactly this front view), the scorer returns
az=25°, el=8° with score 0.0141 vs 0.0036 at declared (matted input; 22.5° on the raw photo).
The margin gate would need to be ~0.011 to reject this single case — but then it would reject
everything. The photo's shading gradients (studio soft light, hair shadows) do not match the
Lambertian gray render's geometry gradients well enough for this scorer to have a usable
peak at frontal poses; its landscape rewards silhouette-interior asymmetries (hair parting).

There is no small-constant fix; the signal is wrong on real faces, the same conclusion the
repo itself already recorded for the earlier NCC tie-break in the perspective branch
(texturing.py:2097–2103). Verified patch = do not apply the offset (delete or bypass the
application block); with the block bypassed the source pose stays at the ground-truth 0.

```python
# BEFORE (texturing.py, orthographic branch of bake_projection_texture)
        if views:
            photometric_pose = estimate_pose_photometric(
                mesh, views[0]["rgba"],
                border_ratio=float(canonical_border_ratio), azimuth_window_deg=40.0,
            )
            source_pose = {...}
            if photometric_pose["estimated"]:
                views[0]["azimuth_deg"] = float(views[0].get("azimuth_deg", 0.0)) + float(
                    photometric_pose["azimuth_deg"])
                views[0]["elevation_deg"] = float(views[0].get("elevation_deg", 0.0)) + float(
                    photometric_pose["elevation_deg"])

# AFTER (verified: G3b ground truth restored trivially; E2E F1/F2 unaffected)
        # Photometric source-pose estimation REMOVED from the default path:
        # on the checked ground-truth case (2mv geometry conditioned on its
        # own front photo) the gradient-correlation scorer preferred a pose
        # 25 degrees off frontal with 5x the acceptance margin. Same failure
        # class as the reverted perspective-branch NCC tie-break.
```

If pose recovery is genuinely needed for turned-head sources, it must be validated on ground
truth photos with known yaw before it can gate anything, and its acceptance must require the
declared pose to be REJECTED by an interpretable measure (e.g. silhouette + landmark evidence),
not a 0.002 correlation delta.

### 3.2 Finding 2 — `filter_projection_outliers` self-vote (texturing.py:1845)

Intended math: a texel is dropped when its winning view differs from the 2-hop neighborhood's
dominant view AND its color deviates >0.3 from the neighborhood consensus.

Wrong term: `reach = adjacency + adjacency @ adjacency` — (i) the diagonal of `A@A` equals the
vertex degree, so every texel votes for ITSELF with weight ≈6 in both the view histogram and
the color consensus; (ii) off-diagonal entries are path counts (up to 2 for triangles), further
weighting nearby same-island members. Measured on a planted 3×3 foreign island: the island
center's own-view vote is 28 vs 14 from outside, and its color consensus deviation is diluted
to 0.183 < 0.3 → never dropped (the whole filter returned an empty mask in every island test).

```python
# BEFORE (texturing.py:1845)
    reach = adjacency + adjacency @ adjacency

# AFTER (verified: C5-fix island 9/9 dropped, detail kept, strip fully eroded,
#        agreement regression drops nothing)
    reach = adjacency + adjacency @ adjacency
    # A texel must not vote in its own consensus: the diagonal of A@A equals
    # the vertex degree, which lets any island >= a one-ring dominate its own
    # neighborhood histogram and dilute the color deviation below threshold.
    reach.setdiag(0.0)
    reach.eliminate_zeros()
    # Binarize: 2-hop REACH, not 2-hop path counts (triangles create weight-2
    # entries that overweight an island's mutual support).
    reach.data[:] = 1.0
```

Residual limitation (measured, not fixed): a SAME-VIEW island at deviation 0.58 erodes only
partially (2/9) because interior members' consensus still contains island neighbors and the
erosion reaches a fixpoint; the same-view "extreme" criterion (threshold+0.1) is inherently
weak. The primary target (foreign-view islands — hair tips stamped by the front photo, skin
stamped on hair shells by profiles) is fully handled by the patch.

Bookkeeping audit (requested): the iteration bookkeeping is correct — `alive` indexes the
fixed `texel_sel` ordering, `np.add.at` re-aggregates only living texels each pass, and
`drop[texel_sel] = ~alive` writes back in the same construction order (row-major over the
boolean mask, matching `positions[...][texel_sel]` extraction). No index aliasing exists;
the C6 misses were consensus-stall, eliminated by the patch above.

### 3.3 Finding 3 — splat-dilation scale bias in registration (texturing.py:74–80 / 921–930)

Intended math: `register_view_2d` maximizes symmetric edge chamfer between the photo mask and
the mesh silhouette; identity must win for an exactly canonical photo.

Wrong term: the mesh mask from `_splat_silhouette` is grown ~1.3 px by `binary_dilation(1)` +
`binary_closing(2)` while the photo mask is not; the chamfer optimum therefore ENLARGES the
photo by 2·1.3/D (D = subject diameter in the 96 grid ≈ 68 px → +3.8%), and the residual scale
grid {0.96, 1.0, 1.04} locks onto 1.04. Measured: for a pixel-perfect canonical photo of a
centered sphere, current code returns `applied=True, scale=1.04` (G1); the default grid picks
1.05 (A6a). Every view gets a slightly different radial feature displacement (up to ~17 px
@1024 at the silhouette, ~8 px at eye level).

```python
# BEFORE (texturing.py, _splat_silhouette tail)
        mask = binary_dilation(mask, iterations=1)
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)

# AFTER (verified: G1 — registration returns identity (applied=False) on the
#        canonical photo; A3 geometry checks unchanged within bounds)
        mask = binary_dilation(mask, iterations=1)
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
        # The dilation grows the silhouette rim by ~1.3 px, which biases any
        # edge-based registration toward enlarging the photo by 2r/D. Erode
        # once to restore the true rim; interior gaps stay closed.
        mask = binary_erosion(mask, iterations=1, border_value=False)
```

(`binary_erosion` is already imported in the same `try` block's module; add it to the import.)

### 3.4 Finding 4 — z-buffer slope self-rejection (triposr_runtime.py:1591–1599)

Intended math: strict first-surface visibility; a texel is visible iff its depth is within ε of
the minimum depth in its (3×3 min-filtered) photo-pixel bin.

Wrong term: scalar `epsilon = 0.0025 * diagonal_zb` ignores that within the min-filter's
support (≈2.5 px: ±1 bin + intra-bin offsets) a SMOOTH surface's own depth varies by
`tan(tilt) · pixel_world · support`. Whenever that exceeds ε, the surface occludes itself.
Measured (facing>0.25 metric, in-frame): 39.6% of genuinely visible texels zeroed at 55–75°
tilt for H=512 frames; onset ~72° at the shipped scales (H=1024, half=0.94, ε/px=4.7). The
zeroed texels are demoted to unobserved and harmonically filled (milky).

```python
# BEFORE (triposr_runtime.py:1598)
        epsilon = 0.0025 * diagonal_zb
        visibility = depth_world <= nearest[bins_y, bins_x] + epsilon

# AFTER (verified: 0% wrong rejection at 45-70 deg, 5.1% residual at 75 deg
#        where weights are < 0.01; two-sheet blocking and sphere unchanged)
        # Slope-aware bias (standard shadow-mapping practice): within the
        # 3x3 min-filter's ~2.5 px support a smooth surface's own depth
        # varies by tan(tilt) * pixel size; a scalar epsilon must otherwise
        # choose between leaking hidden sheets and self-rejecting tilted
        # surfaces. facing is the cosine of the local tilt.
        if str(projection_model) == "orthographic":
            pixel_world = 1.0 / ortho_scale
        else:
            pixel_world = np.maximum(-z_cam, 1e-6) / focal
        slope = np.sqrt(np.clip(1.0 - facing ** 2, 0.0, 1.0)) / np.maximum(facing, 0.05)
        epsilon = 0.0025 * diagonal_zb + 2.5 * pixel_world * slope
        visibility = depth_world <= nearest[bins_y, bins_x] + epsilon
```

Trade-off (quantified): at facing 0.26 the added bias is ≈9 px of depth — a hidden sheet closer
than that behind a GRAZING front surface can leak, but such texels carry weight <0.01 and are
dominated by any direct view. Front-on sheets (the ghosting case that motivated the strict
z-buffer) keep the base ε exactly (slope=0).

### 3.5 Finding 5 — shift unit back-conversion (texturing.py:956, same pattern at 1080)

Answer to the scope question: shifts are fitted in a 96×96 square frame into which the photo is
letterboxed with BOTH axes scaled by `size/full_h` — i.e. HEIGHT-normalized; `shift_x` is a
fraction of the frame side = photo height. The PIL back-conversion uses
`center_x - inv_scale * (center_x + shift_x * width)` — WIDTH — so on non-square photos the
horizontal shift is applied W/H too large. Verified: a −40/H fitted shift moves a delta dot by
−40.5 px with the width conversion (wrong, should return to center) and −0.5 px with the height
conversion; end-to-end, a (40,−28) px injected offset on a W=2H photo leaves a −50 px residual
(A6b) vs −4 px on a square photo (A6a).

```python
# BEFORE (texturing.py:953-960, register_view_2d; same fix applies to
#         refine_registration_photometric at 1075-1082)
    matrix = (
        inv_scale, 0.0, center_x - inv_scale * (center_x + shift_x * width),
        0.0, inv_scale, center_y - inv_scale * (center_y + shift_y * height),
    )

# AFTER (verified by A6c: dot returns to center within 0.5 px)
    # The fit frame is height-normalized on BOTH axes (aspect-preserving
    # letterbox), so both shifts are fractions of the photo HEIGHT.
    matrix = (
        inv_scale, 0.0, center_x - inv_scale * (center_x + shift_x * height),
        0.0, inv_scale, center_y - inv_scale * (center_y + shift_y * height),
    )
```

No effect on the ortho path (recentered photos are square); corrupts perspective-path
registration for any non-square photo.

### 3.6 Minor / latent (no patch shipped, guards recommended)

- `estimate_camera_distance`: subjects touching the frame edge bias the estimate (+7% measured
  at d=1.5 with a frame-overflowing sphere). Detect a foreground bbox touching ≥1 frame edge
  and skip to the default (or use the untouched axis only). Perspective path only.
- `mesh_graph_harmonic_fill`: a fully-unobserved DISCONNECTED mesh component makes `L_uu`
  singular; measured behavior fills it black (spsolve emitted zeros; on other platforms it can
  be NaN). Guard: `if not np.isfinite(solved).all(): return None` (falls back to the KD fill),
  or solve per connected component. Latent (the pipeline currently prunes to one component).
- `register_view_by_width_profile`: dead term `(1.0 - scale) * letter_offset * 0.0` (line 737)
  — delete for clarity (the centered letterbox makes the offset cancel; the `*0.0` is a
  confusing leftover).
- Projector signature: `depth_tolerance=0.02` is dead since the strict z-buffer; remove.
- `harmonize_and_gate_projection` defaults (0.16/0.34) differ from the values the bake actually
  passes (0.24/0.4); the defaults are dead in the pipeline — align them or drop the defaults.

---

## 4. Threshold system audit

Units legend: [cos] = cosine of angle between normal and view ray; [w] = blend-weight units
(alpha·((facing−0.2)/0.8)², so [w] is a nonlinear function of [cos]); [rgb] = mean absolute
RGB difference in [0,1]; [fr] = fraction of frame side; [tex] = atlas texels; [dw] = world
distance; [diag] = fraction of a bounding-box diagonal.

| Constant | Where | Value | Unit / scale basis | Assessment |
|---|---|---|---|---|
| facing_threshold | projector | 0.2 | [cos] → 78.5° | OK. Interacts with ε (see below). |
| weight law exponent | projector | 2 | — | Verified exact (B3). |
| z-buffer ε | projector | 0.0025 | [diag of surface bbox] → 0.0087 [dw] on the face | **Scale-inconsistent** with the facing cut: self-rejection onset (72.4° at 1024/0.94, 53° at 512-px frames) lands INSIDE the accepted facing range (<78.5°). The band is resolution-dependent because ε is mesh-scaled while the bin spread is pixel-scaled. Fix 3.4 makes it facing-aware; alternatively raising facing_threshold to ≥cos(onset) would be scale-fragile. |
| z-buffer min-filter | projector | 3×3 | photo px | OK (conservative widening); its 2.5 px support is exactly the factor in fix 3.4. |
| occluder window | projector | [−0.5, W−0.5] | photo px | OK; wider than the paint window [0, W−1] by design (occluders need no bilinear support). |
| z_cam cut | projector | −1e-4 | [dw] | OK at unit scales. |
| overlap membership | harmonize | 0.05 | [w] ≡ facing 0.38 (67.7°) at alpha 1 | OK; consistent with "compare only reliable texels". |
| min_overlap_texels | harmonize | 400 | [tex] @2048 atlas | **Not resolution-scaled**: at 512 atlas the same physical patch is 16× fewer texels. Recommend `max(64, (res/2048)² · 400)`. |
| usable ref floor | harmonize | 0.02 | mean rgb of texel | OK (guards log-ratio explosion; verified C3). |
| spread gate | harmonize | 0.7 | log-ratio IQR ≡ ratio 2.01 | Coherent with the gain clamp [0.5, 2.0] (spread beyond the clamp's dynamic range = content). Verified separation: 0.0 pure exposure vs 1.61 content (C2). |
| gain clamp | harmonize | [0.5, 2.0] | ratio | OK. |
| accept-after-gain | harmonize | ≤attenuate or −0.02 | [rgb] | OK. |
| attenuate_above (live) | bake→harmonize | 0.24 | [rgb] | **Ordering bug vs conflict 0.25**: the comment says the global gate "engages later than its per-texel counterpart" but 0.24 < 0.25 engages EARLIER. Set to 0.30 (or ≥0.25) to match the stated design. |
| reject_above (live) | bake→harmonize | 0.4 | [rgb] | OK given attenuate fix; span 0.24→0.4 verified linear (C2b). |
| conflict_threshold | conflicts | 0.25 | [rgb] | OK as the per-texel gate; see ordering above. |
| conflict min_weight | conflicts | 0.05 | [w] | Same value as overlap membership — coherent. |
| priority_floor | conflicts | 0.45 | [w] ≡ facing 0.74 (42.6°) at alpha 1 | OK conceptually ("solid facing"); note it is compared on PRE-feather weights (order verified) but POST-harmonize attenuation for references — intended asymmetry. |
| outlier color_threshold | outlier filter | 0.3 / extreme +0.1 | [rgb] vs 2-hop consensus | Unreachable for islands ≥ one ring in the current code (Finding 2); with the fix, 0.3/0.4 verified to separate a 0.55-dev island from a 0.35-dev genuine detail. |
| min_neighbor_support | outlier filter | 3 | texel count in 2-hop | OK. |
| outlier weight floor | outlier filter | 0.05 | [w] | Consistent with the other 0.05 floors. |
| speckle floor | blend | max(4, HW/262144) = 16 @2048 | [tex], resolution-scaled | OK (verified law C7). |
| speckle strong_weight | blend | 0.5 | [w] | OK. |
| feather | bake | max(4, res/512·3) = 12 @2048 | [tex], resolution-scaled | OK; ramp law verified (the 6.0 in the function signature is a dead default — the bake always passes the scaled value). Feathered weights are what `mirror_fill` sees (below). |
| rim reclaim | blend | feather+1 | [tex] | OK (bounded gather; verified semantics C8b). |
| blend sharpness | bake | 3.0 | 1/[w] | Verified algebra (C8). |
| symmetry gate | bake | 0.55 | score = 1 − median-mirror-NN/0.10 [diag of vertex bbox] → gate ≡ 4.5% diag | OK; shipped face scored 0.966. |
| mirror twin distance | mirror fill | 0.02 | [diag of OBSERVED positions bbox] | **Different diagonal basis** than the symmetry score (vertex bbox): with sparse observation the threshold shrinks — conservative direction, acceptable, but document it. |
| mirror min_source_weight | mirror fill | 0.35 | [w] POST-feather (blend's max feathered weight) | Unit subtlety: texels within ~4 texels of a seam fall below 0.35 by the ramp alone and are excluded as sources — conservative, OK. Confident-set floor 500 [tex] is not resolution-scaled (same remark as min_overlap). |
| mirror added weight | bake | 0.85 | [w] | OK (below priority floor semantics don't apply post-blend). |
| pose min_iou | estimate_view_pose | 0.45 (source) / 0.4 (refine) | IoU | OK. |
| pose prior_strength | estimate_view_pose | 0.12 / 0.06 | IoU per window | OK. |
| refine accept margin | bake | +0.02 | IoU | OK (empirically motivated per comments). |
| register gate | register_view_2d | +0.25 | 96-grid px (chamfer is negative mean edge distance) | OK once the splat bias (3.3) is fixed; today the bias manufactures gains larger than the gate. |
| register grids | register_view_2d | scales 0.7–1.45/0.05 (refs, perspective), (0.96,1.0,1.04) + shift ±0.06/0.02 (ortho) | [fr] | Residual ±0.06 range equals the F2-measured need for a 0.12 world offset — at its EDGE; meshes off-center by more than ~6% of the frame will clip. Acceptable given F3 measured 0.2–0.6% on the shipped mesh. |
| width-profile scale range | width profile | 0.45–1.6 / 0.02 | ratio | OK. |
| photometric min_margin | estimate_pose_photometric | 0.002 | NCC units | **Ineffective**: measured false-positive margin 0.0105 (5×) on ground truth. No constant fixes this scorer (Finding 1). |
| erode_view_alpha | erode | min(8, min(size)/256) → 4 px @1024 | photo px, resolution-scaled | OK. |
| recenter border_ratio | bake/caller | 0.15 | fraction | Matches upstream Hunyuan exactly (A1). |
| distance clamps | estimate_camera_distance | [0.5, 4]×default; conv 0.005 | ratio | OK (A5). |
| splat distance | estimate_view_pose | ≥ r/sin(20°)·1.05 | [dw] | OK (bounding-sphere containment for fov 40). |

Coherent-set recommendation (minimal changes): keep the [rgb] family at
`conflict 0.25 < attenuate 0.30 < reject 0.45` (restores the documented per-texel-first
ordering with the same spans), scale `min_overlap_texels` and the mirror confident-set floor
by `(resolution/2048)²`, adopt the slope-aware ε (3.4) so the visibility test no longer
contradicts the facing threshold at any resolution, and remove the two dead knobs
(`depth_tolerance`, harmonize defaults 0.16/0.34).

---

## 5. Tests worth adding to the repo suite

From the audit harnesses (all in `/tmp/fixer4/`, self-contained, seconds each):

1. **Recenter parity** (A1): synthetic ellipse photo → `recenter_to_canonical_frame` vs vendored
   `ImageProcessorV2.recenter`, bbox size/center within 2 px @512. Guards the canonical-frame
   contract the whole ortho path rests on.
2. **Projector dot identity** (B1/B2r): 1-px dot photo → exactly the analytically-mapped texel
   receives it, ortho and perspective, W≠H. Locks the −0.5/center conventions and height-based
   focal.
3. **Facing weight law** (B3): measured weights ≡ `((f−0.2)/0.8)²` to 1e-6.
4. **Two-sheet z-buffer** (B4r): gap>ε fully blocked, gap<ε passes; front coverage 100%.
5. **Tilted-plane self-rejection** (B5): facing>0.25 texels on a smooth plane must keep weight
   at all tilts ≤70° for photo sizes 512/1024 — currently FAILS; passes with slope-aware ε.
6. **Outlier island** (C5/C6): planted foreign 3×3 island fully dropped, 0.35-dev same-view
   detail kept, agreement scenario drops nothing — currently FAILS; passes with self-vote fix.
7. **Harmonize laws** (C1/C2): exact gain recovery; bimodal-content spread gate; attenuation
   linearity.
8. **Registration identity** (G1): canonical photo of a centered sphere → `register_view_2d`
   must return identity — currently FAILS (scale 1.04).
9. **Shift unit round trip** (A6c): delta-dot warp back-conversion on a W=2H photo — currently
   FAILS with `shift_x*width`.
10. **Harmonic fill vs independent solve** (D2i): implementation ≡ dense Dirichlet solve to 1e-6;
    constant-boundary identity; disconnected-component finiteness guard.
11. **Mirror twin analytic** (D5): position-coded colors → filled color ≡ f(mirror(p)); zero
    visible-side writes.
12. **V-flip delta-texel** (E1/E2): atlas delta → CPU sample AND moderngl render read it at the
    analytically correct location.
13. **E2E ortho bake** (F1/F2): centered + off-center sphere with analytic canonical photo →
    mean baked RGB error <0.02, implied world offset <0.02.
14. **Pose-stage ground truth** (G3-style): any photometric/silhouette source-pose estimator must
    return ≈0 for a photo the geometry was conditioned on (use a cached small mesh + photo pair).

---

## 6. Scope-item cross-reference

- `estimate_view_pose` symmetric-offsets grid: PASS (A4) — center always scored, symmetric,
  window-bounded, including non-multiple window/step pairs.
- `estimate_camera_distance` fixed-point: PASS in-frame (≤0.06%); frame-touching bias documented.
- `register_view_2d` letterbox: height-normalized on both axes (verified); crop lines tracked
  through the warp (code-read + G2 cropped case); chamfer gate constant fine once the splat bias
  is fixed; **PIL back-conversion uses width for x — wrong for W≠H** (fix 3.5).
- `harmonize_and_gate_projection` log-ratio spread gate: PASS (clean separation, exact
  attenuation law); live thresholds' ordering vs conflict threshold flagged.
- `resolve_projection_conflicts` priority logic: PASS (both directions of the floor).
- `filter_projection_outliers`: **FAIL — self-vote consensus**; drop-index bookkeeping verified
  correct (no aliasing between iterations); fix verified.
- `mesh_graph_harmonic_fill`: implementation exact vs independent solve; conductance floor keeps
  the system nonsingular whenever unknowns connect to boundary; disconnected-component
  singularity documented with guard; near-zero conductance (0.0025) stable (direct solver).
- `mirror_fill_from_observed`: PASS incl. weight gating both ways and the 500-texel fallback.
- `inpaint_unseen_texels`: PASS incl. empty-observation fallback.
- `blend_projections`/`feather`/`speckle` with zeroed weights: PASS — raw coverage tracks
  post-zeroing weights; feather never zeroes; rim reclaim bounded.
- `bake_projection_texture` ortho orchestration: frame conventions mutually consistent
  (E2E F1/F2 ≤0.005 mean RGB); per-view half-extents and distances flow to both the wide
  (width-profile) and residual (register_view_2d) searches consistently — both height-normalized
  (G2 + code read); **the new photometric pose stage is the orchestration-level defect**.
- `_tripo_project_observed_texture`: sampling formulas PASS; strict z-buffer: bins/epsilon/
  min-filter audited — slope band FAIL + fix; occluder set correctly alpha/facing-independent;
  sub-pixel consistency of sampling vs bin coords verified (round vs floor+bilinear share the
  −0.5 center convention); photo-resolution vs atlas-density regimes measured (dense regime
  triggers the band, sparse does not).
- `_tripo_render_camera_depth_map`: ortho matrix rows vs perspective PASS (E3), orientation PASS.
- V-flip chain: PASS end-to-end (E1/E2a/E2b), rasterizer conventions PASS (E5, CPU≡GL).

---

## Appendix — artifacts

- `/tmp/fixer4/results.jsonl` — every check with status/error/detail.
- `/tmp/fixer4/audit_a_frame.py`, `audit_a2_fix.py` — frame/registration group.
- `/tmp/fixer4/audit_b_projector.py`, `audit_b2_fix.py`, `audit_b5_fixverify.py` — projector.
- `/tmp/fixer4/audit_c_blend.py`, `audit_c5_debug.py`, `audit_c5_fixverify.py`,
  `audit_c5b_fixverify.py` — blend/gates/outliers.
- `/tmp/fixer4/audit_d_fill.py`, `audit_d2_fix.py`, `audit_d2_independent.py` — fills.
- `/tmp/fixer4/audit_e_vflip.py` — V-flip + depth matrices + rasterizers.
- `/tmp/fixer4/audit_f_orchestration.py` — E2E ortho bakes + shipped-mesh offsets.
- `/tmp/fixer4/audit_g_bias.py`, `audit_g3b.py` — registration bias, width profile, photometric
  pose on the shipped case.
- `/tmp/fixer4/snapshot/` — the exact source state audited.
