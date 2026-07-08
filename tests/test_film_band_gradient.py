from __future__ import annotations

import numpy as np

from abstract3d import film_band, film_band_gradient


def test_skin_side_profile_measures_descending_falloff() -> None:
    # Photo: skin plateau on the right, dark body on the left, linear
    # transition ramp between them.
    size = 128
    image = np.zeros((size, size, 4), dtype=np.float32)
    image[:, :, 3] = 1.0
    image[:, :40, :3] = 0.1          # dark body
    image[:, 80:, :3] = 0.8          # skin plateau
    ramp = np.linspace(0.1, 0.8, 40, dtype=np.float32)
    image[:, 40:80, :3] = ramp[None, :, None]

    got = film_band.skin_side_profile(image)

    assert got is not None
    distances, profile = got
    assert profile[0] > 0.6                  # near the skin: high
    assert profile[-1] <= 0.15               # deep into the hair: low
    assert (np.diff(profile) <= 1e-6).all()  # isotonic descending


def test_photo_feature_components_split_brow_from_hairline() -> None:
    size = 160
    image = np.zeros((size, size, 4), dtype=np.float32)
    image[:, :, 3] = 1.0
    image[:, :, :3] = 0.8
    image[:, :30, :3] = 0.08                  # hair body (large, near-black)
    image[40:44, 32:60, :3] = 0.45            # wispy fringe INSIDE corridor
    image[100:106, 90:130, :3] = 0.45         # brow far from the body

    features = film_band.photo_feature_components(image, transition_px=20.0)

    assert features[100:106, 90:130].all()    # the brow is a feature
    assert not features[40:44, 32:60].any()   # the fringe is hairline


def _flat_positions(shape):
    positions = np.zeros((*shape, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    positions[:, :, 0] = xs / max(shape[1] - 1, 1)
    positions[:, :, 1] = ys / max(shape[0] - 1, 1)
    positions[:, :, 3] = 1.0
    return positions


def _projection(shape, rgb, *, weight, first, photo_products,
                veto=None, azimuth=0.0):
    rgba = np.zeros((*shape, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 1.0
    return {
        "rgba": rgba,
        "weight": np.full(shape, weight, dtype=np.float32),
        "azimuth_deg": azimuth,
        "elevation_deg": 0.0,
        "film_band": {
            "zone_texel": np.zeros(shape, dtype=bool),
            "added_texel": np.zeros(shape, dtype=bool),
            "commit_texel": np.zeros(shape, dtype=bool),
            "veto_texel": (np.zeros(shape, dtype=bool)
                           if veto is None else veto),
            "img_first_texel": first,
            "contested_texel": np.zeros(shape, dtype=bool),
            "body_weight_texel": np.zeros(shape, dtype=np.float32),
            "photo_products": photo_products,
        },
    }


def _band_scene(size: int = 96):
    """Synthetic hairline: dark mass rows 0..32, apron rows 32..64 filled
    with putty, skin rows 64..96. The source view witnesses the apron with
    its own gradient content."""
    shape = (size, size)
    positions = _flat_positions(shape)
    colors = np.zeros((*shape, 4), dtype=np.float32)
    colors[:, :, 3] = 1.0
    colors[:32, :, :3] = 0.10                      # observed hair mass
    colors[32:64, :, :3] = 0.55                    # putty apron (fill)
    colors[64:, :, :3] = 0.80                      # observed skin
    observed = np.zeros(shape, dtype=bool)
    observed[:32] = True
    observed[64:] = True

    # Source view: witnesses everything; its own content carries the
    # transition (0.1 at the mass edge -> 0.8 at the skin edge).
    src_rgb = np.zeros((*shape, 3), dtype=np.float32)
    src_rgb[:32] = 0.10
    src_rgb[64:] = 0.80
    ramp = np.linspace(0.12, 0.78, 32, dtype=np.float32)
    src_rgb[32:64] = ramp[:, None, None]
    first = np.ones(shape, dtype=bool)

    profile = (np.arange(1.0, 33.0, 2.0),
               np.clip(1.0 - np.arange(1.0, 33.0, 2.0) / 24.0, 0.0, 1.0))
    bins_y, bins_x = np.mgrid[0:size, 0:size]
    photo = {
        "bright_median": 0.8,
        "profile": profile,
        "transition_px": 24.0,
        "in_body_texel": np.zeros(shape, dtype=bool),
        "beyond_transition_texel": np.zeros(shape, dtype=bool),
        "feature_texel": np.zeros(shape, dtype=bool),
        "bins_y": bins_y.astype(np.int32),
        "bins_x": bins_x.astype(np.int32),
        "photo_shape": shape,
    }
    photo["in_body_texel"][:32] = True
    photo["beyond_transition_texel"][68:] = True

    source = _projection(shape, src_rgb, weight=0.8, first=first,
                         photo_products=photo)
    # Reference view: images everything, no film products needed beyond
    # the required keys; zero weight except a confident patch used by the
    # standoff test.
    ref_rgb = np.full((*shape, 3), 0.7, dtype=np.float32)
    reference = _projection(shape, ref_rgb, weight=0.0, first=first,
                            photo_products=photo, azimuth=90.0)
    return positions, colors, observed, source, reference


def test_repaint_replaces_apron_with_source_gradient() -> None:
    positions, colors, observed, source, reference = _band_scene()

    got = film_band_gradient.repaint_film_band(
        [source, reference],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=positions.shape[0],
    )

    assert got is not None
    out, stats, applied_mask = got
    assert stats["applied"]
    assert applied_mask.any()
    band = out[36:60, :, :3].mean(axis=2)
    src = np.asarray(source["rgba"], dtype=np.float32)[36:60, :, :3].mean(axis=2)
    # The apron carries the source's own gradient (authority stamps),
    # far from the flat 0.55 putty it started with.
    assert np.abs(band - src).mean() < 0.05
    # Monotone hair-to-skin read across the apron.
    rows = out[32:64, :, :3].mean(axis=2).mean(axis=1)
    assert rows[0] < 0.3 and rows[-1] > 0.6
    # Observed mass and skin are untouched.
    assert np.allclose(out[:30, :, :3], 0.10, atol=1e-5)
    assert np.allclose(out[66:, :, :3], 0.80, atol=1e-5)


def test_repaint_stands_off_confident_reference_territory() -> None:
    positions, colors, observed, source, reference = _band_scene()
    weight = np.asarray(reference["weight"], dtype=np.float32)
    weight[36:48, 30:60] = 0.9        # the reference owns this patch
    reference["weight"] = weight
    # Give the source DISTINCT content in the owned patch (much darker
    # than the profile tone there): with the standoff the patch must NOT
    # take it; without the standoff it would copy 0.15 verbatim.
    src_rgba = np.asarray(source["rgba"], dtype=np.float32)
    src_rgba[38:46, 34:56, :3] = 0.15
    source["rgba"] = src_rgba

    got = film_band_gradient.repaint_film_band(
        [source, reference],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=positions.shape[0],
    )

    assert got is not None
    out, stats, _ = got
    patch = out[38:46, 34:56, :3]
    assert np.abs(patch - 0.15).mean() > 0.05


def test_displacement_veto_rejects_vetoed_dark_stamps_in_skin_half() -> None:
    """FACE-20: a dark-stamp component that (a) another view positively
    vetoes as base material and (b) sits in the skin half of the field
    (S median >= 0.35) is parallax-displaced content. It must not print
    verbatim; the refill must stay strictly above the dark class so it
    cannot render as a stroke at any pose. Equally vetoed dark stamps
    near the mass, and unvetoed dark stamps anywhere, still stamp
    verbatim (the identity contract of cycle 3)."""
    positions, colors, observed, source, reference = _band_scene()
    src_rgba = np.asarray(source["rgba"], dtype=np.float32)
    # displaced candidate: skin-half apron (S ~ 0.7), elongated, dark
    src_rgba[54:62, 20:40, :3] = 0.08
    # control: same dark content at the same S, NO veto — the S gate
    # alone must not displacement-reject it
    src_rgba[54:62, 60:80, :3] = 0.08
    source["rgba"] = src_rgba
    veto = np.zeros(positions.shape[:2], dtype=bool)
    veto[54:62, 20:40] = True     # base-material witness at the candidate
    veto[33:41, 20:40] = True     # veto near the mass: S gate must keep it
    reference["film_band"]["veto_texel"] = veto
    # mass-side dark stamp under full veto (valid wisp class, S ~ 0.2);
    # dark-connected to the mass so island guards do not revert it
    src_rgba[33:41, 20:40, :3] = 0.08

    got = film_band_gradient.repaint_film_band(
        [source, reference],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=positions.shape[0],
    )

    assert got is not None
    out, stats, _ = got
    # exactly the VETOED skin-half component (8x20 texels) is rejected:
    # the unvetoed control at the same S and the vetoed mass-side stamps
    # pass the displacement gate (the control's later island-guard fate
    # is that guard's own, pre-existing semantics)
    assert stats["displaced_dark_vetoed"] == 160
    assert stats["displaced_refilled"] == 160
    dark_threshold = 0.55 * 0.8   # DARK_LUMINANCE_RATIO x bright median
    displaced_site = out[55:61, 22:38, :3].mean(axis=2)
    # not the verbatim stroke, and strictly above the dark class
    assert (displaced_site > dark_threshold).all()
    mass_side_site = out[34:40, 22:38, :3].mean(axis=2)
    assert np.abs(mass_side_site - 0.08).max() < 1e-5


def test_repaint_field_support_bound_refuses_far_treatment() -> None:
    """FACE-22: the S field is a distance RATIO and reaches arbitrarily far
    from the hair mass; treatment must stop within FIELD_SUPPORT_TRANSITIONS
    pooled transition lengths of the mass (where the profile was measured).
    A bright source stamp INSIDE the support still applies; the same stamp
    class beyond the support is refused and the surface keeps its
    composite color."""
    size = 96
    shape = (size, size)
    positions = _flat_positions(shape)
    colors = np.zeros((*shape, 4), dtype=np.float32)
    colors[:, :, 3] = 1.0
    colors[:24, :, :3] = 0.10                     # observed hair mass
    colors[24:64, :, :3] = 0.55                   # putty apron (fill)
    colors[64:84, :, :3] = 0.80                   # observed skin belt
    colors[84:92, :, :3] = 0.55                   # FAR unobserved putty strip
    colors[92:, :, :3] = 0.80                     # observed skin
    observed = np.zeros(shape, dtype=bool)
    observed[:24] = True
    observed[64:84] = True
    observed[92:] = True

    src_rgb = np.zeros((*shape, 3), dtype=np.float32)
    src_rgb[:24] = 0.10
    src_rgb[24:] = 0.80
    ramp = np.linspace(0.12, 0.78, 40, dtype=np.float32)
    src_rgb[24:64] = ramp[:, None, None]
    src_rgb[86:90, 24:36] = 0.95                  # far bright stamp content
    first = np.ones(shape, dtype=bool)

    # transition of 8 texels: passes the sampling-adequacy floor (>= 7)
    # with margin while 6 transitions = 48 texels — the far strip at 62+
    # texels from the mass sits beyond the support.
    transition_px = 8.0
    profile = (np.arange(1.0, 33.0, 2.0),
               np.clip(1.0 - np.arange(1.0, 33.0, 2.0) / 24.0, 0.0, 1.0))
    bins_y, bins_x = np.mgrid[0:size, 0:size]
    photo = {
        "bright_median": 0.8,
        "profile": profile,
        "transition_px": transition_px,
        "in_body_texel": np.zeros(shape, dtype=bool),
        "beyond_transition_texel": np.zeros(shape, dtype=bool),
        "feature_texel": np.zeros(shape, dtype=bool),
        "bins_y": bins_y.astype(np.int32),
        "bins_x": bins_x.astype(np.int32),
        "photo_shape": shape,
    }
    photo["in_body_texel"][:24] = True
    photo["beyond_transition_texel"][68:72] = True   # narrow skin ring

    source = _projection(shape, src_rgb, weight=0.8, first=first,
                         photo_products=photo)
    reference = _projection(shape, np.full((*shape, 3), 0.7, np.float32),
                            weight=0.0, first=first, photo_products=photo,
                            azimuth=90.0)

    got = film_band_gradient.repaint_film_band(
        [source, reference],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=size,
    )

    assert got is not None
    out, stats, applied_mask = got
    assert stats["applied"]
    # near apron: treated (authority carries the source's ramp)
    assert applied_mask[30:60, 20:70].mean() > 0.8
    # far strip: beyond the support — untreated, keeps the putty tone
    far = out[86:90, 24:36, :3].mean(axis=2)
    assert not applied_mask[86:90, 24:36].any()
    assert np.abs(far - 0.55).max() < 1e-4


def test_stamp_border_feather_ramps_into_untreated_surface() -> None:
    """FACE-22: authority stamps blend composite -> photo over the border
    band at treated-region edges, so the stamp/composite step cannot print
    as a line; deep interior stays verbatim photo; borders shared with the
    dark mass stay verbatim (the stamp continues the mass content)."""
    positions, colors, observed, source, reference = _band_scene()

    got = film_band_gradient.repaint_film_band(
        [source, reference],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=positions.shape[0],
    )

    assert got is not None
    out, stats, applied_mask = got
    src = np.asarray(source["rgba"], dtype=np.float32)[:, :, :3]
    # interior rows of the apron: verbatim photo
    interior = np.abs(out[40:56, 20:70, :3] - src[40:56, 20:70]).mean()
    assert interior < 1e-4
    # rows adjacent to the mass boundary (top): still verbatim — the mass
    # border is exempt from the feather
    top_edge = np.abs(out[32:34, 20:70, :3] - src[32:34, 20:70]).mean()
    assert top_edge < 1e-4


def test_repaint_noops_without_second_view_or_mass() -> None:
    positions, colors, observed, source, reference = _band_scene()

    assert film_band_gradient.repaint_film_band(
        [source],
        colors.copy(),
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed,
        texture_resolution=positions.shape[0],
    ) is None

    # No dark mass: brighten the "hair" rows.
    bright = colors.copy()
    bright[:32, :, :3] = 0.75
    observed_bright = observed.copy()
    assert film_band_gradient.repaint_film_band(
        [source, reference],
        bright,
        positions_texture=positions,
        normals_texture=None,
        observed_mask=observed_bright,
        texture_resolution=positions.shape[0],
    ) is None
