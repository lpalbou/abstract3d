from __future__ import annotations

import numpy as np

from abstract3d import film_band


def _photo(height: int = 96, width: int = 96) -> np.ndarray:
    """Bright base photo with a large dark body on the left third."""
    image = np.zeros((height, width, 4), dtype=np.float32)
    image[:, :, :3] = 0.8
    image[:, :, 3] = 1.0
    image[:, : width // 3, :3] = 0.1  # dark film body (large component)
    return image


def test_dark_body_mask_keeps_large_components_only() -> None:
    image = _photo()
    image[48:52, 60:64, :3] = 0.05  # small dark island (an "eye")

    body = film_band.dark_body_mask(image)

    assert body[:, :10].all()  # the big body is kept
    assert not body[48:52, 60:64].any()  # the island is not


def test_compute_view_film_maps_extends_zone_and_vetoes_base() -> None:
    image = _photo()
    height, width = image.shape[:2]
    # High-contrast mixture band along the body edge: alternating stripes.
    edge = width // 3
    image[:, edge : edge + 12, :3] = 0.8
    image[::2, edge : edge + 12, :3] = 0.15

    # Texel grid mapping 1:1 onto photo pixels, all first-surface.
    bins_y, bins_x = np.mgrid[0:height, 0:width]
    infront = np.ones((height, width), dtype=bool)
    first = np.ones((height, width), dtype=bool)
    window = 5
    # Strong zone: a seed inside the stripe band.
    zone = np.zeros((height, width), dtype=bool)
    zone[40:56, edge : edge + 4] = True
    # Weak evidence: any layered density at contrast; make density high
    # inside the stripe band only.
    density = np.zeros((height, width), dtype=np.float32)
    density[:, edge : edge + 12] = 0.05
    stripes = image[:, :, :3].mean(axis=2)
    from scipy.ndimage import uniform_filter

    local_mean = uniform_filter(stripes, size=window)
    local_std = np.sqrt(
        np.clip(uniform_filter(stripes**2, size=window) - local_mean**2, 0, None)
    )

    maps = film_band.compute_view_film_maps(
        image_rgba01=image,
        zone_map=zone,
        density=density,
        local_std=local_std,
        window=window,
        min_contrast=0.055,
        bins_y=bins_y,
        bins_x=bins_x,
        infront=infront,
        first_surface=first,
    )

    # The extension grows beyond the strong seed along the stripe band.
    assert maps["zone_texel"].sum() > zone.sum()
    # Deep base territory (far right) is vetoed, the film body is not.
    assert maps["veto_texel"][:, -10:].all()
    assert not maps["veto_texel"][:, :10].any()
    # Wispiness weight is high inside the dark body, low over plain base.
    assert maps["body_weight_texel"][:, :10].mean() > 0.9
    assert maps["body_weight_texel"][:, -10:].mean() < 0.05


def _view(shape, *, commit=None, added=None, veto=None, img_first=None,
          zone=None, weight=1.0, luminance=0.8):
    def full(mask):
        out = np.zeros(shape, dtype=bool)
        if mask is not None:
            out[mask] = True
        return out

    rgba = np.zeros((*shape, 4), dtype=np.float32)
    rgba[:, :, :3] = luminance   # bright mixture content by default
    rgba[:, :, 3] = 1.0
    return {
        "rgba": rgba,
        "weight": np.full(shape, weight, dtype=np.float32),
        "contested": np.zeros(shape, dtype=bool),
        "film_band": {
            "zone_texel": full(zone),
            "added_texel": full(added),
            "commit_texel": full(commit),
            "veto_texel": full(veto),
            "img_first_texel": full(img_first),
            "contested_texel": np.zeros(shape, dtype=bool),
            "body_weight_texel": np.ones(shape, dtype=np.float32),
        },
    }


def _flat_positions(shape) -> np.ndarray:
    positions = np.zeros((*shape, 4), dtype=np.float32)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    positions[:, :, 0] = xs / max(shape[1] - 1, 1)
    positions[:, :, 1] = ys / max(shape[0] - 1, 1)
    positions[:, :, 3] = 1.0
    return positions


def test_commit_film_band_requires_flag_consensus_and_no_veto() -> None:
    shape = (16, 16)
    surface = np.ones(shape, dtype=bool)
    committed = (slice(0, 4), slice(0, 16))     # both views flag rows 0-3
    floater = (slice(8, 12), slice(0, 16))      # only view A flags rows 8-11
    vetoed = (slice(4, 6), slice(0, 16))        # view B vetoes rows 4-5

    # View A carries bright mixture claims at moderate weight; view B is
    # the winning witness with dark film content on the LEFT half and
    # bright base on the RIGHT half (the observed-context dominance must
    # separate them).
    view_a = _view(shape, commit=None, added=committed,
                   img_first=(slice(0, 16), slice(0, 16)),
                   weight=0.6, luminance=0.8)
    view_a["film_band"]["commit_texel"][committed] = True
    view_a["film_band"]["commit_texel"][floater] = True
    view_a["film_band"]["commit_texel"][vetoed] = True
    view_a["film_band"]["added_texel"][floater] = True
    view_a["film_band"]["added_texel"][vetoed] = True
    view_b = _view(shape, commit=committed, veto=vetoed,
                   img_first=(slice(0, 16), slice(0, 16)),
                   weight=1.0, luminance=0.15)
    view_b["rgba"][:, 8:, :3] = 0.85

    state = film_band.commit_film_band(
        [view_a, view_b], surface_mask=surface,
        positions_texture=_flat_positions(shape))

    assert state is not None
    commit = state["commit_mask"]
    assert commit[0:4].all()          # consensus + no veto -> committed
    assert not commit[8:12].any()     # flag dissent (floater) -> blocked
    assert not commit[4:6].any()      # base witness veto -> blocked
    # Bright claims are vacated on committed texels inside the dark-
    # dominated context; bright-context commits keep their claims (no
    # flake-island contrast), and the film witness keeps its claims.
    assert (view_a["weight"][0:4, 0:6] == 0.0).all()
    assert (view_a["weight"][0:4, 11:] > 0.0).all()
    assert (view_a["weight"][8:12] > 0.0).all()
    assert (view_a["weight"][4:6] > 0.0).all()
    assert (view_b["weight"] == 1.0).all()


def test_commit_film_band_noop_for_single_view() -> None:
    shape = (8, 8)
    view = _view(shape, commit=(slice(0, 4), slice(0, 8)))
    state = film_band.commit_film_band(
        [view], surface_mask=np.ones(shape, bool),
        positions_texture=_flat_positions(shape))
    assert state is None
    assert (view["weight"] == 1.0).all()


def test_retone_film_band_pulls_committed_fill_toward_dark_anchors() -> None:
    # Texel pitch must be fine relative to the voxel-ball scales (which
    # are fractions of the mesh diagonal), as in a real atlas.
    size = 96
    positions = np.zeros((size, size, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    positions[:, :, 0] = xs / (size - 1)
    positions[:, :, 1] = ys / (size - 1)
    positions[:, :, 3] = 1.0

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, 3] = 1.0
    observed = np.zeros((size, size), dtype=bool)
    observed[:, :44] = True          # dark film body, observed
    colors[:, :44, :3] = 0.1
    observed[:, 50:] = True          # bright base, observed
    colors[:, 50:, :3] = 0.85
    colors[:, 44:50, :3] = 0.55      # membrane-filled band between them

    commit = np.zeros((size, size), dtype=bool)
    commit[:, 44:50] = True
    body_weight = np.ones((size, size), dtype=np.float32)

    out, stats = film_band.retone_film_band(
        colors.copy(),
        positions_texture=positions,
        observed_mask=observed,
        commit_mask=commit,
        body_weight=body_weight,
    )

    assert stats["applied"]
    # Dark-dominated interior of the band reads as the film...
    assert out[:, 44:46, :3].mean() < 0.3
    # ...while texels whose observed context is bright-contested keep
    # (most of) the membrane: no hard dark step against bright neighbors.
    assert out[:, 49, :3].mean() > 0.4
    assert np.allclose(out[:, 50:, :3], 0.85)  # observed base untouched
