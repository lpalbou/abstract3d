"""Robust subject segmentation for texture and conditioning inputs.

Background removal quality bounds everything downstream of it: the alpha
mask drives canonical recentering, silhouette registration, projection
weighting, and the multi-view geometry conditioning. rembg's default
``u2net`` checkpoint is a general salient-object model that routinely
amputates large low-contrast regions (the classic failure is a dark hair
mass against a light studio background — empirically it dropped 40% of the
subject on the checked profile photo). The ``isnet-general-use`` checkpoint
segments the same inputs correctly while remaining general-purpose, so it
is preferred, with graceful fallback to the default model when the ONNX
checkpoint cannot be fetched.

The raw matte also needs hygiene before geometric use: stray floater blobs
shift bounding boxes (breaking canonical framing), and pinholes create
false silhouette edges. `clean_alpha_mask` keeps the dominant connected
component and closes small holes without touching the soft matte edge.
"""

from __future__ import annotations

from typing import Any, Optional

_SESSION_CACHE: dict = {}
_PREFERRED_MODELS = ("isnet-general-use",)


def _session(model_name: str) -> Optional[Any]:
    if model_name in _SESSION_CACHE:
        return _SESSION_CACHE[model_name]
    try:
        import rembg

        session = rembg.new_session(model_name)
    except Exception:
        session = None
    _SESSION_CACHE[model_name] = session
    return session


def clean_alpha_mask(image: Any, *, min_component_ratio: float = 0.02) -> Any:
    """Keep the dominant alpha component(s) and close pinholes.

    Components smaller than `min_component_ratio` of the largest one are
    floaters (matting noise, background remnants); they shift the subject
    bounding box that canonical framing depends on. Holes inside the
    subject create false silhouette edges for registration. Both are
    removed on the BINARY support; the soft matte values are preserved
    where the support survives.
    """
    import numpy as np

    rgba = image.convert("RGBA") if hasattr(image, "convert") else image
    array = np.asarray(rgba, dtype=np.uint8).copy()
    alpha = array[:, :, 3]
    support = alpha > 12
    if not support.any():
        return rgba
    try:
        from scipy import ndimage

        labels, count = ndimage.label(support)
        if count > 1:
            sizes = ndimage.sum(support, labels, index=np.arange(1, count + 1))
            keep = np.zeros(count + 1, dtype=bool)
            keep[1:] = sizes >= float(sizes.max()) * float(min_component_ratio)
            support = keep[labels]
        support = ndimage.binary_closing(support, structure=np.ones((5, 5), dtype=bool))
        support = ndimage.binary_fill_holes(support)
    except Exception:
        return rgba
    # Keep original matte values on surviving support; holes that were
    # closed had zero alpha, so they become opaque interior.
    array[:, :, 3] = np.where(support, np.where(alpha > 12, alpha, 255), 0)
    from PIL import Image

    return Image.fromarray(array, mode="RGBA")


def remove_background_robust(image: Any) -> Any:
    """Segment the subject with the strongest available matte model.

    Tries the preferred general-purpose checkpoint first and falls back to
    rembg's default when unavailable (e.g. offline hosts that never cached
    the ONNX file). The returned image always has a cleaned alpha channel.
    """
    import rembg

    source = image.convert("RGBA") if hasattr(image, "convert") else image
    result = None
    for model_name in _PREFERRED_MODELS:
        session = _session(model_name)
        if session is None:
            continue
        try:
            result = rembg.remove(source, session=session)
            break
        except Exception:
            continue
    if result is None:
        result = rembg.remove(source)
    return clean_alpha_mask(result.convert("RGBA"))
