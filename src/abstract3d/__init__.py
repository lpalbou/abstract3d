"""Abstract3D local-first 3D generation for AbstractFramework."""

from .scene3d_manager import Scene3DManager

__all__ = [
    "Scene3DManager",
    "Hunyuan3DShapeBackend",
    "Step1XGeometryBackend",
    "Trellis2LocalBackend",
    "TripoSRBackend",
]

__version__ = "0.3.0"

_LAZY_BACKEND_EXPORTS = {
    "Hunyuan3DShapeBackend",
    "Step1XGeometryBackend",
    "Trellis2LocalBackend",
    "TripoSRBackend",
}


def __getattr__(name: str):
    if name in _LAZY_BACKEND_EXPORTS:
        from . import backends as _backends

        value = getattr(_backends, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
