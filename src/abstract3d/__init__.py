"""Abstract3D local-first 3D generation for AbstractFramework."""

from .scene3d_manager import Scene3DManager

__all__ = [
    "Scene3DManager",
    "Hunyuan3DShapeBackend",
    "Step1XGeometryBackend",
    "Trellis2LocalBackend",
    "TripoSRBackend",
    "mesh_ops",
    "tools",
]

__version__ = "0.3.1"

_LAZY_BACKEND_EXPORTS = {
    "Hunyuan3DShapeBackend",
    "Step1XGeometryBackend",
    "Trellis2LocalBackend",
    "TripoSRBackend",
}

# Submodules exposed lazily so `import abstract3d` stays dependency-light.
_LAZY_SUBMODULES = {"mesh_ops", "tools"}


def __getattr__(name: str):
    if name in _LAZY_BACKEND_EXPORTS:
        from . import backends as _backends

        value = getattr(_backends, name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        import importlib

        value = importlib.import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
