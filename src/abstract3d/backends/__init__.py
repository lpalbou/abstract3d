"""Backend implementations and selectors for Abstract3D."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple


_BACKEND_EXPORTS: Dict[str, Tuple[str, str]] = {
    "TripoSRBackend": ("abstract3d.backends.triposr_runtime", "TripoSRBackend"),
    "Step1XGeometryBackend": ("abstract3d.backends.step1x_runtime", "Step1XGeometryBackend"),
    "Trellis2LocalBackend": ("abstract3d.backends.trellis2_runtime", "Trellis2LocalBackend"),
    "Hunyuan3DShapeBackend": ("abstract3d.backends.hunyuan3d_runtime", "Hunyuan3DShapeBackend"),
}

_BACKEND_CLASS_BY_ID: Dict[str, Tuple[str, str]] = {
    "abstract3d:triposr": _BACKEND_EXPORTS["TripoSRBackend"],
    "triposr": _BACKEND_EXPORTS["TripoSRBackend"],
    "abstract3d:step1x-local": _BACKEND_EXPORTS["Step1XGeometryBackend"],
    "step1x": _BACKEND_EXPORTS["Step1XGeometryBackend"],
    "step1x-local": _BACKEND_EXPORTS["Step1XGeometryBackend"],
    "abstract3d:trellis2-local": _BACKEND_EXPORTS["Trellis2LocalBackend"],
    "trellis2": _BACKEND_EXPORTS["Trellis2LocalBackend"],
    "abstract3d:hunyuan3d21-local": _BACKEND_EXPORTS["Hunyuan3DShapeBackend"],
    "hunyuan3d21": _BACKEND_EXPORTS["Hunyuan3DShapeBackend"],
    "hunyuan3d": _BACKEND_EXPORTS["Hunyuan3DShapeBackend"],
    "hunyuan": _BACKEND_EXPORTS["Hunyuan3DShapeBackend"],
}


def _load_backend_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _backend_class_for_id(backend_id: str):
    module_name, class_name = _BACKEND_CLASS_BY_ID[backend_id]
    return _load_backend_class(module_name, class_name)


DEFAULT_BACKEND_ID = "abstract3d:triposr"
BACKEND_FACTORIES: Dict[str, Callable[[Any], Any]] = {
    backend_id: (lambda owner, backend_id=backend_id: _backend_class_for_id(backend_id)(owner))
    for backend_id in _BACKEND_CLASS_BY_ID
}


def make_backend(backend_id: str | None, owner: Any):
    selected = str(backend_id or DEFAULT_BACKEND_ID).strip().lower()
    factory = BACKEND_FACTORIES.get(selected)
    if factory is None:
        raise KeyError(f"Unknown scene3d backend id: {backend_id!r}")
    return factory(owner)


def __getattr__(name: str):
    spec = _BACKEND_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _load_backend_class(*spec)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_BACKEND_ID",
    "BACKEND_FACTORIES",
    "Hunyuan3DShapeBackend",
    "Step1XGeometryBackend",
    "TripoSRBackend",
    "Trellis2LocalBackend",
    "make_backend",
]
