"""Thin public manager wrapper for scene3d backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .backends import DEFAULT_BACKEND_ID, make_backend
from .errors import BackendNotConfiguredError


@dataclass
class Scene3DManager:
    backend: Optional[Any] = None
    backend_id: Optional[str] = None
    owner: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.backend is None:
            selected = self.backend_id
            if selected is None and self.owner is not None:
                config = getattr(self.owner, "config", None)
                if isinstance(config, Mapping):
                    selected = config.get("scene3d_backend")
            self.backend = make_backend(selected or DEFAULT_BACKEND_ID, self.owner)

    def _require_backend(self) -> Any:
        if self.backend is None:
            raise BackendNotConfiguredError("No scene3d backend configured.")
        return self.backend

    def available_providers(self, *, task: Optional[str] = None):
        return self._require_backend().available_providers(task=task)

    def list_models(self, *, task: Optional[str] = None, provider: Optional[str] = None):
        return self._require_backend().list_models(task=task, provider=provider)

    def list_provider_models(self, *, task: Optional[str] = None, provider: Optional[str] = None):
        return self._require_backend().list_provider_models(task=task, provider=provider)

    def list_operations(self, *, task: Optional[str] = None):
        return self._require_backend().list_operations(task=task)

    def capability_catalog(self, *, task: Optional[str] = None):
        backend = self._require_backend()
        return {
            "capability": "scene3d",
            "backend_id": getattr(backend, "backend_id", None),
            "task": task,
            "providers": backend.available_providers(task=task),
            "models": backend.list_models(task=task),
            "operations": backend.list_operations(task=task),
        }

    def load_resident_model(self, request: Mapping[str, Any]):
        return self._require_backend().load_resident_model(request)

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None):
        return self._require_backend().list_loaded_models(filters)

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None):
        return self._require_backend().list_resident_models(filters)

    def unload_resident_model(self, request: Mapping[str, Any]):
        return self._require_backend().unload_resident_model(request)

    def t23d(self, prompt: str, **kwargs: Any):
        return self._require_backend().t23d(prompt, **kwargs)

    def i23d(self, image: Any, **kwargs: Any):
        return self._require_backend().i23d(image, **kwargs)

    def generate(self, prompt: str = "", *, task: Optional[str] = None, **kwargs: Any):
        return self._require_backend().generate(prompt, task=task, **kwargs)

    def validate_suite(self, **kwargs: Any):
        return self._require_backend().validate_suite(**kwargs)
