from __future__ import annotations

import pytest

from abstract3d.errors import BackendNotConfiguredError
from abstract3d.scene3d_manager import Scene3DManager


class _FakeBackend:
    backend_id = "fake-scene3d"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def available_providers(self, *, task=None):
        self.calls.append(("providers", task))
        return [{"provider_id": "fake", "task": task}]

    def list_models(self, *, task=None, provider=None):
        self.calls.append(("models", {"task": task, "provider": provider}))
        return [{"model_id": "mesh", "task": task, "provider": provider}]

    def list_provider_models(self, *, task=None, provider=None):
        self.calls.append(("provider_models", {"task": task, "provider": provider}))
        return [{"model_id": "mesh", "task": task, "provider": provider}]

    def list_operations(self, *, task=None):
        self.calls.append(("operations", task))
        return [{"operation_id": task or "text_to_scene3d"}]

    def load_resident_model(self, request):
        self.calls.append(("load", dict(request)))
        return {"state": "loaded"}

    def list_loaded_models(self, filters=None):
        self.calls.append(("loaded", dict(filters or {})))
        return [{"model": "resident"}]

    def list_resident_models(self, filters=None):
        self.calls.append(("resident", dict(filters or {})))
        return [{"model": "resident"}]

    def unload_resident_model(self, request):
        self.calls.append(("unload", dict(request)))
        return {"state": "unloaded"}

    def t23d(self, prompt: str, **kwargs):
        self.calls.append(("t23d", {"prompt": prompt, "kwargs": dict(kwargs)}))
        return {"mode": "t23d", "prompt": prompt}

    def i23d(self, image, **kwargs):
        self.calls.append(("i23d", {"image": image, "kwargs": dict(kwargs)}))
        return {"mode": "i23d", "image": image}

    def generate(self, prompt: str = "", *, task=None, **kwargs):
        self.calls.append(("generate", {"prompt": prompt, "task": task, "kwargs": dict(kwargs)}))
        return {"mode": "generate", "task": task}

    def validate_suite(self, **kwargs):
        self.calls.append(("validate", dict(kwargs)))
        return {"rows": []}


def test_scene3d_manager_delegates_complete_backend_surface() -> None:
    backend = _FakeBackend()
    manager = Scene3DManager(backend=backend)

    assert manager.available_providers(task="t23d")[0]["provider_id"] == "fake"
    assert manager.list_models(task="text_to_scene3d", provider="fake")[0]["model_id"] == "mesh"
    assert manager.list_provider_models(task="image_to_scene3d", provider="fake")[0]["provider"] == "fake"
    assert manager.list_operations(task="image_to_scene3d")[0]["operation_id"] == "image_to_scene3d"
    assert manager.capability_catalog(task="text_to_scene3d")["backend_id"] == "fake-scene3d"
    assert manager.load_resident_model({"task": "t23d"})["state"] == "loaded"
    assert manager.list_loaded_models({"task": "t23d"})[0]["model"] == "resident"
    assert manager.list_resident_models({"task": "i23d"})[0]["model"] == "resident"
    assert manager.unload_resident_model({"task": "t23d"})["state"] == "unloaded"
    assert manager.t23d("chair")["mode"] == "t23d"
    assert manager.i23d("image.png")["mode"] == "i23d"
    assert manager.generate("chair", task="text_to_scene3d")["mode"] == "generate"
    assert manager.validate_suite(prompts=["chair"], images=[], output_dir="/tmp")["rows"] == []


def test_scene3d_manager_raises_when_backend_missing() -> None:
    manager = Scene3DManager(backend=_FakeBackend())
    manager.backend = None

    with pytest.raises(BackendNotConfiguredError):
        manager.t23d("missing")


def test_scene3d_manager_can_select_backend_by_id(monkeypatch) -> None:
    selected: dict[str, object] = {}

    def _fake_make_backend(backend_id, owner):
        selected["backend_id"] = backend_id
        selected["owner"] = owner
        return _FakeBackend()

    monkeypatch.setattr("abstract3d.scene3d_manager.make_backend", _fake_make_backend)

    manager = Scene3DManager(backend=None, backend_id="trellis2", owner=None)

    assert isinstance(manager.backend, _FakeBackend)
    assert selected["backend_id"] == "trellis2"


def test_scene3d_manager_can_select_step1x_alias(monkeypatch) -> None:
    selected: dict[str, object] = {}

    def _fake_make_backend(backend_id, owner):
        selected["backend_id"] = backend_id
        selected["owner"] = owner
        return _FakeBackend()

    monkeypatch.setattr("abstract3d.scene3d_manager.make_backend", _fake_make_backend)

    manager = Scene3DManager(backend=None, backend_id="step1x", owner=None)

    assert isinstance(manager.backend, _FakeBackend)
    assert selected["backend_id"] == "step1x"
