from __future__ import annotations

from abstract3d.backends.hunyuan3d_runtime import Hunyuan3DShapeBackend
from abstract3d.backends.step1x_runtime import Step1XGeometryBackend
from abstract3d.backends.trellis2_runtime import Trellis2LocalBackend
from abstract3d.backends.triposr_runtime import TripoSRBackend
from abstract3d.integrations.abstractcore_plugin import register


def test_plugin_registers_scene3d_backends_with_expected_contract() -> None:
    recorded: list[dict[str, object]] = []

    class _Registry:
        def register_scene3d_backend(self, **kwargs):
            recorded.append(dict(kwargs))

    register(_Registry())

    assert [item["backend_id"] for item in recorded] == [
        "abstract3d:triposr",
        "abstract3d:step1x-local",
        "abstract3d:hunyuan3d21-local",
        "abstract3d:trellis2-local",
    ]
    assert recorded[0]["priority"] == 10
    assert 'abstract3d[triposr]' in recorded[0]["install_hint"]
    assert 'pip install abstract3d' in recorded[0]["install_hint"]
    assert 'abstract3d[apple]' in recorded[0]["install_hint"]
    assert recorded[1]["priority"] == 7
    assert 'abstract3d[step1x]' in recorded[1]["install_hint"]
    assert 'abstract3d[gpu]' in recorded[1]["install_hint"]
    # Hunyuan registers BELOW TripoSR (validated default keeps priority) and
    # its hints must carry the license gate loudly.
    assert recorded[2]["priority"] == 8
    assert 'abstract3d[hunyuan3d]' in recorded[2]["install_hint"]
    assert "License gate" in recorded[2]["config_hint"]
    assert "ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE" in recorded[2]["config_hint"]
    assert "License-gated" in recorded[2]["description"]
    assert recorded[3]["priority"] == 5
    assert 'abstract3d[trellis2]' in recorded[3]["install_hint"]
    assert 'AbstractVision composition contract' in recorded[3]["install_hint"]
    triposr_backend = recorded[0]["factory"](owner=None)
    step1x_backend = recorded[1]["factory"](owner=None)
    hunyuan_backend = recorded[2]["factory"](owner=None)
    trellis_backend = recorded[3]["factory"](owner=None)
    assert isinstance(triposr_backend, TripoSRBackend)
    assert isinstance(step1x_backend, Step1XGeometryBackend)
    assert isinstance(hunyuan_backend, Hunyuan3DShapeBackend)
    assert isinstance(trellis_backend, Trellis2LocalBackend)


def test_default_backend_selection_prefers_validated_triposr() -> None:
    # Priority order must keep the validated default selected even with the
    # license-gated strongest-geometry backend registered.
    registrations: dict[str, int] = {}

    class _Registry:
        def register_scene3d_backend(self, *, backend_id, priority, **kwargs):
            registrations[backend_id] = priority

    register(_Registry())
    best = max(registrations.items(), key=lambda kv: kv[1])
    assert best[0] == "abstract3d:triposr"
