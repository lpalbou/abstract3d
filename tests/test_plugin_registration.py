from __future__ import annotations

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
        "abstract3d:trellis2-local",
    ]
    assert recorded[0]["priority"] == 10
    assert 'abstract3d[triposr]' in recorded[0]["install_hint"]
    assert 'pip install abstract3d' in recorded[0]["install_hint"]
    assert 'abstract3d[apple]' in recorded[0]["install_hint"]
    assert recorded[1]["priority"] == 7
    assert 'abstract3d[step1x]' in recorded[1]["install_hint"]
    assert 'abstract3d[gpu]' in recorded[1]["install_hint"]
    assert recorded[2]["priority"] == 5
    assert 'abstract3d[trellis2]' in recorded[2]["install_hint"]
    assert 'AbstractVision composition contract' in recorded[2]["install_hint"]
    triposr_backend = recorded[0]["factory"](owner=None)
    step1x_backend = recorded[1]["factory"](owner=None)
    trellis_backend = recorded[2]["factory"](owner=None)
    assert isinstance(triposr_backend, TripoSRBackend)
    assert isinstance(step1x_backend, Step1XGeometryBackend)
    assert isinstance(trellis_backend, Trellis2LocalBackend)
