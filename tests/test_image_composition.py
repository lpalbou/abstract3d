from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstract3d.image_composition import (
    default_image_generator,
    describe_image_binding,
    parse_lora_adapters,
    pop_image_generation_request,
    resolve_image_generation_request,
)


def test_resolve_image_generation_request_is_provider_neutral_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_IMAGE_PROVIDER", raising=False)
    monkeypatch.delenv("ABSTRACT3D_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACT3D_IMAGE_WIDTH", raising=False)
    monkeypatch.delenv("ABSTRACT3D_IMAGE_HEIGHT", raising=False)
    monkeypatch.delenv("ABSTRACT3D_IMAGE_SEED", raising=False)

    out = resolve_image_generation_request(owner=None)

    assert out == {"width": 768, "height": 768}


def test_resolve_image_generation_request_prefers_explicit_then_owner_then_env(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACT3D_IMAGE_PROVIDER", "env-provider")
    monkeypatch.setenv("ABSTRACT3D_IMAGE_MODEL", "env-model")
    monkeypatch.setenv("ABSTRACT3D_IMAGE_WIDTH", "320")
    monkeypatch.setenv("ABSTRACT3D_IMAGE_HEIGHT", "321")
    monkeypatch.setenv("ABSTRACT3D_IMAGE_SEED", "13")
    owner = type(
        "_Owner",
        (),
        {
            "config": {
                "scene3d_image_provider": "owner-provider",
                "scene3d_image_model": "owner-model",
                "scene3d_image_width": "640",
                "scene3d_image_height": "641",
                "scene3d_image_seed": "7",
            }
        },
    )()

    out = resolve_image_generation_request(
        owner,
        provider="request-provider",
        model="request-model",
        width=960,
        height=961,
        seed=99,
    )

    assert out == {
        "provider": "request-provider",
        "model": "request-model",
        "width": 960,
        "height": 961,
        "seed": 99,
    }


def test_pop_image_generation_request_consumes_generic_keys() -> None:
    kwargs = {
        "image_provider": "mlx-gen",
        "image_model": "model-id",
        "image_width": 640,
        "image_height": 512,
        "image_seed": 42,
        "other": "keep",
    }

    out = pop_image_generation_request(None, kwargs)

    assert out == {
        "provider": "mlx-gen",
        "model": "model-id",
        "width": 640,
        "height": 512,
        "seed": 42,
    }
    assert kwargs == {"other": "keep"}


def test_parse_lora_adapters_accepts_json_paths_and_structured_lists() -> None:
    json_form = parse_lora_adapters(
        '[{"source": "/loras/consistency.safetensors", "scale": 1.0}]')
    assert json_form == [
        {"source": "/loras/consistency.safetensors", "scale": 1.0}]

    text_form = parse_lora_adapters(
        "/loras/a.safetensors; /loras/b.safetensors@0.8")
    assert text_form == [
        {"source": "/loras/a.safetensors", "scale": 1.0},
        {"source": "/loras/b.safetensors", "scale": 0.8},
    ]

    structured = parse_lora_adapters(
        [{"source": "/loras/c.safetensors"}, "/loras/d.safetensors@0.5"])
    assert structured == [
        {"source": "/loras/c.safetensors", "scale": 1.0},
        {"source": "/loras/d.safetensors", "scale": 0.5},
    ]

    assert parse_lora_adapters(None) is None
    assert parse_lora_adapters("") is None
    # A declared adapter without a source path is a loud error — an
    # operator-requested LoRA must never be dropped silently.
    with pytest.raises(ValueError, match="source"):
        parse_lora_adapters([{"scale": 1.0}])


def test_resolve_image_generation_request_carries_lora_adapters(monkeypatch) -> None:
    """The LoRA knob rides the same config/env chain as provider/model
    (viewgen bench section 6: klein + consistency LoRA is the winning view
    recipe) and is byte-absent when unset."""
    monkeypatch.delenv("ABSTRACT3D_IMAGE_LORA_ADAPTERS", raising=False)
    assert "lora_adapters" not in resolve_image_generation_request(owner=None)

    monkeypatch.setenv(
        "ABSTRACT3D_IMAGE_LORA_ADAPTERS", "/loras/env.safetensors@0.9")
    out = resolve_image_generation_request(owner=None)
    assert out["lora_adapters"] == [
        {"source": "/loras/env.safetensors", "scale": 0.9}]

    # Owner config (structured value allowed) wins over env.
    owner = SimpleNamespace(config={
        "scene3d_image_lora_adapters": [
            {"source": "/loras/owner.safetensors", "scale": 1.0}],
    })
    out = resolve_image_generation_request(owner)
    assert out["lora_adapters"] == [
        {"source": "/loras/owner.safetensors", "scale": 1.0}]


def test_default_image_generator_prefers_owner_vision() -> None:
    seen: dict[str, object] = {}

    def _t2i(prompt: str, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = dict(kwargs)
        return b"image"

    owner = SimpleNamespace(vision=SimpleNamespace(t2i=_t2i))

    generator = default_image_generator(owner)

    assert generator("rocket", width=640) == b"image"
    assert seen == {"prompt": "rocket", "kwargs": {"width": 640}}


def test_default_image_generator_uses_public_abstractvision_plugin_registration(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _Capability:
        def __init__(self, owner):
            calls["owner"] = owner

        def t2i(self, prompt: str, **kwargs):
            calls["prompt"] = prompt
            calls["kwargs"] = dict(kwargs)
            return b"generated"

    def _register(registry) -> None:
        registry.register_vision_backend(
            backend_id="abstractvision:openai",
            factory=lambda owner: _Capability(owner),
        )

    monkeypatch.setattr(
        "abstract3d.image_composition.importlib.import_module",
        lambda name: SimpleNamespace(register=_register),
    )

    generator = default_image_generator(owner=None)

    assert generator("chair", model="gpt-image-1", width=768) == b"generated"
    assert calls["prompt"] == "chair"
    assert calls["kwargs"] == {"model": "gpt-image-1", "width": 768}


def test_describe_image_binding_formats_unset_defaults() -> None:
    assert describe_image_binding(None) == "configured AbstractVision default"
    assert describe_image_binding("mlx-gen") == "mlx-gen"
