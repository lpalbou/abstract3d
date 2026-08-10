"""Provider-neutral composed image-generation settings for `t23d`."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple

from .errors import DependencyUnavailableError

DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 768
COMPOSITION_INSTALL_HINT = (
    'Composed text-to-scene3d uses AbstractVision. Install "abstract3d" for the '
    'lightweight base contract, or use "abstract3d[apple]" / "abstract3d[gpu]" '
    "for local image-composition profiles."
)


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _config_text(owner: Any, key: str) -> Optional[str]:
    try:
        config = getattr(owner, "config", None)
        if isinstance(config, dict):
            return _clean_text(config.get(key))
    except Exception:
        return None
    return None


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        text = _clean_text(value)
        if text is not None:
            return text
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _coerce_int(value)
        if parsed is not None:
            return parsed
    return None


def parse_lora_adapters(value: Any) -> Optional[list]:
    """Parse a LoRA-adapter declaration into `[{"source", "scale"}, ...]`.

    Accepted forms (config `scene3d_image_lora_adapters` / env
    `ABSTRACT3D_IMAGE_LORA_ADAPTERS`, or an already-structured value):
      * a JSON list: `[{"source": "/path/a.safetensors", "scale": 1.0}]`
      * plain text: `;`-separated `path` or `path@scale` entries
        (`~` expands; `@` never occurs in adapter paths, unlike `:`).
      * a Python list of dicts/paths (programmatic callers).

    The result is the abstractvision `lora_adapters` request shape
    (`source` required, `scale` defaulting to 1.0); validation of the
    files themselves stays with the provider. Malformed declarations
    raise: a LoRA the operator asked for must never be dropped silently
    (the 0-of-1680-keys class is invisible enough already).
    """

    if value is None:
        return None
    entries: Any
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            import json

            entries = json.loads(text)
        else:
            entries = [part.strip() for part in text.split(";") if part.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        entries = [value]
    adapters = []
    for entry in entries:
        if isinstance(entry, dict):
            source = str(entry.get("source") or "").strip()
            scale = entry.get("scale", 1.0)
        else:
            text = str(entry).strip()
            source, _, scale_text = text.partition("@")
            scale = scale_text or 1.0
        if not source:
            raise ValueError(
                f"lora_adapters entry {entry!r} has no 'source' path")
        adapters.append({
            "source": str(Path(source).expanduser()),
            "scale": float(scale),
        })
    return adapters or None


def resolve_image_generation_request(
    owner: Any,
    *,
    provider: Any = None,
    model: Any = None,
    width: Any = None,
    height: Any = None,
    seed: Any = None,
) -> Dict[str, Any]:
    """Resolve generic text-to-image kwargs for composed `t23d`.

    The contract intentionally stays provider-neutral. `abstract3d` only owns
    generic selection and sizing here; provider-specific controls remain in the
    configured vision backend.
    """

    resolved: Dict[str, Any] = {
        "width": _first_int(
            width,
            _config_text(owner, "scene3d_image_width"),
            os.environ.get("ABSTRACT3D_IMAGE_WIDTH"),
            DEFAULT_IMAGE_WIDTH,
        )
        or DEFAULT_IMAGE_WIDTH,
        "height": _first_int(
            height,
            _config_text(owner, "scene3d_image_height"),
            os.environ.get("ABSTRACT3D_IMAGE_HEIGHT"),
            DEFAULT_IMAGE_HEIGHT,
        )
        or DEFAULT_IMAGE_HEIGHT,
    }
    resolved_provider = _first_text(
        provider,
        _config_text(owner, "scene3d_image_provider"),
        os.environ.get("ABSTRACT3D_IMAGE_PROVIDER"),
    )
    if resolved_provider is not None:
        resolved["provider"] = resolved_provider
    resolved_model = _first_text(
        model,
        _config_text(owner, "scene3d_image_model"),
        os.environ.get("ABSTRACT3D_IMAGE_MODEL"),
    )
    if resolved_model is not None:
        resolved["model"] = resolved_model
    resolved_seed = _first_int(
        seed,
        _config_text(owner, "scene3d_image_seed"),
        os.environ.get("ABSTRACT3D_IMAGE_SEED"),
    )
    if resolved_seed is not None:
        resolved["seed"] = resolved_seed
    # LoRA adapters (viewgen bench 2026-07-21 section 6: the winning view
    # recipe is klein + the consistency LoRA at scale 1.0). One knob for
    # both synthesis stages, like provider/model above; the value forwards
    # verbatim to the abstractvision request (`lora_adapters`), which every
    # local mlx route consumes. The config value may be structured (a list)
    # — read it raw, never through the text coercion. Parse errors raise:
    # an operator-declared adapter must never be silently dropped.
    raw_lora: Any = None
    try:
        config = getattr(owner, "config", None)
        if isinstance(config, dict):
            raw_lora = config.get("scene3d_image_lora_adapters")
    except Exception:
        raw_lora = None
    if raw_lora is None:
        raw_lora = os.environ.get("ABSTRACT3D_IMAGE_LORA_ADAPTERS")
    resolved_lora = parse_lora_adapters(raw_lora)
    if resolved_lora:
        resolved["lora_adapters"] = resolved_lora
    return resolved


# The composed-t23d option surface: backends pop exactly these keys from a
# generation call before the strict unknown-option check runs.
IMAGE_REQUEST_KEYS: Tuple[str, ...] = (
    "image_provider", "image_model", "image_width", "image_height", "image_seed",
)


def pop_image_generation_request(owner: Any, kwargs: MutableMapping[str, Any]) -> Dict[str, Any]:
    return resolve_image_generation_request(
        owner,
        provider=kwargs.pop("image_provider", None),
        model=kwargs.pop("image_model", None),
        width=kwargs.pop("image_width", None),
        height=kwargs.pop("image_height", None),
        seed=kwargs.pop("image_seed", None),
    )


def pop_composition_kwargs(kwargs: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Extract the composed-image options from a generation call's kwargs.

    Backends forward ONLY these to `_make_source_image`, so the composition
    keys are consumed from the caller's dict (previously `**kwargs` passed a
    copy and the keys survived, which would trip the strict option check).
    """

    return {key: kwargs.pop(key) for key in IMAGE_REQUEST_KEYS if key in kwargs}


def _owner_vision(owner: Any) -> Optional[Any]:
    if owner is None:
        return None
    try:
        return getattr(owner, "vision", None)
    except Exception:
        return None


def _owner_or_shim(owner: Any) -> Any:
    if owner is not None:
        return owner

    class _Owner:
        def __init__(self) -> None:
            self.config = {}

    return _Owner()


def _abstractvision_capability(owner: Any) -> Any:
    try:
        plugin = importlib.import_module("abstractvision.integrations.abstractcore_plugin")
    except Exception as exc:
        raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT) from exc

    register = getattr(plugin, "register", None)
    if not callable(register):
        raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT)

    backends: list[dict[str, Any]] = []

    class _Registry:
        def register_vision_backend(self, **kwargs: Any) -> None:
            backends.append(dict(kwargs))

    register(_Registry())
    by_id = {str(item.get("backend_id") or "").strip(): item for item in backends}
    selected = by_id.get("abstractvision:openai") or by_id.get("abstractvision:openai-compatible")
    factory = selected.get("factory") if isinstance(selected, dict) else None
    if not callable(factory):
        raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT)
    return factory(_owner_or_shim(owner))


def default_image_generator(owner: Any) -> Callable[..., Any]:
    vision = _owner_vision(owner)
    if vision is not None and callable(getattr(vision, "t2i", None)):
        return lambda prompt, **kwargs: vision.t2i(prompt, **kwargs)

    capability = _abstractvision_capability(owner)
    if not callable(getattr(capability, "t2i", None)):
        raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT)
    return lambda prompt, **kwargs: capability.t2i(prompt, **kwargs)


def has_image_composer(owner: Any) -> bool:
    vision = _owner_vision(owner)
    if vision is not None and callable(getattr(vision, "t2i", None)):
        return True
    return importlib.util.find_spec("abstractvision.integrations.abstractcore_plugin") is not None


def describe_image_binding(value: Any) -> str:
    text = _clean_text(value)
    if text is not None:
        return text
    return "configured AbstractVision default"
