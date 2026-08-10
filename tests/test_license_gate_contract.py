"""Pins the license-refusal contract HTTP hosts depend on.

AbstractCore's scene3d endpoint maps license refusals to 403 using two
signals, in order:
1. the STABLE `error_class` attribute ("license_acknowledgment_required"),
2. the message phrase "license acknowledgment" as a fallback.

Both are contracts: `error_class` never changes; the phrase may not be
reworded (text around it may evolve). This test exists so a message edit
cannot silently turn a policy-gate 403 into a generic 500 host-side.
"""

from __future__ import annotations

import pytest

from abstract3d.errors import CapabilityNotSupportedError, LicenseAcknowledgmentRequiredError


def test_license_error_class_is_stable() -> None:
    assert LicenseAcknowledgmentRequiredError.error_class == "license_acknowledgment_required"
    # Subclass relationship keeps existing except-clauses working.
    assert issubclass(LicenseAcknowledgmentRequiredError, CapabilityNotSupportedError)


def test_hunyuan_refusal_carries_both_signals(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", raising=False)
    from abstract3d.backends.hunyuan3d_runtime import _require_license_acceptance

    with pytest.raises(LicenseAcknowledgmentRequiredError) as excinfo:
        _require_license_acceptance(owner=None)

    err = excinfo.value
    assert getattr(err, "error_class", None) == "license_acknowledgment_required"
    message = str(err).lower()
    assert "license acknowledgment" in message
    assert "abstract3d_hunyuan_accept_license" in message
