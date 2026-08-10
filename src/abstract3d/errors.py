"""Abstract3D error types.

`error_class` contract: request-class errors carry a STABLE machine-readable
`error_class` attribute so HTTP hosts (e.g. AbstractCore's scene3d endpoint)
can map them to 4xx statuses without importing abstract3d or keying on
message prose. The values never change once shipped:

- ``invalid_request``: the caller's options/inputs were wrong (HTTP 422).
- ``capability_not_supported``: the backend cannot do what was asked (422).
- ``license_acknowledgment_required``: policy gate, explicit opt-in missing (403).
"""

from __future__ import annotations


class Abstract3DError(RuntimeError):
    """Base error for Abstract3D."""


class DependencyUnavailableError(Abstract3DError):
    """Raised when an optional runtime dependency is missing."""


class SourceBootstrapError(Abstract3DError):
    """Raised when the pinned upstream runtime source cannot be prepared."""


class InvalidRequestError(Abstract3DError):
    """A request carried options the selected backend does not support.

    Raised instead of silently ignoring unknown keyword options: a caller
    tuning `guidance_scale` on a feed-forward backend (or mistyping
    `texure_mode`) must learn immediately that the option changed nothing.
    """

    error_class = "invalid_request"


class CapabilityNotSupportedError(Abstract3DError):
    """Raised when a backend does not support the requested task or option."""

    error_class = "capability_not_supported"


class LicenseAcknowledgmentRequiredError(CapabilityNotSupportedError):
    """Raised when a license-gated backend runs without the explicit opt-in.

    Carries a STABLE machine-readable marker (`error_class`) so HTTP hosts can
    map the refusal to a policy-gate status (403) without keying on message
    prose. Contract: `error_class` never changes, and the message always
    contains the phrase "license acknowledgment" (hosts may use it as a
    fallback; wording around it may evolve, the phrase may not).
    """

    error_class = "license_acknowledgment_required"


class BackendNotConfiguredError(Abstract3DError):
    """Raised when no backend is configured for a manager call."""
