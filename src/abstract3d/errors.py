"""Abstract3D error types."""

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


class CapabilityNotSupportedError(Abstract3DError):
    """Raised when a backend does not support the requested task or option."""


class BackendNotConfiguredError(Abstract3DError):
    """Raised when no backend is configured for a manager call."""
