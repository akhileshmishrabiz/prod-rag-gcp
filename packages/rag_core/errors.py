"""Stable, sanitized domain errors."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import ValidationError


class ErrorCode(StrEnum):
    """Frozen public error codes."""

    INVALID_REQUEST = "INVALID_REQUEST"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    ACCESS_DENIED = "ACCESS_DENIED"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_CONFLICT = "DOCUMENT_CONFLICT"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    UNPROCESSABLE_DOCUMENT = "UNPROCESSABLE_DOCUMENT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    STALE_GENERATION = "STALE_GENERATION"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    LEASE_CONTENDED = "LEASE_CONTENDED"
    CONTRACT_VERSION_UNSUPPORTED = "CONTRACT_VERSION_UNSUPPORTED"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    DEPENDENCY_TIMEOUT = "DEPENDENCY_TIMEOUT"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    DEPENDENCY_REJECTED = "DEPENDENCY_REJECTED"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    INTERNAL = "INTERNAL"


_RETRYABLE = {
    ErrorCode.RATE_LIMITED,
    ErrorCode.LEASE_CONTENDED,
    ErrorCode.DEPENDENCY_TIMEOUT,
    ErrorCode.DEPENDENCY_UNAVAILABLE,
}


class RagError(RuntimeError):
    """Error safe to route across an application boundary."""

    def __init__(self, code: ErrorCode, *, detail: str | None = None) -> None:
        self.code = code
        self.retryable = code in _RETRYABLE
        self.detail = detail or code.value.replace("_", " ").title()
        super().__init__(self.detail)


class RetryableDependencyError(RagError):
    """A bounded dependency operation can be retried."""

    def __init__(self, code: ErrorCode = ErrorCode.DEPENDENCY_UNAVAILABLE) -> None:
        if code not in _RETRYABLE:
            raise ValueError("retryable dependency error requires a retryable code")
        super().__init__(code)


class TerminalDependencyError(RagError):
    """A dependency rejected a valid bounded request."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.DEPENDENCY_REJECTED)


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration fails closed."""

    code = "CONFIGURATION_INVALID"

    def __init__(self, error_count: int) -> None:
        self.error_count = error_count
        super().__init__("Runtime configuration is invalid")

    @classmethod
    def from_validation_error(cls, error: ValidationError) -> ConfigurationError:
        """Discard values and locations that could contain sensitive material."""
        return cls(error.error_count())

    def as_problem(self) -> dict[str, Any]:
        """Return the frozen, sanitized problem-details shape."""
        return {
            "type": "https://example.invalid/problems/configuration-invalid",
            "title": "Service configuration is invalid",
            "status": 503,
            "code": self.code,
            "detail": "Readiness checks failed; inspect sanitized service logs.",
            "trace_id": "readiness",
            "request_id": "readiness",
            "retryable": False,
        }
