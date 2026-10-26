"""Metadata-only telemetry primitives."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Protocol

_ALLOWED_FIELDS = frozenset(
    {
        "trace_id",
        "request_id",
        "resource_hash",
        "operation",
        "status",
        "error_code",
        "retryable",
        "latency_ms",
        "attempt",
        "count",
        "model_id",
        "pipeline_version",
        "index_version",
    }
)


def resource_hash(value: str, *, salt: bytes) -> str:
    """Return a stable pseudonymous identifier; never log the source value."""
    return hashlib.sha256(salt + value.encode()).hexdigest()[:24]


def safe_fields(**fields: object) -> dict[str, object]:
    """Enforce an allow-list so content cannot reach structured logs."""
    unknown = set(fields) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"unsafe telemetry fields: {', '.join(sorted(unknown))}")
    return {key: value for key, value in fields.items() if value is not None}


class TelemetrySink(Protocol):
    def emit(self, event: str, fields: Mapping[str, object]) -> None: ...


class JsonTelemetry:
    """Compact JSON logger accepting only metadata fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        payload = safe_fields(**dict(fields))
        self.logger.info(
            json.dumps({"event": event, **payload}, sort_keys=True, separators=(",", ":"))
        )
