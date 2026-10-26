"""Cloud Run IAM authentication boundary for Pub/Sub push."""

from __future__ import annotations

from fastapi import Request

from packages.rag_core.errors import ErrorCode, RagError


def require_push_identity(request: Request) -> str:
    """Require a configured service; push auth is enforced by Cloud Run IAM."""
    settings = request.app.state.settings
    dependencies = request.app.state.dependencies
    if settings is None or dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    return "cloud-run-iam-boundary"
