"""Private push authentication boundary for local and Cloud Run modes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime

from fastapi import Request

from packages.rag_core.config import AppMode, AuthMode, RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError
from services.worker.app.dependencies import WorkerDependencies

LOCAL_PUSH_IDENTITY = "local-pubsub-push"


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_local_push_token(
    *,
    key: bytes,
    tenant_id: str,
    audience: str,
    expires_at: datetime,
) -> str:
    """Issue the deliberately small, local-only push identity token."""
    payload = json.dumps(
        {
            "audience": audience,
            "expires_at": int(expires_at.timestamp()),
            "identity": LOCAL_PUSH_IDENTITY,
            "tenant_id": tenant_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signature = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{_encode(payload)}.{_encode(signature)}"


def require_push_identity(request: Request) -> str:
    """Validate explicit local push auth or rely only on Cloud Run IAM."""
    settings: RuntimeSettings | None = request.app.state.settings
    dependencies: WorkerDependencies | None = request.app.state.dependencies
    if settings is None or dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    if settings.app_mode is AppMode.GCP:
        if settings.auth_mode is not AuthMode.CLOUD_RUN_IAM:
            raise RagError(ErrorCode.CONFIGURATION_INVALID)
        return "cloud-run-iam-boundary"
    if (
        settings.app_mode is not AppMode.LOCAL
        or settings.auth_mode is not AuthMode.LOCAL_SIGNED
        or settings.local_auth_signing_key_file is None
    ):
        raise RagError(ErrorCode.CONFIGURATION_INVALID)
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        raise RagError(ErrorCode.AUTHENTICATION_REQUIRED)
    try:
        payload_part, signature_part = authorization.removeprefix("Bearer ").split(".", 1)
        payload = _decode(payload_part)
        signature = _decode(signature_part)
        key = settings.local_auth_signing_key_file.read_bytes()
        decoded = json.loads(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise RagError(ErrorCode.AUTHENTICATION_REQUIRED) from None
    expected_keys = {"audience", "expires_at", "identity", "tenant_id"}
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    if (
        not key
        or not hmac.compare_digest(signature, expected)
        or not isinstance(decoded, dict)
        or set(decoded) != expected_keys
        or decoded.get("identity") != LOCAL_PUSH_IDENTITY
        or decoded.get("tenant_id") != settings.dev_tenant_id
        or decoded.get("audience") != str(settings.pubsub_worker_audience)
        or not isinstance(decoded.get("expires_at"), int)
        or decoded["expires_at"] <= int(dependencies.clock.now().timestamp())
    ):
        raise RagError(ErrorCode.AUTHENTICATION_REQUIRED)
    return LOCAL_PUSH_IDENTITY
