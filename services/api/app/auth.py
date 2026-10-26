"""Cloud IAM boundary declaration and isolated signed local test identity."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime

from fastapi import Request

from packages.rag_core.config import AppMode, AuthMode, RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError
from services.api.app.dependencies import ApiDependencies


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_local_token(*, key: bytes, identity: str, tenant_id: str, expires_at: datetime) -> str:
    """Create the deliberately small local-only bearer token used by tests/Compose."""
    payload = json.dumps(
        {
            "identity": identity,
            "tenant_id": tenant_id,
            "expires_at": int(expires_at.timestamp()),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signature = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{_encode(payload)}.{_encode(signature)}"


def require_identity(request: Request) -> str:
    """Require local signature, or rely exclusively on Cloud Run's IAM boundary."""
    settings: RuntimeSettings | None = request.app.state.settings
    dependencies: ApiDependencies | None = request.app.state.dependencies
    if settings is None or dependencies is None:
        raise RagError(ErrorCode.CONFIGURATION_INVALID)

    if settings.app_mode is AppMode.GCP:
        if settings.auth_mode is not AuthMode.CLOUD_RUN_IAM:
            raise RagError(ErrorCode.CONFIGURATION_INVALID)
        # Cloud Run validates IAM identity tokens before the request reaches
        # this process. No caller-controlled header is treated as an identity.
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
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    now = int(dependencies.clock.now().timestamp())
    if (
        not key
        or not hmac.compare_digest(signature, expected)
        or decoded.get("identity") != dependencies.policy.local_test_identity
        or decoded.get("tenant_id") != settings.dev_tenant_id
        or not isinstance(decoded.get("expires_at"), int)
        or decoded["expires_at"] <= now
    ):
        raise RagError(ErrorCode.AUTHENTICATION_REQUIRED)
    return dependencies.policy.local_test_identity
