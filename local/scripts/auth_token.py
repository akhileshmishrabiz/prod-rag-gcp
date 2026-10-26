"""Create the fixed-tenant local signed identity used by Compose."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

SIGNING_KEY = b"0123456789abcdef0123456789abcdef\n"


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def issue_token(audience: str = "rag-api-local") -> str:
    """Issue a short-lived identity; tenant is signed and never caller-selected."""
    del audience  # Retained for a stable local script call surface.
    payload = json.dumps(
        {
            "expires_at": int(time.time()) + 300,
            "identity": "local-rag-tester",
            "tenant_id": "dev-tenant",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signature = hmac.new(SIGNING_KEY, payload, hashlib.sha256).digest()
    return f"{_encode(payload)}.{_encode(signature)}"


if __name__ == "__main__":
    print(issue_token())
