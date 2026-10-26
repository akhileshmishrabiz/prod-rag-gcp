"""Local-only Pub/Sub push authenticator forwarding the unmodified envelope."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 1_000_000


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def token(key: bytes, tenant_id: str, audience: str) -> str:
    payload = json.dumps(
        {
            "audience": audience,
            "expires_at": int(time.time()) + 60,
            "identity": "local-pubsub-push",
            "tenant_id": tenant_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"{_encode(payload)}.{_encode(hmac.new(key, payload, hashlib.sha256).digest())}"


def canonicalize_emulator_envelope(body: bytes) -> bytes:
    """Remove duplicate snake-case fields emitted only by the Pub/Sub emulator."""
    try:
        envelope = json.loads(body)
        message = envelope["message"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise ValueError("invalid emulator envelope") from None
    if not isinstance(envelope, dict) or not isinstance(message, dict):
        raise ValueError("invalid emulator envelope")
    for emulator_name, canonical_name in (
        ("message_id", "messageId"),
        ("publish_time", "publishTime"),
    ):
        if emulator_name not in message:
            continue
        if canonical_name not in message or message[emulator_name] != message[canonical_name]:
            raise ValueError("conflicting emulator envelope fields")
        del message[emulator_name]
    return json.dumps(envelope, separators=(",", ":")).encode()


class Gateway(BaseHTTPRequestHandler):
    target: str
    key: bytes
    tenant_id: str
    audience: str

    def log_message(self, format_: str, *args: object) -> None:
        print(f"local_push_gateway {self.command} {self.path} {args[1]}", flush=True)

    def do_GET(self) -> None:
        status = 200 if self.path in {"/healthz", "/readyz"} else 404
        self.send_response(status)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        if self.path != "/v1/pubsub/push" or length < 1 or length > MAX_BODY_BYTES:
            self.send_response(400)
            self.end_headers()
            return
        try:
            body = canonicalize_emulator_envelope(self.rfile.read(length))
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        request = urllib.request.Request(  # noqa: S310 - fixed Compose target
            self.target,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token(self.key, self.tenant_id, self.audience)}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(  # nosec B310  # noqa: S310 - fixed local target
                request, timeout=30
            ) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        except (TimeoutError, urllib.error.URLError):
            status = 503
        self.send_response(status)
        self.end_headers()


def main() -> None:
    target = os.environ.get("WORKER_PUSH_ENDPOINT", "")
    if target != "http://worker:8081/v1/pubsub/push":
        raise SystemExit("push gateway target must be the local worker")
    key_path = Path(os.environ["LOCAL_AUTH_SIGNING_KEY_FILE"])
    Gateway.target = target
    Gateway.key = key_path.read_bytes()
    Gateway.tenant_id = os.environ["DEV_TENANT_ID"]
    Gateway.audience = os.environ["PUBSUB_WORKER_AUDIENCE"]
    server = ThreadingHTTPServer(
        ("0.0.0.0", 8086),  # nosec B104  # noqa: S104 - container namespace only
        Gateway,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
