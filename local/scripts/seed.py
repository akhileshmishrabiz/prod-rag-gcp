"""Idempotently seed one non-sensitive object in the local GCS-compatible service."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = (ROOT / "local/emulator/fixtures/handbook.txt").read_bytes()
CHECKSUM = hashlib.sha256(CONTENT).hexdigest()
STORAGE = "http://127.0.0.1:4443"
BUCKET = "rag-local"
OBJECT = "tenants/dev-tenant/uploads/handbook.txt"


def main() -> None:
    bucket_request = urllib.request.Request(  # noqa: S310 - fixed loopback URL
        f"{STORAGE}/storage/v1/b?project=local-rag-project",
        data=json.dumps({"name": BUCKET}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(  # nosec B310  # noqa: S310 - fixed loopback URL
            bucket_request, timeout=5
        ).close()
    except urllib.error.HTTPError as error:
        if error.code != 409:
            raise

    boundary = "rag-local-seed-boundary"
    metadata = json.dumps(
        {
            "name": OBJECT,
            "contentType": "text/markdown",
            "metadata": {"sha256": CHECKSUM},
        },
        separators=(",", ":"),
    ).encode()
    multipart = (
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
        + metadata
        + f"\r\n--{boundary}\r\nContent-Type: text/markdown\r\n\r\n".encode()
        + CONTENT
        + f"\r\n--{boundary}--\r\n".encode()
    )
    query = urllib.parse.urlencode({"uploadType": "multipart"})
    upload = urllib.request.Request(  # noqa: S310 - fixed loopback URL
        f"{STORAGE}/upload/storage/v1/b/{BUCKET}/o?{query}",
        data=multipart,
        method="POST",
        headers={
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(  # nosec B310  # noqa: S310 - fixed loopback URL
        upload, timeout=5
    ) as response:
        if response.status not in {200, 201}:
            raise RuntimeError(f"storage seed failed with {response.status}")
        result = json.loads(response.read())
    print(
        json.dumps(
            {
                "bucket": BUCKET,
                "checksum_sha256": CHECKSUM,
                "object": OBJECT,
                "object_generation": int(result["generation"]),
                "status": "seeded",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
