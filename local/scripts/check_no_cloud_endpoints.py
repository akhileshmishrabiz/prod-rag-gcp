"""Static assertions for the Compose zero-cloud runtime boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "local/compose.yaml").read_text()
GUARD = (ROOT / "local/scripts/assert-local-env.sh").read_text()


def main() -> None:
    published = re.findall(r'"([^"]+:[0-9]+:[0-9]+)"', COMPOSE)
    if not published or any(not binding.startswith("127.0.0.1:") for binding in published):
        raise SystemExit("every published port must bind to loopback")
    required = {
        "LOCAL_METADATA_ENDPOINT",
        "LOCAL_GCS_ENDPOINT",
        "LOCAL_PUBSUB_ENDPOINT",
        "LOCAL_EMBEDDING_ENDPOINT",
        "LOCAL_VECTOR_ENDPOINT",
        "LOCAL_GENERATION_ENDPOINT",
    }
    if any(name not in COMPOSE or name not in GUARD for name in required):
        raise SystemExit("explicit endpoints and startup assertions are required")
    if 'GOOGLE_APPLICATION_CREDENTIALS: ""' not in COMPOSE:
        raise SystemExit("cloud credential fallback must be disabled")
    if COMPOSE.count("http://127.0.0.1:9") < 2:
        raise SystemExit("HTTP and HTTPS egress must use the closed loopback proxy")
    blocked_hosts = {
        "metadata.google.internal",
        "oauth2.googleapis.com",
        "aiplatform.googleapis.com",
        "firestore.googleapis.com",
        "pubsub.googleapis.com",
        "storage.googleapis.com",
    }
    if any(f'"{hostname}=127.0.0.1"' not in COMPOSE for hostname in blocked_hosts):
        raise SystemExit("known Google metadata, auth, and data endpoints must be sinkholed")
    if "googleapis.com" not in GUARD:
        raise SystemExit("startup guard must reject public Google endpoints")
    print("zero-cloud assertions passed: endpoint sinkholes, loopback ports, explicit emulators")


if __name__ == "__main__":
    main()
