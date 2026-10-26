from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "local/compose.yaml").read_text()
API_DOCKERFILE = (ROOT / "local/docker/Dockerfile.api").read_text()
WORKER_DOCKERFILE = (ROOT / "local/docker/Dockerfile.worker").read_text()
EMULATOR_DOCKERFILE = (ROOT / "local/emulator/Dockerfile").read_text()
STARTUP_GUARD = (ROOT / "local/scripts/assert-local-env.sh").read_text()


def test_runtime_is_loopback_only_without_cloud_credentials() -> None:
    published = re.findall(r'"([^"]+:[0-9]+:[0-9]+)"', COMPOSE)
    assert published
    assert all(item.startswith("127.0.0.1:") for item in published)
    assert 'GOOGLE_APPLICATION_CREDENTIALS: ""' in COMPOSE
    assert "HTTP_PROXY: http://127.0.0.1:9" in COMPOSE
    assert "HTTPS_PROXY: http://127.0.0.1:9" in COMPOSE
    assert "credentials.json" not in COMPOSE
    assert "/.config/gcloud" not in COMPOSE
    for host in (
        "metadata.google.internal",
        "oauth2.googleapis.com",
        "aiplatform.googleapis.com",
        "firestore.googleapis.com",
        "pubsub.googleapis.com",
        "storage.googleapis.com",
    ):
        assert f'"{host}=127.0.0.1"' in COMPOSE


def test_required_services_and_exactly_one_model_emulator() -> None:
    for service in (
        "api",
        "worker",
        "firestore",
        "pubsub",
        "pubsub-bootstrap",
        "storage",
        "vertex",
        "push-gateway",
    ):
        assert re.search(rf"^  {service}:$", COMPOSE, re.MULTILINE)
    assert len(re.findall(r"^  vertex:$", COMPOSE, re.MULTILINE)) == 1
    assert "LOCAL_EMBEDDING_ENDPOINT: http://vertex:8090" in COMPOSE
    assert "LOCAL_VECTOR_ENDPOINT: http://vertex:8090" in COMPOSE
    assert "LOCAL_GENERATION_ENDPOINT: http://vertex:8090" in COMPOSE


def test_owned_images_are_digest_pinned_multistage_and_non_root() -> None:
    for dockerfile in (API_DOCKERFILE, WORKER_DOCKERFILE, EMULATOR_DOCKERFILE):
        assert dockerfile.count("FROM ") >= 2
        assert "@sha256:" in dockerfile
        assert "USER 10001:10001" in dockerfile
    assert "@sha256:" in COMPOSE


def test_startup_guard_rejects_cloud_fallback() -> None:
    for variable in (
        "LOCAL_METADATA_ENDPOINT",
        "LOCAL_GCS_ENDPOINT",
        "LOCAL_PUBSUB_ENDPOINT",
        "LOCAL_EMBEDDING_ENDPOINT",
        "LOCAL_VECTOR_ENDPOINT",
        "LOCAL_GENERATION_ENDPOINT",
    ):
        assert variable in STARTUP_GUARD
    assert "googleapis.com" in STARTUP_GUARD
    assert "GOOGLE_APPLICATION_CREDENTIALS" in STARTUP_GUARD
