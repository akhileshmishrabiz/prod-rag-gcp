"""API and worker process-probe tests."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ConfigurationError
from packages.rag_core.models import ActiveIndexRecord
from packages.rag_core.telemetry import JsonTelemetry
from services.api.app.main import create_app as create_api
from services.worker.app.dependencies import WorkerDependencies
from services.worker.app.main import create_app as create_worker
from tests.fakes import (
    FakeClock,
    FakeEmbeddingModel,
    FakeMetadataStore,
    FakeObjectStore,
    FakeVectorIndex,
)

AppFactory = Callable[[Callable[[], RuntimeSettings]], FastAPI]


def invalid_loader() -> RuntimeSettings:
    raise ConfigurationError(1)


@pytest.mark.parametrize("factory", [create_api, create_worker])
def test_liveness_is_dependency_free_when_configuration_is_invalid(factory: AppFactory) -> None:
    response = TestClient(factory(invalid_loader)).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("factory", [create_api, create_worker])
def test_readiness_succeeds_with_valid_configuration(
    factory: AppFactory, runtime_settings: RuntimeSettings
) -> None:
    response = TestClient(factory(lambda: runtime_settings)).get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("factory", [create_api, create_worker])
def test_readiness_failure_is_structured_and_sanitized(factory: AppFactory) -> None:
    response = TestClient(factory(invalid_loader)).get("/readyz")
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json() == {
        "type": "https://example.invalid/problems/configuration-invalid",
        "title": "Service configuration is invalid",
        "status": 503,
        "code": "CONFIGURATION_INVALID",
        "detail": "Readiness checks failed; inspect sanitized service logs.",
        "trace_id": "readiness",
        "request_id": "readiness",
        "retryable": False,
    }
    body = response.text.lower()
    assert "traceback" not in body
    assert "secret" not in body


def test_worker_does_not_accept_ingestion_payloads(runtime_settings: RuntimeSettings) -> None:
    response = TestClient(create_worker(lambda: runtime_settings)).post("/v1/ingest", json={})
    assert response.status_code == 404


def test_worker_readiness_rejects_incompatible_active_index(
    runtime_settings: RuntimeSettings,
) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    metadata = FakeMetadataStore(
        tenant_id=runtime_settings.dev_tenant_id,
        active_index=ActiveIndexRecord(
            index_version=runtime_settings.active_index_version,
            pipeline_version=runtime_settings.pipeline_version,
            parser_version=runtime_settings.parser_version,
            chunker_version=runtime_settings.chunker_version,
            embedding_model_id=runtime_settings.embedding_model_id,
            embedding_dimensions=runtime_settings.embedding_dimensions,
            updated_at=now,
        ),
    )
    metadata.active_index = metadata.active_index.model_copy(update={"chunker_version": "other"})
    logger = logging.getLogger("test.worker")
    dependencies = WorkerDependencies(
        metadata=metadata,
        objects=FakeObjectStore(),
        embeddings=FakeEmbeddingModel(),
        vectors=FakeVectorIndex(),
        clock=FakeClock(now),
        telemetry=JsonTelemetry(logger),
        owner="probe-test",
        sleep=lambda _seconds: None,
        jitter=lambda _attempt: 0.0,
    )
    response = TestClient(create_worker(lambda: runtime_settings, dependencies)).get("/readyz")
    assert response.status_code == 503
    assert response.json()["code"] == "CONFIGURATION_INVALID"
