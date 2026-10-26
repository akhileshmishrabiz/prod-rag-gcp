"""API and worker process-probe tests."""

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ConfigurationError
from services.api.app.main import create_app as create_api
from services.worker.app.main import create_app as create_worker

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
    factory: AppFactory, local_settings: RuntimeSettings
) -> None:
    response = TestClient(factory(lambda: local_settings)).get("/readyz")
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


def test_worker_does_not_accept_ingestion_payloads(local_settings: RuntimeSettings) -> None:
    response = TestClient(create_worker(lambda: local_settings)).post("/v1/ingest", json={})
    assert response.status_code == 404
