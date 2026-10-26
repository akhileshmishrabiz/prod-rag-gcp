"""Cloud Run entrypoint wiring and IAM boundary tests."""

from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError
from services.api.app.auth import require_identity
from services.api.app.main import create_app as create_api_app
from services.worker.app.auth import require_push_identity
from services.worker.app.main import create_app as create_worker_app


def test_api_wires_gcp_dependencies(
    runtime_settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = MagicMock()
    monkeypatch.setattr(
        "services.composition.gcp.build_api_dependencies",
        lambda _settings: bundle,
    )
    app = create_api_app(lambda: runtime_settings, wire_gcp=True)
    assert app.state.dependencies is bundle


def test_worker_wires_gcp_dependencies(
    runtime_settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = MagicMock()
    monkeypatch.setattr(
        "services.composition.gcp.build_worker_dependencies",
        lambda _settings: bundle,
    )
    app = create_worker_app(lambda: runtime_settings, wire_gcp=True)
    assert app.state.dependencies is bundle


def _request(settings: object | None, dependencies: object | None) -> Request:
    app = MagicMock()
    app.state.settings = settings
    app.state.dependencies = dependencies
    return Request({"type": "http", "headers": [], "app": app})


def test_api_auth_requires_configuration() -> None:
    with pytest.raises(RagError) as error:
        require_identity(_request(None, MagicMock()))
    assert error.value.code is ErrorCode.CONFIGURATION_INVALID


def test_worker_push_auth_requires_configuration() -> None:
    with pytest.raises(RagError) as error:
        require_push_identity(_request(MagicMock(), None))
    assert error.value.code is ErrorCode.CONFIGURATION_INVALID
