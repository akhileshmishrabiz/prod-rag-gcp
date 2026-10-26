"""Runtime configuration contract tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from packages.rag_core.config import RuntimeSettings, load_settings
from packages.rag_core.errors import ConfigurationError

ROOT = Path(__file__).parents[2]


def test_runtime_profile_matches_frozen_json_schema(runtime_values: dict[str, Any]) -> None:
    settings = RuntimeSettings.model_validate(runtime_values)
    schema = json.loads(
        (ROOT / "contracts/config/runtime-config.v1.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(settings.model_dump(by_alias=True, exclude_none=True, mode="json"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CONFIG_CONTRACT_VERSION", "2.0.0"),
        ("DEV_TENANT_ID", "not valid"),
        ("GCS_DOCUMENT_BUCKET", "Invalid_Bucket"),
        ("FIXED_CORPUS", "another-corpus"),
        ("PUBSUB_WORKER_AUDIENCE", "http://worker.example.com"),
        ("MAX_QUERY_TOP_K", 51),
        ("GCP_PROJECT_ID", "PROJECT_PLACEHOLDER"),
        ("GENERATION_MODEL_ID", "gemini-latest"),
    ],
)
def test_invalid_configuration_fails_closed(
    runtime_values: dict[str, Any], key: str, value: object
) -> None:
    runtime_values[key] = value
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(runtime_values)


def test_requires_https_worker_audience(runtime_values: dict[str, Any]) -> None:
    runtime_values["PUBSUB_WORKER_AUDIENCE"] = "http://worker.example.com"
    with pytest.raises(ValidationError, match="HTTPS"):
        RuntimeSettings.model_validate(runtime_values)


def test_embedding_retry_window_must_fit_deadline(runtime_values: dict[str, Any]) -> None:
    runtime_values.update(
        {
            "PROCESSING_LEASE_SECONDS": 50,
            "PROCESSING_DEADLINE_SECONDS": 100,
            "DEPENDENCY_TIMEOUT_SECONDS": 30,
            "EMBEDDING_MAX_RETRIES": 3,
        }
    )
    with pytest.raises(ValidationError, match="retry window"):
        RuntimeSettings.model_validate(runtime_values)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TENANT_UPLOAD_PREFIX", "tenants/another-tenant/uploads"),
        ("CHUNK_OVERLAP_TOKENS", 512),
        ("PROCESSING_LEASE_SECONDS", 601),
    ],
)
def test_cross_field_limits_fail_closed(
    runtime_values: dict[str, Any], key: str, value: object
) -> None:
    runtime_values[key] = value
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(runtime_values)


def test_environment_loader_accepts_explicit_profile(
    runtime_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for key, value in runtime_values.items():
        monkeypatch.setenv(key, str(value))
    assert load_settings().dev_tenant_id == "dev-tenant"


def test_environment_loader_sanitizes_invalid_values(
    runtime_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for key, value in runtime_values.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("DEV_TENANT_ID", "super-secret invalid value")
    with pytest.raises(ConfigurationError) as captured:
        load_settings()
    assert "super-secret" not in str(captured.value)
    assert captured.value.error_count > 0


def test_unknown_dotenv_key_is_rejected(
    runtime_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in runtime_values:
        monkeypatch.delenv(key, raising=False)
    dotenv = "\n".join(f"{key}={value}" for key, value in runtime_values.items())
    (tmp_path / ".env").write_text(f"{dotenv}\nUNKNOWN_SETTING=value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError):
        load_settings()
