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
LOCAL_ONLY_KEYS = (
    "LOCAL_AUTH_SIGNING_KEY_FILE",
    "LOCAL_METADATA_ENDPOINT",
    "LOCAL_GCS_ENDPOINT",
    "LOCAL_PUBSUB_ENDPOINT",
    "LOCAL_EMBEDDING_ENDPOINT",
    "LOCAL_VECTOR_ENDPOINT",
    "LOCAL_GENERATION_ENDPOINT",
)


def gcp_config(local_values: dict[str, Any]) -> dict[str, Any]:
    values = {
        **local_values,
        "APP_MODE": "gcp",
        "AUTH_MODE": "cloud_run_iam",
        "GCP_PROJECT_ID": "my-dev-project",
        "GCP_REGION": "us-central1",
        "TF_STATE_PROJECT_ID": "my-state-project",
        "TF_STATE_BUCKET_NAME": "my-state-bucket",
        "TF_STATE_BUCKET_LOCATION": "US",
        "ARTIFACT_REGISTRY_REPOSITORY": "rag-images",
        "DEV_OPERATOR_GROUP": "rag-operators@example.com",
        "ALERT_EMAIL": "alerts@example.com",
        "BUDGET_AMOUNT_USD": 100,
        "VECTOR_SEARCH_SHARD_SIZE": "SHARD_SIZE_SMALL",
        "LOG_RETENTION_DAYS": 30,
        "SOURCE_REPOSITORY": "N/A",
        "PUBSUB_WORKER_AUDIENCE": "https://worker.example.com",
        "GENERATION_MODEL_ID": "gemini-2.5-flash-001",
        "EMBEDDING_MODEL_ID": "text-embedding-005",
    }
    for key in LOCAL_ONLY_KEYS:
        values.pop(key)
    return values


def test_local_profile_matches_frozen_json_schema(local_values: dict[str, Any]) -> None:
    settings = RuntimeSettings.model_validate(local_values)
    schema = json.loads(
        (ROOT / "contracts/config/runtime-config.v1.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(settings.model_dump(by_alias=True, exclude_none=True, mode="json"))


def test_local_mode_needs_no_cloud_values_or_credentials(local_values: dict[str, Any]) -> None:
    settings = RuntimeSettings.model_validate(local_values)
    assert settings.app_mode == "local"
    assert settings.gcp_project_id is None
    assert "credentials" not in settings.model_dump()


def test_gcp_profile_matches_frozen_json_schema(local_values: dict[str, Any]) -> None:
    settings = RuntimeSettings.model_validate(gcp_config(local_values))
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
        ("PUBSUB_WORKER_AUDIENCE", "worker:8080"),
        ("LOCAL_GCS_ENDPOINT", "grpc://gcs:4443"),
        ("MAX_QUERY_TOP_K", 51),
        ("LOCAL_AUTH_SIGNING_KEY_FILE", "relative.key"),
    ],
)
def test_invalid_local_configuration_fails_closed(
    local_values: dict[str, Any], key: str, value: object
) -> None:
    local_values[key] = value
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(local_values)


def test_gcp_mode_requires_all_cloud_gate_values(local_values: dict[str, Any]) -> None:
    values = gcp_config(local_values)
    values.pop("GCP_REGION")
    with pytest.raises(ValidationError, match="GCP_REGION"):
        RuntimeSettings.model_validate(values)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("GCP_PROJECT_ID", "PROJECT_PLACEHOLDER"),
        ("GCS_DOCUMENT_BUCKET", "bucket-placeholder"),
        ("TENANT_UPLOAD_PREFIX", "tenants/dev-tenant/todo"),
        ("GENERATION_MODEL_ID", "gemini-latest"),
        ("EMBEDDING_MODEL_ID", "TODO"),
    ],
)
def test_gcp_mode_rejects_placeholders_and_mutable_models(
    local_values: dict[str, Any], key: str, value: str
) -> None:
    values = gcp_config(local_values)
    values[key] = value
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(values)


def test_gcp_mode_forbids_local_signing_key(local_values: dict[str, Any]) -> None:
    values = gcp_config(local_values)
    values["LOCAL_AUTH_SIGNING_KEY_FILE"] = "/tmp/secret"
    with pytest.raises(ValidationError, match="forbidden"):
        RuntimeSettings.model_validate(values)


@pytest.mark.parametrize("key", LOCAL_ONLY_KEYS[1:])
def test_gcp_mode_forbids_local_endpoints(local_values: dict[str, Any], key: str) -> None:
    values = gcp_config(local_values)
    values[key] = "http://local-only:8080"
    with pytest.raises(ValidationError, match="forbidden"):
        RuntimeSettings.model_validate(values)


@pytest.mark.parametrize("key", LOCAL_ONLY_KEYS[1:])
def test_local_mode_requires_every_endpoint(local_values: dict[str, Any], key: str) -> None:
    local_values.pop(key)
    with pytest.raises(ValidationError, match=key):
        RuntimeSettings.model_validate(local_values)


def test_local_mode_requires_local_auth_and_signing_key(local_values: dict[str, Any]) -> None:
    wrong_auth = {**local_values, "AUTH_MODE": "cloud_run_iam"}
    with pytest.raises(ValidationError, match="local_signed"):
        RuntimeSettings.model_validate(wrong_auth)

    local_values.pop("LOCAL_AUTH_SIGNING_KEY_FILE")
    with pytest.raises(ValidationError, match="signing-key"):
        RuntimeSettings.model_validate(local_values)


def test_gcp_mode_requires_cloud_run_auth(local_values: dict[str, Any]) -> None:
    values = gcp_config(local_values)
    values["AUTH_MODE"] = "local_signed"
    with pytest.raises(ValidationError, match="cloud_run_iam"):
        RuntimeSettings.model_validate(values)


def test_embedding_retry_window_must_fit_deadline(local_values: dict[str, Any]) -> None:
    local_values.update(
        {
            "PROCESSING_LEASE_SECONDS": 50,
            "PROCESSING_DEADLINE_SECONDS": 100,
            "DEPENDENCY_TIMEOUT_SECONDS": 30,
            "EMBEDDING_MAX_RETRIES": 3,
        }
    )
    with pytest.raises(ValidationError, match="retry window"):
        RuntimeSettings.model_validate(local_values)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TENANT_UPLOAD_PREFIX", "tenants/another-tenant/uploads"),
        ("CHUNK_OVERLAP_TOKENS", 512),
        ("PROCESSING_LEASE_SECONDS", 601),
        ("PROCESSING_DEADLINE_SECONDS", 100),
    ],
)
def test_cross_field_limits_fail_closed(
    local_values: dict[str, Any], key: str, value: object
) -> None:
    local_values[key] = value
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(local_values)


def test_gcp_mode_requires_https_worker_audience(local_values: dict[str, Any]) -> None:
    values = gcp_config(local_values)
    values["PUBSUB_WORKER_AUDIENCE"] = "http://worker.example.com"
    with pytest.raises(ValidationError, match="HTTPS"):
        RuntimeSettings.model_validate(values)


def test_environment_loader_accepts_explicit_local_profile(
    local_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for key, value in local_values.items():
        monkeypatch.setenv(key, str(value))
    assert load_settings().dev_tenant_id == "dev-tenant"


def test_environment_loader_sanitizes_invalid_values(
    local_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for key, value in local_values.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("DEV_TENANT_ID", "super-secret invalid value")
    with pytest.raises(ConfigurationError) as captured:
        load_settings()
    assert "super-secret" not in str(captured.value)
    assert captured.value.error_count > 0


def test_unknown_dotenv_key_is_rejected(
    local_values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in local_values:
        monkeypatch.delenv(key, raising=False)
    dotenv = "\n".join(f"{key}={value}" for key, value in local_values.items())
    (tmp_path / ".env").write_text(f"{dotenv}\nUNKNOWN_SETTING=value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError):
        load_settings()
