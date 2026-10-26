"""Runtime configuration matching contract version 1.0.0."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AnyUrl,
    EmailStr,
    Field,
    StringConstraints,
    UrlConstraints,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from packages.rag_core.errors import ConfigurationError

CONTRACT_VERSION = "1.0.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9._-]+$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")
_UPLOAD_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_PLACEHOLDER = re.compile(r"(placeholder|changeme|todo|tbd)", re.IGNORECASE)
_MUTABLE_MODEL_ALIASES = re.compile(
    r"(^|[-_@.:/])(latest|preview|stable)($|[-_@.:/])", re.IGNORECASE
)

TenantId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=_IDENTIFIER)]
VersionId = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=_VERSION)]
ModelId = Annotated[str, StringConstraints(min_length=1, max_length=256)]
CloudString = Annotated[str, StringConstraints(min_length=1, max_length=256)]
BucketName = Annotated[str, StringConstraints(min_length=3, max_length=63, pattern=_BUCKET)]
UploadPrefix = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=_UPLOAD_PREFIX)
]
HttpEndpoint = Annotated[AnyUrl, UrlConstraints(max_length=2048, allowed_schemes=["http", "https"])]


class AppMode(StrEnum):
    """Supported runtime adapter modes."""

    LOCAL = "local"
    GCP = "gcp"


class AuthMode(StrEnum):
    """Authentication mode paired with the runtime mode."""

    LOCAL_SIGNED = "local_signed"
    CLOUD_RUN_IAM = "cloud_run_iam"


class VectorSearchShardSize(StrEnum):
    """Frozen Vector Search shard-size values."""

    SMALL = "SHARD_SIZE_SMALL"
    MEDIUM = "SHARD_SIZE_MEDIUM"
    LARGE = "SHARD_SIZE_LARGE"


class RuntimeSettings(BaseSettings):
    """Strict process settings loaded only from explicit values or environment."""

    model_config = SettingsConfigDict(
        alias_generator=str.upper,
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=True,
    )

    config_contract_version: str
    app_mode: AppMode
    auth_mode: AuthMode
    dev_tenant_id: TenantId
    gcs_document_bucket: BucketName
    tenant_upload_prefix: UploadPrefix
    fixed_corpus: Literal["default"]
    pubsub_worker_audience: HttpEndpoint
    local_auth_signing_key_file: Path | None = None
    local_metadata_endpoint: HttpEndpoint | None = None
    local_gcs_endpoint: HttpEndpoint | None = None
    local_pubsub_endpoint: HttpEndpoint | None = None
    local_embedding_endpoint: HttpEndpoint | None = None
    local_vector_endpoint: HttpEndpoint | None = None
    local_generation_endpoint: HttpEndpoint | None = None

    gcp_project_id: CloudString | None = None
    gcp_region: CloudString | None = None
    tf_state_project_id: CloudString | None = None
    tf_state_bucket_name: CloudString | None = None
    tf_state_bucket_location: CloudString | None = None
    artifact_registry_repository: CloudString | None = None
    dev_operator_group: CloudString | None = None
    alert_email: EmailStr | None = None
    budget_amount_usd: Annotated[float, Field(gt=0)] | None = None
    vector_search_shard_size: VectorSearchShardSize | None = None
    log_retention_days: Annotated[int, Field(ge=1, le=3650)] | None = None
    source_repository: Annotated[str, StringConstraints(min_length=1, max_length=2048)] | None = (
        None
    )

    pipeline_version: VersionId
    parser_version: VersionId
    chunker_version: VersionId
    generation_model_id: ModelId
    embedding_model_id: ModelId
    embedding_dimensions: Annotated[int, Field(ge=1, le=10_000)]
    max_document_bytes: Annotated[int, Field(ge=1, le=52_428_800)]
    max_document_chunks: Annotated[int, Field(ge=1, le=10_000)]
    max_embedding_batch_size: Annotated[int, Field(ge=1, le=250)]
    max_query_input_chars: Annotated[int, Field(ge=1, le=16_000)]
    max_query_top_k: Annotated[int, Field(ge=1, le=50)]
    max_active_generations: Annotated[int, Field(ge=1, le=10_000)]
    max_candidates_per_document: Annotated[int, Field(ge=1, le=100)]
    max_context_chars: Annotated[int, Field(ge=1, le=200_000)]
    max_context_tokens: Annotated[int, Field(ge=1, le=100_000)]
    max_evidence_chars: Annotated[int, Field(ge=1, le=200_000)]
    max_output_tokens: Annotated[int, Field(ge=1, le=8192)]
    max_pdf_pages: Annotated[int, Field(ge=1, le=2000)]
    chunk_size_tokens: Annotated[int, Field(ge=64, le=8192)]
    chunk_overlap_tokens: Annotated[int, Field(ge=0, le=2048)]
    dependency_timeout_seconds: Annotated[float, Field(ge=0.1, le=300)]
    processing_deadline_seconds: Annotated[int, Field(ge=10, le=3600)]
    processing_lease_seconds: Annotated[int, Field(ge=10, le=3600)]
    max_processing_attempts: Annotated[int, Field(ge=1, le=20)]
    embedding_max_retries: Annotated[int, Field(ge=0, le=10)]
    worker_concurrency: Annotated[int, Field(ge=1, le=100)]
    api_max_instances: Annotated[int, Field(ge=1, le=100)]
    worker_max_instances: Annotated[int, Field(ge=1, le=100)]

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Enforce mode pairing and reject unresolved cloud configuration."""
        if self.config_contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported configuration contract version")
        if self.tenant_upload_prefix != f"tenants/{self.dev_tenant_id}/uploads":
            raise ValueError("tenant upload prefix must match the configured tenant")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.processing_lease_seconds > self.processing_deadline_seconds:
            raise ValueError("processing lease must not exceed processing deadline")
        retry_window = (self.embedding_max_retries + 1) * self.dependency_timeout_seconds
        if retry_window > self.processing_deadline_seconds:
            raise ValueError("embedding retry window must fit within processing deadline")

        if self.app_mode is AppMode.LOCAL:
            if self.auth_mode is not AuthMode.LOCAL_SIGNED:
                raise ValueError("local mode requires local_signed authentication")
            if self.local_auth_signing_key_file is None:
                raise ValueError("local mode requires a mounted signing-key file path")
            if not self.local_auth_signing_key_file.is_absolute():
                raise ValueError("local signing-key file path must be absolute")
            local_endpoints = {
                "LOCAL_METADATA_ENDPOINT": self.local_metadata_endpoint,
                "LOCAL_GCS_ENDPOINT": self.local_gcs_endpoint,
                "LOCAL_PUBSUB_ENDPOINT": self.local_pubsub_endpoint,
                "LOCAL_EMBEDDING_ENDPOINT": self.local_embedding_endpoint,
                "LOCAL_VECTOR_ENDPOINT": self.local_vector_endpoint,
                "LOCAL_GENERATION_ENDPOINT": self.local_generation_endpoint,
            }
            missing = sorted(name for name, value in local_endpoints.items() if value is None)
            if missing:
                raise ValueError(f"local mode requires endpoints: {', '.join(missing)}")
            return self

        if self.auth_mode is not AuthMode.CLOUD_RUN_IAM:
            raise ValueError("gcp mode requires cloud_run_iam authentication")
        if self.local_auth_signing_key_file is not None:
            raise ValueError("local signing-key configuration is forbidden in gcp mode")
        local_endpoint_values = (
            self.local_metadata_endpoint,
            self.local_gcs_endpoint,
            self.local_pubsub_endpoint,
            self.local_embedding_endpoint,
            self.local_vector_endpoint,
            self.local_generation_endpoint,
        )
        if any(endpoint is not None for endpoint in local_endpoint_values):
            raise ValueError("local endpoint configuration is forbidden in gcp mode")
        if self.pubsub_worker_audience.scheme != "https":
            raise ValueError("gcp mode requires an HTTPS worker audience")

        cloud_values = {
            "GCP_PROJECT_ID": self.gcp_project_id,
            "GCP_REGION": self.gcp_region,
            "TF_STATE_PROJECT_ID": self.tf_state_project_id,
            "TF_STATE_BUCKET_NAME": self.tf_state_bucket_name,
            "TF_STATE_BUCKET_LOCATION": self.tf_state_bucket_location,
            "ARTIFACT_REGISTRY_REPOSITORY": self.artifact_registry_repository,
            "DEV_OPERATOR_GROUP": self.dev_operator_group,
            "ALERT_EMAIL": self.alert_email,
            "BUDGET_AMOUNT_USD": self.budget_amount_usd,
            "VECTOR_SEARCH_SHARD_SIZE": self.vector_search_shard_size,
            "LOG_RETENTION_DAYS": self.log_retention_days,
            "SOURCE_REPOSITORY": self.source_repository,
            "GCS_DOCUMENT_BUCKET": self.gcs_document_bucket,
            "TENANT_UPLOAD_PREFIX": self.tenant_upload_prefix,
            "PUBSUB_WORKER_AUDIENCE": str(self.pubsub_worker_audience),
        }
        missing = sorted(name for name, value in cloud_values.items() if value is None)
        if missing:
            raise ValueError(f"gcp mode requires cloud values: {', '.join(missing)}")

        text_values = (value for value in cloud_values.values() if isinstance(value, str))
        if any(_PLACEHOLDER.search(value) for value in text_values):
            raise ValueError("gcp mode rejects placeholder cloud values")
        if _PLACEHOLDER.search(self.generation_model_id) or _PLACEHOLDER.search(
            self.embedding_model_id
        ):
            raise ValueError("gcp mode rejects placeholder model IDs")
        if _MUTABLE_MODEL_ALIASES.search(self.generation_model_id) or _MUTABLE_MODEL_ALIASES.search(
            self.embedding_model_id
        ):
            raise ValueError("gcp mode requires pinned model IDs")
        return self


def load_settings() -> RuntimeSettings:
    """Load environment settings and expose only a sanitized failure."""
    try:
        return RuntimeSettings()
    except ValidationError as error:
        raise ConfigurationError.from_validation_error(error) from None
