"""Runtime configuration matching contract version 1.0.0."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import (
    AnyUrl,
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
ResourceName = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class RuntimeSettings(BaseSettings):
    """Cloud Run runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        alias_generator=str.upper,
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=True,
    )

    config_contract_version: str
    dev_tenant_id: TenantId
    gcp_project_id: CloudString
    gcp_region: CloudString
    gcs_document_bucket: BucketName
    tenant_upload_prefix: UploadPrefix
    fixed_corpus: Literal["default"]
    pubsub_topic: VersionId
    pubsub_worker_audience: HttpEndpoint
    active_index_version: VersionId
    vector_search_index_resource: ResourceName
    vector_search_endpoint_resource: ResourceName
    vector_search_deployed_index_id: VersionId

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
    def validate_configuration(self) -> Self:
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
        if self.pubsub_worker_audience.scheme != "https":
            raise ValueError("worker audience must be an HTTPS URL")

        cloud_strings = (
            self.gcp_project_id,
            self.gcp_region,
            self.gcs_document_bucket,
            self.vector_search_index_resource,
            self.vector_search_endpoint_resource,
            self.vector_search_deployed_index_id,
            self.generation_model_id,
            self.embedding_model_id,
        )
        if any(_PLACEHOLDER.search(value) for value in cloud_strings):
            raise ValueError("placeholder values are not allowed")
        if _MUTABLE_MODEL_ALIASES.search(self.generation_model_id) or _MUTABLE_MODEL_ALIASES.search(
            self.embedding_model_id
        ):
            raise ValueError("pinned model IDs are required")
        return self


def load_settings() -> RuntimeSettings:
    """Load environment settings and expose only a sanitized failure."""
    try:
        return RuntimeSettings()
    except ValidationError as error:
        raise ConfigurationError.from_validation_error(error) from None
