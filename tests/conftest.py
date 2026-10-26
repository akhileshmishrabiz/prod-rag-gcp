"""Shared deterministic foundation fixtures."""

from collections.abc import Mapping
from typing import Any

import pytest

from packages.rag_core.config import RuntimeSettings


def local_config() -> dict[str, Any]:
    """Return the explicit local-safe profile from contract v1.0.0."""
    return {
        "CONFIG_CONTRACT_VERSION": "1.0.0",
        "APP_MODE": "local",
        "AUTH_MODE": "local_signed",
        "DEV_TENANT_ID": "dev-tenant",
        "LOCAL_AUTH_SIGNING_KEY_FILE": "/run/secrets/local-auth.key",
        "GCS_DOCUMENT_BUCKET": "dev-rag-documents",
        "TENANT_UPLOAD_PREFIX": "tenants/dev-tenant/uploads",
        "FIXED_CORPUS": "default",
        "PUBSUB_WORKER_AUDIENCE": "http://worker:8080",
        "PIPELINE_VERSION": "pipeline-1",
        "PARSER_VERSION": "parser-1",
        "CHUNKER_VERSION": "chunker-1",
        "GENERATION_MODEL_ID": "deterministic-generator-v1",
        "EMBEDDING_MODEL_ID": "deterministic-embedding-v1",
        "EMBEDDING_DIMENSIONS": 32,
        "MAX_DOCUMENT_BYTES": 10_485_760,
        "MAX_DOCUMENT_CHUNKS": 1000,
        "MAX_EMBEDDING_BATCH_SIZE": 100,
        "MAX_QUERY_INPUT_CHARS": 4000,
        "MAX_QUERY_TOP_K": 20,
        "MAX_ACTIVE_GENERATIONS": 1000,
        "MAX_CANDIDATES_PER_DOCUMENT": 5,
        "MAX_CONTEXT_CHARS": 40_000,
        "MAX_CONTEXT_TOKENS": 8000,
        "MAX_EVIDENCE_CHARS": 6000,
        "MAX_OUTPUT_TOKENS": 1024,
        "MAX_PDF_PAGES": 200,
        "CHUNK_SIZE_TOKENS": 512,
        "CHUNK_OVERLAP_TOKENS": 64,
        "DEPENDENCY_TIMEOUT_SECONDS": 30,
        "PROCESSING_DEADLINE_SECONDS": 600,
        "PROCESSING_LEASE_SECONDS": 300,
        "MAX_PROCESSING_ATTEMPTS": 5,
        "EMBEDDING_MAX_RETRIES": 3,
        "WORKER_CONCURRENCY": 4,
        "API_MAX_INSTANCES": 2,
        "WORKER_MAX_INSTANCES": 2,
        "LOCAL_METADATA_ENDPOINT": "http://metadata:8080",
        "LOCAL_GCS_ENDPOINT": "http://gcs:4443",
        "LOCAL_PUBSUB_ENDPOINT": "http://pubsub:8085",
        "LOCAL_EMBEDDING_ENDPOINT": "http://embeddings:8080",
        "LOCAL_VECTOR_ENDPOINT": "http://vector:8080",
        "LOCAL_GENERATION_ENDPOINT": "http://generation:8080",
    }


@pytest.fixture
def local_values() -> dict[str, Any]:
    return local_config()


@pytest.fixture
def local_settings(local_values: Mapping[str, Any]) -> RuntimeSettings:
    return RuntimeSettings.model_validate(local_values)
