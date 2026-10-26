"""Shared deterministic test fixtures."""

from typing import Any

import pytest

from packages.rag_core.config import RuntimeSettings


def runtime_config() -> dict[str, Any]:
    """Minimal valid runtime profile for unit tests."""
    return {
        "CONFIG_CONTRACT_VERSION": "1.0.0",
        "DEV_TENANT_ID": "dev-tenant",
        "GCP_PROJECT_ID": "my-dev-project",
        "GCP_REGION": "us-central1",
        "GCS_DOCUMENT_BUCKET": "dev-rag-documents",
        "TENANT_UPLOAD_PREFIX": "tenants/dev-tenant/uploads",
        "FIXED_CORPUS": "default",
        "PUBSUB_TOPIC": "ingestion",
        "PUBSUB_WORKER_AUDIENCE": "https://worker.example.com",
        "ACTIVE_INDEX_VERSION": "index-1",
        "VECTOR_SEARCH_INDEX_RESOURCE": "projects/p/locations/us-central1/indexes/i",
        "VECTOR_SEARCH_ENDPOINT_RESOURCE": "projects/p/locations/us-central1/indexEndpoints/e",
        "VECTOR_SEARCH_DEPLOYED_INDEX_ID": "rag_deployed",
        "PIPELINE_VERSION": "pipeline-1",
        "PARSER_VERSION": "parser-1",
        "CHUNKER_VERSION": "chunker-1",
        "GENERATION_MODEL_ID": "gemini-2.5-flash-001",
        "EMBEDDING_MODEL_ID": "text-embedding-005",
        "EMBEDDING_DIMENSIONS": 768,
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
    }


@pytest.fixture
def runtime_values() -> dict[str, Any]:
    return runtime_config()


@pytest.fixture
def runtime_settings(runtime_values: dict[str, Any]) -> RuntimeSettings:
    return RuntimeSettings.model_validate(runtime_values)
