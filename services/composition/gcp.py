"""Build API and worker dependencies from GCP settings and ADC credentials."""

from __future__ import annotations

import logging
import secrets
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from google.cloud import firestore

from packages.rag_core.config import RuntimeSettings
from packages.rag_core.embeddings import VertexEmbeddingModel
from packages.rag_core.firestore_store import FirestoreMetadataStore
from packages.rag_core.generation import VertexGenerator
from packages.rag_core.models import ActiveIndexRecord
from packages.rag_core.policy import GcsObjectPolicy
from packages.rag_core.retrieval import RetrievalLimits, RetrievalService
from packages.rag_core.storage import GcsObjectStore, PubSubEventPublisher
from packages.rag_core.telemetry import JsonTelemetry
from packages.rag_core.vector_store import VertexVectorIndex
from services.api.app.dependencies import ApiDependencies
from services.worker.app.dependencies import WorkerDependencies


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class SystemIds:
    def new_event_id(self) -> UUID:
        return uuid4()

    def new_trace_id(self) -> str:
        return f"trace-{secrets.token_hex(16)}"


def _ensure_active_index(settings: RuntimeSettings, metadata: FirestoreMetadataStore) -> None:
    """Seed the active index document when the project is fresh."""
    reference = metadata.client.document("config/active_index")
    if reference.get(timeout=settings.dependency_timeout_seconds, retry=None).exists:
        return
    record = ActiveIndexRecord(
        index_version=settings.active_index_version,
        pipeline_version=settings.pipeline_version,
        parser_version=settings.parser_version,
        chunker_version=settings.chunker_version,
        embedding_model_id=settings.embedding_model_id,
        embedding_dimensions=settings.embedding_dimensions,
        updated_at=datetime.now(UTC),
    )
    reference.set(record.model_dump(mode="python"), timeout=settings.dependency_timeout_seconds)


def _metadata(settings: RuntimeSettings) -> FirestoreMetadataStore:
    client = firestore.Client(project=settings.gcp_project_id)
    store = FirestoreMetadataStore(
        client=client,
        tenant_id=settings.dev_tenant_id,
        dependency_timeout=settings.dependency_timeout_seconds,
        max_active_generations=settings.max_active_generations,
        max_chunks_per_document=settings.max_document_chunks,
    )
    _ensure_active_index(settings, store)
    return store


def _objects(settings: RuntimeSettings) -> GcsObjectStore:
    return GcsObjectStore.from_project(
        project=settings.gcp_project_id,
        policy=GcsObjectPolicy(
            bucket=settings.gcs_document_bucket,
            tenant_id=settings.dev_tenant_id,
            max_bytes=settings.max_document_bytes,
        ),
    )


def _publisher(settings: RuntimeSettings) -> PubSubEventPublisher:
    return PubSubEventPublisher.from_project_topic(
        project=settings.gcp_project_id,
        topic=settings.pubsub_topic,
    )


def _retrieval(settings: RuntimeSettings, metadata: FirestoreMetadataStore) -> RetrievalService:
    embeddings = VertexEmbeddingModel.from_project(
        project=settings.gcp_project_id,
        region=settings.gcp_region,
        model_id=settings.embedding_model_id,
        max_batch_size=settings.max_embedding_batch_size,
    )
    vectors = VertexVectorIndex.from_resources(
        region=settings.gcp_region,
        index_resource=settings.vector_search_index_resource,
        endpoint_resource=settings.vector_search_endpoint_resource,
        deployed_index_id=settings.vector_search_deployed_index_id,
        max_batch_size=settings.max_embedding_batch_size,
    )
    generator = VertexGenerator.from_project(
        project=settings.gcp_project_id,
        region=settings.gcp_region,
        model_id=settings.generation_model_id,
    )
    return RetrievalService(
        metadata=metadata,
        embeddings=embeddings,
        vectors=vectors,
        generator=generator,
        tenant_id=settings.dev_tenant_id,
        pipeline_version=settings.pipeline_version,
        parser_version=settings.parser_version,
        chunker_version=settings.chunker_version,
        generation_model_id=settings.generation_model_id,
        embedding_model_id=settings.embedding_model_id,
        embedding_dimensions=settings.embedding_dimensions,
        limits=RetrievalLimits(
            max_active_generations=settings.max_active_generations,
            max_candidates_per_document=settings.max_candidates_per_document,
            max_context_chars=settings.max_context_chars,
            max_context_tokens=settings.max_context_tokens,
            max_evidence_chars=settings.max_evidence_chars,
            max_output_tokens=settings.max_output_tokens,
            dependency_timeout_seconds=settings.dependency_timeout_seconds,
        ),
    )


def build_api_dependencies(settings: RuntimeSettings) -> ApiDependencies:
    metadata = _metadata(settings)
    return ApiDependencies(
        metadata=metadata,
        objects=_objects(settings),
        publisher=_publisher(settings),
        retrieval=_retrieval(settings, metadata),
        clock=SystemClock(),
        ids=SystemIds(),
    )


def build_worker_dependencies(settings: RuntimeSettings) -> WorkerDependencies:
    metadata = _metadata(settings)
    embeddings = VertexEmbeddingModel.from_project(
        project=settings.gcp_project_id,
        region=settings.gcp_region,
        model_id=settings.embedding_model_id,
        max_batch_size=settings.max_embedding_batch_size,
    )
    vectors = VertexVectorIndex.from_resources(
        region=settings.gcp_region,
        index_resource=settings.vector_search_index_resource,
        endpoint_resource=settings.vector_search_endpoint_resource,
        deployed_index_id=settings.vector_search_deployed_index_id,
        max_batch_size=settings.max_embedding_batch_size,
    )
    telemetry_logger = logging.getLogger("rag.worker")
    telemetry_logger.setLevel(logging.INFO)
    return WorkerDependencies(
        metadata=metadata,
        objects=_objects(settings),
        embeddings=embeddings,
        vectors=vectors,
        clock=SystemClock(),
        telemetry=JsonTelemetry(telemetry_logger),
        owner=f"worker-{settings.gcp_region}",
        sleep=time.sleep,
        jitter=lambda attempt: min(float(attempt), 5.0),
    )
