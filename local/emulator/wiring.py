"""Local-only cross-process adapters and dependency composition."""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, NoReturn, cast
from uuid import UUID, uuid4

import google.cloud.pubsub_v1 as pubsub_v1  # type: ignore[import-untyped]
import google.cloud.storage as storage  # type: ignore[import-untyped]
from google.api_core.client_options import ClientOptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore

from packages.rag_core.config import AppMode, RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.firestore_store import FirestoreMetadataStore
from packages.rag_core.generation import Generator
from packages.rag_core.models import (
    ActiveIndexRecord,
    AuthorizedContext,
    GeneratedAnswer,
    RetrievalCandidate,
    VectorDatapoint,
    VectorRestrictions,
    VerificationResult,
)
from packages.rag_core.policy import GcsObjectPolicy
from packages.rag_core.retrieval import RetrievalLimits, RetrievalService
from packages.rag_core.storage import GcsObjectStore, PubSubEventPublisher
from packages.rag_core.telemetry import JsonTelemetry
from services.api.app.dependencies import ApiDependencies, ApiPolicy
from services.worker.app.dependencies import WorkerDependencies

_PROJECT = "local-rag-project"
_INDEX_VERSION = "local-index-1"


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


def _local_endpoint(value: object, name: str) -> str:
    endpoint = str(value).rstrip("/")
    if (
        not endpoint.startswith("http://")
        or "googleapis.com" in endpoint
        or "google.com" in endpoint
    ):
        raise ValueError(f"{name} must be a local HTTP endpoint")
    return endpoint


class _JsonClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = _local_endpoint(endpoint, "model endpoint")

    @staticmethod
    def _raise_http(error: urllib.error.HTTPError) -> NoReturn:
        if error.code >= 500:
            raise RetryableDependencyError() from error
        raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error

    def post(self, path: str, body: dict[str, object], deadline: float) -> dict[str, Any]:
        if deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        request = urllib.request.Request(  # noqa: S310 - endpoint is constrained at construction
            f"{self.endpoint}{path}",
            data=json.dumps(body, separators=(",", ":")).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(  # nosec B310  # noqa: S310 - local endpoint only
                request, timeout=deadline
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            self._raise_http(error)
        except (TimeoutError, urllib.error.URLError) as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error
        if not isinstance(payload, dict) or payload.get("non_production") is not True:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return cast(dict[str, Any], payload)


class LocalEmbeddingModel:
    def __init__(self, endpoint: str, max_batch_size: int) -> None:
        self.client = _JsonClient(endpoint)
        self.max_batch_size = max_batch_size

    def embed(self, texts: Sequence[str], dimensions: int, deadline: float) -> list[list[float]]:
        if not texts or len(texts) > self.max_batch_size:
            raise RagError(ErrorCode.INVALID_REQUEST)
        payload = self.client.post(
            "/v1/embeddings",
            {"texts": list(texts), "dimensions": dimensions},
            deadline,
        )
        vectors = payload.get("vectors")
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(texts)
            or any(
                not isinstance(vector, list)
                or len(vector) != dimensions
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
                for vector in vectors
            )
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return [[float(value) for value in vector] for vector in vectors]


class LocalVectorIndex:
    def __init__(self, endpoint: str, max_batch_size: int) -> None:
        self.client = _JsonClient(endpoint)
        self.max_batch_size = max_batch_size

    def upsert(self, datapoints: Sequence[VectorDatapoint], deadline: float) -> None:
        if not datapoints or len(datapoints) > self.max_batch_size:
            raise RagError(ErrorCode.INVALID_REQUEST)
        self.client.post(
            "/v1/vectors:upsert",
            {"datapoints": [item.model_dump(mode="json") for item in datapoints]},
            deadline,
        )

    def delete(self, vector_ids: Sequence[str], deadline: float) -> None:
        if not vector_ids or len(vector_ids) > self.max_batch_size:
            raise RagError(ErrorCode.INVALID_REQUEST)
        self.client.post("/v1/vectors:delete", {"vector_ids": list(vector_ids)}, deadline)

    def search(
        self,
        vector: list[float],
        restrictions: VectorRestrictions,
        top_k: int,
        deadline: float,
    ) -> list[RetrievalCandidate]:
        if not all(
            (
                restrictions.tenant_id,
                restrictions.corpus,
                restrictions.index_version,
                restrictions.active_generations,
            )
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        payload = self.client.post(
            "/v1/vectors:search",
            {
                "vector": vector,
                "restrictions": {
                    "tenant_id": restrictions.tenant_id,
                    "corpus": restrictions.corpus,
                    "index_version": restrictions.index_version,
                    "active_generations": list(restrictions.active_generations),
                },
                "top_k": top_k,
            },
            deadline,
        )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        try:
            return [RetrievalCandidate.model_validate(item) for item in candidates]
        except ValueError as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error

    def read(self, vector_ids: Sequence[str], limit: int, deadline: float) -> list[VectorDatapoint]:
        if (
            not vector_ids
            or len(vector_ids) > limit
            or limit > self.max_batch_size
            or len(set(vector_ids)) != len(vector_ids)
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        payload = self.client.post(
            "/v1/vectors:read",
            {"vector_ids": list(vector_ids), "limit": limit},
            deadline,
        )
        datapoints = payload.get("datapoints")
        if not isinstance(datapoints, list):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        try:
            return [VectorDatapoint.model_validate(item) for item in datapoints]
        except ValueError as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error

    def verify(
        self,
        expected_vector_ids: Sequence[str],
        expected_count: int,
        sample_size: int,
        deadline: float,
    ) -> VerificationResult:
        unique = sorted(set(expected_vector_ids), key=str.encode)
        if expected_count != len(expected_vector_ids) or len(unique) != expected_count:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if not unique:
            return VerificationResult(verified=True, expected_count=0, checked_vector_ids=())
        sample = unique if sample_size >= len(unique) else [*unique[: sample_size - 1], unique[-1]]
        found = self.read(sample, len(sample), deadline)
        found_ids = {item.vector_id for item in found}
        missing = tuple(item for item in sample if item not in found_ids)
        return VerificationResult(
            verified=not missing,
            expected_count=expected_count,
            checked_vector_ids=tuple(sample),
            missing_vector_ids=missing,
        )


class LocalGenerator(Generator):
    def __init__(self, endpoint: str) -> None:
        self.client = _JsonClient(endpoint)

    def generate(
        self,
        question: str,
        authorized_context: Sequence[AuthorizedContext],
        max_output_tokens: int,
        deadline: float,
    ) -> GeneratedAnswer:
        payload = self.client.post(
            "/v1/generation",
            {
                "question": question,
                "authorized_context": [item.model_dump(mode="json") for item in authorized_context],
                "max_output_tokens": max_output_tokens,
            },
            deadline,
        )
        try:
            return GeneratedAnswer.model_validate(
                {
                    "answer": payload.get("answer"),
                    "citations": payload.get("citations"),
                }
            )
        except ValueError as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error


def _metadata(settings: RuntimeSettings) -> FirestoreMetadataStore:
    endpoint = _local_endpoint(settings.local_metadata_endpoint, "metadata endpoint")
    os.environ["FIRESTORE_EMULATOR_HOST"] = endpoint.removeprefix("http://")
    client = firestore.Client(
        project=_PROJECT,
        credentials=AnonymousCredentials(),  # type: ignore[no-untyped-call]
    )
    active_index = ActiveIndexRecord(
        index_version=_INDEX_VERSION,
        pipeline_version=settings.pipeline_version,
        parser_version=settings.parser_version,
        chunker_version=settings.chunker_version,
        embedding_model_id=settings.embedding_model_id,
        embedding_dimensions=settings.embedding_dimensions,
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    reference = client.document("config/active_index")
    if not reference.get(timeout=settings.dependency_timeout_seconds, retry=None).exists:
        reference.set(
            active_index.model_dump(mode="python"), timeout=settings.dependency_timeout_seconds
        )
    return FirestoreMetadataStore(
        client=client,
        tenant_id=settings.dev_tenant_id,
        dependency_timeout=settings.dependency_timeout_seconds,
        max_active_generations=settings.max_active_generations,
        max_chunks_per_document=settings.max_document_chunks,
    )


def _objects(settings: RuntimeSettings) -> GcsObjectStore:
    endpoint = _local_endpoint(settings.local_gcs_endpoint, "GCS endpoint")
    client = storage.Client(
        project=_PROJECT,
        credentials=AnonymousCredentials(),  # type: ignore[no-untyped-call]
        client_options=ClientOptions(api_endpoint=endpoint),
    )
    return GcsObjectStore(
        client=client,
        policy=GcsObjectPolicy(
            bucket=settings.gcs_document_bucket,
            tenant_id=settings.dev_tenant_id,
            max_bytes=settings.max_document_bytes,
        ),
    )


def _publisher(settings: RuntimeSettings) -> PubSubEventPublisher:
    endpoint = _local_endpoint(settings.local_pubsub_endpoint, "Pub/Sub endpoint")
    os.environ["PUBSUB_EMULATOR_HOST"] = endpoint.removeprefix("http://")
    client = pubsub_v1.PublisherClient()
    return PubSubEventPublisher(
        publisher=client,
        topic_path=client.topic_path(_PROJECT, "ingestion"),
    )


def _models(
    settings: RuntimeSettings,
) -> tuple[LocalEmbeddingModel, LocalVectorIndex, LocalGenerator]:
    return (
        LocalEmbeddingModel(
            str(settings.local_embedding_endpoint), settings.max_embedding_batch_size
        ),
        LocalVectorIndex(str(settings.local_vector_endpoint), settings.max_embedding_batch_size),
        LocalGenerator(str(settings.local_generation_endpoint)),
    )


def build_api_dependencies(settings: RuntimeSettings) -> ApiDependencies:
    if settings.app_mode is not AppMode.LOCAL:
        raise ValueError("local dependency factory is forbidden outside local mode")
    metadata = _metadata(settings)
    objects = _objects(settings)
    embeddings, vectors, generator = _models(settings)
    retrieval = RetrievalService(
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
    return ApiDependencies(
        metadata=metadata,
        objects=objects,
        publisher=_publisher(settings),
        retrieval=retrieval,
        clock=SystemClock(),
        ids=SystemIds(),
        policy=ApiPolicy(local_test_identity="local-rag-tester"),
    )


def build_worker_dependencies(settings: RuntimeSettings) -> WorkerDependencies:
    if settings.app_mode is not AppMode.LOCAL:
        raise ValueError("local dependency factory is forbidden outside local mode")
    embeddings, vectors, _ = _models(settings)
    telemetry_logger = logging.getLogger("rag.worker")
    telemetry_logger.setLevel(logging.INFO)
    return WorkerDependencies(
        metadata=_metadata(settings),
        objects=_objects(settings),
        embeddings=embeddings,
        vectors=vectors,
        clock=SystemClock(),
        telemetry=JsonTelemetry(telemetry_logger),
        owner="local-worker",
        sleep=time.sleep,
        jitter=lambda attempt: min(float(attempt), 5.0),
    )
