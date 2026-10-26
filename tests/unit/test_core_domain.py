"""Core domain, adapter request-shape, and deterministic fake tests."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from packages.rag_core.embeddings import VertexEmbeddingModel
from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.generation import VertexGenerator
from packages.rag_core.models import (
    ActiveIndexRecord,
    AuthorizedContext,
    ChunkRecord,
    Claim,
    DocumentRecord,
    DocumentRegistration,
    DocumentStatus,
    IngestionMessage,
    IngestionOperation,
    MediaType,
    MessageMetadata,
    ObjectMetadata,
    Provenance,
    RetrievalCandidate,
    SanitizedFailure,
    VectorRestrictions,
    VerificationResult,
    chunk_ids_sha256,
)
from packages.rag_core.policy import (
    GcsObjectPolicy,
    active_generation_token,
    build_datapoint,
    build_search_restrictions,
    require_tenant,
    validate_scope,
)
from packages.rag_core.storage import GcsObjectStore, PubSubEventPublisher, encode_push_data
from packages.rag_core.telemetry import JsonTelemetry, resource_hash, safe_fields
from packages.rag_core.vector_store import VertexVectorIndex
from tests.fakes import (
    FakeClock,
    FakeEmbeddingModel,
    FakeEventPublisher,
    FakeGenerator,
    FakeIds,
    FakeMetadataStore,
    FakeObjectStore,
    FakeVectorIndex,
)

NOW = datetime(2025, 1, 1, tzinfo=UTC)
CHECKSUM = hashlib.sha256(b"content").hexdigest()
EVENT_ID = UUID("00000000-0000-0000-0000-000000000001")


def active_index() -> ActiveIndexRecord:
    return ActiveIndexRecord(
        index_version="index-1",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model_id="embedding-001",
        embedding_dimensions=2,
        updated_at=NOW,
    )


def registration() -> DocumentRegistration:
    return DocumentRegistration(
        tenant_id="dev-tenant",
        document_id="doc-1",
        corpus="docs",
        object_uri="gs://bucket/tenants/dev-tenant/doc-1.txt",
        object_generation=7,
        checksum_sha256=CHECKSUM,
        media_type=MediaType.TEXT,
        pipeline_version="pipeline-1",
    )


def message(
    generation: int, operation: IngestionOperation = IngestionOperation.UPSERT
) -> IngestionMessage:
    return IngestionMessage(
        operation=operation,
        tenant_id="dev-tenant",
        document_id="doc-1",
        object_uri="gs://bucket/tenants/dev-tenant/doc-1.txt",
        generation=generation,
        object_generation=7,
        checksum_sha256=CHECKSUM,
        corpus="docs",
        pipeline_version="pipeline-1",
        trace_id="trace-1",
        message_metadata=MessageMetadata(
            event_id=EVENT_ID,
            occurred_at=NOW,
        ),
    )


def chunk(generation: int = 1) -> ChunkRecord:
    return ChunkRecord(
        tenant_id="dev-tenant",
        corpus="docs",
        document_id="doc-1",
        generation=generation,
        object_generation=7,
        chunk_id="chunk-1",
        ordinal=0,
        text="authorized evidence",
        text_sha256=hashlib.sha256(b"authorized evidence").hexdigest(),
        vector_id="chunk-1",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        staged_at=NOW,
        created_at=NOW,
    )


def ready_store() -> tuple[FakeMetadataStore, Claim]:
    store = FakeMetadataStore(tenant_id="dev-tenant", active_index=active_index())
    store.register(registration(), "key-1", EVENT_ID, NOW)
    result = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert result.claim is not None
    expected_hash = chunk_ids_sha256(["chunk-1"])
    store.stage_chunks(result.claim, [chunk()], expected_hash, 1, NOW)
    store.commit_ready(
        result.claim,
        ["chunk-1"],
        1,
        Provenance(
            pipeline_version="pipeline-1",
            parser_version="parser-1",
            chunker_version="chunker-1",
            embedding_model_id="embedding-001",
            embedding_dimensions=2,
            active_index_version="index-1",
        ),
        VerificationResult(verified=True, expected_count=1, checked_vector_ids=("chunk-1",)),
        NOW,
    )
    return store, result.claim


def test_models_enforce_ready_and_sanitized_failure_invariants() -> None:
    values = {
        **registration().model_dump(),
        "generation": 1,
        "status": DocumentStatus.READY,
        "tombstoned": False,
        "created_at": NOW,
        "updated_at": NOW,
    }
    with pytest.raises(ValidationError, match="compatibility"):
        DocumentRecord.model_validate(values)
    values["status"] = DocumentStatus.FAILED_TERMINAL
    with pytest.raises(ValidationError, match="failure"):
        DocumentRecord.model_validate(values)
    values.update(
        failure=SanitizedFailure(
            code="INVALID_REQUEST", retryable=False, attempts=1, last_failed_at=NOW
        )
    )
    assert DocumentRecord.model_validate(values).failure is not None


def test_policy_rejects_wrong_bucket_prefix_generation_size_and_checksum() -> None:
    policy = GcsObjectPolicy(bucket="bucket", tenant_id="dev-tenant", max_bytes=7)
    metadata = ObjectMetadata(
        canonical_uri=registration().object_uri,
        bucket="bucket",
        name="tenants/dev-tenant/doc-1.txt",
        object_generation=2,
        size=7,
        content_type=MediaType.TEXT,
        checksum_sha256=CHECKSUM,
        updated_at=NOW,
    )
    policy.validate(
        metadata,
        expected_uri=registration().object_uri,
        expected_object_generation=2,
    )
    with pytest.raises(RagError) as wrong_bucket:
        policy.parse_uri("gs://other/tenants/dev-tenant/doc.txt")
    assert wrong_bucket.value.code is ErrorCode.ACCESS_DENIED
    with pytest.raises(RagError) as stale:
        policy.validate(
            metadata,
            expected_uri=registration().object_uri,
            expected_object_generation=1,
        )
    assert stale.value.code is ErrorCode.STALE_GENERATION
    with pytest.raises(RagError) as mismatch:
        policy.validate(
            metadata,
            expected_uri=registration().object_uri,
            expected_object_generation=2,
            expected_checksum="0" * 64,
        )
    assert mismatch.value.code is ErrorCode.DEPENDENCY_REJECTED


def test_restrictions_are_mandatory_and_collision_safe() -> None:
    restrictions = build_search_restrictions(
        tenant_id="dev-tenant",
        corpus="docs",
        index_version="index-1",
        active_generations=[("doc-1", 1)],
    )
    assert restrictions.active_generations == ("5:doc-1:1",)
    assert active_generation_token("ab:1", 2) != active_generation_token("ab", 12)
    with pytest.raises(RagError):
        build_search_restrictions(
            tenant_id="", corpus="docs", index_version="index-1", active_generations=[]
        )
    datapoint = build_datapoint(
        vector_id="chunk-1",
        vector=[1.0, 0.0],
        tenant_id="dev-tenant",
        corpus="docs",
        document_id="doc-1",
        generation=1,
        index_version="index-1",
    )
    assert datapoint.active_generation == "5:doc-1:1"


def test_metadata_generation_barrier_and_tombstone_hydration() -> None:
    store, old_claim = ready_store()
    assert store.list_active_generations("dev-tenant", "docs", 2) == [("doc-1", 1)]
    tombstone = store.tombstone("dev-tenant", "doc-1", NOW + timedelta(seconds=1))
    assert tombstone.generation == 2
    assert tombstone.tombstoned
    with pytest.raises(RagError) as stale:
        store.commit_ready(
            old_claim,
            ["chunk-1"],
            1,
            Provenance(
                pipeline_version="pipeline-1",
                parser_version="parser-1",
                chunker_version="chunker-1",
                embedding_model_id="embedding-001",
                embedding_dimensions=2,
                active_index_version="index-1",
            ),
            VerificationResult(verified=True, expected_count=1, checked_vector_ids=("chunk-1",)),
            NOW + timedelta(seconds=2),
        )
    assert stale.value.code is ErrorCode.STALE_GENERATION
    candidate = RetrievalCandidate(
        vector_id="chunk-1",
        score=1,
        document_id="doc-1",
        generation=1,
        chunk_id="chunk-1",
    )
    assert store.hydrate_authorized("dev-tenant", "docs", [candidate], "index-1") == []


def test_metadata_idempotency_conflict_failure_and_delete() -> None:
    store = FakeMetadataStore(tenant_id="dev-tenant", active_index=active_index())
    original = store.register(registration(), "key", EVENT_ID, NOW)
    assert store.register(registration(), "key", EVENT_ID, NOW) == original
    published = store.mark_registration_published(
        "dev-tenant",
        original.idempotency.idempotency_key_sha256,
        original.idempotency.request_sha256,
        EVENT_ID,
        NOW,
    )
    assert published.outcome.value == "PUBLISHED"
    assert (
        store.mark_registration_published(
            "dev-tenant",
            published.idempotency_key_sha256,
            published.request_sha256,
            EVENT_ID,
            NOW,
        )
        == published
    )
    with pytest.raises(RagError):
        store.mark_registration_published(
            "dev-tenant", published.idempotency_key_sha256, "0" * 64, EVENT_ID, NOW
        )
    changed = registration().model_copy(update={"corpus": "other"})
    with pytest.raises(RagError) as conflict:
        store.register(changed, "key", EVENT_ID, NOW)
    assert conflict.value.code is ErrorCode.DOCUMENT_CONFLICT
    result = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert result.claim is not None
    failure = SanitizedFailure(
        code="DEPENDENCY_UNAVAILABLE", retryable=True, attempts=1, last_failed_at=NOW
    )
    store.commit_retryable_failure(result.claim, failure, NOW)
    retry = store.claim_event(message(1), "worker", NOW + timedelta(minutes=2), NOW)
    assert retry.claim is not None
    store.commit_terminal_failure(retry.claim, failure, NOW)
    duplicate = store.claim_event(message(1), "worker", NOW + timedelta(minutes=2), NOW)
    assert duplicate.disposition.value == "DUPLICATE_TERMINAL"


def test_deterministic_fakes_filter_record_and_inject_failures() -> None:
    embeddings = FakeEmbeddingModel()
    assert embeddings.embed(["same"], 3, 1) == embeddings.embed(["same"], 3, 1)
    embeddings.fail_next("embed", RagError(ErrorCode.RATE_LIMITED))
    with pytest.raises(RagError):
        embeddings.embed(["x"], 2, 1)
    index = FakeVectorIndex()
    point = build_datapoint(
        vector_id="chunk-1",
        vector=[1.0, 0.0],
        tenant_id="dev-tenant",
        corpus="docs",
        document_id="doc-1",
        generation=1,
        index_version="index-1",
    )
    index.upsert([point], 1)
    assert index.read(["chunk-1"], 1, 1) == [point]
    assert index.verify(["chunk-1"], 1, 1, 1).verified
    calls_before_invalid = len(index.calls)
    with pytest.raises(RagError):
        index.read(["chunk-1"], 0, 1)
    assert len(index.calls) == calls_before_invalid
    restrictions = build_search_restrictions(
        tenant_id="dev-tenant",
        corpus="docs",
        index_version="index-1",
        active_generations=[("doc-1", 1)],
    )
    assert [item.vector_id for item in index.search([1.0, 0.0], restrictions, 1, 1)] == ["chunk-1"]
    index.delete(["chunk-1"], 1)
    assert index.search([1.0, 0.0], restrictions, 1, 1) == []


def test_clock_ids_generator_object_and_publisher_fakes() -> None:
    clock = FakeClock()
    clock.advance(2)
    assert clock.monotonic() == 2
    assert clock.now() == NOW + timedelta(seconds=2)
    ids = FakeIds()
    assert str(ids.new_event_id()).endswith("0001")
    assert ids.new_trace_id() == "trace-00000002"
    assert ids.chunk_id("doc", 1, 0, CHECKSUM) == ids.chunk_id("doc", 1, 0, CHECKSUM)
    generator = FakeGenerator("answer")
    generated = generator.generate(
        "question", [AuthorizedContext(citation_id="c", text="text")], 5, 1
    )
    assert generated.citations[0].quote == "text"
    objects = FakeObjectStore()
    metadata = ObjectMetadata(
        canonical_uri=registration().object_uri,
        bucket="bucket",
        name="tenants/dev-tenant/doc-1.txt",
        object_generation=1,
        size=7,
        content_type=MediaType.TEXT,
        checksum_sha256=CHECKSUM,
        updated_at=NOW,
    )
    objects.put(metadata, b"content")
    assert objects.read_bounded(metadata.canonical_uri, 1, 7, 1) == b"content"
    assert objects.delete(metadata.canonical_uri, 1, 1).deleted
    publisher = FakeEventPublisher()
    assert publisher.publish(message(1), 1).message_id == "message-00000001"


class EmbeddingClient:
    def __init__(self, values: list[list[float]]) -> None:
        self.models = self
        self.values = values
        self.request: dict[str, object] = {}

    def embed_content(self, **kwargs: object) -> object:
        self.request = kwargs
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=values) for values in self.values]
        )


def test_vertex_embedding_request_and_validation_are_offline() -> None:
    client = EmbeddingClient([[1.0, 0.0]])
    adapter = VertexEmbeddingModel(
        client=client, model_id="text-embedding-005", region="us-central1", max_batch_size=2
    )
    assert adapter.embed(["text"], 2, 3) == [[1.0, 0.0]]
    assert client.request["config"] == {"output_dimensionality": 2}
    with pytest.raises(ValueError, match="regional"):
        VertexEmbeddingModel(
            client=client, model_id="model-latest", region="global", max_batch_size=1
        )
    client.values = [[float("nan"), 0]]
    with pytest.raises(RagError) as rejected:
        adapter.embed(["text"], 2, 3)
    assert rejected.value.code is ErrorCode.DEPENDENCY_REJECTED


class GenerationClient:
    def __init__(self, text: str) -> None:
        self.models = self
        self.text = text
        self.request: dict[str, object] = {}

    def generate_content(self, **kwargs: object) -> object:
        self.request = kwargs
        return SimpleNamespace(text=self.text)


def test_vertex_generation_request_is_structured_and_offline() -> None:
    client = GenerationClient(
        json.dumps({"answer": "answer", "citations": [{"citation_id": "c1", "quote": "evidence"}]})
    )
    adapter = VertexGenerator(client=client, model_id="gemini-2.5-flash-001", region="us-central1")
    answer = adapter.generate(
        "question", [AuthorizedContext(citation_id="c1", text="evidence")], 100, 2
    )
    assert answer.answer == "answer"
    assert client.request["config"]["response_mime_type"] == "application/json"  # type: ignore[index]
    with pytest.raises(RagError):
        adapter.generate("question", [], 100, 2)


class VectorClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []

    def upsert_datapoints(self, *, request: dict[str, object], **_: object) -> None:
        self.requests.append(("upsert", request))

    def remove_datapoints(self, *, request: dict[str, object], **_: object) -> None:
        self.requests.append(("delete", request))

    def find_neighbors(self, *, request: dict[str, object], **_: object) -> object:
        self.requests.append(("search", request))
        restricts = [
            SimpleNamespace(namespace="document_id", allow_list=["doc-1"]),
            SimpleNamespace(namespace="generation", allow_list=["1"]),
        ]
        datapoint = SimpleNamespace(datapoint_id="chunk-1", restricts=restricts)
        neighbor = SimpleNamespace(datapoint=datapoint, distance=0.9)
        return SimpleNamespace(nearest_neighbors=[SimpleNamespace(neighbors=[neighbor])])

    def read_index_datapoints(self, *, request: dict[str, object], **_: object) -> object:
        self.requests.append(("read", request))
        datapoints = []
        for vector_id in cast(list[str], request["ids"]):
            if vector_id == "missing":
                continue
            restricts = [
                SimpleNamespace(namespace="tenant_id", allow_list=["dev-tenant"]),
                SimpleNamespace(namespace="corpus", allow_list=["docs"]),
                SimpleNamespace(namespace="document_id", allow_list=["doc-1"]),
                SimpleNamespace(namespace="generation", allow_list=["1"]),
                SimpleNamespace(namespace="active_generation", allow_list=["5:doc-1:1"]),
                SimpleNamespace(namespace="index_version", allow_list=["index-1"]),
            ]
            datapoints.append(
                SimpleNamespace(
                    datapoint_id=vector_id,
                    feature_vector=[1.0, 0.0],
                    restricts=restricts,
                )
            )
        return SimpleNamespace(datapoints=datapoints)


def test_vertex_vector_request_has_every_restrict_and_no_call_on_missing() -> None:
    client = VectorClient()
    adapter = VertexVectorIndex(
        index_client=client,
        endpoint_client=client,
        index_resource="projects/p/locations/us/indexes/i",
        endpoint_resource="projects/p/locations/us/indexEndpoints/e",
        deployed_index_id="deployed",
        region="us-central1",
    )
    point = build_datapoint(
        vector_id="chunk-1",
        vector=[1.0, 0.0],
        tenant_id="dev-tenant",
        corpus="docs",
        document_id="doc-1",
        generation=1,
        index_version="index-1",
    )
    adapter.upsert([point], 2)
    namespaces = {
        item["namespace"] for item in cast(Any, client.requests[0][1]["datapoints"])[0]["restricts"]
    }
    assert namespaces == {
        "tenant_id",
        "corpus",
        "document_id",
        "generation",
        "active_generation",
        "index_version",
    }
    restrictions = build_search_restrictions(
        tenant_id="dev-tenant",
        corpus="docs",
        index_version="index-1",
        active_generations=[("doc-1", 1)],
    )
    assert adapter.search([1.0, 0.0], restrictions, 1, 2)[0].document_id == "doc-1"
    adapter.delete(["chunk-1"], 2)
    before = len(client.requests)
    invalid = VectorRestrictions(
        tenant_id="", corpus="docs", index_version="index-1", active_generations=("5:doc-1:1",)
    )
    with pytest.raises(RagError):
        adapter.search([1.0, 0.0], invalid, 1, 2)
    assert len(client.requests) == before
    assert adapter.read(["chunk-1"], 1, 2) == [point]
    verified = adapter.verify(["chunk-1", "missing"], 2, 2, 2)
    assert not verified.verified
    assert verified.missing_vector_ids == ("missing",)
    before = len(client.requests)
    with pytest.raises(RagError):
        adapter.read(["chunk-1", "missing"], 1, 2)
    with pytest.raises(RagError):
        adapter.verify(["chunk-1"], 2, 1, 2)
    assert len(client.requests) == before


class Blob:
    def __init__(self) -> None:
        self.bucket = SimpleNamespace(name="bucket")
        self.name = "tenants/dev-tenant/doc-1.txt"
        self.generation = 1
        self.size = 7
        self.content_type = "text/plain"
        self.metadata = {"sha256": CHECKSUM}
        self.updated = NOW
        self.deleted = False

    def reload(self, **_: object) -> None: ...

    def download_as_bytes(self, **_: object) -> bytes:
        return b"content"

    def delete(self, **_: object) -> None:
        self.deleted = True


class StorageClient:
    def __init__(self, blob: Blob) -> None:
        self._blob = blob

    def bucket(self, _: str) -> object:
        return SimpleNamespace(blob=lambda _name: self._blob)


class Future:
    def result(self, timeout: float) -> str:
        assert timeout == 2
        return "provider-id"


class Publisher:
    def __init__(self) -> None:
        self.call: tuple[object, ...] = ()

    def publish(self, *args: object, **kwargs: object) -> Future:
        self.call = (*args, kwargs)
        return Future()


def test_storage_and_pubsub_adapters_are_bounded_and_content_free() -> None:
    policy = GcsObjectPolicy(bucket="bucket", tenant_id="dev-tenant", max_bytes=7)
    blob = Blob()
    storage = GcsObjectStore(client=StorageClient(blob), policy=policy)
    assert storage.stat(registration().object_uri, 2).object_generation == 1
    assert storage.read_bounded(registration().object_uri, 1, 7, 2) == b"content"
    assert storage.delete(registration().object_uri, 1, 2).deleted
    publisher_client = Publisher()
    publisher = PubSubEventPublisher(publisher=publisher_client, topic_path="projects/p/topics/t")
    assert publisher.publish(message(1), 2).message_id == "provider-id"
    raw = cast(bytes, publisher_client.call[1])
    assert b"content" not in raw
    assert "operation" in cast(dict[str, str], publisher_client.call[2])
    assert (
        json.loads(__import__("base64").b64decode(encode_push_data(message(1))))["operation"]
        == "UPSERT"
    )


def test_telemetry_allowlist_never_accepts_content(caplog: pytest.LogCaptureFixture) -> None:
    assert resource_hash("doc-1", salt=b"salt") == resource_hash("doc-1", salt=b"salt")
    with pytest.raises(ValueError, match="unsafe"):
        safe_fields(prompt="secret")
    logger = logging.getLogger("safe-test")
    with caplog.at_level(logging.INFO):
        JsonTelemetry(logger).emit("completed", {"operation": "UPSERT", "count": 1})
    assert "UPSERT" in caplog.text
    assert "secret" not in caplog.text


def test_errors_are_sanitized() -> None:
    error = RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT)
    assert error.retryable
    assert "secret" not in str(error)


def test_metadata_hydration_delete_and_bounded_active_set() -> None:
    store, _ = ready_store()
    candidate = RetrievalCandidate(
        vector_id="chunk-1",
        score=1,
        document_id="doc-1",
        generation=1,
        chunk_id="chunk-1",
    )
    assert store.hydrate_authorized("dev-tenant", "docs", [candidate], "index-1") == [chunk()]
    with pytest.raises(RagError) as bounded:
        store.list_active_generations("dev-tenant", "docs", 0)
    assert bounded.value.code is ErrorCode.INVALID_REQUEST
    tombstone = store.tombstone("dev-tenant", "doc-1", NOW)
    delete_message = message(tombstone.generation, IngestionOperation.DELETE).model_copy(
        update={
            "message_metadata": MessageMetadata(
                event_id=UUID("00000000-0000-0000-0000-000000000002"),
                occurred_at=NOW,
            )
        }
    )
    claimed = store.claim_event(delete_message, "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    failure = SanitizedFailure(
        code="DEPENDENCY_UNAVAILABLE", retryable=True, attempts=1, last_failed_at=NOW
    )
    preserved = store.record_retryable_delete_failure(claimed.claim, failure, NOW)
    assert preserved.status is DocumentStatus.DELETE_PENDING
    assert preserved.tombstoned
    retried = store.claim_event(delete_message, "worker-2", NOW + timedelta(minutes=1), NOW)
    assert retried.claim is not None
    deleted = store.commit_deleted(retried.claim, NOW)
    assert deleted.status is DocumentStatus.DELETED
    assert store.chunks == {}


def test_fake_fail_closed_edges() -> None:
    embeddings = FakeEmbeddingModel(max_batch_size=1)
    with pytest.raises(RagError):
        embeddings.embed([], 1, 1)
    generator = FakeGenerator()
    with pytest.raises(RagError):
        generator.generate("", [], 0, 0)
    objects = FakeObjectStore()
    with pytest.raises(RagError) as missing:
        objects.stat("gs://bucket/missing", 1)
    assert missing.value.code is ErrorCode.DOCUMENT_NOT_FOUND
    metadata = ObjectMetadata(
        canonical_uri=registration().object_uri,
        bucket="bucket",
        name="tenants/dev-tenant/doc-1.txt",
        object_generation=1,
        size=7,
        content_type=MediaType.TEXT,
        checksum_sha256=CHECKSUM,
        updated_at=NOW,
    )
    objects.put(metadata, b"content")
    with pytest.raises(RagError) as large:
        objects.read_bounded(metadata.canonical_uri, 1, 6, 1)
    assert large.value.code is ErrorCode.DOCUMENT_TOO_LARGE
    with pytest.raises(RagError) as stale:
        objects.delete(metadata.canonical_uri, 2, 1)
    assert stale.value.code is ErrorCode.STALE_GENERATION
    publisher = FakeEventPublisher()
    with pytest.raises(RagError):
        publisher.publish(message(1), 0)


class TimeoutModels:
    models: TimeoutModels

    def __init__(self) -> None:
        self.models = self

    def embed_content(self, **_: object) -> object:
        raise TimeoutError

    def generate_content(self, **_: object) -> object:
        raise TimeoutError


class TimeoutVector:
    def upsert_datapoints(self, **_: object) -> None:
        raise TimeoutError

    def remove_datapoints(self, **_: object) -> None:
        raise TimeoutError

    def find_neighbors(self, **_: object) -> object:
        raise TimeoutError


def test_cloud_adapters_classify_timeouts_without_network() -> None:
    embeddings = VertexEmbeddingModel(
        client=TimeoutModels(),
        model_id="text-embedding-005",
        region="us-central1",
        max_batch_size=1,
    )
    with pytest.raises(RetryableDependencyError) as embedding_error:
        embeddings.embed(["x"], 2, 1)
    assert embedding_error.value.code is ErrorCode.DEPENDENCY_TIMEOUT
    generator = VertexGenerator(
        client=TimeoutModels(), model_id="gemini-2.5-flash-001", region="us-central1"
    )
    with pytest.raises(RetryableDependencyError):
        generator.generate("q", [AuthorizedContext(citation_id="c", text="x")], 1, 1)
    point = build_datapoint(
        vector_id="v",
        vector=[1.0],
        tenant_id="dev-tenant",
        corpus="docs",
        document_id="doc-1",
        generation=1,
        index_version="index-1",
    )
    vector = VertexVectorIndex(
        index_client=TimeoutVector(),
        endpoint_client=TimeoutVector(),
        index_resource="index",
        endpoint_resource="endpoint",
        deployed_index_id="deployed",
        region="us-central1",
    )
    with pytest.raises(RetryableDependencyError):
        vector.upsert([point], 1)
    with pytest.raises(RetryableDependencyError):
        vector.delete(["v"], 1)
    restrictions = build_search_restrictions(
        tenant_id="dev-tenant",
        corpus="docs",
        index_version="index-1",
        active_generations=[("doc-1", 1)],
    )
    with pytest.raises(RetryableDependencyError):
        vector.search([1.0], restrictions, 1, 1)


def test_policy_all_fail_closed_paths() -> None:
    with pytest.raises(RagError):
        require_tenant("other", "dev-tenant")
    with pytest.raises(RagError):
        active_generation_token("", 0)
    with pytest.raises(RagError):
        build_datapoint(
            vector_id="",
            vector=[],
            tenant_id="",
            corpus="",
            document_id="",
            generation=1,
            index_version="",
        )
    with pytest.raises(RagError):
        validate_scope("", "docs", "doc-1")
    policy = GcsObjectPolicy(bucket="bucket", tenant_id="dev-tenant", max_bytes=6)
    for uri in (
        "https://bucket/tenants/dev-tenant/a",
        "gs://bucket/tenants/other/a",
        "gs://bucket/tenants/dev-tenant/../a",
        "gs://bucket/tenants/dev-tenant/",
    ):
        with pytest.raises(RagError):
            policy.parse_uri(uri)
    metadata = ObjectMetadata(
        canonical_uri=registration().object_uri,
        bucket="bucket",
        name="tenants/dev-tenant/doc-1.txt",
        object_generation=1,
        size=7,
        content_type=MediaType.TEXT,
        checksum_sha256=CHECKSUM,
        updated_at=NOW,
    )
    with pytest.raises(RagError) as too_large:
        policy.validate(
            metadata,
            expected_uri=registration().object_uri,
            expected_object_generation=1,
        )
    assert too_large.value.code is ErrorCode.DOCUMENT_TOO_LARGE
    wrong_name = metadata.model_copy(update={"name": "tenants/dev-tenant/other.txt"})
    with pytest.raises(RagError) as rejected:
        policy.validate(
            wrong_name,
            expected_uri=registration().object_uri,
            expected_object_generation=1,
        )
    assert rejected.value.code is ErrorCode.DEPENDENCY_REJECTED


class ErrorBlob(Blob):
    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error

    def reload(self, **_: object) -> None:
        if self.error:
            raise self.error

    def download_as_bytes(self, **_: object) -> bytes:
        if self.error:
            raise self.error
        return super().download_as_bytes()

    def delete(self, **_: object) -> None:
        if self.error:
            raise self.error
        super().delete()


class NotFoundError(Exception):
    pass


class ErrorPublisher:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def publish(self, *_: object, **__: object) -> Future:
        raise self.error


def test_storage_adapter_rejects_bounds_and_classifies_failures() -> None:
    policy = GcsObjectPolicy(bucket="bucket", tenant_id="dev-tenant", max_bytes=7)
    adapter = GcsObjectStore(client=StorageClient(Blob()), policy=policy)
    with pytest.raises(RagError):
        adapter.stat(registration().object_uri, 0)
    with pytest.raises(RagError):
        adapter.read_bounded(registration().object_uri, 1, 0, 1)
    with pytest.raises(RagError):
        adapter.delete(registration().object_uri, 0, 1)
    timeout = GcsObjectStore(client=StorageClient(ErrorBlob(TimeoutError())), policy=policy)
    with pytest.raises(RetryableDependencyError):
        timeout.stat(registration().object_uri, 1)
    with pytest.raises(RetryableDependencyError):
        timeout.read_bounded(registration().object_uri, 1, 7, 1)
    with pytest.raises(RetryableDependencyError):
        timeout.delete(registration().object_uri, 1, 1)
    not_found = NotFoundError()
    not_found.__class__.__name__ = "NotFound"
    missing = GcsObjectStore(client=StorageClient(ErrorBlob(not_found)), policy=policy)
    assert not missing.delete(registration().object_uri, 1, 1).deleted
    publisher = PubSubEventPublisher(
        publisher=ErrorPublisher(TimeoutError()), topic_path="projects/p/topics/t"
    )
    with pytest.raises(RetryableDependencyError):
        publisher.publish(message(1), 1)
    with pytest.raises(RagError):
        publisher.publish(message(1), 0)
