from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from local.emulator import app as emulator_app
from local.emulator import push_gateway
from local.emulator.app import (
    ApiError,
    EmulatorServer,
    EmulatorStore,
    _finite_vector,
    deterministic_embedding,
)
from local.emulator.wiring import (
    LocalEmbeddingModel,
    LocalGenerator,
    LocalVectorIndex,
    SystemClock,
    SystemIds,
    _JsonClient,
    _local_endpoint,
    _metadata,
    _objects,
    _publisher,
    build_api_dependencies,
    build_worker_dependencies,
)
from local.runtime.apps import build_local_api, build_local_worker
from packages.rag_core.config import AppMode
from packages.rag_core.embeddings import VertexEmbeddingModel
from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.generation import VertexGenerator
from packages.rag_core.models import (
    ActiveIndexRecord,
    AuthorizedContext,
    ChunkRecord,
    GeneratedAnswer,
    MediaType,
    ObjectMetadata,
    VectorDatapoint,
    VectorRestrictions,
)
from packages.rag_core.parsing import _bounded_section
from packages.rag_core.policy import GcsObjectPolicy
from packages.rag_core.storage import GcsObjectStore, PubSubEventPublisher
from packages.rag_core.vector_store import VertexVectorIndex
from tests.fakes import (
    FakeEventPublisher,
    FakeMetadataStore,
    FakeObjectStore,
    FakeVectorIndex,
)


def _datapoint(vector_id: str = "vector-1") -> VectorDatapoint:
    return VectorDatapoint(
        vector_id=vector_id,
        vector=[1.0, 0.0],
        tenant_id="dev-tenant",
        corpus="default",
        document_id="handbook",
        generation=1,
        active_generation="8:handbook:1",
        index_version="local-index-1",
    )


def _restrictions() -> VectorRestrictions:
    return VectorRestrictions(
        tenant_id="dev-tenant",
        corpus="default",
        index_version="local-index-1",
        active_generations=("8:handbook:1",),
    )


def test_local_endpoint_clock_and_ids_are_safe() -> None:
    assert _local_endpoint("http://vertex:8090/", "endpoint") == "http://vertex:8090"
    for endpoint in ("https://aiplatform.googleapis.com", "ftp://vertex"):
        with pytest.raises(ValueError, match="local HTTP"):
            _local_endpoint(endpoint, "endpoint")
    assert SystemClock().now().tzinfo is UTC
    assert SystemClock().monotonic() > 0
    ids = SystemIds()
    assert ids.new_event_id() != ids.new_event_id()
    assert ids.new_trace_id().startswith("trace-")
    assert _bounded_section(" \n ", page=None, remaining_chars=1).text == ""
    with pytest.raises(RagError):
        _bounded_section("too long", page=None, remaining_chars=1)


def test_json_client_success_and_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"non_production":true,"ok":true}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    client = _JsonClient("http://vertex:8090")
    assert client.post("/test", {"x": 1}, 1)["ok"] is True
    with pytest.raises(RagError) as invalid:
        client.post("/test", {}, 0)
    assert invalid.value.code is ErrorCode.INVALID_REQUEST

    def http_error(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("http://vertex", 503, "failure", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    with pytest.raises(RetryableDependencyError):
        client.post("/test", {}, 1)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(Response, "read", lambda _self: b'{"non_production":false}')
    with pytest.raises(RagError):
        client.post("/test", {}, 1)

    def bad_request(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("http://vertex", 400, "failure", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    with pytest.raises(RagError) as rejected:
        client.post("/test", {}, 1)
    assert rejected.value.code is ErrorCode.DEPENDENCY_REJECTED

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(RetryableDependencyError):
        client.post("/test", {}, 1)


def test_http_model_adapters_validate_and_translate(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding = LocalEmbeddingModel("http://vertex:8090", 2)
    monkeypatch.setattr(embedding.client, "post", lambda *_args: {"vectors": [[1.0, 0.0]]})
    assert embedding.embed(["text"], 2, 1) == [[1.0, 0.0]]
    with pytest.raises(RagError):
        embedding.embed([], 2, 1)
    monkeypatch.setattr(embedding.client, "post", lambda *_args: {"vectors": [[1.0]]})
    with pytest.raises(RagError):
        embedding.embed(["text"], 2, 1)

    vector = LocalVectorIndex("http://vertex:8090", 10)
    calls: list[tuple[str, dict[str, object]]] = []

    def vector_post(path: str, body: dict[str, object], _deadline: float) -> dict[str, Any]:
        calls.append((path, body))
        if path.endswith(":search"):
            return {
                "candidates": [
                    {
                        "vector_id": "vector-1",
                        "score": 1.0,
                        "document_id": "handbook",
                        "generation": 1,
                        "chunk_id": "vector-1",
                    }
                ]
            }
        if path.endswith(":read"):
            return {"datapoints": [_datapoint().model_dump(mode="json")]}
        return {}

    monkeypatch.setattr(vector.client, "post", vector_post)
    vector.upsert([_datapoint()], 1)
    vector.delete(["vector-1"], 1)
    assert vector.search([1.0, 0.0], _restrictions(), 1, 1)[0].vector_id == "vector-1"
    assert vector.read(["vector-1"], 1, 1)[0].vector_id == "vector-1"
    assert vector.verify(["vector-1"], 1, 1, 1).verified is True
    assert vector.verify([], 0, 1, 1).verified is True
    assert [path for path, _ in calls] == [
        "/v1/vectors:upsert",
        "/v1/vectors:delete",
        "/v1/vectors:search",
        "/v1/vectors:read",
        "/v1/vectors:read",
    ]
    with pytest.raises(RagError):
        vector.search(
            [1.0],
            VectorRestrictions(tenant_id="", corpus="", index_version="", active_generations=()),
            1,
            1,
        )
    with pytest.raises(RagError):
        vector.read([], 1, 1)
    with pytest.raises(RagError):
        vector.verify(["one"], 2, 1, 1)

    generator = LocalGenerator("http://vertex:8090")
    monkeypatch.setattr(
        generator.client,
        "post",
        lambda *_args: {
            "answer": "Grounded answer.",
            "citations": [{"citation_id": "citation", "quote": "evidence"}],
        },
    )
    result = generator.generate(
        "question", [AuthorizedContext(citation_id="citation", text="evidence")], 10, 1
    )
    assert isinstance(result, GeneratedAnswer)


def test_dependency_factories_are_local_only(
    monkeypatch: pytest.MonkeyPatch, local_settings: Any
) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    metadata = FakeMetadataStore(
        tenant_id=local_settings.dev_tenant_id,
        active_index=ActiveIndexRecord(
            index_version="local-index-1",
            pipeline_version=local_settings.pipeline_version,
            parser_version=local_settings.parser_version,
            chunker_version=local_settings.chunker_version,
            embedding_model_id=local_settings.embedding_model_id,
            embedding_dimensions=local_settings.embedding_dimensions,
            updated_at=now,
        ),
    )
    objects = FakeObjectStore()
    embeddings = LocalEmbeddingModel("http://vertex:8090", 10)
    vectors = LocalVectorIndex("http://vertex:8090", 10)
    generator = LocalGenerator("http://vertex:8090")
    monkeypatch.setattr("local.emulator.wiring._metadata", lambda _settings: metadata)
    monkeypatch.setattr("local.emulator.wiring._objects", lambda _settings: objects)
    monkeypatch.setattr(
        "local.emulator.wiring._models",
        lambda _settings: (embeddings, vectors, generator),
    )
    monkeypatch.setattr(
        "local.emulator.wiring._publisher",
        lambda _settings: FakeEventPublisher(),
    )
    api = build_api_dependencies(local_settings)
    worker = build_worker_dependencies(local_settings)
    assert api.metadata is metadata
    assert worker.metadata is metadata
    cloud = local_settings.model_copy(update={"app_mode": AppMode.GCP})
    with pytest.raises(ValueError, match="forbidden"):
        build_api_dependencies(cloud)
    with pytest.raises(ValueError, match="forbidden"):
        build_worker_dependencies(cloud)


def test_fake_metadata_chunk_pagination_paths(local_settings: Any) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    metadata = FakeMetadataStore(
        tenant_id="dev-tenant",
        active_index=ActiveIndexRecord(
            index_version="local-index-1",
            pipeline_version=local_settings.pipeline_version,
            parser_version=local_settings.parser_version,
            chunker_version=local_settings.chunker_version,
            embedding_model_id=local_settings.embedding_model_id,
            embedding_dimensions=local_settings.embedding_dimensions,
            updated_at=now,
        ),
        max_chunks_per_document=10,
    )
    for ordinal in range(3):
        chunk_id = f"chunk-{ordinal}"
        metadata.chunks[("doc", 1, chunk_id)] = ChunkRecord(
            tenant_id="dev-tenant",
            corpus="default",
            document_id="doc",
            generation=1,
            object_generation=1,
            chunk_id=chunk_id,
            ordinal=ordinal,
            text=f"text {ordinal}",
            text_sha256=hashlib.sha256(f"text {ordinal}".encode()).hexdigest(),
            vector_id=chunk_id,
            pipeline_version=local_settings.pipeline_version,
            parser_version=local_settings.parser_version,
            chunker_version=local_settings.chunker_version,
            staged_at=now,
            created_at=now,
        )
    first = metadata.list_document_chunks("dev-tenant", "doc", None, 2)
    assert [item.chunk_id for item in first.items] == ["chunk-0", "chunk-1"]
    assert first.next_cursor
    second = metadata.list_document_chunks("dev-tenant", "doc", first.next_cursor, 2)
    assert [item.chunk_id for item in second.items] == ["chunk-2"]
    with pytest.raises(RagError):
        metadata.list_document_chunks("dev-tenant", "doc", "invalid", 2)
    with pytest.raises(RagError):
        metadata.list_document_chunks("dev-tenant", "doc", None, 0)


def test_task02_fake_adapter_failure_paths() -> None:
    vectors = FakeVectorIndex(max_batch_size=2)
    with pytest.raises(RagError):
        vectors.upsert([], 1)
    with pytest.raises(RagError):
        vectors.upsert([_datapoint().model_copy(update={"active_generation": "wrong"})], 1)
    with pytest.raises(RagError):
        vectors.delete([], 1)
    with pytest.raises(RagError):
        vectors.search([], _restrictions(), 1, 1)
    with pytest.raises(RagError):
        vectors.verify(["duplicate", "duplicate"], 2, 2, 1)
    assert vectors.verify([], 0, 1, 1).verified is True
    assert vectors.verify(["missing"], 1, 1, 1).verified is False
    vectors.upsert([_datapoint("one"), _datapoint("two")], 1)
    assert vectors.verify(["one", "two"], 2, 2, 1).verified is True
    assert vectors.verify(["one"], 1, 1, 1).checked_vector_ids == ("one",)
    assert (
        vectors.search(
            [1.0],
            _restrictions().model_copy(update={"tenant_id": "other"}),
            1,
            1,
        )
        == []
    )

    objects = FakeObjectStore()
    uri = "gs://rag-local/tenants/dev-tenant/uploads/doc.txt"
    with pytest.raises(RagError):
        objects.stat(uri, 0)
    with pytest.raises(RagError):
        objects.stat(uri, 1)
    with pytest.raises(RagError):
        objects.read_bounded(uri, 1, 10, 1)
    record = ObjectMetadata(
        canonical_uri=uri,
        bucket="rag-local",
        name="tenants/dev-tenant/uploads/doc.txt",
        object_generation=1,
        size=4,
        content_type=MediaType.TEXT,
        checksum_sha256=hashlib.sha256(b"text").hexdigest(),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    objects.put(record, b"text")
    with pytest.raises(RagError):
        objects.read_bounded(uri, 2, 10, 1)
    with pytest.raises(RagError):
        objects.read_bounded(uri, 0, 10, 1)
    with pytest.raises(RagError):
        objects.read_bounded(uri, 1, 2, 1)
    objects.objects[uri] = (record, b"bad!")
    with pytest.raises(RagError):
        objects.read_bounded(uri, 1, 10, 1)
    with pytest.raises(RagError):
        objects.delete(uri, 2, 1)
    assert objects.delete("gs://rag-local/missing", 1, 1).deleted is False
    sampled = FakeVectorIndex(max_batch_size=3)
    sampled.upsert([_datapoint("a"), _datapoint("b"), _datapoint("c")], 1)
    assert sampled.verify(["a", "b", "c"], 3, 2, 1).checked_vector_ids == ("a", "c")


def test_concrete_local_client_factories(
    monkeypatch: pytest.MonkeyPatch, local_settings: Any
) -> None:
    class Snapshot:
        exists = False

    class Reference:
        def get(self, **_kwargs: object) -> Snapshot:
            return Snapshot()

        def set(self, value: object, **_kwargs: object) -> None:
            assert value

    class FirestoreClient:
        def document(self, _path: str) -> Reference:
            return Reference()

    monkeypatch.setattr(
        "local.emulator.wiring.firestore.Client",
        lambda **_kwargs: FirestoreClient(),
    )
    metadata = _metadata(local_settings)
    assert metadata.tenant_id == "dev-tenant"

    storage_client = object()
    monkeypatch.setattr(
        "local.emulator.wiring.storage.Client",
        lambda **_kwargs: storage_client,
    )
    assert _objects(local_settings).client is storage_client

    class Publisher:
        @staticmethod
        def topic_path(project: str, topic: str) -> str:
            return f"projects/{project}/topics/{topic}"

    monkeypatch.setattr(
        "local.emulator.wiring.pubsub_v1.PublisherClient",
        lambda: Publisher(),
    )
    assert _publisher(local_settings).topic_path.endswith("/topics/ingestion")


def test_local_runtime_wiring_is_local_only(
    monkeypatch: pytest.MonkeyPatch, local_settings: Any
) -> None:
    api_dependencies = object()
    worker_dependencies = object()
    monkeypatch.setattr(
        "local.runtime.apps.build_api_dependencies",
        lambda _settings: api_dependencies,
    )
    monkeypatch.setattr(
        "local.runtime.apps.build_worker_dependencies",
        lambda _settings: worker_dependencies,
    )
    assert build_local_api(lambda: local_settings).state.dependencies is api_dependencies
    assert build_local_worker(lambda: local_settings).state.dependencies is worker_dependencies
    cloud = local_settings.model_copy(update={"app_mode": AppMode.GCP})
    with pytest.raises(RuntimeError, match="APP_MODE=local"):
        build_local_api(lambda: cloud)
    with pytest.raises(RuntimeError, match="APP_MODE=local"):
        build_local_worker(lambda: cloud)


def test_vertex_sdk_adapters_translate_responses() -> None:
    class EmbeddingModels:
        @staticmethod
        def embed_content(**_kwargs: object) -> object:
            return type("Response", (), {"embeddings": [type("E", (), {"values": [1.0]})()]})()

        @staticmethod
        def generate_content(**_kwargs: object) -> object:
            return type(
                "Response",
                (),
                {
                    "text": json.dumps(
                        {
                            "answer": "answer",
                            "citations": [{"citation_id": "citation", "quote": "evidence"}],
                        }
                    )
                },
            )()

    client = type("Client", (), {"models": EmbeddingModels()})()
    embedding = VertexEmbeddingModel(
        client=client,
        model_id="embedding-001",
        region="us-central1",
        max_batch_size=2,
    )
    assert embedding.embed(["text"], 1, 1) == [[1.0]]
    with pytest.raises(RagError):
        embedding.embed([], 1, 1)
    generator = VertexGenerator(client=client, model_id="generation-001", region="us-central1")
    assert (
        generator.generate(
            "question",
            [AuthorizedContext(citation_id="citation", text="evidence")],
            10,
            1,
        ).answer
        == "answer"
    )
    with pytest.raises(RagError):
        generator.generate("question", [], 10, 1)
    with pytest.raises(ValueError, match="regional endpoint"):
        VertexEmbeddingModel(
            client=client,
            model_id="latest",
            region="global",
            max_batch_size=2,
        )


def test_vertex_sdk_adapter_failures_are_sanitized() -> None:
    class Models:
        mode = "timeout"

        def embed_content(self, **_kwargs: object) -> object:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "error":
                raise RuntimeError
            return type("Response", (), {"embeddings": []})()

        def generate_content(self, **_kwargs: object) -> object:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "error":
                raise RuntimeError
            return type("Response", (), {"text": "not-json"})()

    models = Models()
    client = type("Client", (), {"models": models})()
    embedding = VertexEmbeddingModel(
        client=client,
        model_id="embedding-001",
        region="us-central1",
        max_batch_size=2,
    )
    generator = VertexGenerator(client=client, model_id="generation-001", region="us-central1")
    context = [AuthorizedContext(citation_id="citation", text="evidence")]
    with pytest.raises(RetryableDependencyError):
        embedding.embed(["text"], 1, 1)
    with pytest.raises(RetryableDependencyError):
        generator.generate("question", context, 10, 1)
    models.mode = "error"
    with pytest.raises(RetryableDependencyError):
        embedding.embed(["text"], 1, 1)
    with pytest.raises(RetryableDependencyError):
        generator.generate("question", context, 10, 1)
    models.mode = "malformed"
    with pytest.raises(RagError):
        embedding.embed(["text"], 1, 1)
    with pytest.raises(RagError):
        generator.generate("question", context, 10, 1)
    with pytest.raises(RagError):
        embedding.embed([""], 1, 1)


def test_gcs_and_pubsub_local_sdk_adapters() -> None:
    class Blob:
        generation = 7
        size = 4
        content_type = "text/plain"
        metadata: ClassVar[dict[str, str]] = {"sha256": hashlib.sha256(b"text").hexdigest()}
        updated = datetime(2025, 1, 1, tzinfo=UTC)
        name = "tenants/dev-tenant/uploads/doc.txt"
        bucket = type("Bucket", (), {"name": "rag-local"})()

        def reload(self, **_kwargs: object) -> None:
            return None

        def download_as_bytes(self, **_kwargs: object) -> bytes:
            return b"text"

        def delete(self, **_kwargs: object) -> None:
            return None

    blob = Blob()

    class Bucket:
        name = "rag-local"

        @staticmethod
        def blob(_name: str) -> Blob:
            return blob

    class StorageClient:
        @staticmethod
        def bucket(_name: str) -> Bucket:
            return Bucket()

    store = GcsObjectStore(
        client=StorageClient(),
        policy=GcsObjectPolicy(bucket="rag-local", tenant_id="dev-tenant", max_bytes=100),
    )
    uri = "gs://rag-local/tenants/dev-tenant/uploads/doc.txt"
    assert store.stat(uri, 1).object_generation == 7
    assert store.read_bounded(uri, 7, 100, 1) == b"text"
    assert store.delete(uri, 7, 1).deleted is True
    with pytest.raises(RagError):
        store.read_bounded(uri, 0, 100, 1)

    class Future:
        @staticmethod
        def result(timeout: float) -> str:
            assert timeout == 1
            return "message-1"

    class Publisher:
        @staticmethod
        def publish(*_args: object, **_kwargs: object) -> Future:
            return Future()

    publisher = PubSubEventPublisher(
        publisher=Publisher(), topic_path="projects/local-rag-project/topics/ingestion"
    )
    from packages.rag_core.models import IngestionMessage, IngestionOperation, MessageMetadata

    message = IngestionMessage(
        operation=IngestionOperation.UPSERT,
        tenant_id="dev-tenant",
        document_id="doc",
        object_uri=uri,
        object_generation=7,
        generation=1,
        checksum_sha256=hashlib.sha256(b"text").hexdigest(),
        corpus="default",
        pipeline_version="pipeline-1",
        trace_id="trace",
        message_metadata=MessageMetadata(
            event_id=SystemIds().new_event_id(),
            occurred_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )
    assert publisher.publish(message, 1).message_id == "message-1"
    with pytest.raises(RagError):
        publisher.publish(message, 0)


def test_gcs_adapter_rejects_bad_dependency_responses() -> None:
    class Blob:
        generation = 7
        size = 4
        content_type = "text/plain"
        updated = datetime(2025, 1, 1, tzinfo=UTC)
        name = "tenants/dev-tenant/uploads/doc.txt"
        bucket = type("Bucket", (), {"name": "rag-local"})()

        def __init__(self) -> None:
            self.metadata: dict[str, str] = {}
            self.error: Exception | None = None

        def reload(self, **_kwargs: object) -> None:
            if self.error:
                raise self.error

        def download_as_bytes(self, **_kwargs: object) -> bytes:
            if self.error:
                raise self.error
            return b"wrong"

        def delete(self, **_kwargs: object) -> None:
            if self.error:
                raise self.error

    blob = Blob()
    bucket = type("Bucket", (), {"blob": lambda _self, _name: blob})()
    client = type("Client", (), {"bucket": lambda _self, _name: bucket})()
    store = GcsObjectStore(
        client=client,
        policy=GcsObjectPolicy(bucket="rag-local", tenant_id="dev-tenant", max_bytes=100),
    )
    uri = "gs://rag-local/tenants/dev-tenant/uploads/doc.txt"
    with pytest.raises(RagError) as missing_checksum:
        store.stat(uri, 1)
    assert missing_checksum.value.code is ErrorCode.DEPENDENCY_REJECTED
    blob.metadata = {"sha256": hashlib.sha256(b"text").hexdigest()}
    with pytest.raises(RagError):
        store.read_bounded(uri, 7, 100, 1)
    blob.error = TimeoutError()
    with pytest.raises(RetryableDependencyError):
        store.stat(uri, 1)
    with pytest.raises(RetryableDependencyError):
        store.read_bounded(uri, 7, 100, 1)
    with pytest.raises(RetryableDependencyError):
        store.delete(uri, 7, 1)
    blob.error = type("NotFound", (Exception,), {})()
    assert store.delete(uri, 7, 1).deleted is False
    blob.error = type("PreconditionFailed", (Exception,), {})()
    with pytest.raises(RagError) as stale:
        store.delete(uri, 7, 1)
    assert stale.value.code is ErrorCode.STALE_GENERATION


def test_vertex_vector_adapter_success_and_failure_paths() -> None:
    class IndexClient:
        mode = "ok"

        def upsert_datapoints(self, **_kwargs: object) -> None:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "error":
                raise RuntimeError

        def remove_datapoints(self, **_kwargs: object) -> None:
            self.upsert_datapoints()

    class EndpointClient:
        mode = "ok"

        def find_neighbors(self, **_kwargs: object) -> object:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "error":
                raise RuntimeError
            return type("Response", (), {"nearest_neighbors": []})()

        def read_index_datapoints(self, **_kwargs: object) -> object:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "error":
                raise RuntimeError
            return type("Response", (), {"datapoints": []})()

    index_client = IndexClient()
    endpoint_client = EndpointClient()
    adapter = VertexVectorIndex(
        index_client=index_client,
        endpoint_client=endpoint_client,
        index_resource="projects/p/locations/r/indexes/i",
        endpoint_resource="projects/p/locations/r/indexEndpoints/e",
        deployed_index_id="deployed",
        region="us-central1",
        max_batch_size=10,
    )
    adapter.upsert([_datapoint()], 1)
    adapter.delete(["vector-1"], 1)
    assert adapter.search([1.0, 0.0], _restrictions(), 1, 1) == []
    assert adapter.read(["vector-1"], 1, 1) == []
    assert adapter.verify(["vector-1"], 1, 1, 1).verified is False
    with pytest.raises(RagError):
        adapter.upsert([], 1)
    with pytest.raises(RagError):
        adapter.delete([], 1)
    with pytest.raises(RagError):
        adapter.read([], 1, 1)
    with pytest.raises(RagError):
        adapter.verify(["same", "same"], 2, 2, 1)
    with pytest.raises(ValueError, match="regional"):
        VertexVectorIndex(
            index_client=index_client,
            endpoint_client=endpoint_client,
            index_resource="",
            endpoint_resource="",
            deployed_index_id="",
            region="global",
        )
    index_client.mode = "timeout"
    with pytest.raises(RetryableDependencyError):
        adapter.upsert([_datapoint()], 1)
    with pytest.raises(RetryableDependencyError):
        adapter.delete(["vector-1"], 1)
    index_client.mode = "error"
    with pytest.raises(RetryableDependencyError):
        adapter.upsert([_datapoint()], 1)
    endpoint_client.mode = "timeout"
    with pytest.raises(RetryableDependencyError):
        adapter.search([1.0], _restrictions(), 1, 1)
    with pytest.raises(RetryableDependencyError):
        adapter.read(["vector-1"], 1, 1)
    endpoint_client.mode = "error"
    with pytest.raises(RetryableDependencyError):
        adapter.search([1.0], _restrictions(), 1, 1)


class _Target(BaseHTTPRequestHandler):
    authorization = ""

    def do_POST(self) -> None:
        type(self).authorization = self.headers["Authorization"]
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return None


def test_push_gateway_forwards_signed_identity() -> None:
    target = ThreadingHTTPServer(("127.0.0.1", 0), _Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    push_gateway.Gateway.target = f"http://127.0.0.1:{target.server_port}/push"
    push_gateway.Gateway.key = b"local-key"
    push_gateway.Gateway.tenant_id = "dev-tenant"
    push_gateway.Gateway.audience = "http://worker:8081"
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), push_gateway.Gateway)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{gateway.server_port}/v1/pubsub/push",
            data=b'{"message":{}}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
            assert response.status == 204
        assert _Target.authorization.startswith("Bearer ")
        with urllib.request.urlopen(
            f"http://127.0.0.1:{gateway.server_port}/readyz", timeout=2
        ) as response:
            assert response.status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{gateway.server_port}/missing", timeout=2)
        invalid = urllib.request.Request(
            f"http://127.0.0.1:{gateway.server_port}/wrong",
            data=b"{}",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(invalid, timeout=2)  # noqa: S310
    finally:
        gateway.shutdown()
        target.shutdown()


def test_push_token_is_deterministic_shape(tmp_path: Path) -> None:
    value = push_gateway.token(b"key", "dev-tenant", "http://worker:8081")
    assert len(value.split(".")) == 2


def test_push_gateway_canonicalizes_emulator_duplicate_fields() -> None:
    body = json.dumps(
        {
            "message": {
                "messageId": "123",
                "message_id": "123",
                "publishTime": "2026-10-23T00:00:00Z",
                "publish_time": "2026-10-23T00:00:00Z",
            },
            "subscription": "projects/local/subscriptions/worker",
        }
    ).encode()
    canonical = json.loads(push_gateway.canonicalize_emulator_envelope(body))
    assert set(canonical["message"]) == {"messageId", "publishTime"}

    conflicting = body.replace(b'"message_id": "123"', b'"message_id": "different"')
    with pytest.raises(ValueError, match="conflicting"):
        push_gateway.canonicalize_emulator_envelope(conflicting)
    with pytest.raises(ValueError, match="invalid"):
        push_gateway.canonicalize_emulator_envelope(b"{}")


def test_push_gateway_main_configures_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "key"
    key.write_bytes(b"local-key")
    served: list[bool] = []

    class Server:
        def __init__(self, address: object, handler: object) -> None:
            assert address == ("0.0.0.0", 8086)  # noqa: S104
            assert handler is push_gateway.Gateway

        def serve_forever(self) -> None:
            served.append(True)

    monkeypatch.setenv("WORKER_PUSH_ENDPOINT", "http://worker:8081/v1/pubsub/push")
    monkeypatch.setenv("LOCAL_AUTH_SIGNING_KEY_FILE", str(key))
    monkeypatch.setenv("DEV_TENANT_ID", "dev-tenant")
    monkeypatch.setenv("PUBSUB_WORKER_AUDIENCE", "http://worker:8081")
    monkeypatch.setattr(push_gateway, "ThreadingHTTPServer", Server)
    push_gateway.main()
    assert served == [True]
    monkeypatch.setenv("WORKER_PUSH_ENDPOINT", "https://worker.example")
    with pytest.raises(SystemExit, match="local worker"):
        push_gateway.main()


def test_model_emulator_http_surface(tmp_path: Path) -> None:
    server = EmulatorServer(
        ("127.0.0.1", 0),
        EmulatorStore(str(tmp_path / "vectors.sqlite3")),
        "local-emulator-admin-only",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(
        path: str,
        body: dict[str, object],
        *,
        admin: bool = False,
        expected: int = 200,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["X-Local-Admin-Token"] = "local-emulator-admin-only"
        request = urllib.request.Request(  # noqa: S310
            f"{base}{path}",
            data=json.dumps(body).encode(),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
                assert response.status == expected
                return cast(dict[str, Any], json.loads(response.read()))
        except urllib.error.HTTPError as error:
            if error.code != expected:
                raise
            return cast(dict[str, Any], json.loads(error.read()))

    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:  # noqa: S310
            assert response.status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/missing", timeout=2)  # noqa: S310
        assert len(post("/v1/embeddings", {"texts": ["text"], "dimensions": 4})["vectors"]) == 1
        assert (
            post(
                "/v1/embeddings",
                {"texts": [], "dimensions": 4},
                expected=400,
            )["code"]
            == "INVALID_REQUEST"
        )
        point = _datapoint().model_dump(mode="json")
        assert post("/v1/vectors:upsert", {"datapoints": [point]})["upserted"] == 1
        assert (
            post(
                "/v1/vectors:read",
                {"vector_ids": ["vector-1"], "limit": 1},
            )["datapoints"][0]["vector_id"]
            == "vector-1"
        )
        assert (
            post(
                "/v1/vectors:search",
                {
                    "vector": [1.0, 0.0],
                    "restrictions": _restrictions().model_dump(mode="json"),
                    "top_k": 1,
                },
            )["candidates"][0]["vector_id"]
            == "vector-1"
        )
        context = [{"citation_id": "citation", "text": "authorized evidence"}]
        assert post(
            "/v1/generation",
            {
                "question": "What is authorized?",
                "authorized_context": context,
                "max_output_tokens": 20,
            },
        )["citations"]
        assert (
            post(
                "/v1/generation",
                {
                    "question": "What is the lunar launch code?",
                    "authorized_context": context,
                    "max_output_tokens": 20,
                },
            )["citations"]
            == []
        )
        assert (
            post(
                "/_test/failures",
                {"operation": "search", "count": 1, "code": "DEPENDENCY_UNAVAILABLE"},
                expected=401,
            )["code"]
            == "UNAUTHENTICATED"
        )
        post(
            "/_test/failures",
            {"operation": "search", "count": 1, "code": "DEPENDENCY_UNAVAILABLE"},
            admin=True,
        )
        assert (
            post(
                "/v1/vectors:search",
                {
                    "vector": [1.0, 0.0],
                    "restrictions": _restrictions().model_dump(mode="json"),
                    "top_k": 1,
                },
                expected=503,
            )["code"]
            == "DEPENDENCY_UNAVAILABLE"
        )
        assert post("/v1/vectors:delete", {"vector_ids": ["vector-1"]})["deleted"] == 1
        assert post("/_test/reset", {}, admin=True)["reset"] is True
        assert post("/unknown", {}, expected=404)["code"] == "NOT_FOUND"
    finally:
        server.shutdown()


def test_emulator_store_rejects_invalid_operations(tmp_path: Path) -> None:
    with pytest.raises(ApiError):
        deterministic_embedding("", 1)
    with pytest.raises(ApiError):
        _finite_vector([True])
    store = EmulatorStore(str(tmp_path / "invalid.sqlite3"))
    with pytest.raises(ApiError):
        store.inject_failure("unknown", 1, "FAIL")
    with pytest.raises(ApiError):
        store.inject_failure("search", 0, "FAIL")
    with pytest.raises(ApiError):
        store.upsert([])
    point = _datapoint().model_dump(mode="json")
    point["active_generation"] = "wrong"
    with pytest.raises(ApiError):
        store.upsert([point])
    with pytest.raises(ApiError):
        store.delete([])
    with pytest.raises(ApiError):
        store.read([], 1)
    with pytest.raises(ApiError):
        store.search([], {}, 0)
    store.inject_failure("search", 2, "DEPENDENCY_UNAVAILABLE")
    with pytest.raises(ApiError):
        store.maybe_fail("search")
    with pytest.raises(ApiError):
        store.maybe_fail("search")
    store.maybe_fail("search")


def test_emulator_main_requires_local_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APP_MODE", "gcp")
    with pytest.raises(SystemExit, match="APP_MODE"):
        emulator_app.main()
    monkeypatch.setenv("APP_MODE", "local")
    monkeypatch.setenv("LOCAL_EMULATOR_ADMIN_TOKEN", "short")
    with pytest.raises(SystemExit, match="at least"):
        emulator_app.main()

    served: list[bool] = []

    class Server:
        def __init__(self, address: object, store: object, token: str) -> None:
            assert address == ("0.0.0.0", 8090)  # noqa: S104
            assert isinstance(store, EmulatorStore)
            assert token == "local-emulator-admin-only"

        def serve_forever(self) -> None:
            served.append(True)

    monkeypatch.setenv("LOCAL_EMULATOR_ADMIN_TOKEN", "local-emulator-admin-only")
    monkeypatch.setenv("LOCAL_EMULATOR_DB", str(tmp_path / "main.sqlite3"))
    monkeypatch.setattr(emulator_app, "EmulatorServer", Server)
    emulator_app.main()
    assert served == [True]
