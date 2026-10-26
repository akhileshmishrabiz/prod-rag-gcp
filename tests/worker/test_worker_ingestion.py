"""Offline worker unit, contract, integration, and security tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from packages.rag_core.chunking import TextChunk, chunk_sections
from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import (
    ActiveIndexRecord,
    DocumentRegistration,
    DocumentStatus,
    IngestionMessage,
    IngestionOperation,
    MediaType,
    MessageMetadata,
    ObjectMetadata,
)
from packages.rag_core.parsing import ParsedSection, parse_document
from packages.rag_core.storage import encode_push_data
from services.worker.app.dependencies import WorkerDependencies
from services.worker.app.envelope import decode_push_envelope
from services.worker.app.main import create_app
from tests.fakes import (
    FakeClock,
    FakeEmbeddingModel,
    FakeMetadataStore,
    FakeObjectStore,
    FakeVectorIndex,
)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        self.events.append((event, dict(fields)))


def _active_index(settings: RuntimeSettings, now: datetime) -> ActiveIndexRecord:
    return ActiveIndexRecord(
        index_version=settings.active_index_version,
        pipeline_version=settings.pipeline_version,
        parser_version=settings.parser_version,
        chunker_version=settings.chunker_version,
        embedding_model_id=settings.embedding_model_id,
        embedding_dimensions=settings.embedding_dimensions,
        updated_at=now,
    )


@pytest.fixture
def worker_runtime(
    runtime_values: dict[str, Any],
) -> tuple[
    RuntimeSettings,
    FakeClock,
    FakeMetadataStore,
    FakeObjectStore,
    FakeEmbeddingModel,
    FakeVectorIndex,
    RecordingTelemetry,
    WorkerDependencies,
]:
    values = dict(runtime_values)
    values["CHUNK_SIZE_TOKENS"] = 64
    values["CHUNK_OVERLAP_TOKENS"] = 8
    settings = RuntimeSettings.model_validate(values)
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    metadata = FakeMetadataStore(
        tenant_id=settings.dev_tenant_id,
        active_index=_active_index(settings, clock.now()),
        max_chunks_per_document=settings.max_document_chunks,
    )
    objects = FakeObjectStore()
    embeddings = FakeEmbeddingModel(max_batch_size=settings.max_embedding_batch_size)
    vectors = FakeVectorIndex(max_batch_size=settings.max_embedding_batch_size)
    telemetry = RecordingTelemetry()
    dependencies = WorkerDependencies(
        metadata=metadata,
        objects=objects,
        embeddings=embeddings,
        vectors=vectors,
        clock=clock,
        telemetry=telemetry,
        owner="worker-test",
        sleep=lambda _seconds: None,
        jitter=lambda _attempt: 0.0,
    )
    return (
        settings,
        clock,
        metadata,
        objects,
        embeddings,
        vectors,
        telemetry,
        dependencies,
    )


def _register(
    settings: RuntimeSettings,
    clock: FakeClock,
    metadata: FakeMetadataStore,
    objects: FakeObjectStore,
    *,
    content: bytes = b"alpha beta gamma delta",
    document_id: str = "handbook",
) -> IngestionMessage:
    checksum = hashlib.sha256(content).hexdigest()
    uri = f"gs://{settings.gcs_document_bucket}/{settings.tenant_upload_prefix}/{document_id}.txt"
    result = metadata.register(
        DocumentRegistration(
            tenant_id=settings.dev_tenant_id,
            document_id=document_id,
            corpus=settings.fixed_corpus,
            object_uri=uri,
            object_generation=101,
            checksum_sha256=checksum,
            media_type=MediaType.TEXT,
            pipeline_version=settings.pipeline_version,
        ),
        f"register-{document_id}",
        UUID(int=1),
        clock.now(),
    )
    objects.put(
        ObjectMetadata(
            canonical_uri=uri,
            bucket=settings.gcs_document_bucket,
            name=f"{settings.tenant_upload_prefix}/{document_id}.txt",
            object_generation=101,
            size=len(content),
            content_type=MediaType.TEXT,
            checksum_sha256=checksum,
            updated_at=clock.now(),
        ),
        content,
    )
    assert result.document is not None
    return IngestionMessage(
        operation=IngestionOperation.UPSERT,
        tenant_id=settings.dev_tenant_id,
        document_id=document_id,
        object_uri=uri,
        object_generation=101,
        generation=result.document.generation,
        checksum_sha256=checksum,
        corpus=settings.fixed_corpus,
        pipeline_version=settings.pipeline_version,
        trace_id=f"trace-{document_id}",
        message_metadata=MessageMetadata(
            event_id=UUID(int=2),
            occurred_at=clock.now(),
        ),
    )


def _envelope(payload: IngestionMessage, *, message_id: str = "message-1") -> dict[str, object]:
    return {
        "message": {
            "data": encode_push_data(payload),
            "messageId": message_id,
            "publishTime": "2026-01-01T00:00:00Z",
            "attributes": {
                "contract_version": "1.0.0",
                "operation": payload.operation.value,
                "trace_id": payload.trace_id,
            },
        },
        "subscription": "projects/local/subscriptions/worker",
        "deliveryAttempt": 1,
    }


def _client(
    settings: RuntimeSettings,
    dependencies: WorkerDependencies,
) -> tuple[TestClient, dict[str, str]]:
    return TestClient(create_app(lambda: settings, dependencies)), {}


def _text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)

    target = BytesIO()
    writer.write(target)
    return target.getvalue()


def test_parser_pdf_provenance_utf8_and_bounds() -> None:
    sections = parse_document(
        _text_pdf("Page one"),
        MediaType.PDF,
        max_pdf_pages=1,
        max_chars=100,
    )
    assert sections == [ParsedSection(text="Page one", source_page=1)]
    with pytest.raises(RagError, match="Unprocessable"):
        parse_document(b"\xff", MediaType.TEXT, max_pdf_pages=1, max_chars=10)
    with pytest.raises(RagError, match="Limit"):
        parse_document(
            _text_pdf("too long"),
            MediaType.PDF,
            max_pdf_pages=1,
            max_chars=3,
        )


def test_chunking_is_deterministic_overlapping_and_bounded() -> None:
    def make_chunks(max_chunks: int) -> list[TextChunk]:
        return chunk_sections(
            [ParsedSection(text="one two three four five", source_page=3)],
            tenant_id="dev-tenant",
            document_id="doc",
            generation=2,
            object_generation=9,
            pipeline_version="pipeline-1",
            chunker_version="chunker-1",
            chunk_size_tokens=3,
            overlap_tokens=1,
            max_chunks=max_chunks,
        )

    first = make_chunks(3)
    assert first == make_chunks(3)
    assert [chunk.text for chunk in first] == ["one two three", "three four five"]
    assert all(chunk.source_page == 3 for chunk in first)
    with pytest.raises(RagError, match="Limit"):
        make_chunks(1)


def test_envelope_is_strict_and_matches_attributes() -> None:
    payload = IngestionMessage(
        operation=IngestionOperation.DELETE,
        tenant_id="dev-tenant",
        document_id="doc",
        object_uri="gs://bucket/tenants/dev-tenant/uploads/doc.txt",
        object_generation=1,
        generation=2,
        checksum_sha256="a" * 64,
        corpus="default",
        pipeline_version="pipeline-1",
        trace_id="trace",
        message_metadata=MessageMetadata(
            event_id=UUID(int=1),
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    envelope = _envelope(payload)
    decoded_envelope, decoded = decode_push_envelope(json.dumps(envelope).encode())
    assert decoded == payload
    assert decoded_envelope.delivery_attempt == 1
    wire = cast(dict[str, Any], envelope)
    message = cast(dict[str, Any], wire["message"])
    google_wire = {
        "deliveryAttempt": 2,
        "message": {
            **message,
            "message_id": message["messageId"],
            "publish_time": message["publishTime"],
        },
        "subscription": wire["subscription"],
    }
    decoded_envelope, decoded = decode_push_envelope(json.dumps(google_wire).encode())
    assert decoded == payload
    assert decoded_envelope.delivery_attempt == 2
    message["data"] = "not-valid-base64!!!"
    with pytest.raises(RagError, match="Invalid Request"):
        decode_push_envelope(json.dumps(wire).encode())


def test_upsert_duplicate_is_idempotent_and_content_safe(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, _, vectors, telemetry, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects, content=b"secret alpha beta")
    client, headers = _client(settings, dependencies)
    response = client.post("/v1/pubsub/push", json=_envelope(payload), headers=headers)
    assert response.status_code == 204
    assert metadata.documents[payload.document_id].status is DocumentStatus.READY
    assert len(metadata.chunks) == len(vectors.datapoints) == 1
    duplicate = client.post("/v1/pubsub/push", json=_envelope(payload), headers=headers)
    assert duplicate.status_code == 204
    assert len(metadata.chunks) == len(vectors.datapoints) == 1
    assert "secret alpha beta" not in repr(telemetry.events)


def test_security_mismatches_fail_before_object_or_vector_calls(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, _, vectors, _, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects)
    client, headers = _client(settings, dependencies)
    hostile = payload.model_copy(
        update={"object_uri": "gs://other-bucket/tenants/dev-tenant/uploads/doc.txt"}
    )
    response = client.post("/v1/pubsub/push", json=_envelope(hostile), headers=headers)
    assert response.status_code == 204
    assert objects.calls == []
    assert vectors.calls == []
    assert [name for name, _ in metadata.calls].count("claim_event") == 0


def test_retryable_embedding_failure_returns_non_2xx_and_records_failure(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, embeddings, _, _, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects)
    for _ in range(settings.embedding_max_retries + 1):
        embeddings.fail_next("embed", RetryableDependencyError())
    client, headers = _client(settings, dependencies)
    response = client.post("/v1/pubsub/push", json=_envelope(payload), headers=headers)
    assert response.status_code == 503
    document = metadata.documents[payload.document_id]
    assert document.status is DocumentStatus.FAILED_RETRYABLE
    assert document.failure is not None
    assert document.failure.retryable
    assert payload.object_uri not in response.text


def test_vector_upsert_failure_prevents_ready(worker_runtime: tuple[Any, ...]) -> None:
    settings, clock, metadata, objects, _, vectors, _, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects)
    vectors.fail_next("upsert", RetryableDependencyError())
    client, headers = _client(settings, dependencies)
    response = client.post("/v1/pubsub/push", json=_envelope(payload), headers=headers)
    assert response.status_code == 503
    assert metadata.documents[payload.document_id].status is DocumentStatus.FAILED_RETRYABLE
    assert not any(name == "commit_ready" for name, _ in metadata.calls)


def test_terminal_poison_is_acknowledged_and_durably_failed(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, _, vectors, _, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects, content=b"")
    client, headers = _client(settings, dependencies)
    response = client.post("/v1/pubsub/push", json=_envelope(payload), headers=headers)
    assert response.status_code == 204
    document = metadata.documents[payload.document_id]
    assert document.status is DocumentStatus.FAILED_TERMINAL
    assert document.failure is not None
    assert document.failure.code == ErrorCode.UNPROCESSABLE_DOCUMENT.value
    assert vectors.calls == []


def test_stale_generation_is_acknowledged_before_storage(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, _, vectors, _, dependencies = worker_runtime
    payload = _register(settings, clock, metadata, objects)
    stale = payload.model_copy(update={"generation": payload.generation + 1})
    client, headers = _client(settings, dependencies)
    response = client.post("/v1/pubsub/push", json=_envelope(stale), headers=headers)
    assert response.status_code == 204
    assert objects.calls == []
    assert vectors.calls == []


def test_delete_is_idempotent_and_preserves_tombstone_on_retry(
    worker_runtime: tuple[Any, ...],
) -> None:
    settings, clock, metadata, objects, _, vectors, _, dependencies = worker_runtime
    upsert = _register(settings, clock, metadata, objects)
    client, headers = _client(settings, dependencies)
    upsert_response = client.post("/v1/pubsub/push", json=_envelope(upsert), headers=headers)
    assert upsert_response.status_code == 204
    tombstone = metadata.tombstone(settings.dev_tenant_id, upsert.document_id, clock.now())
    delete = upsert.model_copy(
        update={
            "operation": IngestionOperation.DELETE,
            "generation": tombstone.generation,
            "message_metadata": MessageMetadata(
                event_id=UUID(int=3),
                occurred_at=clock.now(),
            ),
        }
    )
    vectors.fail_next("delete", RetryableDependencyError())
    retry = client.post("/v1/pubsub/push", json=_envelope(delete), headers=headers)
    assert retry.status_code == 503
    assert metadata.documents[delete.document_id].status is DocumentStatus.DELETE_PENDING
    assert metadata.documents[delete.document_id].tombstoned
    success = client.post("/v1/pubsub/push", json=_envelope(delete), headers=headers)
    assert success.status_code == 204
    assert metadata.documents[delete.document_id].status is DocumentStatus.DELETED
    assert metadata.chunks == {}
    assert vectors.datapoints == {}
    duplicate = client.post("/v1/pubsub/push", json=_envelope(delete), headers=headers)
    assert duplicate.status_code == 204
