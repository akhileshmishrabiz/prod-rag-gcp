"""Offline Firestore transaction and authorization behavior tests."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.firestore_store import FirestoreMetadataStore
from packages.rag_core.models import (
    ActiveIndexRecord,
    ChunkRecord,
    DocumentRegistration,
    DocumentStatus,
    IngestionMessage,
    IngestionOperation,
    MediaType,
    MessageMetadata,
    Provenance,
    RegistrationOutcome,
    RetrievalCandidate,
    SanitizedFailure,
    VerificationResult,
    chunk_ids_sha256,
)

NOW = datetime(2025, 1, 1, tzinfo=UTC)
EVENT_ID = UUID("00000000-0000-0000-0000-000000000001")
DELETE_EVENT_ID = UUID("00000000-0000-0000-0000-000000000002")
CHECKSUM = hashlib.sha256(b"content").hexdigest()


class Snapshot:
    def __init__(self, reference: DocumentReference, data: dict[str, Any] | None) -> None:
        self.reference = reference
        self._data = copy.deepcopy(data)
        self.exists = data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._data)


class DocumentReference:
    def __init__(self, client: MemoryFirestore, path: str) -> None:
        self.client = client
        self.path = path
        self.id = path.rsplit("/", 1)[-1]

    def get(self, **_: object) -> Snapshot:
        return Snapshot(self, self.client.records.get(self.path))

    def collection(self, name: str) -> Query:
        return Query(self.client, f"{self.path}/{name}")


class Query:
    def __init__(
        self,
        client: MemoryFirestore,
        path: str,
        *,
        filters: tuple[tuple[str, object], ...] = (),
        orders: tuple[str, ...] = (),
        query_limit: int | None = None,
        after: tuple[int, int, str] | None = None,
    ) -> None:
        self.client = client
        self.path = path
        self.filters = filters
        self.orders = orders
        self.query_limit = query_limit
        self.after = after

    def where(self, field: str, operator: str, value: object) -> Query:
        assert operator == "=="
        return Query(
            self.client,
            self.path,
            filters=(*self.filters, (field, value)),
            orders=self.orders,
            query_limit=self.query_limit,
            after=self.after,
        )

    def order_by(self, field: str) -> Query:
        return Query(
            self.client,
            self.path,
            filters=self.filters,
            orders=(*self.orders, field),
            query_limit=self.query_limit,
            after=self.after,
        )

    def limit(self, value: int) -> Query:
        return Query(
            self.client,
            self.path,
            filters=self.filters,
            orders=self.orders,
            query_limit=value,
            after=self.after,
        )

    def start_after(self, values: dict[str, object]) -> Query:
        return Query(
            self.client,
            self.path,
            filters=self.filters,
            orders=self.orders,
            query_limit=self.query_limit,
            after=(
                cast(int, values["generation"]),
                cast(int, values["ordinal"]),
                str(values["chunk_id"]),
            ),
        )

    def stream(self, **_: object) -> list[Snapshot]:
        prefix = f"{self.path}/"
        expected_parts = len(self.path.split("/")) + 1
        matching = [
            (path, data)
            for path, data in self.client.records.items()
            if path.startswith(prefix) and len(path.split("/")) == expected_parts
        ]
        for field, value in self.filters:
            matching = [(path, data) for path, data in matching if data.get(field) == value]
        for field in reversed(self.orders):
            matching.sort(key=lambda item: item[1][field])
        if self.after is not None:
            matching = [
                (path, data)
                for path, data in matching
                if (
                    int(data["generation"]),
                    int(data["ordinal"]),
                    str(data["chunk_id"]),
                )
                > self.after
            ]
        if self.query_limit is not None:
            matching = matching[: self.query_limit]
        return [Snapshot(DocumentReference(self.client, path), data) for path, data in matching]


class Transaction:
    def __init__(self, client: MemoryFirestore) -> None:
        self.client = client
        self.writes: list[tuple[str, DocumentReference, dict[str, Any] | None]] = []
        self.has_written = False

    def get(self, reference: DocumentReference | Query, **_: object) -> Iterator[Snapshot]:
        if self.has_written:
            raise AssertionError("Firestore transactions require all reads before writes")
        if isinstance(reference, Query):
            return iter(reference.stream())
        return iter((reference.get(),))

    def set(self, reference: DocumentReference, data: dict[str, Any]) -> None:
        self.has_written = True
        self.writes.append(("set", reference, copy.deepcopy(data)))

    def create(self, reference: DocumentReference, data: dict[str, Any]) -> None:
        self.has_written = True
        self.writes.append(("create", reference, copy.deepcopy(data)))

    def delete(self, reference: DocumentReference) -> None:
        self.has_written = True
        self.writes.append(("delete", reference, None))

    def commit(self) -> None:
        for operation, reference, data in self.writes:
            if operation == "create" and reference.path in self.client.records:
                raise AssertionError("create precondition failed")
            if operation == "delete":
                self.client.records.pop(reference.path, None)
            else:
                assert data is not None
                self.client.records[reference.path] = copy.deepcopy(data)
        self.client.commits.append(
            tuple((operation, reference.path) for operation, reference, _ in self.writes)
        )


class MemoryFirestore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.commits: list[tuple[tuple[str, str], ...]] = []
        self.transactions: list[Transaction] = []

    def document(self, path: str) -> DocumentReference:
        return DocumentReference(self, path)

    def collection(self, path: str) -> Query:
        return Query(self, path)

    def transaction(self, *, max_attempts: int) -> Transaction:
        assert max_attempts == 3
        transaction = Transaction(self)
        self.transactions.append(transaction)
        return transaction

    def get_all(self, references: list[DocumentReference], **_: object) -> list[Snapshot]:
        return [reference.get() for reference in references]


def run_transaction(transaction: Transaction, callback: Any) -> Any:
    result = callback(transaction)
    transaction.commit()
    return result


def registration(document_id: str = "doc-1") -> DocumentRegistration:
    return DocumentRegistration(
        tenant_id="dev-tenant",
        document_id=document_id,
        corpus="docs",
        object_uri=f"gs://bucket/tenants/dev-tenant/{document_id}.txt",
        object_generation=7,
        checksum_sha256=CHECKSUM,
        media_type=MediaType.TEXT,
        pipeline_version="pipeline-1",
    )


def message(
    generation: int,
    *,
    event_id: UUID = EVENT_ID,
    operation: IngestionOperation = IngestionOperation.UPSERT,
) -> IngestionMessage:
    return IngestionMessage(
        operation=operation,
        tenant_id="dev-tenant",
        document_id="doc-1",
        object_uri="gs://bucket/tenants/dev-tenant/doc-1.txt",
        object_generation=7,
        generation=generation,
        checksum_sha256=CHECKSUM,
        corpus="docs",
        pipeline_version="pipeline-1",
        trace_id="trace",
        message_metadata=MessageMetadata(event_id=event_id, occurred_at=NOW),
    )


def chunk(document_id: str = "doc-1", generation: int = 1) -> ChunkRecord:
    return ChunkRecord(
        tenant_id="dev-tenant",
        corpus="docs",
        document_id=document_id,
        generation=generation,
        object_generation=7,
        chunk_id="chunk-1",
        ordinal=0,
        text="authorized",
        text_sha256=hashlib.sha256(b"authorized").hexdigest(),
        vector_id="chunk-1",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        staged_at=NOW,
        created_at=NOW,
    )


def provenance(*, embedding_model_id: str = "embedding-001") -> Provenance:
    return Provenance(
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model_id=embedding_model_id,
        embedding_dimensions=2,
        active_index_version="index-1",
    )


def verification(count: int = 1) -> VerificationResult:
    return VerificationResult(
        verified=True,
        expected_count=count,
        checked_vector_ids=tuple(f"chunk-{index + 1}" for index in range(count)),
    )


def adapter(*, maximum: int = 10) -> tuple[FirestoreMetadataStore, MemoryFirestore]:
    client = MemoryFirestore()
    client.records["config/active_index"] = ActiveIndexRecord(
        index_version="index-1",
        pipeline_version="pipeline-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        embedding_model_id="embedding-001",
        embedding_dimensions=2,
        updated_at=NOW,
    ).model_dump(mode="python", exclude_none=True)
    store = FirestoreMetadataStore(
        client=client,
        tenant_id="dev-tenant",
        dependency_timeout=2,
        max_active_generations=maximum,
        transaction_runner=run_transaction,
    )
    return store, client


def make_ready(store: FirestoreMetadataStore) -> None:
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    expected_hash = chunk_ids_sha256(["chunk-1"])
    store.stage_chunks(claimed.claim, [chunk()], expected_hash, 1, NOW)
    store.commit_ready(
        claimed.claim,
        ["chunk-1"],
        1,
        provenance(),
        verification(),
        NOW,
    )


def test_document_transaction_reads_require_exactly_one_generated_snapshot() -> None:
    store, client = adapter()
    reference = client.document("tenants/dev-tenant/documents/doc-1")
    faithful = Transaction(client)
    snapshot = store._get(faithful, reference)
    assert snapshot.reference.path == reference.path
    assert not snapshot.exists

    class InvalidTransaction:
        def __init__(self, result: object) -> None:
            self.result = result

        def get(self, _reference: object, **_: object) -> object:
            return self.result

    for result in (
        reference.get(),
        iter(()),
        iter((reference.get(), reference.get())),
        iter((object(),)),
    ):
        with pytest.raises(RagError) as rejected:
            store._get(InvalidTransaction(result), reference)
        assert rejected.value.code is ErrorCode.DEPENDENCY_REJECTED

    query = client.collection("tenants/dev-tenant/documents")
    assert store._get_query(faithful, query) == []
    for query_result in (object(), iter((object(),))):
        with pytest.raises(RagError) as rejected:
            store._get_query(InvalidTransaction(query_result), query)
        assert rejected.value.code is ErrorCode.DEPENDENCY_REJECTED


def test_registration_replay_publish_and_conflict_are_transactional() -> None:
    store, client = adapter()
    first = store.register(registration(), "key", EVENT_ID, NOW)
    assert first.document is not None
    commit_count = len(client.commits)
    replay = store.register(registration(), "key", UUID(int=99), NOW + timedelta(seconds=1))
    assert replay == first
    assert replay.idempotency.event_id == EVENT_ID
    assert len(client.commits) == commit_count + 1
    assert client.commits[-1] == ()
    changed = registration().model_copy(update={"corpus": "other"})
    with pytest.raises(RagError) as conflict:
        store.register(changed, "key", EVENT_ID, NOW)
    assert conflict.value.code is ErrorCode.DOCUMENT_CONFLICT
    assert store.get_document("dev-tenant", "doc-1") == first.document
    published = store.mark_registration_published(
        "dev-tenant",
        first.idempotency.idempotency_key_sha256,
        first.idempotency.request_sha256,
        EVENT_ID,
        NOW + timedelta(seconds=2),
    )
    assert published.outcome is RegistrationOutcome.PUBLISHED
    assert (
        store.mark_registration_published(
            "dev-tenant",
            published.idempotency_key_sha256,
            published.request_sha256,
            EVENT_ID,
            NOW + timedelta(seconds=3),
        )
        == published
    )
    with pytest.raises(RagError):
        store.mark_registration_published(
            "dev-tenant",
            published.idempotency_key_sha256,
            "0" * 64,
            EVENT_ID,
            NOW,
        )


def test_stale_generation_tombstone_and_lease_comparisons_fail_closed() -> None:
    store, _ = adapter()
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker-a", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    with pytest.raises(RagError) as contended:
        store.claim_event(message(1), "worker-b", NOW + timedelta(minutes=2), NOW)
    assert contended.value.code is ErrorCode.LEASE_CONTENDED
    wrong_owner = claimed.claim.model_copy(update={"owner": "worker-b"})
    failure = SanitizedFailure(
        code="DEPENDENCY_UNAVAILABLE", retryable=True, attempts=1, last_failed_at=NOW
    )
    with pytest.raises(RagError) as lease_mismatch:
        store.commit_retryable_failure(wrong_owner, failure, NOW)
    assert lease_mismatch.value.code is ErrorCode.STALE_GENERATION
    tombstone = store.tombstone("dev-tenant", "doc-1", NOW + timedelta(minutes=2))
    assert tombstone.tombstoned
    with pytest.raises(RagError) as stale:
        store.commit_ready(
            claimed.claim,
            ["chunk-1"],
            1,
            provenance(),
            verification(),
            NOW,
        )
    assert stale.value.code is ErrorCode.STALE_GENERATION
    with pytest.raises(RagError):
        store.claim_event(message(1), "worker", NOW + timedelta(minutes=3), NOW)


def test_active_listing_overflow_and_hydration_reauthorization() -> None:
    store, client = adapter(maximum=2)
    make_ready(store)
    assert store.list_active_generations("dev-tenant", "docs", 2) == [("doc-1", 1)]
    candidate = RetrievalCandidate(
        vector_id="chunk-1",
        score=1,
        document_id="doc-1",
        generation=1,
        chunk_id="chunk-1",
    )
    assert store.hydrate_authorized("dev-tenant", "docs", [candidate], "index-1") == [chunk()]
    for document_id in ("doc-2", "doc-3"):
        document = store.register(
            registration(document_id),
            f"key-{document_id}",
            UUID(int=int(document_id[-1]) + 10),
            NOW,
        ).document
        assert document is not None
        client.records[f"tenants/dev-tenant/documents/{document_id}"] = document.model_copy(
            update={
                "status": DocumentStatus.READY,
                "chunk_count": 1,
                "expected_chunk_ids_sha256": chunk_ids_sha256(["chunk-1"]),
                "parser_version": "parser-1",
                "chunker_version": "chunker-1",
                "embedding_model_id": "embedding-001",
                "embedding_dimensions": 2,
                "active_index_version": "index-1",
                "ready_at": NOW,
            }
        ).model_dump(mode="python", exclude_none=True)
    with pytest.raises(RagError) as overflow:
        store.list_active_generations("dev-tenant", "docs", 2)
    assert overflow.value.code is ErrorCode.LIMIT_EXCEEDED
    client.records["tenants/dev-tenant/documents/doc-1"]["tombstoned"] = True
    client.records["tenants/dev-tenant/documents/doc-1"]["status"] = "DELETE_PENDING"
    client.records["tenants/dev-tenant/documents/doc-1"]["deleted_at"] = NOW
    assert store.hydrate_authorized("dev-tenant", "docs", [candidate], "index-1") == []


def test_delete_commit_removes_chunks_and_completes_event() -> None:
    store, client = adapter()
    make_ready(store)
    tombstone = store.tombstone("dev-tenant", "doc-1", NOW + timedelta(minutes=1))
    claimed = store.claim_event(
        message(
            tombstone.generation,
            event_id=DELETE_EVENT_ID,
            operation=IngestionOperation.DELETE,
        ),
        "worker",
        NOW + timedelta(minutes=2),
        NOW + timedelta(minutes=1),
    )
    assert claimed.claim is not None
    deleted = store.commit_deleted(claimed.claim, NOW + timedelta(minutes=1))
    assert deleted.status is DocumentStatus.DELETED
    assert not any("/chunks/" in path for path in client.records)


def test_failure_commits_duplicate_dispositions_and_expired_lease() -> None:
    store, _ = adapter()
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    failure = SanitizedFailure(
        code="DEPENDENCY_UNAVAILABLE", retryable=True, attempts=1, last_failed_at=NOW
    )
    store.commit_retryable_failure(claimed.claim, failure, NOW)
    retried = store.claim_event(
        message(1), "worker-2", NOW + timedelta(minutes=2), NOW + timedelta(minutes=1)
    )
    assert retried.claim is not None
    with pytest.raises(RagError) as expired:
        store.commit_terminal_failure(retried.claim, failure, NOW + timedelta(minutes=3))
    assert expired.value.code is ErrorCode.STALE_GENERATION
    store.commit_terminal_failure(retried.claim, failure, NOW + timedelta(minutes=1))
    duplicate = store.claim_event(
        message(1), "worker-3", NOW + timedelta(minutes=4), NOW + timedelta(minutes=3)
    )
    assert duplicate.disposition.value == "DUPLICATE_TERMINAL"


def test_adapter_rejects_invalid_inputs_missing_records_and_incompatible_ready() -> None:
    client = MemoryFirestore()
    with pytest.raises(ValueError, match="invalid"):
        FirestoreMetadataStore(
            client=client,
            tenant_id="",
            dependency_timeout=0,
            transaction_runner=run_transaction,
        )
    store, client = adapter(maximum=2)
    with pytest.raises(RagError):
        store.register(registration(), "", EVENT_ID, NOW)
    assert store.get_document("dev-tenant", "missing") is None
    with pytest.raises(RagError) as missing:
        store.tombstone("dev-tenant", "missing", NOW)
    assert missing.value.code is ErrorCode.DOCUMENT_NOT_FOUND
    with pytest.raises(RagError):
        store.claim_event(message(1), "", NOW, NOW)
    with pytest.raises(RagError):
        store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    store.register(registration(), "key", EVENT_ID, NOW)
    wrong_generation = message(2)
    with pytest.raises(RagError) as stale:
        store.claim_event(wrong_generation, "worker", NOW + timedelta(minutes=1), NOW)
    assert stale.value.code is ErrorCode.STALE_GENERATION
    wrong_checksum = message(1).model_copy(update={"checksum_sha256": "0" * 64})
    with pytest.raises(RagError) as conflict:
        store.claim_event(wrong_checksum, "worker", NOW + timedelta(minutes=1), NOW)
    assert conflict.value.code is ErrorCode.DOCUMENT_CONFLICT
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    expected_hash = chunk_ids_sha256(["chunk-1"])
    store.stage_chunks(claimed.claim, [chunk()], expected_hash, 1, NOW)
    with pytest.raises(RagError) as incompatible:
        store.commit_ready(
            claimed.claim,
            ["chunk-1"],
            1,
            provenance(embedding_model_id="wrong-model"),
            verification(),
            NOW,
        )
    assert incompatible.value.code is ErrorCode.DEPENDENCY_REJECTED
    invalid_chunk = chunk().model_copy(update={"tenant_id": "other"})
    with pytest.raises(RagError) as invalid:
        store.stage_chunks(
            claimed.claim,
            [invalid_chunk],
            expected_hash,
            1,
            NOW,
        )
    assert invalid.value.code is ErrorCode.INVALID_REQUEST
    with pytest.raises(RagError):
        store.list_active_generations("dev-tenant", "docs", 3)
    with pytest.raises(RagError):
        store.hydrate_authorized(
            "dev-tenant",
            "docs",
            [
                RetrievalCandidate(
                    vector_id="chunk-1",
                    score=1,
                    document_id="doc-1",
                    generation=1,
                    chunk_id="chunk-1",
                )
            ],
            "wrong-index",
        )
    del client.records["config/active_index"]
    with pytest.raises(RagError) as no_index:
        store.get_active_index()
    assert no_index.value.code is ErrorCode.CONFIGURATION_INVALID


def test_registration_replay_after_new_generation_returns_durable_result() -> None:
    store, _ = adapter()
    first = store.register(registration(), "first-key", EVENT_ID, NOW)
    store.register(
        registration().model_copy(update={"checksum_sha256": "0" * 64}),
        "second-key",
        UUID(int=3),
        NOW + timedelta(seconds=1),
    )
    replay = store.register(registration(), "first-key", UUID(int=99), NOW)
    assert replay.document is None
    assert replay.idempotency.generation == first.idempotency.generation
    assert replay.idempotency.event_id == EVENT_ID


def test_firestore_edge_paths_remain_fail_closed() -> None:
    store, client = adapter(maximum=1)
    result = store.register(registration(), "key", EVENT_ID, NOW)
    assert result.document is not None
    assert store.tombstone("dev-tenant", "doc-1", NOW) == store.tombstone(
        "dev-tenant", "doc-1", NOW
    )
    missing_hash = "f" * 64
    with pytest.raises(RagError):
        store.mark_registration_published("dev-tenant", missing_hash, missing_hash, EVENT_ID, NOW)
    assert store.hydrate_authorized("dev-tenant", "docs", [], "index-1") == []
    candidates = [
        RetrievalCandidate(
            vector_id="missing",
            score=1,
            document_id="doc-1",
            generation=1,
            chunk_id="missing",
        ),
        RetrievalCandidate(
            vector_id="other",
            score=0,
            document_id="doc-1",
            generation=1,
            chunk_id="other",
        ),
    ]
    with pytest.raises(RagError) as bounded:
        store.hydrate_authorized("dev-tenant", "docs", candidates, "index-1")
    assert bounded.value.code is ErrorCode.LIMIT_EXCEEDED
    client.records.pop("tenants/dev-tenant/documents/doc-1")
    with pytest.raises(RagError) as missing_document:
        store.register(registration(), "key", EVENT_ID, NOW)
    assert missing_document.value.code is ErrorCode.DEPENDENCY_REJECTED


def test_transaction_errors_and_commit_bounds_are_classified() -> None:
    client = MemoryFirestore()

    def timeout_runner(_: Transaction, __: Any) -> Any:
        raise TimeoutError

    timeout_store = FirestoreMetadataStore(
        client=client,
        tenant_id="dev-tenant",
        dependency_timeout=1,
        transaction_runner=timeout_runner,
    )
    with pytest.raises(RagError) as timeout:
        timeout_store.register(registration(), "key", EVENT_ID, NOW)
    assert timeout.value.code is ErrorCode.DEPENDENCY_TIMEOUT

    class AbortedError(Exception):
        pass

    aborted = AbortedError()
    aborted.__class__.__name__ = "Aborted"

    def aborted_runner(_: Transaction, __: Any) -> Any:
        raise aborted

    aborted_store = FirestoreMetadataStore(
        client=client,
        tenant_id="dev-tenant",
        dependency_timeout=1,
        transaction_runner=aborted_runner,
    )
    with pytest.raises(RagError) as unavailable:
        aborted_store.register(registration(), "key", EVENT_ID, NOW)
    assert unavailable.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE

    store, _ = adapter()
    store.max_chunks_per_document = 1
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    with pytest.raises(RagError) as too_many:
        store.commit_ready(
            claimed.claim,
            ["chunk-1", "chunk-2"],
            2,
            provenance(),
            verification(2),
            NOW,
        )
    assert too_many.value.code is ErrorCode.LIMIT_EXCEEDED


def test_staging_is_idempotent_conflict_safe_and_required_for_ready() -> None:
    store, _ = adapter()
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    expected_hash = chunk_ids_sha256(["chunk-1"])
    first = store.stage_chunks(claimed.claim, [chunk()], expected_hash, 1, NOW)
    assert store.stage_chunks(claimed.claim, [chunk()], expected_hash, 1, NOW) == first
    changed = chunk().model_copy(
        update={"text": "changed", "text_sha256": hashlib.sha256(b"changed").hexdigest()}
    )
    with pytest.raises(RagError) as conflict:
        store.stage_chunks(claimed.claim, [changed], expected_hash, 1, NOW)
    assert conflict.value.code is ErrorCode.DOCUMENT_CONFLICT
    with pytest.raises(RagError):
        store.commit_ready(
            claimed.claim,
            ["chunk-1", "missing"],
            2,
            provenance(),
            verification(2),
            NOW,
        )
    processing = store.get_document("dev-tenant", "doc-1")
    assert processing is not None
    assert processing.status is DocumentStatus.PROCESSING
    ready = store.commit_ready(
        claimed.claim,
        ["chunk-1"],
        1,
        provenance(),
        verification(),
        NOW,
    )
    assert ready.ready_at == NOW


def test_chunk_paging_is_bounded_and_orders_all_generations() -> None:
    store, _ = adapter()
    store.register(registration(), "key", EVENT_ID, NOW)
    claimed = store.claim_event(message(1), "worker", NOW + timedelta(minutes=1), NOW)
    assert claimed.claim is not None
    second = chunk().model_copy(
        update={
            "chunk_id": "chunk-2",
            "ordinal": 1,
            "vector_id": "vector-2",
            "text": "second",
            "text_sha256": hashlib.sha256(b"second").hexdigest(),
        }
    )
    expected_hash = chunk_ids_sha256(["chunk-1", "chunk-2"])
    store.stage_chunks(claimed.claim, [second, chunk()], expected_hash, 2, NOW)
    first_page = store.list_document_chunks("dev-tenant", "doc-1", None, 1)
    assert [item.chunk_id for item in first_page.items] == ["chunk-1"]
    assert first_page.next_cursor is not None
    second_page = store.list_document_chunks("dev-tenant", "doc-1", first_page.next_cursor, 1)
    assert [item.vector_id for item in second_page.items] == ["vector-2"]
    assert second_page.next_cursor is None
    with pytest.raises(RagError):
        store.list_document_chunks("dev-tenant", "doc-1", "not-base64", 1)


def test_retryable_delete_failure_preserves_tombstone_and_deleted_at() -> None:
    store, _ = adapter()
    make_ready(store)
    tombstone = store.tombstone("dev-tenant", "doc-1", NOW + timedelta(minutes=1))
    claimed = store.claim_event(
        message(
            tombstone.generation,
            event_id=DELETE_EVENT_ID,
            operation=IngestionOperation.DELETE,
        ),
        "worker",
        NOW + timedelta(minutes=2),
        NOW + timedelta(minutes=1),
    )
    assert claimed.claim is not None
    failure = SanitizedFailure(
        code="DEPENDENCY_UNAVAILABLE",
        retryable=True,
        attempts=1,
        last_failed_at=NOW,
    )
    preserved = store.record_retryable_delete_failure(
        claimed.claim, failure, NOW + timedelta(minutes=1)
    )
    assert preserved.status is DocumentStatus.DELETE_PENDING
    assert preserved.tombstoned
    assert preserved.deleted_at == tombstone.deleted_at
    assert preserved.lease is None
    assert (
        store.hydrate_authorized(
            "dev-tenant",
            "docs",
            [
                RetrievalCandidate(
                    vector_id="chunk-1",
                    score=1,
                    document_id="doc-1",
                    generation=1,
                    chunk_id="chunk-1",
                )
            ],
            "index-1",
        )
        == []
    )
