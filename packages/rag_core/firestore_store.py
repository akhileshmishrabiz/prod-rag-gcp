"""Metadata transaction port and concrete client-injected Firestore adapter."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from itertools import islice
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID

from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import (
    ActiveIndexRecord,
    ChunkPage,
    ChunkPointer,
    ChunkRecord,
    Claim,
    ClaimDisposition,
    ClaimResult,
    DocumentRecord,
    DocumentRegistration,
    DocumentStatus,
    IdempotencyOutcome,
    IdempotencyRecord,
    IngestionMessage,
    IngestionOperation,
    Lease,
    Provenance,
    RegistrationIdempotencyRecord,
    RegistrationOutcome,
    RegistrationResult,
    RetrievalCandidate,
    SanitizedFailure,
    StagedChunkResult,
    VerificationResult,
    chunk_ids_sha256,
)
from packages.rag_core.policy import require_tenant

_T = TypeVar("_T")
_TransactionCallback = Callable[[Any], _T]
_TransactionRunner = Callable[[Any, Callable[[Any], Any]], Any]


class MetadataStore(Protocol):
    """Atomic, generation-aware metadata operations."""

    def register(
        self,
        registration: DocumentRegistration,
        idempotency_key: str,
        event_id: UUID,
        now: datetime,
    ) -> RegistrationResult: ...

    def mark_registration_published(
        self,
        tenant_id: str,
        idempotency_key_sha256: str,
        request_sha256: str,
        event_id: UUID,
        published_at: datetime,
    ) -> RegistrationIdempotencyRecord: ...

    def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None: ...

    def tombstone(self, tenant_id: str, document_id: str, now: datetime) -> DocumentRecord: ...

    def claim_event(
        self,
        payload: IngestionMessage,
        owner: str,
        lease_until: datetime,
        now: datetime,
    ) -> ClaimResult: ...

    def commit_ready(
        self,
        claim: Claim,
        expected_chunk_ids: Sequence[str],
        expected_chunk_count: int,
        provenance: Provenance,
        vector_verification: VerificationResult,
        now: datetime,
    ) -> DocumentRecord: ...

    def stage_chunks(
        self,
        claim: Claim,
        chunks: Sequence[ChunkRecord],
        expected_chunk_ids_sha256: str,
        expected_chunk_count: int,
        now: datetime,
    ) -> StagedChunkResult: ...

    def commit_retryable_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None: ...

    def record_retryable_delete_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> DocumentRecord: ...

    def commit_terminal_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None: ...

    def commit_deleted(self, claim: Claim, now: datetime) -> DocumentRecord: ...

    def list_document_chunks(
        self, tenant_id: str, document_id: str, after: str | None, limit: int
    ) -> ChunkPage: ...

    def hydrate_authorized(
        self,
        tenant_id: str,
        corpus: str,
        candidates: Sequence[RetrievalCandidate],
        active_index_version: str,
    ) -> list[ChunkRecord]: ...

    def get_active_index(self) -> ActiveIndexRecord: ...

    def list_active_generations(
        self, tenant_id: str, corpus: str, limit: int
    ) -> list[tuple[str, int]]: ...


class FirestoreMetadataStore:
    """Firestore implementation with all compare-and-set writes in transactions."""

    def __init__(
        self,
        *,
        client: Any,
        tenant_id: str,
        dependency_timeout: float,
        event_idempotency_ttl: timedelta = timedelta(days=7),
        max_active_generations: int = 1000,
        max_chunks_per_document: int = 10_000,
        max_stage_batch_size: int = 450,
        transaction_runner: _TransactionRunner | None = None,
    ) -> None:
        if (
            not tenant_id
            or dependency_timeout <= 0
            or event_idempotency_ttl <= timedelta(0)
            or max_active_generations < 1
            or max_chunks_per_document < 1
            or max_stage_batch_size < 1
            or max_stage_batch_size > 450
        ):
            raise ValueError("invalid Firestore adapter configuration")
        self.client = client
        self.tenant_id = tenant_id
        self.timeout = dependency_timeout
        self.event_idempotency_ttl = event_idempotency_ttl
        self.max_active_generations = max_active_generations
        self.max_chunks_per_document = max_chunks_per_document
        self.max_stage_batch_size = max_stage_batch_size
        self._transaction_runner = transaction_runner or self._official_transaction_runner

    @staticmethod  # pragma: no cover - official SDK decorator integration
    def _official_transaction_runner(transaction: Any, callback: _TransactionCallback[_T]) -> _T:
        from google.cloud.firestore_v1 import transactional

        result: _T = transactional(callback)(transaction)
        return result

    def _run_transaction(self, callback: _TransactionCallback[_T]) -> _T:
        transaction = self.client.transaction(max_attempts=3)
        try:
            return cast(_T, self._transaction_runner(transaction, callback))
        except RagError:
            raise
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            if error.__class__.__name__ in {"Aborted", "DeadlineExceeded", "ServiceUnavailable"}:
                raise RetryableDependencyError() from error
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error

    def _document_ref(self, document_id: str) -> Any:
        return self.client.document(f"tenants/{self.tenant_id}/documents/{document_id}")

    def _registration_ref(self, key_sha256: str) -> Any:
        return self.client.document(
            f"tenants/{self.tenant_id}/document_registration_idempotency/{key_sha256}"
        )

    def _event_ref(self, event_id: UUID) -> Any:
        return self.client.document(f"tenants/{self.tenant_id}/event_idempotency/{event_id}")

    def _chunk_ref(self, document_id: str, generation: int, chunk_id: str) -> Any:
        return self.client.document(
            f"tenants/{self.tenant_id}/documents/{document_id}/chunks/{generation}:{chunk_id}"
        )

    def _active_index_ref(self) -> Any:
        return self.client.document("config/active_index")

    def _get(self, transaction: Any, reference: Any) -> Any:
        result = transaction.get(reference, retry=None, timeout=self.timeout)
        try:
            snapshots = list(islice(iter(result), 2))
        except TypeError as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error
        if len(snapshots) != 1:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        snapshot = snapshots[0]
        if (
            not hasattr(snapshot, "exists")
            or not callable(getattr(snapshot, "to_dict", None))
            or getattr(getattr(snapshot, "reference", None), "path", None)
            != getattr(reference, "path", None)
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return snapshot

    def _get_query(self, transaction: Any, query: Any) -> list[Any]:
        result = transaction.get(query, retry=None, timeout=self.timeout)
        try:
            snapshots = list(iter(result))
        except TypeError as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error
        if any(
            not hasattr(snapshot, "exists")
            or not callable(getattr(snapshot, "to_dict", None))
            or getattr(getattr(snapshot, "reference", None), "path", None) is None
            for snapshot in snapshots
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return snapshots

    @staticmethod
    def _data(snapshot: Any) -> dict[str, Any] | None:
        if not snapshot.exists:
            return None
        value = snapshot.to_dict()
        return dict(value) if value is not None else None

    @staticmethod
    def _dump(model: Any) -> dict[str, Any]:
        def firestore_value(value: Any) -> Any:
            if isinstance(value, UUID):
                return str(value)
            if isinstance(value, dict):
                return {key: firestore_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [firestore_value(item) for item in value]
            return value

        return cast(
            dict[str, Any],
            firestore_value(model.model_dump(mode="python", exclude_none=True)),
        )

    @staticmethod
    def _registration_hash(registration: DocumentRegistration) -> str:
        canonical = {
            "checksum_sha256": registration.checksum_sha256,
            "corpus": registration.corpus,
            "document_id": registration.document_id,
            "media_type": registration.media_type.value,
            "object_generation": registration.object_generation,
            "object_uri": registration.object_uri,
        }
        encoded = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def register(
        self,
        registration: DocumentRegistration,
        idempotency_key: str,
        event_id: UUID,
        now: datetime,
    ) -> RegistrationResult:
        require_tenant(registration.tenant_id, self.tenant_id)
        if not idempotency_key:
            raise RagError(ErrorCode.INVALID_REQUEST)
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        request_hash = self._registration_hash(registration)
        registration_ref = self._registration_ref(key_hash)
        document_ref = self._document_ref(registration.document_id)

        def operation(transaction: Any) -> RegistrationResult:
            idempotency_data = self._data(self._get(transaction, registration_ref))
            document_data = self._data(self._get(transaction, document_ref))
            if idempotency_data is not None:
                stored = RegistrationIdempotencyRecord.model_validate(idempotency_data)
                if (
                    stored.tenant_id != self.tenant_id
                    or stored.idempotency_key_sha256 != key_hash
                    or stored.request_sha256 != request_hash
                    or stored.document_id != registration.document_id
                ):
                    raise RagError(ErrorCode.DOCUMENT_CONFLICT)
                if document_data is None:
                    raise RagError(ErrorCode.DEPENDENCY_REJECTED)
                document = DocumentRecord.model_validate(document_data)
                if document.generation != stored.generation:
                    return RegistrationResult(idempotency=stored)
                return RegistrationResult(idempotency=stored, document=document)

            current = (
                DocumentRecord.model_validate(document_data) if document_data is not None else None
            )
            generation = 1 if current is None else current.generation + 1
            document = DocumentRecord(
                **registration.model_dump(),
                generation=generation,
                status=DocumentStatus.PENDING,
                tombstoned=False,
                created_at=current.created_at if current else now,
                updated_at=now,
            )
            idempotency = RegistrationIdempotencyRecord(
                tenant_id=self.tenant_id,
                idempotency_key_sha256=key_hash,
                request_sha256=request_hash,
                document_id=registration.document_id,
                generation=generation,
                event_id=event_id,
                outcome=RegistrationOutcome.COMMITTED,
                created_at=now,
                updated_at=now,
            )
            transaction.set(document_ref, self._dump(document))
            transaction.create(registration_ref, self._dump(idempotency))
            return RegistrationResult(document=document, idempotency=idempotency)

        return self._run_transaction(operation)

    def mark_registration_published(
        self,
        tenant_id: str,
        idempotency_key_sha256: str,
        request_sha256: str,
        event_id: UUID,
        published_at: datetime,
    ) -> RegistrationIdempotencyRecord:
        require_tenant(tenant_id, self.tenant_id)
        reference = self._registration_ref(idempotency_key_sha256)

        def operation(transaction: Any) -> RegistrationIdempotencyRecord:
            data = self._data(self._get(transaction, reference))
            if data is None:
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            current = RegistrationIdempotencyRecord.model_validate(data)
            if (
                current.idempotency_key_sha256 != idempotency_key_sha256
                or current.request_sha256 != request_sha256
                or current.event_id != event_id
            ):
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            if current.outcome is RegistrationOutcome.PUBLISHED:
                return current
            published = current.model_copy(
                update={
                    "outcome": RegistrationOutcome.PUBLISHED,
                    "updated_at": published_at,
                    "published_at": published_at,
                }
            )
            published = RegistrationIdempotencyRecord.model_validate(published.model_dump())
            transaction.set(reference, self._dump(published))
            return published

        return self._run_transaction(operation)

    def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None:
        require_tenant(tenant_id, self.tenant_id)
        snapshot = self._document_ref(document_id).get(retry=None, timeout=self.timeout)
        data = self._data(snapshot)
        return DocumentRecord.model_validate(data) if data is not None else None

    def tombstone(self, tenant_id: str, document_id: str, now: datetime) -> DocumentRecord:
        require_tenant(tenant_id, self.tenant_id)
        reference = self._document_ref(document_id)

        def operation(transaction: Any) -> DocumentRecord:
            data = self._data(self._get(transaction, reference))
            if data is None:
                raise RagError(ErrorCode.DOCUMENT_NOT_FOUND)
            current = DocumentRecord.model_validate(data)
            if current.status in {DocumentStatus.DELETE_PENDING, DocumentStatus.DELETED}:
                return current
            tombstone = current.model_copy(
                update={
                    "generation": current.generation + 1,
                    "status": DocumentStatus.DELETE_PENDING,
                    "tombstoned": True,
                    "deleted_at": now,
                    "updated_at": now,
                    "lease": None,
                    "failure": None,
                }
            )
            tombstone = DocumentRecord.model_validate(tombstone.model_dump())
            transaction.set(reference, self._dump(tombstone))
            return tombstone

        return self._run_transaction(operation)

    def claim_event(
        self,
        payload: IngestionMessage,
        owner: str,
        lease_until: datetime,
        now: datetime,
    ) -> ClaimResult:
        require_tenant(payload.tenant_id, self.tenant_id)
        if not owner or lease_until <= now:
            raise RagError(ErrorCode.INVALID_REQUEST)
        document_ref = self._document_ref(payload.document_id)
        event_ref = self._event_ref(payload.message_metadata.event_id)
        payload_hash = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()

        def operation(transaction: Any) -> ClaimResult:
            event_data = self._data(self._get(transaction, event_ref))
            document_data = self._data(self._get(transaction, document_ref))
            if document_data is None:
                raise RagError(ErrorCode.DOCUMENT_NOT_FOUND)
            document = DocumentRecord.model_validate(document_data)
            if payload.generation != document.generation:
                raise RagError(ErrorCode.STALE_GENERATION)
            if (
                payload.corpus != document.corpus
                or payload.object_generation != document.object_generation
                or payload.checksum_sha256 != document.checksum_sha256
                or payload.pipeline_version != document.pipeline_version
            ):
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            event = IdempotencyRecord.model_validate(event_data) if event_data is not None else None
            if event is not None:
                if (
                    event.tenant_id != self.tenant_id
                    or event.payload_sha256 != payload_hash
                    or event.document_id != payload.document_id
                    or event.generation != payload.generation
                    or event.operation is not payload.operation
                ):
                    raise RagError(ErrorCode.DOCUMENT_CONFLICT)
                if event.outcome is IdempotencyOutcome.SUCCEEDED:
                    return ClaimResult(disposition=ClaimDisposition.DUPLICATE_SUCCESS)
                if event.outcome is IdempotencyOutcome.TERMINAL_FAILURE:
                    return ClaimResult(disposition=ClaimDisposition.DUPLICATE_TERMINAL)
            if document.lease is not None and document.lease.expires_at > now:
                raise RagError(ErrorCode.LEASE_CONTENDED)
            if payload.operation is IngestionOperation.UPSERT:
                if document.tombstoned or document.status not in {
                    DocumentStatus.PENDING,
                    DocumentStatus.FAILED_RETRYABLE,
                    DocumentStatus.PROCESSING,
                }:
                    raise RagError(ErrorCode.STALE_GENERATION)
                status = DocumentStatus.PROCESSING
            else:
                if document.status is not DocumentStatus.DELETE_PENDING or not document.tombstoned:
                    raise RagError(ErrorCode.STALE_GENERATION)
                status = DocumentStatus.DELETE_PENDING
            lease = Lease(owner=owner, expires_at=lease_until, generation=document.generation)
            claimed_document = document.model_copy(
                update={"status": status, "lease": lease, "updated_at": now, "failure": None}
            )
            claimed_document = DocumentRecord.model_validate(claimed_document.model_dump())
            claim = Claim(
                tenant_id=self.tenant_id,
                event_id=payload.message_metadata.event_id,
                operation=payload.operation,
                document_id=payload.document_id,
                generation=payload.generation,
                owner=owner,
                lease_expires_at=lease_until,
            )
            if event is None:
                event = IdempotencyRecord(
                    tenant_id=self.tenant_id,
                    event_id=payload.message_metadata.event_id,
                    operation=payload.operation,
                    document_id=payload.document_id,
                    generation=payload.generation,
                    payload_sha256=payload_hash,
                    outcome=IdempotencyOutcome.IN_PROGRESS,
                    created_at=now,
                    updated_at=now,
                    expires_at=now + self.event_idempotency_ttl,
                )
                transaction.create(event_ref, self._dump(event))
            else:
                transaction.set(
                    event_ref,
                    self._dump(event.model_copy(update={"updated_at": now})),
                )
            transaction.set(document_ref, self._dump(claimed_document))
            return ClaimResult(disposition=ClaimDisposition.CLAIMED, claim=claim)

        return self._run_transaction(operation)

    def _claimed_records(
        self, transaction: Any, claim: Claim, now: datetime
    ) -> tuple[DocumentRecord, IdempotencyRecord, Any, Any]:
        require_tenant(claim.tenant_id, self.tenant_id)
        document_ref = self._document_ref(claim.document_id)
        event_ref = self._event_ref(claim.event_id)
        document_data = self._data(self._get(transaction, document_ref))
        event_data = self._data(self._get(transaction, event_ref))
        if document_data is None or event_data is None:
            raise RagError(ErrorCode.STALE_GENERATION)
        document = DocumentRecord.model_validate(document_data)
        event = IdempotencyRecord.model_validate(event_data)
        if (
            document.generation != claim.generation
            or document.lease is None
            or document.lease.owner != claim.owner
            or document.lease.generation != claim.generation
            or document.lease.expires_at != claim.lease_expires_at
            or document.lease.expires_at < now
            or event.tenant_id != claim.tenant_id
            or event.event_id != claim.event_id
            or event.operation is not claim.operation
            or event.document_id != claim.document_id
            or event.generation != claim.generation
            or event.outcome is not IdempotencyOutcome.IN_PROGRESS
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        return document, event, document_ref, event_ref

    def stage_chunks(
        self,
        claim: Claim,
        chunks: Sequence[ChunkRecord],
        expected_chunk_ids_sha256: str,
        expected_chunk_count: int,
        now: datetime,
    ) -> StagedChunkResult:
        if (
            expected_chunk_count < 0
            or expected_chunk_count > self.max_chunks_per_document
            or len(chunks) > self.max_stage_batch_size
            or (not chunks and expected_chunk_count != 0)
            or re.fullmatch(r"[a-f0-9]{64}", expected_chunk_ids_sha256) is None
        ):
            raise RagError(ErrorCode.LIMIT_EXCEEDED)

        def operation(transaction: Any) -> StagedChunkResult:
            document, _, document_ref, _ = self._claimed_records(transaction, claim, now)
            if (
                claim.operation is not IngestionOperation.UPSERT
                or document.status is not DocumentStatus.PROCESSING
                or document.tombstoned
            ):
                raise RagError(ErrorCode.STALE_GENERATION)
            if (
                document.expected_chunk_ids_sha256 is not None
                and document.expected_chunk_ids_sha256 != expected_chunk_ids_sha256
            ) or (
                document.chunk_count is not None and document.chunk_count != expected_chunk_count
            ):
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            references = [
                self._chunk_ref(chunk.document_id, chunk.generation, chunk.chunk_id)
                for chunk in chunks
            ]
            existing = [self._data(self._get(transaction, reference)) for reference in references]
            parser_versions = {chunk.parser_version for chunk in chunks}
            chunker_versions = {chunk.chunker_version for chunk in chunks}
            if len(parser_versions) > 1 or len(chunker_versions) > 1:
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            for chunk, stored_data in zip(chunks, existing, strict=True):
                if (
                    chunk.tenant_id != self.tenant_id
                    or chunk.document_id != claim.document_id
                    or chunk.generation != claim.generation
                    or chunk.object_generation != document.object_generation
                    or chunk.corpus != document.corpus
                    or chunk.pipeline_version != document.pipeline_version
                ):
                    raise RagError(ErrorCode.INVALID_REQUEST)
                if stored_data is not None:
                    stored = ChunkRecord.model_validate(stored_data)
                    immutable_fields = (
                        "tenant_id",
                        "corpus",
                        "document_id",
                        "generation",
                        "object_generation",
                        "chunk_id",
                        "ordinal",
                        "text",
                        "text_sha256",
                        "vector_id",
                        "pipeline_version",
                        "parser_version",
                        "chunker_version",
                        "source_page",
                        "source_start",
                        "source_end",
                    )
                    if any(
                        getattr(stored, field) != getattr(chunk, field)
                        for field in immutable_fields
                    ):
                        raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            processing = document.model_copy(
                update={
                    "expected_chunk_ids_sha256": expected_chunk_ids_sha256,
                    "chunk_count": expected_chunk_count,
                    "parser_version": next(iter(parser_versions), document.parser_version),
                    "chunker_version": next(iter(chunker_versions), document.chunker_version),
                    "updated_at": now,
                }
            )
            for chunk, reference, stored_data in zip(chunks, references, existing, strict=True):
                if stored_data is None:
                    staged_chunk = chunk.model_copy(update={"staged_at": now, "created_at": now})
                    transaction.create(reference, self._dump(staged_chunk))
            transaction.set(document_ref, self._dump(processing))
            return StagedChunkResult(
                staged_count=len(chunks),
                expected_chunk_count=expected_chunk_count,
                expected_chunk_ids_sha256=expected_chunk_ids_sha256,
            )

        return self._run_transaction(operation)

    def commit_ready(
        self,
        claim: Claim,
        expected_chunk_ids: Sequence[str],
        expected_chunk_count: int,
        provenance: Provenance,
        vector_verification: VerificationResult,
        now: datetime,
    ) -> DocumentRecord:
        try:
            expected_hash = chunk_ids_sha256(expected_chunk_ids)
        except ValueError as error:
            raise RagError(ErrorCode.INVALID_REQUEST) from error
        if expected_chunk_count > self.max_chunks_per_document:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)
        if expected_chunk_count < 0 or len(expected_chunk_ids) != expected_chunk_count:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if (
            not vector_verification.verified
            or vector_verification.expected_count != expected_chunk_count
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)

        def operation(transaction: Any) -> DocumentRecord:
            document, event, document_ref, event_ref = self._claimed_records(
                transaction, claim, now
            )
            if (
                claim.operation is not IngestionOperation.UPSERT
                or document.status is not DocumentStatus.PROCESSING
                or document.tombstoned
            ):
                raise RagError(ErrorCode.STALE_GENERATION)
            index_data = self._data(self._get(transaction, self._active_index_ref()))
            if index_data is None:
                raise RagError(ErrorCode.CONFIGURATION_INVALID)
            active_index = ActiveIndexRecord.model_validate(index_data)
            query = (
                document_ref.collection("chunks")
                .where("generation", "==", claim.generation)
                .order_by("ordinal")
                .order_by("chunk_id")
                .limit(self.max_chunks_per_document + 1)
            )
            snapshots = self._get_query(transaction, query)
            if len(snapshots) > self.max_chunks_per_document:
                raise RagError(ErrorCode.LIMIT_EXCEEDED)
            staged = [ChunkRecord.model_validate(self._data(snapshot)) for snapshot in snapshots]
            staged_ids = sorted((chunk.chunk_id for chunk in staged), key=str.encode)
            staged_vector_ids = sorted((chunk.vector_id for chunk in staged), key=str.encode)
            checked_vector_ids = set(vector_verification.checked_vector_ids)
            if (
                staged_ids != sorted(expected_chunk_ids, key=str.encode)
                or len(staged_ids) != expected_chunk_count
                or chunk_ids_sha256(staged_ids) != expected_hash
                or document.chunk_count != expected_chunk_count
                or document.expected_chunk_ids_sha256 != expected_hash
                or not checked_vector_ids.issubset(staged_vector_ids)
                or (
                    staged_vector_ids
                    and (
                        staged_vector_ids[0] not in checked_vector_ids
                        or staged_vector_ids[-1] not in checked_vector_ids
                    )
                )
                or provenance.embedding_model_id != active_index.embedding_model_id
                or provenance.embedding_dimensions != active_index.embedding_dimensions
                or provenance.active_index_version != active_index.index_version
                or provenance.pipeline_version != active_index.pipeline_version
                or provenance.parser_version != active_index.parser_version
                or provenance.chunker_version != active_index.chunker_version
                or document.pipeline_version != active_index.pipeline_version
                or (
                    document.parser_version is not None
                    and document.parser_version != active_index.parser_version
                )
                or (
                    document.chunker_version is not None
                    and document.chunker_version != active_index.chunker_version
                )
                or any(
                    chunk.object_generation != document.object_generation
                    or chunk.pipeline_version != provenance.pipeline_version
                    or chunk.parser_version != provenance.parser_version
                    or chunk.chunker_version != provenance.chunker_version
                    for chunk in staged
                )
            ):
                raise RagError(ErrorCode.DEPENDENCY_REJECTED)
            ready = document.model_copy(
                update={
                    "status": DocumentStatus.READY,
                    "lease": None,
                    "chunk_count": expected_chunk_count,
                    "expected_chunk_ids_sha256": expected_hash,
                    "parser_version": provenance.parser_version,
                    "chunker_version": provenance.chunker_version,
                    "embedding_model_id": provenance.embedding_model_id,
                    "embedding_dimensions": provenance.embedding_dimensions,
                    "active_index_version": provenance.active_index_version,
                    "ready_at": now,
                    "updated_at": now,
                }
            )
            ready = DocumentRecord.model_validate(ready.model_dump())
            succeeded = event.model_copy(
                update={"outcome": IdempotencyOutcome.SUCCEEDED, "updated_at": now}
            )
            transaction.set(document_ref, self._dump(ready))
            transaction.set(event_ref, self._dump(succeeded))
            return ready

        return self._run_transaction(operation)

    def commit_retryable_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None:
        self._commit_failure(claim, failure, now, terminal=False)

    def record_retryable_delete_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> DocumentRecord:
        if not failure.retryable:
            raise RagError(ErrorCode.INVALID_REQUEST)

        def operation(transaction: Any) -> DocumentRecord:
            document, event, document_ref, event_ref = self._claimed_records(
                transaction, claim, now
            )
            if (
                claim.operation is not IngestionOperation.DELETE
                or document.status is not DocumentStatus.DELETE_PENDING
                or not document.tombstoned
            ):
                raise RagError(ErrorCode.STALE_GENERATION)
            preserved = document.model_copy(
                update={
                    "status": DocumentStatus.DELETE_PENDING,
                    "tombstoned": True,
                    "lease": None,
                    "failure": failure,
                    "updated_at": now,
                }
            )
            preserved = DocumentRecord.model_validate(preserved.model_dump())
            event = event.model_copy(update={"updated_at": now})
            transaction.set(document_ref, self._dump(preserved))
            transaction.set(event_ref, self._dump(event))
            return preserved

        return self._run_transaction(operation)

    def commit_terminal_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None:
        self._commit_failure(claim, failure, now, terminal=True)

    def _commit_failure(
        self,
        claim: Claim,
        failure: SanitizedFailure,
        now: datetime,
        *,
        terminal: bool,
    ) -> None:
        def operation(transaction: Any) -> None:
            document, event, document_ref, event_ref = self._claimed_records(
                transaction, claim, now
            )
            if (
                claim.operation is not IngestionOperation.UPSERT
                or document.status is not DocumentStatus.PROCESSING
            ):
                raise RagError(ErrorCode.STALE_GENERATION)
            status = DocumentStatus.FAILED_TERMINAL if terminal else DocumentStatus.FAILED_RETRYABLE
            outcome = (
                IdempotencyOutcome.TERMINAL_FAILURE if terminal else IdempotencyOutcome.IN_PROGRESS
            )
            failed = document.model_copy(
                update={"status": status, "lease": None, "failure": failure, "updated_at": now}
            )
            event = event.model_copy(update={"outcome": outcome, "updated_at": now})
            transaction.set(document_ref, self._dump(failed))
            transaction.set(event_ref, self._dump(event))

        self._run_transaction(operation)

    def commit_deleted(self, claim: Claim, now: datetime) -> DocumentRecord:
        chunks = list(
            self._document_ref(claim.document_id)
            .collection("chunks")
            .limit(self.max_chunks_per_document + 1)
            .stream(timeout=self.timeout, retry=None)
        )
        if len(chunks) > self.max_chunks_per_document:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)

        def operation(transaction: Any) -> DocumentRecord:
            document, event, document_ref, event_ref = self._claimed_records(
                transaction, claim, now
            )
            if (
                claim.operation is not IngestionOperation.DELETE
                or document.status is not DocumentStatus.DELETE_PENDING
                or not document.tombstoned
            ):
                raise RagError(ErrorCode.STALE_GENERATION)
            for snapshot in chunks:
                transaction.delete(snapshot.reference)
            deleted = document.model_copy(
                update={"status": DocumentStatus.DELETED, "lease": None, "updated_at": now}
            )
            succeeded = event.model_copy(
                update={"outcome": IdempotencyOutcome.SUCCEEDED, "updated_at": now}
            )
            transaction.set(document_ref, self._dump(deleted))
            transaction.set(event_ref, self._dump(succeeded))
            return DocumentRecord.model_validate(deleted.model_dump())

        return self._run_transaction(operation)

    @staticmethod
    def _encode_chunk_cursor(generation: int, ordinal: int, chunk_id: str) -> str:
        raw = json.dumps(
            [generation, ordinal, chunk_id], ensure_ascii=False, separators=(",", ":")
        ).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_chunk_cursor(cursor: str) -> tuple[int, int, str]:
        try:
            padding = "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(cursor + padding))
            generation, ordinal, chunk_id = value
            if (
                not isinstance(generation, int)
                or generation < 1
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(chunk_id, str)
                or not chunk_id
            ):
                raise ValueError
            return generation, ordinal, chunk_id
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as error:
            raise RagError(ErrorCode.INVALID_REQUEST) from error

    def list_document_chunks(
        self, tenant_id: str, document_id: str, after: str | None, limit: int
    ) -> ChunkPage:
        require_tenant(tenant_id, self.tenant_id)
        if limit < 1 or limit > self.max_chunks_per_document:
            raise RagError(ErrorCode.INVALID_REQUEST)
        query = (
            self._document_ref(document_id)
            .collection("chunks")
            .order_by("generation")
            .order_by("ordinal")
            .order_by("chunk_id")
        )
        if after is not None:
            generation, ordinal, chunk_id = self._decode_chunk_cursor(after)
            query = query.start_after(
                {
                    "generation": generation,
                    "ordinal": ordinal,
                    "chunk_id": chunk_id,
                }
            )
        snapshots = list(query.limit(limit + 1).stream(timeout=self.timeout, retry=None))
        records = [
            ChunkRecord.model_validate(self._data(snapshot)) for snapshot in snapshots[:limit]
        ]
        items = [
            ChunkPointer(
                chunk_id=record.chunk_id,
                generation=record.generation,
                ordinal=record.ordinal,
                vector_id=record.vector_id,
            )
            for record in records
        ]
        next_cursor = None
        if len(snapshots) > limit and records:
            last = records[-1]
            next_cursor = self._encode_chunk_cursor(last.generation, last.ordinal, last.chunk_id)
        return ChunkPage(items=items, next_cursor=next_cursor)

    def hydrate_authorized(
        self,
        tenant_id: str,
        corpus: str,
        candidates: Sequence[RetrievalCandidate],
        active_index_version: str,
    ) -> list[ChunkRecord]:
        require_tenant(tenant_id, self.tenant_id)
        if not candidates:
            return []
        if len(candidates) > self.max_active_generations:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)
        index = self.get_active_index()
        if active_index_version != index.index_version:
            raise RagError(ErrorCode.STALE_GENERATION)
        references: list[Any] = []
        for candidate in candidates:
            references.extend(
                (
                    self._document_ref(candidate.document_id),
                    self._chunk_ref(
                        candidate.document_id, candidate.generation, candidate.chunk_id
                    ),
                )
            )
        snapshots = self.client.get_all(references, retry=None, timeout=self.timeout)
        records = {snapshot.reference.path: self._data(snapshot) for snapshot in snapshots}
        hydrated: list[ChunkRecord] = []
        for candidate in candidates:
            document_data = records.get(self._document_ref(candidate.document_id).path)
            chunk_data = records.get(
                self._chunk_ref(
                    candidate.document_id, candidate.generation, candidate.chunk_id
                ).path
            )
            if document_data is None or chunk_data is None:
                continue
            document = DocumentRecord.model_validate(document_data)
            chunk = ChunkRecord.model_validate(chunk_data)
            if (
                document.tenant_id != tenant_id
                or document.corpus != corpus
                or document.generation != candidate.generation
                or document.status is not DocumentStatus.READY
                or document.tombstoned
                or document.active_index_version != active_index_version
                or document.pipeline_version != index.pipeline_version
                or document.parser_version != index.parser_version
                or document.chunker_version != index.chunker_version
                or chunk.tenant_id != tenant_id
                or chunk.corpus != corpus
                or chunk.document_id != candidate.document_id
                or chunk.generation != candidate.generation
                or chunk.chunk_id != candidate.chunk_id
                or chunk.vector_id != candidate.vector_id
                or chunk.object_generation != document.object_generation
                or chunk.pipeline_version != index.pipeline_version
                or chunk.parser_version != index.parser_version
                or chunk.chunker_version != index.chunker_version
            ):
                continue
            hydrated.append(chunk)
        return hydrated

    def get_active_index(self) -> ActiveIndexRecord:
        snapshot = self._active_index_ref().get(retry=None, timeout=self.timeout)
        data = self._data(snapshot)
        if data is None:
            raise RagError(ErrorCode.CONFIGURATION_INVALID)
        return ActiveIndexRecord.model_validate(data)

    def list_active_generations(
        self, tenant_id: str, corpus: str, limit: int
    ) -> list[tuple[str, int]]:
        require_tenant(tenant_id, self.tenant_id)
        if limit < 1 or limit > self.max_active_generations:
            raise RagError(ErrorCode.INVALID_REQUEST)
        index = self.get_active_index()
        query = self.client.collection(f"tenants/{self.tenant_id}/documents")
        query = query.where("corpus", "==", corpus)
        query = query.where("status", "==", DocumentStatus.READY.value)
        query = query.where("tombstoned", "==", False)
        query = query.where("active_index_version", "==", index.index_version)
        query = query.where("pipeline_version", "==", index.pipeline_version)
        query = query.where("parser_version", "==", index.parser_version)
        query = query.where("chunker_version", "==", index.chunker_version)
        query = query.order_by("document_id").order_by("generation").limit(limit + 1)
        snapshots = list(query.stream(timeout=self.timeout, retry=None))
        if len(snapshots) > limit:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)
        records = [DocumentRecord.model_validate(self._data(snapshot)) for snapshot in snapshots]
        active = [
            (record.document_id, record.generation)
            for record in records
            if record.pipeline_version == index.pipeline_version
            and record.parser_version == index.parser_version
            and record.chunker_version == index.chunker_version
        ]
        return sorted(active, key=lambda item: (item[0].encode(), item[1]))
