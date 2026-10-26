"""Deterministic local substitutes with failure injection and call recording."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import (
    ActiveIndexRecord,
    AuthorizedContext,
    ChunkPage,
    ChunkPointer,
    ChunkRecord,
    Claim,
    ClaimDisposition,
    ClaimResult,
    DeleteResult,
    DocumentRecord,
    DocumentRegistration,
    DocumentStatus,
    GeneratedAnswer,
    GeneratedCitation,
    IdempotencyOutcome,
    IngestionMessage,
    IngestionOperation,
    Lease,
    ObjectMetadata,
    Provenance,
    PublishReceipt,
    RegistrationIdempotencyRecord,
    RegistrationOutcome,
    RegistrationResult,
    RetrievalCandidate,
    SanitizedFailure,
    StagedChunkResult,
    Vector,
    VectorDatapoint,
    VectorRestrictions,
    VerificationResult,
    chunk_ids_sha256,
)
from packages.rag_core.policy import active_generation_token, require_tenant


class FakeBase:
    """Shared deterministic fault queue and call recorder."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._failures: dict[str, deque[Exception]] = defaultdict(deque)

    def fail_next(self, method: str, error: Exception) -> None:
        self._failures[method].append(error)

    def _record(self, method: str, **arguments: object) -> None:
        self.calls.append((method, arguments))
        if self._failures[method]:
            raise self._failures[method].popleft()


class FakeClock:
    def __init__(self, now: datetime | None = None, monotonic: float = 0.0) -> None:
        self._now = now or datetime(2025, 1, 1, tzinfo=UTC)
        self._monotonic = monotonic

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds


class FakeIds:
    def __init__(self, seed: int = 1) -> None:
        self._next = seed

    def new_event_id(self) -> UUID:
        result = UUID(int=self._next)
        self._next += 1
        return result

    def new_trace_id(self) -> str:
        result = f"trace-{self._next:08d}"
        self._next += 1
        return result

    def chunk_id(self, document_id: str, generation: int, ordinal: int, text_sha256: str) -> str:
        material = f"{document_id}\0{generation}\0{ordinal}\0{text_sha256}".encode()
        return hashlib.sha256(material).hexdigest()


class FakeEmbeddingModel(FakeBase):
    def __init__(self, *, max_batch_size: int = 250) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size

    def embed(self, texts: Sequence[str], dimensions: int, deadline: float) -> list[Vector]:
        self._record("embed", count=len(texts), dimensions=dimensions, deadline=deadline)
        if not texts or len(texts) > self.max_batch_size or dimensions < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = [
                ((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(dimensions)
            ]
            magnitude = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / magnitude for value in vector])
        return vectors


class FakeGenerator(FakeBase):
    def __init__(self, answer: str = "Grounded answer.") -> None:
        super().__init__()
        self.answer = answer

    def generate(
        self,
        question: str,
        authorized_context: Sequence[AuthorizedContext],
        max_output_tokens: int,
        deadline: float,
    ) -> GeneratedAnswer:
        self._record(
            "generate",
            question_length=len(question),
            context_ids=tuple(item.citation_id for item in authorized_context),
            max_output_tokens=max_output_tokens,
            deadline=deadline,
        )
        if not question or not authorized_context or max_output_tokens < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        first = authorized_context[0]
        return GeneratedAnswer(
            answer=self.answer,
            citations=[GeneratedCitation(citation_id=first.citation_id, quote=first.text[:2000])],
        )


class FakeVectorIndex(FakeBase):
    def __init__(self, *, max_batch_size: int = 100) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.datapoints: dict[str, VectorDatapoint] = {}

    def upsert(self, datapoints: Sequence[VectorDatapoint], deadline: float) -> None:
        self._record(
            "upsert", vector_ids=tuple(item.vector_id for item in datapoints), deadline=deadline
        )
        if not datapoints or len(datapoints) > self.max_batch_size or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        for item in datapoints:
            expected = active_generation_token(item.document_id, item.generation)
            if item.active_generation != expected:
                raise RagError(ErrorCode.INVALID_REQUEST)
            self.datapoints[item.vector_id] = item

    def delete(self, vector_ids: Sequence[str], deadline: float) -> None:
        self._record("delete", vector_ids=tuple(vector_ids), deadline=deadline)
        if not vector_ids or len(vector_ids) > self.max_batch_size or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        for vector_id in vector_ids:
            self.datapoints.pop(vector_id, None)

    def search(
        self,
        vector: list[float],
        restrictions: VectorRestrictions,
        top_k: int,
        deadline: float,
    ) -> list[RetrievalCandidate]:
        self._record("search", restrictions=restrictions, top_k=top_k, deadline=deadline)
        if (
            not vector
            or not restrictions.tenant_id
            or not restrictions.corpus
            or not restrictions.index_version
            or not restrictions.active_generations
            or top_k < 1
            or deadline <= 0
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        allowed = set(restrictions.active_generations)
        matches: list[RetrievalCandidate] = []
        for item in self.datapoints.values():
            if (
                item.tenant_id != restrictions.tenant_id
                or item.corpus != restrictions.corpus
                or item.index_version != restrictions.index_version
                or item.active_generation not in allowed
                or len(item.vector) != len(vector)
            ):
                continue
            score = sum(left * right for left, right in zip(vector, item.vector, strict=True))
            matches.append(
                RetrievalCandidate(
                    vector_id=item.vector_id,
                    score=score,
                    document_id=item.document_id,
                    generation=item.generation,
                    chunk_id=item.vector_id,
                )
            )
        return sorted(matches, key=lambda item: (-item.score, item.vector_id))[:top_k]

    def read(self, vector_ids: Sequence[str], limit: int, deadline: float) -> list[VectorDatapoint]:
        if (
            not vector_ids
            or limit < 1
            or limit > self.max_batch_size
            or len(vector_ids) > limit
            or len(set(vector_ids)) != len(vector_ids)
            or any(not vector_id for vector_id in vector_ids)
            or deadline <= 0
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        self._record("read", vector_ids=tuple(vector_ids), limit=limit, deadline=deadline)
        return [
            self.datapoints[vector_id] for vector_id in vector_ids if vector_id in self.datapoints
        ]

    def verify(
        self,
        expected_vector_ids: Sequence[str],
        expected_count: int,
        sample_size: int,
        deadline: float,
    ) -> VerificationResult:
        unique = sorted(set(expected_vector_ids), key=str.encode)
        if (
            expected_count < 0
            or expected_count != len(expected_vector_ids)
            or len(unique) != expected_count
            or sample_size < 1
            or (expected_count > 1 and sample_size < 2)
            or sample_size > self.max_batch_size
            or deadline <= 0
            or any(not vector_id for vector_id in unique)
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        self._record(
            "verify",
            expected_count=expected_count,
            sample_size=sample_size,
            deadline=deadline,
        )
        if not unique:
            sample: list[str] = []
        elif sample_size >= len(unique):
            sample = unique
        elif sample_size == 1:
            sample = [unique[0]]
        else:
            sample = [*unique[: sample_size - 1], unique[-1]]
        missing = tuple(vector_id for vector_id in sample if vector_id not in self.datapoints)
        return VerificationResult(
            verified=not missing,
            expected_count=expected_count,
            checked_vector_ids=tuple(sample),
            missing_vector_ids=missing,
        )


class FakeObjectStore(FakeBase):
    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, tuple[ObjectMetadata, bytes]] = {}

    def put(self, metadata: ObjectMetadata, content: bytes) -> None:
        self.objects[metadata.canonical_uri] = (metadata, content)

    def stat(self, canonical_uri: str, deadline: float) -> ObjectMetadata:
        self._record("stat", canonical_uri=canonical_uri, deadline=deadline)
        if deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        try:
            return self.objects[canonical_uri][0]
        except KeyError as error:
            raise RagError(ErrorCode.DOCUMENT_NOT_FOUND) from error

    def read_bounded(
        self, canonical_uri: str, object_generation: int, max_bytes: int, deadline: float
    ) -> bytes:
        self._record(
            "read_bounded",
            canonical_uri=canonical_uri,
            object_generation=object_generation,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        try:
            metadata, content = self.objects[canonical_uri]
        except KeyError as error:
            raise RagError(ErrorCode.DOCUMENT_NOT_FOUND) from error
        if deadline <= 0 or max_bytes < 1 or object_generation < 1:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if metadata.object_generation != object_generation:
            raise RagError(ErrorCode.STALE_GENERATION)
        if metadata.size > max_bytes or len(content) > max_bytes:
            raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
        if (
            metadata.checksum_sha256
            and hashlib.sha256(content).hexdigest() != metadata.checksum_sha256
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return content

    def delete(self, canonical_uri: str, object_generation: int, deadline: float) -> DeleteResult:
        self._record(
            "delete",
            canonical_uri=canonical_uri,
            object_generation=object_generation,
            deadline=deadline,
        )
        existing = self.objects.get(canonical_uri)
        if existing is None:
            return DeleteResult(deleted=False)
        if existing[0].object_generation != object_generation:
            raise RagError(ErrorCode.STALE_GENERATION)
        del self.objects[canonical_uri]
        return DeleteResult(deleted=True)


class FakeEventPublisher(FakeBase):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[IngestionMessage] = []

    def publish(self, payload: IngestionMessage, deadline: float) -> PublishReceipt:
        self._record("publish", event_id=str(payload.message_metadata.event_id), deadline=deadline)
        if deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        self.published.append(payload)
        return PublishReceipt(message_id=f"message-{len(self.published):08d}")


class FakeMetadataStore(FakeBase):
    """Generation-aware, tombstone-safe in-memory metadata transaction fake."""

    def __init__(
        self,
        *,
        tenant_id: str,
        active_index: ActiveIndexRecord,
        max_chunks_per_document: int = 10_000,
        max_active_generations: int = 10_000,
        max_stage_batch_size: int = 450,
    ) -> None:
        super().__init__()
        self.tenant_id = tenant_id
        self.active_index = active_index
        self.max_chunks_per_document = max_chunks_per_document
        self.max_active_generations = max_active_generations
        self.max_stage_batch_size = max_stage_batch_size
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[tuple[str, int, str], ChunkRecord] = {}
        self._registrations: dict[str, RegistrationIdempotencyRecord] = {}
        self._events: dict[UUID, tuple[str, IdempotencyOutcome]] = {}

    def register(
        self,
        registration: DocumentRegistration,
        idempotency_key: str,
        event_id: UUID,
        now: datetime,
    ) -> RegistrationResult:
        self._record(
            "register", document_id=registration.document_id, idempotency_key=idempotency_key
        )
        require_tenant(registration.tenant_id, self.tenant_id)
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        canonical = {
            "checksum_sha256": registration.checksum_sha256,
            "corpus": registration.corpus,
            "document_id": registration.document_id,
            "media_type": registration.media_type.value,
            "object_generation": registration.object_generation,
            "object_uri": registration.object_uri,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        previous = self._registrations.get(key_hash)
        if previous:
            if previous.request_sha256 != fingerprint:
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            document = self.documents[previous.document_id]
            if document.generation != previous.generation:
                return RegistrationResult(idempotency=previous)
            return RegistrationResult(idempotency=previous, document=document)
        current = self.documents.get(registration.document_id)
        generation = 1 if current is None else current.generation + 1
        record = DocumentRecord(
            **registration.model_dump(),
            generation=generation,
            status=DocumentStatus.PENDING,
            tombstoned=False,
            created_at=current.created_at if current else now,
            updated_at=now,
        )
        self.documents[registration.document_id] = record
        idempotency = RegistrationIdempotencyRecord(
            tenant_id=self.tenant_id,
            idempotency_key_sha256=key_hash,
            request_sha256=fingerprint,
            document_id=registration.document_id,
            generation=generation,
            event_id=event_id,
            outcome=RegistrationOutcome.COMMITTED,
            created_at=now,
            updated_at=now,
        )
        self._registrations[key_hash] = idempotency
        return RegistrationResult(document=record, idempotency=idempotency)

    def mark_registration_published(
        self,
        tenant_id: str,
        idempotency_key_sha256: str,
        request_sha256: str,
        event_id: UUID,
        published_at: datetime,
    ) -> RegistrationIdempotencyRecord:
        self._record("mark_registration_published", event_id=str(event_id))
        require_tenant(tenant_id, self.tenant_id)
        current = self._registrations.get(idempotency_key_sha256)
        if (
            current is None
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
        self._registrations[idempotency_key_sha256] = published
        return published

    def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None:
        self._record("get_document", document_id=document_id)
        require_tenant(tenant_id, self.tenant_id)
        return self.documents.get(document_id)

    def tombstone(self, tenant_id: str, document_id: str, now: datetime) -> DocumentRecord:
        self._record("tombstone", document_id=document_id)
        require_tenant(tenant_id, self.tenant_id)
        current = self.documents.get(document_id)
        if current is None:
            raise RagError(ErrorCode.DOCUMENT_NOT_FOUND)
        if current.status in {DocumentStatus.DELETE_PENDING, DocumentStatus.DELETED}:
            return current
        record = current.model_copy(
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
        self.documents[document_id] = DocumentRecord.model_validate(record.model_dump())
        return self.documents[document_id]

    @staticmethod
    def _payload_hash(payload: IngestionMessage) -> str:
        return hashlib.sha256(payload.model_dump_json().encode()).hexdigest()

    def claim_event(
        self,
        payload: IngestionMessage,
        owner: str,
        lease_until: datetime,
        now: datetime,
    ) -> ClaimResult:
        self._record("claim_event", event_id=str(payload.message_metadata.event_id), owner=owner)
        require_tenant(payload.tenant_id, self.tenant_id)
        current = self.documents.get(payload.document_id)
        if current is None:
            raise RagError(ErrorCode.DOCUMENT_NOT_FOUND)
        if payload.generation != current.generation:
            raise RagError(ErrorCode.STALE_GENERATION)
        if (
            payload.corpus != current.corpus
            or payload.object_generation != current.object_generation
            or payload.checksum_sha256 != current.checksum_sha256
            or payload.pipeline_version != current.pipeline_version
        ):
            raise RagError(ErrorCode.DOCUMENT_CONFLICT)
        digest = self._payload_hash(payload)
        existing = self._events.get(payload.message_metadata.event_id)
        if existing:
            if existing[0] != digest:
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            if existing[1] is IdempotencyOutcome.SUCCEEDED:
                return ClaimResult(disposition=ClaimDisposition.DUPLICATE_SUCCESS)
            if existing[1] is IdempotencyOutcome.TERMINAL_FAILURE:
                return ClaimResult(disposition=ClaimDisposition.DUPLICATE_TERMINAL)
        if current.lease and current.lease.expires_at > now:
            raise RagError(ErrorCode.LEASE_CONTENDED)
        if lease_until <= now:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if payload.operation is IngestionOperation.UPSERT:
            if current.tombstoned:
                raise RagError(ErrorCode.STALE_GENERATION)
            next_record = current.model_copy(
                update={
                    "status": DocumentStatus.PROCESSING,
                    "lease": Lease(
                        owner=owner, expires_at=lease_until, generation=current.generation
                    ),
                    "updated_at": now,
                    "failure": None,
                }
            )
            self.documents[payload.document_id] = DocumentRecord.model_validate(
                next_record.model_dump()
            )
        else:
            if current.status is not DocumentStatus.DELETE_PENDING or not current.tombstoned:
                raise RagError(ErrorCode.STALE_GENERATION)
            next_record = current.model_copy(
                update={
                    "lease": Lease(
                        owner=owner, expires_at=lease_until, generation=current.generation
                    ),
                    "updated_at": now,
                }
            )
            self.documents[payload.document_id] = DocumentRecord.model_validate(
                next_record.model_dump()
            )
        self._events[payload.message_metadata.event_id] = (digest, IdempotencyOutcome.IN_PROGRESS)
        claim = Claim(
            tenant_id=payload.tenant_id,
            event_id=payload.message_metadata.event_id,
            operation=payload.operation,
            document_id=payload.document_id,
            generation=payload.generation,
            owner=owner,
            lease_expires_at=lease_until,
        )
        return ClaimResult(disposition=ClaimDisposition.CLAIMED, claim=claim)

    def _current_for_claim(self, claim: Claim, now: datetime) -> DocumentRecord:
        require_tenant(claim.tenant_id, self.tenant_id)
        current = self.documents.get(claim.document_id)
        if current is None or current.generation != claim.generation:
            raise RagError(ErrorCode.STALE_GENERATION)
        if (
            claim.lease_expires_at < now
            or current.lease is None
            or current.lease.owner != claim.owner
            or current.lease.expires_at != claim.lease_expires_at
            or current.lease.generation != claim.generation
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        if (
            claim.operation is IngestionOperation.UPSERT
            and current.status is not DocumentStatus.PROCESSING
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        return current

    def stage_chunks(
        self,
        claim: Claim,
        chunks: Sequence[ChunkRecord],
        expected_chunk_ids_sha256: str,
        expected_chunk_count: int,
        now: datetime,
    ) -> StagedChunkResult:
        self._record("stage_chunks", event_id=str(claim.event_id), chunk_count=len(chunks))
        current = self._current_for_claim(claim, now)
        if (
            claim.operation is not IngestionOperation.UPSERT
            or current.tombstoned
            or expected_chunk_count < 0
            or expected_chunk_count > self.max_chunks_per_document
            or len(chunks) > self.max_stage_batch_size
            or re.fullmatch(r"[a-f0-9]{64}", expected_chunk_ids_sha256) is None
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        if (
            current.expected_chunk_ids_sha256 is not None
            and current.expected_chunk_ids_sha256 != expected_chunk_ids_sha256
        ) or (current.chunk_count is not None and current.chunk_count != expected_chunk_count):
            raise RagError(ErrorCode.DOCUMENT_CONFLICT)
        parser_versions = {chunk.parser_version for chunk in chunks}
        chunker_versions = {chunk.chunker_version for chunk in chunks}
        if len(parser_versions) > 1 or len(chunker_versions) > 1:
            raise RagError(ErrorCode.DOCUMENT_CONFLICT)
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
        for chunk in chunks:
            if (
                chunk.tenant_id != claim.tenant_id
                or chunk.document_id != claim.document_id
                or chunk.generation != claim.generation
                or chunk.object_generation != current.object_generation
                or chunk.corpus != current.corpus
                or chunk.pipeline_version != current.pipeline_version
            ):
                raise RagError(ErrorCode.INVALID_REQUEST)
            key = (chunk.document_id, chunk.generation, chunk.chunk_id)
            existing = self.chunks.get(key)
            if existing is not None and any(
                getattr(existing, field) != getattr(chunk, field) for field in immutable_fields
            ):
                raise RagError(ErrorCode.DOCUMENT_CONFLICT)
            if existing is None:
                self.chunks[key] = chunk.model_copy(update={"staged_at": now, "created_at": now})
        processing = current.model_copy(
            update={
                "expected_chunk_ids_sha256": expected_chunk_ids_sha256,
                "chunk_count": expected_chunk_count,
                "parser_version": next(iter(parser_versions), current.parser_version),
                "chunker_version": next(iter(chunker_versions), current.chunker_version),
                "updated_at": now,
            }
        )
        self.documents[claim.document_id] = DocumentRecord.model_validate(processing.model_dump())
        return StagedChunkResult(
            staged_count=len(chunks),
            expected_chunk_count=expected_chunk_count,
            expected_chunk_ids_sha256=expected_chunk_ids_sha256,
        )

    def commit_ready(
        self,
        claim: Claim,
        expected_chunk_ids: Sequence[str],
        expected_chunk_count: int,
        provenance: Provenance,
        vector_verification: VerificationResult,
        now: datetime,
    ) -> DocumentRecord:
        self._record(
            "commit_ready",
            event_id=str(claim.event_id),
            chunk_count=expected_chunk_count,
        )
        if expected_chunk_count > self.max_chunks_per_document:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)
        current = self._current_for_claim(claim, now)
        if claim.operation is not IngestionOperation.UPSERT or current.tombstoned:
            raise RagError(ErrorCode.STALE_GENERATION)
        try:
            expected_hash = chunk_ids_sha256(expected_chunk_ids)
        except ValueError as error:
            raise RagError(ErrorCode.INVALID_REQUEST) from error
        staged = sorted(
            (
                chunk
                for (document_id, generation, _), chunk in self.chunks.items()
                if document_id == claim.document_id and generation == claim.generation
            ),
            key=lambda item: (item.ordinal, item.chunk_id.encode()),
        )
        staged_ids = sorted((chunk.chunk_id for chunk in staged), key=str.encode)
        staged_vector_ids = sorted((chunk.vector_id for chunk in staged), key=str.encode)
        checked_vector_ids = set(vector_verification.checked_vector_ids)
        if (
            len(staged) != expected_chunk_count
            or staged_ids != sorted(expected_chunk_ids, key=str.encode)
            or current.chunk_count != expected_chunk_count
            or current.expected_chunk_ids_sha256 != expected_hash
            or not vector_verification.verified
            or vector_verification.expected_count != expected_chunk_count
            or not checked_vector_ids.issubset(staged_vector_ids)
            or (
                staged_vector_ids
                and (
                    staged_vector_ids[0] not in checked_vector_ids
                    or staged_vector_ids[-1] not in checked_vector_ids
                )
            )
            or provenance.pipeline_version != self.active_index.pipeline_version
            or provenance.parser_version != self.active_index.parser_version
            or provenance.chunker_version != self.active_index.chunker_version
            or provenance.embedding_model_id != self.active_index.embedding_model_id
            or provenance.embedding_dimensions != self.active_index.embedding_dimensions
            or provenance.active_index_version != self.active_index.index_version
            or (
                current.parser_version is not None
                and current.parser_version != provenance.parser_version
            )
            or (
                current.chunker_version is not None
                and current.chunker_version != provenance.chunker_version
            )
            or any(
                chunk.object_generation != current.object_generation
                or chunk.pipeline_version != provenance.pipeline_version
                or chunk.parser_version != provenance.parser_version
                or chunk.chunker_version != provenance.chunker_version
                for chunk in staged
            )
        ):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        ready = current.model_copy(
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
        self.documents[claim.document_id] = DocumentRecord.model_validate(ready.model_dump())
        digest = self._events[claim.event_id][0]
        self._events[claim.event_id] = (digest, IdempotencyOutcome.SUCCEEDED)
        return self.documents[claim.document_id]

    def _commit_failure(
        self,
        claim: Claim,
        failure: SanitizedFailure,
        now: datetime,
        *,
        terminal: bool,
    ) -> None:
        current = self._current_for_claim(claim, now)
        if (
            claim.operation is not IngestionOperation.UPSERT
            or current.status is not DocumentStatus.PROCESSING
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        status = DocumentStatus.FAILED_TERMINAL if terminal else DocumentStatus.FAILED_RETRYABLE
        failed = current.model_copy(
            update={"status": status, "lease": None, "failure": failure, "updated_at": now}
        )
        self.documents[claim.document_id] = DocumentRecord.model_validate(failed.model_dump())
        digest = self._events[claim.event_id][0]
        outcome = (
            IdempotencyOutcome.TERMINAL_FAILURE if terminal else IdempotencyOutcome.IN_PROGRESS
        )
        self._events[claim.event_id] = (digest, outcome)

    def commit_retryable_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None:
        self._record("commit_retryable_failure", event_id=str(claim.event_id))
        self._commit_failure(claim, failure, now, terminal=False)

    def record_retryable_delete_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> DocumentRecord:
        self._record("record_retryable_delete_failure", event_id=str(claim.event_id))
        current = self._current_for_claim(claim, now)
        if (
            claim.operation is not IngestionOperation.DELETE
            or current.status is not DocumentStatus.DELETE_PENDING
            or not current.tombstoned
            or not failure.retryable
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        preserved = current.model_copy(
            update={"lease": None, "failure": failure, "updated_at": now}
        )
        preserved = DocumentRecord.model_validate(preserved.model_dump())
        self.documents[claim.document_id] = preserved
        return preserved

    def commit_terminal_failure(
        self, claim: Claim, failure: SanitizedFailure, now: datetime
    ) -> None:
        self._record("commit_terminal_failure", event_id=str(claim.event_id))
        self._commit_failure(claim, failure, now, terminal=True)

    def commit_deleted(self, claim: Claim, now: datetime) -> DocumentRecord:
        self._record("commit_deleted", event_id=str(claim.event_id))
        current = self._current_for_claim(claim, now)
        if (
            claim.operation is not IngestionOperation.DELETE
            or current.status is not DocumentStatus.DELETE_PENDING
            or not current.tombstoned
        ):
            raise RagError(ErrorCode.STALE_GENERATION)
        for key in [key for key in self.chunks if key[0] == claim.document_id]:
            del self.chunks[key]
        deleted = current.model_copy(
            update={"status": DocumentStatus.DELETED, "lease": None, "updated_at": now}
        )
        self.documents[claim.document_id] = DocumentRecord.model_validate(deleted.model_dump())
        digest = self._events[claim.event_id][0]
        self._events[claim.event_id] = (digest, IdempotencyOutcome.SUCCEEDED)
        return self.documents[claim.document_id]

    def list_document_chunks(
        self, tenant_id: str, document_id: str, after: str | None, limit: int
    ) -> ChunkPage:
        self._record("list_document_chunks", document_id=document_id, limit=limit)
        require_tenant(tenant_id, self.tenant_id)
        if limit < 1 or limit > self.max_chunks_per_document:
            raise RagError(ErrorCode.INVALID_REQUEST)
        cursor: tuple[int, int, str] | None = None
        if after is not None:
            try:
                padding = "=" * (-len(after) % 4)
                decoded = json.loads(base64.urlsafe_b64decode(after + padding))
                if (
                    not isinstance(decoded, list)
                    or len(decoded) != 3
                    or not isinstance(decoded[0], int)
                    or decoded[0] < 1
                    or not isinstance(decoded[1], int)
                    or decoded[1] < 0
                    or not isinstance(decoded[2], str)
                    or not decoded[2]
                ):
                    raise ValueError
                cursor = (decoded[0], decoded[1], decoded[2])
            except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as error:
                raise RagError(ErrorCode.INVALID_REQUEST) from error
        records = sorted(
            (
                chunk
                for (stored_document_id, _, _), chunk in self.chunks.items()
                if stored_document_id == document_id
            ),
            key=lambda item: (item.generation, item.ordinal, item.chunk_id.encode()),
        )
        if cursor is not None:
            records = [
                chunk
                for chunk in records
                if (chunk.generation, chunk.ordinal, chunk.chunk_id) > cursor
            ]
        page_records = records[:limit]
        next_cursor = None
        if len(records) > limit:
            last = page_records[-1]
            raw = json.dumps(
                [last.generation, last.ordinal, last.chunk_id], separators=(",", ":")
            ).encode()
            next_cursor = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        return ChunkPage(
            items=[
                ChunkPointer(
                    chunk_id=chunk.chunk_id,
                    generation=chunk.generation,
                    ordinal=chunk.ordinal,
                    vector_id=chunk.vector_id,
                )
                for chunk in page_records
            ],
            next_cursor=next_cursor,
        )

    def hydrate_authorized(
        self,
        tenant_id: str,
        corpus: str,
        candidates: Sequence[RetrievalCandidate],
        active_index_version: str,
    ) -> list[ChunkRecord]:
        self._record("hydrate_authorized", candidate_count=len(candidates))
        require_tenant(tenant_id, self.tenant_id)
        hydrated = []
        for candidate in candidates:
            document = self.documents.get(candidate.document_id)
            chunk = self.chunks.get(
                (candidate.document_id, candidate.generation, candidate.chunk_id)
            )
            if (
                document is None
                or chunk is None
                or document.status is not DocumentStatus.READY
                or document.tombstoned
                or document.tenant_id != tenant_id
                or document.corpus != corpus
                or document.generation != candidate.generation
                or document.active_index_version != active_index_version
                or document.pipeline_version != self.active_index.pipeline_version
                or document.parser_version != self.active_index.parser_version
                or document.chunker_version != self.active_index.chunker_version
                or chunk.vector_id != candidate.vector_id
                or chunk.object_generation != document.object_generation
                or chunk.pipeline_version != self.active_index.pipeline_version
                or chunk.parser_version != self.active_index.parser_version
                or chunk.chunker_version != self.active_index.chunker_version
            ):
                continue
            hydrated.append(chunk)
        return hydrated

    def get_active_index(self) -> ActiveIndexRecord:
        self._record("get_active_index")
        return self.active_index

    def list_active_generations(
        self, tenant_id: str, corpus: str, limit: int
    ) -> list[tuple[str, int]]:
        self._record("list_active_generations", corpus=corpus, limit=limit)
        require_tenant(tenant_id, self.tenant_id)
        if limit < 1 or limit > self.max_active_generations:
            raise RagError(ErrorCode.INVALID_REQUEST)
        active = sorted(
            (document.document_id, document.generation)
            for document in self.documents.values()
            if document.corpus == corpus
            and document.status is DocumentStatus.READY
            and not document.tombstoned
            and document.active_index_version == self.active_index.index_version
            and document.pipeline_version == self.active_index.pipeline_version
            and document.parser_version == self.active_index.parser_version
            and document.chunker_version == self.active_index.chunker_version
        )
        if len(active) > limit:
            raise RagError(ErrorCode.LIMIT_EXCEEDED)
        return active
