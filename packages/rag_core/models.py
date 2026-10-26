"""Validated domain records shared by services and adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT_VERSION: Literal["1.0.0"] = "1.0.0"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_CORPUS = r"^[a-z][a-z0-9_-]*$"
_ERROR = r"^[A-Z][A-Z0-9_]*$"
_SHA256 = r"^[a-f0-9]{64}$"
_URI = re.compile(r"^gs://[^\s]+$")

TenantId = Annotated[str, StringConstraints(min_length=1, max_length=128)]
DocumentId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=_IDENTIFIER)]
Corpus = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=_CORPUS)]
Checksum = Annotated[str, StringConstraints(pattern=_SHA256)]
Generation = Annotated[int, Field(ge=1)]
ObjectGeneration = Annotated[int, Field(ge=1)]
Vector = list[float]


def chunk_ids_sha256(chunk_ids: Sequence[str]) -> str:
    """Hash a sorted, duplicate-free chunk-ID set using canonical JSON."""
    ordered = sorted(chunk_ids, key=str.encode)
    if len(set(ordered)) != len(ordered):
        raise ValueError("chunk IDs must be unique")
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class DomainModel(BaseModel):
    """Strict immutable value object."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    DELETE_PENDING = "DELETE_PENDING"
    DELETED = "DELETED"


class IngestionOperation(StrEnum):
    UPSERT = "UPSERT"
    DELETE = "DELETE"


class IdempotencyOutcome(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class RegistrationOutcome(StrEnum):
    COMMITTED = "COMMITTED"
    PUBLISHED = "PUBLISHED"


class ClaimDisposition(StrEnum):
    CLAIMED = "CLAIMED"
    DUPLICATE_SUCCESS = "DUPLICATE_SUCCESS"
    DUPLICATE_TERMINAL = "DUPLICATE_TERMINAL"


class AbstentionReason(StrEnum):
    NO_AUTHORIZED_EVIDENCE = "NO_AUTHORIZED_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"


class MediaType(StrEnum):
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF = "application/pdf"


class SanitizedFailure(DomainModel):
    code: Annotated[str, StringConstraints(pattern=_ERROR)]
    retryable: bool
    attempts: Annotated[int, Field(ge=1)]
    last_failed_at: datetime


class Lease(DomainModel):
    owner: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    expires_at: datetime
    generation: Generation


class DocumentRegistration(DomainModel):
    tenant_id: TenantId
    document_id: DocumentId
    corpus: Corpus
    object_uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    object_generation: ObjectGeneration
    checksum_sha256: Checksum
    media_type: MediaType
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def canonical_uri(self) -> Self:
        if not _URI.fullmatch(self.object_uri):
            raise ValueError("object_uri must be canonical")
        return self


class DocumentRecord(DocumentRegistration):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    generation: Generation
    status: DocumentStatus
    tombstoned: bool
    created_at: datetime
    updated_at: datetime
    chunk_count: Annotated[int, Field(ge=0)] | None = None
    expected_chunk_ids_sha256: Checksum | None = None
    parser_version: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None
    chunker_version: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None
    embedding_model_id: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = (
        None
    )
    embedding_dimensions: Annotated[int, Field(ge=1)] | None = None
    active_index_version: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = (
        None
    )
    lease: Lease | None = None
    failure: SanitizedFailure | None = None
    deleted_at: datetime | None = None
    ready_at: datetime | None = None

    @model_validator(mode="after")
    def state_fields(self) -> Self:
        if self.status is DocumentStatus.READY:
            required = (
                self.chunk_count,
                self.expected_chunk_ids_sha256,
                self.parser_version,
                self.chunker_version,
                self.embedding_model_id,
                self.embedding_dimensions,
                self.active_index_version,
                self.ready_at,
            )
            if any(value is None for value in required) or self.tombstoned:
                raise ValueError("READY requires compatibility fields and no tombstone")
        if self.status is DocumentStatus.PROCESSING and self.lease is None:
            raise ValueError("PROCESSING requires a lease")
        if self.status in {DocumentStatus.FAILED_RETRYABLE, DocumentStatus.FAILED_TERMINAL}:
            if self.failure is None:
                raise ValueError("failed state requires sanitized failure")
        if self.tombstoned != (
            self.status in {DocumentStatus.DELETE_PENDING, DocumentStatus.DELETED}
        ):
            raise ValueError("tombstone and deletion state must agree")
        if self.tombstoned and self.deleted_at is None:
            raise ValueError("tombstone requires deleted_at")
        if (
            self.status is DocumentStatus.DELETE_PENDING
            and self.failure is not None
            and not self.failure.retryable
        ):
            raise ValueError("DELETE_PENDING failure must be retryable")
        return self


class ChunkRecord(DomainModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    tenant_id: TenantId
    corpus: Corpus
    document_id: DocumentId
    generation: Generation
    object_generation: ObjectGeneration
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    ordinal: Annotated[int, Field(ge=0)]
    text: Annotated[str, StringConstraints(min_length=1)]
    text_sha256: Checksum
    vector_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    parser_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    chunker_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    staged_at: datetime
    created_at: datetime
    source_page: Annotated[int, Field(ge=1)] | None = None
    source_start: Annotated[int, Field(ge=0)] | None = None
    source_end: Annotated[int, Field(ge=0)] | None = None


class RegistrationIdempotencyRecord(DomainModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    tenant_id: TenantId
    idempotency_key_sha256: Checksum
    request_sha256: Checksum
    document_id: DocumentId
    generation: Generation
    event_id: UUID
    outcome: RegistrationOutcome
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None = None

    @model_validator(mode="after")
    def publication_fields(self) -> Self:
        if (self.outcome is RegistrationOutcome.PUBLISHED) != (self.published_at is not None):
            raise ValueError("PUBLISHED outcome and published_at must agree")
        return self


class RegistrationResult(DomainModel):
    idempotency: RegistrationIdempotencyRecord
    document: DocumentRecord | None = None

    @model_validator(mode="after")
    def matching_result(self) -> Self:
        if self.document is not None and (
            self.document.tenant_id != self.idempotency.tenant_id
            or self.document.document_id != self.idempotency.document_id
            or self.document.generation != self.idempotency.generation
        ):
            raise ValueError("registration document and idempotency result must agree")
        return self


class MessageMetadata(DomainModel):
    event_id: UUID
    occurred_at: datetime
    producer: Literal["rag-api"] = "rag-api"


class IngestionMessage(DomainModel):
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    operation: IngestionOperation
    tenant_id: TenantId
    document_id: DocumentId
    object_uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    object_generation: ObjectGeneration
    generation: Generation
    checksum_sha256: Checksum
    corpus: Corpus
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    message_metadata: MessageMetadata

    @model_validator(mode="after")
    def canonical_uri(self) -> Self:
        if not _URI.fullmatch(self.object_uri):
            raise ValueError("object_uri must be canonical")
        return self


class IdempotencyRecord(DomainModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    tenant_id: TenantId
    event_id: UUID
    operation: IngestionOperation
    document_id: DocumentId
    generation: Generation
    payload_sha256: Checksum
    outcome: IdempotencyOutcome
    created_at: datetime
    expires_at: datetime
    updated_at: datetime | None = None


class Claim(DomainModel):
    tenant_id: TenantId
    event_id: UUID
    operation: IngestionOperation
    document_id: DocumentId
    generation: Generation
    owner: str
    lease_expires_at: datetime


class ClaimResult(DomainModel):
    disposition: ClaimDisposition
    claim: Claim | None = None


class ActiveIndexRecord(DomainModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    index_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    parser_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    chunker_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    embedding_model_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    embedding_dimensions: Annotated[int, Field(ge=1)]
    updated_at: datetime
    vector_endpoint_resource: (
        Annotated[str, StringConstraints(min_length=1, max_length=1024)] | None
    ) = None
    deployed_index_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None


class RetrievalCandidate(DomainModel):
    vector_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    score: float
    document_id: DocumentId
    generation: Generation
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def finite_score(self) -> Self:
        if not math.isfinite(self.score):
            raise ValueError("candidate score must be finite")
        return self


class Citation(DomainModel):
    document_id: DocumentId
    generation: Generation
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    quote: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class GroundedAnswer(DomainModel):
    outcome: Literal["ANSWER"] = "ANSWER"
    answer: Annotated[str, StringConstraints(min_length=1)]
    citations: Annotated[list[Citation], Field(min_length=1)]
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class Abstention(DomainModel):
    outcome: Literal["ABSTAIN"] = "ABSTAIN"
    reason: AbstentionReason
    citations: list[Citation] = Field(default_factory=list, max_length=0)
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class AuthorizedContext(DomainModel):
    citation_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    text: Annotated[str, StringConstraints(min_length=1)]


class GeneratedCitation(DomainModel):
    citation_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    quote: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class GeneratedAnswer(DomainModel):
    answer: Annotated[str, StringConstraints(min_length=1)]
    citations: list[GeneratedCitation]


class ObjectMetadata(DomainModel):
    canonical_uri: str
    bucket: str
    name: str
    object_generation: ObjectGeneration
    size: Annotated[int, Field(ge=0)]
    content_type: MediaType
    checksum_sha256: Checksum
    updated_at: datetime


class DeleteResult(DomainModel):
    deleted: bool


class PublishReceipt(DomainModel):
    message_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class VectorRestrictions(DomainModel):
    tenant_id: str
    corpus: str
    index_version: str
    active_generations: tuple[str, ...]


class VectorDatapoint(DomainModel):
    vector_id: str
    vector: Vector
    tenant_id: str
    corpus: str
    document_id: str
    generation: Generation
    active_generation: str
    index_version: str


class Provenance(DomainModel):
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    parser_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    chunker_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    embedding_model_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    embedding_dimensions: Annotated[int, Field(ge=1)]
    active_index_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class StagedChunkResult(DomainModel):
    staged_count: Annotated[int, Field(ge=0)]
    expected_chunk_count: Annotated[int, Field(ge=0)]
    expected_chunk_ids_sha256: Checksum


class ChunkPointer(DomainModel):
    chunk_id: str
    generation: Generation
    ordinal: Annotated[int, Field(ge=0)]
    vector_id: str


class ChunkPage(DomainModel):
    items: list[ChunkPointer]
    next_cursor: str | None = None


class VerificationResult(DomainModel):
    verified: bool
    expected_count: Annotated[int, Field(ge=0)]
    checked_vector_ids: tuple[str, ...]
    missing_vector_ids: tuple[str, ...] = ()
    mismatched_vector_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent_verification(self) -> Self:
        if self.verified and (self.missing_vector_ids or self.mismatched_vector_ids):
            raise ValueError("verified result cannot contain failures")
        return self
