"""Strict HTTP request and response schemas for API contract v1."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from packages.rag_core.models import (
    DocumentId,
    DocumentRecord,
    DocumentStatus,
    MediaType,
)
from packages.rag_core.retrieval import RetrievalResult


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceUri = Annotated[str, StringConstraints(max_length=2048, pattern=r"^gs://[^\s]+$")]
Version = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class RegisterDocumentRequest(ApiModel):
    document_id: DocumentId
    corpus: Literal["default"] = "default"
    object_uri: Annotated[
        str,
        StringConstraints(min_length=1, max_length=2048, pattern=r"^gs://[^\s]+$"),
    ]
    object_generation: Annotated[int, Field(ge=1)]
    media_type: MediaType
    checksum_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    idempotency_key: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class QueryRequest(ApiModel):
    corpus: Literal["default"] = "default"
    question: Annotated[str, StringConstraints(min_length=1, max_length=16_000)]
    top_k: Annotated[int, Field(ge=1, le=50)] = 8


class FailureResponse(ApiModel):
    code: str
    retryable: bool


class DocumentResponse(ApiModel):
    document_id: DocumentId
    corpus: Literal["default"]
    source_uri: SourceUri
    object_generation: Annotated[int, Field(ge=1)]
    generation: Annotated[int, Field(ge=1)]
    status: DocumentStatus
    tombstoned: bool
    created_at: datetime
    updated_at: datetime
    ready_at: datetime | None
    deleted_at: datetime | None
    parser_version: Version | None
    chunker_version: Version | None
    embedding_model_version: Annotated[str, StringConstraints(max_length=256)] | None
    pipeline_version: Version
    index_version: Version | None
    chunk_count: Annotated[int, Field(ge=0)] | None
    failure: FailureResponse | None

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentResponse:
        failure = None
        if record.failure is not None:
            failure = FailureResponse(code=record.failure.code, retryable=record.failure.retryable)
        return cls(
            document_id=record.document_id,
            corpus=cast(Literal["default"], record.corpus),
            source_uri=record.object_uri,
            object_generation=record.object_generation,
            generation=record.generation,
            status=record.status,
            tombstoned=record.tombstoned,
            created_at=record.created_at,
            updated_at=record.updated_at,
            ready_at=record.ready_at,
            deleted_at=record.deleted_at,
            parser_version=record.parser_version,
            chunker_version=record.chunker_version,
            embedding_model_version=record.embedding_model_id,
            pipeline_version=record.pipeline_version,
            index_version=record.active_index_version,
            chunk_count=record.chunk_count,
            failure=failure,
        )


class ProblemResponse(ApiModel):
    type: str
    title: str
    status: int
    code: str
    detail: str | None = None
    trace_id: str
    request_id: str
    retryable: bool


class ProbeResponse(ApiModel):
    status: Literal["ok"]


class CitationResponse(ApiModel):
    document_id: DocumentId
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    source_uri: SourceUri
    page: Annotated[int, Field(ge=1)] | None = None
    quote: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class QueryResponse(ApiModel):
    answer: str
    citations: list[CitationResponse]
    abstained: bool
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    model_version: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    index_generation: Version

    @classmethod
    def from_result(cls, result: RetrievalResult) -> QueryResponse:
        return cls(
            answer=result.answer,
            citations=[
                CitationResponse(
                    document_id=citation.document_id,
                    chunk_id=citation.chunk_id,
                    source_uri=citation.source_uri,
                    page=citation.page,
                    quote=citation.quote,
                )
                for citation in result.citations
            ],
            abstained=result.abstained,
            trace_id=result.trace_id,
            model_version=result.model_version,
            index_generation=result.index_generation,
        )
