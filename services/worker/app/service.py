"""Bounded, idempotent ingestion orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlsplit

from packages.rag_core.chunking import (
    CHUNKER_IMPLEMENTATION_VERSION,
    chunk_sections,
)
from packages.rag_core.config import RuntimeSettings
from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import (
    ActiveIndexRecord,
    ChunkRecord,
    Claim,
    ClaimDisposition,
    IngestionMessage,
    IngestionOperation,
    Provenance,
    SanitizedFailure,
    VectorDatapoint,
    chunk_ids_sha256,
)
from packages.rag_core.parsing import PARSER_IMPLEMENTATION_VERSION, parse_document
from packages.rag_core.policy import GcsObjectPolicy, build_datapoint, require_tenant
from packages.rag_core.telemetry import resource_hash, safe_fields
from services.worker.app.dependencies import WorkerDependencies


@dataclass(frozen=True, slots=True)
class ProcessResult:
    disposition: str
    count: int = 0


class _Deadline:
    def __init__(self, dependencies: WorkerDependencies, seconds: float) -> None:
        self._dependencies = dependencies
        self._end = dependencies.clock.monotonic() + seconds

    def remaining(self, maximum: float | None = None) -> float:
        remaining = self._end - self._dependencies.clock.monotonic()
        if remaining <= 0:
            raise RagError(ErrorCode.DEPENDENCY_TIMEOUT)
        return remaining if maximum is None else min(remaining, maximum)


class IngestionService:
    """Application service whose cloud-facing dependencies are all injected."""

    def __init__(
        self,
        *,
        settings: RuntimeSettings,
        dependencies: WorkerDependencies,
    ) -> None:
        self.settings = settings
        self.dependencies = dependencies
        self.object_policy = GcsObjectPolicy(
            bucket=settings.gcs_document_bucket,
            tenant_id=settings.dev_tenant_id,
            max_bytes=settings.max_document_bytes,
        )

    def _validate_message(self, payload: IngestionMessage) -> None:
        require_tenant(payload.tenant_id, self.settings.dev_tenant_id)
        if (
            payload.pipeline_version != self.settings.pipeline_version
            or payload.corpus != self.settings.fixed_corpus
            or self.settings.parser_version != PARSER_IMPLEMENTATION_VERSION
            or self.settings.chunker_version != CHUNKER_IMPLEMENTATION_VERSION
        ):
            raise RagError(ErrorCode.CONFIGURATION_INVALID)
        _, name = self.object_policy.parse_uri(payload.object_uri)
        prefix = f"{self.settings.tenant_upload_prefix}/"
        if not name.startswith(prefix):
            raise RagError(ErrorCode.ACCESS_DENIED)
        parts = urlsplit(payload.object_uri)
        if parts.path != f"/{name}":
            raise RagError(ErrorCode.ACCESS_DENIED)

    def _active_index(self, deadline: _Deadline) -> ActiveIndexRecord:
        deadline.remaining(self.settings.dependency_timeout_seconds)
        index = self.dependencies.metadata.get_active_index()
        if (
            index.pipeline_version != self.settings.pipeline_version
            or index.parser_version != self.settings.parser_version
            or index.chunker_version != self.settings.chunker_version
            or index.embedding_model_id != self.settings.embedding_model_id
            or index.embedding_dimensions != self.settings.embedding_dimensions
        ):
            raise RagError(ErrorCode.CONFIGURATION_INVALID)
        return index

    def _embed_batch(self, texts: list[str], deadline: _Deadline) -> list[list[float]]:
        retries = 0
        while True:
            try:
                vectors = self.dependencies.embeddings.embed(
                    texts,
                    self.settings.embedding_dimensions,
                    deadline.remaining(self.settings.dependency_timeout_seconds),
                )
                if len(vectors) != len(texts) or any(
                    len(vector) != self.settings.embedding_dimensions
                    or any(not math.isfinite(value) for value in vector)
                    for vector in vectors
                ):
                    raise RagError(ErrorCode.DEPENDENCY_REJECTED)
                return vectors
            except RagError as error:
                if not error.retryable or retries >= self.settings.embedding_max_retries:
                    raise
                retries += 1
                delay = max(0.0, self.dependencies.jitter(retries))
                if delay:
                    if delay >= deadline.remaining():
                        raise RagError(ErrorCode.DEPENDENCY_TIMEOUT) from error
                    self.dependencies.sleep(delay)

    def _upsert(self, payload: IngestionMessage, claim: Claim, deadline: _Deadline) -> int:
        metadata = self.dependencies.objects.stat(
            payload.object_uri,
            deadline.remaining(self.settings.dependency_timeout_seconds),
        )
        self.object_policy.validate(
            metadata,
            expected_uri=payload.object_uri,
            expected_object_generation=payload.object_generation,
            expected_checksum=payload.checksum_sha256,
        )
        data = self.dependencies.objects.read_bounded(
            payload.object_uri,
            payload.object_generation,
            self.settings.max_document_bytes,
            deadline.remaining(self.settings.dependency_timeout_seconds),
        )
        sections = parse_document(
            data,
            metadata.content_type,
            max_pdf_pages=self.settings.max_pdf_pages,
            max_chars=self.settings.max_document_bytes,
        )
        chunks = chunk_sections(
            sections,
            tenant_id=payload.tenant_id,
            document_id=payload.document_id,
            generation=payload.generation,
            object_generation=payload.object_generation,
            pipeline_version=payload.pipeline_version,
            chunker_version=self.settings.chunker_version,
            chunk_size_tokens=self.settings.chunk_size_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
            max_chunks=self.settings.max_document_chunks,
        )
        now = self.dependencies.clock.now()
        records = [
            ChunkRecord(
                tenant_id=payload.tenant_id,
                corpus=payload.corpus,
                document_id=payload.document_id,
                generation=payload.generation,
                object_generation=payload.object_generation,
                chunk_id=chunk.chunk_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                text_sha256=chunk.text_sha256,
                vector_id=chunk.vector_id,
                pipeline_version=payload.pipeline_version,
                parser_version=self.settings.parser_version,
                chunker_version=self.settings.chunker_version,
                staged_at=now,
                created_at=now,
                source_page=chunk.source_page,
                source_start=chunk.source_start,
                source_end=chunk.source_end,
            )
            for chunk in chunks
        ]
        expected_ids = [record.chunk_id for record in records]
        expected_hash = chunk_ids_sha256(expected_ids)
        batch_size = self.settings.max_embedding_batch_size
        for offset in range(0, len(records), batch_size):
            deadline.remaining()
            self.dependencies.metadata.stage_chunks(
                claim,
                records[offset : offset + batch_size],
                expected_hash,
                len(records),
                self.dependencies.clock.now(),
            )

        index = self._active_index(deadline)
        datapoints: list[VectorDatapoint] = []
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            vectors = self._embed_batch([record.text for record in batch], deadline)
            datapoints.extend(
                build_datapoint(
                    vector_id=record.vector_id,
                    vector=vector,
                    tenant_id=record.tenant_id,
                    corpus=record.corpus,
                    document_id=record.document_id,
                    generation=record.generation,
                    index_version=index.index_version,
                )
                for record, vector in zip(batch, vectors, strict=True)
            )
        for offset in range(0, len(datapoints), batch_size):
            self.dependencies.vectors.upsert(
                datapoints[offset : offset + batch_size],
                deadline.remaining(self.settings.dependency_timeout_seconds),
            )
        vector_ids = [item.vector_id for item in datapoints]
        verification = self.dependencies.vectors.verify(
            vector_ids,
            len(vector_ids),
            min(len(vector_ids), 10),
            deadline.remaining(self.settings.dependency_timeout_seconds),
        )
        if not verification.verified:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        self.dependencies.metadata.commit_ready(
            claim,
            expected_ids,
            len(expected_ids),
            Provenance(
                pipeline_version=self.settings.pipeline_version,
                parser_version=self.settings.parser_version,
                chunker_version=self.settings.chunker_version,
                embedding_model_id=self.settings.embedding_model_id,
                embedding_dimensions=self.settings.embedding_dimensions,
                active_index_version=index.index_version,
            ),
            verification,
            self.dependencies.clock.now(),
        )
        return len(records)

    def _delete(self, payload: IngestionMessage, claim: Claim, deadline: _Deadline) -> int:
        cursor: str | None = None
        deleted_count = 0
        batch_size = self.settings.max_embedding_batch_size
        while True:
            deadline.remaining()
            page = self.dependencies.metadata.list_document_chunks(
                payload.tenant_id,
                payload.document_id,
                cursor,
                batch_size,
            )
            if page.items:
                self.dependencies.vectors.delete(
                    [item.vector_id for item in page.items],
                    deadline.remaining(self.settings.dependency_timeout_seconds),
                )
                deleted_count += len(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        self.dependencies.objects.delete(
            payload.object_uri,
            payload.object_generation,
            deadline.remaining(self.settings.dependency_timeout_seconds),
        )
        self.dependencies.metadata.commit_deleted(
            claim,
            self.dependencies.clock.now(),
        )
        return deleted_count

    def _record_failure(
        self,
        *,
        claim: Claim,
        payload: IngestionMessage,
        error: RagError,
        attempt: int,
    ) -> None:
        now = self.dependencies.clock.now()
        failure = SanitizedFailure(
            code=error.code.value,
            retryable=error.retryable,
            attempts=attempt,
            last_failed_at=now,
        )
        if payload.operation is IngestionOperation.DELETE:
            if error.retryable:
                self.dependencies.metadata.record_retryable_delete_failure(claim, failure, now)
            else:
                # DELETE terminal failures cannot clear a tombstone or become READY.
                retry_failure = failure.model_copy(
                    update={
                        "code": ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                        "retryable": True,
                    }
                )
                self.dependencies.metadata.record_retryable_delete_failure(
                    claim, retry_failure, now
                )
            return
        if error.retryable:
            self.dependencies.metadata.commit_retryable_failure(claim, failure, now)
        else:
            self.dependencies.metadata.commit_terminal_failure(claim, failure, now)

    def process(self, payload: IngestionMessage, *, attempt: int) -> ProcessResult:
        """Validate, claim, and complete one message within a monotonic deadline."""
        started = self.dependencies.clock.monotonic()
        self._validate_message(payload)
        deadline = _Deadline(self.dependencies, self.settings.processing_deadline_seconds)
        now = self.dependencies.clock.now()
        claim_result = self.dependencies.metadata.claim_event(
            payload,
            self.dependencies.owner,
            now + timedelta(seconds=self.settings.processing_lease_seconds),
            now,
        )
        if claim_result.disposition is not ClaimDisposition.CLAIMED:
            return ProcessResult(disposition=claim_result.disposition.value)
        if claim_result.claim is None:
            raise RagError(ErrorCode.INTERNAL)
        claim = claim_result.claim
        try:
            count = (
                self._upsert(payload, claim, deadline)
                if payload.operation is IngestionOperation.UPSERT
                else self._delete(payload, claim, deadline)
            )
        except RagError as error:
            self._record_failure(
                claim=claim,
                payload=payload,
                error=error,
                attempt=attempt,
            )
            if payload.operation is IngestionOperation.DELETE and not error.retryable:
                raise RetryableDependencyError() from error
            raise
        except Exception as error:
            internal = RagError(ErrorCode.INTERNAL)
            self._record_failure(
                claim=claim,
                payload=payload,
                error=internal,
                attempt=attempt,
            )
            if payload.operation is IngestionOperation.DELETE:
                raise RetryableDependencyError() from error
            raise internal from error
        self.dependencies.telemetry.emit(
            "worker_ingestion_complete",
            safe_fields(
                trace_id=payload.trace_id,
                resource_hash=resource_hash(payload.document_id, salt=b"worker-v1"),
                operation=payload.operation.value,
                status="SUCCEEDED",
                count=count,
                attempt=attempt,
                pipeline_version=payload.pipeline_version,
                latency_ms=int((self.dependencies.clock.monotonic() - started) * 1000),
            ),
        )
        return ProcessResult(disposition="SUCCEEDED", count=count)
