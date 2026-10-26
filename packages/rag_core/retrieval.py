"""Fail-closed retrieval, evidence selection, and grounding verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from packages.rag_core.embeddings import EmbeddingModel
from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.firestore_store import MetadataStore
from packages.rag_core.generation import Generator
from packages.rag_core.models import (
    AbstentionReason,
    AuthorizedContext,
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
)
from packages.rag_core.policy import build_search_restrictions
from packages.rag_core.vector_store import VectorIndex


@dataclass(frozen=True)
class RetrievalLimits:
    """All query-side bounds not represented by adapter protocols."""

    max_active_generations: int
    max_candidates_per_document: int
    max_context_chars: int
    max_context_tokens: int
    max_evidence_chars: int
    max_output_tokens: int
    dependency_timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            min(
                self.max_active_generations,
                self.max_candidates_per_document,
                self.max_context_chars,
                self.max_context_tokens,
                self.max_evidence_chars,
                self.max_output_tokens,
            )
            < 1
            or self.dependency_timeout_seconds <= 0
        ):
            raise ValueError("retrieval limits must be positive")


@dataclass(frozen=True)
class VerifiedCitation:
    document_id: str
    chunk_id: str
    source_uri: str
    page: int | None
    quote: str


@dataclass(frozen=True)
class RetrievalResult:
    answer: str
    citations: tuple[VerifiedCitation, ...]
    abstained: bool
    trace_id: str
    model_version: str
    index_generation: str
    reason: AbstentionReason | None = None


@dataclass(frozen=True)
class _Evidence:
    chunk: ChunkRecord
    document: DocumentRecord


class RetrievalService:
    """Coordinates only authorized evidence through generation."""

    def __init__(
        self,
        *,
        metadata: MetadataStore,
        embeddings: EmbeddingModel,
        vectors: VectorIndex,
        generator: Generator,
        tenant_id: str,
        pipeline_version: str,
        parser_version: str,
        chunker_version: str,
        generation_model_id: str,
        embedding_model_id: str,
        embedding_dimensions: int,
        limits: RetrievalLimits,
    ) -> None:
        self.metadata = metadata
        self.embeddings = embeddings
        self.vectors = vectors
        self.generator = generator
        self.tenant_id = tenant_id
        self.pipeline_version = pipeline_version
        self.parser_version = parser_version
        self.chunker_version = chunker_version
        self.generation_model_id = generation_model_id
        self.embedding_model_id = embedding_model_id
        self.embedding_dimensions = embedding_dimensions
        self.limits = limits

    def query(self, *, question: str, corpus: str, top_k: int, trace_id: str) -> RetrievalResult:
        """Return a verified grounded answer or an explicit safe abstention."""
        normalized = " ".join(question.split())
        if not normalized or top_k < 1:
            raise RagError(ErrorCode.INVALID_REQUEST)

        index = self.metadata.get_active_index()
        if (
            index.pipeline_version != self.pipeline_version
            or index.parser_version != self.parser_version
            or index.chunker_version != self.chunker_version
            or index.embedding_model_id != self.embedding_model_id
            or index.embedding_dimensions != self.embedding_dimensions
        ):
            raise RagError(ErrorCode.CONFIGURATION_INVALID)

        active = self.metadata.list_active_generations(
            self.tenant_id, corpus, self.limits.max_active_generations
        )
        if not active:
            return self._abstain(
                AbstentionReason.NO_AUTHORIZED_EVIDENCE,
                trace_id,
                index.index_version,
            )

        # Construction is deliberately completed before embedding/search. It is
        # the final guard against an accidentally unscoped vector request.
        restrictions = build_search_restrictions(
            tenant_id=self.tenant_id,
            corpus=corpus,
            index_version=index.index_version,
            active_generations=active,
        )
        vector = self.embeddings.embed(
            [normalized],
            self.embedding_dimensions,
            self.limits.dependency_timeout_seconds,
        )[0]
        candidates = self.vectors.search(
            vector,
            restrictions,
            top_k,
            self.limits.dependency_timeout_seconds,
        )
        if not candidates:
            return self._abstain(
                AbstentionReason.NO_AUTHORIZED_EVIDENCE,
                trace_id,
                index.index_version,
            )

        hydrated = self.metadata.hydrate_authorized(
            self.tenant_id, corpus, candidates, index.index_version
        )
        allowed = {
            (
                candidate.document_id,
                candidate.generation,
                candidate.chunk_id,
                candidate.vector_id,
            )
            for candidate in candidates
        }
        reauthorized = [
            chunk
            for chunk in hydrated
            if chunk.tenant_id == self.tenant_id
            and chunk.corpus == corpus
            and chunk.pipeline_version == index.pipeline_version
            and chunk.parser_version == index.parser_version
            and chunk.chunker_version == index.chunker_version
            and (
                chunk.document_id,
                chunk.generation,
                chunk.chunk_id,
                chunk.vector_id,
            )
            in allowed
        ]
        evidence = self._bound_evidence(reauthorized)
        if not evidence:
            return self._abstain(
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                trace_id,
                index.index_version,
            )

        authorized = self._authorize_sources(evidence, corpus, index.index_version)
        if not authorized:
            return self._abstain(
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                trace_id,
                index.index_version,
            )
        contexts, lookup = self._contexts(authorized)
        generated = self.generator.generate(
            normalized,
            contexts,
            self.limits.max_output_tokens,
            self.limits.dependency_timeout_seconds,
        )
        citations: list[VerifiedCitation] = []
        seen: set[str] = set()
        for generated_citation in generated.citations:
            evidence_item = lookup.get(generated_citation.citation_id)
            quote = generated_citation.quote.strip()
            if evidence_item is None or not quote or quote not in evidence_item.chunk.text:
                return self._abstain(
                    AbstentionReason.CITATION_VALIDATION_FAILED,
                    trace_id,
                    index.index_version,
                )
            if generated_citation.citation_id in seen:
                continue
            seen.add(generated_citation.citation_id)
            citations.append(
                VerifiedCitation(
                    document_id=evidence_item.chunk.document_id,
                    chunk_id=evidence_item.chunk.chunk_id,
                    source_uri=evidence_item.document.object_uri,
                    page=evidence_item.chunk.source_page,
                    quote=quote,
                )
            )
        if not citations:
            return self._abstain(
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                trace_id,
                index.index_version,
            )
        return RetrievalResult(
            answer=generated.answer,
            citations=tuple(citations),
            abstained=False,
            trace_id=trace_id,
            model_version=self.generation_model_id,
            index_generation=index.index_version,
        )

    def _bound_evidence(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        selected: list[ChunkRecord] = []
        per_document: dict[str, int] = {}
        ordinals: dict[str, set[int]] = {}
        total_chars = 0
        total_tokens = 0
        for chunk in chunks:
            count = per_document.get(chunk.document_id, 0)
            used_ordinals = ordinals.setdefault(chunk.document_id, set())
            if count >= self.limits.max_candidates_per_document:
                continue
            if chunk.ordinal - 1 in used_ordinals or chunk.ordinal + 1 in used_ordinals:
                continue
            chars = len(chunk.text)
            tokens = (chars + 3) // 4
            if (
                total_chars + chars > self.limits.max_context_chars
                or total_chars + chars > self.limits.max_evidence_chars
                or total_tokens + tokens > self.limits.max_context_tokens
            ):
                continue
            selected.append(chunk)
            per_document[chunk.document_id] = count + 1
            used_ordinals.add(chunk.ordinal)
            total_chars += chars
            total_tokens += tokens
        return selected

    def _authorize_sources(
        self, chunks: list[ChunkRecord], corpus: str, index_version: str
    ) -> list[_Evidence]:
        documents: dict[str, DocumentRecord | None] = {}
        authorized: list[_Evidence] = []
        for chunk in chunks:
            if chunk.document_id not in documents:
                documents[chunk.document_id] = self.metadata.get_document(
                    self.tenant_id, chunk.document_id
                )
            document = documents[chunk.document_id]
            if (
                document is None
                or document.tenant_id != self.tenant_id
                or document.corpus != corpus
                or document.generation != chunk.generation
                or document.object_generation != chunk.object_generation
                or document.status is not DocumentStatus.READY
                or document.tombstoned
                or document.pipeline_version != self.pipeline_version
                or document.parser_version != self.parser_version
                or document.chunker_version != self.chunker_version
                or document.active_index_version != index_version
            ):
                continue
            authorized.append(_Evidence(chunk=chunk, document=document))
        return authorized

    @staticmethod
    def _contexts(
        evidence: list[_Evidence],
    ) -> tuple[list[AuthorizedContext], dict[str, _Evidence]]:
        contexts: list[AuthorizedContext] = []
        lookup: dict[str, _Evidence] = {}
        for item in evidence:
            chunk = item.chunk
            material = f"{chunk.document_id}\0{chunk.generation}\0{chunk.chunk_id}".encode()
            citation_id = hashlib.sha256(material).hexdigest()
            contexts.append(AuthorizedContext(citation_id=citation_id, text=chunk.text))
            lookup[citation_id] = item
        return contexts, lookup

    def _abstain(
        self, reason: AbstentionReason, trace_id: str, index_generation: str
    ) -> RetrievalResult:
        return RetrievalResult(
            answer="",
            citations=(),
            abstained=True,
            trace_id=trace_id,
            model_version=self.generation_model_id,
            index_generation=index_generation,
            reason=reason,
        )
