"""Deterministic bounded chunking and stable ingestion identifiers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.parsing import ParsedSection

CHUNKER_IMPLEMENTATION_VERSION = "chunker-1"
_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    text: str
    text_sha256: str
    chunk_id: str
    vector_id: str
    source_page: int | None
    source_start: int
    source_end: int


def _stable_id(
    *,
    tenant_id: str,
    document_id: str,
    generation: int,
    object_generation: int,
    pipeline_version: str,
    chunker_version: str,
    ordinal: int,
    text_sha256: str,
) -> str:
    material = "\0".join(
        (
            tenant_id,
            document_id,
            str(generation),
            str(object_generation),
            pipeline_version,
            chunker_version,
            str(ordinal),
            text_sha256,
        )
    ).encode()
    return hashlib.sha256(material).hexdigest()


def chunk_sections(
    sections: list[ParsedSection],
    *,
    tenant_id: str,
    document_id: str,
    generation: int,
    object_generation: int,
    pipeline_version: str,
    chunker_version: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_chunks: int,
) -> list[TextChunk]:
    """Split normalized sections by token windows without crossing page boundaries."""
    if (
        not sections
        or not tenant_id
        or not document_id
        or generation < 1
        or object_generation < 1
        or not pipeline_version
        or not chunker_version
        or chunk_size_tokens < 1
        or overlap_tokens < 0
        or overlap_tokens >= chunk_size_tokens
        or max_chunks < 1
    ):
        raise RagError(ErrorCode.INVALID_REQUEST)

    chunks: list[TextChunk] = []
    step = chunk_size_tokens - overlap_tokens
    for section in sections:
        matches = list(_TOKEN.finditer(section.text))
        for token_start in range(0, len(matches), step):
            window = matches[token_start : token_start + chunk_size_tokens]
            if not window:
                break
            source_start = window[0].start()
            source_end = window[-1].end()
            text = section.text[source_start:source_end]
            text_sha256 = hashlib.sha256(text.encode()).hexdigest()
            ordinal = len(chunks)
            identifier = _stable_id(
                tenant_id=tenant_id,
                document_id=document_id,
                generation=generation,
                object_generation=object_generation,
                pipeline_version=pipeline_version,
                chunker_version=chunker_version,
                ordinal=ordinal,
                text_sha256=text_sha256,
            )
            chunks.append(
                TextChunk(
                    ordinal=ordinal,
                    text=text,
                    text_sha256=text_sha256,
                    chunk_id=identifier,
                    vector_id=identifier,
                    source_page=section.source_page,
                    source_start=source_start,
                    source_end=source_end,
                )
            )
            if len(chunks) > max_chunks:
                raise RagError(ErrorCode.LIMIT_EXCEEDED)
            if token_start + chunk_size_tokens >= len(matches):
                break
    if not chunks:
        raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT)
    return chunks
