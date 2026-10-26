"""Bounded deterministic parsing for worker-supported document types."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader, apply_configuration
from pypdf.errors import LimitReachedError, PdfReadError, PyPdfError

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import MediaType

PARSER_IMPLEMENTATION_VERSION = "parser-1"
_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Normalized text and source coordinates from one logical source section."""

    text: str
    source_page: int | None


def normalize_text(value: str) -> str:
    """Normalize line endings and horizontal whitespace without changing words."""
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_WHITESPACE.sub(" ", line).strip() for line in value.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT) from error


def _bounded_section(text: str, *, page: int | None, remaining_chars: int) -> ParsedSection:
    normalized = normalize_text(text)
    if not normalized:
        return ParsedSection(text="", source_page=page)
    if len(normalized) > remaining_chars:
        raise RagError(ErrorCode.LIMIT_EXCEEDED)
    return ParsedSection(text=normalized, source_page=page)


def parse_document(
    data: bytes,
    media_type: MediaType,
    *,
    max_pdf_pages: int,
    max_chars: int,
) -> list[ParsedSection]:
    """Parse a supported document with hard page and extracted-character limits."""
    if not data or max_pdf_pages < 1 or max_chars < 1:
        raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT)
    if media_type in {MediaType.TEXT, MediaType.MARKDOWN}:
        section = _bounded_section(
            _decode_utf8(data),
            page=None,
            remaining_chars=max_chars,
        )
        if not section.text:
            raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT)
        return [section]
    if media_type is not MediaType.PDF:
        raise RagError(ErrorCode.UNSUPPORTED_MEDIA_TYPE)

    try:
        with apply_configuration(
            maximum_declared_stream_length=max_chars,
            array_based_stream_maximum_output_length=max_chars,
            jbig2_maximum_output_length=max_chars,
            lzw_maximum_output_length=max_chars,
            run_length_maximum_output_length=max_chars,
            zlib_maximum_output_length=max_chars,
            zlib_maximum_recovery_input_length=max_chars,
            brotli_maximum_output_length=max_chars,
            image_maximum_buffer_size=max_chars,
            stream_decoding_work_maximum_length=max_chars * 4,
            stream_parsing_work_maximum_count=max_chars,
        ):
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted or len(reader.pages) > max_pdf_pages:
                raise RagError(ErrorCode.LIMIT_EXCEEDED)
            sections: list[ParsedSection] = []
            total_chars = 0
            for page_number, page in enumerate(reader.pages, start=1):
                section = _bounded_section(
                    page.extract_text() or "",
                    page=page_number,
                    remaining_chars=max_chars - total_chars,
                )
                if section.text:
                    total_chars += len(section.text)
                    sections.append(section)
    except RagError:
        raise
    except LimitReachedError as error:
        raise RagError(ErrorCode.LIMIT_EXCEEDED) from error
    except (PdfReadError, PyPdfError, ValueError, TypeError, KeyError, RecursionError) as error:
        raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT) from error
    if not sections:
        # A PDF with pages but no extractable text is treated as scanned input.
        raise RagError(ErrorCode.UNPROCESSABLE_DOCUMENT)
    return sections
