"""Generation port and regional Gemini implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import AuthorizedContext, GeneratedAnswer


class Generator(Protocol):
    def generate(
        self,
        question: str,
        authorized_context: Sequence[AuthorizedContext],
        max_output_tokens: int,
        deadline: float,
    ) -> GeneratedAnswer:
        """Generate only from non-empty authorized context."""


class VertexGenerator:
    """Regional Gemini adapter returning a validated structured response."""

    def __init__(self, *, client: Any, model_id: str, region: str) -> None:
        if not region or region == "global" or not model_id or "latest" in model_id.lower():
            raise ValueError("regional endpoint and pinned model ID are required")
        self.client = client
        self.model_id = model_id
        self.region = region
        self.calls: list[dict[str, object]] = []

    @classmethod  # pragma: no cover - credential-bearing factory
    def from_project(cls, *, project: str, region: str, model_id: str) -> VertexGenerator:
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=project,
            location=region,
            http_options=types.HttpOptions(api_version="v1"),
        )
        return cls(client=client, model_id=model_id, region=region)

    def generate(
        self,
        question: str,
        authorized_context: Sequence[AuthorizedContext],
        max_output_tokens: int,
        deadline: float,
    ) -> GeneratedAnswer:
        if not question or not authorized_context or max_output_tokens < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        context = [
            {"citation_id": item.citation_id, "text": item.text} for item in authorized_context
        ]
        prompt = (
            "Answer only from the evidence. Return JSON with answer and citations "
            "(citation_id and exact quote).\n"
            f"Question: {question}\nEvidence: {json.dumps(context, ensure_ascii=False)}"
        )
        request = {
            "model": self.model_id,
            "region": self.region,
            "max_output_tokens": max_output_tokens,
            "context_ids": [item.citation_id for item in authorized_context],
            "deadline": deadline,
        }
        self.calls.append(request)
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": GeneratedAnswer.model_json_schema(),
                    "temperature": 0,
                },
            )
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        try:
            return GeneratedAnswer.model_validate_json(response.text)
        except (ValueError, TypeError, AttributeError) as error:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED) from error
