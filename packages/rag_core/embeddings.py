"""Embedding port and regional Vertex AI implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import Vector


class EmbeddingModel(Protocol):
    def embed(self, texts: Sequence[str], dimensions: int, deadline: float) -> list[Vector]:
        """Embed a bounded non-empty batch in input order."""


class VertexEmbeddingModel:
    """Regional Vertex embedding adapter using an injected google-genai client."""

    def __init__(
        self,
        *,
        client: Any,
        model_id: str,
        region: str,
        max_batch_size: int,
    ) -> None:
        if not region or region == "global" or not model_id or "latest" in model_id.lower():
            raise ValueError("regional endpoint and pinned model ID are required")
        self.client = client
        self.model_id = model_id
        self.region = region
        self.max_batch_size = max_batch_size
        self.calls: list[dict[str, object]] = []

    @classmethod  # pragma: no cover - credential-bearing factory
    def from_project(
        cls,
        *,
        project: str,
        region: str,
        model_id: str,
        max_batch_size: int,
    ) -> VertexEmbeddingModel:
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=project,
            location=region,
            http_options=types.HttpOptions(api_version="v1"),
        )
        return cls(
            client=client,
            model_id=model_id,
            region=region,
            max_batch_size=max_batch_size,
        )

    def embed(self, texts: Sequence[str], dimensions: int, deadline: float) -> list[Vector]:
        if not texts or len(texts) > self.max_batch_size or dimensions < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if any(not text for text in texts):
            raise RagError(ErrorCode.INVALID_REQUEST)
        request = {
            "model": self.model_id,
            "contents": list(texts),
            "config": {"output_dimensionality": dimensions},
            "region": self.region,
            "deadline": deadline,
        }
        self.calls.append(request)
        try:
            response = self.client.models.embed_content(
                model=self.model_id,
                contents=list(texts),
                config={"output_dimensionality": dimensions},
            )
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        vectors = [list(item.values) for item in embeddings]
        if len(vectors) != len(texts) or any(len(vector) != dimensions for vector in vectors):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return vectors
