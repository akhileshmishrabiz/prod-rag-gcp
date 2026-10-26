"""Injectable worker business dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from packages.rag_core.embeddings import EmbeddingModel
from packages.rag_core.firestore_store import MetadataStore
from packages.rag_core.storage import ObjectStore
from packages.rag_core.telemetry import TelemetrySink
from packages.rag_core.vector_store import VectorIndex


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


@dataclass(frozen=True, slots=True)
class WorkerDependencies:
    metadata: MetadataStore
    objects: ObjectStore
    embeddings: EmbeddingModel
    vectors: VectorIndex
    clock: Clock
    telemetry: TelemetrySink
    owner: str
    sleep: Callable[[float], None]
    jitter: Callable[[int], float]
