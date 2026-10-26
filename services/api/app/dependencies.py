"""Injectable API adapter bundle and API-owned policy limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from packages.rag_core.firestore_store import MetadataStore
from packages.rag_core.retrieval import RetrievalService
from packages.rag_core.storage import EventPublisher, ObjectStore


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    def new_event_id(self) -> UUID: ...

    def new_trace_id(self) -> str: ...


@dataclass(frozen=True)
class ApiPolicy:
    """Isolated local authentication policy."""

    local_test_identity: str = "local-rag-tester"

    def __post_init__(self) -> None:
        if not self.local_test_identity:
            raise ValueError("local test identity must be non-empty")


@dataclass(frozen=True)
class ApiDependencies:
    metadata: MetadataStore
    objects: ObjectStore
    publisher: EventPublisher
    retrieval: RetrievalService
    clock: Clock
    ids: IdGenerator
    policy: ApiPolicy
