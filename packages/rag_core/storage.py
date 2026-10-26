"""Object and publication ports with Google Cloud adapters."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
from typing import Any, Protocol

from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import (
    DeleteResult,
    IngestionMessage,
    MediaType,
    ObjectMetadata,
    PublishReceipt,
)
from packages.rag_core.policy import GcsObjectPolicy


class ObjectStore(Protocol):
    def stat(self, canonical_uri: str, deadline: float) -> ObjectMetadata: ...

    def write_new(
        self,
        *,
        object_name: str,
        data: bytes,
        content_type: MediaType,
        checksum_sha256: str,
        deadline: float,
    ) -> ObjectMetadata: ...

    def read_bounded(
        self, canonical_uri: str, object_generation: int, max_bytes: int, deadline: float
    ) -> bytes: ...

    def delete(
        self, canonical_uri: str, object_generation: int, deadline: float
    ) -> DeleteResult: ...


class EventPublisher(Protocol):
    def publish(self, payload: IngestionMessage, deadline: float) -> PublishReceipt: ...


class GcsObjectStore:
    """Canonical, generation-aware Cloud Storage access."""

    def __init__(self, *, client: Any, policy: GcsObjectPolicy) -> None:
        self.client = client
        self.policy = policy

    @classmethod  # pragma: no cover - credential-bearing factory
    def from_project(cls, *, project: str, policy: GcsObjectPolicy) -> GcsObjectStore:
        storage = importlib.import_module("google.cloud.storage")
        return cls(client=storage.Client(project=project), policy=policy)

    def _blob(self, uri: str) -> Any:
        bucket, name = self.policy.parse_uri(uri)
        return self.client.bucket(bucket).blob(name)

    def stat(self, canonical_uri: str, deadline: float) -> ObjectMetadata:
        if deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        blob = self._blob(canonical_uri)
        try:
            blob.reload(timeout=deadline, retry=None)
            metadata = getattr(blob, "metadata", None) or {}
            checksum = metadata.get("sha256")
            if not isinstance(checksum, str):
                raise RagError(ErrorCode.DEPENDENCY_REJECTED)
            return ObjectMetadata(
                canonical_uri=canonical_uri,
                bucket=blob.bucket.name,
                name=blob.name,
                object_generation=int(blob.generation),
                size=int(blob.size),
                content_type=MediaType(blob.content_type),
                checksum_sha256=checksum,
                updated_at=blob.updated,
            )
        except RagError:
            raise
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            if error.__class__.__name__ == "PreconditionFailed":
                raise RagError(ErrorCode.STALE_GENERATION) from error
            raise RetryableDependencyError() from error

    def write_new(
        self,
        *,
        object_name: str,
        data: bytes,
        content_type: MediaType,
        checksum_sha256: str,
        deadline: float,
    ) -> ObjectMetadata:
        if deadline <= 0 or not object_name or not checksum_sha256:
            raise RagError(ErrorCode.INVALID_REQUEST)
        if len(data) > self.policy.max_bytes:
            raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
        upload_prefix = f"{self.policy.prefix}uploads/"
        if not object_name.startswith(upload_prefix):
            raise RagError(ErrorCode.ACCESS_DENIED)
        canonical_uri = f"gs://{self.policy.bucket}/{object_name}"
        blob = self.client.bucket(self.policy.bucket).blob(object_name)
        try:
            blob.metadata = {"sha256": checksum_sha256}
            blob.upload_from_string(
                data,
                content_type=content_type.value,
                timeout=deadline,
                retry=None,
            )
            blob.reload(timeout=deadline, retry=None)
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        return ObjectMetadata(
            canonical_uri=canonical_uri,
            bucket=self.policy.bucket,
            name=object_name,
            object_generation=int(blob.generation),
            size=len(data),
            content_type=content_type,
            checksum_sha256=checksum_sha256,
            updated_at=blob.updated,
        )

    def read_bounded(
        self, canonical_uri: str, object_generation: int, max_bytes: int, deadline: float
    ) -> bytes:
        if object_generation < 1 or max_bytes < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        blob = self._blob(canonical_uri)
        try:
            blob.reload(timeout=deadline, retry=None)
            if int(blob.size) > max_bytes:
                raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
            data: bytes = blob.download_as_bytes(
                start=0,
                end=max_bytes,
                if_generation_match=object_generation,
                timeout=deadline,
                retry=None,
            )
        except RagError:
            raise
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            if error.__class__.__name__ == "PreconditionFailed":
                raise RagError(ErrorCode.STALE_GENERATION) from error
            raise RetryableDependencyError() from error
        if len(data) > max_bytes:
            raise RagError(ErrorCode.DOCUMENT_TOO_LARGE)
        expected = (getattr(blob, "metadata", None) or {}).get("sha256")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return data

    def delete(self, canonical_uri: str, object_generation: int, deadline: float) -> DeleteResult:
        if object_generation < 1 or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        blob = self._blob(canonical_uri)
        try:
            blob.delete(if_generation_match=object_generation, timeout=deadline, retry=None)
            return DeleteResult(deleted=True)
        except Exception as error:
            if error.__class__.__name__ == "NotFound":
                return DeleteResult(deleted=False)
            if error.__class__.__name__ == "PreconditionFailed":
                raise RagError(ErrorCode.STALE_GENERATION) from error
            if isinstance(error, TimeoutError):
                raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
            raise RetryableDependencyError() from error


class PubSubEventPublisher:
    """Content-free Pub/Sub publisher with matching frozen attributes."""

    def __init__(self, *, publisher: Any, topic_path: str) -> None:
        if not topic_path:
            raise ValueError("topic_path is required")
        self.publisher = publisher
        self.topic_path = topic_path
        self.calls: list[dict[str, object]] = []

    @classmethod  # pragma: no cover - credential-bearing factory
    def from_project_topic(cls, *, project: str, topic: str) -> PubSubEventPublisher:
        pubsub_v1 = importlib.import_module("google.cloud.pubsub_v1")
        client = pubsub_v1.PublisherClient()
        return cls(publisher=client, topic_path=client.topic_path(project, topic))

    def publish(self, payload: IngestionMessage, deadline: float) -> PublishReceipt:
        if deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        encoded = payload.model_dump_json(exclude_none=True).encode()
        # Re-validation makes accidental model bypasses fail before publishing.
        IngestionMessage.model_validate_json(encoded)
        attributes = {
            "contract_version": payload.contract_version,
            "operation": payload.operation.value,
            "trace_id": payload.trace_id,
        }
        self.calls.append(
            {
                "topic_path": self.topic_path,
                "attributes": attributes,
                "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        try:
            future = self.publisher.publish(self.topic_path, encoded, **attributes)
            message_id = future.result(timeout=deadline)
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        return PublishReceipt(message_id=str(message_id))


def encode_push_data(payload: IngestionMessage) -> str:
    """Encode only a validated decoded payload for local contract tests."""
    raw = json.dumps(payload.model_dump(mode="json"), separators=(",", ":")).encode()
    return base64.b64encode(raw).decode("ascii")
