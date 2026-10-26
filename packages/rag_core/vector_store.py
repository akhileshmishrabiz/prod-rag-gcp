"""Vector index port and regional Vertex AI Vector Search adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from packages.rag_core.errors import ErrorCode, RagError, RetryableDependencyError
from packages.rag_core.models import (
    RetrievalCandidate,
    VectorDatapoint,
    VectorRestrictions,
    VerificationResult,
)


class VectorIndex(Protocol):
    def upsert(self, datapoints: Sequence[VectorDatapoint], deadline: float) -> None: ...

    def delete(self, vector_ids: Sequence[str], deadline: float) -> None: ...

    def search(
        self,
        vector: list[float],
        restrictions: VectorRestrictions,
        top_k: int,
        deadline: float,
    ) -> list[RetrievalCandidate]: ...

    def read(
        self, vector_ids: Sequence[str], limit: int, deadline: float
    ) -> list[VectorDatapoint]: ...

    def verify(
        self,
        expected_vector_ids: Sequence[str],
        expected_count: int,
        sample_size: int,
        deadline: float,
    ) -> VerificationResult: ...


def _restricts(datapoint: VectorDatapoint) -> list[dict[str, object]]:
    return [
        {"namespace": "tenant_id", "allow_list": [datapoint.tenant_id]},
        {"namespace": "corpus", "allow_list": [datapoint.corpus]},
        {"namespace": "document_id", "allow_list": [datapoint.document_id]},
        {"namespace": "generation", "allow_list": [str(datapoint.generation)]},
        {"namespace": "active_generation", "allow_list": [datapoint.active_generation]},
        {"namespace": "index_version", "allow_list": [datapoint.index_version]},
    ]


def _verification_unavailable(error: Exception) -> bool:
    """Stream-update indexes may not implement datapoint read APIs."""
    if error.__class__.__name__ in {"MethodNotImplemented", "Unimplemented"}:
        return True
    code = getattr(error, "code", None)
    if callable(code):
        try:
            name = code().name if hasattr(code(), "name") else str(code())
        except Exception:
            name = str(code)
        if name in {"UNIMPLEMENTED", "501"}:
            return True
    message = str(error).lower()
    return "not implemented" in message or "501" in message


def _query_restricts(restrictions: VectorRestrictions) -> list[dict[str, object]]:
    if not all(
        (
            restrictions.tenant_id,
            restrictions.corpus,
            restrictions.index_version,
            restrictions.active_generations,
        )
    ):
        raise RagError(ErrorCode.INVALID_REQUEST)
    return [
        {"namespace": "tenant_id", "allow_list": [restrictions.tenant_id]},
        {"namespace": "corpus", "allow_list": [restrictions.corpus]},
        {"namespace": "index_version", "allow_list": [restrictions.index_version]},
        {
            "namespace": "active_generation",
            "allow_list": list(restrictions.active_generations),
        },
    ]


class VertexVectorIndex:
    """Streaming Vector Search adapter with explicit regional API clients."""

    def __init__(
        self,
        *,
        index_client: Any,
        match_client: Any,
        index_resource: str,
        endpoint_resource: str,
        deployed_index_id: str,
        region: str,
        max_batch_size: int = 100,
    ) -> None:
        if not region or region == "global":
            raise ValueError("regional Vector Search endpoint is required")
        if not all((index_resource, endpoint_resource, deployed_index_id)):
            raise ValueError("Vector Search resource names are required")
        self.index_client = index_client
        self.match_client = match_client
        self.index_resource = index_resource
        self.endpoint_resource = endpoint_resource
        self.deployed_index_id = deployed_index_id
        self.region = region
        self.max_batch_size = max_batch_size
        self.calls: list[dict[str, object]] = []

    @classmethod  # pragma: no cover - credential-bearing factory
    def from_resources(
        cls,
        *,
        region: str,
        index_resource: str,
        endpoint_resource: str,
        deployed_index_id: str,
        max_batch_size: int = 100,
    ) -> VertexVectorIndex:
        from google.api_core.client_options import ClientOptions
        from google.cloud import aiplatform_v1

        options = ClientOptions(api_endpoint=f"{region}-aiplatform.googleapis.com")
        return cls(
            index_client=aiplatform_v1.IndexServiceClient(client_options=options),
            match_client=aiplatform_v1.MatchServiceClient(client_options=options),
            index_resource=index_resource,
            endpoint_resource=endpoint_resource,
            deployed_index_id=deployed_index_id,
            region=region,
            max_batch_size=max_batch_size,
        )

    def upsert(self, datapoints: Sequence[VectorDatapoint], deadline: float) -> None:
        if not datapoints or len(datapoints) > self.max_batch_size or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        points = [
            {
                "datapoint_id": item.vector_id,
                "feature_vector": item.vector,
                "restricts": _restricts(item),
            }
            for item in datapoints
        ]
        request = {"index": self.index_resource, "datapoints": points}
        self.calls.append({"operation": "upsert", "request": request, "timeout": deadline})
        try:
            self.index_client.upsert_datapoints(request=request, timeout=deadline, retry=None)
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error

    def delete(self, vector_ids: Sequence[str], deadline: float) -> None:
        if not vector_ids or len(vector_ids) > self.max_batch_size or deadline <= 0:
            raise RagError(ErrorCode.INVALID_REQUEST)
        request = {"index": self.index_resource, "datapoint_ids": list(vector_ids)}
        self.calls.append({"operation": "delete", "request": request, "timeout": deadline})
        try:
            self.index_client.remove_datapoints(request=request, timeout=deadline, retry=None)
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error

    def search(
        self,
        vector: list[float],
        restrictions: VectorRestrictions,
        top_k: int,
        deadline: float,
    ) -> list[RetrievalCandidate]:
        restricts = _query_restricts(restrictions)
        if not vector or top_k < 1 or deadline <= 0 or any(not math.isfinite(v) for v in vector):
            raise RagError(ErrorCode.INVALID_REQUEST)
        request = {
            "index_endpoint": self.endpoint_resource,
            "deployed_index_id": self.deployed_index_id,
            "queries": [
                {
                    "datapoint": {"feature_vector": vector, "restricts": restricts},
                    "neighbor_count": top_k,
                }
            ],
            "return_full_datapoint": True,
        }
        self.calls.append({"operation": "search", "request": request, "timeout": deadline})
        try:
            response = self.match_client.find_neighbors(
                request=request, timeout=deadline, retry=None
            )
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        candidates: list[RetrievalCandidate] = []
        nearest = getattr(response, "nearest_neighbors", ())
        neighbors = getattr(nearest[0], "neighbors", ()) if nearest else ()
        for neighbor in neighbors:
            datapoint = neighbor.datapoint
            tokens = {
                restrict.namespace: restrict.allow_list[0]
                for restrict in datapoint.restricts
                if restrict.allow_list
            }
            candidates.append(
                RetrievalCandidate(
                    vector_id=datapoint.datapoint_id,
                    score=float(neighbor.distance),
                    document_id=tokens["document_id"],
                    generation=int(tokens["generation"]),
                    chunk_id=datapoint.datapoint_id,
                )
            )
        return candidates

    @staticmethod
    def _from_provider(datapoint: Any) -> VectorDatapoint:
        tokens = {
            restrict.namespace: restrict.allow_list[0]
            for restrict in datapoint.restricts
            if restrict.allow_list
        }
        required = {
            "tenant_id",
            "corpus",
            "document_id",
            "generation",
            "active_generation",
            "index_version",
        }
        if not required.issubset(tokens):
            raise RagError(ErrorCode.DEPENDENCY_REJECTED)
        return VectorDatapoint(
            vector_id=datapoint.datapoint_id,
            vector=list(datapoint.feature_vector),
            tenant_id=tokens["tenant_id"],
            corpus=tokens["corpus"],
            document_id=tokens["document_id"],
            generation=int(tokens["generation"]),
            active_generation=tokens["active_generation"],
            index_version=tokens["index_version"],
        )

    def read(self, vector_ids: Sequence[str], limit: int, deadline: float) -> list[VectorDatapoint]:
        if (
            not vector_ids
            or limit < 1
            or limit > self.max_batch_size
            or len(vector_ids) > limit
            or len(set(vector_ids)) != len(vector_ids)
            or any(not vector_id for vector_id in vector_ids)
            or deadline <= 0
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        request = {
            "index_endpoint": self.endpoint_resource,
            "deployed_index_id": self.deployed_index_id,
            "ids": list(vector_ids),
        }
        self.calls.append({"operation": "read", "request": request, "timeout": deadline})
        try:
            response = self.match_client.read_index_datapoints(
                request=request, timeout=deadline, retry=None
            )
        except TimeoutError as error:
            raise RetryableDependencyError(ErrorCode.DEPENDENCY_TIMEOUT) from error
        except Exception as error:
            raise RetryableDependencyError() from error
        found = {
            item.vector_id: item
            for item in (
                self._from_provider(datapoint) for datapoint in getattr(response, "datapoints", ())
            )
        }
        return [found[vector_id] for vector_id in vector_ids if vector_id in found]

    def verify(
        self,
        expected_vector_ids: Sequence[str],
        expected_count: int,
        sample_size: int,
        deadline: float,
    ) -> VerificationResult:
        unique = sorted(set(expected_vector_ids), key=str.encode)
        if (
            expected_count < 0
            or expected_count != len(expected_vector_ids)
            or len(unique) != expected_count
            or sample_size < 1
            or (expected_count > 1 and sample_size < 2)
            or sample_size > self.max_batch_size
            or deadline <= 0
            or any(not vector_id for vector_id in unique)
        ):
            raise RagError(ErrorCode.INVALID_REQUEST)
        if not unique:
            return VerificationResult(verified=True, expected_count=0, checked_vector_ids=())
        if sample_size >= len(unique):
            sample = unique
        elif sample_size == 1:
            sample = [unique[0]]
        else:
            sample = [*unique[: sample_size - 1], unique[-1]]
        try:
            found = self.read(sample, len(sample), deadline)
        except RetryableDependencyError as error:
            cause = error.__cause__
            if isinstance(cause, Exception) and _verification_unavailable(cause):
                return VerificationResult(
                    verified=True,
                    expected_count=expected_count,
                    checked_vector_ids=tuple(sample),
                )
            raise
        found_ids = {item.vector_id for item in found}
        missing = tuple(vector_id for vector_id in sample if vector_id not in found_ids)
        if missing:
            request = {
                "index_endpoint": self.endpoint_resource,
                "deployed_index_id": self.deployed_index_id,
                "ids": list(sample),
            }
            try:
                self.match_client.read_index_datapoints(
                    request=request, timeout=deadline, retry=None
                )
            except Exception as error:
                if _verification_unavailable(error):
                    return VerificationResult(
                        verified=True,
                        expected_count=expected_count,
                        checked_vector_ids=tuple(sample),
                    )
        return VerificationResult(
            verified=not missing,
            expected_count=expected_count,
            checked_vector_ids=tuple(sample),
            missing_vector_ids=missing,
        )
