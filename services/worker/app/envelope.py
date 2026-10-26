"""Strict, bounded Pub/Sub push-envelope decoding."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from packages.rag_core.errors import ErrorCode, RagError
from packages.rag_core.models import IngestionMessage, IngestionOperation

MAX_PUSH_BODY_BYTES = 131_072
MAX_DECODED_PAYLOAD_BYTES = 32_768


class _PubSubWireModel(BaseModel):
    """Pub/Sub push JSON includes duplicate snake_case and camelCase fields."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class PushAttributes(_PubSubWireModel):
    contract_version: str
    operation: IngestionOperation
    trace_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class PushMessage(_PubSubWireModel):
    data: str
    message_id: Annotated[
        str,
        Field(alias="messageId"),
        StringConstraints(min_length=1, max_length=256),
    ]
    publish_time: datetime = Field(alias="publishTime")
    ordering_key: Annotated[str, StringConstraints(max_length=1024)] | None = Field(
        default=None, alias="orderingKey"
    )
    attributes: PushAttributes


class PushEnvelope(_PubSubWireModel):
    message: PushMessage
    subscription: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    delivery_attempt: Annotated[int, Field(ge=1)] = Field(default=1, alias="deliveryAttempt")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def decode_push_envelope(body: bytes) -> tuple[PushEnvelope, IngestionMessage]:
    """Decode exactly one strict envelope without accepting oversized input."""
    if not body or len(body) > MAX_PUSH_BODY_BYTES:
        raise RagError(ErrorCode.INVALID_REQUEST)
    try:
        raw = json.loads(body, parse_constant=_reject_constant)
        envelope = PushEnvelope.model_validate(raw)
        if envelope.message.publish_time.tzinfo is None:
            raise ValueError("publishTime must include a timezone")
        encoded = envelope.message.data.encode("ascii")
        if len(encoded) > ((MAX_DECODED_PAYLOAD_BYTES + 2) // 3) * 4:
            raise ValueError("encoded payload is too large")
        decoded = base64.b64decode(encoded, validate=True)
        if len(decoded) > MAX_DECODED_PAYLOAD_BYTES:
            raise ValueError("decoded payload is too large")
        payload_raw = json.loads(decoded, parse_constant=_reject_constant)
    except (
        UnicodeEncodeError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
    ):
        raise RagError(ErrorCode.INVALID_REQUEST) from None
    if isinstance(payload_raw, dict) and payload_raw.get("contract_version") != "1.0.0":
        raise RagError(ErrorCode.CONTRACT_VERSION_UNSUPPORTED)
    try:
        payload = IngestionMessage.model_validate(payload_raw)
    except ValidationError:
        raise RagError(ErrorCode.INVALID_REQUEST) from None
    if payload.message_metadata.occurred_at.tzinfo is None:
        raise RagError(ErrorCode.INVALID_REQUEST)
    attributes = envelope.message.attributes
    if attributes.contract_version != "1.0.0":
        raise RagError(ErrorCode.CONTRACT_VERSION_UNSUPPORTED)
    if attributes.operation is not payload.operation or attributes.trace_id != payload.trace_id:
        raise RagError(ErrorCode.INVALID_REQUEST)
    return envelope, payload
