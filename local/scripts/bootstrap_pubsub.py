"""Idempotently create the local Pub/Sub topology through the emulator REST API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def request(method: str, url: str, body: dict[str, object] | None = None) -> None:
    if not url.startswith("http://pubsub:8085/"):
        raise RuntimeError("only the local Pub/Sub emulator is permitted")
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    call = urllib.request.Request(  # noqa: S310 - URL is constrained above
        url,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # nosec B310  # noqa: S310 - constrained URL
            call, timeout=3
        ) as response:
            if response.status not in {200, 201}:
                raise RuntimeError(f"unexpected emulator status {response.status}")
    except urllib.error.HTTPError as error:
        if error.code != 409:
            raise


def main() -> None:
    host = os.environ["PUBSUB_EMULATOR_HOST"]
    if "googleapis.com" in host or not host.startswith("pubsub:"):
        raise SystemExit("refusing non-local Pub/Sub endpoint")
    project = os.environ["PUBSUB_PROJECT_ID"]
    base = f"http://{host}/v1/projects/{project}"
    topic = os.environ["PUBSUB_TOPIC"]
    dlq_topic = os.environ["PUBSUB_DLQ_TOPIC"]
    subscription = os.environ["PUBSUB_SUBSCRIPTION"]
    dlq_subscription = os.environ["PUBSUB_DLQ_SUBSCRIPTION"]
    push_endpoint = os.environ["PUBSUB_PUSH_ENDPOINT"]

    for attempt in range(30):
        try:
            request("PUT", f"{base}/topics/{topic}", {})
            request("PUT", f"{base}/topics/{dlq_topic}", {})
            break
        except (OSError, urllib.error.URLError):
            if attempt == 29:
                raise
            time.sleep(0.2)

    request(
        "PUT",
        f"{base}/subscriptions/{subscription}",
        {
            "topic": f"projects/{project}/topics/{topic}",
            "pushConfig": {"pushEndpoint": push_endpoint},
            "ackDeadlineSeconds": 10,
            "deadLetterPolicy": {
                "deadLetterTopic": f"projects/{project}/topics/{dlq_topic}",
                "maxDeliveryAttempts": 5,
            },
            "retryPolicy": {
                "minimumBackoff": "1s",
                "maximumBackoff": "5s",
            },
        },
    )
    request(
        "PUT",
        f"{base}/subscriptions/{dlq_subscription}",
        {
            "topic": f"projects/{project}/topics/{dlq_topic}",
            "ackDeadlineSeconds": 10,
        },
    )
    print("local Pub/Sub topics, push subscription, retry policy, and DLQ are ready")


if __name__ == "__main__":
    main()
