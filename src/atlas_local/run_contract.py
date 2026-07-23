from __future__ import annotations

import queue
import threading
from datetime import UTC, datetime
from typing import Any, TypeVar, TypedDict

TERMINAL_EVENT_TYPES = {"run_completed", "run_failed"}
DEFAULT_SUBSCRIBER_QUEUE_SIZE = 256

QueueItem = TypeVar("QueueItem")


class RunEvent(TypedDict):
    type: str
    timestamp: str
    payload: dict[str, Any]


class RunTraceItem(TypedDict, total=False):
    timestamp: str
    stage: str
    rationale: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    artifacts: dict[str, Any]


class RunHub:
    def __init__(self, *, subscriber_queue_size: int = DEFAULT_SUBSCRIBER_QUEUE_SIZE) -> None:
        self._lock = threading.Lock()
        self._queues: dict[str, list[queue.Queue[RunEvent]]] = {}
        self._subscriber_queue_size = max(1, int(subscriber_queue_size))

    def subscribe(self, run_id: str) -> queue.Queue[RunEvent]:
        subscriber: queue.Queue[RunEvent] = queue.Queue(maxsize=self._subscriber_queue_size)
        with self._lock:
            self._queues.setdefault(run_id, []).append(subscriber)
        return subscriber

    def unsubscribe(self, run_id: str, subscriber: queue.Queue[RunEvent]) -> None:
        with self._lock:
            queues = self._queues.get(run_id, [])
            self._queues[run_id] = [item for item in queues if item is not subscriber]
            if not self._queues[run_id]:
                self._queues.pop(run_id, None)

    def publish(self, run_id: str, event: RunEvent) -> None:
        with self._lock:
            subscribers = list(self._queues.get(run_id, []))
        for subscriber in subscribers:
            put_bounded_queue(subscriber, event)


def put_bounded_queue(target: queue.Queue[QueueItem], item: QueueItem) -> None:
    """Publish without allowing a slow subscriber to grow memory without bound."""
    try:
        target.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        target.get_nowait()
    except queue.Empty:
        pass

    try:
        target.put_nowait(item)
    except queue.Full:
        # Another producer won the newly freed slot. Dropping this item is
        # preferable to blocking a model or runner output thread.
        pass


def make_run_event(event_type: str, payload: dict[str, Any], *, timestamp: str | None = None) -> RunEvent:
    return {
        "type": event_type,
        "timestamp": timestamp or now_timestamp(),
        "payload": payload,
    }


def make_trace_item(item: dict[str, Any], *, timestamp: str | None = None) -> RunTraceItem:
    enriched: RunTraceItem = dict(item)
    enriched.setdefault("timestamp", timestamp or now_timestamp())
    return enriched


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat()
