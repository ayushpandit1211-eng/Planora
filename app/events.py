"""Per-request event bus + Server-Sent Events serialisation (Section 38).

SSE is used because the stream is one-directional (backend -> dashboard). Each request has
an isolated history; late subscribers get the full replay. Event messages are redacted and
must never contain raw SQL or literals.
"""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import Any, AsyncIterator

from app.contracts import StreamEvent
from app.logging_setup import redact
from app.pii import PIIDetector

PIPELINE_EVENTS = [
    "REQUEST_RECEIVED", "SECURITY_SCAN_STARTED", "SECURITY_SCAN_COMPLETE", "PII_SCAN_COMPLETE",
    "MASKING_STARTED", "MASKING_COMPLETE", "SCHEMA_INSPECTION_COMPLETE", "EXPLAIN_STARTED",
    "EXPLAIN_COMPLETE", "GRAPH_CONVERSION_COMPLETE", "GNN_ANALYSIS_COMPLETE", "SUPERVISOR_STARTED",
    "SQL_AGENT_STARTED", "SQL_AGENT_COMPLETE", "SCHEMA_AGENT_STARTED", "SCHEMA_AGENT_COMPLETE",
    "INDEX_AGENT_STARTED", "INDEX_AGENT_COMPLETE", "RL_ANALYSIS_STARTED", "RL_ANALYSIS_COMPLETE",
    "SANDBOX_STARTED", "SANDBOX_COMPLETE", "BLAST_RADIUS_STARTED", "BLAST_RADIUS_COMPLETE", "WRITE_LATENCY_SIMULATED",
    "PARTITION_SIMULATION_STARTED", "PARTITION_SIMULATION_COMPLETE",
    "FINAL_REPORT_READY", "ERROR",
]
EXTRA_EVENTS = [
    "AGENT_SKIPPED", "VALIDATION_COMPLETE", "MIGRATION_GENERATED", "PLAN_REGRESSION",
    "WORKLOAD_DRIFT", "SEARCH_BUDGET_EXHAUSTED", "CAPABILITY_DENIED", "PRIVACY_BLOCKED",
    "SECURITY_REJECTED", "LLM_CALL", "HEARTBEAT",
    # Final Upgrade P13 (Ask Your Database / BYOD progress)
    "NLQ_STARTED", "NLQ_SCHEMA_LOADED", "NLQ_SQL_VALIDATED", "NLQ_SQL_REJECTED", "NLQ_OPTIMIZATION_STARTED",
    "NLQ_OPTIMIZATION_COMPLETE", "NLQ_EXECUTED", "NLQ_COMPLETE", "PRIVACY_BUDGET_EXHAUSTED",
]
ALL_EVENTS = set(PIPELINE_EVENTS) | set(EXTRA_EVENTS)
TERMINAL_EVENTS = {"FINAL_REPORT_READY", "ERROR", "SECURITY_REJECTED", "NLQ_COMPLETE", "NLQ_SQL_REJECTED"}

_detector = PIIDetector()


def safe_message(message: str) -> str:
    msg = redact(message)
    scan = _detector.scan(msg)
    if scan.pii_detected or scan.secret_detected:
        return "[message withheld: privacy filter]"
    return msg[:500]


class RequestChannel:
    def __init__(self) -> None:
        self.history: list[StreamEvent] = []
        self.queues: list[asyncio.Queue] = []
        self.closed = False


class EventBus:
    def __init__(self, max_requests: int = 500):
        self._channels: OrderedDict[str, RequestChannel] = OrderedDict()
        self.max_requests = max_requests

    def _channel(self, request_id: str) -> RequestChannel:
        ch = self._channels.get(request_id)
        if ch is None:
            ch = RequestChannel()
            self._channels[request_id] = ch
            while len(self._channels) > self.max_requests:
                self._channels.popitem(last=False)
        return ch

    def publish(self, request_id: str, event: str, stage: str, message: str,
                data: dict[str, Any] | None = None) -> StreamEvent:
        if event not in ALL_EVENTS:
            raise ValueError(f"unknown event type {event}")
        ev = StreamEvent(event=event, request_id=request_id, stage=stage,
                         message=safe_message(message), data=_safe_data(data or {}))
        ch = self._channel(request_id)
        ch.history.append(ev)
        for q in list(ch.queues):
            q.put_nowait(ev)
        if event in TERMINAL_EVENTS:
            ch.closed = True
        return ev

    def history(self, request_id: str) -> list[StreamEvent]:
        ch = self._channels.get(request_id)
        return list(ch.history) if ch else []

    def known(self, request_id: str) -> bool:
        return request_id in self._channels

    def ensure(self, request_id: str) -> None:
        self._channel(request_id)

    async def subscribe(self, request_id: str, heartbeat_s: float = 15.0,
                        timeout_s: float = 600.0) -> AsyncIterator[StreamEvent]:
        ch = self._channel(request_id)
        q: asyncio.Queue = asyncio.Queue()
        replay = list(ch.history)
        ch.queues.append(q)
        try:
            for ev in replay:
                yield ev
                if ev.event in TERMINAL_EVENTS:
                    return
            # Events published during replay are queued, including completion.
            # Drain that queue even if the publisher has already closed the channel.
            waited = 0.0
            while waited < timeout_s:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=heartbeat_s)
                except asyncio.TimeoutError:
                    waited += heartbeat_s
                    yield StreamEvent(event="HEARTBEAT", request_id=request_id, stage="stream",
                                      message="waiting for pipeline events")
                    continue
                yield ev
                if ev.event in TERMINAL_EVENTS:
                    return
        finally:
            if q in ch.queues:
                ch.queues.remove(q)


def _safe_data(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = safe_message(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, default=str)
            scan = _detector.scan(s)
            out[k] = "[withheld: privacy filter]" if (scan.pii_detected or scan.secret_detected) else v
        else:
            out[k] = safe_message(str(v))
    return out


def to_sse(ev: StreamEvent) -> str:
    """Serialise one event in text/event-stream format."""
    payload = ev.model_dump(mode="json")
    return f"event: {ev.event}\ndata: {json.dumps(payload)}\n\n"


event_bus = EventBus()
