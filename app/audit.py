"""Audit trail (Section 37). Every optimisation request has a request ID; major steps are
recorded with safe, anonymised metadata only. Events are kept in memory (bounded) and,
when a database writer is attached, persisted to optimizer_meta.audit_events."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from app.contracts import AuditEvent
from app.logging_setup import get_logger, redact

log = get_logger("audit")

AUDIT_EVENT_TYPES = {
    "REQUEST_RECEIVED", "SECURITY_SCAN", "SECURITY_REJECTION", "PII_DETECTION", "MASKING",
    "SCHEMA_INSPECTION", "EXPLAIN", "GRAPH_CONVERSION", "GNN_MODE", "AGENT_ROUTING",
    "RL_RECOMMENDATION", "SANDBOX_EXPERIMENT", "BLAST_RADIUS", "FINAL_RECOMMENDATION",
    "MIGRATION_GENERATION", "PRIVACY_BLOCK", "CAPABILITY_DENIED", "BUDGET_EXHAUSTED",
    "PLAN_REGRESSION", "WORKLOAD_DRIFT", "INGESTION", "LLM_CALL", "ERROR", "BENCHMARK",
}


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<truncated>"
    if isinstance(value, str):
        return redact(value)[:2000]
    if isinstance(value, dict):
        return {str(k): _sanitize(v, depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, depth + 1) for v in list(value)[:100]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(str(value))[:500]


Persister = Callable[[AuditEvent], Awaitable[None]]


class AuditLogger:
    def __init__(self, max_per_request: int = 500, max_requests: int = 2000):
        self._events: dict[str, deque[AuditEvent]] = defaultdict(lambda: deque(maxlen=max_per_request))
        self._order: deque[str] = deque(maxlen=max_requests)
        self._persister: Persister | None = None
        self._pending: set[asyncio.Task] = set()

    def attach_persister(self, persister: Persister | None) -> None:
        self._persister = persister

    def record(self, request_id: str, event_type: str, stage: str | None = None,
               severity: str = "info", details: dict[str, Any] | None = None) -> AuditEvent:
        if event_type not in AUDIT_EVENT_TYPES:
            event_type = "ERROR" if severity == "error" else event_type.upper()
        ev = AuditEvent(request_id=request_id, event_type=event_type, stage=stage,
                        severity=severity, details=_sanitize(details or {}))
        if request_id not in self._events:
            self._order.append(request_id)
        self._events[request_id].append(ev)
        log.info(event_type, extra={"request_id": request_id, "stage": stage, "event": event_type})
        if self._persister is not None:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._safe_persist(ev))
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)
            except RuntimeError:
                pass  # no running loop (sync context / tests): memory only
        return ev

    async def _safe_persist(self, ev: AuditEvent) -> None:
        try:
            await self._persister(ev)  # type: ignore[misc]
        except Exception as exc:  # persistence must never break the pipeline
            log.warning("audit persistence failed", extra={"request_id": ev.request_id,
                                                           "error_category": type(exc).__name__})

    async def flush(self) -> None:
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def events_for(self, request_id: str) -> list[AuditEvent]:
        return list(self._events.get(request_id, ()))

audit_logger = AuditLogger()
