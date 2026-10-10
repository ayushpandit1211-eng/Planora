"""Workload intelligence store (Addendum R1).

Holds ONLY safe, anonymised workload metadata: fingerprints, plan shape, time bucket,
call counts, timings when legitimately available, planner cost, row estimates, operator
histogram, access-pattern features and recommendation outcomes. Never raw literals or rows.
In-memory by default; persisted to optimizer_meta.query_history when a DB writer is attached.
"""
from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class WorkloadRecord(BaseModel):
    query_fingerprint: str
    plan_fingerprint: str | None = None
    plan_shape: str | None = None
    timestamp_bucket: str                      # ISO hour bucket
    call_count: int = 1
    mean_exec_time_ms: float | None = None
    estimated_cost: float | None = None
    estimated_rows: float | None = None
    actual_rows: float | None = None
    operator_histogram: dict[str, int] = Field(default_factory=dict)
    access_features: dict[str, Any] = Field(default_factory=dict)
    join_features: list[dict[str, Any]] = Field(default_factory=list)
    sort_features: list[str] = Field(default_factory=list)
    index_features: list[str] = Field(default_factory=list)
    table_size_bucket: str | None = None
    write_frequency_bucket: str | None = None
    anonymized_sql: str | None = None
    statement_type: str = "SelectStmt"
    source: str = "ingestion"
    recommendation_outcome: str | None = None
    data_source_id: str = "demo"               # Final Upgrade: workload namespace (additive, default demo)


def hour_bucket(ts: str | datetime | None) -> str:
    if ts is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()


Persister = Callable[[WorkloadRecord], Awaitable[None]]


class WorkloadStore:
    def __init__(self, namespace: str = "demo") -> None:
        self._records: list[WorkloadRecord] = []
        self._lock = threading.Lock()
        self.persister: Persister | None = None
        self.namespace = namespace

    def scoped(self, namespace: str) -> "WorkloadStore":
        """View of the same store restricted to one data source's workload namespace (Final Upgrade P5).
        Records added through the view are tagged with that namespace, so sandbox/dynamic analyses never
        pollute the demo workload."""
        view = WorkloadStore.__new__(WorkloadStore)
        view._records, view._lock, view.persister, view.namespace = self._records, self._lock, self.persister, namespace
        return view

    def _tag(self, rec: WorkloadRecord) -> WorkloadRecord:
        if self.namespace != "demo" and rec.data_source_id == "demo":
            return rec.model_copy(update={"data_source_id": self.namespace})
        return rec

    def add(self, rec: WorkloadRecord) -> None:
        with self._lock:
            self._records.append(self._tag(rec))

    def extend(self, recs: list[WorkloadRecord]) -> None:
        with self._lock:
            self._records.extend(self._tag(r) for r in recs)

    def all(self) -> list[WorkloadRecord]:
        with self._lock:
            return [r for r in self._records if r.data_source_id == self.namespace]

    def purge_source(self, namespace: str) -> int:
        with self._lock:
            before = len(self._records)
            self._records[:] = [r for r in self._records if r.data_source_id != namespace]
            return before - len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def history(self, query_fingerprint: str) -> list[WorkloadRecord]:
        return sorted((r for r in self.all() if r.query_fingerprint == query_fingerprint),
                      key=lambda r: r.timestamp_bucket)

    def frequency(self, query_fingerprint: str) -> int:
        return sum(r.call_count for r in self.all() if r.query_fingerprint == query_fingerprint)

    def total_calls(self) -> int:
        return sum(r.call_count for r in self.all())

    def frequency_weight(self, query_fingerprint: str) -> float:
        """Documented workload weight used by the RL reward (R7):
        w = 1 + log10(1 + calls_for_fingerprint)  (1.0 when no history)."""
        return 1.0 + math.log10(1 + self.frequency(query_fingerprint))

    def fingerprints(self) -> dict[str, int]:
        agg: dict[str, int] = defaultdict(int)
        for r in self.all():
            agg[r.query_fingerprint] += r.call_count
        return dict(sorted(agg.items(), key=lambda kv: -kv[1]))

    def split_by_time(self, cutoff_iso: str | None = None) -> tuple[list[WorkloadRecord], list[WorkloadRecord]]:
        recs = sorted(self.all(), key=lambda r: r.timestamp_bucket)
        if not recs:
            return [], []
        if cutoff_iso is None:
            buckets = sorted({r.timestamp_bucket for r in recs})
            cutoff_iso = buckets[len(buckets) // 2] if len(buckets) > 1 else buckets[0]
        hist = [r for r in recs if r.timestamp_bucket < cutoff_iso]
        recent = [r for r in recs if r.timestamp_bucket >= cutoff_iso]
        return hist, recent

    def context_for(self, query_fingerprint: str) -> dict[str, Any]:
        """Historical workload context combined with the current query (R1)."""
        hist = self.history(query_fingerprint)
        if not hist:
            return {"available": False, "records": 0, "frequency_weight": 1.0}
        costs = [r.estimated_cost for r in hist if r.estimated_cost is not None]
        times = [r.mean_exec_time_ms for r in hist if r.mean_exec_time_ms is not None]
        return {
            "available": True,
            "records": len(hist),
            "total_calls": sum(r.call_count for r in hist),
            "frequency_weight": round(self.frequency_weight(query_fingerprint), 3),
            "share_of_workload_calls": round(sum(r.call_count for r in hist) / max(self.total_calls(), 1), 4),
            "plan_fingerprints": sorted({r.plan_fingerprint for r in hist if r.plan_fingerprint}),
            "first_seen": hist[0].timestamp_bucket, "last_seen": hist[-1].timestamp_bucket,
            "historical_cost_range": [min(costs), max(costs)] if costs else None,
            "historical_mean_exec_time_ms_range": [min(times), max(times)] if times else None,
            "timing_source": "ingested workload metadata (observed)" if times else None,
        }

    def summary(self) -> dict[str, Any]:
        recs = self.all()
        return {"records": len(recs), "distinct_fingerprints": len({r.query_fingerprint for r in recs}),
                "total_calls": sum(r.call_count for r in recs),
                "sources": sorted({r.source for r in recs})}


def record_to_row(r: WorkloadRecord) -> tuple:
    return (r.query_fingerprint, r.plan_fingerprint, r.plan_shape, datetime.fromisoformat(r.timestamp_bucket),
            r.call_count, r.mean_exec_time_ms, r.estimated_cost, r.estimated_rows, r.actual_rows,
            json.dumps(r.operator_histogram), json.dumps(r.access_features), r.table_size_bucket,
            r.write_frequency_bucket, r.anonymized_sql, r.source, r.recommendation_outcome)


workload_store = WorkloadStore()
