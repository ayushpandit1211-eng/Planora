"""In-process operational metrics (stage durations, counters, model modes).

Exposed at GET /api/v1/metrics (JSON) and in Prometheus text format at /api/v1/metrics/prometheus.
Only aggregate, non-sensitive values are recorded - never SQL, literals or identifiers.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.durations: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, stage: str, ms: float) -> None:
        with self._lock:
            d = self.durations[stage]
            d.append(ms)
            if len(d) > 1000:
                del d[: len(d) - 1000]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {}
            for k, v in self.durations.items():
                s = sorted(v)
                stages[k] = {"count": len(s), "avg_ms": round(sum(s) / len(s), 2) if s else None,
                             "p50_ms": round(s[len(s) // 2], 2) if s else None,
                             "p95_ms": round(s[min(int(len(s) * 0.95), len(s) - 1)], 2) if s else None}
            return {"counters": dict(self.counters), "stage_durations": stages}

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = ["# TYPE querysentinel_events_total counter"]
        for k, v in sorted(snap["counters"].items()):
            lines.append(f'querysentinel_events_total{{name="{k}"}} {v}')
        lines.append("# TYPE querysentinel_stage_duration_ms summary")
        for k, v in sorted(snap["stage_durations"].items()):
            if v["count"]:
                lines.append(f'querysentinel_stage_duration_ms{{stage="{k}",quantile="0.5"}} {v["p50_ms"]}')
                lines.append(f'querysentinel_stage_duration_ms{{stage="{k}",quantile="0.95"}} {v["p95_ms"]}')
                lines.append(f'querysentinel_stage_duration_ms_count{{stage="{k}"}} {v["count"]}')
        return "\n".join(lines) + "\n"


metrics = Metrics()
