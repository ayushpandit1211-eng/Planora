"""Workload drift detection (Addendum R8).

Compares RECENT vs HISTORICAL distributions of safe structural features - query fingerprints,
predicate patterns, join patterns, plan fingerprints, table access - with the Jensen-Shannon
divergence (log base 2, bounded in [0, 1]), weighted by call counts. A WORKLOAD_DRIFT event is
emitted only when a measured divergence crosses the configured threshold (DRIFT_JS_THRESHOLD)
and both windows hold at least MIN_RECORDS records; otherwise the result says the data is
insufficient. Scores are always computed, never fabricated.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from app.config import get_settings
from app.workload_store import WorkloadRecord

MIN_RECORDS = 5


def js_divergence(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not keys or sp == 0 or sq == 0:
        return 0.0
    js = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a > 0:
            js += 0.5 * a * math.log2(a / m)
        if b > 0:
            js += 0.5 * b * math.log2(b / m)
    return round(min(max(js, 0.0), 1.0), 4)


def _features(records: list[WorkloadRecord]) -> dict[str, Counter]:
    f = {k: Counter() for k in ("query_fingerprints", "predicate_patterns", "join_patterns",
                                "plan_fingerprints", "table_access")}
    for r in records:
        w = max(r.call_count, 1)
        f["query_fingerprints"][r.query_fingerprint] += w
        if r.plan_fingerprint:
            f["plan_fingerprints"][r.plan_fingerprint] += w
        per = (r.access_features or {}).get("per_table", {})
        for t, roles in per.items():
            f["table_access"][t] += w
            for role in ("eq", "range", "in", "like"):
                for c in roles.get(role, []):
                    f["predicate_patterns"][f"{t}.{c}:{role}"] += w
        for j in r.join_features or []:
            f["join_patterns"][f"{j.get('left_table')}.{j.get('left_column')}={j.get('right_table')}.{j.get('right_column')}"] += w
    return f


def detect_drift(historical: list[WorkloadRecord], recent: list[WorkloadRecord],
                 threshold: float | None = None) -> dict[str, Any]:
    thr = get_settings().drift_js_threshold if threshold is None else threshold
    if len(historical) < MIN_RECORDS or len(recent) < MIN_RECORDS:
        return {"status": "insufficient_data", "drift_detected": False, "threshold": thr,
                "detection_method": "threshold-based (not a statistical test)",
                "historical_records": len(historical), "recent_records": len(recent),
                "min_records_per_window": MIN_RECORDS, "scores": {}}
    fh, fr = _features(historical), _features(recent)
    scores = {k: js_divergence(fh[k], fr[k]) for k in fh}
    crossed = {k: v for k, v in scores.items() if v >= thr}
    new_fps = sorted(set(fr["query_fingerprints"]) - set(fh["query_fingerprints"]))
    return {"status": "measured", "drift_detected": bool(crossed), "threshold": thr,
            "detection_method": "threshold-based (not a statistical test)",
            "metric": "Jensen-Shannon divergence (base 2), call-weighted", "scores": scores,
            "dimensions_over_threshold": sorted(crossed), "new_query_fingerprints": new_fps[:20],
            "historical_records": len(historical), "recent_records": len(recent),
            "recommendation": ("WORKLOAD_DRIFT: re-evaluate / retrain the RL policy on recent workload"
                               if crossed else "no drift above threshold")}
