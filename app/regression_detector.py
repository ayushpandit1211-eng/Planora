"""Plan regression detection (Addendum R2).

For the same query fingerprint, detects changes in plan shape/fingerprint and in performance
metrics across time. A plan change is called a PLAN_REGRESSION only when evidence supports it:
  * planner cost increased by >= COST_REGRESSION_PCT, or
  * observed mean execution time increased by >= TIME_REGRESSION_PCT (only when timings were
    legitimately ingested), or
  * an index-based access path was replaced by a sequential scan AND cost/time did not improve.
Otherwise the change is reported as PLAN_CHANGE (neutral) or PLAN_IMPROVEMENT.
"""
from __future__ import annotations

from statistics import mean
from typing import Any

from app.workload_store import WorkloadRecord

COST_REGRESSION_PCT = 20.0
TIME_REGRESSION_PCT = 50.0
INDEX_OPS = ("Index Scan", "Index Only Scan", "Bitmap Index Scan")


def _avg(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return mean(v) if v else None


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0:
        return None
    return round((b - a) / a * 100.0, 2)


def _uses_index(hist: dict[str, int]) -> bool:
    return any(hist.get(k, 0) for k in INDEX_OPS)


def detect_regressions(history: list[WorkloadRecord]) -> list[dict[str, Any]]:
    """`history`: records of ONE query fingerprint. Consecutive plan-fingerprint segments are compared."""
    recs = sorted([r for r in history if r.plan_fingerprint], key=lambda r: r.timestamp_bucket)
    if len(recs) < 2:
        return []
    segments: list[list[WorkloadRecord]] = [[recs[0]]]
    for r in recs[1:]:
        if r.plan_fingerprint == segments[-1][-1].plan_fingerprint:
            segments[-1].append(r)
        else:
            segments.append([r])
    events = []
    for prev, cur in zip(segments, segments[1:]):
        pc, cc = _avg([r.estimated_cost for r in prev]), _avg([r.estimated_cost for r in cur])
        pt, ct = _avg([r.mean_exec_time_ms for r in prev]), _avg([r.mean_exec_time_ms for r in cur])
        cost_delta, time_delta = _pct(pc, cc), _pct(pt, ct)
        lost_index = _uses_index(prev[-1].operator_histogram) and not _uses_index(cur[0].operator_histogram) \
            and cur[0].operator_histogram.get("Seq Scan", 0) > 0
        evidence = [f"plan fingerprint {prev[-1].plan_fingerprint} -> {cur[0].plan_fingerprint}",
                    f"plan shape: {prev[-1].plan_shape} -> {cur[0].plan_shape}"]
        if cost_delta is not None:
            evidence.append(f"mean planner cost {pc:.2f} -> {cc:.2f} ({cost_delta:+.1f}%)")
        if time_delta is not None:
            evidence.append(f"mean observed execution time {pt:.3f} ms -> {ct:.3f} ms ({time_delta:+.1f}%) [ingested]")
        if lost_index:
            evidence.append("index access path replaced by a sequential scan")
        worse = (cost_delta is not None and cost_delta >= COST_REGRESSION_PCT) or \
                (time_delta is not None and time_delta >= TIME_REGRESSION_PCT)
        better = (cost_delta is not None and cost_delta <= -COST_REGRESSION_PCT) or \
                 (time_delta is not None and time_delta <= -TIME_REGRESSION_PCT)
        if worse or (lost_index and not better and (cost_delta or 0) >= 0):
            kind = "PLAN_REGRESSION"
        elif better:
            kind = "PLAN_IMPROVEMENT"
        else:
            kind = "PLAN_CHANGE"
        events.append({"event": kind, "query_fingerprint": cur[0].query_fingerprint,
                       "changed_at_bucket": cur[0].timestamp_bucket,
                       "previous_plan_fingerprint": prev[-1].plan_fingerprint,
                       "current_plan_fingerprint": cur[0].plan_fingerprint,
                       "previous_operators": prev[-1].operator_histogram, "current_operators": cur[0].operator_histogram,
                       "planner_cost_change_pct": cost_delta, "observed_time_change_pct": time_delta,
                       "lost_index_access": lost_index, "evidence": evidence,
                       "thresholds": {"cost_pct": COST_REGRESSION_PCT, "time_pct": TIME_REGRESSION_PCT},
                       "detection_method": "threshold-based (not a statistical test)"})
    return events


def detect_all(records: list[WorkloadRecord]) -> list[dict[str, Any]]:
    by_fp: dict[str, list[WorkloadRecord]] = {}
    for r in records:
        by_fp.setdefault(r.query_fingerprint, []).append(r)
    out = []
    for fp, recs in by_fp.items():
        out += detect_regressions(recs)
    return out


def compare_current(history: list[WorkloadRecord], current_plan_fingerprint: str, current_cost: float,
                    current_histogram: dict[str, int]) -> dict[str, Any] | None:
    """Compare the plan produced NOW with the most recent historical plan of the same fingerprint."""
    hist = sorted([r for r in history if r.plan_fingerprint], key=lambda r: r.timestamp_bucket)
    if not hist:
        return None
    last = hist[-1]
    probe = WorkloadRecord(query_fingerprint=last.query_fingerprint, plan_fingerprint=current_plan_fingerprint,
                           timestamp_bucket="9999-12-31T00:00:00+00:00", estimated_cost=current_cost,
                           operator_histogram=current_histogram, plan_shape="(current)")
    ev = detect_regressions([*hist[-5:], probe])
    return ev[-1] if ev and ev[-1]["current_plan_fingerprint"] == current_plan_fingerprint else None
