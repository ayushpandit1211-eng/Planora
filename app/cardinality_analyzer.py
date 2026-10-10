"""Cardinality-error analysis (Addendum R5).

Compares planner-estimated rows with ACTUAL rows only when actual values exist (EXPLAIN ANALYZE
from the opt-in benchmark sandbox, or ingested historical plans that contain them). Actual row
counts are never invented. Misestimation ratio = max(est, act) / max(min(est, act), 1), with
direction. Large errors on scans with multiple ANDed filter columns flag possible correlated
predicates -> an ADVISORY `CREATE STATISTICS` suggestion for the Schema Expert.
"""
from __future__ import annotations

from typing import Any

from app.contracts import PlanAnalysis

SEVERE_RATIO = 10.0
MODERATE_RATIO = 3.0


def misestimation_ratio(estimated: float, actual: float) -> tuple[float, str]:
    est, act = max(estimated, 0.0), max(actual, 0.0)
    hi, lo = max(est, act), max(min(est, act), 1.0)
    direction = "overestimate" if est > act else "underestimate" if act > est else "exact"
    return round(hi / lo, 2), direction


def analyze_cardinality(plan: PlanAnalysis, filter_columns: dict[str, list[str]] | None = None) -> dict[str, Any]:
    if not plan.has_actuals:
        return {"available": False,
                "reason": "no actual row counts (EXPLAIN without ANALYZE); run the opt-in benchmark or ingest "
                          "plans that include actual rows",
                "nodes": [], "flags": [], "extended_statistics_candidates": []}
    nodes, flags, ext = [], [], []
    filter_columns = filter_columns or {}
    for n in plan.nodes:
        if n.actual_rows is None:
            continue
        loops = n.actual_loops or 1.0
        actual_total = n.actual_rows          # PostgreSQL reports per-loop average rows
        ratio, direction = misestimation_ratio(n.plan_rows, actual_total)
        sev = "severe" if ratio >= SEVERE_RATIO else "moderate" if ratio >= MODERATE_RATIO else "ok"
        nodes.append({"node_id": n.node_id, "node_type": n.node_type, "relation": n.relation_name,
                      "estimated_rows": n.plan_rows, "actual_rows": actual_total, "actual_loops": loops,
                      "misestimation_ratio": ratio, "direction": direction, "severity": sev})
        if sev != "ok":
            flags.append(f"{n.node_type}{' on ' + n.relation_name if n.relation_name else ''}: {direction} x{ratio}")
            cols = filter_columns.get(n.relation_name or "", [])
            if n.relation_name and len(cols) >= 2 and n.node_type in ("Seq Scan", "Bitmap Heap Scan", "Index Scan"):
                ext.append({"table": n.relation_name, "columns": cols[:3],
                            "statistics_kinds": ["dependencies", "ndistinct"],
                            "reason": f"{direction} x{ratio} on a scan filtering {len(cols)} columns together "
                                      "suggests correlated predicates",
                            "status": "advisory - validate separately before deployment"})
    worst = max((x["misestimation_ratio"] for x in nodes), default=None)
    return {"available": True, "nodes": nodes, "flags": flags, "worst_ratio": worst,
            "extended_statistics_candidates": ext, "evidence_kind": "benchmark-measured"}
