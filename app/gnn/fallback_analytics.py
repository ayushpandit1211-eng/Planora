"""Deterministic analytic fallback for bottleneck diagnosis (Section 20, MODE 2).

Used when no valid GNN checkpoint exists, and always run alongside the GNN to provide
human-readable evidence. Output uses a RULE-CONFIDENCE LABEL (high/medium/low), never a
probability. Thresholds are explicit constants so the behaviour is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.contracts import BOTTLENECK_CLASSES, PlanAnalysis, PlanNode

EFFICIENT_TOTAL_COST = 1_000.0        # whole plan cheaper than this -> no major bottleneck
DOMINANT_SHARE = 0.30                 # a node owning >=30% of effective cost is "dominant"
LARGE_SCAN_ROWS_SCANNED = 50_000.0    # seq scan considered large above this (rows/pages proxy)
SCAN_TYPES = {"Seq Scan", "Bitmap Heap Scan"}
JOIN_TYPES = {"Nested Loop", "Hash Join", "Merge Join"}
SORT_AGG_TYPES = {"Sort", "Incremental Sort", "Aggregate", "Hash", "WindowAgg", "Group", "Unique", "SetOp"}


@dataclass
class RuleResult:
    class_id: int
    rule: str
    confidence_label: str
    focus_node: PlanNode
    evidence: list[str] = field(default_factory=list)


def _share(n: PlanNode, root_cost: float) -> float:
    return min(n.effective_cost / root_cost, 1.0) if root_cost > 0 else 0.0


def _label(share: float) -> str:
    return "high" if share >= 0.6 else "medium" if share >= DOMINANT_SHARE else "low"


def _children(plan: PlanAnalysis, nid: int) -> list[PlanNode]:
    ids = [c for p, c in plan.edges if p == nid]
    return [plan.nodes[i] for i in ids]


def classify(plan: PlanAnalysis) -> RuleResult:
    root = plan.nodes[0]
    root_cost = max(plan.total_cost, 1e-9)
    ranked = sorted(plan.nodes, key=lambda n: n.effective_cost, reverse=True)
    top = ranked[0]
    share = _share(top, root_cost)
    ev = [f"total planner cost = {plan.total_cost:.2f}", f"estimated result rows = {root.plan_rows:.0f}",
          f"highest effective-cost operator = {top.node_type}"
          + (f" on {top.relation_name}" if top.relation_name else "")
          + f" (effective cost {top.effective_cost:.2f}, share {share:.1%})"]
    if top.estimated_loops > 1:
        ev.append(f"operator is executed ~{top.estimated_loops:.0f} times "
                  f"({'correlated SubPlan' if _in_subplan(plan, top) else 'nested-loop inner side'})")

    if plan.total_cost < EFFICIENT_TOTAL_COST:
        ev.append(f"whole plan cost below {EFFICIENT_TOTAL_COST:.0f}")
        return RuleResult(3, "R3_cheap_plan", "high", top, ev)

    if top.node_type in SCAN_TYPES and share >= DOMINANT_SHARE:
        if top.filter:
            ev.append(f"scan applies a residual filter: {top.filter}")
        if top.node_type == "Bitmap Heap Scan":
            ev.append("index-assisted scan still visits many heap rows; filter column not covered by the index")
        if top.node_type == "Seq Scan" and not top.filter:
            ev.append("full scan without a predicate (index cannot reduce rows read)")
        return RuleResult(0, "R0_dominant_scan", _label(share), top, ev)

    if top.node_type in JOIN_TYPES and share >= DOMINANT_SHARE:
        kids = _children(plan, top.node_id)
        ev.append(f"join type {top.node_type}" + (f" ({top.join_type})" if top.join_type else ""))
        for k in kids:
            ev.append(f"  child {k.parent_relationship}: {k.node_type} rows={k.plan_rows:.0f}")
        return RuleResult(1, "R1_dominant_join", _label(share), top, ev)

    if top.node_type in SORT_AGG_TYPES and share >= DOMINANT_SHARE:
        if top.sort_key:
            ev.append(f"sort key: {', '.join(top.sort_key)}")
        if top.group_key:
            ev.append(f"group key: {', '.join(top.group_key)}")
        return RuleResult(2, "R2_dominant_sort_agg", _label(share), top, ev)

    # inner side of a nested loop that is itself a scan repeated many times
    for n in ranked[:3]:
        if n.node_type in SCAN_TYPES and n.estimated_loops > 100 and _share(n, root_cost) >= DOMINANT_SHARE:
            return RuleResult(0, "R0_repeated_scan", _label(_share(n, root_cost)), n, ev)
    # largest scan by share if nothing dominates
    scans = [n for n in ranked if n.node_type in SCAN_TYPES]
    if scans and _share(scans[0], root_cost) >= 0.2 and scans[0].plan_rows * scans[0].estimated_loops >= LARGE_SCAN_ROWS_SCANNED:
        return RuleResult(0, "R0_largest_scan", "low", scans[0], ev)
    ev.append("no single operator dominates the plan cost")
    return RuleResult(3, "R3_no_dominant_operator", "medium" if share < 0.2 else "low", top, ev)


def _in_subplan(plan: PlanAnalysis, node: PlanNode) -> bool:
    cur = node
    while cur.parent_id is not None:
        if cur.parent_relationship == "SubPlan":
            return True
        cur = plan.nodes[cur.parent_id]
    return False


def summarize(result: RuleResult, plan: PlanAnalysis, mode_note: str) -> str:
    n = result.focus_node
    where = f" on {n.relation_name}" if n.relation_name else ""
    loops = f", executed ~{n.estimated_loops:.0f} times" if n.estimated_loops > 1 else ""
    base = (f"{BOTTLENECK_CLASSES[result.class_id]}: the dominant operator is {n.node_type}{where}"
            f"{loops}, accounting for ~{_share(n, max(plan.total_cost, 1e-9)):.0%} of the planner-estimated cost.")
    advice = {
        0: " An index on the filtered/joined columns may reduce rows read; verify with HypoPG.",
        1: " Join strategy is costly; check join-column indexes and row-estimate accuracy.",
        2: " Sorting/aggregation dominates; an index matching the ORDER BY / GROUP BY or pre-aggregation may help.",
        3: " No operator dominates; structural changes are unlikely to give large planner-cost gains.",
    }[result.class_id]
    return base + advice + f" ({mode_note})"
