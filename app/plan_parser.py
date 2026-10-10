"""PostgreSQL EXPLAIN (FORMAT JSON) analysis (Section 18).

Builds a reusable flat plan tree (nodes + parent/child edges) with exclusive cost, operator
histogram, plan fingerprint and shape. `sanitize_plan` replaces literals in expressions and
pseudonymises relation/column/index names before anything leaves the trusted backend.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from app.contracts import PlanAnalysis, PlanNode
from app.masking_vault import DataObfuscator


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_explain(doc: dict | list) -> PlanAnalysis:
    """Parse one EXPLAIN JSON document ({'Plan': {...}} or [{'Plan': ...}])."""
    if isinstance(doc, list):
        if not doc:
            raise ValueError("empty EXPLAIN document")
        doc = doc[0]
    if "Plan" not in doc:
        raise ValueError("EXPLAIN document has no 'Plan' key")
    nodes: list[PlanNode] = []
    edges: list[tuple[int, int]] = []

    def walk(p: dict, parent: int | None, depth: int, loops: float) -> int:
        nid = len(nodes)
        node = PlanNode(
            node_id=nid, parent_id=parent, depth=depth, node_type=p.get("Node Type", "Unknown"),
            parent_relationship=p.get("Parent Relationship"),
            startup_cost=_f(p.get("Startup Cost")), total_cost=_f(p.get("Total Cost")),
            plan_rows=_f(p.get("Plan Rows")), plan_width=int(_f(p.get("Plan Width"))),
            relation_name=p.get("Relation Name"), alias=p.get("Alias"), index_name=p.get("Index Name"),
            filter=p.get("Filter"), index_cond=p.get("Index Cond"), recheck_cond=p.get("Recheck Cond"),
            join_type=p.get("Join Type"), hash_cond=p.get("Hash Cond"), merge_cond=p.get("Merge Cond"),
            join_filter=p.get("Join Filter"), sort_key=list(p.get("Sort Key") or []),
            group_key=list(p.get("Group Key") or []), strategy=p.get("Strategy"),
            shared_hit_blocks=p.get("Shared Hit Blocks"), shared_read_blocks=p.get("Shared Read Blocks"),
            actual_rows=p.get("Actual Rows"), actual_loops=p.get("Actual Loops"),
            actual_total_time=p.get("Actual Total Time"), subplan_name=p.get("Subplan Name"),
            parallel_aware=bool(p.get("Parallel Aware", False)), estimated_loops=loops,
        )
        nodes.append(node)
        if parent is not None:
            edges.append((parent, nid))
        kids = p.get("Plans", []) or []
        child_cost = 0.0
        outer_rows = None
        for idx, child in enumerate(kids):
            rel = child.get("Parent Relationship")
            mult = 1.0
            if rel == "SubPlan":
                # correlated SubPlan: executed once per row produced by this node (planner estimate)
                mult = max(node.plan_rows, 1.0)
            elif node.node_type == "Nested Loop" and rel == "Inner" and outer_rows is not None:
                mult = max(outer_rows, 1.0)
            elif node.node_type == "Limit" and _f(child.get("Total Cost")) > node.total_cost > 0:
                # LIMIT stops its input early: only the fraction of the child's cost the planner
                # expects to consume is attributed (Limit total / child total).
                mult = node.total_cost / _f(child.get("Total Cost"))
            cid = walk(child, nid, depth + 1, loops * mult)
            if rel == "Outer":
                outer_rows = nodes[cid].plan_rows
            child_cost += nodes[cid].total_cost * (mult if rel in ("SubPlan", "Inner") or node.node_type == "Limit" else 1.0)
        # Planner charges rescans of a nested-loop inner side more cheaply than full runs
        # (e.g. Materialize / Memoize); clamp so attribution never goes negative.
        node.exclusive_cost = max(node.total_cost - child_cost, 0.0)
        node.effective_cost = node.exclusive_cost * loops
        return nid

    walk(doc["Plan"], None, 0, 1.0)
    root = nodes[0]
    hist = Counter(n.node_type for n in nodes)
    top = max(nodes, key=lambda n: n.effective_cost)
    shape = plan_shape(nodes, edges)
    seq_scans = [{"node_id": n.node_id, "relation": n.relation_name, "total_cost": n.total_cost,
                  "exclusive_cost": n.exclusive_cost, "estimated_loops": n.estimated_loops,
                  "effective_cost": n.effective_cost, "plan_rows": n.plan_rows, "filter": n.filter,
                  "parallel": n.parallel_aware,
                  "cost_share": round(min(n.effective_cost / root.total_cost, 1.0), 4) if root.total_cost else None}
                 for n in nodes if n.node_type == "Seq Scan"]
    planning = doc.get("Planning") or {}
    return PlanAnalysis(
        plan_fingerprint=plan_fingerprint(nodes),
        plan_shape=shape,
        total_cost=root.total_cost, startup_cost=root.startup_cost, plan_rows=root.plan_rows,
        node_count=len(nodes), nodes=nodes, edges=edges, operator_histogram=dict(hist),
        highest_cost_node=_node_summary(top, root.total_cost),
        seq_scans=seq_scans,
        has_actuals=any(n.actual_rows is not None for n in nodes),
        sanitized=False,
        planning_shared_hit_blocks=planning.get("Shared Hit Blocks"),
    )


def _node_summary(n: PlanNode, root_cost: float) -> dict[str, Any]:
    return {"node_id": n.node_id, "node_type": n.node_type, "relation_name": n.relation_name,
            "index_name": n.index_name, "total_cost": n.total_cost, "exclusive_cost": round(n.exclusive_cost, 2),
            "estimated_loops": n.estimated_loops, "effective_cost": round(n.effective_cost, 2),
            "cost_share": round(min(n.effective_cost / root_cost, 1.0), 4) if root_cost else None,
            "parent_relationship": n.parent_relationship,
            "plan_rows": n.plan_rows, "filter": n.filter, "join_type": n.join_type,
            "sort_key": n.sort_key, "group_key": n.group_key, "depth": n.depth}


def plan_shape(nodes: list[PlanNode], edges: list[tuple[int, int]]) -> str:
    """Identifier-free structural shape, e.g. 'Limit(Sort(Nested Loop(Seq Scan,Index Scan)))'."""
    children: dict[int, list[int]] = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)

    def render(i: int) -> str:
        ch = children.get(i, [])
        label = nodes[i].node_type
        return f"{label}({','.join(render(c) for c in ch)})" if ch else label

    return render(0) if nodes else ""


def plan_fingerprint(nodes: list[PlanNode]) -> str:
    """Structure-only fingerprint: operator types, relations, indexes, join types, depth.
    Costs/rows are excluded so the same plan shape on the same objects hashes equal."""
    parts = [f"{n.depth}:{n.node_type}:{n.relation_name or ''}:{n.index_name or ''}:{n.join_type or ''}"
             for n in nodes]
    return "plan_" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- sanitizer
_MULTIWORD_TYPES = r"(?:timestamp|time)(?:\(\d+\))? with(?:out)? time zone|character varying(?:\(\d+\))?|double precision|bit varying"
_TYPED_LITERAL = re.compile(r"'(?:[^']|'')*'(?:::(?:" + _MULTIWORD_TYPES + r'|"?[\w.]+"?(?:\[\])?(?:\(\d+(?:,\d+)?\))?))*')
_NUMBER = re.compile(r"(?<![\w$.])-?\d+(?:\.\d+)?(?![\w.])")


def sanitize_expression(expr: str | None, ident_map: dict[str, str]) -> str | None:
    """Replace literals with '?' and real identifiers with pseudonyms (longest first)."""
    if expr is None:
        return None
    out = _TYPED_LITERAL.sub("?", expr)
    out = _NUMBER.sub("?", out)
    for original in sorted(ident_map, key=len, reverse=True):
        out = re.sub(rf"(?<![A-Za-z0-9_$]){re.escape(original)}(?![A-Za-z0-9_])", ident_map[original], out)
    return out


def sanitize_plan(analysis: PlanAnalysis, obfuscator: DataObfuscator, request_id: str) -> PlanAnalysis:
    vault = obfuscator.get_vault(request_id)
    forward = dict(vault.forward) if vault else {}
    extra: dict[str, str] = {}
    for n in analysis.nodes:
        if n.relation_name and n.relation_name not in forward:
            extra[n.relation_name] = "table"
        if n.alias and n.alias not in forward:
            extra[n.alias] = "alias"
        if n.index_name and n.index_name not in forward:
            extra[n.index_name] = "index"
    if extra:
        forward.update(obfuscator.pseudonymize_many(request_id, extra))
    # 'public' schema qualifiers that VERBOSE output adds are not sensitive but are removed for brevity
    def clean(e):
        return sanitize_expression(e, forward)

    new_nodes = []
    for n in analysis.nodes:
        new_nodes.append(n.model_copy(update={
            "relation_name": forward.get(n.relation_name, n.relation_name) if n.relation_name else None,
            "alias": forward.get(n.alias, n.alias) if n.alias else None,
            "index_name": forward.get(n.index_name, n.index_name) if n.index_name else None,
            "filter": clean(n.filter), "index_cond": clean(n.index_cond), "recheck_cond": clean(n.recheck_cond),
            "hash_cond": clean(n.hash_cond), "merge_cond": clean(n.merge_cond), "join_filter": clean(n.join_filter),
            "sort_key": [clean(s) for s in n.sort_key], "group_key": [clean(s) for s in n.group_key],
        }))
    root_cost = analysis.total_cost
    top = max(new_nodes, key=lambda x: x.effective_cost)
    seq = [{**s, "relation": forward.get(s["relation"], s["relation"]), "filter": clean(s["filter"])}
           for s in analysis.seq_scans]
    # fingerprint recomputed over pseudonymised names so stored fingerprints never embed real names
    return analysis.model_copy(update={"nodes": new_nodes, "highest_cost_node": _node_summary(top, root_cost),
                                       "plan_fingerprint": plan_fingerprint(new_nodes),
                                       "seq_scans": seq, "sanitized": True})


def plan_to_tree_text(analysis: PlanAnalysis) -> str:
    """Indented text tree for the dashboard."""
    lines = []
    for n in analysis.nodes:
        rel = f" on {n.relation_name}" if n.relation_name else ""
        idx = f" using {n.index_name}" if n.index_name else ""
        loops = f" loops~{n.estimated_loops:.0f}" if n.estimated_loops > 1 else ""
        sub = f"[{n.parent_relationship}] " if n.parent_relationship in ("SubPlan", "InitPlan") else ""
        lines.append(f"{'  ' * n.depth}-> {sub}{n.node_type}{idx}{rel}  (cost={n.startup_cost:.2f}..{n.total_cost:.2f} "
                     f"rows={n.plan_rows:.0f} self={n.exclusive_cost:.2f}{loops})")
        for label, val in (("Filter", n.filter), ("Index Cond", n.index_cond), ("Hash Cond", n.hash_cond),
                           ("Join Filter", n.join_filter)):
            if val:
                lines.append(f"{'  ' * n.depth}     {label}: {val}")
    return "\n".join(lines)
