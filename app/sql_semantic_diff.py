"""SQL semantic-diff safety (Addendum R6).

Compares the pglast ASTs of the original and the rewritten statement and reports structural
changes: tables, output columns, WHERE predicates, JOIN structure, GROUP BY, ORDER BY, LIMIT,
subquery count. A structural diff cannot prove semantic equivalence on its own; each rewrite
rule declares which changes it is ALLOWED to make, and any other change causes rejection /
mandatory human review.
"""
from __future__ import annotations

from typing import Any

from pglast import ast, parse_sql, visitors
from pglast.stream import RawStream


class _Collect(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.tables: list[str] = []
        self.joins: list[str] = []
        self.sublinks = 0
        self.subselects = 0

    def visit_RangeVar(self, anc, node):
        self.tables.append(node.relname)

    def visit_JoinExpr(self, anc, node):
        self.joins.append(str(getattr(node.jointype, "name", node.jointype)))

    def visit_SubLink(self, anc, node):
        self.sublinks += 1

    def visit_RangeSubselect(self, anc, node):
        self.subselects += 1


def _conjuncts(node) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.BoolExpr) and int(node.boolop) == 0:   # AND
        out = []
        for a in node.args:
            out += _conjuncts(a)
        return out
    return [RawStream()(node)]


def structure(sql: str) -> dict[str, Any]:
    stmt = parse_sql(sql)[0].stmt
    c = _Collect()
    c(stmt)
    targets = []
    for rt in getattr(stmt, "targetList", None) or ():
        targets.append(rt.name or (RawStream()(rt.val) if rt.val is not None else "?"))
    return {
        "tables": sorted(set(c.tables)),
        "output_columns": targets,
        "where": sorted(_conjuncts(getattr(stmt, "whereClause", None))),
        "join_types": sorted(c.joins),
        "group_by": [RawStream()(g) for g in getattr(stmt, "groupClause", None) or ()],
        "order_by": [RawStream()(s) for s in getattr(stmt, "sortClause", None) or ()],
        "limit": RawStream()(stmt.limitCount) if getattr(stmt, "limitCount", None) is not None else None,
        "distinct": getattr(stmt, "distinctClause", None) is not None,
        "subquery_count": c.sublinks + c.subselects,
        "offset": RawStream()(stmt.limitOffset) if getattr(stmt, "limitOffset", None) is not None else None,
        "aggregates": sorted(_aggregates(stmt)),
        "having": sorted(_conjuncts(getattr(stmt, "havingClause", None))),
    }


_AGG = {"count", "sum", "avg", "min", "max", "array_agg", "string_agg", "bool_and", "bool_or", "stddev",
        "variance", "percentile_cont", "percentile_disc", "mode", "every"}


def _aggregates(stmt) -> list[str]:
    found: list[str] = []

    class V(visitors.Visitor):
        def visit_FuncCall(self, anc, node):
            n = [getattr(p, "sval", "") for p in node.funcname][-1].lower()
            if n in _AGG:
                found.append(f"{n}{'(distinct)' if node.agg_distinct else ''}{'(*)' if node.agg_star else ''}")

    V()(stmt)
    return found


def diff(original_sql: str, rewritten_sql: str) -> dict[str, Any]:
    a, b = structure(original_sql), structure(rewritten_sql)
    report = {
        "tables_preserved": a["tables"] == b["tables"],
        "output_column_count_preserved": len(a["output_columns"]) == len(b["output_columns"]),
        "output_names_preserved": [x for x in a["output_columns"]] == [x for x in b["output_columns"]],
        "where_preserved": a["where"] == b["where"],
        "join_structure_changed": a["join_types"] != b["join_types"],
        "group_by_changed": a["group_by"] != b["group_by"],
        "order_by_changed": a["order_by"] != b["order_by"],
        "limit_changed": a["limit"] != b["limit"],
        "distinct_changed": a["distinct"] != b["distinct"],
        "subquery_count_changed": a["subquery_count"] != b["subquery_count"],
        "subquery_count": {"before": a["subquery_count"], "after": b["subquery_count"]},
        "tables": {"before": a["tables"], "after": b["tables"]},
        "where_removed": sorted(set(a["where"]) - set(b["where"])),
        "where_added": sorted(set(b["where"]) - set(a["where"])),
    }
    return report


# Changes each rewrite rule is permitted to make. Everything else must be preserved.
ALLOWED_CHANGES: dict[str, set[str]] = {
    "correlated_count_subquery_to_left_join": {"join_structure_changed", "subquery_count_changed",
                                               "output_names_preserved"},
    "cast_date_equality_to_range": {"where_preserved"},
    "remove_duplicate_predicates": {"where_preserved"},
    "or_equalities_to_in_list": {"where_preserved"},
}
MUST_HOLD = ("tables_preserved", "output_column_count_preserved")
MUST_NOT_CHANGE = ("group_by_changed", "order_by_changed", "limit_changed", "distinct_changed")


def verify(rule: str, report: dict[str, Any]) -> tuple[bool, list[str]]:
    """True when only the rule's declared changes occurred."""
    allowed = ALLOWED_CHANGES.get(rule, set())
    problems = [k for k in MUST_HOLD if not report[k]]
    problems += [k for k in MUST_NOT_CHANGE if report[k]]
    if not report["where_preserved"] and "where_preserved" not in allowed:
        problems.append("where_changed")
    if report["join_structure_changed"] and "join_structure_changed" not in allowed:
        problems.append("join_structure_changed")
    if report["subquery_count_changed"] and "subquery_count_changed" not in allowed:
        problems.append("subquery_count_changed")
    if not report["output_names_preserved"] and "output_names_preserved" not in allowed:
        problems.append("output_names_changed")
    return not problems, problems


# ------------------------------------------------------------------ Final Upgrade P6: semantic-risk screening
def risk_screen(original_sql: str, rewritten_sql: str, rule: str | None = None) -> dict[str, Any]:
    """Semantic-risk screening (NOT a proof of equivalence). Returns LOW / MEDIUM / HIGH with reasons.
    HIGH = blocked (never applied, never sent to HypoPG/sandbox as a rewrite); MEDIUM = human review."""
    try:
        a, b = structure(original_sql), structure(rewritten_sql)
    except Exception:  # noqa: BLE001 - unparsable rewrite is the highest risk
        return {"risk": "HIGH", "reasons": ["rewrite could not be parsed"], "blocked": True,
                "terminology": "semantic-risk screening (not a formal equivalence proof)"}
    allowed = ALLOWED_CHANGES.get(rule or "", set())
    high, medium = [], []
    if a["tables"] != b["tables"]:
        high.append(f"referenced tables changed {a['tables']} -> {b['tables']}")
    if len(a["output_columns"]) != len(b["output_columns"]):
        high.append("number of output columns changed")
    elif a["output_columns"] != b["output_columns"]:
        (medium if "output_names_preserved" in allowed else medium).append("output column names/aliases changed")
    removed = sorted(set(a["where"]) - set(b["where"]))
    added = sorted(set(b["where"]) - set(a["where"]))
    if removed or added:
        if "where_preserved" in allowed:
            medium.append(f"predicates restructured by rule {rule}: removed {removed}, added {added}")
        else:
            if removed:
                high.append(f"predicate(s) dropped or altered: {removed}")
            if added:
                high.append(f"predicate(s) added: {added}")
    if a["group_by"] != b["group_by"]:
        high.append("grouping keys changed")
    if a["aggregates"] != b["aggregates"]:
        high.append(f"aggregation changed {a['aggregates']} -> {b['aggregates']}")
    if a["having"] != b["having"]:
        high.append("HAVING changed")
    if a["limit"] != b["limit"] or a["offset"] != b["offset"]:
        high.append("LIMIT/OFFSET changed")
    if a["distinct"] != b["distinct"]:
        high.append("DISTINCT changed")
    if a["join_types"] != b["join_types"]:
        (medium if "join_structure_changed" in allowed else high).append(
            f"join types changed {a['join_types']} -> {b['join_types']}")
    if a["subquery_count"] != b["subquery_count"]:
        (medium if "subquery_count_changed" in allowed else high).append("subquery structure changed")
    if a["order_by"] != b["order_by"]:
        (high if a["limit"] is not None else medium).append("ORDER BY changed")
    risk = "HIGH" if high else "MEDIUM" if medium else "LOW"
    return {"risk": risk, "reasons": high + medium, "blocked": risk == "HIGH",
            "requires_human_review": risk != "LOW",
            "terminology": "semantic-risk screening (not a formal equivalence proof)"}

