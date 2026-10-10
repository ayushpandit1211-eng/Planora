"""Curriculum levels from the real structure of parsed workload queries (Final Upgrade P9).

Level 1 single-table filtering | 2 multi-predicate filtering | 3 single join | 4 multi-table joins |
5 join + aggregation + sorting | 6 complex evolving workload (level-5 shape with a drift flag or >= 4 tables).
Difficulty is derived by parsing the query (pglast); it is never invented. Unparsable -> None (excluded).
"""
from __future__ import annotations

from typing import Any

from pglast import ast, parse_sql, visitors


def query_features(sql: str) -> dict[str, Any] | None:
    try:
        stmt = parse_sql(sql)[0].stmt
    except Exception:  # noqa: BLE001
        return None
    f = {"tables": 0, "joins": 0, "predicates": 0, "aggregates": 0, "group_by": 0, "order_by": 0}

    class V(visitors.Visitor):
        def visit_RangeVar(self, anc, node):
            f["tables"] += 1

        def visit_JoinExpr(self, anc, node):
            f["joins"] += 1

        def visit_A_Expr(self, anc, node):
            f["predicates"] += 1

        def visit_FuncCall(self, anc, node):
            n = [getattr(p, "sval", "") for p in node.funcname][-1].lower()
            if n in ("count", "sum", "avg", "min", "max"):
                f["aggregates"] += 1

    V()(stmt)
    if isinstance(stmt, ast.SelectStmt):
        f["group_by"] = len(stmt.groupClause or ())
        f["order_by"] = len(stmt.sortClause or ())
        if f["joins"] == 0 and f["tables"] > 1 and stmt.whereClause is not None:
            f["joins"] = f["tables"] - 1          # implicit joins in WHERE
    return f


def level_for(sql: str, drift: bool = False) -> int | None:
    f = query_features(sql)
    if f is None:
        return None
    joined = max(f["joins"], f["tables"] - 1)
    agg_sort = (f["aggregates"] or f["group_by"]) and f["order_by"]
    if joined >= 1 and agg_sort:
        return 6 if (drift or f["tables"] >= 4) else 5
    if joined >= 2:
        return 4
    if joined == 1:
        return 3
    return 2 if f["predicates"] >= 2 else 1


def order_by_level(contexts: list, sql_key: str = "sql") -> list[tuple[int, Any]]:
    out = []
    for c in contexts:
        lvl = level_for(c.meta.get(sql_key) or "", bool(c.meta.get("drift")))
        if lvl is not None:
            out.append((lvl, c))
    return sorted(out, key=lambda t: t[0])


def split_by_pattern(contexts: list, holdout_patterns: set[str]) -> tuple[list, list]:
    """Train / held-out split by query PATTERN (fingerprint), never by random rows of the same pattern."""
    train = [c for c in contexts if c.meta.get("pattern") not in holdout_patterns]
    held = [c for c in contexts if c.meta.get("pattern") in holdout_patterns]
    return train, held
