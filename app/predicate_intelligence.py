"""Predicate / access-pattern intelligence (Addendum R3).

Derives structural features from (anonymised or real) SQL via the pglast AST:
WHERE / JOIN / ORDER BY / GROUP BY columns with their operator role, column co-occurrence,
correlated subqueries and non-sargable predicates. Only column/table names and parameter
placeholders are recorded - never literal values. Works identically on masked SQL, so the
resulting features are safe to pass to agents.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

from pglast import ast, parse_sql
from pglast.enums import A_Expr_Kind, BoolExprType, SortByDir, SubLinkType

from app.contracts import SchemaMetadata

RANGE_OPS = {"<", ">", "<=", ">="}


@dataclass
class ColumnUse:
    table: str | None
    column: str
    role: str                 # eq | range | in | like | null | neq | join | order | group | select | non_sargable | in_subquery
    param: str | None = None  # "$n" when compared to a parameter
    function: str | None = None
    direction: str | None = None
    correlated: bool = False
    in_or: bool = False
    depth: int = 0


@dataclass
class AccessPattern:
    uses: list[ColumnUse] = field(default_factory=list)
    joins: list[dict[str, Any]] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    subquery_count: int = 0
    correlated_subqueries: list[dict[str, Any]] = field(default_factory=list)
    unresolved_columns: list[str] = field(default_factory=list)
    parse_ok: bool = True

    def per_table(self) -> dict[str, dict[str, list[str]]]:
        out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for u in self.uses:
            if u.table is None:
                continue
            lst = out[u.table][u.role]
            if u.column not in lst:
                lst.append(u.column)
        return {t: dict(r) for t, r in out.items()}

    def cooccurrence(self) -> list[dict[str, Any]]:
        by_table: dict[str, set[str]] = defaultdict(set)
        for u in self.uses:
            if u.table and u.role in ("eq", "range", "in", "like", "null", "join", "in_subquery"):
                by_table[u.table].add(u.column)
        pairs = []
        for t, cols in by_table.items():
            for a, b in combinations(sorted(cols), 2):
                pairs.append({"table": t, "columns": [a, b]})
        return pairs

    def non_sargable(self) -> list[dict[str, Any]]:
        return [{"table": u.table, "column": u.column, "function": u.function}
                for u in self.uses if u.role == "non_sargable"]

    def eq_params(self, table: str) -> dict[str, str]:
        return {u.column: u.param for u in self.uses
                if u.table == table and u.role == "eq" and u.param and not u.in_or}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": self.tables,
            "per_table": self.per_table(),
            "joins": self.joins,
            "cooccurrence": self.cooccurrence(),
            "non_sargable": self.non_sargable(),
            "subquery_count": self.subquery_count,
            "correlated_subqueries": self.correlated_subqueries,
            "unresolved_columns": sorted(set(self.unresolved_columns)),
            "parse_ok": self.parse_ok,
        }


def _names(node) -> list[str]:
    return [getattr(n, "sval", "") for n in (node or ())]


class _Scope:
    def __init__(self, parent: "_Scope | None" = None):
        self.parent = parent
        self.aliases: dict[str, str] = {}     # alias/relname -> table
        self.order: list[str] = []            # tables in this scope (insertion order)

    def add(self, table: str, alias: str | None) -> None:
        self.aliases[alias or table] = table
        self.aliases.setdefault(table, table)
        if table not in self.order:
            self.order.append(table)


class PredicateAnalyzer:
    def __init__(self, schema: SchemaMetadata | None = None, column_index: dict[str, set[str]] | None = None):
        """`column_index`: table -> set(columns), used to resolve unqualified columns.
        For masked SQL pass a pseudonymised index; for real SQL pass `schema`."""
        self.column_index = column_index or {}
        if schema is not None and not self.column_index:
            self.column_index = {t: {c.name for c in m.columns} for t, m in schema.tables.items()}

    # ------------------------------------------------------------------ entry
    def analyze(self, sql: str) -> AccessPattern:
        ap = AccessPattern()
        try:
            stmts = parse_sql(sql)
        except Exception:  # noqa: BLE001
            ap.parse_ok = False
            return ap
        for raw in stmts:
            self._stmt(raw.stmt, None, ap, 0)
        ap.tables = sorted({u.table for u in ap.uses if u.table} | set(ap.tables))
        return ap

    # ------------------------------------------------------------------ statements
    def _stmt(self, stmt, parent: _Scope | None, ap: AccessPattern, depth: int) -> _Scope:
        scope = _Scope(parent)
        if isinstance(stmt, ast.SelectStmt):
            if stmt.withClause is not None:
                for cte in stmt.withClause.ctes or ():
                    self._stmt(cte.ctequery, parent, ap, depth + 1)
            if stmt.larg is not None or stmt.rarg is not None:
                for part in (stmt.larg, stmt.rarg):
                    if part is not None:
                        self._stmt(part, parent, ap, depth)
                return scope
            for item in stmt.fromClause or ():
                self._from_item(item, scope, ap, depth)
            if stmt.whereClause is not None:
                self._expr(stmt.whereClause, scope, ap, depth, in_or=False)
            for rt in stmt.targetList or ():
                val = rt.val
                if isinstance(val, ast.ColumnRef):
                    self._record_col(val, scope, ap, "select", depth)
                elif val is not None:
                    self._expr(val, scope, ap, depth, in_or=False, context="select")
            for g in stmt.groupClause or ():
                if isinstance(g, ast.ColumnRef):
                    self._record_col(g, scope, ap, "group", depth)
                else:
                    self._expr(g, scope, ap, depth, in_or=False, context="group")
            if stmt.havingClause is not None:
                self._expr(stmt.havingClause, scope, ap, depth, in_or=False, context="having")
            for sb in stmt.sortClause or ():
                d = "desc" if sb.sortby_dir == SortByDir.SORTBY_DESC else "asc"
                if isinstance(sb.node, ast.ColumnRef):
                    self._record_col(sb.node, scope, ap, "order", depth, direction=d)
        elif isinstance(stmt, (ast.UpdateStmt, ast.DeleteStmt, ast.InsertStmt)):
            rel = stmt.relation
            scope.add(rel.relname, rel.alias.aliasname if rel.alias else None)
            ap.tables.append(rel.relname)
            where = getattr(stmt, "whereClause", None)
            if where is not None:
                self._expr(where, scope, ap, depth, in_or=False)
        return scope

    def _from_item(self, item, scope: _Scope, ap: AccessPattern, depth: int) -> None:
        if isinstance(item, ast.RangeVar):
            scope.add(item.relname, item.alias.aliasname if item.alias else None)
            ap.tables.append(item.relname)
        elif isinstance(item, ast.JoinExpr):
            self._from_item(item.larg, scope, ap, depth)
            self._from_item(item.rarg, scope, ap, depth)
            if item.quals is not None:
                self._expr(item.quals, scope, ap, depth, in_or=False)
            for s in item.usingClause or ():
                col = s.sval
                owners = [t for t in scope.order if col in self.column_index.get(t, set())] or scope.order[-2:]
                for t in owners:
                    ap.uses.append(ColumnUse(t, col, "join", depth=depth))
                if len(owners) >= 2:
                    ap.joins.append({"left_table": owners[0], "left_column": col, "right_table": owners[1],
                                     "right_column": col, "type": "using"})
        elif isinstance(item, ast.RangeSubselect):
            ap.subquery_count += 1
            self._stmt(item.subquery, scope if item.lateral else None, ap, depth + 1)
            if item.alias is not None:
                scope.aliases[item.alias.aliasname] = f"derived:{item.alias.aliasname}"

    # ------------------------------------------------------------------ expressions
    def _resolve(self, cref: ast.ColumnRef, scope: _Scope) -> tuple[str | None, str | None, bool]:
        """-> (table, column, is_outer_reference)"""
        parts = [getattr(f, "sval", None) for f in cref.fields or ()]
        if not parts or parts[-1] is None:
            return None, None, False
        col = parts[-1]
        qual = parts[-2] if len(parts) >= 2 else None
        s, outer = scope, False
        while s is not None:
            if qual is not None:
                if qual in s.aliases:
                    t = s.aliases[qual]
                    return (None if t.startswith("derived:") else t), col, outer
            else:
                cands = [t for t in s.order if col in self.column_index.get(t, set())]
                if len(cands) == 1:
                    return cands[0], col, outer
                if not self.column_index and len(s.order) == 1:
                    return s.order[0], col, outer
            s, outer = s.parent, True
        return None, col, False

    def _record_col(self, cref, scope, ap, role, depth, direction=None, param=None, in_or=False,
                    function=None) -> tuple[str | None, str | None, bool]:
        t, c, outer = self._resolve(cref, scope)
        if c is None:
            return None, None, False
        if t is None:
            ap.unresolved_columns.append(c)
        ap.uses.append(ColumnUse(t, c, role, param=param, direction=direction, in_or=in_or,
                                 depth=depth, function=function, correlated=outer))
        return t, c, outer

    @staticmethod
    def _value_kind(node) -> tuple[str, str | None]:
        """-> ('const'|'param'|'column'|'func_col'|'expr'|'sublink', param_or_fn)"""
        if isinstance(node, ast.TypeCast):
            inner_kind, extra = PredicateAnalyzer._value_kind(node.arg)
            if inner_kind == "column":
                return "func_col", "cast:" + ".".join(_names(node.typeName.names)[-1:])
            return inner_kind, extra
        if isinstance(node, ast.A_Const):
            return "const", None
        if isinstance(node, ast.ParamRef):
            return "param", f"${node.number}"
        if isinstance(node, ast.ColumnRef):
            return "column", None
        if isinstance(node, ast.FuncCall):
            if any(isinstance(a, (ast.ColumnRef, ast.TypeCast)) for a in node.args or ()):
                return "func_col", (_names(node.funcname)[-1:] or ["fn"])[0]
            return "expr", None
        if isinstance(node, ast.SubLink):
            return "sublink", None
        return "expr", None

    @staticmethod
    def _inner_col(node):
        if isinstance(node, ast.ColumnRef):
            return node
        if isinstance(node, ast.TypeCast):
            return PredicateAnalyzer._inner_col(node.arg)
        if isinstance(node, ast.FuncCall):
            for a in node.args or ():
                c = PredicateAnalyzer._inner_col(a)
                if c is not None:
                    return c
        return None

    def _expr(self, node, scope: _Scope, ap: AccessPattern, depth: int, in_or: bool, context: str = "where") -> None:
        if node is None:
            return
        if isinstance(node, ast.BoolExpr):
            or_here = in_or or node.boolop == BoolExprType.OR_EXPR
            for a in node.args or ():
                self._expr(a, scope, ap, depth, or_here, context)
            return
        if isinstance(node, ast.A_Expr) and context == "where":
            self._a_expr(node, scope, ap, depth, in_or)
            return
        if isinstance(node, ast.NullTest) and context == "where":
            if isinstance(node.arg, ast.ColumnRef):
                self._record_col(node.arg, scope, ap, "null", depth, in_or=in_or)
            return
        if isinstance(node, ast.SubLink):
            ap.subquery_count += 1
            self._sublink(node, scope, ap, depth, in_or)
            return
        # generic descent (function args, CASE, casts...) to find nested sublinks
        for name in getattr(node, "__slots__", {}):
            child = getattr(node, name, None)
            if isinstance(child, ast.Node):
                self._expr(child, scope, ap, depth, in_or, context if isinstance(child, ast.BoolExpr) else "generic")
            elif isinstance(child, tuple):
                for c in child:
                    if isinstance(c, ast.Node):
                        self._expr(c, scope, ap, depth, in_or, "generic")

    def _sublink(self, node: ast.SubLink, scope: _Scope, ap: AccessPattern, depth: int, in_or: bool) -> None:
        before = len(ap.uses)
        inner_scope = self._stmt(node.subselect, scope, ap, depth + 1)
        correlated = [u for u in ap.uses[before:] if u.correlated]
        if node.subLinkType == SubLinkType.ANY_SUBLINK and isinstance(node.testexpr, ast.ColumnRef):
            self._record_col(node.testexpr, scope, ap, "in_subquery", depth, in_or=in_or)
        if correlated:
            inner_tables = sorted({u.table for u in ap.uses[before:] if not u.correlated and u.table})
            ap.correlated_subqueries.append({
                "type": SubLinkType(node.subLinkType).name,
                "inner_tables": inner_tables,
                "outer_columns": sorted({f"{u.table}.{u.column}" for u in correlated}),
                "inner_join_columns": sorted({f"{u.table}.{u.column}" for u in ap.uses[before:]
                                              if u.role == "join" and not u.correlated}),
            })
        del inner_scope

    def _a_expr(self, node: ast.A_Expr, scope: _Scope, ap: AccessPattern, depth: int, in_or: bool) -> None:
        kind = node.kind
        op = (_names(node.name) or [""])[-1]
        lk, lx = self._value_kind(node.lexpr) if node.lexpr is not None else ("expr", None)
        rk, rx = self._value_kind(node.rexpr) if not isinstance(node.rexpr, tuple) else ("list", None)
        for side in (node.lexpr, node.rexpr):
            if isinstance(side, ast.SubLink):
                ap.subquery_count += 1
                self._sublink(side, scope, ap, depth, in_or)
        if kind == A_Expr_Kind.AEXPR_OP and lk == "column" and rk == "column":
            lt, lc, lo = self._resolve(node.lexpr, scope)
            rt, rc, ro = self._resolve(node.rexpr, scope)
            if op == "=" and lt and rt and (lt != rt or lo != ro):
                ap.uses.append(ColumnUse(lt, lc, "join", depth=depth, correlated=lo, in_or=in_or))
                ap.uses.append(ColumnUse(rt, rc, "join", depth=depth, correlated=ro, in_or=in_or))
                ap.joins.append({"left_table": lt, "left_column": lc, "right_table": rt, "right_column": rc,
                                 "type": "correlated" if (lo or ro) else "inner"})
            return
        # orient: column on one side, value on the other
        col_node, val_kind, val_extra, func = None, None, None, None
        if lk in ("column", "func_col") and rk not in ("column", "func_col"):
            col_node, val_kind, val_extra, func = node.lexpr, rk, rx, lx if lk == "func_col" else None
        elif rk in ("column", "func_col") and lk not in ("column", "func_col"):
            col_node, val_kind, val_extra, func = node.rexpr, lk, lx, rx if rk == "func_col" else None
        if col_node is None:
            return
        cref = self._inner_col(col_node)
        if cref is None:
            return
        param = val_extra if val_kind == "param" else None
        if func is not None:
            self._record_col(cref, scope, ap, "non_sargable", depth, in_or=in_or, function=func, param=param)
            return
        if kind == A_Expr_Kind.AEXPR_OP:
            role = "eq" if op == "=" else "range" if op in RANGE_OPS else "neq" if op in ("<>", "!=") else "other"
        elif kind in (A_Expr_Kind.AEXPR_IN, A_Expr_Kind.AEXPR_OP_ANY):
            role = "in"
        elif kind in (A_Expr_Kind.AEXPR_LIKE, A_Expr_Kind.AEXPR_ILIKE):
            role = "like"
        elif kind in (A_Expr_Kind.AEXPR_BETWEEN, A_Expr_Kind.AEXPR_BETWEEN_SYM):
            role = "range"
        else:
            role = "other"
        self._record_col(cref, scope, ap, role, depth, in_or=in_or, param=param)


# ---------------------------------------------------------------------- workload aggregation
def aggregate_workload(patterns: Iterable[tuple[AccessPattern, float]]) -> dict[str, Any]:
    """Weighted predicate frequency over a workload: [(pattern, weight=calls)]."""
    freq: Counter = Counter()
    combos: Counter = Counter()
    for ap, w in patterns:
        seen = set()
        for u in ap.uses:
            if u.table and u.role in ("eq", "range", "in", "join", "order", "group", "like"):
                seen.add((u.table, u.column, u.role))
        for key in seen:
            freq[key] += w
        for pair in ap.cooccurrence():
            combos[(pair["table"], tuple(pair["columns"]))] += w
    return {
        "predicate_frequency": [{"table": t, "column": c, "role": r, "weight": wt}
                                for (t, c, r), wt in freq.most_common(50)],
        "cooccurrence_frequency": [{"table": t, "columns": list(cols), "weight": wt}
                                   for (t, cols), wt in combos.most_common(30)],
    }
