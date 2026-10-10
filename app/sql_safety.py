"""SQL safety gate for all generated SQL (Final Upgrade P4).

Ten stages, in order, fail fast:
  1 input hygiene + exactly one statement      6 column binding (friendly suggestions; EXPLAIN is authoritative)
  2 parse with the PostgreSQL parser           7 function allowlist (denylist from the existing gateway)
  3 top-level node must be SELECT              8 EXPLAIN (FORMAT JSON, no ANALYZE) as the query role; cost cap;
  4 AST walk (DML CTEs, SELECT INTO, locking,    unconstrained cross joins rejected
    catalogs, temp schemas, nesting depth)     9 row cap (LIMIT added unless single-row aggregate)
  5 relations exist in the active schema       10 execute as the restricted role in a READ ONLY transaction with
    snapshot; qualified names must point to       statement_timeout, lock_timeout, pinned search_path; rolled back
    the active schema
Stages 1-4 reuse the existing `SecurityGateway` (no second gateway). Parser validation is the first wall; the
restricted database role (qs_nlq_reader / per-sandbox role, read-only transaction) is the real wall.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any

import asyncpg
from pglast import ast, parse_sql, visitors
from pglast.stream import RawStream

from app.config import get_settings
from app.datasource import DataSourceContext
from app.errors import ErrorCategory, QuerySentinelError, SecurityRejection
from app.security import SecurityGateway

STAGES = {1: "input hygiene", 2: "parse", 3: "statement type", 4: "AST safety walk", 5: "relation check",
          6: "column binding", 7: "function allowlist", 8: "EXPLAIN plan check", 9: "row cap", 10: "execution"}
ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "round", "trunc", "coalesce", "nullif", "greatest", "least", "abs", "ceil",
    "ceiling", "floor", "sign", "sqrt", "power", "mod", "ln", "log", "exp", "date_trunc", "date_part", "extract",
    "age", "make_date", "make_timestamp", "make_interval", "to_char", "to_date", "to_timestamp", "to_number", "now",
    "lower", "upper", "initcap", "length", "char_length", "substring", "substr", "trim", "btrim", "ltrim", "rtrim",
    "replace", "split_part", "left", "right", "concat", "concat_ws", "position", "strpos", "lpad", "rpad",
    "string_agg", "array_agg", "array_length", "unnest", "cardinality", "jsonb_extract_path_text",
    "json_extract_path_text", "jsonb_array_length", "bool_and", "bool_or", "every", "stddev", "stddev_pop",
    "stddev_samp", "variance", "var_pop", "var_samp", "corr", "covar_pop", "regr_slope", "percentile_cont",
    "percentile_disc", "mode", "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag",
    "lead", "first_value", "last_value", "nth_value", "date", "generate_series", "width_bucket", "md5",
    "timezone", "isfinite", "justify_interval", "num_nonnulls", "num_nulls", "format",
}
MAX_NESTING = 8
MAX_NLQ_SQL_LENGTH = 8000


class SQLRejected(QuerySentinelError):
    def __init__(self, stage: int, message: str, suggestions: list[str] | None = None, **details: Any):
        super().__init__(ErrorCategory.SQL_REJECTED, message, stage=f"sql_safety.{stage}",
                         details={"stage": stage, "stage_name": STAGES[stage],
                                  "suggestions": suggestions or [], **details})
        self.stage_no = stage


@dataclass
class SafetyResult:
    sql: str                                 # final SQL (row cap applied)
    relations: list[str]
    functions: list[str]
    plan: dict | None = None
    plan_total_cost: float | None = None
    limit_added: bool = False
    single_row_aggregate: bool = False
    stages_passed: list[int] = field(default_factory=list)
    validated_sql: str = ""                  # validated SQL WITHOUT the added row cap (what the optimizer analyses)
    notes: list[str] = field(default_factory=list)   # e.g. automatic table-name corrections shown to the user


class _Walk(visitors.Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.relations: list[tuple[str | None, str, str | None]] = []   # (schema, name, alias)
        self.cte_names: set[str] = set()
        self.colrefs: list[list[str]] = []
        self.funcs: list[str] = []
        self.violations: list[str] = []
        self.cross_joins = 0
        self.has_subquery = False

    def visit_RangeVar(self, ancestors, node: ast.RangeVar):
        sch = node.schemaname.lower() if node.schemaname else None
        name = node.relname
        if sch in ("pg_catalog", "information_schema") or (name or "").lower().startswith("pg_"):
            self.violations.append(f"system catalog access ({name}) is not allowed")
        if sch and (sch.startswith("pg_temp") or sch.startswith("pg_toast")):
            self.violations.append("temporary / internal schemas are not allowed")
        if node.relpersistence == "t":
            self.violations.append("temporary tables are not allowed")
        self.relations.append((sch, name, node.alias.aliasname if node.alias else None))

    def visit_CommonTableExpr(self, ancestors, node):
        self.cte_names.add(node.ctename)

    def visit_ColumnRef(self, ancestors, node: ast.ColumnRef):
        parts = []
        for f in node.fields or ():
            if isinstance(f, ast.String):
                parts.append(f.sval)
            else:
                parts.append("*")
        self.colrefs.append(parts)

    def visit_FuncCall(self, ancestors, node):
        parts = [getattr(p, "sval", "") for p in (node.funcname or ())]
        self.funcs.append(".".join(p.lower() for p in parts))

    def visit_SubLink(self, ancestors, node):
        self.has_subquery = True

    def visit_RangeSubselect(self, ancestors, node):
        self.has_subquery = True

    def visit_JoinExpr(self, ancestors, node: ast.JoinExpr):
        if node.quals is None and not node.usingClause and not node.isNatural:
            self.cross_joins += 1


def _depth(node, d: int = 0) -> int:
    """Nesting depth of SELECT statements (subqueries, derived tables, CTE bodies)."""
    best = d
    if isinstance(node, ast.Node):
        for name in node:
            v = getattr(node, name)
            nd = d + 1 if isinstance(v, ast.SelectStmt) else d
            best = max(best, _depth(v, nd))
    elif isinstance(node, tuple):
        for v in node:
            nd = d + 1 if isinstance(v, ast.SelectStmt) else d
            best = max(best, _depth(v, nd))
    return best


class SQLSafetyGate:
    def __init__(self, ctx: DataSourceContext, snapshot: dict[str, list[str]], settings=None):
        """`snapshot` = {table: [columns]} of the ACTIVE schema (SchemaInspector snapshot, hidden cols removed)."""
        self.ctx = ctx
        self.snapshot = {t.lower(): [c.lower() for c in cols] for t, cols in snapshot.items()}
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ stages 1-7 (static)
    def check_static(self, sql: str) -> tuple[ast.SelectStmt, _Walk, SafetyResult]:
        if sql is None or not str(sql).strip():
            raise SQLRejected(1, "empty SQL")
        if len(sql) > MAX_NLQ_SQL_LENGTH:
            raise SQLRejected(1, f"SQL longer than {MAX_NLQ_SQL_LENGTH} characters")
        try:
            vq = SecurityGateway(max_query_length=MAX_NLQ_SQL_LENGTH).validate(sql)
        except SecurityRejection as exc:
            rule = (exc.details or {}).get("rule")
            if exc.category == ErrorCategory.INVALID_SQL and rule is None and "parsed" in exc.message:
                raise SQLRejected(2, "SQL could not be parsed by the PostgreSQL parser",
                                  position=(exc.details or {}).get("position")) from None
            stage = {"max_query_length": 1, "single_statement": 1, "statement_type": 3, "content": 4}.get(rule, 1)
            raise SQLRejected(stage, exc.message) from None
        res = SafetyResult(sql=vq.sql, relations=[], functions=[], stages_passed=[1, 2, 3])
        stmt = parse_sql(vq.sql)[0].stmt
        fixed = self._autofix_tables(stmt)
        if fixed:
            res.sql = RawStream()(stmt)
            res.notes.extend(fixed)
        w = _Walk()
        w(stmt)
        if w.violations:
            raise SQLRejected(4, "; ".join(sorted(set(w.violations))))
        if _depth(stmt) > MAX_NESTING:
            raise SQLRejected(4, f"query nesting deeper than {MAX_NESTING} levels")
        res.stages_passed.append(4)
        # ---- stage 5: relations must exist in the active schema snapshot
        alias_map: dict[str, str] = {}
        tables_used: list[str] = []
        for sch, name, alias in w.relations:
            lname = (name or "").lower()
            if sch is None and lname in {c.lower() for c in w.cte_names}:
                continue
            if sch is not None and sch != self.ctx.schema_name:
                raise SQLRejected(5, f"relation '{sch}.{name}' is outside the active data source schema "
                                     f"'{self.ctx.schema_name}'")
            if lname not in self.snapshot:
                near = [t for t in self.snapshot if lname in t or t in lname][:3] or \
                    difflib.get_close_matches(lname, list(self.snapshot), n=3, cutoff=0.5)
                raise SQLRejected(5, f"table '{name}' does not exist in the active data source",
                                  suggestions=[f"did you mean '{n}'?" for n in near] or
                                  [f"available tables: {', '.join(sorted(self.snapshot)[:15])}"])
            tables_used.append(lname)
            alias_map[(alias or lname).lower()] = lname
            alias_map[lname] = lname
        res.relations = sorted(set(tables_used))
        res.stages_passed.append(5)
        # ---- stage 6: column binding (friendly; EXPLAIN in stage 8 is authoritative)
        out_aliases = {t.name.lower() for t in (stmt.targetList or ()) if isinstance(t, ast.ResTarget) and t.name}
        if not w.has_subquery and not w.cte_names:
            all_cols = {c for t in res.relations for c in self.snapshot[t]}
            for parts in w.colrefs:
                if parts[-1] == "*":
                    continue
                col = parts[-1].lower()
                if len(parts) >= 2:
                    tbl = alias_map.get(parts[-2].lower())
                    if tbl is None:
                        raise SQLRejected(6, f"unknown table or alias '{parts[-2]}'",
                                          suggestions=[f"tables in this query: {', '.join(res.relations)}"])
                    if col not in self.snapshot[tbl]:
                        near = difflib.get_close_matches(col, self.snapshot[tbl], n=3, cutoff=0.5)
                        raise SQLRejected(6, f"column '{parts[-2]}.{parts[-1]}' does not exist",
                                          suggestions=[f"did you mean '{tbl}.{n}'?" for n in near] or
                                          [f"columns of {tbl}: {', '.join(self.snapshot[tbl][:20])}"])
                elif col not in all_cols and col not in out_aliases:
                    near = difflib.get_close_matches(col, sorted(all_cols), n=3, cutoff=0.5)
                    raise SQLRejected(6, f"column '{parts[-1]}' does not exist in {', '.join(res.relations)}",
                                      suggestions=[f"did you mean '{n}'?" for n in near])
        res.stages_passed.append(6)
        # ---- stage 7: function allowlist
        extra = {f.strip().lower() for f in self.settings.nlq_allowed_functions.split(",") if f.strip()}
        for f in w.funcs:
            base = f.split(".")[-1]
            if f.startswith("pg_catalog.") and base in ALLOWED_FUNCTIONS | extra:
                continue
            if base not in ALLOWED_FUNCTIONS | extra or "." in f:
                raise SQLRejected(7, f"function '{f}' is not on the allowlist")
        res.functions = sorted(set(w.funcs))
        res.stages_passed.append(7)
        if w.cross_joins or (len(stmt.fromClause or ()) > 1 and stmt.whereClause is None):
            raise SQLRejected(8, "unconstrained cross join (add a join condition)")
        return stmt, w, res

    def table_candidate(self, name: str) -> str | None:
        """The one table of the active source a mistyped/guessed name clearly means, else None."""
        lname = name.lower()
        tables = list(self.snapshot)
        if len(tables) == 1:
            return tables[0]
        hits = [t for t in tables if lname and (lname in t or t in lname)]
        return hits[0] if len(hits) == 1 else None    # typos in multi-table sources -> rejected with suggestions

    def _autofix_tables(self, stmt: Any) -> list[str]:
        """Unknown unqualified table names are mapped to the source's table when the match is unambiguous (the only
        table, or the only table containing the name). The original name stays usable as an alias, so `survey.col`
        keeps working. Nothing is fixed when any doubt exists - stage 5 then rejects with suggestions."""
        w = _Walk()
        w(stmt)
        ctes = {c.lower() for c in w.cte_names}
        notes: list[str] = []
        todo: list[tuple[Any, str]] = []
        for node in _range_vars(stmt):
            if node.schemaname is not None or not node.relname:
                continue
            ln = node.relname.lower()
            if ln in ctes or ln in self.snapshot:
                continue
            cand = self.table_candidate(ln)
            if cand is None:
                return []
            todo.append((node, cand))
        for node, cand in todo:
            old = node.relname
            if node.alias is None:
                node.alias = ast.Alias(aliasname=old)
            node.relname = cand
            note = f"table '{old}' does not exist here - used this data source's table '{cand}' instead"
            if note not in notes:
                notes.append(note)
        return notes

    # ------------------------------------------------------------------ stages 8-10 (database)
    async def _enter(self, conn: asyncpg.Connection, timeout_ms: int) -> None:
        await conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        await conn.execute("SET LOCAL lock_timeout = 2000")
        await conn.execute(f"SET LOCAL search_path = {_qi(self.ctx.schema_name)}")
        if self.ctx.query_role:
            await conn.execute(f"SET LOCAL ROLE {_qi(self.ctx.query_role)}")

    def apply_row_cap(self, stmt: ast.SelectStmt, res: SafetyResult) -> None:
        single = _single_row_agg(stmt)
        res.single_row_aggregate = single
        res.validated_sql = res.sql
        if stmt.limitCount is None and not single:
            res.sql = f"{res.sql}\nLIMIT {int(self.settings.nlq_max_rows)}"
            res.limit_added = True
        res.stages_passed.append(9)

    async def validate(self, sql: str) -> SafetyResult:
        stmt, _w, res = self.check_static(sql)
        try:
            async with self.ctx.db.connection() as conn:
                tx = conn.transaction(readonly=True)
                await tx.start()
                try:
                    await self._enter(conn, self.settings.nlq_statement_timeout_ms)
                    raw = await conn.fetchval("EXPLAIN (FORMAT JSON) " + res.sql)
                finally:
                    await tx.rollback()
        except asyncpg.PostgresError as exc:
            if exc.sqlstate == "42703":           # undefined column: EXPLAIN is the authoritative binder
                raise SQLRejected(6, f"column binding failed: {exc.message}") from None
            if exc.sqlstate == "42P01":
                raise SQLRejected(5, f"relation binding failed: {exc.message}") from None
            if exc.sqlstate == "42501":
                raise SQLRejected(8, "the restricted query role may not read this object") from None
            raise SQLRejected(8, f"PostgreSQL rejected the query during EXPLAIN ({exc.sqlstate})") from None
        doc = json.loads(raw) if isinstance(raw, str) else raw
        plan = doc[0] if isinstance(doc, list) else doc
        cost = float(plan["Plan"]["Total Cost"])
        if cost > self.settings.nlq_max_plan_cost:
            raise SQLRejected(8, f"estimated plan cost {cost:,.0f} exceeds NLQ_MAX_PLAN_COST "
                                 f"({self.settings.nlq_max_plan_cost:,.0f})")
        res.plan, res.plan_total_cost = plan, cost
        res.stages_passed.append(8)
        self.apply_row_cap(stmt, res)
        return res

    async def execute(self, res: SafetyResult) -> dict[str, Any]:
        """Stage 10: run as the restricted role in a READ ONLY transaction, capped rows, always rolled back."""
        cap = int(self.settings.nlq_max_rows)
        try:
            async with self.ctx.db.connection() as conn:
                tx = conn.transaction(readonly=True)
                await tx.start()
                try:
                    await self._enter(conn, self.settings.nlq_statement_timeout_ms)
                    stmt = await conn.prepare(res.sql)
                    cols = [a.name for a in stmt.get_attributes()]
                    cur = await stmt.cursor()
                    rows = await cur.fetch(cap + 1)
                finally:
                    await tx.rollback()
        except asyncpg.QueryCanceledError:
            raise SQLRejected(10, "query exceeded NLQ_STATEMENT_TIMEOUT_MS") from None
        except asyncpg.PostgresError as exc:
            raise SQLRejected(10, f"execution failed ({exc.sqlstate})") from None
        keep = [i for i, c in enumerate(cols) if not c.startswith("_qs_")]   # hidden lineage never shown
        truncated = len(rows) > cap
        out_rows = [[_jsonable(r[i]) for i in keep] for r in rows[:cap]]
        res.stages_passed.append(10)
        return {"columns": [cols[i] for i in keep], "rows": out_rows, "row_count": len(out_rows),
                "truncated": truncated, "row_cap": cap}


def _single_row_agg(stmt: ast.SelectStmt) -> bool:
    if stmt.groupClause or stmt.valuesLists or getattr(stmt, "larg", None) is not None:
        return False
    found = {"agg": False}

    class V(visitors.Visitor):
        def visit_FuncCall(self, anc, node):
            n = [getattr(p, "sval", "") for p in node.funcname][-1].lower()
            if n in ("count", "sum", "avg", "min", "max", "bool_and", "bool_or", "string_agg", "array_agg",
                     "percentile_cont", "percentile_disc", "stddev", "variance", "mode", "every") \
                    and node.over is None:
                found["agg"] = True

    for t in stmt.targetList or ():
        V()(t)
    return found["agg"]


def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _jsonable(v: Any) -> Any:
    import datetime as dt
    import decimal
    import uuid
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, (uuid.UUID, dt.timedelta)):
        return str(v)
    if isinstance(v, (bytes, memoryview)):
        return "<binary>"
    return v


def _range_vars(node: Any) -> list[Any]:
    """Every RangeVar (table reference) in the tree, including subqueries and CTE bodies."""
    out: list[Any] = []

    class _V(visitors.Visitor):
        def visit_RangeVar(self, ancestors, n):  # noqa: N802
            out.append(n)

    _V()(node)
    return out
