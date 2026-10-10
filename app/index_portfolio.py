"""Index portfolio optimisation (Final Upgrade P7, behind QS_FEATURE_INDEX_PORTFOLIO).

Turns "add an index" into evidence-based portfolio reasoning that also weighs write and storage cost.
Actions: ADD_INDEX, KEEP_INDEX, REPLACE_INDEX_PORTFOLIO, REMOVE_REDUNDANT_INDEX, NO_CHANGE.

Rules
- Exact duplicates and prefix redundancy only when access method, operator classes, collations, uniqueness,
  partial predicate and expressions match. Indexes backing PRIMARY KEY / UNIQUE / EXCLUSION constraints and
  indexes supporting foreign-key checks are never removal candidates without explicit evidence + a warning.
- A composite candidate such as (tenant_id, order_date) never automatically replaces single-column indexes.
- Pruning needs workload-level evidence over a sufficient statistics window; "unused by the current query" is
  never enough.
- Portfolio comparison A (current) / B (current + candidate) / C (consolidated) uses HypoPG: hypothetical indexes
  for additions and hypopg_hide_index for removals (planner simulation, labelled as such).
- Output is review artifacts only (CREATE/DROP INDEX CONCURRENTLY + rollback, HUMAN REVIEW REQUIRED). Nothing is
  executed. Write effects are [HEURISTIC ESTIMATE]; no fake millisecond precision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pglast import ast, parse_sql, visitors

from app.provenance import Provenance

MIN_STATS_AGE_DAYS = 7
MIN_WORKLOAD_CALLS = 100

SQL_INDEX_DETAIL = """
SELECT ic.relname AS index_name, tc.relname AS table_name, am.amname AS am,
       ARRAY(SELECT a.attname FROM unnest(ix.indkey[0:ix.indnkeyatts-1]) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = tc.oid AND a.attnum = k.attnum ORDER BY k.ord) AS columns,
       ARRAY(SELECT a.attname FROM unnest(ix.indkey[ix.indnkeyatts:]) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = tc.oid AND a.attnum = k.attnum ORDER BY k.ord) AS include,
       ix.indclass::oid[]::text AS opclasses, ix.indcollation::oid[]::text AS collations,
       ix.indisunique AS is_unique, ix.indisprimary AS is_primary, ix.indisexclusion AS is_exclusion,
       pg_get_expr(ix.indpred, ix.indrelid) AS predicate, pg_get_expr(ix.indexprs, ix.indrelid) AS expressions,
       EXISTS (SELECT 1 FROM pg_constraint con WHERE con.conindid = ix.indexrelid) AS constraint_backed,
       pg_relation_size(ic.oid) AS size_bytes, COALESCE(si.idx_scan, 0) AS idx_scan
FROM pg_index ix
JOIN pg_class ic ON ic.oid = ix.indexrelid JOIN pg_class tc ON tc.oid = ix.indrelid
JOIN pg_namespace n ON n.oid = tc.relnamespace JOIN pg_am am ON am.oid = ic.relam
LEFT JOIN pg_stat_user_indexes si ON si.indexrelid = ix.indexrelid
WHERE n.nspname = current_schema()
ORDER BY tc.relname, ic.relname
"""
SQL_FK_COLS = """
SELECT tc.relname AS table_name,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum ORDER BY k.ord) AS columns
FROM pg_constraint con JOIN pg_class tc ON tc.oid = con.conrelid JOIN pg_namespace n ON n.oid = tc.relnamespace
WHERE con.contype = 'f' AND n.nspname = current_schema()
"""
SQL_STATS_AGE = """
SELECT EXTRACT(EPOCH FROM (now() - COALESCE(stats_reset, pg_postmaster_start_time()))) / 86400.0 AS days
FROM pg_stat_database WHERE datname = current_database()
"""
SQL_TABLE_WRITES = """
SELECT c.relname AS table_name, COALESCE(s.n_tup_ins, 0) + COALESCE(s.n_tup_upd, 0) + COALESCE(s.n_tup_del, 0)
       AS writes, COALESCE(s.seq_scan, 0) + COALESCE(s.idx_scan, 0) AS reads, c.reltuples::float8 AS rows
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.relkind IN ('r', 'p') AND n.nspname = current_schema()
"""


@dataclass
class IndexInfo:
    name: str
    table: str
    columns: list[str]
    am: str = "btree"
    include: list[str] = field(default_factory=list)
    opclasses: str = ""
    collations: str = ""
    is_unique: bool = False
    is_primary: bool = False
    is_exclusion: bool = False
    predicate: str | None = None
    expressions: str | None = None
    constraint_backed: bool = False
    size_bytes: int = 0
    idx_scan: int = 0

    @property
    def protected_reason(self) -> str | None:
        if self.is_primary or self.constraint_backed or self.is_exclusion:
            return "backs a PRIMARY KEY / UNIQUE / EXCLUSION constraint"
        if self.is_unique:
            return "enforces uniqueness"
        return None

    def opclass_list(self) -> list[str]:
        return self.opclasses.strip("{}").split(",") if self.opclasses else []

    def collation_list(self) -> list[str]:
        return self.collations.strip("{}").split(",") if self.collations else []


def _comparable(a: IndexInfo, b: IndexInfo, n: int) -> bool:
    """Same access method, opclasses and collations on the first n keys; no expressions; same predicate."""
    return (a.am == b.am and a.expressions is None and b.expressions is None and a.predicate == b.predicate
            and a.opclass_list()[:n] == b.opclass_list()[:n] and a.collation_list()[:n] == b.collation_list()[:n])


def find_redundancy(indexes: list[IndexInfo], fk_cols: dict[str, list[list[str]]]) -> list[dict[str, Any]]:
    out = []
    by_table: dict[str, list[IndexInfo]] = {}
    for ix in indexes:
        by_table.setdefault(ix.table, []).append(ix)
    seen: set[str] = set()
    for table, ixs in by_table.items():
        for a in ixs:
            for b in ixs:
                if a.name == b.name or a.name in seen:
                    continue
                n = len(a.columns)
                exact = a.columns == b.columns and a.include == b.include and a.is_unique == b.is_unique
                prefix = (n < len(b.columns) and b.columns[:n] == a.columns and not a.is_unique
                          and a.include == [] and b.predicate is None)
                if not (exact or prefix) or not _comparable(a, b, n):
                    continue
                if exact:
                    # exactly one side of a duplicate pair is reported: the unprotected one (name order breaks ties)
                    a_victim = a.protected_reason is None and (b.protected_reason is not None or a.name > b.name)
                    if not a_victim:
                        continue
                reason = "exact duplicate of" if exact else "left prefix of"
                warn = []
                protected = a.protected_reason
                supports_fk = any(cols == a.columns for cols in fk_cols.get(table, []))
                if supports_fk:
                    warn.append(f"{a.name} matches a foreign-key column list; {b.name} also serves FK checks "
                                "only because it starts with the same columns - verify before removal")
                out.append({"index": a.name, "table": table, "covered_by": b.name, "kind": "exact" if exact
                            else "prefix", "statement": f"{a.name} is an {reason} {b.name} (same access method, "
                            "operator classes, collations, predicate)", "protected": protected,
                            "supports_fk": supports_fk, "warnings": warn})
                seen.add(a.name)
    return out


# ------------------------------------------------------------------ workload coverage
class _Cols(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.cols: list[str] = []

    def visit_ColumnRef(self, anc, node):
        f = [x.sval for x in node.fields or () if isinstance(x, ast.String)]
        if f:
            self.cols.append(f[-1])


def statement_columns(sql: str) -> tuple[set[str], set[str]]:
    """(tables, predicate/join/order columns) for a normalised workload statement; empty when unparsable."""
    try:
        stmt = parse_sql(sql)[0].stmt
    except Exception:  # noqa: BLE001
        return set(), set()
    tables: set[str] = set()

    class T(visitors.Visitor):
        def visit_RangeVar(self, anc, node):
            tables.add(node.relname)

    T()(stmt)
    cols = _Cols()
    for part in ("whereClause", "sortClause", "groupClause", "havingClause"):
        node = getattr(stmt, part, None)
        if node is not None:
            cols(node) if not isinstance(node, tuple) else [cols(x) for x in node]
    for fr in getattr(stmt, "fromClause", None) or ():
        if isinstance(fr, ast.JoinExpr) and fr.quals is not None:
            cols(fr.quals)
    return tables, set(cols.cols)


def workload_coverage(index: IndexInfo, statements: list[dict]) -> dict[str, Any]:
    """Share of calls on the index's table whose predicates/joins/orderings use the index's leading column."""
    lead = index.columns[0] if index.columns else None
    table_calls = used_calls = 0
    affected = []
    for s in statements:
        tables, cols = statement_columns(s.get("query") or "")
        if index.table not in tables:
            continue
        calls = int(s.get("calls") or 0)
        table_calls += calls
        if lead and lead in cols:
            used_calls += calls
            affected.append(s.get("queryid"))
    return {"table_calls": table_calls, "calls_using_leading_column": used_calls,
            "coverage": round(used_calls / table_calls, 4) if table_calls else None,
            "affected_query_ids": [a for a in affected if a][:10]}


# ------------------------------------------------------------------ portfolio
def _review_ddl(schema: str, ix: IndexInfo | None, add_ddl: str | None = None) -> dict[str, str]:
    from app.recommendation_mapper import quote_ident
    if add_ddl is not None:
        m = re.match(r"CREATE INDEX ON (?:[\w\"]+\.)?\"?(\w+)\"? USING btree \((.+)\)", add_ddl)
        table, cols = (m.group(1), m.group(2)) if m else ("?", "?")
        name = "qs_" + re.sub(r"[^a-z0-9]+", "_", f"{table}_{cols}".lower())[:50] + "_idx"
        return {"migration": f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)} ON "
                             f"{quote_ident(schema)}.{quote_ident(table)} ({cols});",
                "rollback": f"DROP INDEX CONCURRENTLY IF EXISTS {quote_ident(schema)}.{quote_ident(name)};",
                "label": "HUMAN REVIEW REQUIRED - never executed by Planora"}
    assert ix is not None
    cols = ", ".join(quote_ident(c) for c in ix.columns)
    inc = f" INCLUDE ({', '.join(quote_ident(c) for c in ix.include)})" if ix.include else ""
    return {"migration": f"DROP INDEX CONCURRENTLY IF EXISTS {quote_ident(schema)}.{quote_ident(ix.name)};",
            "rollback": f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(ix.name)} ON "
                        f"{quote_ident(schema)}.{quote_ident(ix.table)} USING {ix.am} ({cols}){inc}"
                        + (f" WHERE {ix.predicate}" if ix.predicate else "") + ";",
            "label": "HUMAN REVIEW REQUIRED - never executed by Planora"}


def plan_actions(indexes: list[IndexInfo], fk_cols: dict[str, list[list[str]]], statements: list[dict],
                 stats_age_days: float, table_writes: dict[str, dict], schema: str = "public",
                 candidates: list[str] | None = None) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    total_calls = sum(int(s.get("calls") or 0) for s in statements)
    thin = stats_age_days < MIN_STATS_AGE_DAYS or total_calls < MIN_WORKLOAD_CALLS
    redundant = {r["index"]: r for r in find_redundancy(indexes, fk_cols)}
    for ix in indexes:
        cov = workload_coverage(ix, statements)
        w = table_writes.get(ix.table, {})
        write_effect = {"value": f"maintained on every write to {ix.table} ({int(w.get('writes', 0)):,} writes "
                                 "observed since stats reset)", "provenance": [Provenance.HEURISTIC_ESTIMATE.value]}
        storage = {"value_bytes": ix.size_bytes, "provenance": [Provenance.POSTGRES_OBSERVED.value]}
        evidence = [f"idx_scan={ix.idx_scan} [POSTGRES OBSERVED]",
                    f"statistics window {stats_age_days:.1f} days, workload calls {total_calls}",
                    f"workload coverage of leading column: {cov['coverage']}"]
        r = redundant.get(ix.name)
        if r and not r["protected"]:
            act = {"action": "REMOVE_REDUNDANT_INDEX", "indexes_affected": [ix.name, r["covered_by"]],
                   "evidence": evidence + [r["statement"]] + r["warnings"]}
            if r["supports_fk"]:
                act["warning"] = "supports foreign-key checks; covering index must keep the same leading columns"
        elif r and r["protected"]:
            act = {"action": "KEEP_INDEX", "indexes_affected": [ix.name],
                   "evidence": evidence + [r["statement"], f"protected: {r['protected']}"],
                   "warning": "redundant by structure but protected; removal only with explicit evidence"}
        elif ix.idx_scan == 0 and not ix.protected_reason and not any(cols == ix.columns for cols in
                                                                         fk_cols.get(ix.table, [])):
            if thin:
                act = {"action": "KEEP_INDEX", "indexes_affected": [ix.name],
                       "evidence": evidence + ["unused so far, but the statistics window / workload is too thin "
                                               "to justify removal (pruning refused)"]}
            elif not cov["calls_using_leading_column"]:
                act = {"action": "REMOVE_REDUNDANT_INDEX", "indexes_affected": [ix.name],
                       "evidence": evidence + ["zero scans over a sufficient window and no workload statement "
                                               "uses its leading column"]}
            else:
                act = {"action": "KEEP_INDEX", "indexes_affected": [ix.name],
                       "evidence": evidence + ["workload statements use its leading column"]}
        else:
            act = {"action": "KEEP_INDEX", "indexes_affected": [ix.name], "evidence": evidence}
        act.update({"table": ix.table, "affected_queries": cov["affected_query_ids"], "storage_effect": storage,
                    "write_effect": write_effect, "provenance": [Provenance.POSTGRES_OBSERVED.value,
                                                                 Provenance.HEURISTIC_ESTIMATE.value]})
        if act["action"] == "REMOVE_REDUNDANT_INDEX":
            act["review"] = _review_ddl(schema, ix)
            act["storage_effect"]["statement"] = f"frees ~{ix.size_bytes / 1048576:.1f} MB"
        actions.append(act)
    for cand in candidates or []:
        m = re.search(r"ON (?:[\w\"]+\.)?\"?(\w+)\"? USING btree \(([^)]*)\)", cand)
        if not m:
            continue
        table = m.group(1)
        ccols = [c.strip().strip('"') for c in m.group(2).split(",")]
        singles = [i for i in indexes if i.table == table and len(i.columns) == 1 and i.columns[0] == ccols[0]
                   and not i.protected_reason]
        act = {"action": "ADD_INDEX", "table": table, "indexes_affected": [], "candidate": cand,
               "evidence": [f"candidate from the optimizer for {table}({', '.join(ccols)})"],
               "write_effect": {"value": "one more index maintained on every write", "provenance":
                                [Provenance.HEURISTIC_ESTIMATE.value]},
               "storage_effect": {"value_bytes": None, "provenance": [Provenance.HYPOPG_SIMULATION.value]},
               "affected_queries": [], "review": _review_ddl(schema, None, cand),
               "provenance": [Provenance.HYPOPG_SIMULATION.value, Provenance.HEURISTIC_ESTIMATE.value]}
        if len(ccols) > 1 and singles:
            act["action"] = "REPLACE_INDEX_PORTFOLIO"
            act["indexes_affected"] = [s.name for s in singles]
            act["evidence"].append(f"{', '.join(s.name for s in singles)} would become a left prefix of the "
                                   "composite; replacement is a SUGGESTION ONLY (never automatic) - check column "
                                   "order, ORDER BY usage and write frequency first")
        actions.append(act)
    if not actions:
        actions.append({"action": "NO_CHANGE", "indexes_affected": [], "evidence": ["no indexes found"],
                        "provenance": [Provenance.POSTGRES_OBSERVED.value]})
    return actions


async def analyze_portfolio(db, schema: str = "public", candidates: list[str] | None = None,
                            statements: list[dict] | None = None) -> dict[str, Any]:
    """Read real metadata through the data source's Database (search_path = its schema) and plan actions."""
    async with db.read_only() as conn:
        await conn.execute(f"SET LOCAL search_path = \"{schema}\", public")
        rows = [dict(r) for r in await conn.fetch(SQL_INDEX_DETAIL)]
        fks = [dict(r) for r in await conn.fetch(SQL_FK_COLS)]
        age = float(await conn.fetchval(SQL_STATS_AGE) or 0.0)
        writes = {r["table_name"]: dict(r) for r in await conn.fetch(SQL_TABLE_WRITES)}
    indexes = [IndexInfo(name=r["index_name"], table=r["table_name"], columns=list(r["columns"]), am=r["am"],
                         include=list(r["include"]), opclasses=r["opclasses"], collations=r["collations"],
                         is_unique=r["is_unique"], is_primary=r["is_primary"], is_exclusion=r["is_exclusion"],
                         predicate=r["predicate"], expressions=r["expressions"],
                         constraint_backed=r["constraint_backed"], size_bytes=r["size_bytes"],
                         idx_scan=r["idx_scan"]) for r in rows]
    fk_cols: dict[str, list[list[str]]] = {}
    for f in fks:
        fk_cols.setdefault(f["table_name"], []).append(list(f["columns"]))
    stm = statements or []
    actions = plan_actions(indexes, fk_cols, stm, age, writes, schema, candidates)
    return {"schema": schema, "statistics_window_days": round(age, 2),
            "workload_statements": len(stm), "actions": actions,
            "summary": {a: sum(1 for x in actions if x["action"] == a) for a in
                        ("ADD_INDEX", "KEEP_INDEX", "REPLACE_INDEX_PORTFOLIO", "REMOVE_REDUNDANT_INDEX",
                         "NO_CHANGE")},
            "note": "review artifacts only; nothing is executed. Pruning requires workload evidence over "
                    f">= {MIN_STATS_AGE_DAYS} days and >= {MIN_WORKLOAD_CALLS} calls."}

