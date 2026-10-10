"""Blast-radius analysis (Section 33) and write-overhead estimation (Section 34).

pg_stat_statements -> candidate INSERT/UPDATE/DELETE statements -> pglast parsing -> written /
referenced tables -> match target table -> rank (calls) -> analyse up to the top three.

All outputs are ESTIMATES derived from transparent inputs (index column count, estimated index
size from HypoPG, write frequency, operation mix, table size). No write-latency measurement is
made or claimed. Statement text in the output is anonymised.
"""
from __future__ import annotations

from typing import Any

from pglast import ast, parse_sql, visitors

from app.contracts import BlastRadiusResult, SchemaMetadata
from app.errors import QuerySentinelError
from app.masking_vault import DataObfuscator
from app.postgres_instrumentation import PostgresInstrumentation

WRITE_TYPES = {"InsertStmt": "INSERT", "UpdateStmt": "UPDATE", "DeleteStmt": "DELETE", "MergeStmt": "MERGE"}
METHODOLOGY = (
    "Write statements are taken from pg_stat_statements (calls since the last stats reset), parsed with "
    "pglast to find the written table, matched to the index's table and ranked by calls (top 3). "
    "Per-statement impact: INSERT/MERGE add one index entry per inserted row; UPDATE adds index work only "
    "when it modifies an indexed column (it also stops being HOT-eligible); DELETE leaves dead index "
    "entries for VACUUM. Relative maintenance increase = 1 / (existing indexes on the table). Storage "
    "impact = HypoPG hypopg_relation_size estimate. Risk band combines write share, indexed-column "
    "updates and relative index size (size only counts for indexes >= 10 MB / 50 MB). All values are ESTIMATES, not measured latencies."
)


class _Refs(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.tables: set[str] = set()

    def visit_RangeVar(self, anc, node):
        self.tables.add(node.relname)


def parse_write(sql: str) -> dict[str, Any] | None:
    try:
        stmt = parse_sql(sql)[0].stmt
    except Exception:  # noqa: BLE001
        return None
    kind = WRITE_TYPES.get(type(stmt).__name__)
    if kind is None:
        return None
    target = stmt.relation.relname if getattr(stmt, "relation", None) is not None else None
    refs = _Refs()
    refs(stmt)
    set_cols: list[str] = []
    if isinstance(stmt, ast.UpdateStmt):
        set_cols = [rt.name for rt in stmt.targetList or () if rt.name]
    if isinstance(stmt, ast.InsertStmt):
        set_cols = [rt.name for rt in stmt.cols or () if rt.name]
    return {"kind": kind, "target_table": target, "referenced_tables": sorted(refs.tables - {target}),
            "set_columns": set_cols}


MATERIAL_INDEX_MB_HIGH = 50.0     # size ratio only matters once an index is materially large
MATERIAL_INDEX_MB_MEDIUM = 10.0


def classify_risk(write_calls: int, write_share: float, touches_indexed: bool, size_ratio: float | None,
                  table_writes: int, index_size_mb: float | None = None) -> str:
    big_hi = (index_size_mb is None or index_size_mb >= MATERIAL_INDEX_MB_HIGH) and (size_ratio or 0) > 0.5
    big_med = (index_size_mb is None or index_size_mb >= MATERIAL_INDEX_MB_MEDIUM) and (size_ratio or 0) > 0.15
    if (touches_indexed and write_calls > 1000) or big_hi or (write_share > 0.5 and write_calls > 100):
        return "high"
    if write_calls > 100 or touches_indexed or big_med or table_writes > 100_000:
        return "medium"
    return "low"


class BlastRadiusAnalyzer:
    def __init__(self, instrumentation: PostgresInstrumentation | None, obfuscator: DataObfuscator):
        self.instr = instrumentation
        self.obf = obfuscator

    async def fetch_statements(self) -> list[dict]:
        if self.instr is None:
            return []
        return await self.instr.statements(500)

    def analyze_with(self, statements: list[dict], request_id: str, target_table: str, index_columns: list[str],
                     schema: SchemaMetadata, index_size_bytes: float | None,
                     include_columns: list[str] | None = None) -> BlastRadiusResult:
        tmeta = schema.tables.get(target_table)
        existing = len(schema.indexes_for(target_table))
        matched, total_calls_all = [], 0
        for s in statements:
            info = parse_write(s.get("query") or "")
            calls = int(s.get("calls") or 0)
            total_calls_all += calls
            if info is None or info["target_table"] != target_table:
                continue
            matched.append({**info, "calls": calls, "rows": int(s.get("rows") or 0),
                            "mean_exec_time_ms": s.get("mean_exec_time"), "query": s.get("query")})
        matched.sort(key=lambda m: -m["calls"])
        top = matched[:3]
        write_calls = sum(m["calls"] for m in matched)
        idx_cols = set(index_columns) | set(include_columns or [])
        analysed = []
        touches_any = False
        for m in top:
            touches = m["kind"] == "UPDATE" and bool(idx_cols & set(m["set_columns"]))
            touches_any |= touches
            if m["kind"] in ("INSERT", "MERGE"):
                impact = "adds one entry to the new index per inserted row"
            elif m["kind"] == "UPDATE":
                impact = ("modifies an indexed column: extra index insert per updated row and no HOT update"
                          if touches else "does not modify indexed columns: HOT updates remain possible")
            else:
                impact = "leaves dead index entries until VACUUM"
            masked = self.obf.mask_sql(m["query"], request_id)
            analysed.append({"operation": m["kind"], "calls": m["calls"], "rows_affected_total": m["rows"],
                             "mean_exec_time_ms_observed": m["mean_exec_time_ms"],
                             "anonymized_sql": masked["masked_sql"] if masked["privacy_audit"]["safe_for_llm"] else None,
                             "modifies_indexed_columns": touches, "estimated_impact": impact})
        table_writes = 0
        if tmeta:
            table_writes = (tmeta.n_tup_ins or 0) + (tmeta.n_tup_upd or 0) + (tmeta.n_tup_del or 0)
        size_ratio = None
        if index_size_bytes and tmeta and tmeta.table_size_bytes:
            size_ratio = index_size_bytes / tmeta.table_size_bytes
        write_share = write_calls / total_calls_all if total_calls_all else 0.0
        risk = classify_risk(write_calls, write_share, touches_any, size_ratio, table_writes,
                             None if index_size_bytes is None else index_size_bytes / 1048576)
        hot_ratio = round((tmeta.n_tup_hot_upd or 0) / tmeta.n_tup_upd, 3) if tmeta and tmeta.n_tup_upd else None
        return BlastRadiusResult(
            target_table=target_table,
            write_queries_analyzed=analysed,
            write_frequency_summary={
                "write_statements_matched": len(matched), "write_calls_on_table": write_calls,
                "share_of_all_tracked_calls": round(write_share, 4),
                "operation_mix": {k: sum(m["calls"] for m in matched if m["kind"] == k) for k in ("INSERT", "UPDATE", "DELETE", "MERGE")},
                "table_rows_written_since_stats_reset": table_writes, "hot_update_ratio": hot_ratio,
                "source": "pg_stat_statements + pg_stat_user_tables (observed counters)",
            },
            estimated_write_impact={
                "index_columns": len(index_columns), "include_columns": len(include_columns or []),
                "existing_indexes_on_table": existing,
                "relative_index_maintenance_increase_pct": round(100.0 / max(existing, 1), 1),
                "index_size_to_table_ratio": None if size_ratio is None else round(size_ratio, 4),
                "updates_touch_indexed_columns": touches_any,
                "qualitative_band": risk,
                "label": "ESTIMATED - heuristic, not a measured write latency",
            },
            estimated_storage_impact_mb=None if index_size_bytes is None else round(index_size_bytes / 1048576, 2),
            risk_level=risk, methodology=METHODOLOGY)

    async def analyze(self, request_id: str, target_table: str, index_columns: list[str], schema: SchemaMetadata,
                      index_size_bytes: float | None, include_columns: list[str] | None = None) -> BlastRadiusResult:
        try:
            statements = await self.fetch_statements()
            note = None
        except QuerySentinelError as exc:
            statements, note = [], exc.message
        res = self.analyze_with(statements, request_id, target_table, index_columns, schema, index_size_bytes,
                                include_columns)
        if note:
            res.methodology += f" NOTE: {note}; write workload unknown, risk band from table counters only."
        return res
