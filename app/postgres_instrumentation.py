"""PostgreSQL instrumentation layer (Section 35).

This prototype does NOT implement a native C-level planner hook. Its instrumentation is
extension/API based: pg_stat_statements, HypoPG, EXPLAIN (FORMAT JSON) and the system
catalogs / planner statistics. All catalog queries below read metadata only. In pg_stats
only n_distinct, null_frac, correlation and the top most_common_FREQ (a number) are read -
most_common_vals and histogram_bounds
contain raw data values and are deliberately never selected.
"""
from __future__ import annotations

import re

from typing import Any

from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError

SQL_VERSION = "SELECT current_setting('server_version') AS version, current_setting('server_version_num')::int AS num"
SQL_EXTENSIONS = "SELECT extname, extversion FROM pg_extension"

SQL_TABLES = """
SELECT n.nspname AS schema_name, c.relname AS table_name, c.reltuples::float8 AS estimated_rows,
       pg_relation_size(c.oid) AS table_size_bytes, pg_total_relation_size(c.oid) AS total_size_bytes,
       s.n_tup_ins, s.n_tup_upd, s.n_tup_del, s.n_tup_hot_upd, s.seq_scan, s.idx_scan,
       to_char(greatest(s.last_analyze, s.last_autoanalyze), 'YYYY-MM-DD"T"HH24:MI:SS') AS last_analyze
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.relkind IN ('r', 'p') AND n.nspname = 'public'
  AND ($1::text[] IS NULL OR c.relname = ANY($1::text[]))
ORDER BY c.relname
"""

SQL_COLUMNS = """
SELECT c.relname AS table_name, a.attname AS column_name, format_type(a.atttypid, a.atttypmod) AS data_type,
       NOT a.attnotnull AS nullable, st.n_distinct::float8 AS n_distinct, st.null_frac::float8 AS null_frac,
       st.correlation::float8 AS correlation,
       (st.most_common_freqs)[1]::float8 AS top_value_freq, a.attnum
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stats st ON st.schemaname = n.nspname AND st.tablename = c.relname AND st.attname = a.attname
WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relkind IN ('r','p') AND n.nspname = 'public'
  AND ($1::text[] IS NULL OR c.relname = ANY($1::text[]))
ORDER BY c.relname, a.attnum
"""

SQL_INDEXES = """
SELECT ic.relname AS index_name, tc.relname AS table_name, ix.indisunique AS is_unique,
       ix.indisprimary AS is_primary, (ix.indpred IS NOT NULL) AS is_partial,
       am.amname AS am, pg_relation_size(ic.oid) AS size_bytes, si.idx_scan,
       ARRAY(SELECT a.attname FROM unnest(ix.indkey[0:ix.indnkeyatts-1]) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = tc.oid AND a.attnum = k.attnum ORDER BY k.ord) AS key_columns,
       ARRAY(SELECT a.attname FROM unnest(ix.indkey[ix.indnkeyatts:]) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = tc.oid AND a.attnum = k.attnum ORDER BY k.ord) AS include_columns,
       (0 = ANY(ix.indkey)) AS has_expression
FROM pg_index ix
JOIN pg_class ic ON ic.oid = ix.indexrelid
JOIN pg_class tc ON tc.oid = ix.indrelid
JOIN pg_namespace n ON n.oid = tc.relnamespace
JOIN pg_am am ON am.oid = ic.relam
LEFT JOIN pg_stat_user_indexes si ON si.indexrelid = ix.indexrelid
WHERE n.nspname = 'public' AND ($1::text[] IS NULL OR tc.relname = ANY($1::text[]))
ORDER BY tc.relname, ic.relname
"""

SQL_FKS = """
SELECT tc.relname AS table_name, rc.relname AS ref_table,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum ORDER BY k.ord) AS columns,
       ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum ORDER BY k.ord) AS ref_columns
FROM pg_constraint con
JOIN pg_class tc ON tc.oid = con.conrelid
JOIN pg_class rc ON rc.oid = con.confrelid
JOIN pg_namespace n ON n.oid = tc.relnamespace
WHERE con.contype = 'f' AND n.nspname = 'public'
  AND ($1::text[] IS NULL OR tc.relname = ANY($1::text[]) OR rc.relname = ANY($1::text[]))
"""

SQL_PK = """
SELECT tc.relname AS table_name,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum ORDER BY k.ord) AS columns
FROM pg_constraint con JOIN pg_class tc ON tc.oid = con.conrelid
JOIN pg_namespace n ON n.oid = tc.relnamespace
WHERE con.contype = 'p' AND n.nspname = 'public'
"""

SQL_STATEMENTS = """
SELECT s.queryid::text AS queryid, s.query, s.calls, s.total_exec_time, s.mean_exec_time, s.rows,
       s.shared_blks_hit, s.shared_blks_read
FROM pg_stat_statements s
JOIN pg_database d ON d.oid = s.dbid
WHERE d.datname = current_database() AND s.toplevel
  AND s.query NOT LIKE '%qs_sim\_%'      -- Planora's own temp-copy simulations are not workload
ORDER BY s.total_exec_time DESC
LIMIT $1
"""


_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class PostgresInstrumentation:
    def __init__(self, db: Database, schema: str = "public"):
        """`schema` (Final Upgrade P1, default 'public') selects the catalog namespace that tables/columns/
        indexes/foreign keys are read from. It must be a plain lower-case identifier."""
        if not _IDENT.match(schema):
            raise ValueError("invalid schema name")
        self.db = db
        self.schema = schema

    def _q(self, sql: str) -> str:
        if self.schema == "public":
            return sql
        out = sql.replace("n.nspname = 'public'", f"n.nspname = '{self.schema}'")
        # hidden lineage columns of ingested datasets (_qs_*) never appear in schema context
        return out.replace("WHERE a.attnum > 0 AND NOT a.attisdropped",
                           "WHERE a.attnum > 0 AND NOT a.attisdropped AND a.attname NOT LIKE '\\_qs\\_%'")

    async def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {"postgresql": {"ok": False}, "hypopg": {"ok": False},
                               "pg_stat_statements": {"ok": False}}
        if not self.db.available:
            out["postgresql"]["error"] = self.db.last_error or "pool not initialised"
            return out
        try:
            row = (await self.db.fetch(SQL_VERSION))[0]
            out["postgresql"] = {"ok": True, "version": row["version"], "version_num": row["num"]}
            exts = {r["extname"]: r["extversion"] for r in await self.db.fetch(SQL_EXTENSIONS)}
        except QuerySentinelError as exc:
            out["postgresql"] = {"ok": False, "error": exc.category.value}
            return out
        except Exception as exc:  # noqa: BLE001 - health must not raise
            out["postgresql"] = {"ok": False, "error": type(exc).__name__}
            return out
        # HypoPG: installed AND callable by the restricted role
        if "hypopg" in exts:
            try:
                await self.db.fetchval("SELECT count(*) FROM hypopg_list_indexes")
                out["hypopg"] = {"ok": True, "version": exts["hypopg"]}
            except Exception as exc:  # noqa: BLE001
                out["hypopg"] = {"ok": False, "version": exts["hypopg"], "error": type(exc).__name__}
        else:
            out["hypopg"] = {"ok": False, "error": "extension not installed"}
        # pg_stat_statements: extension created AND view actually queryable. Reading the view
        # raises "must be loaded via shared_preload_libraries" when the library is not
        # preloaded, so a successful read is a functional proof of correct preload config
        # (shared_preload_libraries itself is not readable by the least-privilege role).
        if "pg_stat_statements" in exts:
            try:
                n = await self.db.fetchval("SELECT count(*) FROM pg_stat_statements")
                out["pg_stat_statements"] = {"ok": True, "version": exts["pg_stat_statements"],
                                             "preloaded": True, "tracked_statements": n}
            except Exception as exc:  # noqa: BLE001
                out["pg_stat_statements"] = {"ok": False, "preloaded": False, "error": type(exc).__name__}
        else:
            out["pg_stat_statements"] = {"ok": False, "error": "extension not installed"}
        return out

    async def tables(self, names: list[str] | None = None) -> list[dict]:
        return [dict(r) for r in await self.db.fetch(self._q(SQL_TABLES), names)]

    async def columns(self, names: list[str] | None = None) -> list[dict]:
        return [dict(r) for r in await self.db.fetch(self._q(SQL_COLUMNS), names)]

    async def indexes(self, names: list[str] | None = None) -> list[dict]:
        return [dict(r) for r in await self.db.fetch(self._q(SQL_INDEXES), names)]

    async def foreign_keys(self, names: list[str] | None = None) -> list[dict]:
        return [dict(r) for r in await self.db.fetch(self._q(SQL_FKS), names)]

    async def primary_keys(self) -> dict[str, list[str]]:
        return {r["table_name"]: list(r["columns"]) for r in await self.db.fetch(self._q(SQL_PK))}

    async def statements(self, limit: int = 200) -> list[dict]:
        """pg_stat_statements rows (normalised query text: literals already replaced by $n)."""
        try:
            return [dict(r) for r in await self.db.fetch(SQL_STATEMENTS, limit)]
        except QuerySentinelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise QuerySentinelError(ErrorCategory.PG_STAT_UNAVAILABLE,
                                     f"pg_stat_statements unavailable ({type(exc).__name__})",
                                     stage="instrumentation") from None

    async def explain(self, validated_sql: str, generic: bool = False) -> dict:
        return await self.db.explain(validated_sql, generic=generic)
