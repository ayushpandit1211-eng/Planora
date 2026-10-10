"""Benchmark lab on BYOD sandboxes / uploaded datasets (Final Upgrade P15, optional).

Variants: BASELINE (current schema), INDEX (recommended index created inside a transaction that is rolled
back), FULL (index + validated rewrite when one is supplied). Every number comes from a run in this process:
median, min, max over `repetitions` warm runs, with dataset size and environment recorded and tagged
[MEASURED SANDBOX]. Never available on the demo database. Nothing persists: each variant runs in a transaction
that is always rolled back.
"""
from __future__ import annotations

import platform
import statistics
import time
from typing import Any

from app.datasource import DataSourceContext
from app.errors import ErrorCategory, QuerySentinelError


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def run_benchmark(ctx: DataSourceContext, sql: str, index_ddls: list[str], rewrite_sql: str | None = None,
                        repetitions: int = 5) -> dict[str, Any]:
    if ctx.kind != "sandbox":
        raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "the benchmark lab runs on BYOD sandboxes only",
                                 stage="benchmark")
    from pglast import ast, parse_sql
    for d in index_ddls:
        if not isinstance(parse_sql(d)[0].stmt, ast.IndexStmt):
            raise QuerySentinelError(ErrorCategory.INVALID_DDL, "only CREATE INDEX statements", stage="benchmark")

    async def timed(conn, q: str) -> list[float]:
        await conn.fetch(q)                         # warm-up (not counted)
        out = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            await conn.fetch(q)
            out.append((time.perf_counter() - t0) * 1000)
        return out

    variants: dict[str, list[float]] = {}
    async with ctx.db.connection() as conn:
        await conn.execute(f"SET search_path = {_q(ctx.schema_name)}")
        sizes = {r["relname"]: int(r["n"]) for r in await conn.fetch(
            "SELECT c.relname, c.reltuples::bigint AS n FROM pg_class c JOIN pg_namespace s ON s.oid = c.relnamespace "
            "WHERE s.nspname = $1 AND c.relkind = 'r'", ctx.schema_name)}
        for name, ddls, q in (("BASELINE", [], sql), ("INDEX", index_ddls, sql),
                              ("FULL", index_ddls, rewrite_sql or sql)):
            if name == "FULL" and not rewrite_sql:
                continue
            tx = conn.transaction()
            await tx.start()
            try:
                for d in ddls:
                    await conn.execute(d)
                if ddls:
                    await conn.execute("ANALYZE")
                variants[name] = await timed(conn, q)
            finally:
                await tx.rollback()               # nothing persists
    out = {}
    for k, v in variants.items():
        out[k] = {"median_ms": round(statistics.median(v), 3), "min_ms": round(min(v), 3),
                  "max_ms": round(max(v), 3), "runs": len(v), "provenance": ["MEASURED SANDBOX", "SYNTHETIC DATA"]
                  if ctx.synthetic else ["MEASURED SANDBOX"]}
    return {"variants": out, "dataset_rows": sizes, "repetitions": repetitions,
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            "server": "sandbox-pg (isolated)", "cache": "warm (1 discarded warm-up run)"},
            "note": "real timings from this run only; small synthetic datasets are illustrative, not production"}
