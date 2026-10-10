"""Structural-change simulator: write latency (ms) and partition pruning, on THROWAWAY copies.

HypoPG can only estimate read-side planner costs. To answer the problem statement's
"simulated increase in write latency" and to simulate partitioning, this module works on
session-private TEMPORARY tables inside ONE transaction that is ALWAYS rolled back:

* write latency: a temp copy of the target table (sample rows + the table's existing indexes,
  unique ones rebuilt as non-unique) is timed for INSERTs before and after the proposed
  index(es) are added. Reported as ms per 1,000 inserted rows and per single-row write.
* partitioning: a temp sample copy and a temp PARTITIONED copy (monthly range or hash) are
  ANALYZEd; a representative query (latest-month window / one key value) is EXPLAINed on both.
  Reported: partitions scanned vs total and planner cost on each copy.

Safety: the application role cannot write to business tables (enforced by PostgreSQL). Only
temp objects are created; the transaction is rolled back in a finally block, so nothing
persists and no other session can see the copies. Sample values never leave the database
server: only timings, counts and costs are returned.
"""
from __future__ import annotations

import re
import statistics
import time
from typing import Any

import asyncpg

from app.db import Database

SAMPLE_ROWS = 20_000          # rows copied into the temp table (write-latency test)
INSERT_BATCH = 4_000          # rows inserted per timed trial
TRIALS = 7
PART_SAMPLE_ROWS = 60_000
MAX_MONTH_PARTITIONS = 48
HASH_PARTITIONS = 8
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def qi(name: str) -> str:
    """Quote an identifier (names come from the validated schema / vault on the trusted side)."""
    if not _IDENT.match(name):
        raise ValueError("unexpected identifier")
    return '"' + name.replace('"', '""') + '"'


async def _sample_pct(conn: asyncpg.Connection, table: str, want: int) -> float:
    rows = await conn.fetchval("SELECT GREATEST(reltuples, 1)::float FROM pg_class WHERE oid = $1::regclass", table)
    return max(0.01, min(100.0, want / (rows or 1) * 100 * 1.5))


async def _fill_sample(conn: asyncpg.Connection, table: str, target: str, want: int) -> int:
    pct = await _sample_pct(conn, table, want)
    await conn.execute(f"INSERT INTO {target} SELECT * FROM {qi(table)} TABLESAMPLE SYSTEM ({pct:.4f}) LIMIT {want}")
    return await conn.fetchval(f"SELECT count(*) FROM {target}")


async def _existing_index_defs(conn: asyncpg.Connection, table: str) -> list[str]:
    rows = await conn.fetch("SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = $1", table)
    out = []
    for r in rows:
        m = re.search(r"\bUSING\s+(\w+)\s+(\(.*)$", r["indexdef"])
        if m:
            out.append(f"USING {m.group(1)} {m.group(2)}")
    return out


async def _time_inserts(conn: asyncpg.Connection) -> float:
    """Median ms to insert INSERT_BATCH rows into qs_sim_w (each trial rolled back to a savepoint)."""
    samples = []
    for _ in range(TRIALS):
        await conn.execute("SAVEPOINT qs_w")
        t0 = time.perf_counter()
        await conn.execute(f"INSERT INTO qs_sim_w SELECT * FROM qs_sim_src LIMIT {INSERT_BATCH}")
        samples.append((time.perf_counter() - t0) * 1000)
        await conn.execute("ROLLBACK TO SAVEPOINT qs_w")
    return statistics.median(samples)


class StructuralSimulator:
    def __init__(self, db: Database, timeout_ms: int = 60_000):
        self.db = db
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------ write latency
    async def write_latency(self, table: str, new_indexes: list[dict[str, Any]]) -> dict[str, Any]:
        """new_indexes: [{"columns": [...], "include": [...], "where": str|None}] (real names, validated)."""
        out: dict[str, Any] = {"status": "failed", "evidence_kind": "simulated",
                               "method": (f"temp copy of the table ({SAMPLE_ROWS:,} sampled rows + its existing indexes); "
                                          f"median of {TRIALS} timed INSERTs of {INSERT_BATCH:,} rows, baseline, with "
                                          "the proposed index(es), baseline again (A-B-A); transaction rolled back")}
        try:
            async with self.db.connection() as conn:
                tx = conn.transaction()
                await tx.start()
                try:
                    await conn.execute(f"SET LOCAL statement_timeout = {self.timeout_ms}")
                    await conn.execute(f"CREATE TEMP TABLE qs_sim_src (LIKE {qi(table)}) ON COMMIT DROP")
                    n = await _fill_sample(conn, table, "qs_sim_src", SAMPLE_ROWS)
                    if n < 100:
                        out.update(status="skipped", reason="table too small to sample")
                        return out
                    await conn.execute(f"CREATE TEMP TABLE qs_sim_w (LIKE {qi(table)}) ON COMMIT DROP")
                    await conn.execute("INSERT INTO qs_sim_w SELECT * FROM qs_sim_src")
                    existing = await _existing_index_defs(conn, table)
                    for d in existing:      # unique indexes rebuilt as non-unique: same maintenance work, no conflicts
                        await conn.execute(f"CREATE INDEX ON qs_sim_w {d}")
                    await conn.execute("ANALYZE qs_sim_w")
                    await _time_inserts(conn)                       # warm-up
                    before1 = await _time_inserts(conn)
                    for k, ix in enumerate(new_indexes):
                        cols = ", ".join(qi(c) for c in ix["columns"])
                        inc = f" INCLUDE ({', '.join(qi(c) for c in ix.get('include') or [])})" if ix.get("include") else ""
                        where = f" WHERE {ix['where']}" if ix.get("where") else ""
                        await conn.execute(f"CREATE INDEX qs_sim_new_{k} ON qs_sim_w ({cols}){inc}{where}")
                    await _time_inserts(conn)
                    after = await _time_inserts(conn)
                    for k in range(len(new_indexes)):               # baseline again (A-B-A) to cancel drift
                        await conn.execute(f"DROP INDEX qs_sim_new_{k}")
                    await _time_inserts(conn)
                    before2 = await _time_inserts(conn)
                    before = (before1 + before2) / 2
                finally:
                    await tx.rollback()
        except Exception as exc:  # noqa: BLE001 - simulation is best-effort, never breaks the pipeline
            out["reason"] = f"simulation unavailable ({getattr(exc, 'sqlstate', None) or type(exc).__name__})"
            return out
        noise_ms = abs(before1 - before2)
        within_noise = (after - before) <= noise_ms
        if after < before:          # an extra index cannot make inserts faster: timing noise
            after = before
        per_k_before = before / INSERT_BATCH * 1000
        per_k_after = after / INSERT_BATCH * 1000
        out.update(
            status="completed", sample_rows=n, existing_indexes=len(existing), new_indexes=len(new_indexes),
            ms_per_1000_rows_before=round(per_k_before, 2), ms_per_1000_rows_after=round(per_k_after, 2),
            added_ms_per_1000_rows=round(per_k_after - per_k_before, 2),
            added_ms_per_row=round((after - before) / INSERT_BATCH, 4),
            write_latency_change_pct=round((after - before) / before * 100, 1) if before > 0 else None,
            within_measurement_noise=within_noise,
            baseline_spread_ms_per_1000_rows=round(noise_ms / INSERT_BATCH * 1000, 2),
            caveat=("Measured on a small temp copy: B-tree depth and cache state differ from the full table, so "
                    "treat this as an order-of-magnitude estimate; validate on staging."))
        return out

    # ------------------------------------------------------------------ partitioning
    async def partitioning(self, table: str, key: str, strategy: str) -> dict[str, Any]:
        out: dict[str, Any] = {"status": "failed", "strategy": strategy, "evidence_kind": "simulated"}
        try:
            async with self.db.connection() as conn:
                tx = conn.transaction()
                await tx.start()
                try:
                    await conn.execute(f"SET LOCAL statement_timeout = {self.timeout_ms}")
                    await conn.execute(f"CREATE TEMP TABLE qs_sim_flat (LIKE {qi(table)}) ON COMMIT DROP")
                    n = await _fill_sample(conn, table, "qs_sim_flat", PART_SAMPLE_ROWS)
                    if n < 1000:
                        out.update(status="skipped", reason="table too small to sample")
                        return out
                    k = qi(key)
                    if strategy == "range":
                        lo, hi = await conn.fetchrow(f"SELECT date_trunc('month', min({k})), "
                                                     f"date_trunc('month', max({k})) + interval '1 month' FROM qs_sim_flat")
                        months = await conn.fetchval("SELECT (EXTRACT(year FROM age($2::timestamp, $1::timestamp)) * 12 "
                                                     "+ EXTRACT(month FROM age($2::timestamp, $1::timestamp)))::int", lo, hi)
                        if not months or months > MAX_MONTH_PARTITIONS:
                            lo = await conn.fetchval("SELECT $1::timestamp - make_interval(months => $2)", hi, MAX_MONTH_PARTITIONS)
                            months = MAX_MONTH_PARTITIONS
                            await conn.execute(f"DELETE FROM qs_sim_flat WHERE {k} < $1", lo)
                        await conn.execute(f"CREATE TEMP TABLE qs_sim_part (LIKE {qi(table)}) PARTITION BY RANGE ({k})")
                        for i in range(months):
                            await conn.execute(
                                f"CREATE TEMP TABLE qs_sim_part_{i} PARTITION OF qs_sim_part FOR VALUES FROM "
                                f"('{lo}'::timestamp + interval '{i} month') TO ('{lo}'::timestamp + interval '{i + 1} month')")
                        total_parts = months
                        win_lo = await conn.fetchval("SELECT $1::timestamp - interval '1 month'", hi)
                        pred = f"{k} >= '{win_lo}'::timestamp"
                        out["query_shape"] = "latest one-month window (typical weekly/monthly report)"
                    else:
                        await conn.execute(f"CREATE TEMP TABLE qs_sim_part (LIKE {qi(table)}) PARTITION BY HASH ({k})")
                        for i in range(HASH_PARTITIONS):
                            await conn.execute(f"CREATE TEMP TABLE qs_sim_part_{i} PARTITION OF qs_sim_part "
                                               f"FOR VALUES WITH (MODULUS {HASH_PARTITIONS}, REMAINDER {i})")
                        total_parts = HASH_PARTITIONS
                        val = await conn.fetchval(f"SELECT quote_literal({k}::text) FROM qs_sim_flat "
                                                  f"GROUP BY {k} ORDER BY count(*) DESC LIMIT 1")
                        typ = await conn.fetchval("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                                                  "WHERE attrelid = $1::regclass AND attname = $2", table, key)
                        pred = f"{k} = {val}::{typ}"
                        out["query_shape"] = "equality on the busiest key value (tenant-style lookup)"
                    await conn.execute("INSERT INTO qs_sim_part SELECT * FROM qs_sim_flat")
                    await conn.execute("ANALYZE qs_sim_flat")
                    await conn.execute("ANALYZE qs_sim_part")
                    q = "SELECT count(*) FROM {t} WHERE " + pred
                    flat = (await conn.fetchval("EXPLAIN (FORMAT JSON) " + q.format(t="qs_sim_flat")))
                    part = (await conn.fetchval("EXPLAIN (FORMAT JSON) " + q.format(t="qs_sim_part")))
                finally:
                    await tx.rollback()
        except Exception as exc:  # noqa: BLE001 - simulation is best-effort, never breaks the pipeline
            out["reason"] = f"simulation unavailable ({getattr(exc, 'sqlstate', None) or type(exc).__name__})"
            return out
        import json
        fp, pp = json.loads(flat)[0]["Plan"], json.loads(part)[0]["Plan"]
        scanned = set()

        def walk(node):
            rel = node.get("Relation Name")
            if rel and rel.startswith("qs_sim_part_"):
                scanned.add(rel)
            for c in node.get("Plans", []):
                walk(c)
        walk(pp)
        cf, cp = fp["Total Cost"], pp["Total Cost"]
        out.update(status="completed", sample_rows=n, partitions_total=total_parts, partitions_scanned=len(scanned),
                   partitions_pruned=total_parts - len(scanned), cost_unpartitioned=round(cf, 2),
                   cost_partitioned=round(cp, 2),
                   planner_cost_reduction_pct=round((cf - cp) / cf * 100, 2) if cf > 0 else None,
                   method=(f"temp sample copy ({n:,} rows) vs temp {strategy}-partitioned copy "
                           f"({total_parts} partitions); both ANALYZEd and EXPLAINed; transaction rolled back"),
                   caveat="Sample-based planner estimate; the full migration must still be benchmarked on staging.")
        return out
