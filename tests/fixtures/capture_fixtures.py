#!/usr/bin/env python3
"""Re-captures the offline test fixtures from a seeded demo database (run from repo root):

    python tests/fixtures/capture_fixtures.py

Writes demo_catalog.json (schema metadata rows exactly as returned by the instrumentation
queries) and demo_plans.json (EXPLAIN JSON of the demonstration scenarios + a plan of scenario C
with a hypothetical order_items(order_id) index). No row data is captured.
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.benchmark import load_demo_queries  # noqa: E402
from app.db import Database, explain_on  # noqa: E402
from app.postgres_instrumentation import PostgresInstrumentation  # noqa: E402


async def main():
    db = Database()
    assert await db.connect(), "database unavailable"
    ins = PostgresInstrumentation(db)
    cat = {"tables": await ins.tables(None), "columns": await ins.columns(None), "indexes": await ins.indexes(None),
           "foreign_keys": await ins.foreign_keys(None), "primary_keys": await ins.primary_keys()}
    stmts = [s for s in await ins.statements(300)
             if s["query"].lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))]
    cat["write_statements"] = [{k: s[k] for k in ("query", "calls", "rows", "mean_exec_time", "total_exec_time")} for s in stmts]
    plans = {}
    for i, q in enumerate(load_demo_queries()):
        plans[f"scenario_{i}"] = await db.explain(q)
    async with db.read_only() as conn:
        await conn.fetchrow("SELECT * FROM hypopg_create_index('CREATE INDEX ON public.order_items USING btree (order_id)')")
        plans["scenario_2_with_fk_index"] = await explain_on(conn, load_demo_queries()[2])
        await conn.execute("SELECT hypopg_reset()")
    out = Path(__file__).parent
    (out / "demo_catalog.json").write_text(json.dumps(cat, indent=1, default=str))
    (out / "demo_plans.json").write_text(json.dumps(plans, indent=1, default=str))
    await db.close()
    print("fixtures written")


asyncio.run(main())
