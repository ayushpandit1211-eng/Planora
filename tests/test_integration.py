"""Live integration tests against PostgreSQL 16 + HypoPG + pg_stat_statements (skipped when no
database is reachable). Run after `scripts/seed_data.py` with QS_APP_PASSWORD / POSTGRES_* set."""
import asyncpg
import pytest

from app.agents.supervisor import Pipeline
from app.contracts import IndexCandidate, ValidationOutcome
from app.db import Database
from app.pipeline import Services
from app.postgres_instrumentation import PostgresInstrumentation
from app.sandbox import HypoPGSandbox

from tests.conftest import DEMO, requires_db

pytestmark = [pytest.mark.integration, requires_db]


async def _db():
    db = Database()
    assert await db.connect()
    return db


async def test_extensions_operational():
    db = await _db()
    h = await PostgresInstrumentation(db).health()
    await db.close()
    assert h["postgresql"]["ok"] and h["hypopg"]["ok"]
    assert h["pg_stat_statements"]["ok"] and h["pg_stat_statements"]["preloaded"]


async def test_least_privilege_role_cannot_write_or_ddl():
    db = await _db()
    async with db.connection() as conn:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("INSERT INTO tenants VALUES (gen_random_uuid(), 'x', 'y', now())")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("CREATE INDEX ON orders (customer_email)")
    async with db.read_only() as conn:
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await conn.execute("INSERT INTO optimizer_meta.audit_events (request_id, event_type) VALUES ('x','y')")
    await db.close()


async def test_real_hypopg_sandbox_and_cleanup(obfuscator):
    db = await _db()
    obfuscator.mask_sql(DEMO[2], "int-1")
    o = ValidationOutcome(candidate=IndexCandidate(table="t", columns=["c"], ddl="CREATE INDEX ON tbl_x (col_y)"),
                          valid=True, real_ddl="CREATE INDEX ON public.order_items USING btree (order_id)")
    res = await HypoPGSandbox(db, obfuscator).run("int-1", DEMO[2], [o])
    async with db.read_only() as conn:
        left = await conn.fetchval("SELECT count(*) FROM hypopg_list_indexes")
    await db.close()
    assert res.status == "completed" and res.planner_cost_reduction_pct > 90 and res.cleanup_ok and left == 0


async def test_live_end_to_end_pipeline():
    s = Services()
    await s.startup()
    try:
        res = await Pipeline(s).run("int-e2e-1", DEMO[1])
    finally:
        await s.shutdown()
    assert res["status"] == "completed"
    assert res["sandbox_simulation"]["status"] == "completed"
    assert "tenant_id, order_date" in res["deployment"]["migration_sql"]
