"""P15 major test: benchmark lab produces only real measured values, nothing persists, repeatable."""
import uuid

import pytest

from app.db import Database
from app.sandbox_benchmark import run_benchmark
from app.sandbox_manager import SandboxManager
from tests.upgrade_helpers import admin_db_available, sandbox_pg_available

pytestmark = pytest.mark.skipif(not (admin_db_available() and sandbox_pg_available()),
                                reason="DB / sandbox-pg not reachable (NOT RUN)")


async def test_benchmark_on_generated_sandbox_table():
    from app.datasource import DataSourceResolver
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    db = Database()
    assert await db.connect()
    r = DataSourceResolver(db)
    m = SandboxManager(db, resolver=r)
    r.sandbox_db_factory = m.db_factory
    sess = "sess-" + uuid.uuid4().hex[:10]
    sb = await m.create(sess, "CREATE TABLE ev (id bigint primary key, code text, amount numeric(8,2));",
                        scale="medium")
    try:
        ctx = await r.resolve(sb["data_source_id"], sess)
        q = "SELECT sum(amount) FROM ev WHERE code = 'code_7'"
        a = await run_benchmark(ctx, q, ["CREATE INDEX ON ev (code)"], repetitions=5)
        b = await run_benchmark(ctx, q, ["CREATE INDEX ON ev (code)"], repetitions=5)
        for res in (a, b):
            v = res["variants"]
            assert set(v) == {"BASELINE", "INDEX"} and all(x["runs"] == 5 and x["median_ms"] > 0 for x in v.values())
            assert v["BASELINE"]["provenance"][0] == "MEASURED SANDBOX" and res["dataset_rows"]["ev"] > 0
        ratio = a["variants"]["BASELINE"]["median_ms"] / b["variants"]["BASELINE"]["median_ms"]
        assert 0.2 < ratio < 5                                   # reproducible within a loose tolerance
        async with ctx.db.connection() as c:                     # nothing persisted
            assert await c.fetchval("SELECT count(*) FROM pg_indexes WHERE tablename = 'ev'") == 1   # PK only
    finally:
        await m.delete(sb["data_source_id"], sess)
        await r.close()
        await db.close()
