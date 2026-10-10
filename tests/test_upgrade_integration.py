"""P5 major tests: NL -> validated SQL -> EXISTING optimizer on demo and dynamic sources (+ search budget)."""
from __future__ import annotations

import os
import uuid

import pytest

from app.config import reset_settings_cache
from app.datasource import DataSourceResolver
from app.db import Database
from app.nlq import NLQService
from app.nlq_optimizer import optimize
from app.pipeline import Services
from tests.upgrade_helpers import admin_db_available, admin_exec

pytestmark = pytest.mark.skipif(not admin_db_available(), reason="live DB not reachable (NOT RUN)")


@pytest.fixture
async def env():
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    svc = Services()
    await svc.startup()
    r = DataSourceResolver(svc.db)
    yield svc, r
    await r.close()
    await svc.shutdown()


async def test_demo_nlq_flows_through_existing_optimizer(env):
    svc, r = env
    ctx = await r.resolve(None)
    assert svc.for_source(ctx) is svc                         # demo path untouched
    out = await NLQService().ask("total amount by tenant in 2025", ctx,
                                 optimizer=lambda sql, c, o: optimize(svc, c, sql, svc.db))
    opt = out["optimization"]
    assert opt["status"] == "completed"
    tr = opt["component_trace"]
    assert set(tr) == {"security_gateway", "privacy_layer", "gnn", "rl", "hypopg"}
    assert tr["gnn"]["status"] in ("used", "fallback") and tr["hypopg"]["status"] in ("used", "skipped")
    assert opt["workload_namespace"] == "demo" and opt["executed_sql_is_original"]
    assert out["execution_result"]["row_count"] > 0


async def test_dynamic_large_table_hypopg_and_namespace(env, tmp_path, monkeypatch):
    from app.dataset_ingestion import DatasetIngestionService
    from tests.test_upgrade_ingestion import _cleanup
    svc, r = env
    ds = "p5big" + uuid.uuid4().hex[:6]
    sess = "sess-" + uuid.uuid4().hex[:10]
    p = tmp_path / "events.csv"
    with p.open("w") as f:
        f.write("event_id,customer_code,amount,event_day\n")
        for i in range(60000):
            f.write(f"{i},C{i % 5000:05d},{(i % 97) + 0.5},2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}\n")
    try:
        ing = DatasetIngestionService()
        pv = await ing.preview(p, "events.csv", dataset_name=ds, session_id=sess)
        res = await ing.commit(pv["preview_id"], session_id=sess)
        ctx = await r.resolve(res["data_source_id"], sess)
        sql = f'SELECT sum(amount) FROM "{res["table"].split(".")[1]}" WHERE customer_code = \'C00042\''
        opt = await optimize(svc, ctx, sql, svc.db)
        assert opt["status"] == "completed", opt
        assert opt["component_trace"]["hypopg"]["status"] == "used"
        pct = opt["hypopg_comparison"]["planner_cost_reduction_pct"]
        assert pct["value"] > 50 and "HYPOPG SIMULATION" in pct["provenance"]
        assert any("customer_code" in d for d in opt["recommendations"]["indexes"])
        assert "qs_dynamic" in (opt["recommendations"]["migration_sql"] or "")
        assert opt["workload_namespace"] == ctx.id and opt["workload_persisted"]
        assert admin_exec("SELECT count(*) FROM optimizer_meta.query_history WHERE data_source_id = %s",
                          ctx.id)[0][0] >= 1
        assert all(rec.data_source_id == "demo" for rec in svc.store.all())   # demo view not polluted
        # tiny search budget -> SEARCH_BUDGET_EXHAUSTED emitted and recorded
        try:
            with monkeypatch.context() as scoped:
                scoped.setenv("BUDGET_MAX_HYPOPG_EXPERIMENTS", "0")
                scoped.setenv("MAX_HYPOPG_EXPERIMENTS", "0")
                reset_settings_cache()
                opt2 = await optimize(svc, ctx, sql, svc.db)
        finally:
            reset_settings_cache()
        assert "SEARCH_BUDGET_EXHAUSTED" in opt2.get("events", [])
        assert admin_exec("SELECT count(*) FROM qs_meta.audit_log WHERE data_source_id = %s AND "
                          "event_type = 'SEARCH_BUDGET_EXHAUSTED'", ctx.id)[0][0] >= 1
    finally:
        admin_exec("DELETE FROM optimizer_meta.query_history WHERE data_source_id LIKE 'dyn_%%'")
        _cleanup(ds)
