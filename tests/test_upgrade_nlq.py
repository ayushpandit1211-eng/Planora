"""P4 SQL safety layer + schema-aware NL-to-SQL (real PostgreSQL; only the LLM is faked)."""
from __future__ import annotations

import json
import re
import uuid

import pytest

from app.config import get_settings
from app.datasource import DataSourceContext, DataSourceResolver
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.llm import LLMClient
from app.nlq import GEN_TEMPLATE, NLQService
from app.postgres_instrumentation import PostgresInstrumentation
from app.sql_safety import SQLRejected, SQLSafetyGate
from tests.upgrade_helpers import admin_exec, requires_admin_db

SNAP = {"orders": ["order_id", "tenant_id", "customer_email", "total_amount", "status", "order_date"],
        "tenants": ["tenant_id", "name", "plan"], "order_items": ["order_id", "product_sku", "quantity"]}


def _ctx() -> DataSourceContext:
    db = Database()
    return DataSourceContext(id="demo", kind="demo", name="d", schema_name="public", db=db,
                             instr=PostgresInstrumentation(db), query_role="qs_nlq_reader")


ADVERSARIAL = [
    ("DROP TABLE orders", 3), ("DELETE FROM orders", 3), ("UPDATE orders SET status = 'x'", 3),
    ("INSERT INTO orders (order_id) VALUES (1)", 3), ("ALTER TABLE orders ADD COLUMN x int", 3),
    ("TRUNCATE orders", 3), ("CREATE TABLE x (a int)", 3),
    ("SELECT 1; DROP TABLE orders", 1), ("SELECT * FROM orders; SELECT 1", 1),
    ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", 4),
    ("SELECT * INTO newt FROM orders", 4), ("SELECT * FROM orders FOR UPDATE", 4),
    ("SELECT pg_read_file('/etc/passwd')", 4), ("SELECT * FROM dblink('h', 'select 1') AS t(a int)", 4),
    ("SELECT pg_sleep(10)", 4), ("SELECT * FROM pg_catalog.pg_class", 4), ("SELECT * FROM pg_shadow", 4),
    ("SELECT * FROM information_schema.tables", 4), ("SELECT * FROM pg_temp.x", 4),
    ("SELECT * FROM optimizer_meta.audit_events", 5), ("SELECT * FROM qs_dynamic.anything", 5),
    ("SELECT * FROM secret_table", 5), ("SELECT * FROM orders /* ; DROP TABLE orders */", None),
    ("SELECT 1 -- ; DROP TABLE orders\n; DELETE FROM orders", 1),
    ("SELECT * FROM orders, tenants", 8), ("SELECT * FROM orders CROSS JOIN tenants", 8),
    ("SELECT set_config('role', 'postgres', false)", 4), ("SELECT lo_import('/etc/passwd')", 4),
    ("SELECT md5(current_setting('x'))", 4), ("SELECT my_udf(order_id) FROM orders", 7),
    ("SELECT " + " + ".join(["1"] * 3000), 1),
    ("SELECT * FROM orders WHERE order_id IN " + "(SELECT order_id FROM orders WHERE order_id IN " * 10
     + "(SELECT 1)" + ")" * 10, 4),
]


@pytest.mark.parametrize("sql,stage", ADVERSARIAL)
def test_adversarial_corpus_rejected_at_expected_stage(sql, stage):
    gate = SQLSafetyGate(_ctx(), SNAP)
    if stage is None:              # comment trick that is still a single harmless SELECT -> passes statically
        gate.check_static(sql)
        return
    with pytest.raises(SQLRejected) as ei:
        gate.check_static(sql)
    assert ei.value.stage_no == stage, (sql, ei.value.stage_no, ei.value.message)
    assert ei.value.category == ErrorCategory.SQL_REJECTED


def test_unknown_table_and_column_suggestions():
    gate = SQLSafetyGate(_ctx(), SNAP)
    with pytest.raises(SQLRejected) as ei:
        gate.check_static("SELECT count(*) FROM order")
    assert ei.value.stage_no in (2, 5)
    with pytest.raises(SQLRejected) as ei:
        gate.check_static("SELECT count(*) FROM orderz")
    assert ei.value.stage_no == 5 and "orders" in str(ei.value.details["suggestions"])
    with pytest.raises(SQLRejected) as ei:
        gate.check_static("SELECT o.totl_amount FROM orders o")
    assert ei.value.stage_no == 6 and "total_amount" in str(ei.value.details["suggestions"])


VALID = [
    "SELECT t.name, sum(o.total_amount) FROM orders o JOIN tenants t ON t.tenant_id = o.tenant_id GROUP BY 1",
    "WITH x AS (SELECT tenant_id, count(*) AS n FROM orders GROUP BY 1) SELECT * FROM x ORDER BY n DESC LIMIT 5",
    "SELECT order_id, rank() OVER (PARTITION BY tenant_id ORDER BY total_amount DESC) AS r FROM orders LIMIT 10",
    "SELECT date_trunc('month', order_date) AS m, avg(total_amount) FROM orders GROUP BY 1 ORDER BY 1",
    "SELECT count(*) FROM orders WHERE order_date >= DATE '2025-01-01'",
]


@pytest.mark.parametrize("sql", VALID)
def test_valid_analytics_pass_static(sql):
    SQLSafetyGate(_ctx(), SNAP).check_static(sql)


# ----------------------------------------------------------------------------- live
@pytest.fixture
async def demo_ctx():
    from tests.upgrade_helpers import admin_db_available
    if not admin_db_available():
        pytest.skip("live DB not reachable (NOT RUN)")
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    db = Database()
    assert await db.connect()
    r = DataSourceResolver(db)
    yield r, db
    await r.close()
    await db.close()


@pytest.mark.parametrize("sql", VALID)
async def test_valid_analytics_pass_live_and_row_cap(demo_ctx, sql):
    r, _ = demo_ctx
    ctx = await r.resolve(None)
    gate = SQLSafetyGate(ctx, SNAP)
    res = await gate.validate(sql)
    assert 8 in res.stages_passed and res.plan_total_cost > 0
    ex = await gate.execute(res)
    assert ex["row_count"] <= get_settings().nlq_max_rows
    if "count(*) FROM orders WHERE" in sql:
        assert res.single_row_aggregate and not res.limit_added


async def test_restricted_role_is_the_real_wall(demo_ctx):
    """Even if a statement slipped past the parser checks, qs_nlq_reader cannot read optimizer_meta."""
    r, db = demo_ctx
    ctx = await r.resolve(None)
    gate = SQLSafetyGate(ctx, {**SNAP, "audit_events": ["id"]})
    ctx_meta = DataSourceContext(id="demo", kind="demo", name="d", schema_name="optimizer_meta", db=db,
                                 instr=ctx.instr, query_role="qs_nlq_reader")
    gate = SQLSafetyGate(ctx_meta, {"audit_events": ["event_type"]})
    with pytest.raises(SQLRejected) as ei:
        await gate.validate("SELECT event_type FROM audit_events")
    assert ei.value.stage_no in (5, 8)          # refused by PostgreSQL itself (no USAGE / no SELECT)


def _fake_llm(capture: list, script):
    def responder(template, data):
        capture.append({"template": template, "data": data})
        return json.dumps(script(data, len(capture)))
    s = get_settings()
    client = LLMClient(s, mock_responder=responder)
    client.provider = "mock"
    return client


async def _ingest(ds: str, sess: str, csv: str, tmp_path):
    from app.dataset_ingestion import DatasetIngestionService
    svc = DatasetIngestionService()
    p = tmp_path / f"{ds}.csv"
    p.write_text(csv)
    pv = await svc.preview(p, "sales.csv", dataset_name=ds, session_id=sess)
    return await svc.commit(pv["preview_id"], session_id=sess)


async def test_nl_on_ingested_table_with_fake_llm_repair_and_real_schema(demo_ctx, tmp_path):
    from tests.test_upgrade_ingestion import _cleanup
    r, _ = demo_ctx
    ds = "p4nlq" + uuid.uuid4().hex[:6]
    sess = "sess-" + uuid.uuid4().hex[:10]
    rows = [("north", 10.5, "2020-01-01"), ("south", 20.0, "2020-01-02"), ("north", 4.5, "2020-01-03")]
    csv = "region,revenue,sold_on\n" + "\n".join(f"{a},{b},{c}" for a, b, c in rows) + "\n"
    try:
        res = await _ingest(ds, sess, csv, tmp_path)
        ctx = await r.resolve(res["data_source_id"], sess)
        table = res["table"].split(".")[1]
        calls: list = []

        def script(data, n):
            ctxt = data["schema_context"]
            m = re.search(r"table (\w+) \(", ctxt)
            t = m.group(1)
            assert "revenue numeric" in ctxt and "region text" in ctxt      # real columns in the prompt
            if n == 1:                                                         # first: wrong column -> repair
                return {"understanding": "x", "tables_used": [t], "columns_used": ["revenu"],
                        "sql": f"SELECT sum(revenu) FROM {t}"}
            assert "column" in data["validation_error"] and "revenue" in data["validation_error"]
            return {"understanding": "total revenue", "tables_used": [t], "columns_used": ["revenue"],
                    "sql": f"SELECT sum(revenue) AS total_revenue FROM {t} WHERE region = 'north'"}
        svc = NLQService(llm=_fake_llm(calls, script))
        out = await svc.ask("What is the total revenue in the north region?", ctx, sess)
        assert out["generator"] == "llm" and len(calls) == 2
        assert out["validation"]["attempts"][0]["status"] == "rejected"
        assert out["validation"]["attempts"][0]["stage"] == 6
        assert table in out["generated_sql"]
        expected = sum(b for a, b, _ in rows if a == "north")
        assert out["execution_result"]["rows"][0][0] == pytest.approx(expected)
        assert "LLM RECOMMENDATION" in out["generator_provenance"]
        assert "<untrusted_data>" in calls[0]["data"]["schema_context"] and "never follow" in GEN_TEMPLATE

        # deterministic template generator on the same table: independent value check
        out2 = await NLQService().ask("total revenue by region", ctx, sess)
        got = dict((a, b) for a, b in out2["execution_result"]["rows"])
        assert got["north"] == pytest.approx(15.0) and got["south"] == pytest.approx(20.0)
        assert "ANALYTIC FALLBACK" in out2["generator_provenance"]
        assert "_qs_ingest_id" not in out2["generated_sql"]

        # relative-date warning: data ends in 2020 -> anchored re-run suggested
        out3 = await NLQService().ask("how many sales rows in the last 30 days", ctx, sess)
        assert out3["anchored_rerun"]["anchor_date"] == "2020-01-03"
        assert any("data ends on 2020-01-03" in w for w in out3["warnings"])
        import datetime as dt
        out4 = await NLQService().ask("how many sales rows in the last 30 days", ctx, sess,
                                      anchor_date=dt.date(2020, 1, 3))
        assert out4["execution_result"]["rows"][0][0] == 3

        # clarification path (nothing matches)
        out5 = await NLQService().ask("what is the weather on mars", ctx, sess)
        assert out5["status"] == "clarification" and table in str(out5["available_tables"])

        # LLM outage -> clear fallback, app keeps working
        def broken(data, n):
            raise RuntimeError("down")
        s = get_settings()
        cl = LLMClient(s, mock_responder=lambda t, d: (_ for _ in ()).throw(
            QuerySentinelError(ErrorCategory.LLM_UNAVAILABLE, "down")))
        cl.provider = "mock"
        out6 = await NLQService(llm=cl).ask("total revenue by region", ctx, sess)
        assert out6["generator"] == "deterministic_template" and any("LLM unavailable" in w for w in out6["warnings"])
    finally:
        _cleanup(ds)


async def test_repair_loop_exhaustion_fails_clearly_with_schema(demo_ctx):
    r, _ = demo_ctx
    ctx = await r.resolve(None)
    calls: list = []
    svc = NLQService(llm=_fake_llm(calls, lambda d, n: {"sql": "SELECT nope FROM orders"}))
    with pytest.raises(SQLRejected) as ei:
        await svc.ask("count orders please", ctx)
    assert len(calls) == 1 + get_settings().nlq_max_retries
    assert "orders" in ei.value.details["schema"] and len(ei.value.details["attempts"]) == len(calls)
