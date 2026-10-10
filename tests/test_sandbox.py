"""HypoPG sandbox, blast radius and migration generator (Sections 31-34, 50-51, 60)."""
import pytest

from app.blast_radius import BlastRadiusAnalyzer, classify_risk, parse_write
from app.contracts import IndexCandidate, ValidationOutcome
from app.migration_generator import MigrationGenerator, deterministic_index_name
from app.sandbox import HypoPGSandbox, cost_reduction_pct
from app.schema_inspector import build_schema_metadata

from tests.conftest import CATALOG, DEMO, FakeConn, FakeDB


def outcome(ddl_real, masked="CREATE INDEX ON tbl_x (col_y)"):
    return ValidationOutcome(candidate=IndexCandidate(table="tbl_x", columns=["col_y"], ddl=masked), valid=True,
                             real_ddl=ddl_real)


def test_cost_reduction_formula_safe():
    assert cost_reduction_pct(200, 50) == 75.0
    assert cost_reduction_pct(0, 10) is None and cost_reduction_pct(None, 1) is None


async def test_hypothetical_index_creation_cost_comparison_cleanup(obfuscator):
    db = FakeDB()
    obfuscator.mask_sql(DEMO[2], "r1")
    res = await HypoPGSandbox(db, obfuscator).run(
        "r1", DEMO[2], [outcome("CREATE INDEX ON public.order_items USING btree (order_id)")])
    assert res.status == "completed"
    assert res.before_total_cost > res.after_total_cost
    assert res.planner_cost_reduction_pct > 99
    assert res.virtual_indexes_used == ["CREATE INDEX ON tbl_x (col_y)"]
    assert res.plan_changed and res.estimated_storage_impact_mb == 5.0
    assert res.cleanup_ok and db.conn.created == [] and db.conn.resets >= 2
    assert "not a runtime speedup" in res.metric_label


async def test_unhelpful_candidate_not_kept(obfuscator):
    db = FakeDB()
    res = await HypoPGSandbox(db, obfuscator).run(
        "r1", DEMO[2], [outcome("CREATE INDEX ON public.orders USING btree (total_amount)")])
    assert res.virtual_indexes_used == [] and res.planner_cost_reduction_pct == 0.0


async def test_cleanup_attempted_even_on_failure(obfuscator):
    db = FakeDB(FakeConn(fail_on_explain=True))
    res = await HypoPGSandbox(db, obfuscator).run("r1", DEMO[2], [outcome("CREATE INDEX ON public.order_items (order_id)")])
    assert res.status == "failed" and res.cleanup_ok and db.conn.resets >= 2


async def test_skipped_without_candidates(obfuscator):
    res = await HypoPGSandbox(FakeDB(), obfuscator).run("r1", DEMO[2], [])
    assert res.status == "skipped"


# ------------------------------------------------------------------ blast radius
def schema():
    return build_schema_metadata(CATALOG["tables"], CATALOG["columns"], CATALOG["indexes"],
                                 CATALOG["foreign_keys"], CATALOG["primary_keys"])


def test_write_query_discovery_and_table_matching():
    assert parse_write("INSERT INTO orders (order_id) VALUES ($1)")["target_table"] == "orders"
    u = parse_write("UPDATE orders SET status = $1 WHERE order_id = (SELECT order_id FROM orders LIMIT 1)")
    assert u["kind"] == "UPDATE" and u["set_columns"] == ["status"]
    assert parse_write("SELECT 1") is None


def test_blast_radius_ranking_and_risk(obfuscator):
    stmts = [{"query": "INSERT INTO order_items (item_id, order_id) VALUES ($1, $2)", "calls": 500, "rows": 500},
             {"query": "UPDATE orders SET status = $1 WHERE order_id = $2", "calls": 2000, "rows": 2000},
             {"query": "INSERT INTO orders (order_id) VALUES ($1)", "calls": 100, "rows": 100},
             {"query": "SELECT * FROM orders", "calls": 50, "rows": 1}]
    br = BlastRadiusAnalyzer(None, obfuscator)
    r = br.analyze_with(stmts, "r1", "orders", ["status", "order_date"], schema(), 30 * 1048576)
    assert [w["operation"] for w in r.write_queries_analyzed] == ["UPDATE", "INSERT"]
    assert r.write_queries_analyzed[0]["modifies_indexed_columns"] is True
    assert r.risk_level == "high"
    assert "ESTIMATED" in r.estimated_write_impact["label"]
    assert all("status" not in (w["anonymized_sql"] or "") for w in r.write_queries_analyzed)
    low = br.analyze_with(stmts, "r1", "tenants", ["tier"], schema(), 8192)
    assert low.risk_level == "low" and low.write_queries_analyzed == []


def test_risk_classification_bands():
    assert classify_risk(5000, 0.1, True, 0.1, 0) == "high"
    assert classify_risk(200, 0.1, False, 0.05, 0) == "medium"
    assert classify_risk(0, 0.0, False, 0.01, 10) == "low"


# ------------------------------------------------------------------ migration generator
def test_migration_and_rollback_generation():
    gen = MigrationGenerator()
    dep = gen.generate(request_id="r1", query_fingerprint="abc123",
                       validated_ddls=[{"real_ddl": "CREATE INDEX ON public.orders USING btree (tenant_id, order_date)",
                                        "rationale": "eq then range", "planner_cost_reduction_pct": 21.3}],
                       partition={"strategy_type": "range", "candidate_key": "col_x",
                                  "label": "Advisory - Requires isolated design/benchmark validation"})
    name = deterministic_index_name("orders", ["tenant_id", "order_date"])
    assert name == deterministic_index_name("orders", ["tenant_id", "order_date"]) and len(name) <= 63
    assert f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name}" in dep.migration_sql
    assert f"DROP INDEX CONCURRENTLY IF EXISTS public.{name};" in dep.rollback_sql
    assert "abc123" in dep.migration_sql and dep.migration_id in dep.migration_sql
    assert dep.requires_human_review and dep.approval_status == "PENDING_REVIEW"
    assert "NOT included" in dep.advisory_notes[0]
    assert "PARTITION" not in dep.migration_sql.upper().replace("PARTITION ", "")


def test_identifier_quoting_in_migration():
    dep = MigrationGenerator().generate(request_id="r", query_fingerprint="f",
                                        validated_ddls=[{"real_ddl": 'CREATE INDEX ON public."Order" USING btree ("select")',
                                                         "rationale": "q"}])
    assert 'public."Order"' in dep.migration_sql and '("select")' in dep.migration_sql
