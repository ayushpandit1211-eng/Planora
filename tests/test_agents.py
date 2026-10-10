"""LangGraph agents (Section 22-26/60): state transitions, routing, validation, mocked LLM."""
import json

import pytest

from app.agents.index_strategist import IndexStrategist
from app.agents.sql_rewriter import SQLRewriter
from app.agents.supervisor import Pipeline
from app.contracts import IndexCandidate
from app.llm import LLMClient
from app.recommendation_mapper import RecommendationValidator
from app.schema_inspector import build_schema_metadata, pseudonymize_metadata

from tests.conftest import CATALOG, DEMO


def real_schema():
    return build_schema_metadata(CATALOG["tables"], CATALOG["columns"], CATALOG["indexes"],
                                 CATALOG["foreign_keys"], CATALOG["primary_keys"])


async def test_full_graph_executes_state_transitions(fake_services):
    res = await Pipeline(fake_services).run("t-graph-1", DEMO[2])
    assert res["status"] == "completed"
    st = res["agent_status"]
    for node in ("supervisor", "sql_rewriter", "schema_expert", "index_strategist", "rl_optimizer",
                 "recommendation_validator", "hypopg_sandbox", "blast_radius", "final_report"):
        assert st[node] == "COMPLETED", (node, st)
    events = [e.event for e in fake_services.events.history("t-graph-1")]
    assert events[0] == "REQUEST_RECEIVED" and events[-1] == "FINAL_REPORT_READY"
    for e in ("SECURITY_SCAN_COMPLETE", "MASKING_COMPLETE", "EXPLAIN_COMPLETE", "GNN_ANALYSIS_COMPLETE",
              "SUPERVISOR_STARTED", "RL_ANALYSIS_COMPLETE", "SANDBOX_COMPLETE", "BLAST_RADIUS_COMPLETE"):
        assert e in events
    sb = res["sandbox_simulation"]
    assert sb["planner_cost_reduction_pct"] > 90 and sb["cleanup_ok"]
    assert "CREATE INDEX CONCURRENTLY" in res["deployment"]["migration_sql"]
    assert "DROP INDEX CONCURRENTLY" in res["deployment"]["rollback_sql"]
    assert res["deployment"]["requires_human_review"] is True
    assert fake_services.trusted == {} and fake_services.obfuscator.get_vault("t-graph-1") is None


async def test_routing_skips_rewriter_and_index_for_efficient_query(fake_services):
    res = await Pipeline(fake_services).run("t-graph-2", DEMO[4])
    assert res["status"] == "completed"
    assert res["agent_status"]["sql_rewriter"] == "SKIPPED"
    assert res["agent_status"]["index_strategist"] == "SKIPPED"
    assert res["agent_status"]["hypopg_sandbox"] == "SKIPPED"
    assert "-- no validated index change" in res["deployment"]["migration_sql"]


async def test_unsafe_sql_rejected_before_any_agent(fake_services):
    res = await Pipeline(fake_services).run("t-graph-3", "DROP TABLE orders")
    assert res["status"] == "rejected" and res["error"]["error"] == "unsafe_sql"
    assert res["agent_status"]["final_report"] == "REJECTED"
    ev = fake_services.audit.events_for("t-graph-3")
    assert any(e.event_type == "SECURITY_REJECTION" and e.severity == "security" for e in ev)


async def test_report_contains_no_raw_literals_outside_internal_fields(fake_services):
    res = await Pipeline(fake_services).run("t-graph-4", DEMO[3])
    rep = dict(res)
    dep = dict(rep.pop("deployment"))
    dep.pop("rewritten_sql_for_review", None)
    dep.pop("migration_sql", None)
    dep.pop("rollback_sql", None)
    rep["final_report"] = {k: v for k, v in rep["final_report"].items() if k != "deployment"}
    blob = json.dumps(rep) + json.dumps(dep)
    assert "customer1234@example.com" not in blob and "2026-08-15" not in blob


def test_index_strategist_uses_only_known_identifiers(obfuscator):
    from app.predicate_intelligence import PredicateAnalyzer
    rs = real_schema()
    m = obfuscator.mask_sql(DEMO[1], "r1")
    ps = pseudonymize_metadata(rs, obfuscator, "r1")
    ap = PredicateAnalyzer(column_index={t: {c.name for c in v.columns} for t, v in ps.tables.items()}).analyze(m["masked_sql"])
    cands = IndexStrategist(ps).propose(ap, {})
    assert cands
    comp = [c for c in cands if c.kind == "composite"][0]
    assert comp.roles[:2] == ["eq", "range"]                          # equality first, then range
    for c in cands:
        assert c.table in ps.tables and all(ps.has_column(c.table, col) for col in c.columns)
    real = RecommendationValidator(obfuscator, rs).validate(comp, "r1")
    assert real.valid and real.real_ddl == "CREATE INDEX ON public.orders USING btree (tenant_id, order_date)"


def test_validator_rejects_invalid_and_redundant(obfuscator):
    rs = real_schema()
    m = obfuscator.mask_sql("SELECT * FROM orders o WHERE o.status = 'x' AND o.order_date > '2026-01-01'", "r1")
    tbl = next(iter(m["mapped_tables"]))
    status_col = obfuscator.to_masked_identifier("r1", "status")
    v = RecommendationValidator(obfuscator, rs)
    red = v.validate(IndexCandidate(table=tbl, columns=[status_col], ddl=f"CREATE INDEX ON {tbl} ({status_col})"), "r1")
    assert not red.valid and any("redundant" in e or "duplicate" in e for e in red.errors)
    uniq = v.validate(IndexCandidate(table=tbl, columns=[status_col],
                                     ddl=f"CREATE UNIQUE INDEX ON {tbl} ({status_col}, {obfuscator.to_masked_identifier('r1', 'order_date')})"), "r1")
    assert not uniq.valid
    fake = v.validate(IndexCandidate(table=tbl, columns=["col_ffffff"], ddl=f"CREATE INDEX ON {tbl} (col_ffffff)"), "r1")
    assert not fake.valid


def test_sql_rewriter_correlated_subquery_semantics_checked(obfuscator):
    m = obfuscator.mask_sql(DEMO[2], "r1")
    p = SQLRewriter().apply(m["masked_sql"])
    assert p and p.rules_applied == ["correlated_count_subquery_to_left_join"]
    assert p.semantics_verified and "LEFT JOIN" in p.rewritten_masked_sql and "COALESCE" in p.rewritten_masked_sql
    real = obfuscator.translate_llm_sql_to_real(p.rewritten_masked_sql, "r1")
    assert "'pending'" in real and "order_items" in real


def test_llm_rewrite_with_invented_identifier_rejected(obfuscator):
    m = obfuscator.mask_sql(DEMO[2], "r1")
    bad = m["masked_sql"].replace("SELECT", "SELECT tbl_123456.col_abcdef,", 1)
    assert SQLRewriter().validate_llm_rewrite(m["masked_sql"], bad) is None


async def test_mocked_llm_payload_is_anonymised_and_validated(fake_services):
    seen = {}

    def responder(template, data):
        seen["payload"] = json.dumps(data)
        tbl = next(iter((data.get("schema") or {}).get("tables", {})), "tbl_x")
        return json.dumps({"rewritten_sql": None, "proposed_indexes": [{"ddl": f"CREATE INDEX ON {tbl} (col_zzzzzz)",
                                                                       "reason": "llm", "columns": ["col_zzzzzz"]}],
                           "technical_reasoning": "mock", "assumptions": [], "risks": [], "confidence": 0.5})

    fake_services.settings.llm_provider = "mock"
    fake_services.llm = LLMClient(fake_services.settings, guard=fake_services.guard, mock_responder=responder)
    res = await Pipeline(fake_services).run("t-llm-1", DEMO[1])
    fake_services.settings.llm_provider = "none"
    assert res["status"] == "completed"
    payload = seen["payload"]
    assert "bdd640fb" not in payload and "orders" not in payload and "tenant_id" not in payload
    assert any(c["source"] == "llm" for c in res["agent_recommendation"]["proposed_indexes"])
    rejected = res["final_report"]["validation"]["validator"]["rejected"]
    assert any("col_zzzzzz" in r["candidate"] for r in rejected)          # invented column rejected
    assert res["final_report"]["model_modes"]["llm"]["provider"] == "mock"


async def test_malformed_llm_json_falls_back(fake_services):
    fake_services.settings.llm_provider = "mock"
    fake_services.llm = LLMClient(fake_services.settings, guard=fake_services.guard,
                                  mock_responder=lambda t, d: "this is not json")
    res = await Pipeline(fake_services).run("t-llm-2", DEMO[1])
    fake_services.settings.llm_provider = "none"
    assert res["status"] == "completed"
    calls = res["agent_recommendation"]["llm"]["calls"]
    assert calls and all(not c["success"] and c["error_category"] == "llm_malformed_json" for c in calls)
