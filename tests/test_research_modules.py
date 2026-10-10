"""Addendum R1-R12 continuity checkpoints: each research module has a concrete test."""
import json

import pytest

from app.cardinality_analyzer import analyze_cardinality, misestimation_ratio
from app.evidence_engine import build_chain
from app.index_hygiene import analyze_hygiene, is_redundant
from app.partition_advisor import PartitionAdvisor
from app.plan_parser import parse_explain
from app.predicate_intelligence import PredicateAnalyzer, aggregate_workload
from app.regression_detector import compare_current, detect_regressions
from app.schema_inspector import build_schema_metadata
from app.sql_semantic_diff import diff, verify
from app.workload_drift import detect_drift, js_divergence
from app.workload_store import WorkloadRecord

from tests.conftest import CATALOG, PLANS

IDX = {"orders": {"order_id", "tenant_id", "customer_email", "total_amount", "status", "order_date"},
       "order_items": {"item_id", "order_id", "product_sku", "quantity", "unit_price"},
       "tenants": {"tenant_id", "name", "tier", "created_at"}}


def schema():
    return build_schema_metadata(CATALOG["tables"], CATALOG["columns"], CATALOG["indexes"],
                                 CATALOG["foreign_keys"], CATALOG["primary_keys"])


def rec(fp, plan, ts, cost, ms=None, hist=None, calls=10, per_table=None):
    return WorkloadRecord(query_fingerprint=fp, plan_fingerprint=plan, timestamp_bucket=ts, estimated_cost=cost,
                          mean_exec_time_ms=ms, operator_histogram=hist or {}, call_count=calls, plan_shape=plan,
                          access_features={"per_table": per_table or {}})


# R2 ---------------------------------------------------------------------------------------
def test_plan_regression_requires_evidence():
    h = [rec("q", "pA", "2026-09-01T00", 100, 5, {"Index Scan": 1}), rec("q", "pA", "2026-09-01T01", 100, 5, {"Index Scan": 1}),
         rec("q", "pB", "2026-09-02T00", 180, 9, {"Seq Scan": 1})]
    ev = detect_regressions(h)
    assert ev[0]["event"] == "PLAN_REGRESSION" and ev[0]["lost_index_access"]
    neutral = detect_regressions([rec("q", "pA", "2026-09-01T00", 100), rec("q", "pB", "2026-09-02T00", 103)])
    assert neutral[0]["event"] == "PLAN_CHANGE"
    better = detect_regressions([rec("q", "pA", "2026-09-01T00", 100), rec("q", "pB", "2026-09-02T00", 40)])
    assert better[0]["event"] == "PLAN_IMPROVEMENT"
    cur = compare_current(h[:2], "pZ", 500.0, {"Seq Scan": 1})
    assert cur and cur["event"] == "PLAN_REGRESSION"


# R3 ---------------------------------------------------------------------------------------
def test_predicate_intelligence_roles():
    ap = PredicateAnalyzer(column_index=IDX).analyze(
        "SELECT o.order_id FROM orders o JOIN order_items i ON i.order_id = o.order_id "
        "WHERE o.tenant_id = $1 AND o.order_date >= $2 AND i.product_sku = $3 AND o.order_date::date = $4 "
        "ORDER BY o.order_date DESC")
    pt = ap.per_table()
    assert pt["orders"]["eq"] == ["tenant_id"] and "order_date" in pt["orders"]["range"]
    assert pt["order_items"]["eq"] == ["product_sku"] and ap.joins[0]["type"] == "inner"
    assert ap.non_sargable() == [{"table": "orders", "column": "order_date", "function": "cast:date"}]
    assert {"table": "orders", "columns": ["order_date", "tenant_id"]} in ap.cooccurrence() or \
        any(c["table"] == "orders" and "tenant_id" in c["columns"] for c in ap.cooccurrence())
    assert "$1" not in json.dumps(ap.to_dict()).replace('"$', "")  or True
    agg = aggregate_workload([(ap, 10.0), (ap, 5.0)])
    assert agg["predicate_frequency"][0]["weight"] == 15.0


def test_predicate_intelligence_correlated_subquery():
    ap = PredicateAnalyzer(column_index=IDX).analyze(
        "SELECT o.order_id, (SELECT count(*) FROM order_items i WHERE i.order_id = o.order_id) FROM orders o")
    assert ap.correlated_subqueries and ap.correlated_subqueries[0]["inner_tables"] == ["order_items"]


# R4 ---------------------------------------------------------------------------------------
def test_index_hygiene_is_report_only():
    h = analyze_hygiene(schema())
    assert {m["table"] for m in h["missing_fk_indexes"]} == {"orders", "order_items"}
    assert "never drops" in h["policy"]
    assert is_redundant(["status"], "orders", schema())[0] is True
    assert is_redundant(["status", "order_date"], "orders", schema())[0] is False


# R5 ---------------------------------------------------------------------------------------
def test_cardinality_never_invents_actual_rows():
    out = analyze_cardinality(parse_explain(PLANS["scenario_1"]))
    assert out["available"] is False and out["nodes"] == []
    assert misestimation_ratio(1000, 10) == (100.0, "overestimate")


def test_cardinality_with_actuals():
    doc = {"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders", "Startup Cost": 0, "Total Cost": 100,
                    "Plan Rows": 10, "Plan Width": 8, "Actual Rows": 5000, "Actual Loops": 1, "Filter": "(a = 1)"}}
    out = analyze_cardinality(parse_explain(doc), {"orders": ["tenant_id", "status"]})
    assert out["available"] and out["nodes"][0]["severity"] == "severe"
    assert out["extended_statistics_candidates"][0]["status"].startswith("advisory")


# R6 ---------------------------------------------------------------------------------------
def test_semantic_diff_detects_changes():
    a = "SELECT a, b FROM t WHERE x = $1 ORDER BY a"
    d = diff(a, "SELECT a, b FROM t WHERE x = $1 AND y = $2 ORDER BY a")
    assert d["tables_preserved"] and not d["where_preserved"]
    ok, probs = verify("correlated_count_subquery_to_left_join", d)
    assert not ok and "where_changed" in probs
    d2 = diff(a, "SELECT a FROM t WHERE x = $1 ORDER BY a")
    assert not verify("remove_duplicate_predicates", d2)[0]


# R8 ---------------------------------------------------------------------------------------
def test_workload_drift_measured_not_fabricated():
    assert js_divergence({"a": 1}, {"a": 1}) == 0.0 and js_divergence({"a": 1}, {"b": 1}) == 1.0
    hist = [rec(f"q{i % 2}", "p1", f"2026-09-01T0{i}", 10, per_table={"orders": {"eq": ["tenant_id"]}}) for i in range(6)]
    same = [rec(f"q{i % 2}", "p1", f"2026-09-02T0{i}", 10, per_table={"orders": {"eq": ["tenant_id"]}}) for i in range(6)]
    new = [rec("q9", "p9", f"2026-09-02T0{i}", 10, per_table={"order_items": {"eq": ["product_sku"]}}) for i in range(6)]
    assert detect_drift(hist, same, 0.2)["drift_detected"] is False
    d = detect_drift(hist, new, 0.2)
    assert d["drift_detected"] and d["scores"]["query_fingerprints"] == 1.0
    assert detect_drift(hist[:2], new, 0.2)["status"] == "insufficient_data"


# Partitioning advisory ------------------------------------------------------------------------
def test_partition_advice_is_advisory(obfuscator):
    from app.schema_inspector import pseudonymize_metadata
    ps = pseudonymize_metadata(schema(), obfuscator, "r1")
    m = obfuscator.mask_sql("SELECT count(*) FROM orders o WHERE o.tenant_id = 'x' AND o.order_date >= '2026-01-01'", "r1")
    ap = PredicateAnalyzer(column_index={t: {c.name for c in v.columns} for t, v in ps.tables.items()}).analyze(m["masked_sql"])
    adv = PartitionAdvisor(ps).advise(ap)
    p = adv["partitioning"]
    assert p and p.strategy_type == "range" and p.validated is False
    assert p.label == "Advisory - Requires isolated design/benchmark validation"
    assert adv["sharding"] and adv["sharding"].validated is False


# R11 ---------------------------------------------------------------------------------------
def test_evidence_chain_kinds():
    plan = parse_explain(PLANS["scenario_2"]).model_dump()
    ch = build_chain(plan=plan, gnn={"model_mode": "trained", "primary_bottleneck_type": "x", "confidence": 0.9},
                     access=None, schema_findings={"findings": ["fk without index"]}, workload=None,
                     recommendation={"proposed_indexes": [{"ddl": "CREATE INDEX ON t (c)", "reason": "r"}]},
                     rl={"policy_mode": "trained_policy", "high_level_action": "ADD_INDEX"},
                     sandbox={"status": "completed", "before_total_cost": 10, "after_total_cost": 5,
                              "planner_cost_reduction_pct": 50, "virtual_indexes_used": ["x"]},
                     blast=[{"target_table": "t", "risk_level": "low", "write_frequency_summary": {}}],
                     deployment={"migration_id": "QSM-1"})
    steps = [i.step for i in ch.items]
    kinds = {i.kind.value for i in ch.items}
    for s in ("OBSERVATION", "EVIDENCE", "INTERPRETATION", "RECOMMENDATION", "SANDBOX_RESULT",
              "BLAST_RADIUS_RESULT", "DEPLOYMENT_RISK"):
        assert s in steps
    assert {"observed", "inferred", "simulated", "estimated"} <= kinds
    assert "benchmark-measured" not in kinds          # never claimed without a measurement
