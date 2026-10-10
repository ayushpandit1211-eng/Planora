"""EXPLAIN JSON parsing (Section 18/60) incl. nested plans, cost attribution and sanitisation."""
from app.plan_parser import parse_explain, plan_shape, sanitize_expression, sanitize_plan

from tests.conftest import PLANS


def test_explain_json_parsing():
    pa = parse_explain(PLANS["scenario_0"])
    assert pa.node_count >= 5 and pa.total_cost > 0
    root = pa.nodes[0]
    assert root.node_type == "Limit" and root.parent_id is None
    seq = [n for n in pa.nodes if n.node_type == "Seq Scan"]
    assert seq and seq[0].relation_name == "order_items" and seq[0].filter
    assert all(n.startup_cost <= n.total_cost for n in pa.nodes)
    assert pa.operator_histogram["Seq Scan"] >= 1


def test_nested_plan_and_subplan_attribution():
    pa = parse_explain(PLANS["scenario_2"])
    sub = [n for n in pa.nodes if n.parent_relationship == "SubPlan"]
    assert sub, "correlated subquery must appear as SubPlan"
    inner_scan = [n for n in pa.nodes if n.node_type == "Seq Scan"][0]
    assert inner_scan.estimated_loops > 1000            # executed once per outer row
    assert pa.highest_cost_node["node_type"] == "Seq Scan"
    assert len(pa.edges) == pa.node_count - 1


def test_plan_list_form_and_errors():
    assert parse_explain([PLANS["scenario_4"]]).nodes[0].node_type == "Index Scan"
    import pytest
    with pytest.raises(ValueError):
        parse_explain({"NoPlan": 1})


def test_fingerprint_stable_and_shape():
    a, b = parse_explain(PLANS["scenario_1"]), parse_explain(PLANS["scenario_1"])
    assert a.plan_fingerprint == b.plan_fingerprint
    assert parse_explain(PLANS["scenario_2"]).plan_fingerprint != parse_explain(PLANS["scenario_2_with_fk_index"]).plan_fingerprint
    assert plan_shape(a.nodes, a.edges).startswith(a.nodes[0].node_type)


def test_sanitize_expression():
    out = sanitize_expression("((orders.order_date >= '2026-06-01 00:00:00'::timestamp without time zone) "
                              "AND (orders.customer_email = 'x@y.com'::text) AND (total_amount > 5000))",
                              {"orders": "tbl_1", "order_date": "col_2", "customer_email": "col_3", "total_amount": "col_4"})
    assert "2026" not in out and "x@y.com" not in out and "5000" not in out and "without time zone" not in out
    assert "tbl_1.col_2" in out and "orders" not in out


def test_sanitize_plan_removes_names_and_literals(obfuscator):
    pa = parse_explain(PLANS["scenario_1"])
    obfuscator.mask_sql("SELECT 1 FROM orders o WHERE o.tenant_id = 'x' AND o.order_date >= '2026-01-01'", "r1")
    safe = sanitize_plan(pa, obfuscator, "r1")
    blob = safe.model_dump_json()
    assert safe.sanitized and "orders" not in blob and "idx_orders" not in blob and "2026-06" not in blob
    assert safe.plan_fingerprint != pa.plan_fingerprint   # fingerprint computed over pseudonyms
