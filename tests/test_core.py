"""Core infrastructure: config, errors, logging redaction, audit, budget (R9), firewall (R10),
schema inspector, workload store."""
import json
import logging

import pytest

from app.audit import AuditLogger
from app.capability_firewall import AGENT_CAPABILITIES, Capability, CapabilityFirewall
from app.config import Settings
from app.errors import CapabilityDenied, ErrorCategory, QuerySentinelError
from app.logging_setup import RedactingJsonFormatter, redact
from app.optimization_budget import OptimizationBudget
from app.schema_inspector import anonymize, build_schema_metadata, estimated_distinct, pseudonymize_metadata
from app.workload_store import WorkloadRecord, WorkloadStore, hour_bucket

from tests.conftest import CATALOG


def test_settings_redacted_view_hides_secrets():
    s = Settings(llm_api_key="sk-supersecretvalue1234567890", qs_app_password="pw")
    r = json.dumps(s.redacted())
    assert "supersecret" not in r and "pw" not in r.replace("pwd", "")
    assert s.redacted()["llm_api_key_configured"] is True


def test_structured_errors():
    e = QuerySentinelError(ErrorCategory.HYPOPG_UNAVAILABLE, "HypoPG extension unavailable", stage="sandbox")
    assert e.http_status == 503 and e.to_dict("r1")["error"] == "hypopg_unavailable"


def test_log_redaction():
    t = redact("postgresql://u:p@h/db password=abc 'literal' a@b.com bc8960a9-23b8-41e9-b924-56de3eb13b90")
    assert "p@h" not in t and "abc" not in t and "literal" not in t and "a@b.com" not in t and "bc8960a9" not in t
    rec = logging.LogRecord("x", logging.INFO, "", 0, "SELECT * FROM t WHERE e = 'x@y.z'", None, None)
    rec.request_id = "r1"
    out = json.loads(RedactingJsonFormatter().format(rec))
    assert out["request_id"] == "r1" and "x@y.z" not in out["msg"]


def test_audit_trail_sanitised():
    a = AuditLogger()
    a.record("r1", "MASKING", stage="privacy", details={"sql": "WHERE e = 'secret@corp.com'"})
    ev = a.events_for("r1")[0]
    assert "secret@corp.com" not in json.dumps(ev.details) and ev.request_id == "r1"


def test_budget_enforced_and_reported():
    hits = []
    b = OptimizationBudget(limits={"index_candidates": 2, "hypopg_experiments": 1, "agent_steps": 3,
                                   "llm_retries": 0, "workload_queries": 5},
                           on_exhausted=lambda k, s: hits.append(k))
    assert b.take("index_candidates", [1, 2, 3]) == [1, 2]
    assert b.consume("hypopg_experiments") and not b.consume("hypopg_experiments")
    snap = b.snapshot()
    assert set(hits) == {"index_candidates", "hypopg_experiments"} and "bounded" in snap["statement"]
    assert snap["remaining"]["index_candidates"] == 0


def test_capability_firewall():
    a = AuditLogger()
    fw = CapabilityFirewall(a)
    fw.require("r1", "index_strategist", Capability.PROPOSE_INDEX)
    with pytest.raises(CapabilityDenied):
        fw.require("r1", "sql_rewriter", Capability.RUN_HYPOPG)
    with pytest.raises(CapabilityDenied):
        fw.require("r1", "supervisor", Capability.PRODUCTION_WRITE)
    assert all(Capability.PRODUCTION_WRITE not in c for c in AGENT_CAPABILITIES.values())
    assert [e.event_type for e in a.events_for("r1")] == ["CAPABILITY_DENIED", "CAPABILITY_DENIED"]


def test_schema_inspector_build_and_anonymize(obfuscator):
    meta = build_schema_metadata(CATALOG["tables"], CATALOG["columns"], CATALOG["indexes"],
                                 CATALOG["foreign_keys"], CATALOG["primary_keys"])
    assert set(meta.tables) == {"orders", "order_items", "tenants"}
    fk = [f for f in meta.foreign_keys if f.table == "order_items"][0]
    assert fk.has_supporting_index is False
    nd = estimated_distinct(meta.column("orders", "status"), meta.tables["orders"].estimated_rows)
    assert nd and nd < 20
    anon = json.dumps(anonymize(meta, obfuscator, "r1"))
    for real in ("orders", "tenant_id", "idx_orders_status", "customer_email"):
        assert real not in anon
    ps = pseudonymize_metadata(meta, obfuscator, "r1")
    assert len(ps.tables) == 3 and all(t.startswith("tbl_") for t in ps.tables)


def test_workload_store_context():
    s = WorkloadStore()
    for h in range(3):
        s.add(WorkloadRecord(query_fingerprint="q1", plan_fingerprint="p1", timestamp_bucket=hour_bucket(f"2026-09-01T0{h}:10:00"),
                             call_count=10, estimated_cost=100.0 + h))
    ctx = s.context_for("q1")
    assert ctx["available"] and ctx["total_calls"] == 30 and ctx["frequency_weight"] > 1
    assert s.context_for("zz")["available"] is False


def test_metrics_snapshot_and_prometheus():
    from app.metrics import Metrics
    m = Metrics()
    m.inc("requests")
    for v in (10, 20, 30):
        m.observe("explain", v)
    snap = m.snapshot()
    assert snap["counters"]["requests"] == 1 and snap["stage_durations"]["explain"]["p50_ms"] == 20
    assert 'querysentinel_stage_duration_ms_count{stage="explain"} 3' in m.prometheus()
