"""FastAPI endpoints (Section 39/60): health, mask, analyze, optimize, ingestion, stream."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app

from tests.conftest import DEMO

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def client(fake_services):
    app = create_app(services=fake_services, connect=False)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["hypopg"] is True


def test_status_reports_model_modes(client):
    s = client.get("/api/v1/status").json()
    assert s["gnn"]["mode"] in ("trained", "analytic_fallback") and "llm" in s and s["langgraph"]["ok"]


def test_mask(client):
    r = client.post("/api/v1/mask", json={"sql": "SELECT * FROM orders WHERE customer_email = 'a@b.com'"})
    body = r.json()
    assert r.status_code == 200 and "a@b.com" not in json.dumps(body) and body["literal_count"] == 1


def test_analyze(client):
    r = client.post("/api/v1/analyze", json={"sql": DEMO[0]})
    b = r.json()
    assert r.status_code == 200 and b["gnn_analysis"]["primary_bottleneck_type"]
    assert b["plan_analysis"]["sanitized"] is True


def test_analyze_rejects_ddl(client):
    r = client.post("/api/v1/analyze", json={"sql": "DROP TABLE orders"})
    assert r.status_code == 400 and r.json()["error"] == "unsafe_sql"


def test_optimize_contract(client):
    r = client.post("/api/v1/optimize", json={"sql": DEMO[2]})
    b = r.json()
    assert r.status_code == 200
    for k in ("request_id", "privacy_audit", "schema_analysis", "plan_analysis", "gnn_analysis",
              "agent_recommendation", "rl_recommendation", "sandbox_simulation", "blast_radius", "deployment",
              "limitations"):
        assert k in b
    assert b["deployment"]["requires_human_review"] is True and b["deployment"]["migration_sql"]
    fr = b["final_report"]
    for k in ("query_fingerprint", "privacy_audit", "diagnosis", "optimization", "validation", "blast_radius",
              "deployment", "limitations", "evidence_chain"):
        assert k in fr
    assert fr["validation"]["measured_execution_time_reduction_pct"] is None


def test_optimize_rejects_multi_statement(client):
    r = client.post("/api/v1/optimize", json={"sql": "SELECT 1; DROP TABLE orders"})
    assert r.status_code == 400 and r.json()["status"] == "rejected"


def test_ingestion_json_and_csv(client):
    r = client.post("/api/v1/ingest/logs", json={"format": "json", "content": (EX / "sample_history.json").read_text()})
    assert r.status_code == 200 and r.json()["accepted"] > 0
    r = client.post("/api/v1/ingest/plans", json={"format": "csv", "content": (EX / "sample_history.csv").read_text()})
    assert r.status_code == 200 and r.json()["plans_parsed"] > 0
    w = client.get("/api/v1/workload").json()
    assert w["summary"]["records"] > 0 and w["drift"]["status"] in ("measured", "insufficient_data")


def test_stream_async_and_report(client):
    rid = client.post("/api/v1/optimize/async", json={"sql": DEMO[1]}).json()["request_id"]
    with client.stream("GET", f"/api/v1/stream/{rid}") as s:
        text = "".join(s.iter_text())
    events = [l.split(": ", 1)[1] for l in text.splitlines() if l.startswith("event: ")]
    assert events[0] == "REQUEST_RECEIVED" and events[-1] == "FINAL_REPORT_READY"
    data = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")]
    assert all(set(d) >= {"event", "timestamp", "request_id", "stage", "message"} for d in data)
    assert "bdd640fb" not in text      # no raw literals in SSE payloads
    rep = client.get(f"/api/v1/report/{rid}")
    assert rep.status_code == 200 and rep.json()["status"] == "completed"


def test_metrics_endpoint(client):
    client.post("/api/v1/optimize", json={"sql": DEMO[4]})
    m = client.get("/api/v1/metrics").json()
    assert m["counters"]["requests"] >= 1 and "explain" in m["stage_durations"]
    assert "querysentinel_events_total" in client.get("/api/v1/metrics/prometheus").text


def test_unknown_report_404(client):
    assert client.get("/api/v1/report/does-not-exist").status_code == 404


def test_api_key_enforced(fake_services, monkeypatch):
    from pydantic import SecretStr
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "qs_api_key", SecretStr("k1"))
    app = create_app(services=fake_services, connect=False)
    with TestClient(app) as c:
        assert c.post("/api/v1/mask", json={"sql": "SELECT 1"}).status_code == 401
        assert c.post("/api/v1/mask", json={"sql": "SELECT 1"}, headers={"x-api-key": "k1"}).status_code == 200
