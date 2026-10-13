"""P13 major API tests: one end-to-end flow per data-source kind + error codes + SSE ordering."""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from tests.upgrade_helpers import admin_db_available, admin_exec

pytestmark = pytest.mark.skipif(not admin_db_available(), reason="live DB not reachable (NOT RUN)")


@pytest.fixture(scope="module")
def client():
    from app.api import create_app
    with TestClient(create_app()) as c:
        yield c


def H(sess):
    return {"X-Session-Id": sess}


def test_excel_options_preview_commit_and_query(client, tmp_path):
    from tests.test_structured_formats import fixture_file
    path = fixture_file(tmp_path, "xlsx")
    content = path.read_bytes()
    headers = H("sess-" + uuid.uuid4().hex[:12])
    params = {"filename": "sales & returns.xlsx"}
    assert client.post("/api/v1/data/options", params=params, content=content).status_code == 403
    response = client.post("/api/v1/data/options", params=params, content=content, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["tables"] == ["Sales & returns"]
    name = "sheet_" + uuid.uuid4().hex[:8]
    params.update(selection="Sales & returns", dataset_name=name)
    response = client.post("/api/v1/data/preview", params=params, content=content, headers=headers)
    assert response.status_code == 200, response.text
    preview = response.json()
    response = client.post("/api/v1/data/commit", json={"preview_id": preview["preview_id"]}, headers=headers)
    assert response.status_code == 200, response.text
    source = response.json()["data_source_id"]
    try:
        result = client.post("/api/v1/nlq/run", headers=headers, json={"data_source_id": source,
            "sql": f"SELECT name, amount FROM {name} ORDER BY id", "optimize": False})
        assert result.status_code == 200, result.text
        assert "Delhi" in result.text and "Pune" in result.text
    finally:
        assert client.delete("/api/v1/data/datasets/" + source, headers=headers).status_code == 200


def test_status_demo_flow_errors_and_sse(client):
    st = client.get("/api/v1/upgrade/status").json()
    assert st["raw_data_exposed_to_ai"] == 0 and "ingestion" in st["feature_flags"]
    assert client.get("/api/v1/health").status_code == 200                    # existing endpoint unchanged
    src = client.get("/api/v1/sources").json()["sources"]
    assert src[0]["id"] == "demo"
    r = client.post("/api/v1/nlq/ask", json={"question": "total amount by tenant"})
    out = r.json()
    assert r.status_code == 200 and out["status"] == "completed", out
    assert out["optimization"]["component_trace"]["security_gateway"]["status"] == "used"
    assert out["execution_result"]["provenance"] == ["POSTGRES OBSERVED"]
    # error codes
    assert client.get("/api/v1/sources/dyn_doesnotexist").json()["error"] == "source_not_found"
    bad = client.post("/api/v1/nlq/validate", json={"sql": "DELETE FROM orders"})
    assert bad.status_code == 422 and bad.json()["error"] == "sql_rejected" and bad.json()["details"]["stage"] == 3
    assert client.post("/api/v1/nlq/apply-index", json={"sql": "SELECT 1 FROM orders", "ddl":
                       "CREATE INDEX ON orders (status)", "data_source_id": "demo"}).json()["error"] == "source_forbidden"
    # async + SSE in order
    a = client.post("/api/v1/nlq/ask/async", json={"question": "how many orders", "optimize": False}).json()
    with client.stream("GET", a["stream_url"]) as s:
        evs = [json.loads(l[5:])["event"] for l in s.iter_lines() if l.startswith("data:")]
    assert evs[0] == "NLQ_STARTED" and evs[-1] == "NLQ_COMPLETE"
    assert evs.index("NLQ_SQL_VALIDATED") < evs.index("NLQ_EXECUTED")
    assert client.get(a["report_url"]).json()["status"] == "completed"


def test_dynamic_upload_flow(client):
    from tests.test_upgrade_ingestion import _cleanup
    sess = "sess-" + uuid.uuid4().hex[:10]
    ds = "p13dyn" + uuid.uuid4().hex[:5]
    body = "city,sales,day\n" + "\n".join(f"c{i % 50},{i % 13},2025-01-{i % 28 + 1:02d}" for i in range(20000))
    try:
        assert client.post("/api/v1/data/preview?filename=x.csv", content=b"a,b\n1,2\n").json()["error"] == \
            "source_forbidden"                                                   # session required
        pv = client.post(f"/api/v1/data/preview?filename=sales.csv&dataset_name={ds}", content=body.encode(),
                         headers=H(sess)).json()
        assert pv["plan"]["action"] == "create_table" and len(pv["columns"]) == 3
        cm = client.post("/api/v1/data/commit", json={"preview_id": pv["preview_id"]}, headers=H(sess)).json()
        sid = cm["data_source_id"]
        assert cm["rows_loaded"] == 20000
        assert client.get(f"/api/v1/sources/{sid}", headers=H("sess-someoneelse")).json()["error"] == \
            "source_forbidden"
        schema = client.get(f"/api/v1/sources/{sid}/schema", headers=H(sess)).json()
        assert [c["name"] for c in schema["tables"][0]["columns"]] == ["city", "sales", "day"]
        out = client.post("/api/v1/nlq/ask", json={"question": "total sales by city", "data_source_id": sid},
                          headers=H(sess)).json()
        assert out["status"] == "completed" and out["execution_result"]["row_count"] == 50
        tbl = cm["table"].split(".")[1]
        m = client.post("/api/v1/nlq/apply-index", headers=H(sess), json={
            "data_source_id": sid, "sql": f"SELECT sum(sales) FROM {tbl} WHERE city = 'c7'",
            "ddl": f"CREATE INDEX ON qs_dynamic.{tbl} USING btree (city)"}).json()
        assert m["provenance"][0] == "MEASURED SANDBOX" and m["runs"] == 5 and m["before_ms"]["median"] > 0
    finally:
        _cleanup(ds)


