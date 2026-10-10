"""Write-latency / partitioning simulation, zero-exposure proof and headline answer."""
import pytest

from app.agents.supervisor import Pipeline, _headline
from app.db import Database
from app.structural_sim import StructuralSimulator, qi

from tests.conftest import DEMO, requires_db


def test_identifier_quoting_rejects_injection():
    assert qi("orders") == '"orders"'
    with pytest.raises(ValueError):
        qi('orders"; DROP TABLE x; --')


def test_headline_contains_ps_elements():
    sb = {"planner_cost_reduction_pct": 72.8, "virtual_indexes_used": ["CREATE INDEX ON tbl_a (col_b, col_c)"],
          "estimated_storage_impact_mb": 12.0,
          "write_latency_simulation": [{"status": "completed", "added_ms_per_1000_rows": 1.5, "added_ms_per_row": 0.0015}]}
    part = {"status": "completed", "strategy": "range", "table": "tbl_a", "key": "col_c",
            "partitions_scanned": 1, "partitions_total": 36, "planner_cost_reduction_pct": 94.2}
    proof = {"zero_raw_data_exposed": True, "statement": "Zero raw data was exposed during this analysis."}
    out = _headline({"agent_recommendation": {}, "gnn_diagnosis": {}}, sb, part, proof)
    s = out["sentence"]
    assert "composite index on tbl_a(col_b, col_c)" in s and "72.8%" in s
    assert "+1.50 ms simulated write latency" in s and "1 of 36 partitions" in s
    assert s.endswith("Zero raw data was exposed during this analysis.")


async def test_simulator_failure_is_non_fatal():
    class Broken:
        def connection(self):
            raise RuntimeError("no db")
    res = await StructuralSimulator(Broken()).write_latency("orders", [{"columns": ["status"]}])
    assert res["status"] == "failed" and "unavailable" in res["reason"]


@pytest.mark.integration
@requires_db
async def test_live_write_latency_and_partitioning_leave_nothing_behind():
    db = Database()
    assert await db.connect()
    sim = StructuralSimulator(db)
    wl = await sim.write_latency("orders", [{"columns": ["tenant_id", "order_date"]}])
    pr = await sim.partitioning("orders", "order_date", "range")
    left = await db.fetchval("SELECT count(*) FROM pg_class WHERE relname LIKE 'qs_sim%'")
    idx = await db.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'orders'")
    await db.close()
    assert wl["status"] == "completed" and wl["ms_per_1000_rows_after"] > 0
    assert pr["status"] == "completed" and pr["partitions_scanned"] < pr["partitions_total"]
    assert left == 0 and idx == 3        # temp objects gone, no real index created


@pytest.mark.integration
@requires_db
async def test_live_report_has_answer_and_zero_exposure_proof():
    from app.pipeline import Services
    s = Services()
    await s.startup()
    try:
        res = await Pipeline(s).run("t-ans-1", DEMO[2])
    finally:
        await s.shutdown()
    assert res["status"] == "completed"
    assert res["zero_exposure_proof"]["zero_raw_data_exposed"] is True
    assert res["zero_exposure_proof"]["ai_payload_check"]["allowed"] is True
    assert "Zero raw data was exposed" in res["answer"]["sentence"]
    assert "ms simulated write latency" in res["answer"]["sentence"]
