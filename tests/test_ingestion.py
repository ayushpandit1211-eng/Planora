"""Ingestion engine (Section 16/60): JSON + CSV, privacy sanitisation, fingerprints."""
import json
from pathlib import Path

import pytest

from app.ingestion import IngestionEngine, PgStatStatementsCollector
from app.llm_guard import OutgoingPayloadGuard
from app.workload_store import WorkloadStore

from tests.conftest import FakeInstrumentation

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def engine(obfuscator):
    return IngestionEngine(obfuscator, WorkloadStore(), OutgoingPayloadGuard(obfuscator))


def test_json_ingestion(engine):
    rep = engine.ingest_json((EX / "sample_history.json").read_text())
    assert rep.accepted == len(json.loads((EX / "sample_history.json").read_text())["records"])
    assert rep.plans_parsed == rep.accepted and rep.quarantined == 0
    assert len(engine.store.all()) == rep.accepted


def test_csv_ingestion(engine):
    rep = engine.ingest_csv((EX / "sample_history.csv").read_text())
    assert rep.accepted > 0 and rep.plans_parsed == rep.accepted


def test_privacy_sanitisation(engine):
    rep = engine.ingest([{"query": "SELECT * FROM orders WHERE customer_email = 'leak@example.com' AND total_amount > 777",
                          "calls": 3, "mean_time_ms": 1.5, "timestamp": "2026-09-01T10:15:00"}], "test")
    assert rep.accepted == 1 and rep.pii_records == 1
    blob = json.dumps(rep.records[0].model_dump())
    assert "leak@example.com" not in blob and "777" not in blob and "orders" not in blob
    assert rep.records[0].timestamp_bucket.startswith("2026-09-01T10:00:00")


def test_fingerprint_ignores_literals(engine):
    rep = engine.ingest([{"query": "SELECT * FROM orders WHERE status = 'paid'"},
                         {"query": "SELECT * FROM orders WHERE status = 'pending'"},
                         {"query": "SELECT * FROM orders WHERE tenant_id = 'x'"}], "test")
    fps = [r.query_fingerprint for r in rep.records]
    assert fps[0] == fps[1] != fps[2]


def test_no_raw_rows_and_bad_records(engine):
    rep = engine.ingest([{"calls": 1}, {"query": "SELECT 1", "rows_data": [["secret"]]}], "test")
    assert rep.rejected == 1
    assert "secret" not in json.dumps([r.model_dump() for r in rep.records])


def test_vaults_cleared_after_batch(engine):
    engine.ingest([{"query": "SELECT * FROM orders WHERE status = 'paid'"}], "test")
    assert len(engine.obf.backend) == 0


async def test_pg_stat_statements_collector(engine):
    rows = [{"query": "SELECT * FROM orders WHERE status = $1", "calls": 10, "mean_exec_time": 5.0, "rows": 1,
             "total_exec_time": 50.0},
            {"query": "CREATE INDEX x ON orders(status)", "calls": 1, "mean_exec_time": 1, "rows": 0, "total_exec_time": 1},
            {"query": "SELECT * FROM pg_stat_activity", "calls": 1, "mean_exec_time": 1, "rows": 0, "total_exec_time": 1}]
    rep = await PgStatStatementsCollector(FakeInstrumentation(rows), engine).collect(10)
    assert rep.accepted == 1
    assert rep.records[0].source == "pg_stat_statements"
