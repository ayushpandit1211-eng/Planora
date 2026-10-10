"""P12 major test: threshold-based labels, no false regression, old workload rows still load."""
from app.regression_detector import detect_regressions
from app.workload_drift import detect_drift
from app.workload_store import WorkloadRecord, WorkloadStore
from tests.test_research_modules import rec


def test_labels_no_false_positive_and_old_rows_readable():
    same = [rec("q", "pA", f"2026-09-0{d}T00", 100, 5, {"Index Scan": 1}) for d in (1, 2, 3)]
    assert detect_regressions(same) == []                       # identical plans: no regression event
    changed = same + [rec("q", "pB", "2026-09-04T00", 300, 9, {"Seq Scan": 1})]
    ev = detect_regressions(changed)
    assert ev[0]["event"] == "PLAN_REGRESSION" and ev[0]["detection_method"].startswith("threshold-based")
    assert detect_drift(same, same, 0.2)["detection_method"].startswith("threshold-based")
    old = WorkloadRecord.model_validate({"query_fingerprint": "q_old", "timestamp_bucket": "2026-01-01T00:00:00"})
    assert old.data_source_id == "demo"                         # pre-upgrade rows default to the demo namespace
    st = WorkloadStore()
    st.add(old)
    st.scoped("sbx_abcdef123456").add(rec("q_s", "p", "2026-09-01T00", 1))
    assert [r.query_fingerprint for r in st.all()] == ["q_old"]
    assert [r.query_fingerprint for r in st.scoped("sbx_abcdef123456").all()] == ["q_s"]
    assert st.purge_source("sbx_abcdef123456") == 1
