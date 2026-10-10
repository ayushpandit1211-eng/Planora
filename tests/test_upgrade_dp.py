"""P10 major tests: Laplace mechanism, budget accounting/exhaustion, AI payload carries no exact counts."""
import os
import random
import statistics

import pytest

from app.config import reset_settings_cache
from app.dp import DPAccountant, dp_release, laplace, protect_schema_payload, set_test_rng, validate_params
from app.privacy_release import release_aggregate


@pytest.fixture(autouse=True)
def seeded():
    set_test_rng(random.Random(7))
    yield
    set_test_rng(None)


def test_validation_clipping_noise_scale_and_budget():
    for bad in [(0, 1, 0, None), (-1, 1, 0, None), (1, 0, 0, None), (1, 1, 5, 1)]:
        with pytest.raises(ValueError):
            validate_params(*bad)
    samples = [laplace(2.0) for _ in range(20000)]
    assert abs(statistics.mean(samples)) < 0.1
    assert abs(statistics.mean(abs(x) for x in samples) - 2.0) < 0.1      # E|Lap(b)| = b (loose, seeded)
    acc = DPAccountant(total=1.0)
    events = []
    acc.audit_hook = lambda scope, snap: events.append((scope, snap))
    v = dp_release("rows", 10**9, upper=1000, epsilon=0.5, scope="s1", accountant=acc)
    assert v <= 1000 + 60                                                 # clipped to the declared bound first
    assert dp_release("rows", 5, epsilon=0.5, scope="s1", accountant=acc) is not None
    assert acc.snapshot("s1")["epsilon_spent"] == 1.0 and acc.snapshot("s1")["releases"] == 2
    assert dp_release("rows", 5, epsilon=0.5, scope="s1", accountant=acc) is None   # exhausted -> no aggregate
    assert dp_release("rows", 5, epsilon=0.5, scope="s1", accountant=acc) is None
    assert len(events) == 1 and events[0][1]["exhausted"]                # audit event written once
    assert dp_release("rows", 5, epsilon=0.5, scope="s2", accountant=acc) is not None  # per-scope budgets


def test_ai_payload_has_no_exact_counts_and_fallback_sends_none():
    schema = {"tables": {"tbl_a": {"estimated_rows": 123456.0, "columns": [
        {"name": "col_b", "type": "integer", "n_distinct": 4321.0}, {"name": "col_c", "type": "text",
                                                                       "n_distinct": -0.5}]}},
        "indexes": [{"name": "ix_1", "table": "tbl_a", "columns": ["col_b"], "idx_scan": 999}]}
    acc = DPAccountant(total=100)
    out, rep = protect_schema_payload(schema, "x", acc)
    assert out["tables"]["tbl_a"]["estimated_rows"] != 123456.0
    assert out["tables"]["tbl_a"]["columns"][0]["n_distinct"] != 4321.0
    assert out["tables"]["tbl_a"]["columns"][1]["n_distinct"] == -0.5          # ratio, not a count: untouched
    assert out["indexes"][0]["columns"] == ["col_b"] and out["indexes"][0]["idx_scan"] != 999   # structure kept
    assert schema["tables"]["tbl_a"]["estimated_rows"] == 123456.0             # trusted copy untouched
    assert rep["provenance"] == ["DP-PROTECTED"] and rep["released"] == 3
    empty = DPAccountant(total=0.1)
    out2, _ = protect_schema_payload(schema, "x", empty)
    assert out2["tables"]["tbl_a"]["estimated_rows"] is None and out2["indexes"][0]["idx_scan"] is None


def test_flag_controls_choke_point():
    assert release_aggregate("rows", 42) == 42                                 # DP off: identity (old behaviour)
    os.environ["QS_FEATURE_DP"] = "true"
    reset_settings_cache()
    try:
        v = release_aggregate("rows", 42, scope="flagtest")
        assert isinstance(v, float)
    finally:
        os.environ.pop("QS_FEATURE_DP")
        reset_settings_cache()
