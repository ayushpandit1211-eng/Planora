"""P11 major tests: evidence chain provenance validator; blast radius maps writes via the parser."""
import pytest

from app.blast_radius import parse_write
from app.contracts import EvidenceKind
from app.evidence_engine import EvidenceChain, build_chain, validate_item


def test_chain_items_carry_provenance_and_artifacts_and_validator_rejects():
    ch = build_chain(plan={"total_cost": 10.0, "highest_cost_node": {"node_type": "Seq Scan"}, "seq_scans": []},
                     gnn={"model_mode": "trained", "primary_bottleneck_type": "seq_scan", "confidence": 0.9,
                          "attention": {"model_mode": "trained", "bottleneck_node": {"id": 1, "operator": "Seq Scan"}}},
                     access=None, schema_findings={"findings": ["f1"]}, workload={"available": True, "records": 2},
                     recommendation={"proposed_indexes": [{"ddl": "CREATE INDEX ON tbl_a (col_b)", "reason": "r"}]},
                     rl={"policy_mode": "analytic_fallback", "high_level_action": "ADD_INDEX"},
                     sandbox={"status": "completed", "before_total_cost": 10, "after_total_cost": 2},
                     blast=[{"target_table": "tbl_a", "risk_level": "low"}], deployment={"migration_id": "QSM-1"})
    types = {i.evidence_type for i in ch.items}
    assert {"OBSERVATION", "GAT_EVIDENCE", "WORKLOAD_EVIDENCE", "SCHEMA_EVIDENCE", "RL_DECISION", "HYPOPG",
            "BLAST_RADIUS", "FINAL_RECOMMENDATION"} <= types
    for i in ch.items:
        validate_item(i)
    rl = [i for i in ch.items if i.evidence_type == "RL_DECISION"][0]
    assert rl.provenance == ["ANALYTIC FALLBACK"]
    with pytest.raises(ValueError):
        validate_item({"evidence_type": "OBSERVATION", "provenance": [], "artifact_id": "x"})
    with pytest.raises(ValueError):
        validate_item({"evidence_type": "OBSERVATION", "provenance": ["PLANNER ESTIMATE"], "artifact_id": None})


def test_blast_radius_maps_write_to_table_via_parser():
    w = parse_write("UPDATE orders SET status = $1 WHERE order_id = $2")
    assert w["target_table"] == "orders" and w["kind"] == "UPDATE" and w["set_columns"] == ["status"]
    assert parse_write("not sql") is None
