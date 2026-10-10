"""Evidence chain engine (Addendum R11, Section 68).

OBSERVATION -> PLAN/SCHEMA/WORKLOAD EVIDENCE -> GNN/RL/AGENT INTERPRETATION -> RECOMMENDATION
-> SANDBOX RESULT -> BLAST-RADIUS RESULT -> DEPLOYMENT RISK

Every item states its source type and whether it is observed, inferred, simulated, estimated
or benchmark-measured. The chain is built only from sanitised (pseudonymised) data.
"""
from __future__ import annotations

from typing import Any

from app.contracts import EvidenceItem, EvidenceKind


EVIDENCE_TYPES = ("OBSERVATION", "PLAN_EVIDENCE", "GAT_EVIDENCE", "WORKLOAD_EVIDENCE", "SCHEMA_EVIDENCE",
                  "RL_DECISION", "HYPOPG", "SANDBOX", "BLAST_RADIUS", "FINAL_RECOMMENDATION")


def _classify(step: str, source_type: str, kind: EvidenceKind, data: dict) -> tuple[str, list[str], str]:
    """(evidence_type, provenance, artifact_id) derived from where the item came from (code, never the LLM)."""
    st = source_type.lower()
    if step == "OBSERVATION":
        return "OBSERVATION", ["POSTGRES OBSERVED", "PLANNER ESTIMATE"], "explain_plan"
    if step == "EVIDENCE":
        if st.startswith("plan"):
            return "PLAN_EVIDENCE", ["PLANNER ESTIMATE"], "explain_plan"
        if "workload" in st:
            return "WORKLOAD_EVIDENCE", ["POSTGRES OBSERVED"], "workload_history"
        if "ast" in st:
            return "SCHEMA_EVIDENCE", ["POSTGRES OBSERVED"], "anonymised_sql_ast"
        return "SCHEMA_EVIDENCE", ["POSTGRES OBSERVED"], "schema_catalog"
    if step == "INTERPRETATION":
        mode = str(data.get("model_mode") or data.get("policy_mode") or "")
        trained = mode in ("trained", "trained_policy")
        if st.startswith("gat"):
            return "GAT_EVIDENCE", ["TRAINED GNN" if trained else "ANALYTIC FALLBACK"], "gnn_attention"
        if st.startswith("rl"):
            return "RL_DECISION", ["RL POLICY" if trained else "ANALYTIC FALLBACK"], "rl_recommendation"
        return "PLAN_EVIDENCE", ["TRAINED GNN" if trained else "ANALYTIC FALLBACK"], "gnn_diagnosis"
    if step == "RECOMMENDATION":
        return "FINAL_RECOMMENDATION", ["HEURISTIC ESTIMATE"], "agent_recommendation"
    if step == "SANDBOX_RESULT":
        if kind == EvidenceKind.BENCHMARK_MEASURED:
            return "SANDBOX", ["MEASURED SANDBOX"], "benchmark_result"
        return "HYPOPG", ["HYPOPG SIMULATION", "PLANNER ESTIMATE"], "hypopg_result"
    if step == "BLAST_RADIUS_RESULT":
        return "BLAST_RADIUS", ["POSTGRES OBSERVED", "HEURISTIC ESTIMATE"], "blast_radius"
    return "FINAL_RECOMMENDATION", ["HEURISTIC ESTIMATE"], f"migration:{data.get('migration_id') or 'n/a'}"


def validate_item(item: EvidenceItem | dict) -> None:
    """Rejects evidence without a provenance tag or an artifact reference (Final Upgrade P11)."""
    from app.provenance import ALL_TAGS
    d = item.model_dump() if isinstance(item, EvidenceItem) else item
    if not d.get("provenance") or not all(t in ALL_TAGS for t in d["provenance"]):
        raise ValueError("evidence item without a valid provenance tag")
    if not d.get("artifact_id"):
        raise ValueError("evidence item without a source artifact reference")
    if d.get("evidence_type") not in EVIDENCE_TYPES:
        raise ValueError("unknown evidence type")


class EvidenceChain:
    def __init__(self) -> None:
        self.items: list[EvidenceItem] = []

    def add(self, step: str, source_type: str, kind: EvidenceKind, statement: str, **data: Any) -> None:
        et, prov, art = _classify(step, source_type, kind, data)
        item = EvidenceItem(step=step, source_type=source_type, kind=kind, statement=statement, data=data,
                            evidence_type=et, provenance=prov, artifact_id=art)
        validate_item(item)
        self.items.append(item)

def build_chain(*, plan: dict | None, gnn: dict | None, access: dict | None, schema_findings: dict | None,
                workload: dict | None, recommendation: dict | None, rl: dict | None, sandbox: dict | None,
                blast: list[dict] | None, deployment: dict | None, benchmark: dict | None = None) -> EvidenceChain:
    ch = EvidenceChain()
    if plan:
        hc = plan.get("highest_cost_node") or {}
        ch.add("OBSERVATION", "EXPLAIN (FORMAT JSON)", EvidenceKind.OBSERVED,
               f"Planner total cost = {plan.get('total_cost'):.2f}; highest effective-cost operator = "
               f"{hc.get('node_type')} {('on ' + hc['relation_name']) if hc.get('relation_name') else ''} "
               f"(share {hc.get('cost_share')})",
               total_cost=plan.get("total_cost"), plan_rows=plan.get("plan_rows"), highest_cost_node=hc)
        for s in plan.get("seq_scans", [])[:3]:
            ch.add("EVIDENCE", "plan", EvidenceKind.OBSERVED,
                   f"Seq Scan on {s.get('relation')} plan rows {s.get('plan_rows'):.0f}, filter pattern "
                   f"{s.get('filter') or 'none'}, executed ~{s.get('estimated_loops', 1):.0f}x", **s)
    if schema_findings:
        for f in schema_findings.get("findings", [])[:6]:
            ch.add("EVIDENCE", "schema catalog", EvidenceKind.OBSERVED, f)
    if access:
        for t, roles in (access.get("per_table") or {}).items():
            ch.add("EVIDENCE", "SQL AST (anonymised)", EvidenceKind.OBSERVED,
                   f"{t}: " + "; ".join(f"{r}={', '.join(c)}" for r, c in roles.items() if r != "select"))
    if workload and workload.get("available"):
        ch.add("EVIDENCE", "workload history", EvidenceKind.OBSERVED,
               f"fingerprint seen in {workload.get('records')} historical records, {workload.get('total_calls')} calls; "
               f"frequency weight {workload.get('frequency_weight')}", **{k: workload.get(k) for k in ("records", "total_calls")})
    if gnn:
        kind = EvidenceKind.INFERRED
        ch.add("INTERPRETATION", f"GNN ({gnn.get('model_mode')})", kind,
               f"{gnn.get('primary_bottleneck_type')} (confidence {gnn.get('confidence')})",
               model_mode=gnn.get("model_mode"))
    if gnn and gnn.get("attention") and gnn["attention"].get("bottleneck_node"):
        bn = gnn["attention"]["bottleneck_node"]
        ch.add("INTERPRETATION", "GATv2 attention", EvidenceKind.INFERRED,
               f"attention-based structural importance signal (interpretability aid, not causal proof) highest "
               f"at node {bn.get('id')} ({bn.get('operator')})", model_mode=gnn["attention"].get("model_mode"))
    if rl:
        ch.add("INTERPRETATION", f"RL ({rl.get('policy_mode')})", EvidenceKind.INFERRED,
               f"policy chose {rl.get('high_level_action')}" + (f" / {rl.get('low_level_action')}" if rl.get("low_level_action") else ""),
               policy_mode=rl.get("policy_mode"))
    if recommendation:
        for ix in recommendation.get("proposed_indexes", [])[:5]:
            ch.add("RECOMMENDATION", ix.get("source", "index_strategist"), EvidenceKind.INFERRED,
                   f"{ix.get('ddl')} - {ix.get('reason')}", evidence=ix.get("evidence"))
        rw = recommendation.get("rewrite")
        if rw:
            ch.add("RECOMMENDATION", "sql_rewriter", EvidenceKind.INFERRED,
                   f"rewrite ({', '.join(rw.get('rules_applied', []))}); semantics verified by AST diff: "
                   f"{rw.get('semantics_verified')}")
        for k in ("partitioning_recommendation", "sharding_recommendation"):
            if recommendation.get(k):
                a = recommendation[k]
                ch.add("RECOMMENDATION", "schema_expert", EvidenceKind.INFERRED,
                       f"{k.split('_')[0]} {a.get('strategy_type')} on {a.get('candidate_key')} - {a.get('label')}")
    if sandbox:
        if sandbox.get("status") == "completed":
            ch.add("SANDBOX_RESULT", "HypoPG + EXPLAIN", EvidenceKind.SIMULATED,
                   f"planner cost {sandbox.get('before_total_cost')} -> {sandbox.get('after_total_cost')} "
                   f"({sandbox.get('planner_cost_reduction_pct')}% planner-estimated, NOT runtime); "
                   f"virtual indexes used: {len(sandbox.get('virtual_indexes_used') or [])}",
                   plan_changed=sandbox.get("plan_changed"))
            if sandbox.get("rewrite_after_cost") is not None:
                ch.add("SANDBOX_RESULT", "EXPLAIN of rewritten SQL", EvidenceKind.SIMULATED,
                       f"rewrite planner cost {sandbox.get('rewrite_before_cost')} -> {sandbox.get('rewrite_after_cost')} "
                       f"({sandbox.get('rewrite_cost_reduction_pct')}%)")
        else:
            ch.add("SANDBOX_RESULT", "HypoPG", EvidenceKind.SIMULATED, f"sandbox {sandbox.get('status')}: {sandbox.get('error')}")
    if benchmark and benchmark.get("measured"):
        ch.add("SANDBOX_RESULT", "benchmark (opt-in, isolated)", EvidenceKind.BENCHMARK_MEASURED,
               f"measured execution time change {benchmark.get('measured_execution_time_reduction_pct')}%")
    for b in blast or []:
        ch.add("BLAST_RADIUS_RESULT", "pg_stat_statements + catalog", EvidenceKind.ESTIMATED,
               f"{b.get('target_table')}: risk {b.get('risk_level')}, {b.get('write_frequency_summary', {}).get('write_calls_on_table')} "
               f"write calls on table; storage ~{b.get('estimated_storage_impact_mb')} MB (estimated)")
    if deployment:
        ch.add("DEPLOYMENT_RISK", "migration generator", EvidenceKind.INFERRED,
               f"migration {deployment.get('migration_id')} requires human review; staging only; "
               "actual runtime requires benchmark")
    return ch
