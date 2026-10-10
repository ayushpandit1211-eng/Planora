"""Builds RL episodes from real queries using the SAME analysis components as the live pipeline
(security gateway, masking, schema inspection, EXPLAIN, GNN, predicate intelligence, index
strategist, SQL rewriter, partition advisor, validator). Used by scripts/train_rl.py.

Real DDL / SQL needed by the HypoPG evaluator is resolved at build time on the trusted side and
kept in a private side-table keyed by action - it is never part of the saved policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.index_strategist import IndexStrategist
from app.agents.sql_rewriter import SQLRewriter
from app.contracts import IndexCandidate
from app.gnn.analyzer import GNNAnalyzer
from app.masking_vault import DataObfuscator
from app.partition_advisor import PartitionAdvisor
from app.plan_parser import parse_explain, sanitize_plan
from app.postgres_instrumentation import PostgresInstrumentation
from app.predicate_intelligence import PredicateAnalyzer
from app.recommendation_mapper import RecommendationValidator
from app.rl.environment import EpisodeContext, LowAction
from app.rl.state_encoder import encode_state
from app.sandbox import SyncHypoPGEvaluator
from app.schema_inspector import SchemaInspector, pseudonymize_metadata, table_size_bucket, write_frequency_bucket
from app.security import SecurityGateway


@dataclass
class Prepared:
    """Trusted-side analysis of one query (shared by RL training and the benchmark lab)."""
    vq: Any
    masked_sql: str
    fingerprint: str
    plan: Any
    diag: Any
    access: Any
    candidates: list
    actions: list
    real_ddl: dict = field(default_factory=dict)     # action id -> canonical real DDL
    real_sql: dict = field(default_factory=dict)     # rewrite rule -> real SQL
    state: Any = None
    table_bytes: float | None = None
    partition: Any = None


async def prepare(rid: str, sql: str, instr: PostgresInstrumentation, obf: DataObfuscator,
                  gnn: GNNAnalyzer, frequency_weight: float = 1.0) -> Prepared:
    vq = SecurityGateway().validate(sql)
    priv = obf.mask_sql(vq.sql, rid)
    real_schema = await SchemaInspector(instr).inspect(vq.referenced_relations)
    pseudo = pseudonymize_metadata(real_schema, obf, rid)
    plan = sanitize_plan(parse_explain(await instr.explain(vq.sql)), obf, rid)
    diag = gnn.analyze(plan)
    ap = PredicateAnalyzer(column_index={t: {c.name for c in m.columns} for t, m in pseudo.tables.items()}).analyze(priv["masked_sql"])
    scanned = {}
    for n in plan.nodes:
        if n.relation_name and n.node_type in ("Seq Scan", "Bitmap Heap Scan"):
            scanned[n.relation_name] = {"node_type": n.node_type, "loops": n.estimated_loops,
                                        "cost_share": min(n.effective_cost / plan.total_cost, 1.0) if plan.total_cost else 0}
    cands = IndexStrategist(pseudo).propose(ap, scanned, max_candidates=8)
    validator = RecommendationValidator(obf, real_schema)
    real_ddl: dict[int, str] = {}
    actions: list[LowAction] = []
    for i, c in enumerate(cands):
        redundant = c.source.endswith("redundant")
        if not redundant:
            o = validator.validate(IndexCandidate(**c.model_dump()), rid)
            if not o.valid:
                continue
            real_ddl[i] = o.real_ddl
        actions.append(LowAction("ADD_INDEX", c.signature, payload={"id": i, "kind": c.kind, "roles": c.roles,
                                                                    "columns": c.columns}, redundant=redundant))
    real_sql: dict[str, str] = {}
    for op in SQLRewriter(pseudo).candidate_operators(priv["masked_sql"]):
        rule = op.rules_applied[0]
        try:
            real_sql[rule] = obf.translate_llm_sql_to_real(op.rewritten_masked_sql, rid)
        except Exception:  # noqa: BLE001
            continue
        actions.append(LowAction("REWRITE_SQL", rule, payload={"id": rule, "rule": rule}, verified=op.semantics_verified))
    adv = PartitionAdvisor(pseudo).advise(ap, frequency_weight)
    if adv["partitioning"]:
        actions.append(LowAction("PARTITION", adv["partitioning"].strategy_type, payload={"id": "partition"}))
    if adv["sharding"]:
        actions.append(LowAction("SHARD_ADVISORY", "tenant_key", payload={"id": "shard"}))
    hc = plan.highest_cost_node or {}
    target = hc.get("relation_name") or (ap.tables[0] if ap.tables else None)
    tmeta = pseudo.tables.get(target or "")
    per = ap.per_table()
    eqcols = set(per.get(target or "", {}).get("eq", []))
    coverage = 0
    for ix in pseudo.indexes_for(target or ""):
        if ix.columns and ix.columns[0] in eqcols:
            coverage = max(coverage, 1 + int(set(ix.columns[:len(eqcols)]) == eqcols))
    state = encode_state(
        query_fingerprint=priv["ast_signature"], bottleneck_class=diag.bottleneck_class_id,
        plan_total_cost=plan.total_cost, plan_rows=plan.plan_rows, operator_histogram=plan.operator_histogram,
        per_table_access=per, target_table=target, existing_index_coverage=coverage,
        table_size_bucket=table_size_bucket(tmeta.estimated_rows) if tmeta else None,
        write_frequency_bucket=write_frequency_bucket((tmeta.n_tup_ins or 0) + (tmeta.n_tup_upd or 0) + (tmeta.n_tup_del or 0)) if tmeta else None,
        partition_suitable=adv["partitioning"] is not None, non_sargable=bool(ap.non_sargable()),
        has_subplan=bool(ap.correlated_subqueries))
    real_table = next((t for t in real_schema.tables if obf.to_masked_identifier(rid, t) == target), None)
    table_bytes = real_schema.tables[real_table].table_size_bytes if real_table else None
    obf.clear_vault(rid)
    return Prepared(vq=vq, masked_sql=priv["masked_sql"], fingerprint=priv["ast_signature"], plan=plan, diag=diag,
                    access=ap, candidates=cands, actions=actions, real_ddl=real_ddl, real_sql=real_sql,
                    state=state, table_bytes=table_bytes, partition=adv["partitioning"])


async def build_episode(rid: str, sql: str, instr: PostgresInstrumentation, obf: DataObfuscator,
                        gnn: GNNAnalyzer, sync_conn, pattern: str, frequency_weight: float = 1.0) -> EpisodeContext:
    p = await prepare(rid, sql, instr, obf, gnn, frequency_weight)
    evaluator = SyncHypoPGEvaluator(
        sync_conn, p.vq.sql,
        ddl_for=lambda a: p.real_ddl.get(a.payload.get("id")),
        sql_for=lambda a: p.real_sql.get(a.payload.get("id")),
        table_bytes=p.table_bytes)
    meta: dict[str, Any] = {"pattern": pattern, "gnn": p.diag.primary_bottleneck_type, "n_actions": len(p.actions),
                            "baseline_cost": p.plan.total_cost}
    return EpisodeContext(state=p.state, actions=p.actions, evaluator=evaluator, frequency_weight=frequency_weight,
                          meta=meta)
