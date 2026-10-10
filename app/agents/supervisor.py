"""LangGraph multi-agent supervisor (Section 22) - a REAL stateful graph that executes the full
optimisation pipeline:

START -> security_validation -> privacy_masking -> schema_inspection -> explain -> plan_to_graph
-> gnn_analysis -> supervisor --(conditional fan-out)--> {sql_rewriter, schema_expert, index_strategist}
-> supervisor_merge -> rl_optimizer -> recommendation_validator --(conditional)--> hypopg_sandbox
--(conditional)--> blast_radius -> final_report -> END

Conditional routing: the SQL rewriter is skipped when no rewrite applies; index simulation is
skipped when no valid candidate exists; unsafe recommendations are rejected by the validator;
every result requires human review, and high-risk ones are flagged. Graph state holds only
anonymised data; trusted data lives in Services.trusted[request_id].
"""
from __future__ import annotations

import re

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.index_strategist import IndexStrategist
from app.agents.schema_expert import SchemaExpert
from app.agents.sql_rewriter import SQLRewriter
from app.agents.state import QSState, initial_state
from app.blast_radius import BlastRadiusAnalyzer
from app.capability_firewall import Capability
from app.contracts import (AgentRecommendation, FinalOptimizationReport, IndexCandidate, PartitionAdvice,
                           PlanAnalysis, RewriteProposal, ShardingAdvice)
from app.errors import ErrorCategory, QuerySentinelError
from app.evidence_engine import build_chain
from app.gnn.mapper import plan_to_graph as to_graph
from app.metrics import metrics
from app.optimization_budget import OptimizationBudget
from app.pii import PIIDetector
from app.pipeline import Services
from app.plan_parser import parse_explain, plan_to_tree_text, sanitize_plan
from app.predicate_intelligence import PredicateAnalyzer
from app.recommendation_mapper import RecommendationValidator, real_index_parts
from app.regression_detector import compare_current
from app.rl.environment import LowAction
from app.rl.state_encoder import encode_state
from app.sandbox import HypoPGSandbox
from app.schema_inspector import anonymize, pseudonymize_metadata, table_size_bucket, write_frequency_bucket
from app.security import SecurityGateway
from app.structural_sim import StructuralSimulator
from app.workload_drift import detect_drift

NodeFn = Callable[[QSState], Awaitable[dict]]


def _now() -> str:
    return time.strftime("%H:%M:%S")


class Pipeline:
    def __init__(self, services: Services):
        self.s = services
        self.graph = self._build()

    # ================================================================== helpers
    def emit(self, rid: str, event: str, stage: str, message: str, **data) -> None:
        self.s.events.publish(rid, event, stage, message, data)

    def log(self, agent: str, message: str) -> dict:
        return {"ts": _now(), "agent": agent, "message": message}

    def t(self, rid: str) -> dict[str, Any]:
        return self.s.trusted.setdefault(rid, {})

    def node(self, name: str, status_key: str | None = None) -> Callable[[NodeFn], NodeFn]:
        """Wrap a node: skip when an earlier node failed; catch and structure errors; track status."""
        def deco(fn: NodeFn) -> NodeFn:
            @functools.wraps(fn)
            async def wrapper(state: QSState) -> dict:
                rid = state["request_id"]
                key = status_key or name
                if state.get("error"):
                    return {"agent_status": {key: "SKIPPED"}} if key in (state.get("agent_status") or {}) else {}
                budget: OptimizationBudget = self.t(rid)["budget"]
                if not budget.consume("agent_steps"):
                    return {"agent_status": {key: "SKIPPED"},
                            "agent_logs": [self.log(name, "skipped: agent-step budget exhausted")]}
                t0 = time.perf_counter()
                try:
                    out = await fn(state)
                except QuerySentinelError as exc:
                    metrics.inc(f"error.{exc.category.value}")
                    self.s.audit.record(rid, "ERROR", stage=name, severity="error",
                                        details={"category": exc.category.value, "message": exc.message})
                    self.emit(rid, "ERROR", name, f"{name} failed: {exc.message}", category=exc.category.value)
                    return {"error": exc.to_dict(rid) | {"stage": name}, "agent_status": {key: "ERROR"}}
                except Exception as exc:  # noqa: BLE001
                    self.s.audit.record(rid, "ERROR", stage=name, severity="error",
                                        details={"category": "internal_error", "type": type(exc).__name__})
                    self.emit(rid, "ERROR", name, f"{name} failed with an internal error", category="internal_error")
                    return {"error": {"error": "internal_error", "message": f"{type(exc).__name__} in {name}",
                                      "stage": name, "request_id": rid}, "agent_status": {key: "ERROR"}}
                elapsed = (time.perf_counter() - t0) * 1000
                metrics.observe(name, elapsed)
                out.setdefault("agent_status", {})
                if key in (state.get("agent_status") or {}) and key not in out["agent_status"]:
                    out["agent_status"][key] = "COMPLETED"
                out.setdefault("agent_logs", []).append(
                    self.log(name, f"completed in {(time.perf_counter() - t0) * 1000:.0f} ms"))
                return out
            return wrapper
        return deco

    # ================================================================== graph
    def _build(self):
        g = StateGraph(QSState)
        g.add_node("security_validation", self.node("security_validation")(self.n_security))
        g.add_node("privacy_masking", self.node("privacy_masking")(self.n_privacy))
        g.add_node("schema_inspection", self.node("schema_inspection")(self.n_schema))
        g.add_node("explain", self.node("explain")(self.n_explain))
        g.add_node("plan_to_graph", self.node("plan_to_graph")(self.n_graph))
        g.add_node("gnn_analysis", self.node("gnn_analysis")(self.n_gnn))
        g.add_node("supervisor", self.node("supervisor")(self.n_supervisor))
        g.add_node("sql_rewriter", self.node("sql_rewriter")(self.n_sql_rewriter))
        g.add_node("schema_expert", self.node("schema_expert")(self.n_schema_expert))
        g.add_node("index_strategist", self.node("index_strategist")(self.n_index_strategist))
        g.add_node("supervisor_merge", self.node("supervisor_merge", "supervisor")(self.n_merge))
        g.add_node("rl_optimizer", self.node("rl_optimizer")(self.n_rl))
        g.add_node("recommendation_validator", self.node("recommendation_validator")(self.n_validator))
        g.add_node("hypopg_sandbox", self.node("hypopg_sandbox")(self.n_sandbox))
        g.add_node("blast_radius", self.node("blast_radius")(self.n_blast))
        g.add_node("final_report", self.n_final)   # always runs (also produces error reports)

        g.add_edge(START, "security_validation")
        g.add_conditional_edges("security_validation", self.r_ok("privacy_masking"), ["privacy_masking", "final_report"])
        g.add_conditional_edges("privacy_masking", self.r_ok("schema_inspection"), ["schema_inspection", "final_report"])
        g.add_conditional_edges("schema_inspection", self.r_ok("explain"), ["explain", "final_report"])
        g.add_conditional_edges("explain", self.r_ok("plan_to_graph"), ["plan_to_graph", "final_report"])
        g.add_edge("plan_to_graph", "gnn_analysis")
        g.add_conditional_edges("gnn_analysis", self.r_ok("supervisor"), ["supervisor", "final_report"])
        g.add_conditional_edges("supervisor", self.r_experts, ["sql_rewriter", "schema_expert", "index_strategist", "final_report"])
        for expert in ("sql_rewriter", "schema_expert", "index_strategist"):
            g.add_edge(expert, "supervisor_merge")
        g.add_edge("supervisor_merge", "rl_optimizer")
        g.add_edge("rl_optimizer", "recommendation_validator")
        g.add_conditional_edges("recommendation_validator", self.r_after_validation, ["hypopg_sandbox", "final_report"])
        g.add_conditional_edges("hypopg_sandbox", self.r_after_sandbox, ["blast_radius", "final_report"])
        g.add_edge("blast_radius", "final_report")
        g.add_edge("final_report", END)
        return g.compile()

    # ---------------------------------------------------------------- routers
    @staticmethod
    def r_ok(nxt: str):
        def route(state: QSState) -> str:
            return "final_report" if state.get("error") else nxt
        return route

    @staticmethod
    def r_experts(state: QSState) -> list[str]:
        if state.get("error"):
            return ["final_report"]
        return state.get("route") or ["schema_expert"]

    @staticmethod
    def r_after_validation(state: QSState) -> str:
        if state.get("error"):
            return "final_report"
        v = state.get("validation") or {}
        return "hypopg_sandbox" if (v.get("valid_indexes") or v.get("rewrite_real_available")) else "final_report"

    @staticmethod
    def r_after_sandbox(state: QSState) -> str:
        sb = state.get("sandbox_results") or {}
        return "blast_radius" if (not state.get("error") and sb.get("virtual_indexes_used")) else "final_report"

    # ================================================================== nodes
    async def n_security(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.emit(rid, "SECURITY_SCAN_STARTED", "security", "Security validation started")
        try:
            vq = SecurityGateway(self.s.settings.max_query_length).validate(tr["raw_sql"])
        except QuerySentinelError as exc:
            self.s.audit.record(rid, "SECURITY_REJECTION", stage="security", severity="security",
                                details={"category": exc.category.value, "reason": exc.message})
            self.emit(rid, "SECURITY_REJECTED", "security", f"Request rejected: {exc.message}")
            return {"error": exc.to_dict(rid), "agent_logs": [self.log("security", f"rejected: {exc.message}")]}
        tr["validated"] = vq
        self.s.audit.record(rid, "SECURITY_SCAN", stage="security",
                            details={"statement_type": vq.statement_type, "relations": len(vq.referenced_relations),
                                     "generic_plan": vq.has_params})
        self.emit(rid, "SECURITY_SCAN_COMPLETE", "security", "Security validation complete",
                  statement_type=vq.statement_type)
        return {"agent_logs": [self.log("security", "single read-only SELECT validated")]}

    async def n_privacy(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        scan = PIIDetector().scan(tr["raw_sql"])
        self.s.audit.record(rid, "PII_DETECTION", stage="privacy", details=scan.as_dict())
        self.emit(rid, "PII_SCAN_COMPLETE", "privacy", "PII pre-filter complete",
                  pii_detected=scan.pii_detected, pii_types=sorted(scan.pii_types))
        self.emit(rid, "MASKING_STARTED", "privacy", "AST parameterisation and pseudonymisation started")
        priv = self.s.obfuscator.mask_sql(tr["validated"].sql, rid)
        self.s.audit.record(rid, "MASKING", stage="privacy",
                            details={"literal_count": priv["literal_count"], "fallback": priv["fallback_used"],
                                     "safe_for_llm": priv["privacy_audit"]["safe_for_llm"],
                                     "bitmask": priv["privacy_bitmask"]})
        if priv["quarantined"] or not priv["privacy_audit"]["safe_for_llm"]:
            self.emit(rid, "PRIVACY_BLOCKED", "privacy", "Request quarantined: safe anonymisation could not be established")
            raise QuerySentinelError(ErrorCategory.QUARANTINED, "safe anonymisation could not be established",
                                     stage="privacy")
        self.emit(rid, "MASKING_COMPLETE", "privacy", "Privacy masking complete",
                  literal_count=priv["literal_count"], pseudonymized=len(priv["mapped_tables"]) + len(priv["mapped_columns"]))
        tr["query_fingerprint"] = priv["ast_signature"]
        return {"masked_sql": priv["masked_sql"], "privacy": priv,
                "agent_logs": [self.log("privacy", f"{priv['literal_count']} literals parameterised")]}

    async def n_schema(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        tables = [t for t in tr["validated"].referenced_relations if not self.s.known_tables or t in self.s.known_tables]
        from app.schema_inspector import SchemaInspector
        real = await SchemaInspector(self.s.instr).inspect(tables or None)
        tr["schema_real"] = real
        pseudo = pseudonymize_metadata(real, self.s.obfuscator, rid)
        tr["schema_pseudo"] = pseudo
        anon = anonymize(real, self.s.obfuscator, rid)
        self.s.audit.record(rid, "SCHEMA_INSPECTION", stage="schema", details={"tables": len(real.tables),
                                                                               "indexes": len(real.indexes)})
        self.emit(rid, "SCHEMA_INSPECTION_COMPLETE", "schema", f"Schema inspected: {len(real.tables)} table(s)",
                  tables=len(real.tables), indexes=len(real.indexes))
        return {"schema_stats": anon}

    async def n_explain(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        vq = tr["validated"]
        self.emit(rid, "EXPLAIN_STARTED", "explain", "EXPLAIN (FORMAT JSON, VERBOSE, BUFFERS) started"
                  + (" with GENERIC_PLAN" if vq.has_params else ""))
        doc = await self.s.instr.explain(vq.sql, generic=vq.has_params)
        real_plan = parse_explain(doc)
        tr["plan_real"] = real_plan
        safe = sanitize_plan(real_plan, self.s.obfuscator, rid)
        tr["plan_safe"] = safe
        self.s.audit.record(rid, "EXPLAIN", stage="explain", details={"total_cost": safe.total_cost,
                                                                      "nodes": safe.node_count,
                                                                      "plan_fingerprint": safe.plan_fingerprint})
        self.emit(rid, "EXPLAIN_COMPLETE", "explain", f"EXPLAIN analysis complete (planner cost {safe.total_cost:.2f})",
                  total_cost=safe.total_cost, nodes=safe.node_count)
        d = safe.model_dump()
        d["tree_text"] = plan_to_tree_text(safe)
        return {"plan_analysis": d}

    async def n_graph(self, state: QSState) -> dict:
        rid = state["request_id"]
        safe: PlanAnalysis = self.t(rid)["plan_safe"]
        data = to_graph(safe)
        self.s.audit.record(rid, "GRAPH_CONVERSION", stage="graph", details={"nodes": int(data.num_nodes),
                                                                             "edges": int(data.edge_index.size(1))})
        self.emit(rid, "GRAPH_CONVERSION_COMPLETE", "graph", f"Plan converted to graph ({data.num_nodes} nodes)")
        return {"graph_summary": {"num_nodes": int(data.num_nodes), "num_edges": int(data.edge_index.size(1)),
                                  "feature_dim": int(data.x.size(1))}}

    async def n_gnn(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        safe: PlanAnalysis = tr["plan_safe"]
        diag = self.s.gnn.analyze(safe)
        self.s.audit.record(rid, "GNN_MODE", stage="gnn", details={"model_mode": diag.model_mode,
                                                                   "class": diag.primary_bottleneck_type})
        self.emit(rid, "GNN_ANALYSIS_COMPLETE", "gnn", f"GNN diagnosis complete ({diag.model_mode}): "
                  f"{diag.primary_bottleneck_type}", model_mode=diag.model_mode)
        pseudo = tr["schema_pseudo"]
        col_index = {t: {c.name for c in m.columns} for t, m in pseudo.tables.items()}
        ap = PredicateAnalyzer(column_index=col_index).analyze(state["masked_sql"])
        tr["access"] = ap
        fp = tr["query_fingerprint"]
        wctx = self.s.store.context_for(fp)
        regression = compare_current(self.s.store.history(fp), safe.plan_fingerprint, safe.total_cost,
                                     safe.operator_histogram)
        if regression and regression["event"] == "PLAN_REGRESSION":
            self.s.audit.record(rid, "PLAN_REGRESSION", stage="workload", details=regression)
            self.emit(rid, "PLAN_REGRESSION", "workload", "Plan regression detected against workload history")
        return {"gnn_diagnosis": diag.model_dump(), "access_pattern": ap.to_dict(), "workload_context": wctx,
                "regression": regression}

    async def n_supervisor(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "supervisor", Capability.READ_PLAN, "route")
        self.emit(rid, "SUPERVISOR_STARTED", "supervisor", "Supervisor routing specialist agents")
        gnn = state["gnn_diagnosis"]
        safe: PlanAnalysis = tr["plan_safe"]
        rewriter = SQLRewriter(tr["schema_pseudo"])
        route = ["schema_expert"]
        status: dict[str, str] = {"supervisor": "RUNNING"}
        reasons = []
        if rewriter.needs_rewrite(state["masked_sql"]) or (self.s.llm.enabled and gnn["bottleneck_class_id"] != 3):
            route.append("sql_rewriter")
            reasons.append("rewrite operator applicable")
        else:
            status["sql_rewriter"] = "SKIPPED"
            reasons.append("SQL rewriter skipped: no applicable rewrite")
            self.emit(rid, "AGENT_SKIPPED", "supervisor", "SQL rewriter skipped (no rewrite needed)")
        dominant_scan = any((s.get("cost_share") or 0) >= 0.2 for s in safe.seq_scans)
        if gnn["bottleneck_class_id"] != 3 or dominant_scan:
            route.append("index_strategist")
            reasons.append(f"bottleneck '{gnn['primary_bottleneck_type']}' -> index strategist")
        else:
            status["index_strategist"] = "SKIPPED"
            reasons.append("index strategist skipped: no bottleneck an index could address")
            self.emit(rid, "AGENT_SKIPPED", "supervisor", "Index strategist skipped (efficient plan)")
        self.s.audit.record(rid, "AGENT_ROUTING", stage="supervisor", details={"route": route})
        return {"route": route, "agent_status": status,
                "agent_logs": [self.log("supervisor", "; ".join(reasons))]}

    def _llm_data(self, state: QSState) -> dict:
        safe = self.t(state["request_id"])["plan_safe"]
        schema = state.get("schema_stats")
        from app import feature_flags
        if feature_flags.enabled("dp"):           # Final Upgrade P10: AI-facing numeric aggregates only
            from app.dp import protect_schema_payload
            scope = self.t(state["request_id"]).get("dp_scope") or "global"
            schema, dp_rep = protect_schema_payload(schema, scope)
            self.t(state["request_id"])["dp_report"] = dp_rep
        return {"anonymized_sql": state["masked_sql"], "schema": schema,
                "plan": {"total_cost": safe.total_cost, "operator_histogram": safe.operator_histogram,
                         "highest_cost_node": safe.highest_cost_node, "seq_scans": safe.seq_scans},
                "gnn_diagnosis": {k: state["gnn_diagnosis"].get(k) for k in
                                  ("primary_bottleneck_type", "model_mode", "natural_language_summary")},
                "access_pattern": state.get("access_pattern")}

    async def _call_llm(self, state: QSState, agent: str, task: str) -> dict | None:
        rid = state["request_id"]
        if not self.s.llm.enabled:
            return None
        self.s.firewall.require(rid, agent, Capability.CALL_LLM, task[:40])
        budget: OptimizationBudget = self.t(rid)["budget"]
        retries = min(self.s.settings.llm_max_retries, budget.remaining("llm_retries"))
        res = await asyncio.to_thread(self.s.llm.recommend, rid, task, self._llm_data(state), retries)
        budget.consume("llm_retries", max(res.attempts - 1, 0))
        self.s.audit.record(rid, "LLM_CALL", stage=agent, details=res.summary())
        self.emit(rid, "LLM_CALL", agent, f"LLM {res.provider}/{res.model}: {'ok' if res.success else res.error}")
        return {"summary": res.summary(), "recommendation": res.recommendation.model_dump() if res.recommendation else None}

    async def n_sql_rewriter(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "sql_rewriter", Capability.PROPOSE_SQL, "rewrite")
        self.emit(rid, "SQL_AGENT_STARTED", "sql_rewriter", "SQL rewriter started")
        rewriter = SQLRewriter(tr["schema_pseudo"])
        proposal = rewriter.apply(state["masked_sql"])
        operators = rewriter.candidate_operators(state["masked_sql"])
        llm = await self._call_llm(state, "sql_rewriter", "Propose a semantically equivalent, faster rewrite of the anonymized SQL, or null.")
        llm_prop = None
        if llm and llm.get("recommendation") and llm["recommendation"].get("rewritten_sql"):
            llm_prop = rewriter.validate_llm_rewrite(state["masked_sql"], llm["recommendation"]["rewritten_sql"])
        tr["rewrite_operators"] = operators
        self.emit(rid, "SQL_AGENT_COMPLETE", "sql_rewriter",
                  f"SQL rewriter complete: {', '.join(proposal.rules_applied) if proposal else 'no deterministic rewrite'}")
        return {"sql_agent": {"proposal": proposal.model_dump() if proposal else None,
                              "operators": [o.model_dump() for o in operators],
                              "llm_proposal": llm_prop.model_dump() if llm_prop else None},
                "proposed_sql": proposal.rewritten_masked_sql if proposal else (llm_prop.rewritten_masked_sql if llm_prop else None),
                "llm_calls": [llm["summary"]] if llm else []}

    async def n_schema_expert(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "schema_expert", Capability.READ_SCHEMA_METADATA, "analyze")
        self.emit(rid, "SCHEMA_AGENT_STARTED", "schema_expert", "Schema expert started")
        out = SchemaExpert(tr["schema_pseudo"]).analyze(tr["access"], tr["plan_safe"],
                                                        (state.get("workload_context") or {}).get("frequency_weight", 1.0))
        self.s.firewall.require(rid, "schema_expert", Capability.PROPOSE_SCHEMA_ADVISORY, "partition_advice")
        self.emit(rid, "SCHEMA_AGENT_COMPLETE", "schema_expert",
                  f"Schema expert complete: {len(out['findings'])} finding(s)")
        return {"schema_agent": out, "partitioning_recommendation": out["partitioning_recommendation"]}

    async def n_index_strategist(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "index_strategist", Capability.PROPOSE_INDEX, "propose")
        self.emit(rid, "INDEX_AGENT_STARTED", "index_strategist", "Index strategist started")
        safe: PlanAnalysis = tr["plan_safe"]
        scanned: dict[str, dict] = {}
        for n in safe.nodes:
            if n.relation_name and n.node_type in ("Seq Scan", "Bitmap Heap Scan"):
                share = min(n.effective_cost / safe.total_cost, 1.0) if safe.total_cost else 0
                prev = scanned.get(n.relation_name)
                if not prev or share > prev["cost_share"]:
                    scanned[n.relation_name] = {"node_type": n.node_type, "cost_share": share, "loops": n.estimated_loops}
        budget: OptimizationBudget = tr["budget"]
        cands = IndexStrategist(tr["schema_pseudo"]).propose(tr["access"], scanned, max_candidates=50)
        cands = budget.take("index_candidates", cands)
        llm = await self._call_llm(state, "index_strategist",
                                   "Propose index DDL (single/composite/partial/covering) for the anonymized query.")
        if llm and llm.get("recommendation"):
            for p in llm["recommendation"].get("proposed_indexes", [])[: max(budget.remaining("index_candidates"), 0)]:
                cands.append(IndexCandidate(table="(llm)", columns=p.get("columns") or ["(llm)"], ddl=p["ddl"],
                                            reason=p.get("reason", ""), source="llm", signature="llm:proposal"))
                budget.consume("index_candidates")
        for i, c in enumerate(cands):
            c.evidence = [e for e in c.evidence if e]
        tr["candidates"] = cands
        self.emit(rid, "INDEX_AGENT_COMPLETE", "index_strategist", f"Index strategist complete: {len(cands)} candidate(s)")
        return {"index_agent": {"candidates": [c.model_dump() for c in cands]},
                "proposed_indexes": [c.model_dump() for c in cands],
                "llm_calls": [llm["summary"]] if llm else []}

    async def n_merge(self, state: QSState) -> dict:
        sql_a = state.get("sql_agent") or {}
        sch = state.get("schema_agent") or {}
        rewrite = sql_a.get("proposal") or sql_a.get("llm_proposal")
        # Final Upgrade P6: a rewrite that screens HIGH semantic risk is never applied or simulated
        blocked = bool(rewrite and ((rewrite.get("semantic_diff") or {}).get("risk_screening") or {}).get("blocked"))
        rec = AgentRecommendation(
            rewritten_sql=rewrite["rewritten_masked_sql"] if rewrite and not blocked else None,
            rewrite=RewriteProposal(**rewrite) if rewrite else None,
            proposed_indexes=[IndexCandidate(**c) for c in state.get("proposed_indexes") or []],
            partitioning_recommendation=PartitionAdvice(**sch["partitioning_recommendation"]) if sch.get("partitioning_recommendation") else None,
            sharding_recommendation=ShardingAdvice(**sch["sharding_recommendation"]) if sch.get("sharding_recommendation") else None,
            extended_statistics=sch.get("extended_statistics", []),
            technical_reasoning=[state["gnn_diagnosis"]["natural_language_summary"]] + sch.get("findings", [])[:6]
            + ([rewrite["reasoning"]] if rewrite else []),
            assumptions=["planner statistics are representative of production data",
                         "anonymised structure is identical to the production schema"],
            risks=(["rewrite requires human review of semantics"] if rewrite else [])
            + (["rewrite BLOCKED: HIGH semantic risk (semantic-risk screening)"] if blocked else [])
            + (["partition/shard advice is advisory only"] if sch.get("partitioning_recommendation") else []),
            confidence=state["gnn_diagnosis"].get("confidence"),
            llm={"calls": state.get("llm_calls") or [], "provider": self.s.llm.provider, "enabled": self.s.llm.enabled},
        )
        return {"agent_recommendation": rec.model_dump(),
                "agent_logs": [self.log("supervisor", "merged specialist outputs")]}

    async def n_rl(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "rl_optimizer", Capability.READ_SANITIZED_STATS, "policy")
        self.emit(rid, "RL_ANALYSIS_STARTED", "rl", "RL policy evaluating candidate actions")
        rec = state["agent_recommendation"]
        safe: PlanAnalysis = tr["plan_safe"]
        pseudo = tr["schema_pseudo"]
        ap = tr["access"]
        actions: list[LowAction] = []
        for i, c in enumerate(rec["proposed_indexes"]):
            actions.append(LowAction("ADD_INDEX", c["signature"] or c["kind"],
                                     payload={"id": i, "ddl": c["ddl"], "kind": c["kind"], "roles": c["roles"],
                                              "columns": c["columns"], "table": c["table"]},
                                     redundant=c["source"].endswith("redundant")))
        for op in tr.get("rewrite_operators", []):
            actions.append(LowAction("REWRITE_SQL", op.rules_applied[0], payload={"rule": op.rules_applied[0]},
                                     verified=op.semantics_verified))
        if rec.get("partitioning_recommendation"):
            p = rec["partitioning_recommendation"]
            actions.append(LowAction("PARTITION", f"{p['strategy_type']}", payload={"table": p["table"], "key": p["candidate_key"]}))
        if rec.get("sharding_recommendation"):
            actions.append(LowAction("SHARD_ADVISORY", "tenant_key", payload={"key": rec["sharding_recommendation"]["candidate_key"]}))
        hc = safe.highest_cost_node or {}
        target = hc.get("relation_name") or (ap.tables[0] if ap.tables else None)
        tmeta = pseudo.tables.get(target or "")
        per = ap.per_table()
        eqcols = set(per.get(target or "", {}).get("eq", []))
        coverage = 0
        for ix in pseudo.indexes_for(target or ""):
            if ix.columns and ix.columns[0] in eqcols:
                coverage = max(coverage, 1 + int(set(ix.columns[:len(eqcols)]) == eqcols))
        st = encode_state(
            query_fingerprint=tr["query_fingerprint"], bottleneck_class=state["gnn_diagnosis"]["bottleneck_class_id"],
            plan_total_cost=safe.total_cost, plan_rows=safe.plan_rows, operator_histogram=safe.operator_histogram,
            per_table_access=per, target_table=target, existing_index_coverage=coverage,
            table_size_bucket=table_size_bucket(tmeta.estimated_rows) if tmeta else None,
            write_frequency_bucket=write_frequency_bucket((tmeta.n_tup_ins or 0) + (tmeta.n_tup_upd or 0) + (tmeta.n_tup_del or 0)) if tmeta else None,
            partition_suitable=bool(rec.get("partitioning_recommendation")), non_sargable=bool(ap.non_sargable()),
            has_subplan=bool(ap.correlated_subqueries))
        hist, recent = self.s.store.split_by_time()
        drift = detect_drift(hist, recent)
        if drift.get("drift_detected"):
            self.s.audit.record(rid, "WORKLOAD_DRIFT", stage="rl", details={"scores": drift["scores"]})
            self.emit(rid, "WORKLOAD_DRIFT", "rl", "Workload drift detected: policy retraining recommended")
        rlrec = self.s.rl.recommend(st, actions, drift_detected=bool(drift.get("drift_detected")))
        tr["rl_state"] = st
        self.s.audit.record(rid, "RL_RECOMMENDATION", stage="rl", details={"mode": rlrec.policy_mode,
                                                                           "action": rlrec.high_level_action})
        self.emit(rid, "RL_ANALYSIS_COMPLETE", "rl", f"RL policy ({rlrec.policy_mode}) evaluated "
                  f"{len(rlrec.valid_actions)} valid action(s): {rlrec.high_level_action}")
        out = rlrec.model_dump()
        out["state"] = st.to_dict()
        out["drift"] = drift
        # RL-selected index candidate goes first (priority for sandbox + migration)
        if rlrec.high_level_action == "ADD_INDEX" and rlrec.selected_candidate:
            sel = rlrec.selected_candidate.get("id")
            idx = rec["proposed_indexes"]
            if isinstance(sel, int) and 0 <= sel < len(idx):
                idx[sel]["source"] = idx[sel]["source"] + "+rl_selected"
                idx.insert(0, idx.pop(sel))
        return {"rl_recommendation": out, "agent_recommendation": rec}

    async def n_validator(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        rec = state["agent_recommendation"]
        validator = RecommendationValidator(self.s.obfuscator, tr["schema_real"],
                                            getattr(self.s, "target_schema", "public"))
        outcomes, rejected = [], []
        for c in rec["proposed_indexes"]:
            if c["source"].startswith("index_strategist:redundant"):
                rejected.append({"candidate": c["ddl"], "errors": ["redundant with an existing index (masked by RL)"]})
                continue
            o = validator.validate(IndexCandidate(**c), rid)
            outcomes.append(o)
            if not o.valid:
                rejected.append({"candidate": c["ddl"], "errors": o.errors})
        tr["outcomes"] = outcomes
        rewrite_real = None
        rw_note = None
        rw = rec.get("rewrite")
        rl = state.get("rl_recommendation") or {}
        if rw:
            try:
                rewrite_real = self.s.obfuscator.translate_llm_sql_to_real(rw["rewritten_masked_sql"], rid)
                SecurityGateway(self.s.settings.max_query_length).validate(rewrite_real)
            except QuerySentinelError as exc:
                rewrite_real, rw_note = None, f"rewrite rejected: {exc.message}"
        tr["rewrite_real"] = rewrite_real
        valid = [o for o in outcomes if o.valid]
        self.emit(rid, "VALIDATION_COMPLETE", "validator",
                  f"Validation: {len(valid)} valid index candidate(s), {len(rejected)} rejected"
                  + ("; rewrite translated" if rewrite_real else ""))
        status = {"recommendation_validator": "COMPLETED" if (valid or rewrite_real) else "REJECTED"}
        if not valid and not rewrite_real:
            status.update({"hypopg_sandbox": "SKIPPED", "blast_radius": "SKIPPED"})
            self.emit(rid, "AGENT_SKIPPED", "validator", "HypoPG simulation skipped: no valid candidate")
        return {"validation": {"valid_indexes": [{"candidate": o.candidate.ddl, "kind": o.candidate.kind,
                                                  "warnings": o.warnings} for o in valid],
                               "rejected": rejected, "rewrite_real_available": bool(rewrite_real),
                               "rewrite_note": rw_note, "rl_action": rl.get("high_level_action")},
                "agent_status": status}

    async def n_sandbox(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.s.firewall.require(rid, "supervisor", Capability.RUN_HYPOPG, "sandbox")
        self.emit(rid, "SANDBOX_STARTED", "sandbox", "HypoPG simulation started")
        sb = HypoPGSandbox(self.s.db, self.s.obfuscator, tr["budget"])
        res = await sb.run(rid, tr["validated"].sql, tr["outcomes"], tr.get("rewrite_real"),
                           generic=tr["validated"].has_params)
        self.s.audit.record(rid, "SANDBOX_EXPERIMENT", stage="sandbox",
                            details={"status": res.status, "before": res.before_total_cost, "after": res.after_total_cost,
                                     "pct": res.planner_cost_reduction_pct, "cleanup_ok": res.cleanup_ok})
        self.emit(rid, "SANDBOX_COMPLETE", "sandbox",
                  f"HypoPG simulation {res.status}: planner-estimated cost reduction "
                  f"{res.planner_cost_reduction_pct if res.planner_cost_reduction_pct is not None else 'n/a'}%",
                  status=res.status)
        status = {"hypopg_sandbox": "COMPLETED" if res.status == "completed" else "ERROR" if res.status == "failed" else "SKIPPED"}
        if not res.virtual_indexes_used:
            status["blast_radius"] = "SKIPPED"
        return {"sandbox_results": res.model_dump(mode="json"), "agent_status": status}

    async def n_blast(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        self.emit(rid, "BLAST_RADIUS_STARTED", "blast_radius", "Blast-radius analysis started")
        sb = state["sandbox_results"]
        used = set(sb.get("virtual_indexes_used") or [])
        sizes = {v["candidate"]: v["estimated_size_mb"] for v in sb.get("virtual_indexes_created", [])}
        analyzer = BlastRadiusAnalyzer(self.s.instr, self.s.obfuscator)
        statements = []
        try:
            statements = await analyzer.fetch_statements()
        except QuerySentinelError:
            statements = []
        results = []
        for o in tr["outcomes"]:
            if not o.valid or o.candidate.ddl not in used:
                continue
            parts = real_index_parts(o.real_ddl)
            size_b = (sizes.get(o.candidate.ddl) or 0) * 1048576
            br = analyzer.analyze_with(statements, rid, parts["table"], parts["columns"], tr["schema_real"],
                                       size_b, parts["include"])
            br.target_table = self.s.obfuscator.to_masked_identifier(rid, parts["table"]) or "(masked)"
            d = br.model_dump(mode="json")
            d["candidate"] = o.candidate.ddl
            results.append(d)
        # --- write latency in ms, measured on a throwaway temp copy (transaction rolled back)
        by_table: dict[str, list[dict]] = {}
        for o in tr["outcomes"]:
            if o.valid and o.candidate.ddl in used:
                parts = real_index_parts(o.real_ddl)
                by_table.setdefault(parts["table"], []).append(parts)
        sim = StructuralSimulator(self.s.db)
        write_lat = []
        for table, parts_list in by_table.items():
            wl = await sim.write_latency(table, parts_list)
            wl["target_table"] = self.s.obfuscator.to_masked_identifier(rid, table) or "(masked)"
            write_lat.append(wl)
            self.emit(rid, "WRITE_LATENCY_SIMULATED", "blast_radius",
                      f"Write-latency simulation {wl['status']}: "
                      + (f"+{wl['added_ms_per_1000_rows']} ms per 1,000 inserted rows ({wl['write_latency_change_pct']:+}%)"
                         if wl["status"] == "completed" else wl.get("reason", "")))
        for r in results:
            wl = next((w for w in write_lat if w["target_table"] == r.get("target_table")), None)
            if wl:
                r["write_latency_simulation"] = wl
        worst = max((r["risk_level"] for r in results), key=["low", "medium", "high"].index, default="low")
        self.s.audit.record(rid, "BLAST_RADIUS", stage="blast_radius", details={"indexes": len(results), "risk": worst})
        self.emit(rid, "BLAST_RADIUS_COMPLETE", "blast_radius", f"Blast-radius analysis complete (risk {worst})")
        sb = dict(sb)
        sb["estimated_write_overhead"] = {"risk_level": worst, "per_index": [
            {"candidate": r["candidate"], **r["estimated_write_impact"]} for r in results]}
        sb["write_latency_simulation"] = write_lat
        return {"blast_radius": {"results": results, "overall_risk": worst}, "sandbox_results": sb}

    async def n_final(self, state: QSState) -> dict:
        rid = state["request_id"]
        tr = self.t(rid)
        if state.get("error"):
            status = {k: ("SKIPPED" if v in ("QUEUED", "RUNNING") else v) for k, v in (state.get("agent_status") or {}).items()}
            status["final_report"] = "REJECTED" if state["error"].get("error") in ("unsafe_sql", "invalid_sql", "quarantined") else "ERROR"
            return {"agent_status": status, "final_report": None}
        self.s.firewall.require(rid, "deployment_generator", Capability.GENERATE_MIGRATION, "migration")
        rec = state.get("agent_recommendation") or {}
        sb = state.get("sandbox_results") or {}
        used = set(sb.get("virtual_indexes_used") or [])
        per = {p["candidate"]: p for p in sb.get("per_candidate", [])}
        ddls = []
        for o in tr.get("outcomes", []):
            if o.valid and o.candidate.ddl in used:
                ddls.append({"real_ddl": o.real_ddl, "rationale": o.candidate.reason,
                             "planner_cost_reduction_pct": (per.get(o.candidate.ddl) or {}).get("planner_cost_reduction_pct")})
        dep = self.s.migrations.generate(request_id=rid, query_fingerprint=tr.get("query_fingerprint", ""),
                                         validated_ddls=ddls, partition=rec.get("partitioning_recommendation"),
                                         sharding=rec.get("sharding_recommendation"),
                                         rewrite_sql_masked=rec.get("rewritten_sql"),
                                         schema=getattr(self.s, "target_schema", "public"))
        self.s.audit.record(rid, "MIGRATION_GENERATION", stage="deployment",
                            details={"migration_id": dep.migration_id, "indexes": len(ddls)})
        self.emit(rid, "MIGRATION_GENERATED", "deployment", f"Staging migration {dep.migration_id} generated (human review required)")
        blast = (state.get("blast_radius") or {}).get("results", [])
        high_risk = any(b.get("risk_level") == "high" for b in blast)
        chain = build_chain(plan=state.get("plan_analysis"), gnn=state.get("gnn_diagnosis"),
                            access=state.get("access_pattern"), schema_findings=state.get("schema_agent"),
                            workload=state.get("workload_context"), recommendation=rec,
                            rl=state.get("rl_recommendation"), sandbox=sb or None, blast=blast,
                            deployment=dep.model_dump())
        part_sim = await self._simulate_partitioning(rid, rec.get("partitioning_recommendation"))
        proof = self._zero_exposure_proof(state)
        budget: OptimizationBudget = tr["budget"]
        if budget.exhausted:
            self.emit(rid, "SEARCH_BUDGET_EXHAUSTED", "budget", budget.snapshot()["statement"])
            self.s.audit.record(rid, "BUDGET_EXHAUSTED", stage="budget", details=budget.snapshot())
        gnn = state.get("gnn_diagnosis") or {}
        rl = state.get("rl_recommendation") or {}
        limitations = [
            "Planner-estimated cost reductions come from EXPLAIN/HypoPG and are NOT runtime speedups.",
            "Actual runtime requires the opt-in benchmark in an isolated environment.",
            "All recommendations require DBA review; nothing is deployed automatically.",
        ]
        if gnn.get("model_mode") == "analytic_fallback":
            limitations.append("GNN checkpoint unavailable: deterministic analytic fallback used.")
        if rl.get("policy_mode") == "analytic_fallback":
            limitations.append("RL policy unavailable: deterministic fallback policy used.")
        if not self.s.llm.enabled:
            limitations.append("LLM disabled: deterministic agents only (no generative proposals).")
        if rec.get("partitioning_recommendation") or rec.get("sharding_recommendation"):
            limitations.append("Partitioning/sharding advice: Advisory - Requires isolated design/benchmark validation.")
        if tr["validated"].has_params:
            limitations.append("Parameterised query analysed with EXPLAIN (GENERIC_PLAN): costs are for a generic plan "
                               "with default selectivity estimates and cannot reflect data skew (e.g. hot tenants). "
                               "Re-run with representative literal values for a value-specific diagnosis.")
        if budget.exhausted:
            limitations.append(budget.snapshot()["statement"])
        if high_risk:
            limitations.append("HIGH write-impact risk: staging benchmark of write workload required before deployment.")
        if sb.get("combination_note"):
            limitations.append(sb["combination_note"])
        rw = rec.get("rewrite")
        if rw and not rw.get("semantics_verified"):
            limitations.append("SQL rewrite semantics could not be fully verified: mandatory human review.")
        report = FinalOptimizationReport(
            request_id=rid, query_fingerprint=tr.get("query_fingerprint", ""),
            privacy_audit={**(state.get("privacy") or {}).get("privacy_audit", {}),
                           "literal_count": (state.get("privacy") or {}).get("literal_count"),
                           "privacy_bitmask": (state.get("privacy") or {}).get("privacy_bitmask")},
            diagnosis={"primary_bottleneck": gnn.get("primary_bottleneck_type"),
                       "model_mode": gnn.get("model_mode"), "confidence": gnn.get("confidence"),
                       "summary": gnn.get("natural_language_summary"),
                       "plan_fingerprint": (state.get("plan_analysis") or {}).get("plan_fingerprint"),
                       "evidence": gnn.get("evidence", [])},
            optimization={"rewritten_sql": rec.get("rewritten_sql"), "proposed_indexes": rec.get("proposed_indexes", []),
                          "partitioning_recommendation": rec.get("partitioning_recommendation"),
                          "sharding_recommendation": rec.get("sharding_recommendation"),
                          "extended_statistics": rec.get("extended_statistics", []),
                          "rl_action": {k: rl.get(k) for k in ("high_level_action", "low_level_action", "policy_mode", "explanation")} if rl else None},
            validation={"hypopg_result": sb, "planner_cost_reduction_pct": sb.get("planner_cost_reduction_pct"),
                        "measured_execution_time_reduction_pct": None,
                        "measurement_note": "not measured (BENCHMARK_ENABLED=false or not requested)",
                        "validator": state.get("validation")},
            blast_radius=blast,
            deployment={**dep.model_dump(), "high_risk": high_risk,
                        "rewritten_sql_for_review": tr.get("rewrite_real"),
                        "rewritten_sql_label": "INTERNAL - real identifiers/literals, DBA review only" if tr.get("rewrite_real") else None},
            evidence_chain=chain.items,
            explanation=_evidence_explanation(state, sb, blast),
            model_modes={"gnn": gnn.get("model_mode"), "rl": rl.get("policy_mode"),
                         "llm": {"provider": self.s.llm.provider, "model": self.s.llm.model,
                                 "enabled": self.s.llm.enabled, "calls": state.get("llm_calls") or []}},
            budget=budget.snapshot(), workload_context={**(state.get("workload_context") or {}),
                                                        "regression": state.get("regression"),
                                                        "drift": (rl.get("drift") if rl else None)},
            limitations=limitations,
            structural_simulation={"partitioning": part_sim,
                                   "write_latency": sb.get("write_latency_simulation") or []},
            zero_exposure_proof=proof,
            answer=_headline(state, sb, part_sim, proof))
        rep = report.model_dump(mode="json")
        rep["answer"]["sentence_dba"] = self._unmask_text(rid, rep["answer"]["sentence"])
        rep["answer"]["sentence_dba_label"] = "DBA view - real names restored on the trusted backend; never sent to AI"
        rep["anonymized_sql"] = state.get("masked_sql")
        status = {"final_report": "COMPLETED"}
        self.s.audit.record(rid, "FINAL_RECOMMENDATION", stage="final",
                            details={"indexes": len(ddls), "planner_pct": sb.get("planner_cost_reduction_pct"),
                                     "high_risk": high_risk})
        return {"final_report": rep, "deployment": dep.model_dump(), "agent_status": status}

    async def _simulate_partitioning(self, rid: str, advice: dict | None) -> dict | None:
        if not advice or advice.get("strategy_type") not in ("range", "hash", "list"):
            return None
        table = self.s.obfuscator.original_identifier(rid, advice["table"])
        key = self.s.obfuscator.original_identifier(rid, advice["candidate_key"])
        if not table or not key:
            return {"status": "skipped", "reason": "partition target not resolvable"}
        self.emit(rid, "PARTITION_SIMULATION_STARTED", "sandbox", "Partitioning simulation on a temp sample copy started")
        strategy = "range" if advice["strategy_type"] == "range" else "hash"
        res = await StructuralSimulator(self.s.db).partitioning(table, key, strategy)
        res.update(table=advice["table"], key=advice["candidate_key"])
        self.s.audit.record(rid, "PARTITION_SIMULATION", stage="sandbox",
                            details={k: res.get(k) for k in ("status", "partitions_total", "partitions_scanned",
                                                             "planner_cost_reduction_pct")})
        self.emit(rid, "PARTITION_SIMULATION_COMPLETE", "sandbox",
                  f"Partitioning simulation {res['status']}: "
                  + (f"{res['partitions_scanned']}/{res['partitions_total']} partitions scanned, planner cost "
                     f"-{res['planner_cost_reduction_pct']}% on the sample" if res["status"] == "completed"
                     else res.get("reason", "")))
        return res

    def _unmask_text(self, rid: str, text: str) -> str:
        return re.sub(r"\b(?:tbl|col|als)_[0-9a-f]{6,12}\b",
                      lambda m: self.s.obfuscator.original_identifier(rid, m.group(0)) or m.group(0), text)

    def _zero_exposure_proof(self, state: QSState) -> dict:
        """Re-check the exact payload an AI model would receive (outgoing guard), and report coverage."""
        rid = state["request_id"]
        priv = state.get("privacy") or {}
        pa = priv.get("privacy_audit") or {}
        payload = self._llm_data(state)
        rep = self.s.guard.check(rid, "", payload)
        lits = priv.get("literal_count") or 0
        idents = len(priv.get("mapped_tables") or {}) + len(priv.get("mapped_columns") or {})
        leaked = bool(pa.get("literal_leak_detected")) or not rep.allowed
        return {"zero_raw_data_exposed": not leaked,
                "literals_total": lits, "literals_masked": 0 if leaked else lits,
                "identifiers_total": idents, "identifiers_hashed": idents,
                "masked_pct": 100.0 if not leaked else None,
                "pii_types_found_and_masked": pa.get("pii_types", []),
                "ai_payload_check": rep.as_dict(),
                "statement": ("Zero raw data was exposed during this analysis: "
                              f"{lits}/{lits} literal values parameterised, {idents}/{idents} table/column names "
                              "replaced by HMAC hashes, and the outgoing AI payload passed the leak check."
                              if not leaked else "Raw-data leak check FAILED - payload blocked, nothing sent to AI.")}

    # ================================================================== run
    async def run(self, request_id: str, raw_sql: str) -> dict[str, Any]:
        self.s.events.ensure(request_id)
        budget = OptimizationBudget.from_settings(
            on_exhausted=lambda key, snap: self.s.audit.record(request_id, "BUDGET_EXHAUSTED", stage="budget",
                                                               details={"key": key}))
        self.s.trusted[request_id] = {"raw_sql": raw_sql, "budget": budget}
        self.s.audit.record(request_id, "REQUEST_RECEIVED", stage="api", details={"length": len(raw_sql)})
        metrics.inc("requests")
        t_run = time.perf_counter()
        self.emit(request_id, "REQUEST_RECEIVED", "api", "Optimization request received")
        try:
            final = await self.graph.ainvoke(initial_state(request_id))
            response = build_response(final)
            if final.get("error"):
                if final["error"].get("error") not in ("unsafe_sql", "invalid_sql", "quarantined"):
                    self.emit(request_id, "ERROR", final["error"].get("stage") or "pipeline",
                              f"Pipeline stopped: {final['error'].get('message')}")
                await self.s.persist_run({"request_id": request_id}, "failed", final["error"].get("error"))
            else:
                await self.s.persist_run(final["final_report"], "completed")
                fp = self.t(request_id).get("query_fingerprint")
                self.emit(request_id, "FINAL_REPORT_READY", "final", "Final report ready",
                          planner_cost_reduction_pct=(final.get("sandbox_results") or {}).get("planner_cost_reduction_pct"))
                del fp
            metrics.inc(f"requests.{response['status']}")
            metrics.observe("pipeline_total", (time.perf_counter() - t_run) * 1000)
            if response["status"] == "completed":
                metrics.inc(f"gnn_mode.{(response.get('gnn_analysis') or {}).get('model_mode')}")
                metrics.inc(f"rl_mode.{(response.get('rl_recommendation') or {}).get('policy_mode')}")
            self.s.reports[request_id] = response
            return response
        finally:
            self.s.obfuscator.clear_vault(request_id)
            self.s.trusted.pop(request_id, None)


def _evidence_explanation(state: QSState, sb: dict, blast: list[dict]) -> dict[str, Any]:
    """Section 68 format: observation -> evidence -> recommendation -> validation -> impact -> uncertainty."""
    plan = state.get("plan_analysis") or {}
    hc = plan.get("highest_cost_node") or {}
    rec = state.get("agent_recommendation") or {}
    idx = rec.get("proposed_indexes") or []
    return {
        "observation": f"{hc.get('node_type')} {('on ' + hc['relation_name']) if hc.get('relation_name') else ''} "
                       f"dominates the plan ({hc.get('cost_share')} of planner cost)".strip(),
        "evidence": {"total_planner_cost": plan.get("total_cost"), "plan_rows": hc.get("plan_rows"),
                     "filter_pattern": hc.get("filter"), "estimated_loops": hc.get("estimated_loops"),
                     "no_matching_composite_index": any(i.get("kind") == "composite" and not i.get("source", "").endswith("redundant") for i in idx)},
        "recommendation": idx[0]["ddl"] if idx else (rec.get("rewritten_sql") and "SQL rewrite") or "no structural change",
        "validation": (f"HypoPG changed the planner plan/cost: {sb.get('before_total_cost')} -> {sb.get('after_total_cost')}"
                       if sb.get("status") == "completed" else f"sandbox {sb.get('status', 'not run')}"),
        "operational_impact": [f"{b.get('target_table')}: write risk {b.get('risk_level')}, ~{b.get('estimated_storage_impact_mb')} MB"
                               for b in blast] or ["no index change simulated"],
        "uncertainty": "Actual runtime requires benchmark; planner estimates depend on statistics and configuration.",
    }


def _fmt_pct(v) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:.2f}%" if v >= 99.5 else f"{v:.1f}%"


def _headline(state: QSState, sb: dict, part: dict | None, proof: dict) -> dict:
    """One evidence-backed answer in the style of the problem statement (masked identifiers)."""
    rec = state.get("agent_recommendation") or {}
    gnn = state.get("gnn_diagnosis") or {}
    actions = []
    rw_pct = sb.get("rewrite_cost_reduction_pct")
    if rec.get("rewritten_sql") and isinstance(rw_pct, (int, float)) and rw_pct > 1:
        actions.append("rewriting the SQL")
    for ddl in (sb.get("virtual_indexes_used") or [])[:2]:
        m = re.search(r"ON\s+(\S+)\s*(?:USING\s+\w+\s*)?\(([^)]*)\)", ddl)
        kind = "a composite index" if m and "," in m.group(2) else "an index"
        actions.append(f"adding {kind} on {m.group(1)}({m.group(2)})" if m else "adding the proposed index")
    best = max([v for v in (sb.get("planner_cost_reduction_pct"), rw_pct) if isinstance(v, (int, float))], default=None)
    wl = [w for w in (sb.get("write_latency_simulation") or []) if w.get("status") == "completed"]
    storage = sb.get("estimated_storage_impact_mb")
    parts = []
    if not actions:
        parts.append("No index or rewrite is needed: the plan is already efficient for this query "
                     f"({gnn.get('primary_bottleneck_type', 'no bottleneck')}).")
    else:
        act = ", ".join(actions[:-1]) + " and " + actions[-1] if len(actions) > 1 else actions[0]
        txt = act[0].upper() + act[1:] + (f" is estimated to cut query cost by {_fmt_pct(best)}" if best is not None
                                          else " is expected to help")
        extras = []
        if wl:
            add = sum(w["added_ms_per_1000_rows"] for w in wl)
            per_row = sum(w["added_ms_per_row"] for w in wl)
            if all(w.get("within_measurement_noise") for w in wl):
                extras.append("no measurable extra write latency in the simulation")
            else:
                extras.append(f"only +{add:.2f} ms simulated write latency per 1,000 inserted rows "
                              f"(~{per_row:.4f} ms per single-row write)")
        if storage:
            extras.append(f"+{storage:.0f} MB of index storage")
        if extras:
            txt += ", with " + " and ".join(extras)
        parts.append(txt + ".")
    if part and part.get("status") == "completed":
        unit = "month" if part["strategy"] == "range" else f"{part['partitions_total']} hash buckets"
        shape = "date-window queries" if part["strategy"] == "range" else "single-key lookups"
        parts.append(f"Optionally, partitioning {part['table']} by {unit} on {part['key']} would let {shape} scan "
                     f"{part['partitions_scanned']} of {part['partitions_total']} partitions "
                     f"(-{_fmt_pct(part.get('planner_cost_reduction_pct'))} cost on a simulated sample copy).")
    parts.append(proof.get("statement", ""))
    return {"sentence": " ".join(p for p in parts if p),
            "bottleneck": gnn.get("primary_bottleneck_type"),
            "planner_cost_reduction_pct": best, "write_latency": wl, "storage_mb": storage,
            "partition_simulation": part, "zero_raw_data_exposed": proof.get("zero_raw_data_exposed"),
            "metric_note": "Cost figures are PostgreSQL planner estimates (HypoPG/EXPLAIN), not stopwatch runtime."}


def build_response(final: QSState) -> dict[str, Any]:
    """Section 39 response contract (+ dashboard fields)."""
    if final.get("error"):
        return {"request_id": final["request_id"], "status": "rejected" if final["error"].get("error") in
                ("unsafe_sql", "invalid_sql", "quarantined") else "error",
                "error": final["error"], "agent_status": final.get("agent_status"),
                "agent_logs": final.get("agent_logs"), "privacy_audit": (final.get("privacy") or {}).get("privacy_audit")}
    rep = final.get("final_report") or {}
    dep = final.get("deployment") or {}
    return {
        "request_id": final["request_id"],
        "status": "completed",
        "privacy_audit": final.get("privacy"),
        "schema_analysis": {"anonymized_schema": final.get("schema_stats"), "schema_expert": final.get("schema_agent")},
        "plan_analysis": final.get("plan_analysis"),
        "gnn_analysis": final.get("gnn_diagnosis"),
        "agent_recommendation": final.get("agent_recommendation"),
        "rl_recommendation": final.get("rl_recommendation"),
        "sandbox_simulation": final.get("sandbox_results"),
        "blast_radius": final.get("blast_radius"),
        "deployment": {"migration_sql": dep.get("migration_sql"), "rollback_sql": dep.get("rollback_sql"),
                       "requires_human_review": True, **{k: v for k, v in (rep.get("deployment") or {}).items()
                                                          if k not in ("migration_sql", "rollback_sql")}},
        "limitations": rep.get("limitations", []),
        "final_report": rep,
        "answer": rep.get("answer"),
        "zero_exposure_proof": rep.get("zero_exposure_proof"),
        "structural_simulation": rep.get("structural_simulation"),
        "access_pattern": final.get("access_pattern"),
        "workload_context": final.get("workload_context"),
        "agent_status": final.get("agent_status"),
        "agent_logs": final.get("agent_logs"),
    }
