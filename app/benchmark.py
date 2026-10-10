"""Optimization efficacy benchmarking + ablation lab (Section 36, 61 and Addendum R12).

Two clearly separated metrics - never converted into one another:
* planner_cost_reduction_pct        - EXPLAIN / HypoPG planner ESTIMATE (always available)
* measured_execution_time_reduction_pct - real timing, ONLY when BENCHMARK_ENABLED=true AND a
  disposable database DSN (BENCHMARK_ADMIN_DSN) is configured. Indexes are then built for real
  INSIDE a transaction that is always rolled back, and timings come from EXPLAIN ANALYZE.

Modes: baseline, index_only, rewrite_only, gnn_assisted, rl_assisted, full
Ablations: without_gnn, without_rl, without_workload_context, without_hypopg,
           without_predicate_intelligence, without_blast_radius
No improvement target is hard-coded; every number is computed from the environment.
"""
from __future__ import annotations

import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.db import Database, explain_on
from app.errors import ErrorCategory, QuerySentinelError
from app.gnn.analyzer import GNNAnalyzer
from app.masking_vault import DataObfuscator
from app.plan_parser import parse_explain, plan_fingerprint
from app.postgres_instrumentation import PostgresInstrumentation
from app.rl import fallback_policy
from app.rl.environment import mask_actions
from app.rl.episode_builder import Prepared, prepare
from app.rl.policy import RLPolicy
from app.sandbox import MIN_USEFUL_REDUCTION_PCT, cost_reduction_pct

MODES = ["baseline", "index_only", "rewrite_only", "gnn_assisted", "rl_assisted", "full"]
ABLATIONS = ["without_gnn", "without_rl", "without_workload_context", "without_hypopg",
             "without_predicate_intelligence", "without_blast_radius"]


class BenchmarkRunner:
    def __init__(self, db: Database, obfuscator: DataObfuscator, gnn: GNNAnalyzer, rl: RLPolicy):
        self.db = db
        self.instr = PostgresInstrumentation(db)
        self.obf = obfuscator
        self.gnn = gnn
        self.rl = rl
        self.settings = get_settings()

    # ------------------------------------------------------------------ planner-estimate experiments
    async def _cost(self, conn, sql: str, ddls: list[str]) -> tuple[float, str, list[str]]:
        await conn.execute("SELECT hypopg_reset()")
        names = {}
        for d in ddls:
            row = await conn.fetchrow("SELECT * FROM hypopg_create_index($1)", d)
            names[row["indexname"]] = d
        plan = parse_explain(await explain_on(conn, sql))
        used = [names[n.index_name] for n in plan.nodes if n.index_name in names]
        await conn.execute("SELECT hypopg_reset()")
        return plan.total_cost, plan_fingerprint(plan.nodes), used

    async def _best_index_set(self, conn, sql: str, ddls: list[str]) -> list[str]:
        """Keep each candidate only if it alone lowers planner cost >= MIN_USEFUL_REDUCTION_PCT."""
        base, _, _ = await self._cost(conn, sql, [])
        keep = []
        for d in ddls:
            c, _, used = await self._cost(conn, sql, [d])
            if used and (cost_reduction_pct(base, c) or 0) >= MIN_USEFUL_REDUCTION_PCT:
                keep.append(d)
        return keep

    def _variant_choice(self, p: Prepared, variant: str) -> tuple[list[str], str | None, str]:
        """-> (candidate real DDLs, rewrite rule or None, note)"""
        valid_ids = [a.payload["id"] for a in p.actions if a.high == "ADD_INDEX" and not a.redundant
                     and a.payload["id"] in p.real_ddl]
        all_ddls = [p.real_ddl[i] for i in valid_ids]
        verified_rw = [a.payload["id"] for a in p.actions if a.high == "REWRITE_SQL" and a.verified]
        rw = verified_rw[0] if verified_rw else None
        bottleneck = p.diag.bottleneck_class_id != 3
        if variant == "baseline":
            return [], None, "no change"
        if variant == "index_only":
            return all_ddls, None, "all validated index candidates (HypoPG-filtered)"
        if variant == "rewrite_only":
            return [], rw, "verified rewrite only"
        if variant == "gnn_assisted":
            return (all_ddls if bottleneck else []), None, "indexes only when GNN diagnoses a bottleneck"
        if variant in ("rl_assisted", "without_gnn", "without_workload_context", "without_blast_radius", "full",
                       "without_rl", "without_hypopg", "without_predicate_intelligence"):
            pass
        if variant == "rl_assisted":
            valid, _ = mask_actions(p.actions, set())
            rec = self.rl.recommend(p.state, p.actions)
            if rec.high_level_action == "ADD_INDEX" and rec.selected_candidate:
                return [p.real_ddl.get(rec.selected_candidate["id"])] if rec.selected_candidate["id"] in p.real_ddl else [], None, \
                    f"RL ({rec.policy_mode}) single action"
            if rec.high_level_action == "REWRITE_SQL" and rec.selected_candidate:
                return [], rec.selected_candidate.get("rule"), f"RL ({rec.policy_mode}) rewrite"
            return [], None, f"RL ({rec.policy_mode}) chose {rec.high_level_action}"
        if variant == "without_rl":
            valid, _ = mask_actions(p.actions, set())
            a, _why = fallback_policy.choose(p.state, valid)
            if a.high == "ADD_INDEX":
                return [p.real_ddl[a.payload["id"]]] if a.payload["id"] in p.real_ddl else [], None, "rule policy instead of RL"
            if a.high == "REWRITE_SQL":
                return [], a.payload.get("rule"), "rule policy instead of RL"
            return [], None, "rule policy: no change"
        if variant == "without_gnn":
            return all_ddls, rw, "full pipeline routing without the GNN gate"
        if variant == "without_hypopg":
            return all_ddls[:1], rw, "top strategist candidate WITHOUT HypoPG filtering (unvalidated)"
        if variant == "without_predicate_intelligence":
            singles = [p.real_ddl[a.payload["id"]] for a in p.actions if a.high == "ADD_INDEX" and not a.redundant
                       and a.payload.get("kind") == "single" and a.payload["id"] in p.real_ddl]
            return singles, rw, "single-column candidates only (no role/co-occurrence analysis)"
        if variant == "without_workload_context":
            return all_ddls, rw, "same candidates; workload weighting only affects RL reward/ranking"
        if variant == "without_blast_radius":
            return all_ddls, rw, "same planner result; write-risk is NOT assessed"
        return all_ddls, rw, "full Planora pipeline (indexes filtered by HypoPG + verified rewrite)"

    async def run(self, sql: str, request_id: str, variants: list[str] | None = None,
                  measure: bool = False) -> dict[str, Any]:
        variants = variants or (MODES + ABLATIONS)
        p = await prepare(request_id, sql, self.instr, self.obf, self.gnn)
        version = (await self.db.fetch("SELECT current_setting('server_version') AS v"))[0]["v"]
        rows = await self.instr.tables(sorted(p.vq.referenced_relations))
        settings_rows = await self.db.fetch(
            "SELECT name, setting FROM pg_settings WHERE name IN ('shared_buffers','work_mem','effective_cache_size',"
            "'random_page_cost','max_parallel_workers_per_gather','jit')")
        results = []
        self._chosen: dict[str, tuple[list[str], str | None]] = {}
        async with self.db.read_only() as conn:
            base_cost, base_fp, _ = await self._cost(conn, p.vq.sql, [])
            for v in variants:
                ddls, rule, note = self._variant_choice(p, v)
                ddls = [d for d in ddls if d]
                if v not in ("without_hypopg", "baseline") and ddls:
                    ddls = await self._best_index_set(conn, p.vq.sql if not rule else p.real_sql[rule], ddls)
                q = p.real_sql.get(rule, p.vq.sql) if rule else p.vq.sql
                self._chosen[v] = (ddls, rule)
                cost, fp, used = await self._cost(conn, q, ddls)
                results.append({"variant": v, "note": note, "indexes": len(ddls), "indexes_used_by_planner": len(used),
                                "rewrite": rule, "planner_cost": cost, "plan_fingerprint": fp,
                                "planner_cost_reduction_pct": cost_reduction_pct(base_cost, cost),
                                "write_risk_assessed": v != "without_blast_radius",
                                "validated_by_hypopg": v != "without_hypopg"})
        measured = await self._measure(p, results) if measure else {
            "measured": False, "reason": "measurement not requested"}
        return {
            "query_fingerprint": p.fingerprint,
            "environment": {"postgresql_version": version, "settings": {r["name"]: r["setting"] for r in settings_rows},
                            "dataset": {r["table_name"]: int(r["estimated_rows"]) for r in rows},
                            "gnn_mode": self.gnn.mode, "rl_mode": self.rl.mode,
                            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "baseline_plan_fingerprint": base_fp,
            "baseline_planner_cost": base_cost,
            "results": results,
            "measurement": measured,
            "metric_separation": {"planner_cost_reduction_pct": "EXPLAIN/HypoPG planner estimate",
                                  "measured_execution_time_reduction_pct": "EXPLAIN ANALYZE timing in a disposable DB only"},
            "caveat": "Results depend on PostgreSQL version, statistics, hardware, data distribution, configuration "
                      "and cache state.",
        }

    # ------------------------------------------------------------------ measured mode (opt-in, disposable DB)
    async def _measure(self, p: Prepared, results: list[dict]) -> dict[str, Any]:
        s = self.settings
        dsn = s.benchmark_admin_dsn.get_secret_value()
        if not s.benchmark_enabled:
            return {"measured": False, "reason": "BENCHMARK_ENABLED=false"}
        if not dsn:
            return {"measured": False, "reason": "BENCHMARK_ADMIN_DSN not configured (disposable database required)"}
        import asyncpg
        full = next((r for r in results if r["variant"] == "full"), None)
        if full is None:
            return {"measured": False, "reason": "full variant not run"}
        ddls, rule = self._chosen["full"]      # identical (HypoPG-filtered) set as the planner run
        conn = await asyncpg.connect(dsn)
        try:
            async def timed(sql: str) -> list[float]:
                out = []
                for _ in range(s.benchmark_repetitions):
                    doc = await conn.fetchval("EXPLAIN (ANALYZE TRUE, FORMAT JSON) " + sql)
                    import json as _json
                    d = _json.loads(doc) if isinstance(doc, str) else doc
                    out.append(float(d[0]["Execution Time"]))
                return out
            t0 = time.time()
            before = await timed(p.vq.sql)
            tx = conn.transaction()
            await tx.start()
            try:
                for d in ddls:
                    await conn.execute(d)          # real index, rolled back below
                await conn.execute("ANALYZE")
                after = await timed(p.real_sql.get(rule, p.vq.sql) if rule else p.vq.sql)
            finally:
                await tx.rollback()
            b, a = statistics.median(before), statistics.median(after)
            return {"measured": True, "repetitions": s.benchmark_repetitions, "before_ms_median": round(b, 3),
                    "after_ms_median": round(a, 3), "before_ms_runs": before, "after_ms_runs": after,
                    "measured_execution_time_reduction_pct": cost_reduction_pct(b, a),
                    "indexes_built_in_rolled_back_transaction": len(ddls), "rewrite": rule,
                    "wall_seconds": round(time.time() - t0, 1),
                    "note": "warm-cache medians; first run may include cold-cache effects"}
        except Exception as exc:  # noqa: BLE001
            raise QuerySentinelError(ErrorCategory.SANDBOX_FAILURE, f"benchmark failed ({type(exc).__name__})",
                                     stage="benchmark") from None
        finally:
            await conn.close()


DEMO_WORKLOAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "slow_queries.sql")


def load_demo_queries(path: str = DEMO_WORKLOAD_PATH) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        txt = "\n".join(line for line in fh.read().splitlines() if not line.strip().startswith("--"))
    return [q.strip() for q in txt.split(";") if q.strip()]
