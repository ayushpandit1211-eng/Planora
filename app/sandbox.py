"""HypoPG simulation sandbox (Section 31).

Workflow: 1 baseline EXPLAIN -> 2 translate + validate candidate DDL (done by the validator) ->
3 create hypothetical index(es) with `SELECT * FROM hypopg_create_index($1)` -> 4 EXPLAIN again
-> 5 compare planner costs -> 6 capture hypothetical index metadata -> 7 cleanup with
hypopg_reset(), ALWAYS attempted (finally block) even when the experiment fails.

Everything runs on ONE connection inside a READ ONLY transaction (hypothetical indexes are
backend-local). planner_cost_reduction_pct = ((before - after) / before) * 100 is a PLANNER
ESTIMATE - never an actual runtime speedup.
"""
from __future__ import annotations

from typing import Any, Callable

import asyncpg

from app.contracts import SandboxResult, ValidationOutcome
from app.db import Database, explain_on
from app.errors import ErrorCategory, QuerySentinelError
from app.masking_vault import DataObfuscator
from app.optimization_budget import OptimizationBudget
from app.plan_parser import parse_explain, sanitize_plan


MIN_USEFUL_REDUCTION_PCT = 1.0   # a candidate must lower planner cost by >= 1 % on its own to be kept


def cost_reduction_pct(before: float | None, after: float | None) -> float | None:
    """((before - after) / before) * 100 with safe handling of zero / null costs."""
    if before is None or after is None or before <= 0:
        return None
    return round((before - after) / before * 100.0, 2)


class HypoPGSandbox:
    def __init__(self, db: Database, obfuscator: DataObfuscator, budget: OptimizationBudget | None = None):
        self.db = db
        self.obf = obfuscator
        self.budget = budget

    async def run(self, request_id: str, real_sql: str, outcomes: list[ValidationOutcome],
                  rewritten_real_sql: str | None = None, generic: bool = False,
                  on_event: Callable[[str, dict[str, Any]], None] | None = None) -> SandboxResult:
        valid = [o for o in outcomes if o.valid and o.real_ddl]
        if not valid and not rewritten_real_sql:
            return SandboxResult(before_total_cost=None, after_total_cost=None, planner_cost_reduction_pct=None,
                                 status="skipped", error="no valid index candidate or rewrite to simulate")
        result = SandboxResult(before_total_cost=None, after_total_cost=None, planner_cost_reduction_pct=None)
        cleanup_ok = False
        try:
            async with self.db.read_only() as conn:
                try:
                    await conn.execute("SELECT hypopg_reset()")
                except asyncpg.UndefinedFunctionError:
                    raise QuerySentinelError(ErrorCategory.HYPOPG_UNAVAILABLE, "HypoPG extension unavailable",
                                             stage="sandbox") from None
                try:
                    base = parse_explain(await explain_on(conn, real_sql, generic))
                    result.before_total_cost = base.total_cost
                    result.baseline_plan_fingerprint = sanitize_plan(base, self.obf, request_id).plan_fingerprint
                    # --- per-candidate isolated experiments (attribution)
                    created_meta: dict[str, dict[str, Any]] = {}
                    for o in valid:
                        if self.budget and not self.budget.consume("hypopg_experiments"):
                            result.per_candidate.append({"candidate": o.candidate.ddl, "status": "skipped: budget exhausted"})
                            continue
                        row = await conn.fetchrow("SELECT * FROM hypopg_create_index($1)", o.real_ddl)
                        size = await conn.fetchval("SELECT hypopg_relation_size($1)", row["indexrelid"])
                        plan = parse_explain(await explain_on(conn, real_sql, generic))
                        used = any(n.index_name == row["indexname"] for n in plan.nodes)
                        created_meta[o.candidate.ddl] = {"size_bytes": size}
                        result.per_candidate.append({
                            "candidate": o.candidate.ddl, "kind": o.candidate.kind, "signature": o.candidate.signature,
                            "before_cost": base.total_cost, "after_cost": plan.total_cost,
                            "planner_cost_reduction_pct": cost_reduction_pct(base.total_cost, plan.total_cost),
                            "used_by_planner": used, "estimated_size_mb": round(size / 1048576, 2),
                            "status": "simulated"})
                        await conn.execute("SELECT hypopg_reset()")
                    # --- combined experiment with every candidate the planner actually used alone
                    useful = [o for o in valid if any(pc.get("candidate") == o.candidate.ddl and pc.get("used_by_planner")
                                                      and (pc.get("planner_cost_reduction_pct") or 0) >= MIN_USEFUL_REDUCTION_PCT
                                                      for pc in result.per_candidate)]
                    hypo_names: dict[str, str] = {}
                    total_size = 0
                    for o in useful:
                        row = await conn.fetchrow("SELECT * FROM hypopg_create_index($1)", o.real_ddl)
                        hypo_names[row["indexname"]] = o.candidate.ddl
                        size = await conn.fetchval("SELECT hypopg_relation_size($1)", row["indexrelid"])
                        total_size += size
                        result.virtual_indexes_created.append({"candidate": o.candidate.ddl,
                                                               "estimated_size_mb": round(size / 1048576, 2)})
                    combined = parse_explain(await explain_on(conn, real_sql, generic)) if useful else base
                    result.after_total_cost = combined.total_cost
                    result.virtual_indexes_used = sorted({hypo_names[n.index_name] for n in combined.nodes
                                                          if n.index_name in hypo_names})
                    result.estimated_storage_impact_mb = round(total_size / 1048576, 2) if useful else 0.0
                    result.optimized_plan_fingerprint = sanitize_plan(combined, self.obf, request_id).plan_fingerprint
                    result.plan_changed = result.optimized_plan_fingerprint != result.baseline_plan_fingerprint
                    result.planner_cost_reduction_pct = cost_reduction_pct(result.before_total_cost, result.after_total_cost)
                    # --- rewrite experiment (EXPLAIN only; never executed)
                    if rewritten_real_sql:
                        await conn.execute("SELECT hypopg_reset()")
                        rw = parse_explain(await explain_on(conn, rewritten_real_sql, generic))
                        result.rewrite_before_cost = base.total_cost
                        result.rewrite_after_cost = rw.total_cost
                        result.rewrite_cost_reduction_pct = cost_reduction_pct(base.total_cost, rw.total_cost)
                        if useful:
                            for o in useful:
                                await conn.fetchrow("SELECT * FROM hypopg_create_index($1)", o.real_ddl)
                            both = parse_explain(await explain_on(conn, rewritten_real_sql, generic))
                            result.rewrite_with_indexes_cost = both.total_cost
                            best = min(result.after_total_cost or base.total_cost, rw.total_cost)
                            if both.total_cost > best * 1.01:
                                result.combination_note = (
                                    "Applying the rewrite TOGETHER with the indexes is estimated worse "
                                    f"({both.total_cost:.2f}) than the best single option ({best:.2f}); "
                                    "deploy one or the other, not both.")
                    if on_event:
                        on_event("sandbox_progress", {"experiments": len(result.per_candidate)})
                finally:
                    try:
                        await conn.execute("SELECT hypopg_reset()")
                        cleanup_ok = True
                    except Exception:  # noqa: BLE001
                        cleanup_ok = False
        except QuerySentinelError as exc:
            result.status = "unavailable" if exc.category == ErrorCategory.HYPOPG_UNAVAILABLE else "failed"
            result.error = exc.message
        except asyncpg.PostgresError as exc:
            result.status = "failed"
            result.error = f"sandbox failure ({exc.sqlstate})"
        result.cleanup_ok = cleanup_ok
        return result


class SyncHypoPGEvaluator:
    """Synchronous HypoPG evaluator for RL training (scripts/train_rl.py) - psycopg2 connection
    in a READ ONLY session. Hypothetical indexes ACCUMULATE within an episode (multi-step MDP)."""

    def __init__(self, conn, real_sql: str, ddl_for: Callable[[Any], str | None],
                 sql_for: Callable[[Any], str | None], table_bytes: float | None):
        self.conn = conn
        self.real_sql = real_sql
        self.current_sql = real_sql
        self.ddl_for = ddl_for
        self.sql_for = sql_for
        self.table_bytes = table_bytes
        self._baseline: float | None = None

    def _explain(self, sql: str) -> float:
        with self.conn.cursor() as cur:
            cur.execute("EXPLAIN (FORMAT JSON) " + sql)
            return float(cur.fetchone()[0][0]["Plan"]["Total Cost"])

    def reset(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT hypopg_reset()")
        self.current_sql = self.real_sql

    def baseline_cost(self) -> float:
        if self._baseline is None:
            self._baseline = self._explain(self.real_sql)
        return self._baseline

    def apply(self, action):
        from app.rl.environment import StepOutcome
        before = self._explain(self.current_sql)
        if action.high == "ADD_INDEX":
            ddl = self.ddl_for(action)
            if not ddl:
                return StepOutcome(before, before, evaluated=False, note="untranslatable candidate")
            with self.conn.cursor() as cur:
                cur.execute("SELECT indexrelid FROM hypopg_create_index(%s)", (ddl,))
                oid = cur.fetchone()[0]
                cur.execute("SELECT hypopg_relation_size(%s)", (oid,))
                size = cur.fetchone()[0]
            after = self._explain(self.current_sql)
            return StepOutcome(before, after, storage_bytes=size, table_bytes=self.table_bytes,
                               n_columns=len(action.payload.get("columns", [])) or 1)
        if action.high == "REWRITE_SQL":
            sql = self.sql_for(action)
            if not sql:
                return StepOutcome(before, before, evaluated=False, note="untranslatable rewrite")
            after = self._explain(sql)
            self.current_sql = sql
            return StepOutcome(before, after)
        return StepOutcome(before, before, evaluated=False, note="advisory action - not physically validated")
