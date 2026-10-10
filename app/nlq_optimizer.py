"""Front door from validated NL-generated SQL into the EXISTING Planora optimizer (Final Upgrade P5).

This is not a second optimizer: it binds the existing `Services` to the active data source
(`Services.for_source`) and runs the existing LangGraph `Pipeline` (security -> privacy -> schema -> EXPLAIN ->
GNN -> agents -> RL -> HypoPG -> blast radius -> final report) on the validated SQL. It then summarises what each
component actually did (component trace), records the workload row under the source's namespace, and reports the
search budget. Index recommendations do not change SQL text, so the original validated query is what executes.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.datasource import DataSourceContext
from app.errors import ErrorCategory, QuerySentinelError
from app.provenance import Provenance
from app.workload_store import WorkloadRecord, hour_bucket

NEGLIGIBLE_PCT = 5.0
_PIPELINES: dict[str, Any] = {}


def forget_source(source_id: str) -> None:
    """Drop the cached optimiser pipeline of a deleted data source."""
    _PIPELINES.pop(source_id, None)


def _pipeline(svc, ctx: DataSourceContext):
    from app.agents.supervisor import Pipeline
    key = ctx.id
    p = _PIPELINES.get(key)
    if p is None or p.s.db is not svc.db:
        p = Pipeline(svc)
        _PIPELINES[key] = p
    return p


def _real_index_ddls(migration_sql: str | None) -> list[str]:
    """Real-name CREATE INDEX statements (trusted side) from the generated migration text."""
    import re
    if not migration_sql:
        return []
    return [" ".join(m.split()) for m in re.findall(r"CREATE INDEX CONCURRENTLY[^;]+", migration_sql)]


def component_trace(rep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per component: used / fallback / skipped, and why. Never claims influence that did not happen."""
    gnn = rep.get("gnn_analysis") or {}
    rl = rep.get("rl_recommendation") or {}
    sb = rep.get("sandbox_simulation") or {}
    proof = rep.get("zero_exposure_proof") or {}
    priv = rep.get("privacy_audit") or {}
    trace = {
        "security_gateway": {"status": "used", "detail": "existing SecurityGateway validated the SQL (SELECT only)"},
        "privacy_layer": {"status": "used" if priv else "skipped",
                          "detail": "literals parameterised + HMAC pseudonyms; zero-exposure proof "
                                    + ("passed" if proof.get("passed", proof.get("ok", True)) else "FAILED")},
    }
    mode = gnn.get("model_mode")
    trace["gnn"] = ({"status": "used", "detail": f"trained checkpoint ({mode})"} if mode and mode != "analytic_fallback"
                    else {"status": "fallback", "detail": "deterministic analytic fallback"} if mode
                    else {"status": "skipped", "detail": "no GNN result"})
    pm = rl.get("policy_mode")
    trace["rl"] = ({"status": "used", "detail": f"policy {pm}: {rl.get('high_level_action')}"} if pm and
                   pm != "analytic_fallback" else {"status": "fallback", "detail": "analytic fallback policy"} if pm
                   else {"status": "skipped", "detail": "no RL decision (no candidate actions)"})
    if sb.get("before_total_cost") is not None and sb.get("virtual_indexes_created"):
        trace["hypopg"] = {"status": "used", "detail": f"{len(sb['virtual_indexes_created'])} hypothetical index(es)"
                                                        " compared on one connection"}
    elif sb:
        trace["hypopg"] = {"status": "skipped", "detail": sb.get("status") or "no valid index candidates"}
    else:
        trace["hypopg"] = {"status": "skipped", "detail": "no index candidates reached the sandbox"}
    return trace


async def _record_workload(svc, ctx: DataSourceContext, rep: dict[str, Any], sql: str) -> WorkloadRecord:
    plan = rep.get("plan_analysis") or {}
    sb = rep.get("sandbox_simulation") or {}
    rec = WorkloadRecord(
        query_fingerprint=(rep.get("final_report") or {}).get("query_fingerprint")
        or "q_" + hashlib.sha256(sql.encode()).hexdigest()[:16],
        plan_fingerprint=sb.get("baseline_plan_fingerprint"), timestamp_bucket=hour_bucket(datetime.now(timezone.utc)),
        estimated_cost=sb.get("before_total_cost") or plan.get("total_cost"), source="nlq",
        data_source_id=ctx.id)
    svc.store.add(rec)
    return rec


async def persist_workload_row(main_db, rec: WorkloadRecord) -> bool:
    if main_db is None or not main_db.available:
        return False
    try:
        await main_db.execute_write(
            "INSERT INTO optimizer_meta.query_history (query_fingerprint, plan_fingerprint, timestamp_bucket, "
            "call_count, estimated_cost, source, data_source_id) VALUES ($1,$2,$3,1,$4,'nlq',$5)",
            rec.query_fingerprint, rec.plan_fingerprint, datetime.fromisoformat(rec.timestamp_bucket),
            rec.estimated_cost, rec.data_source_id)
        return True
    except Exception:  # noqa: BLE001 - persistence is best effort
        return False


async def optimize(services, ctx: DataSourceContext, sql: str, main_db=None) -> dict[str, Any]:
    svc = services.for_source(ctx)
    rid = f"qs-{uuid.uuid4().hex[:16]}"
    rep = await _pipeline(svc, ctx).run(rid, sql)
    if rep.get("status") != "completed":
        return {"request_id": rid, "status": rep.get("status"), "error": rep.get("error"),
                "component_trace": {"security_gateway": {"status": "used", "detail": "rejected"}}}
    sb = rep.get("sandbox_simulation") or {}
    fr = rep.get("final_report") or {}
    pct = sb.get("planner_cost_reduction_pct")
    dep = rep.get("deployment") or {}
    has_index = bool(sb.get("virtual_indexes_used"))
    negligible = pct is None or pct < NEGLIGIBLE_PCT
    budget = fr.get("budget") or {}
    out = {
        "request_id": rid, "status": "completed", "data_source_id": ctx.id,
        "component_trace": component_trace(rep),
        "hypopg_comparison": {
            "before_total_cost": {"value": sb.get("before_total_cost"),
                                  "provenance": [Provenance.PLANNER_ESTIMATE.value]},
            "after_total_cost": {"value": sb.get("after_total_cost"),
                                 "provenance": [Provenance.HYPOPG_SIMULATION.value]},
            "planner_cost_reduction_pct": {"value": pct, "provenance": [Provenance.HYPOPG_SIMULATION.value,
                                                                        Provenance.PLANNER_ESTIMATE.value]},
            "statement": (f"planner cost reduced by {pct:.1f} percent [PLANNER ESTIMATE]" if pct is not None
                          and not negligible else
                          "estimated cost reduction is negligible; no index is recommended for this query"
                          + (" (small tables rarely benefit from indexes)" if negligible else "")),
        },
        "recommendations": {
            "indexes": _real_index_ddls(dep.get("migration_sql")) if has_index and not negligible else [],
            "indexes_as_seen_by_ai": sb.get("virtual_indexes_used") or [],
            "migration_sql": dep.get("migration_sql") if has_index and not negligible else None,
            "rollback_sql": dep.get("rollback_sql") if has_index and not negligible else None,
            "requires_human_review": True, "label": "HUMAN REVIEW REQUIRED",
            "provenance": [Provenance.HYPOPG_SIMULATION.value]},
        "rl": {k: (rep.get("rl_recommendation") or {}).get(k) for k in
               ("high_level_action", "low_level_action", "policy_mode", "explanation")},
        "gnn": {k: (rep.get("gnn_analysis") or {}).get(k) for k in
                ("model_mode", "primary_bottleneck_type", "natural_language_summary")},
        "search_budget": budget,
        "answer_sentence": (rep.get("answer") or {}).get("sentence"),
        "zero_exposure_proof": rep.get("zero_exposure_proof"),
        "executed_sql_is_original": True,
        "full_report_url": f"/api/v1/report/{rid}",
    }
    if budget.get("exhausted"):
        out["events"] = ["SEARCH_BUDGET_EXHAUSTED"]
        if main_db is not None and main_db.available:
            try:
                await main_db.execute_write(
                    "INSERT INTO qs_meta.audit_log (data_source_id, event_type, details) VALUES ($1, "
                    "'SEARCH_BUDGET_EXHAUSTED', $2::jsonb)", ctx.id, json.dumps(budget))
            except Exception:  # noqa: BLE001
                pass
    from app import feature_flags
    if feature_flags.enabled("index_portfolio"):          # P7: portfolio reasoning (flag; default off)
        try:
            from app.index_portfolio import analyze_portfolio
            stm = await svc.instr.statements(200)
            cands = [d.replace("CREATE INDEX CONCURRENTLY IF NOT EXISTS", "CREATE INDEX").split(" ON ", 1)[0]
                     and "CREATE INDEX ON " + d.split(" ON ", 1)[1].replace(" USING btree", " USING btree")
                     for d in out["recommendations"]["indexes"]]
            out["index_portfolio"] = await analyze_portfolio(ctx.db, ctx.schema_name, cands, stm)
        except Exception as exc:  # noqa: BLE001 - portfolio is additive; never breaks the optimisation
            out["index_portfolio"] = {"error": type(exc).__name__}
    rec = await _record_workload(svc, ctx, rep, sql)
    out["workload_namespace"] = rec.data_source_id
    out["workload_persisted"] = await persist_workload_row(main_db, rec)
    return out


# ------------------------------------------------------------------ optional demo action (uploaded datasets only)
async def apply_index_and_measure(ctx: DataSourceContext, sql: str, ddl: str, settings, admin_conn_factory=None,
                                  runs: int = 5) -> dict[str, Any]:
    """Apply ONE recommended index through a trusted path on an uploaded-dataset source, re-run the query and
    report before/after medians as [MEASURED SANDBOX]. Never available on the demo database."""
    from pglast import ast, parse_sql
    from app.recommendation_mapper import quote_ident, real_index_parts
    if ctx.kind == "demo":
        raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "apply-index is never available on the demo "
                                 "database", stage="apply_index")
    stmt = parse_sql(ddl)[0].stmt
    if not isinstance(stmt, ast.IndexStmt):
        raise QuerySentinelError(ErrorCategory.INVALID_DDL, "only CREATE INDEX may be applied", stage="apply_index")
    parts = real_index_parts(ddl)
    if ctx.allowed_tables is not None and parts["table"] not in ctx.allowed_tables:
        raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "index target is outside this data source",
                                 stage="apply_index")
    cols = ", ".join(quote_ident(c) for c in parts["columns"])
    safe_ddl = (f"CREATE INDEX IF NOT EXISTS {quote_ident('qs_demo_' + hashlib.sha1(ddl.encode()).hexdigest()[:10])} "
                f"ON {quote_ident(ctx.schema_name)}.{quote_ident(parts['table'])} ({cols})")

    async def timed(conn) -> list[float]:
        out = []
        for _ in range(runs):
            t0 = time.perf_counter()
            await conn.fetch(sql)
            out.append((time.perf_counter() - t0) * 1000)
        return out

    import asyncpg
    conn = await asyncpg.connect(**settings.admin_dsn_kwargs)   # trusted internal path, qs_dynamic only
    try:
        await conn.execute(f"SET search_path = {quote_ident(ctx.schema_name)}")
        before = await timed(conn)
        await conn.execute(safe_ddl)
        await conn.execute(f"ANALYZE {quote_ident(ctx.schema_name)}.{quote_ident(parts['table'])}")
        after = await timed(conn)
    finally:
        await conn.close()
    prov = [Provenance.MEASURED_SANDBOX.value]
    mb, ma = statistics.median(before), statistics.median(after)
    return {"applied_ddl": safe_ddl, "runs": runs, "before_ms": {"median": round(mb, 3), "values": [round(x, 3)
                                                                                                for x in before]},
            "after_ms": {"median": round(ma, 3), "values": [round(x, 3) for x in after]},
            "runtime_change_pct": round((mb - ma) / mb * 100, 1) if mb else None, "provenance": prov,
            "note": f"measured on this {ctx.kind} source ({runs} runs each, warm cache); not production"}
