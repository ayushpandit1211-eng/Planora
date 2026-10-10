"""Final Upgrade API routers (P13). Additive: every existing endpoint keeps its contract.

Conventions follow app/api.py: prefix /api/v1, JSON error envelope {error, message, stage, details}, optional
X-API-Key. Every request for a data source carries `data_source_id` (absent = demo) and the browser session id in
the `X-Session-Id` header (ownership, NOT authentication). File uploads are sent as the raw request body
(`application/octet-stream`) with `filename` as a query parameter, streamed with a size cap (no extra dependency).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from app import feature_flags
from app.config import get_settings
from app.errors import ErrorCategory, QuerySentinelError
from app.provenance import ALL_TAGS

SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class CommitBody(BaseModel):
    preview_id: str = Field(..., pattern=r"^pv_[0-9a-f]{20}$")
    allow_duplicate: bool = False
    allow_widening: bool = False


class NLQBody(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    data_source_id: str | None = Field(None, max_length=64)
    optimize: bool = True
    execute: bool = True
    anchor_date: dt.date | None = None


class ValidateBody(BaseModel):
    sql: str = Field(..., min_length=1, max_length=8000)
    data_source_id: str | None = Field(None, max_length=64)


class RunSQLBody(BaseModel):
    sql: str = Field(..., min_length=1, max_length=8000)
    data_source_id: str | None = Field(None, max_length=64)
    optimize: bool = True


class ApplyIndexBody(BaseModel):
    sql: str = Field(..., min_length=1, max_length=8000)
    ddl: str = Field(..., min_length=10, max_length=2000)
    data_source_id: str = Field(..., max_length=64)


class OptimizeOnSourceBody(BaseModel):
    sql: str = Field(..., min_length=1, max_length=8000)
    data_source_id: str | None = Field(None, max_length=64)


def _session(x_session_id: str | None) -> str | None:
    if x_session_id is None:
        return None
    if not SESSION_RE.match(x_session_id):
        raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "invalid session id", stage="session")
    return x_session_id


def _require(flag: str) -> None:
    if not feature_flags.enabled(flag):
        raise QuerySentinelError(ErrorCategory.FEATURE_DISABLED, f"feature '{flag}' is disabled", stage="flags")


async def _read_body(request: Request, limit_mb: float) -> bytes:
    cap = int(limit_mb * 1024 * 1024)
    chunks, n = [], 0
    async for chunk in request.stream():
        n += len(chunk)
        if n > cap:
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"upload exceeds {limit_mb} MB", stage="upload")
        chunks.append(chunk)
    if n == 0:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "empty upload", stage="upload")
    return b"".join(chunks)


def build_router(require_key) -> APIRouter:
    r = APIRouter(prefix="/api/v1", dependencies=[Depends(require_key)])

    def st(request: Request):
        return request.app.state

    # ------------------------------------------------------------------ status
    @r.get("/upgrade/status")
    async def upgrade_status(request: Request, x_session_id: str | None = Header(default=None)):
        s = st(request)
        svc = s.services
        from app.dp import accountant
        sess = _session(x_session_id)
        scope = (sess or "global") if get_settings().dp_budget_scope == "session" else "global"
        from app.optimization_budget import OptimizationBudget
        return {"feature_flags": feature_flags.snapshot(),
                "model_modes": {"gnn": svc.gnn.status(), "rl": svc.rl.status(), "llm": svc.llm.status()},
                "search_budget_limits": OptimizationBudget.from_settings().limits,
                "differential_privacy": {"enabled": feature_flags.enabled("dp"), **accountant.snapshot(scope)},
                "provenance_tags": list(ALL_TAGS), "raw_data_exposed_to_ai": 0}

    # ------------------------------------------------------------------ sources
    @r.get("/sources")
    async def list_sources(request: Request, x_session_id: str | None = Header(default=None)):
        sess = _session(x_session_id)
        s = st(request)
        out = [s.resolver.demo().public_view()]
        single = get_settings().single_user
        if s.services.db.available and (sess or single):
            rows = await s.services.db.fetch(
                "SELECT id, kind, name, status, schema_version, expires_at FROM qs_meta.data_sources WHERE "
                "kind = 'dynamic' AND ($2 OR owner_session_id = $1) AND status IN ('creating', 'ready') "
                "AND (expires_at IS NULL OR expires_at > now()) ORDER BY created_at DESC", sess, single)
            out += [{"id": x["id"], "kind": x["kind"], "name": x["name"], "status": x["status"],
                     "schema_version": x["schema_version"],
                     "expires_at": x["expires_at"].isoformat() if x["expires_at"] else None} for x in rows]
        return {"sources": out}

    @r.get("/sources/{source_id}")
    async def source_detail(source_id: str, request: Request, x_session_id: str | None = Header(default=None)):
        ctx = await st(request).resolver.resolve(source_id, _session(x_session_id))
        return ctx.public_view()

    @r.get("/sources/{source_id}/schema")
    async def source_schema(source_id: str, request: Request, x_session_id: str | None = Header(default=None)):
        from app.nlq import load_snapshot
        ctx = await st(request).resolver.resolve(source_id, _session(x_session_id))
        tables = await load_snapshot(ctx)
        prov = ["POSTGRES OBSERVED"]
        return {"data_source": ctx.public_view(), "provenance": prov, "tables": [
            {"name": t.name, "estimated_rows": {"value": t.rows, "provenance": ["POSTGRES OBSERVED"]},
             "columns": [{"name": c, "type": ty, "stats": t.stats.get(c, {})} for c, ty in t.columns],
             "primary_key": t.pk, "foreign_keys": [{"columns": a, "ref_table": b, "ref_columns": c}
                                                   for a, b, c in t.fks]} for t in tables.values()]}

    # ------------------------------------------------------------------ dataset ingestion
    @r.post("/data/preview")
    async def data_preview(request: Request, filename: str = Query(..., max_length=200),
                           dataset_name: str | None = Query(None, max_length=60),
                           entity_name: str | None = Query(None, max_length=60),
                           x_session_id: str | None = Header(default=None)):
        _require("ingestion")
        sess = _session(x_session_id)
        if sess is None:
            raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "X-Session-Id header required", stage="upload")
        s = st(request)
        data = await _read_body(request, get_settings().ingest_max_upload_mb)
        path = s.ingestion.stage_bytes(data, filename)
        return await s.ingestion.preview(path, filename, dataset_name, entity_name, sess)

    @r.post("/data/commit")
    async def data_commit(body: CommitBody, request: Request, x_session_id: str | None = Header(default=None)):
        _require("ingestion")
        return await st(request).ingestion.commit(body.preview_id, _session(x_session_id), body.allow_duplicate,
                                                  body.allow_widening)

    @r.delete("/data/datasets/{source_id}")
    async def delete_dataset(source_id: str, request: Request, x_session_id: str | None = Header(default=None)):
        """Delete an uploaded dataset (its tables and registry rows). The demo database can never be deleted."""
        _require("ingestion")
        from app.datasource import ID_RE
        from app.nlq_optimizer import forget_source
        from app.schema_inspector import schema_cache
        if not ID_RE.match(source_id) or not source_id.startswith("dyn_"):
            raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "only uploaded datasets can be deleted",
                                     stage="dataset_delete")
        out = await st(request).ingestion.delete_source(source_id, _session(x_session_id))
        schema_cache.invalidate(source_id)
        forget_source(source_id)
        return out

    @r.get("/data/datasets")
    async def datasets(request: Request, x_session_id: str | None = Header(default=None)):
        sess = _session(x_session_id)
        rows = await st(request).services.db.fetch(
            "SELECT s.id AS data_source_id, d.name, s.schema_version, t.table_name, t.row_count FROM qs_meta.datasets d "
            "JOIN qs_meta.data_sources s ON s.id = d.data_source_id LEFT JOIN qs_meta.dataset_tables t ON "
            "t.dataset_id = d.id WHERE ($2 OR s.owner_session_id = $1) AND s.status = 'ready' ORDER BY d.name", sess,
            get_settings().single_user)
        return {"datasets": [dict(x) for x in rows]}

    @r.get("/data/runs")
    async def ingestion_runs(request: Request, data_source_id: str, x_session_id: str | None = Header(default=None)):
        await st(request).resolver.resolve(data_source_id, _session(x_session_id))       # ownership check
        rows = await st(request).services.db.fetch(
            "SELECT id, filename, status, rows_read, rows_loaded, rows_rejected, schema_changes, warnings, created_at "
            "FROM qs_meta.ingestion_runs WHERE data_source_id = $1 ORDER BY created_at DESC LIMIT 50", data_source_id)
        return {"runs": [{**dict(x), "created_at": x["created_at"].isoformat()} for x in rows]}

    # ------------------------------------------------------------------ NL analysis
    async def _run_nlq(request: Request, body: NLQBody, sess: str | None, run_id: str | None = None):
        from app.nlq_optimizer import optimize
        s = st(request)
        ctx = await s.resolver.resolve(body.data_source_id, sess)
        events = s.services.events

        def progress(ev: str, msg: str) -> None:
            if run_id:
                events.publish(run_id, ev, "nlq", msg)

        opt = (lambda sql, c, o: optimize(s.services, c, sql, s.services.db)) if body.optimize else None
        out = await s.nlq.ask(body.question, ctx, sess, body.anchor_date, body.execute, opt, progress)
        await s.nlq.persist_run(s.services.db, (out.get("optimization") or {}).get("component_trace"))
        return out

    @r.post("/nlq/ask")
    async def nlq_ask(body: NLQBody, request: Request, x_session_id: str | None = Header(default=None)):
        _require("nlq")
        return await _run_nlq(request, body, _session(x_session_id))

    @r.post("/nlq/ask/async")
    async def nlq_ask_async(body: NLQBody, request: Request, x_session_id: str | None = Header(default=None)):
        _require("nlq")
        sess = _session(x_session_id)
        s = st(request)
        run_id = f"qs-nlq{secrets.token_hex(8)}"
        s.services.events.ensure(run_id)
        s.services.events.publish(run_id, "NLQ_STARTED", "nlq", "question received")

        async def job():
            try:
                out = await _run_nlq(request, body, sess, run_id)
                s.services.reports[run_id] = out
                s.services.events.publish(run_id, "NLQ_COMPLETE", "nlq", "answer ready")
            except QuerySentinelError as exc:
                s.services.reports[run_id] = {"status": "error", **exc.to_dict(run_id)}
                s.services.events.publish(run_id, "NLQ_SQL_REJECTED" if exc.category == ErrorCategory.SQL_REJECTED
                                          else "ERROR", exc.stage or "nlq", exc.message)
            except Exception as exc:  # noqa: BLE001
                s.services.reports[run_id] = {"status": "error", "error": "internal_error",
                                              "message": type(exc).__name__}
                s.services.events.publish(run_id, "ERROR", "nlq", "internal error")
        t = asyncio.create_task(job())
        s.tasks.add(t)
        t.add_done_callback(s.tasks.discard)
        return {"run_id": run_id, "stream_url": f"/api/v1/stream/{run_id}", "report_url": f"/api/v1/report/{run_id}"}

    @r.post("/nlq/run")
    async def nlq_run(body: RunSQLBody, request: Request, x_session_id: str | None = Header(default=None)):
        """User-written SQL on the ACTIVE data source: 10-stage safety gate -> (optional) existing optimizer ->
        execution as the restricted read-only role. Same answer shape as /nlq/ask."""
        from app.nlq import answer_text, load_snapshot
        from app.nlq_optimizer import optimize
        from app.sql_safety import SQLSafetyGate
        s = st(request)
        ctx = await s.resolver.resolve(body.data_source_id, _session(x_session_id))
        tables = await load_snapshot(ctx)
        gate = SQLSafetyGate(ctx, {n: [c for c, _ in t.columns] for n, t in tables.items()})
        res = await gate.validate(body.sql)
        out: dict[str, Any] = {
            "status": "completed", "data_source": ctx.public_view(), "understanding": "SQL written by you",
            "assumptions": [], "detected_tables": res.relations, "detected_columns": [],
            "generated_sql": res.sql, "generator": "user_sql", "generator_provenance": [],
            "validation": {"passed": True, "stages_passed": res.stages_passed, "attempts": [],
                           "limit_added": res.limit_added, "single_row_aggregate": res.single_row_aggregate},
            "query_plan": {"total_cost": res.plan_total_cost,
                           "root_node": (res.plan or {}).get("Plan", {}).get("Node Type"),
                           "provenance": ["PLANNER ESTIMATE"]}, "warnings": list(res.notes)}
        if body.optimize:
            out["optimization"] = await optimize(s.services, ctx, res.validated_sql or res.sql, s.services.db)
        ex = await gate.execute(res)
        out["execution_result"] = {**ex, "provenance": ["POSTGRES OBSERVED"]}
        out["answer"] = answer_text(ex, ctx)
        return out

    @r.post("/nlq/validate")
    async def nlq_validate(body: ValidateBody, request: Request, x_session_id: str | None = Header(default=None)):
        from app.nlq import load_snapshot
        from app.sql_safety import SQLSafetyGate
        ctx = await st(request).resolver.resolve(body.data_source_id, _session(x_session_id))
        tables = await load_snapshot(ctx)
        res = await SQLSafetyGate(ctx, {n: [c for c, _ in t.columns] for n, t in tables.items()}).validate(body.sql)
        return {"passed": True, "stages_passed": res.stages_passed, "final_sql": res.sql,
                "plan_total_cost": {"value": res.plan_total_cost, "provenance": ["PLANNER ESTIMATE"]},
                "limit_added": res.limit_added}

    @r.post("/nlq/optimize")
    async def optimize_on_source(body: OptimizeOnSourceBody, request: Request,
                                 x_session_id: str | None = Header(default=None)):
        """Validated SQL on any source -> existing optimizer (no execution)."""
        from app.nlq import load_snapshot
        from app.nlq_optimizer import optimize
        from app.sql_safety import SQLSafetyGate
        s = st(request)
        ctx = await s.resolver.resolve(body.data_source_id, _session(x_session_id))
        tables = await load_snapshot(ctx)
        res = await SQLSafetyGate(ctx, {n: [c for c, _ in t.columns] for n, t in tables.items()}).validate(body.sql)
        out = await optimize(s.services, ctx, res.validated_sql or res.sql, s.services.db)
        if res.notes:
            out["sql_notes"] = list(res.notes)
            out["corrected_sql"] = res.validated_sql or res.sql
        return out

    @r.post("/nlq/apply-index")
    async def apply_index(body: ApplyIndexBody, request: Request, x_session_id: str | None = Header(default=None)):
        from app.nlq import load_snapshot
        from app.nlq_optimizer import apply_index_and_measure
        from app.optimization_budget import OptimizationBudget
        from app.sql_safety import SQLSafetyGate
        s = st(request)
        ctx = await s.resolver.resolve(body.data_source_id, _session(x_session_id))
        if ctx.kind == "demo":
            raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "apply-index is never available on the demo "
                                     "database", stage="apply_index")
        tables = await load_snapshot(ctx)
        res = await SQLSafetyGate(ctx, {n: [c for c, _ in t.columns] for n, t in tables.items()}).validate(body.sql)
        budget = s.benchmark_budgets.setdefault(ctx.id, OptimizationBudget.from_settings())
        if not budget.consume("sandbox_benchmarks"):
            raise QuerySentinelError(ErrorCategory.SEARCH_BUDGET_EXHAUSTED, "sandbox benchmark budget exhausted "
                                     "for this source", stage="apply_index", details=budget.snapshot())
        out = await apply_index_and_measure(ctx, res.sql, body.ddl, get_settings())
        from app.schema_inspector import schema_cache
        schema_cache.invalidate(ctx.id)
        return out

    # ------------------------------------------------------------------ index portfolio
    @r.get("/index-portfolio")
    async def index_portfolio(request: Request, data_source_id: str | None = None,
                              x_session_id: str | None = Header(default=None)):
        _require("index_portfolio")
        from app.index_portfolio import analyze_portfolio
        s = st(request)
        ctx = await s.resolver.resolve(data_source_id, _session(x_session_id))
        try:
            stm = await ctx.instr.statements(200)
        except QuerySentinelError:
            stm = []
        return await analyze_portfolio(ctx.db, ctx.schema_name, None, stm)

    return r


async def startup_upgrade(app, svc) -> None:
    """Called from the existing lifespan after Services.startup() and the metadata migration."""
    from app.dataset_ingestion import DatasetIngestionService
    from app.datasource import DataSourceResolver
    from app.nlq import NLQService
    resolver = DataSourceResolver(svc.db)
    app.state.resolver = resolver
    app.state.ingestion = DatasetIngestionService(audit=svc.audit)
    app.state.nlq = NLQService(llm=svc.llm, audit=svc.audit)
    app.state.benchmark_budgets = {}


async def shutdown_upgrade(app) -> None:
    res = getattr(app.state, "resolver", None)
    if res is not None:
        await res.close()
