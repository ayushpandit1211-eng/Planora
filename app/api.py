"""FastAPI backend (Section 39) with Server-Sent Events (Section 38).

    uvicorn app.api:app --host 0.0.0.0 --port 8000

Every optimisation request gets a request ID. Raw SQL is accepted only on the trusted backend,
is never echoed in events/logs, and is cleared from the vault when the request finishes.
"""
from __future__ import annotations

import asyncio
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.nl_intent import NLIntentAssistant
from app.agents.supervisor import Pipeline
from app.benchmark import BenchmarkRunner, load_demo_queries
from app.config import get_settings
from app.errors import ErrorCategory, QuerySentinelError
from app.events import to_sse
from app.index_hygiene import analyze_hygiene
from app.ingestion import IngestionEngine, PgStatStatementsCollector
from app.logging_setup import configure_logging, get_logger
from app.metrics import metrics
from app.pipeline import Services
from app.regression_detector import detect_all
from app.schema_inspector import SchemaInspector
from app.workload_drift import detect_drift

log = get_logger("api")
MAX_BODY_SQL = 20000


# ------------------------------------------------------------------ request models
class SQLRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=MAX_BODY_SQL)
    request_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]{6,64}$")


class IngestRequest(BaseModel):
    format: Literal["json", "csv"] = "json"
    content: str | None = Field(None, max_length=5_000_000)
    records: list[dict[str, Any]] | None = Field(None, max_length=10_000)


class BenchmarkRequest(BaseModel):
    sql: str | None = Field(None, max_length=MAX_BODY_SQL)
    demo_query_index: int | None = Field(None, ge=0, le=20)
    variants: list[str] | None = None
    measure: bool = False


class NLRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class NLAnalyzeRequest(BaseModel):
    candidate_id: str = Field(..., pattern=r"^cand-[0-9a-f]{10}$")


class CollectRequest(BaseModel):
    limit: int = Field(50, ge=1, le=500)
    min_mean_ms: float = Field(0.0, ge=0)


# ------------------------------------------------------------------ app factory
def create_app(services: Services | None = None, connect: bool = True) -> FastAPI:
    settings = get_settings()
    if settings.qs_deployment_mode == "production":
        if len(settings.qs_api_key.get_secret_value()) < 32:
            raise ValueError("Production requires a QS_API_KEY of at least 32 characters")
        if len(settings.privacy_hmac_key.get_secret_value()) < 32:
            raise ValueError("Production requires a PRIVACY_HMAC_KEY of at least 32 characters")
    configure_logging(settings.log_level)
    svc = services or Services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if connect:
            await svc.startup()
            if settings.qs_run_meta_migration and svc.db.available:
                # Final Upgrade P1: additive qs_meta migration after the existing initialisation.
                try:
                    from app.meta_migration import run_meta_migration
                    await run_meta_migration(settings)
                except Exception as exc:  # noqa: BLE001 - existing features keep working without it
                    log.warning("metadata migration failed", extra={"error_category": type(exc).__name__})
        app.state.services = svc
        app.state.pipeline = Pipeline(svc)
        app.state.nl = NLIntentAssistant(svc.instr, svc.obfuscator, svc.store, svc.known_tables)
        app.state.tasks = set()
        from app.api_upgrade import shutdown_upgrade, startup_upgrade   # Final Upgrade P13 (additive)
        await startup_upgrade(app, svc)
        yield
        for t in list(app.state.tasks):
            t.cancel()
        await shutdown_upgrade(app)
        if connect:
            await svc.shutdown()

    app = FastAPI(title="Planora API", version="1.0.0", lifespan=lifespan,
                  description="Privacy-aware AI-assisted PostgreSQL DBA decision-support prototype. "
                              "Recommendations require human review; nothing is deployed automatically.")

    async def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = settings.qs_api_key.get_secret_value()
        if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail={"error": "unauthorized", "message": "invalid API key"})

    from app.api_upgrade import build_router            # Final Upgrade P13: additive routers
    app.include_router(build_router(require_key))

    @app.exception_handler(QuerySentinelError)
    async def qs_error(request: Request, exc: QuerySentinelError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(getattr(request.state, "rid", None)))

    @app.exception_handler(Exception)
    async def generic_error(request: Request, exc: Exception):
        log.error("unhandled error", extra={"error_category": type(exc).__name__})
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                      "message": "internal error (details withheld)"})

    def services_(request: Request) -> Services:
        return request.app.state.services

    def new_rid(given: str | None = None) -> str:
        return given or f"qs-{uuid.uuid4().hex[:16]}"

    # ---------------------------------------------------------------- health / status
    @app.get("/api/v1/health")
    async def health(request: Request):
        s = services_(request)
        h = await s.instr.health()
        ok = h["postgresql"]["ok"]
        return JSONResponse(status_code=200 if ok else 503, content={
            "status": "ok" if ok else "degraded", "postgresql": h["postgresql"]["ok"],
            "hypopg": h["hypopg"]["ok"], "pg_stat_statements": h["pg_stat_statements"]["ok"]})

    @app.get("/api/v1/metrics", dependencies=[Depends(require_key)])
    async def metrics_json():
        return metrics.snapshot()

    @app.get("/api/v1/metrics/prometheus", dependencies=[Depends(require_key)])
    async def metrics_prom():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics.prometheus())

    @app.get("/api/v1/status", dependencies=[Depends(require_key)])
    async def status(request: Request):
        return await services_(request).status()

    @app.get("/api/v1/ready", dependencies=[Depends(require_key)])
    async def ready(request: Request):
        status = await services_(request).status()
        checks = {name: bool(status.get(name, {}).get("ok")) for name in
                  ("postgresql", "hypopg", "pg_stat_statements", "privacy_vault", "langgraph")}
        checks["gnn"] = status.get("gnn", {}).get("mode") == "trained"
        checks["pg_stat_statements"] &= bool(status.get("pg_stat_statements", {}).get("preloaded"))
        checks["rl"] = status.get("rl", {}).get("mode") == "trained_policy"
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ready" if ok else "not_ready", "checks": checks})

    # ---------------------------------------------------------------- privacy
    @app.post("/api/v1/mask", dependencies=[Depends(require_key)])
    async def mask(body: SQLRequest, request: Request):
        s = services_(request)
        rid = new_rid(body.request_id)
        request.state.rid = rid
        try:
            out = s.obfuscator.mask_sql(body.sql, rid)
            s.audit.record(rid, "MASKING", stage="api.mask", details={"literal_count": out["literal_count"],
                                                                      "safe_for_llm": out["privacy_audit"]["safe_for_llm"]})
            return {"request_id": rid, **out,
                    "note": "reverse mapping is kept server-side and is never returned"}
        finally:
            s.obfuscator.clear_vault(rid)

    # ---------------------------------------------------------------- analysis
    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
    async def analyze(body: SQLRequest, request: Request):
        """Security -> privacy -> schema -> EXPLAIN -> graph -> GNN (no agents / no sandbox)."""
        s = services_(request)
        p: Pipeline = request.app.state.pipeline
        rid = new_rid(body.request_id)
        request.state.rid = rid
        from app.agents.state import initial_state
        from app.optimization_budget import OptimizationBudget
        s.trusted[rid] = {"raw_sql": body.sql, "budget": OptimizationBudget.from_settings()}
        state: dict = dict(initial_state(rid))
        try:
            for fn in (p.n_security, p.n_privacy, p.n_schema, p.n_explain, p.n_graph, p.n_gnn):
                upd = await fn(state)
                if upd.get("error"):
                    raise QuerySentinelError(ErrorCategory(upd["error"]["error"]), upd["error"]["message"],
                                             stage=upd["error"].get("stage"))
                for k, v in upd.items():
                    if k in ("agent_logs",):
                        state[k] = (state.get(k) or []) + v
                    else:
                        state[k] = v
            return {"request_id": rid, "privacy_audit": state["privacy"], "schema_analysis": state["schema_stats"],
                    "plan_analysis": state["plan_analysis"], "graph": state["graph_summary"],
                    "gnn_analysis": state["gnn_diagnosis"], "access_pattern": state["access_pattern"],
                    "workload_context": state["workload_context"], "regression": state.get("regression")}
        finally:
            s.obfuscator.clear_vault(rid)
            s.trusted.pop(rid, None)

    @app.post("/api/v1/optimize", dependencies=[Depends(require_key)])
    async def optimize(body: SQLRequest, request: Request):
        rid = new_rid(body.request_id)
        request.state.rid = rid
        res = await request.app.state.pipeline.run(rid, body.sql)
        code = 200 if res["status"] == "completed" else 400 if res["status"] == "rejected" else 500
        return JSONResponse(status_code=code, content=res)

    @app.post("/api/v1/optimize/async", dependencies=[Depends(require_key)])
    async def optimize_async(body: SQLRequest, request: Request):
        rid = new_rid(body.request_id)
        s = services_(request)
        s.events.ensure(rid)
        task = asyncio.create_task(request.app.state.pipeline.run(rid, body.sql))
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
        return {"request_id": rid, "stream_url": f"/api/v1/stream/{rid}", "report_url": f"/api/v1/report/{rid}"}

    @app.get("/api/v1/report/{request_id}", dependencies=[Depends(require_key)])
    async def report(request_id: str, request: Request):
        s = services_(request)
        if request_id in s.reports:
            return s.reports[request_id]
        if s.events.known(request_id):
            return JSONResponse(status_code=202, content={"request_id": request_id, "status": "running"})
        raise QuerySentinelError(ErrorCategory.NOT_FOUND, "unknown request id", stage="report")

    @app.get("/api/v1/stream/{request_id}")
    async def stream(request_id: str, request: Request, x_api_key: str | None = Header(default=None)):
        await require_key(x_api_key)
        s = services_(request)

        async def gen():
            try:
                async for ev in s.events.subscribe(request_id, heartbeat_s=10.0):
                    if await request.is_disconnected():
                        break
                    yield to_sse(ev)
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 - SSE failure must not leak internals
                yield "event: ERROR\ndata: {\"event\": \"ERROR\", \"message\": \"stream failure\"}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/v1/audit/{request_id}", dependencies=[Depends(require_key)])
    async def audit(request_id: str, request: Request):
        evs = services_(request).audit.events_for(request_id)
        return {"request_id": request_id, "events": [e.model_dump() for e in evs]}

    # ---------------------------------------------------------------- ingestion / workload
    def _engine(s: Services, schema=None) -> IngestionEngine:
        return IngestionEngine(s.obfuscator, s.store, s.guard, schema)

    async def _schema(s: Services):
        try:
            return await SchemaInspector(s.instr).inspect(None) if s.db.available else None
        except QuerySentinelError:
            return None

    async def _ingest(body: IngestRequest, request: Request, source: str):
        s = services_(request)
        eng = _engine(s, await _schema(s))
        if body.records is not None:
            rep = eng.ingest(body.records, source)
        elif body.content:
            rep = eng.ingest_csv(body.content, source) if body.format == "csv" else eng.ingest_json(body.content, source)
        else:
            raise QuerySentinelError(ErrorCategory.INVALID_SQL, "provide `records` or `content`", stage="ingestion")
        persisted = await s.persist_workload(rep.records)
        s.audit.record(rep.batch_id, "INGESTION", stage="ingestion",
                       details={"accepted": rep.accepted, "quarantined": rep.quarantined, "persisted": persisted})
        out = rep.to_dict(include_records=True)
        out["persisted_to_optimizer_meta"] = persisted
        return out

    @app.post("/api/v1/ingest/logs", dependencies=[Depends(require_key)])
    async def ingest_logs(body: IngestRequest, request: Request):
        return await _ingest(body, request, f"{body.format}_slow_query_log")

    @app.post("/api/v1/ingest/plans", dependencies=[Depends(require_key)])
    async def ingest_plans(body: IngestRequest, request: Request):
        return await _ingest(body, request, f"{body.format}_historical_plans")

    @app.post("/api/v1/ingest/pg_stat_statements", dependencies=[Depends(require_key)])
    async def ingest_pgss(body: CollectRequest, request: Request):
        s = services_(request)
        eng = _engine(s, await _schema(s))
        rep = await PgStatStatementsCollector(s.instr, eng).collect(body.limit, body.min_mean_ms)
        await s.persist_workload(rep.records)
        return rep.to_dict(include_records=True)

    @app.get("/api/v1/workload", dependencies=[Depends(require_key)])
    async def workload(request: Request):
        s = services_(request)
        hist, recent = s.store.split_by_time()
        return {"summary": s.store.summary(), "top_fingerprints": dict(list(s.store.fingerprints().items())[:20]),
                "regressions": detect_all(s.store.all()), "drift": detect_drift(hist, recent)}

    @app.get("/api/v1/hygiene", dependencies=[Depends(require_key)])
    async def hygiene(request: Request):
        s = services_(request)
        schema = await _schema(s)
        if schema is None:
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, "database unavailable", stage="hygiene")
        return analyze_hygiene(schema)

    # ---------------------------------------------------------------- benchmark
    @app.post("/api/v1/benchmark", dependencies=[Depends(require_key)])
    async def benchmark(body: BenchmarkRequest, request: Request):
        s = services_(request)
        if body.sql is None and body.demo_query_index is None:
            raise QuerySentinelError(ErrorCategory.INVALID_SQL, "provide `sql` or `demo_query_index`", stage="benchmark")
        sql = body.sql if body.sql is not None else load_demo_queries()[body.demo_query_index]
        rid = new_rid()
        runner = BenchmarkRunner(s.db, s.obfuscator, s.gnn, s.rl)
        try:
            res = await runner.run(sql, rid, body.variants, measure=body.measure)
        finally:
            s.obfuscator.clear_vault(rid)
        s.audit.record(rid, "BENCHMARK", stage="benchmark", details={"measured": res["measurement"].get("measured")})
        return {"request_id": rid, **res}

    @app.get("/api/v1/examples")
    async def examples():
        names = ["A - unindexed multi-table join", "B - date-range aggregation", "C - nested (correlated) subquery",
                 "D - non-sargable date predicate + PII literal", "E - efficient control query"]
        qs = load_demo_queries()
        return {"examples": [{"index": i, "name": names[i] if i < len(names) else f"query {i}", "sql": q}
                             for i, q in enumerate(qs)]}

    # ---------------------------------------------------------------- natural language
    @app.post("/api/v1/nl/ask", dependencies=[Depends(require_key)])
    async def nl_ask(body: NLRequest, request: Request):
        nl: NLIntentAssistant = request.app.state.nl
        nl.known_tables = services_(request).known_tables
        return await nl.ask(body.question)

    @app.post("/api/v1/nl/analyze", dependencies=[Depends(require_key)])
    async def nl_analyze(body: NLAnalyzeRequest, request: Request):
        sql = request.app.state.nl.resolve_candidate(body.candidate_id)
        if sql is None:
            raise QuerySentinelError(ErrorCategory.NOT_FOUND, "candidate expired or unknown", stage="nl")
        return await optimize_async(SQLRequest(sql=sql), request)

    return app


app = create_app()
