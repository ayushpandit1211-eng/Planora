"""Service container shared by the API, the LangGraph supervisor and the dashboard status bar.

Holds long-lived components (DB pool, vault, guard, models, event bus, audit trail, workload
store) plus a per-request TRUSTED store for data that must never enter the graph state or an
LLM payload (raw SQL, real schema, real DDL).
"""
from __future__ import annotations

import json
from typing import Any

from app.audit import AuditLogger, audit_logger
from app.capability_firewall import CapabilityFirewall
from app.config import Settings, get_settings
from app.contracts import AuditEvent
from app.db import Database
from app.events import EventBus, event_bus
from app.gnn.analyzer import GNNAnalyzer
from app.llm import LLMClient
from app.llm_guard import OutgoingPayloadGuard
from app.logging_setup import get_logger
from app.masking_vault import DataObfuscator, build_vault_backend
from app.migration_generator import MigrationGenerator
from app.postgres_instrumentation import PostgresInstrumentation
from app.rl.policy import RLPolicy
from app.workload_store import WorkloadRecord, WorkloadStore, record_to_row, workload_store

log = get_logger("pipeline")


class Services:
    def __init__(self, settings: Settings | None = None, db: Database | None = None,
                 llm: LLMClient | None = None, gnn: GNNAnalyzer | None = None, rl: RLPolicy | None = None,
                 audit: AuditLogger | None = None, events: EventBus | None = None,
                 store: WorkloadStore | None = None, instrumentation: PostgresInstrumentation | None = None,
                 obfuscator: DataObfuscator | None = None):
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings)
        self.instr = instrumentation or PostgresInstrumentation(self.db)
        self.audit = audit or audit_logger
        self.events = events or event_bus
        self.obfuscator = obfuscator or DataObfuscator(build_vault_backend(), known_tables=())
        self.guard = OutgoingPayloadGuard(self.obfuscator, self.audit)
        self.llm = llm or LLMClient(self.settings, guard=self.guard)
        if self.llm.guard is None:
            self.llm.guard = self.guard
        self.gnn = gnn or GNNAnalyzer()
        self.rl = rl or RLPolicy()
        self.store = store or workload_store
        self.firewall = CapabilityFirewall(self.audit)
        self.migrations = MigrationGenerator()
        self.trusted: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, dict[str, Any]] = {}
        self.known_tables: set[str] = set()
        from app.dp import accountant as _dp_acc          # Final Upgrade P10: PRIVACY_BUDGET_EXHAUSTED audit
        if _dp_acc.audit_hook is None:
            _dp_acc.audit_hook = lambda scope, snap: self.audit.record(
                f"dp-{scope}"[:64], "PRIVACY_BUDGET_EXHAUSTED", stage="dp", details=snap)

    # ------------------------------------------------------------------ Final Upgrade P5
    def for_source(self, ctx) -> "Services":
        """Services bound to one data source (demo returns self, so the demo path is untouched). Shares the
        vault backend, models, audit, events and reports; swaps db/instr, workload namespace and target schema."""
        if ctx is None or ctx.kind == "demo":
            return self
        import copy
        clone = copy.copy(self)
        clone.db, clone.instr = ctx.db, ctx.instr
        clone.store = self.store.scoped(ctx.id)
        clone.target_schema = ctx.schema_name
        clone.data_source_id = ctx.id
        clone.obfuscator = copy.copy(self.obfuscator)
        clone.guard = OutgoingPayloadGuard(clone.obfuscator, self.audit)
        clone.llm = copy.copy(self.llm)
        clone.llm.guard = clone.guard
        clone.known_tables = set(ctx.allowed_tables or [])
        clone.obfuscator.known_tables = set(clone.known_tables)
        return clone

    # ------------------------------------------------------------------ lifecycle
    async def startup(self) -> None:
        ok = await self.db.connect()
        if ok:
            self.audit.attach_persister(self._persist_audit)
            try:
                tables = await self.instr.tables(None)
                self.known_tables = {t["table_name"] for t in tables}
                self.obfuscator.known_tables = set(self.known_tables)
            except Exception as exc:  # noqa: BLE001
                log.warning("table discovery failed", extra={"error_category": type(exc).__name__})

    async def shutdown(self) -> None:
        await self.audit.flush()
        await self.db.close()

    # ------------------------------------------------------------------ persistence (optimizer_meta only)
    async def _persist_audit(self, ev: AuditEvent) -> None:
        await self.db.execute_write(
            "INSERT INTO optimizer_meta.audit_events (request_id, event_type, stage, severity, details) "
            "VALUES ($1, $2, $3, $4, $5::jsonb)",
            ev.request_id, ev.event_type, ev.stage, ev.severity, json.dumps(ev.details))

    async def persist_run(self, report: dict[str, Any], status: str, error_category: str | None = None) -> None:
        if not self.db.available:
            return
        try:
            opt = report.get("optimization", {}) if report else {}
            await self.db.execute_write(
                "INSERT INTO optimizer_meta.optimization_runs (request_id, query_fingerprint, processing_status, "
                "anonymized_sql, plan_fingerprint, recommendation_summary, sandbox_metrics, blast_radius_metrics, "
                "model_modes, error_category) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb,$9::jsonb,$10) "
                "ON CONFLICT (request_id) DO NOTHING",
                report.get("request_id"), report.get("query_fingerprint"), status,
                report.get("anonymized_sql"), (report.get("diagnosis") or {}).get("plan_fingerprint"),
                json.dumps({"proposed_indexes": [i.get("ddl") for i in opt.get("proposed_indexes", [])],
                            "rl_action": opt.get("rl_action"), "rewrite": bool(opt.get("rewritten_sql"))}),
                json.dumps((report.get("validation") or {}).get("hypopg_result") or {}),
                json.dumps(report.get("blast_radius") or []), json.dumps(report.get("model_modes") or {}),
                error_category)
            dep = report.get("deployment") or {}
            if dep.get("migration_id"):
                await self.db.execute_write(
                    "INSERT INTO optimizer_meta.migrations (migration_id, request_id, query_fingerprint, migration_sql, "
                    "rollback_sql, rationale, requires_human_review) VALUES ($1,$2,$3,$4,$5,$6,TRUE) "
                    "ON CONFLICT (migration_id) DO NOTHING",
                    dep["migration_id"], report.get("request_id"), report.get("query_fingerprint"),
                    dep.get("migration_sql", ""), dep.get("rollback_sql", ""), dep.get("rationale", ""))
        except Exception as exc:  # noqa: BLE001 - persistence must not break the response
            log.warning("run persistence failed", extra={"error_category": type(exc).__name__})

    async def persist_workload(self, recs: list[WorkloadRecord]) -> int:
        if not self.db.available or not recs:
            return 0
        n = 0
        for r in recs:
            try:
                await self.db.execute_write(
                    "INSERT INTO optimizer_meta.query_history (query_fingerprint, plan_fingerprint, plan_shape, "
                    "timestamp_bucket, call_count, mean_exec_time_ms, estimated_cost, estimated_rows, actual_rows, "
                    "operator_histogram, access_features, table_size_bucket, write_frequency_bucket, anonymized_sql, "
                    "source, recommendation_outcome) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,$12,$13,$14,$15,$16)",
                    *record_to_row(r))
                n += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("workload persistence failed", extra={"error_category": type(exc).__name__})
                break
        return n

    # ------------------------------------------------------------------ status (dashboard top bar)
    async def status(self) -> dict[str, Any]:
        health = await self.instr.health()
        vault_backend = type(self.obfuscator.backend).__name__
        return {
            "postgresql": health["postgresql"],
            "hypopg": health["hypopg"],
            "pg_stat_statements": health["pg_stat_statements"],
            "privacy_vault": {"ok": True, "backend": vault_backend,
                              "hmac_key_configured": bool(self.settings.privacy_hmac_key.get_secret_value())},
            "llm": {"ok": self.llm.enabled, **self.llm.status()},
            "gnn": {"ok": True, **self.gnn.status()},
            "rl": {"ok": True, **self.rl.status()},
            "langgraph": {"ok": True, "graph": "compiled StateGraph (see app/agents/supervisor.py)"},
            "workload_store": self.store.summary(),
            "settings": self.settings.redacted(),
        }
