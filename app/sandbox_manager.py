"""BYOD sandbox manager (Final Upgrade P3).

Mandatory order: upload (untrusted) -> STRICT VERIFIER -> (only if verified) provision -> apply verified DDL
-> synthetic data -> analysis -> cleanup. Provisioning is never called when verification fails.

Isolation: a separate PostgreSQL service (sandbox-pg, internal network, no published port). Each sandbox is its
own database `sbx_<id>` with its own login role `sbx_<id>_r` (NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION,
CONNECTION LIMIT, CONNECT on its own database only, PUBLIC CONNECT revoked). Role passwords are HMAC-derived from
SANDBOX_SECRET and never returned or logged. Admin credentials are separate and used only here (HypoPG needs a
privileged CREATE EXTENSION; acceptable only because sandbox-pg is isolated and disposable).
This is logical isolation (database + role), not kernel-level isolation. Session ownership is not authentication.
No Docker socket, no Docker-in-Docker, no subprocess.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from typing import Any

import asyncpg

from app.config import Settings, get_settings
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.provenance import Provenance
from app.schema_verifier import verify_schema
from app.structured_log import TraceLogger
from app import synthetic_data

SCALES = ("none", "small", "medium", "large")


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SandboxManager:
    def __init__(self, main_db: Database, settings: Settings | None = None, resolver=None, store=None):
        self.settings = settings or get_settings()
        self.main_db = main_db           # registry (qs_meta) via the restricted app role
        self.resolver = resolver
        self.store = store               # in-memory workload store (purged by data_source_id on delete)
        self._task: asyncio.Task | None = None
        self.provision_calls = 0         # observable for the "never provision unverified" test

    # ------------------------------------------------------------------ helpers
    @property
    def configured(self) -> bool:
        return bool(self.settings.sandbox_pg_host)

    def _require(self) -> None:
        if not self.configured:
            raise QuerySentinelError(ErrorCategory.SANDBOX_UNAVAILABLE,
                                     "sandbox server is not configured (SANDBOX_PG_HOST)", stage="sandbox")

    def role_password(self, sandbox_id: str) -> str:
        # SANDBOX_SECRET; falls back to PRIVACY_HMAC_KEY / POSTGRES_PASSWORD so an existing .env keeps working
        s = self.settings
        key = (s.sandbox_secret.get_secret_value() or s.privacy_hmac_key.get_secret_value()
               or s.postgres_password.get_secret_value())
        if not key:
            raise QuerySentinelError(ErrorCategory.SANDBOX_UNAVAILABLE, "SANDBOX_SECRET is not set", stage="sandbox")
        return hmac.new(key.encode(), sandbox_id.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def role_name(sandbox_id: str) -> str:
        return f"{sandbox_id}_r"

    @staticmethod
    def schema_of(row: dict) -> str:
        """Schema holding the sandbox tables: public, or the single schema the upload created (e.g. shop)."""
        cfg = row.get("config") or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg or "{}")
        schemas = {t.split(".", 1)[0] for t in cfg.get("tables", []) if "." in t}
        return schemas.pop() if len(schemas) == 1 else "public"

    def db_factory(self, row: dict) -> Database:
        """Connection factory used by the DataSourceResolver for kind=sandbox (sandbox role, own database)."""
        s = self.settings
        sch = self.schema_of(row)
        return Database(s, dsn_kwargs={"host": s.sandbox_pg_host, "port": s.sandbox_pg_port,
                                       "database": row["db_name"], "user": row["role_name"],
                                       "password": self.role_password(row["id"])},
                        server_settings={"search_path": f'"{sch}", public'} if sch != "public" else None,
                        max_size=2, label="sandbox")

    async def _admin(self, database: str = "postgres") -> asyncpg.Connection:
        kw = dict(self.settings.sandbox_admin_dsn_kwargs)
        kw["database"] = database
        try:
            return await asyncpg.connect(**kw, timeout=self.settings.db_connect_timeout_s)
        except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
            raise QuerySentinelError(ErrorCategory.SANDBOX_UNAVAILABLE,
                                     f"sandbox server unreachable ({type(exc).__name__})", stage="sandbox") from None

    async def _audit(self, source_id: str | None, session_id: str | None, event: str, details: dict) -> None:
        try:
            await self.main_db.execute_write(
                "INSERT INTO qs_meta.audit_log (data_source_id, session_id, event_type, details) "
                "VALUES ($1, $2, $3, $4::jsonb)", source_id, session_id, event, json.dumps(details))
        except Exception:  # noqa: BLE001 - audit must not break lifecycle operations
            pass

    # ------------------------------------------------------------------ verify only
    def verify(self, text: str) -> dict[str, Any]:
        s = self.settings
        return verify_schema(text, s.sandbox_max_statements, int(s.sandbox_max_upload_mb * 1024 * 1024)).to_dict()

    # ------------------------------------------------------------------ create
    async def create(self, session_id: str, schema_text: str, name: str = "BYOD sandbox", scale: str = "small",
                     ttl_minutes: float | None = None) -> dict[str, Any]:
        s = self.settings
        log = TraceLogger("sandbox")
        if scale not in SCALES:
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"scale must be one of {SCALES}", stage="sandbox")
        rep = verify_schema(schema_text, s.sandbox_max_statements, int(s.sandbox_max_upload_mb * 1024 * 1024))
        if not rep.verified:
            await self._audit(None, session_id, "BYOD_VERIFICATION_REJECTED",
                              {"rejected": len(rep.rejected_statements), "unsupported": len(rep.unsupported_features)})
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "schema verification failed; nothing was "
                                     "provisioned", stage="byod_verify", details={"verification": rep.to_dict()})
        self._require()
        active_session = await self.main_db.fetchval(
            "SELECT count(*) FROM qs_meta.data_sources WHERE kind = 'sandbox' AND status IN ('creating', 'ready') "
            "AND owner_session_id = $1", session_id)
        active_global = await self.main_db.fetchval(
            "SELECT count(*) FROM qs_meta.data_sources WHERE kind = 'sandbox' AND status IN ('creating', 'ready')")
        if active_session >= s.sandbox_max_active_per_session or active_global >= s.sandbox_max_active_global:
            raise QuerySentinelError(ErrorCategory.QUOTA_EXCEEDED, "sandbox quota reached (per session "
                                     f"{s.sandbox_max_active_per_session}, global {s.sandbox_max_active_global})",
                                     stage="sandbox")
        sid = f"sbx_{secrets.token_hex(6)}"
        role = self.role_name(sid)
        ttl = ttl_minutes if ttl_minutes is not None else s.sandbox_ttl_minutes
        await self.main_db.execute_write(
            "INSERT INTO qs_meta.data_sources (id, kind, name, owner_session_id, status, db_name, schema_name, "
            "role_name, expires_at, config) VALUES ($1, 'sandbox', $2, $3, 'creating', $1, 'public', $4, "
            "now() + make_interval(secs => $5), $6::jsonb)",
            sid, name[:100], session_id, role, float(ttl) * 60.0,
            json.dumps({"scale": scale, "synthetic": scale != "none", "tables": rep.tables}))
        log.data_source_id = sid
        try:
            synth = await self._provision(sid, role, rep.apply_statements, scale)
        except QuerySentinelError:
            await self._mark(sid, "failed")
            await self._drop(sid, role)
            raise
        except Exception as exc:  # noqa: BLE001
            await self._mark(sid, "failed")
            await self._drop(sid, role)
            log.error("sandbox provisioning failed", error_category=type(exc).__name__, stage="sandbox")
            raise QuerySentinelError(ErrorCategory.SANDBOX_FAILURE, f"provisioning failed ({type(exc).__name__})",
                                     stage="sandbox") from None
        await self._mark(sid, "ready")
        await self._audit(sid, session_id, "SANDBOX_CREATED", {"tables": len(rep.tables), "scale": scale})
        log.info("sandbox ready", stage="sandbox")
        row = await self.get(sid, session_id)
        return {**row, "verification": rep.to_dict(), "synthetic_data": synth}

    async def _mark(self, sid: str, status: str) -> None:
        await self.main_db.execute_write("UPDATE qs_meta.data_sources SET status = $2 WHERE id = $1", sid, status)

    async def _provision(self, sid: str, role: str, statements: list[str], scale: str) -> dict[str, Any]:
        s = self.settings
        self.provision_calls += 1
        pw = self.role_password(sid)
        admin = await self._admin()
        try:
            await admin.execute(
                f"CREATE ROLE {qi(role)} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT "
                f"CONNECTION LIMIT {int(s.sandbox_connection_limit)} PASSWORD '{pw}'")
            for k, v in (("statement_timeout", int(s.sandbox_statement_timeout_ms)),
                         ("idle_in_transaction_session_timeout", 60000), ("lock_timeout", 5000),
                         ("temp_file_limit", f"'{int(s.sandbox_max_db_size_mb)}MB'"), ("work_mem", "'16MB'")):
                await admin.execute(f"ALTER ROLE {qi(role)} SET {k} = {v}")
            await admin.execute(f"CREATE DATABASE {qi(sid)} TEMPLATE template0 ENCODING 'UTF8'")
            await admin.execute(f"REVOKE ALL ON DATABASE {qi(sid)} FROM PUBLIC")
            await admin.execute(f"GRANT CONNECT, TEMPORARY, CREATE ON DATABASE {qi(sid)} TO {qi(role)}")
        finally:
            await admin.close()
        admin_db = await self._admin(sid)
        try:
            await admin_db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
            await admin_db.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {qi(role)}")
            await admin_db.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
            await admin_db.execute(f"GRANT SELECT ON hypopg_list_indexes TO {qi(role)}")
        finally:
            await admin_db.close()
        conn = await asyncpg.connect(host=s.sandbox_pg_host, port=s.sandbox_pg_port, database=sid, user=role,
                                     password=pw, timeout=s.db_connect_timeout_s)
        try:
            async with conn.transaction():
                for stmt in statements:            # verified DDL only, executed as the sandbox role
                    await conn.execute(stmt)
            await conn.execute("SET search_path = public")
            if scale != "none":
                rows = getattr(s, f"synthetic_rows_{scale}")
                synth = await synthetic_data.generate(conn, rows)
            else:
                synth = {"rows_requested_per_table": 0, "tables": {}, "unsatisfied": [],
                         "note": "schema-only sandbox: plans use default statistics (no data)", "provenance": []}
            size_mb = (await conn.fetchval("SELECT pg_database_size(current_database())")) / 1048576
            if size_mb > s.sandbox_max_db_size_mb:
                raise QuerySentinelError(ErrorCategory.QUOTA_EXCEEDED, f"sandbox size {size_mb:.0f} MB exceeds "
                                         f"SANDBOX_MAX_DB_SIZE_MB ({s.sandbox_max_db_size_mb})", stage="sandbox")
            synth["database_size_mb"] = round(size_mb, 1)
            return synth
        finally:
            await conn.close()

    async def _drop(self, sid: str, role: str | None = None) -> None:
        """Idempotent teardown: terminate sessions, drop database and role."""
        role = role or self.role_name(sid)
        if self.resolver is not None:
            await self.resolver.drop_pool(sid)
        admin = await self._admin()
        try:
            await admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 "
                                "AND pid <> pg_backend_pid()", sid)
            await admin.execute(f"DROP DATABASE IF EXISTS {qi(sid)} WITH (FORCE)")
            await admin.execute(f"DROP ROLE IF EXISTS {qi(role)}")
        finally:
            await admin.close()

    async def _purge_workload(self, sid: str) -> None:
        try:
            await self.main_db.execute_write("DELETE FROM optimizer_meta.query_history WHERE data_source_id = $1", sid)
        except Exception:  # noqa: BLE001
            pass
        if self.store is not None and hasattr(self.store, "purge_source"):
            self.store.purge_source(sid)

    # ------------------------------------------------------------------ read / extend / delete
    async def get(self, sid: str, session_id: str | None) -> dict[str, Any]:
        rows = await self.main_db.fetch(
            "SELECT id, name, status, owner_session_id, created_at, expires_at, extensions_used, config, "
            "GREATEST(0, EXTRACT(EPOCH FROM (expires_at - now())))::int AS seconds_left "
            "FROM qs_meta.data_sources WHERE id = $1 AND kind = 'sandbox'", sid)
        if not rows or rows[0]["status"] == "deleted":
            raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "unknown sandbox", stage="sandbox")
        r = dict(rows[0])
        if r["owner_session_id"] != session_id and not self.settings.single_user:
            raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "sandbox belongs to another session",
                                     stage="sandbox")
        cfg = r["config"] if isinstance(r["config"], dict) else json.loads(r["config"] or "{}")
        return {"data_source_id": r["id"], "name": r["name"], "status": r["status"],
                "created_at": r["created_at"].isoformat(), "expires_at": r["expires_at"].isoformat(),
                "seconds_left": r["seconds_left"], "extensions_used": r["extensions_used"],
                "extensions_allowed": self.settings.sandbox_max_extensions, "tables": cfg.get("tables", []),
                "scale": cfg.get("scale"), "synthetic": cfg.get("synthetic", False),
                "provenance": [Provenance.SYNTHETIC_DATA.value] if cfg.get("synthetic") else [],
                "ownership_note": "session ownership, not authentication"}

    async def list(self, session_id: str) -> list[dict[str, Any]]:
        rows = await self.main_db.fetch("SELECT id FROM qs_meta.data_sources WHERE kind = 'sandbox' AND "
                                        "($2 OR owner_session_id = $1) AND status IN ('creating', 'ready') ORDER BY "
                                        "created_at DESC", session_id, self.settings.single_user)
        return [await self.get(r["id"], session_id) for r in rows]

    async def extend(self, sid: str, session_id: str) -> dict[str, Any]:
        cur = await self.get(sid, session_id)
        if cur["status"] != "ready" or cur["seconds_left"] <= 0:
            raise QuerySentinelError(ErrorCategory.SANDBOX_EXPIRED, "sandbox has expired", stage="sandbox")
        if cur["extensions_used"] >= self.settings.sandbox_max_extensions:
            raise QuerySentinelError(ErrorCategory.QUOTA_EXCEEDED, "maximum TTL extensions reached", stage="sandbox")
        await self.main_db.execute_write(
            "UPDATE qs_meta.data_sources SET expires_at = expires_at + make_interval(secs => $2), "
            "extensions_used = extensions_used + 1 WHERE id = $1", sid, self.settings.sandbox_ttl_minutes * 60.0)
        await self._audit(sid, session_id, "SANDBOX_EXTENDED", {})
        return await self.get(sid, session_id)

    async def delete(self, sid: str, session_id: str) -> dict[str, Any]:
        await self.get(sid, session_id)            # ownership check
        await self._drop(sid)
        await self.main_db.execute_write("UPDATE qs_meta.data_sources SET status = 'deleted' WHERE id = $1", sid)
        await self._purge_workload(sid)
        await self._audit(sid, session_id, "SANDBOX_DELETED", {"reason": "explicit"})
        return {"data_source_id": sid, "status": "deleted"}

    # ------------------------------------------------------------------ reaper / sweep
    async def reap_expired(self) -> list[str]:
        """Drop expired sandboxes. Row locking (FOR UPDATE SKIP LOCKED) means two reapers never collide."""
        if not self.configured or not self.main_db.available:
            return []
        reaped: list[str] = []
        async with self.main_db.connection() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    "SELECT id, role_name FROM qs_meta.data_sources WHERE kind = 'sandbox' AND status IN "
                    "('creating', 'ready', 'failed') AND expires_at < now() FOR UPDATE SKIP LOCKED")
                for r in rows:
                    try:
                        await self._drop(r["id"], r["role_name"])
                    except QuerySentinelError:
                        continue
                    await conn.execute("UPDATE qs_meta.data_sources SET status = 'expired' WHERE id = $1", r["id"])
                    reaped.append(r["id"])
        for sid in reaped:
            await self._purge_workload(sid)
            await self._audit(sid, None, "SANDBOX_EXPIRED", {"reason": "ttl"})
        return reaped

    async def startup_sweep(self) -> dict[str, list[str]]:
        """Drop orphan sandbox databases/roles with no live registry row; fix registry rows whose DB is gone."""
        out: dict[str, list[str]] = {"dropped_orphans": [], "fixed_registry": []}
        if not self.configured or not self.main_db.available:
            return out
        live = {r["id"] for r in await self.main_db.fetch(
            "SELECT id FROM qs_meta.data_sources WHERE kind = 'sandbox' AND status IN ('creating', 'ready')")}
        admin = await self._admin()
        try:
            dbs = {r["datname"] for r in await admin.fetch(
                "SELECT datname FROM pg_database WHERE datname ~ '^sbx_[a-z0-9]+$'")}
            roles = {r["rolname"] for r in await admin.fetch(
                "SELECT rolname FROM pg_roles WHERE rolname ~ '^sbx_[a-z0-9]+_r$'")}
        finally:
            await admin.close()
        for db in sorted(dbs - live):
            await self._drop(db)
            out["dropped_orphans"].append(db)
        for role in sorted(roles):
            sid = role[:-2]
            if sid not in live and sid not in dbs:
                await self._drop(sid, role)
                out["dropped_orphans"].append(role)
        for sid in sorted(live - dbs):
            await self.main_db.execute_write("UPDATE qs_meta.data_sources SET status = 'expired' WHERE id = $1", sid)
            out["fixed_registry"].append(sid)
        return out

    def start(self) -> None:
        if self._task is None and self.configured:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        log = TraceLogger("sandbox_reaper")
        try:
            await self.startup_sweep()
        except Exception as exc:  # noqa: BLE001
            log.warning("startup sweep failed", error_category=type(exc).__name__)
        while True:
            await asyncio.sleep(max(self.settings.sandbox_cleanup_interval_seconds, 1.0))
            try:
                await self.reap_expired()
            except Exception as exc:  # noqa: BLE001 - the reaper must survive transient failures
                log.warning("reaper iteration failed", error_category=type(exc).__name__)
