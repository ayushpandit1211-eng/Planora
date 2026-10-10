"""DataSource context resolver (Final Upgrade Section 8.1).

Turns (data_source_id, session_id) into a validated `DataSourceContext`: kind, connection factory
(`Database` + `PostgresInstrumentation`), allowed schema / search_path, query role, limits, schema_version,
workload namespace, expiry and ownership. Nothing else opens connections for the upgrade features.

There is no global "active database": every new request carries a data_source_id; no id means the demo
database, which resolves without touching the registry so the existing behaviour is unchanged.
Session ownership is NOT authentication; it only keeps one browser session's sources apart from another's.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.postgres_instrumentation import PostgresInstrumentation

DEMO_ID = "demo"
NLQ_READER_ROLE = "qs_nlq_reader"
ID_RE = re.compile(r"^(demo|dyn_[a-z0-9]{6,32}|sbx_[a-z0-9]{6,32})$")
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

SQL_SOURCE = """
SELECT id, kind, name, owner_session_id, status, db_name, schema_name, role_name, schema_version,
       created_at, expires_at, config, (expires_at IS NOT NULL AND expires_at < now()) AS is_expired
FROM qs_meta.data_sources WHERE id = $1
"""
SQL_TABLES = """
SELECT t.table_name FROM qs_meta.dataset_tables t JOIN qs_meta.datasets d ON d.id = t.dataset_id
WHERE d.data_source_id = $1 ORDER BY t.table_name
"""


@dataclass
class DataSourceContext:
    id: str
    kind: str                       # demo | dynamic
    name: str
    schema_name: str
    db: Database
    instr: PostgresInstrumentation
    query_role: str | None          # role used for executing generated SQL (SET LOCAL ROLE) or login role
    schema_version: int = 1
    db_name: str | None = None
    owner_session_id: str | None = None
    expires_at: str | None = None
    status: str = "ready"
    allowed_tables: list[str] | None = None    # None = every table in schema_name
    limits: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def workload_namespace(self) -> str:
        return self.id

    def public_view(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "schema": self.schema_name,
                "schema_version": self.schema_version, "status": self.status, "expires_at": self.expires_at,
                "tables": self.allowed_tables}


def validate_source_row(row: dict | None, session_id: str | None) -> None:
    """Pure validation of a registry row (unit-testable). Raises with distinct error codes."""
    if row is None or row.get("status") == "deleted":
        raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "unknown data source", stage="datasource")
    if row.get("status") == "expired" or row.get("is_expired"):
        raise QuerySentinelError(ErrorCategory.SANDBOX_EXPIRED, "data source has expired", stage="datasource")
    owner = row.get("owner_session_id")
    if owner and owner != session_id and not get_settings().single_user:
        raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "data source belongs to another session",
                                 stage="datasource")
    if row.get("status") != "ready":
        raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, f"data source is {row.get('status')}",
                                 stage="datasource")


class DataSourceResolver:
    def __init__(self, main_db: Database, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.main_db = main_db
        self._pools: dict[str, Database] = {}
        self.sandbox_db_factory = None

    # ------------------------------------------------------------------ pools
    def dynamic_db(self) -> Database:
        key = f"schema:{self.settings.ingest_target_schema}"
        if key not in self._pools:
            # public stays on the path ONLY for the HypoPG functions (extension lives in public); generated SQL is
            # validated against the dynamic snapshot first and SQLSafetyGate pins search_path to qs_dynamic alone.
            self._pools[key] = Database(self.settings, server_settings={
                "search_path": f"{self.settings.ingest_target_schema}, public"}, max_size=4, label="dynamic")
        return self._pools[key]

    async def _ensure(self, db: Database) -> Database:
        if not db.available and not await db.connect():
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, "data source connection failed",
                                     stage="datasource")
        return db

    async def close(self) -> None:
        for db in list(self._pools.values()):
            await db.close()
        self._pools.clear()

    async def drop_pool(self, source_id: str) -> None:
        db = self._pools.pop(source_id, None)
        if db is not None:
            await db.close()
        from app.nlq_optimizer import forget_source
        forget_source(source_id)

    # ------------------------------------------------------------------ resolve
    def demo(self) -> DataSourceContext:
        return DataSourceContext(id=DEMO_ID, kind="demo", name="Demo Database", schema_name="public",
                                 db=self.main_db, instr=PostgresInstrumentation(self.main_db),
                                 query_role=NLQ_READER_ROLE)

    async def resolve(self, data_source_id: str | None = None, session_id: str | None = None) -> DataSourceContext:
        if data_source_id in (None, "", DEMO_ID):
            return self.demo()
        if not ID_RE.match(data_source_id):
            raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "unknown data source", stage="datasource")
        if not self.main_db.available:
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, "registry unavailable", stage="datasource")
        rows = await self.main_db.fetch(SQL_SOURCE, data_source_id)
        row = dict(rows[0]) if rows else None
        validate_source_row(row, session_id)
        assert row is not None
        try:
            await self.main_db.execute_write(
                "UPDATE qs_meta.data_sources SET last_accessed_at = now() WHERE id = $1", data_source_id)
        except Exception:  # noqa: BLE001 - access timestamp is best effort
            pass
        if row["kind"] == "dynamic":
            db = await self._ensure(self.dynamic_db())
            tables = [r["table_name"] for r in await self.main_db.fetch(SQL_TABLES, data_source_id)]
            return DataSourceContext(id=row["id"], kind="dynamic", name=row["name"], schema_name=row["schema_name"],
                                     db=db, instr=PostgresInstrumentation(db, row["schema_name"]),
                                     query_role=NLQ_READER_ROLE, schema_version=row["schema_version"],
                                     owner_session_id=row["owner_session_id"], allowed_tables=tables)
        if row["kind"] == "sandbox":
            if self.sandbox_db_factory is None:
                raise QuerySentinelError(ErrorCategory.SANDBOX_UNAVAILABLE,
                                         "sandbox connection factory is not configured", stage="datasource")
            if data_source_id not in self._pools:
                self._pools[data_source_id] = self.sandbox_db_factory(row)
            db = await self._ensure(self._pools[data_source_id])
            config = row.get("config") or {}
            if isinstance(config, str):
                config = json.loads(config)
            qualified = config.get("tables", [])
            schemas = {t.split(".", 1)[0] for t in qualified if "." in t}
            schema = next(iter(schemas)) if len(schemas) == 1 else "public"
            return DataSourceContext(
                id=row["id"], kind="sandbox", name=row["name"], schema_name=schema,
                db=db, instr=PostgresInstrumentation(db, schema), query_role=None,
                schema_version=row["schema_version"], db_name=row["db_name"],
                owner_session_id=row["owner_session_id"],
                expires_at=row["expires_at"].isoformat() if row.get("expires_at") else None,
                synthetic=bool(config.get("synthetic", False)),
                allowed_tables=[t.split(".", 1)[-1] for t in qualified])
        raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "unknown data source kind", stage="datasource")
