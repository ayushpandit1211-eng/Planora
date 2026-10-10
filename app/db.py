"""Async PostgreSQL connection lifecycle (Section 40).

asyncpg pool created at application startup and closed at shutdown. Connections are always
acquired through context managers (never leaked). Analysis statements run in READ ONLY
transactions with SET LOCAL statement_timeout. Startup retries with backoff; per-call
timeouts map to structured errors.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from app.config import Settings, get_settings
from app.errors import ErrorCategory, QuerySentinelError
from app.logging_setup import get_logger
from app.security import explain_command

log = get_logger("db")


class Database:
    def __init__(self, settings: Settings | None = None, dsn_kwargs: dict | None = None,
                 server_settings: dict | None = None, max_size: int | None = None, label: str = "demo"):
        """`dsn_kwargs` / `server_settings` are optional (Final Upgrade P1): when absent the pool connects as
        the restricted application role to the demo database exactly as before."""
        self.settings = settings or get_settings()
        self.pool: asyncpg.Pool | None = None
        self.last_error: str | None = None
        self._dsn_kwargs = dsn_kwargs
        self._server_settings = server_settings
        self._max_size = max_size
        self.label = label

    @property
    def available(self) -> bool:
        return self.pool is not None

    async def connect(self) -> bool:
        s = self.settings
        delay = 1.0
        for attempt in range(1, s.db_connect_retries + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    **(self._dsn_kwargs or s.app_dsn_kwargs),
                    min_size=min(s.db_pool_min_size, self._max_size or s.db_pool_max_size),
                    max_size=self._max_size or s.db_pool_max_size,
                    timeout=s.db_connect_timeout_s, command_timeout=s.statement_timeout_ms / 1000 + 5,
                    server_settings={"application_name": "querysentinel", **(self._server_settings or {})},
                )
                self.last_error = None
                log.info("database pool created", extra={"stage": "db"})
                return True
            except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
                self.last_error = type(exc).__name__
                log.warning("database connection attempt failed",
                            extra={"stage": "db", "error_category": self.last_error})
                if attempt < s.db_connect_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8)
        return False

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    def _require(self) -> asyncpg.Pool:
        if self.pool is None:
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, "database unavailable", stage="db")
        return self.pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        pool = self._require()
        try:
            conn = await pool.acquire(timeout=self.settings.db_connect_timeout_s)
        except (OSError, asyncio.TimeoutError, asyncpg.PostgresError) as exc:
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, f"could not acquire connection ({type(exc).__name__})",
                                     stage="db") from None
        try:
            yield conn
        finally:
            await pool.release(conn)

    @asynccontextmanager
    async def read_only(self, timeout_ms: int | None = None) -> AsyncIterator[asyncpg.Connection]:
        """Connection inside a READ ONLY transaction with a local statement timeout."""
        async with self.connection() as conn:
            tx = conn.transaction(readonly=True)
            await tx.start()
            try:
                await conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms or self.settings.statement_timeout_ms)}")
                yield conn
            finally:
                await tx.rollback()   # analysis never commits anything

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        try:
            async with self.read_only() as conn:
                return await conn.fetch(sql, *args)
        except asyncpg.QueryCanceledError:
            raise QuerySentinelError(ErrorCategory.DB_UNAVAILABLE, "statement timeout", stage="db") from None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        async with self.read_only() as conn:
            return await conn.fetchval(sql, *args)

    async def execute_write(self, sql: str, *args: Any) -> None:
        """Writes to optimizer_meta only (role has no business-table write privilege)."""
        async with self.connection() as conn:
            await conn.execute(sql, *args)

    async def explain(self, validated_sql: str, conn: asyncpg.Connection | None = None,
                      generic: bool = False) -> dict:
        """EXPLAIN (FORMAT JSON, VERBOSE TRUE, BUFFERS TRUE) - never ANALYZE."""
        if conn is not None:
            return await explain_on(conn, validated_sql, generic)
        try:
            async with self.read_only() as c:
                return await explain_on(c, validated_sql, generic)
        except asyncpg.PostgresError as exc:
            raise QuerySentinelError(ErrorCategory.INVALID_SQL,
                                     f"PostgreSQL rejected the query during EXPLAIN ({exc.sqlstate})",
                                     stage="explain", details={"sqlstate": exc.sqlstate}) from None


async def explain_on(conn: asyncpg.Connection, validated_sql: str, generic: bool = False) -> dict:
    cmd = explain_command(validated_sql, generic)
    if generic:
        # EXPLAIN (GENERIC_PLAN) must reach the server WITHOUT bound parameters: with asyncpg's
        # extended protocol any bound values (even NULLs) turn it into a custom plan
        # ("tenant_id = NULL" -> one-time FALSE -> cost ~0). Run it as a dynamic statement with no
        # USING clause inside a DO block, which is exactly what psql's simple protocol does.
        # The validated SQL is embedded verbatim between a random dollar-quote tag it cannot contain.
        tag = "qs" + secrets.token_hex(6)
        while f"${tag}$" in cmd:
            tag = "qs" + secrets.token_hex(6)
        await conn.execute(
            f"DO ${tag}$ DECLARE p text; BEGIN EXECUTE ${tag}x$" + cmd + f"${tag}x$ INTO p; "
            f"PERFORM set_config('qs.generic_plan', p, true); END ${tag}$")
        raw = await conn.fetchval("SELECT current_setting('qs.generic_plan')")
    else:
        raw = await conn.fetchval(cmd)
    doc = json.loads(raw) if isinstance(raw, str) else raw
    return doc[0] if isinstance(doc, list) else doc
