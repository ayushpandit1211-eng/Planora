"""Runs the additive qs_meta migration (app/migrations/*.sql) with admin credentials.

Called from the backend lifespan after the existing database initialisation, and by tests.
Idempotent: every statement is IF NOT EXISTS / ON CONFLICT DO NOTHING, executed in one transaction under
an advisory lock so two backends starting together never interleave.
"""
from __future__ import annotations

from pathlib import Path

import asyncpg

from app.config import Settings, get_settings
from app.logging_setup import get_logger

log = get_logger("migration")
MIGRATIONS = sorted((Path(__file__).parent / "migrations").glob("*.sql"))
_LOCK_KEY = 74_152_001


async def run_meta_migration(settings: Settings | None = None, conn: asyncpg.Connection | None = None) -> list[str]:
    s = settings or get_settings()
    own = conn is None
    if own:
        conn = await asyncpg.connect(**s.admin_dsn_kwargs, timeout=s.db_connect_timeout_s)
    applied: list[str] = []
    try:
        async with conn.transaction():
            await conn.execute(f"SELECT pg_advisory_xact_lock({_LOCK_KEY})")
            for f in MIGRATIONS:
                await conn.execute(f.read_text())
                applied.append(f.name)
        log.info("metadata migration applied", extra={"stage": "migration"})
        return applied
    finally:
        if own:
            await conn.close()
