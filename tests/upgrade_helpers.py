"""Shared helpers for the Final Upgrade tests (new file; the pre-existing conftest is not modified)."""
from __future__ import annotations

import asyncio

import pytest

from tests.conftest import live_db_available  # noqa: F401  (also sets test env defaults)


def admin_db_available() -> bool:
    try:
        import psycopg2
        from app.config import get_settings
        st = get_settings()
        c = psycopg2.connect(**{k if k != "database" else "dbname": v for k, v in st.admin_dsn_kwargs.items()},
                             connect_timeout=2)
        c.close()
        return True
    except Exception:  # noqa: BLE001
        return False


requires_admin_db = pytest.mark.skipif(not admin_db_available(), reason="admin PostgreSQL not reachable (NOT RUN)")


def admin_exec(sql: str, *args):
    import psycopg2
    from app.config import get_settings
    st = get_settings()
    c = psycopg2.connect(**{k if k != "database" else "dbname": v for k, v in st.admin_dsn_kwargs.items()})
    c.autocommit = True
    try:
        with c.cursor() as cur:
            cur.execute(sql, args or None)
            try:
                return cur.fetchall()
            except psycopg2.ProgrammingError:
                return None
    finally:
        c.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def sandbox_pg_available() -> bool:
    """Check the explicitly configured disposable server, never fall back to the main DB."""
    from app.config import get_settings
    settings = get_settings()
    if not settings.sandbox_pg_host:
        return False
    try:
        import psycopg2
        kw = {("dbname" if k == "database" else k): v
              for k, v in settings.sandbox_admin_dsn_kwargs.items()}
        conn = psycopg2.connect(**kw, connect_timeout=2)
        conn.close()
        return True
    except psycopg2.Error:
        return False


requires_sandbox_pg = pytest.mark.skipif(
    not sandbox_pg_available(), reason="isolated sandbox PostgreSQL not configured/reachable (NOT RUN)")
