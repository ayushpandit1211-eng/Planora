"""Shared fixtures. Tests never need a production environment and never call a real LLM.

Offline fakes replay catalog metadata and EXPLAIN plans captured from the demo database
(tests/fixtures/*.json, re-capturable with tests/fixtures/capture_fixtures.py). Tests marked
`integration` run against a live PostgreSQL+HypoPG only when one is reachable.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PRIVACY_HMAC_KEY", "test-key-not-secret")
os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("BENCHMARK_ENABLED", "false")

from app.config import reset_settings_cache  # noqa: E402

reset_settings_cache()

from app.audit import AuditLogger  # noqa: E402
from app.benchmark import load_demo_queries  # noqa: E402
from app.events import EventBus  # noqa: E402
from app.masking_vault import DataObfuscator, InMemoryVaultBackend  # noqa: E402
from app.workload_store import WorkloadStore  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
CATALOG = json.loads((FIX / "demo_catalog.json").read_text())
PLANS = json.loads((FIX / "demo_plans.json").read_text())
DEMO = load_demo_queries()


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


DEMO_NORM = {_norm(q): i for i, q in enumerate(DEMO)}


def plan_for(sql: str, with_index: bool = False) -> dict:
    i = DEMO_NORM.get(_norm(sql))
    if i is None:
        return copy.deepcopy(PLANS["scenario_4"])
    if with_index and i == 2:
        return copy.deepcopy(PLANS["scenario_2_with_fk_index"])
    return copy.deepcopy(PLANS[f"scenario_{i}"])


class FakeInstrumentation:
    """Serves captured catalog rows / plans with the PostgresInstrumentation interface."""

    def __init__(self, statements: list[dict] | None = None):
        self.statements_rows = statements if statements is not None else CATALOG["write_statements"]

    @staticmethod
    def _f(rows, names, key="table_name"):
        return [dict(r) for r in rows if names is None or r[key] in names]

    async def health(self):
        return {"postgresql": {"ok": True, "version": "16 (fake)"}, "hypopg": {"ok": True, "version": "1.4.0"},
                "pg_stat_statements": {"ok": True, "preloaded": True}}

    async def tables(self, names=None):
        return self._f(CATALOG["tables"], names)

    async def columns(self, names=None):
        return self._f(CATALOG["columns"], names)

    async def indexes(self, names=None):
        return self._f(CATALOG["indexes"], names)

    async def foreign_keys(self, names=None):
        return [dict(r) for r in CATALOG["foreign_keys"]
                if names is None or r["table_name"] in names or r["ref_table"] in names]

    async def primary_keys(self):
        return dict(CATALOG["primary_keys"])

    async def statements(self, limit=200):
        return [dict(s, queryid="1", shared_blks_hit=0, shared_blks_read=0) for s in self.statements_rows][:limit]

    async def explain(self, sql, generic=False):
        return plan_for(sql)


class FakeConn:
    """Mimics the asyncpg calls HypoPGSandbox makes, including hypothetical-index state."""

    def __init__(self, fail_on_explain: bool = False):
        self.created: list[tuple[int, str, str]] = []
        self.resets = 0
        self.fail_on_explain = fail_on_explain

    async def execute(self, sql, *args):
        if "hypopg_reset" in sql:
            self.created.clear()
            self.resets += 1
        return "OK"

    async def fetchrow(self, sql, *args):
        oid = 900 + len(self.created) + self.resets * 10
        name = f"<{oid}>btree_hypo"
        self.created.append((oid, name, args[0] if args else ""))
        return {"indexrelid": oid, "indexname": name}

    async def fetchval(self, sql, *args):
        if "hypopg_relation_size" in sql:
            return 5 * 1048576
        if sql.startswith("EXPLAIN"):
            if self.fail_on_explain:
                import asyncpg
                raise asyncpg.exceptions.SyntaxOrAccessError("boom")
            query = sql.split(") ", 1)[1]
            fk = [c for c in self.created if "order_items" in c[2] and "(order_id)" in c[2]]
            doc = plan_for(query, with_index=bool(fk))
            if fk:
                txt = json.dumps(doc).replace("<13605>btree_order_items_order_id", fk[0][1])
                doc = json.loads(txt)
            return json.dumps([doc])
        return 0


class FakeDB:
    def __init__(self, conn: FakeConn | None = None):
        self.conn = conn or FakeConn()
        self.available = True
        self.last_error = None
        self.writes: list[tuple] = []

    async def connect(self):
        return True

    async def close(self):
        pass

    @asynccontextmanager
    async def read_only(self, timeout_ms=None):
        yield self.conn

    async def explain(self, sql, conn=None, generic=False):
        return plan_for(sql)

    async def execute_write(self, sql, *args):
        self.writes.append((sql.split()[2] if len(sql.split()) > 2 else sql, args))

    async def fetch(self, sql, *args):
        return []


@pytest.fixture
def obfuscator():
    return DataObfuscator(InMemoryVaultBackend(), hmac_key=b"k" * 32,
                          known_tables=["orders", "order_items", "tenants"])


@pytest.fixture
def fake_services():
    from app.gnn.analyzer import GNNAnalyzer
    from app.llm import LLMClient
    from app.pipeline import Services
    from app.rl.policy import RLPolicy
    db = FakeDB()
    s = Services(db=db, instrumentation=FakeInstrumentation(), audit=AuditLogger(), events=EventBus(),
                 store=WorkloadStore(),
                 obfuscator=DataObfuscator(InMemoryVaultBackend(), hmac_key=b"k" * 32,
                                           known_tables=["orders", "order_items", "tenants"]),
                 gnn=GNNAnalyzer(), rl=RLPolicy())
    s.known_tables = {"orders", "order_items", "tenants"}
    s.llm = LLMClient(s.settings, guard=s.guard)
    return s


def live_db_available() -> bool:
    try:
        import psycopg2
        from app.config import get_settings
        st = get_settings()
        c = psycopg2.connect(host=st.postgres_host, port=st.postgres_port, dbname=st.postgres_db,
                             user=st.qs_app_user, password=st.qs_app_password.get_secret_value() or None,
                             connect_timeout=2)
        c.close()
        return True
    except Exception:  # noqa: BLE001
        return False


requires_db = pytest.mark.skipif(not live_db_available(), reason="live PostgreSQL+HypoPG not reachable")
