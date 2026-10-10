"""P1 foundation: migration, resolver, provenance, flags."""
from __future__ import annotations

import pytest

from app.datasource import DataSourceResolver, validate_source_row
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.provenance import Provenance, badge, has_provenance, tagged
from tests.upgrade_helpers import admin_exec, requires_admin_db


def test_provenance_serializes_tags():
    v = tagged(12.5, Provenance.PLANNER_ESTIMATE, "HYPOPG SIMULATION", unit="pct")
    assert v == {"value": 12.5, "provenance": ["PLANNER ESTIMATE", "HYPOPG SIMULATION"], "unit": "pct"}
    assert has_provenance(v)
    assert badge(Provenance.DP_PROTECTED) == "[DP-PROTECTED]"
    with pytest.raises(ValueError):
        tagged(1, "MADE UP")
    with pytest.raises(ValueError):
        tagged(1)
    assert not has_provenance({"value": 1})


def test_feature_flags_snapshot_defaults():
    from app import feature_flags
    snap = feature_flags.snapshot()
    assert snap["ingestion"] and snap["nlq"]
    assert set(snap) == {"ingestion", "nlq", "gatv2", "rl_upgrades", "dp", "index_portfolio"}


@pytest.mark.parametrize("row,session,code", [
    (None, "s1", ErrorCategory.SOURCE_NOT_FOUND),
    ({"status": "deleted"}, "s1", ErrorCategory.SOURCE_NOT_FOUND),
    ({"status": "ready", "is_expired": True}, "s1", ErrorCategory.SANDBOX_EXPIRED),
    ({"status": "expired"}, "s1", ErrorCategory.SANDBOX_EXPIRED),
    ({"status": "ready", "owner_session_id": "other-session"}, "s1-session", ErrorCategory.SOURCE_FORBIDDEN),
])
def test_validate_source_row_distinct_codes(row, session, code):
    with pytest.raises(QuerySentinelError) as ei:
        validate_source_row(row, session)
    assert ei.value.category == code


async def test_resolver_defaults_to_demo_without_registry():
    r = DataSourceResolver(Database())      # pool never opened: demo needs no registry lookup
    for sid in (None, "", "demo"):
        ctx = await r.resolve(sid)
        assert ctx.id == "demo" and ctx.kind == "demo" and ctx.schema_name == "public"
        assert ctx.instr.schema == "public"


async def test_resolver_rejects_malformed_id():
    with pytest.raises(QuerySentinelError) as ei:
        await DataSourceResolver(Database()).resolve("x'; drop table t;--")
    assert ei.value.category == ErrorCategory.SOURCE_NOT_FOUND


@requires_admin_db
async def test_migration_runs_twice_and_resolver_codes_live():
    from app.meta_migration import run_meta_migration
    assert await run_meta_migration() == ["001_qs_meta.sql"]
    assert await run_meta_migration() == ["001_qs_meta.sql"]
    names = {r[0] for r in admin_exec("SELECT table_name FROM information_schema.tables WHERE table_schema='qs_meta'")}
    assert {"data_sources", "datasets", "dataset_tables", "dataset_columns", "schema_versions",
            "ingestion_runs", "audit_log", "query_runs"} <= names
    assert admin_exec("SELECT count(*) FROM optimizer_meta.query_history WHERE data_source_id IS NULL")[0][0] == 0
    admin_exec("DELETE FROM qs_meta.data_sources WHERE id IN ('dyn_p1testexp', 'dyn_p1testown')")
    admin_exec("INSERT INTO qs_meta.data_sources (id, kind, name, schema_name, expires_at) "
               "VALUES ('dyn_p1testexp', 'dynamic', 't', 'qs_dynamic', now() - interval '1 minute')")
    admin_exec("INSERT INTO qs_meta.data_sources (id, kind, name, schema_name, owner_session_id) "
               "VALUES ('dyn_p1testown', 'dynamic', 't', 'qs_dynamic', 'owner-session-1')")
    db = Database()
    assert await db.connect()
    r = DataSourceResolver(db)
    try:
        for sid, sess, code in [("dyn_p1nothere", None, ErrorCategory.SOURCE_NOT_FOUND),
                                ("dyn_p1testexp", None, ErrorCategory.SANDBOX_EXPIRED),
                                ("dyn_p1testown", "other-session-2", ErrorCategory.SOURCE_FORBIDDEN)]:
            with pytest.raises(QuerySentinelError) as ei:
                await r.resolve(sid, sess)
            assert ei.value.category == code
        ctx = await r.resolve("dyn_p1testown", "owner-session-1")
        assert ctx.kind == "dynamic" and ctx.instr.schema == "qs_dynamic" and ctx.allowed_tables == []
    finally:
        await r.close()
        await db.close()
        admin_exec("DELETE FROM qs_meta.data_sources WHERE id IN ('dyn_p1testexp', 'dyn_p1testown')")


def test_single_user_mode_skips_ownership_only_when_enabled():
    import os
    from app.config import reset_settings_cache
    row = {"status": "ready", "owner_session_id": "owner-session-1"}
    with pytest.raises(QuerySentinelError):
        validate_source_row(row, "other-session-2")
    os.environ["QS_SESSION_MODE"] = "single_user"
    reset_settings_cache()
    try:
        validate_source_row(row, "other-session-2")          # same local user, any tab
        with pytest.raises(QuerySentinelError):
            validate_source_row({"status": "ready", "is_expired": True}, None)   # expiry still enforced
    finally:
        os.environ.pop("QS_SESSION_MODE")
        reset_settings_cache()
