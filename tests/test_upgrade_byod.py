"""P3 BYOD: strict verifier, sandbox-pg provisioning, isolation, synthetic data, TTL cleanup."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import asyncpg
import pytest

from app.config import get_settings
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.sandbox_manager import SandboxManager
from app.schema_verifier import verify_schema
from tests.upgrade_helpers import admin_db_available, requires_sandbox_pg

SHOP = (Path(__file__).resolve().parent.parent / "examples" / "byod" / "shop_schema.sql").read_text()


# ----------------------------------------------------------------------------- verifier (pure)
def test_pg_dump_style_file_verifies_with_reported_normalisation():
    r = verify_schema(SHOP)
    assert r.verified, r.to_dict()
    d = r.to_dict()
    assert d["object_counts"]["tables"] == 4 and d["schema_objects"] >= 4
    kinds = " ".join(n["statement"] for n in d["normalized_statements"])
    assert "SET" in kinds and "set_config" in kinds and "OWNER TO" in kinds and "\\restrict" in kinds
    assert d["how_to_fix"] and "--schema-only" in d["how_to_fix"]
    assert "apply_statements" not in d            # trusted-side only


@pytest.mark.parametrize("sql,needle", [
    ("CREATE TABLE a (id int;", "PostgreSQL parser"),
    ("CREATE TABLE a (id int);\nINSERT INTO a VALUES (1);", "INSERT"),
    ("CREATE TABLE a (id int);\nCOPY a FROM '/etc/passwd';", "COPY"),
    ("CREATE TABLE a (id int);\nCOPY a TO PROGRAM 'curl x';", "COPY"),
    ("CREATE TABLE a (id int DEFAULT length(pg_read_file('/etc/passwd')));", "pg_read_file"),
    ("CREATE TABLE a (id int);\nCREATE FUNCTION f() RETURNS int LANGUAGE sql AS $$ select 1 $$;", "FUNCTION"),
    ("CREATE TABLE a (id int);\nCREATE TRIGGER t BEFORE INSERT ON a FOR EACH ROW EXECUTE FUNCTION f();", "TRIGGER"),
    ("CREATE TABLE a (id int);\nGRANT SELECT ON a TO public;", "GRANT"),
    ("CREATE TABLE a (id int);\nREVOKE SELECT ON a FROM public;", "GRANT"),
    ("CREATE TABLE a (id int);\nDROP TABLE a;", "DROP"),
    ("CREATE TABLE a (id int);\nDO $$ BEGIN PERFORM 1; END $$;", "DO"),
    ("CREATE TABLE a (id int);\nALTER SYSTEM SET work_mem = '1GB';", "ALTER SYSTEM"),
    ("CREATE TABLE a (id int);\nCREATE ROLE x;", "role"),
    ("CREATE TABLE a (id int);\nTRUNCATE a;", "TRUNCATE"),
    ("CREATE TABLE a (id int);\n\\copy a from 'x'", "meta-command"),
    ("CREATE TABLE `a` (id int) ENGINE=InnoDB;", "MySQL"),
    ("CREATE TABLE a (id int IDENTITY(1,1));", "SQL Server"),
    ("CREATE TABLE a (id int REFERENCES b(id));\nCREATE TABLE b (id int primary key);", "before it is created"),
    ("CREATE TABLE a (id int);\nCREATE TABLE a (id int);", "duplicate"),
    ("CREATE TABLE other.a (id int);", "schema 'other'"),
    ("CREATE TABLE a (id int);\nCREATE VIEW v AS SELECT pg_sleep(1);", "view body"),
    ("CREATE TABLE a (id int);\nCREATE SERVER s FOREIGN DATA WRAPPER postgres_fdw;", "foreign"),
])
def test_forbidden_constructs_rejected_with_report(sql, needle):
    r = verify_schema(sql)
    assert not r.verified and r.apply_statements == []
    text = str(r.to_dict()["rejected_statements"])
    assert needle.lower() in text.lower(), text
    assert all("line" in x for x in r.rejected_statements)


@pytest.mark.parametrize("sql,feature", [
    ("CREATE EXTENSION postgis;\nCREATE TABLE a (id int);", "extension postgis"),
    ("CREATE TABLE a (g geometry);", "type geometry"),
])
def test_unsupported_features_reported(sql, feature):
    r = verify_schema(sql)
    assert not r.verified and r.unsupported_features[0]["feature"] == feature


def test_system_extension_normalized_not_installed():
    r = verify_schema("CREATE EXTENSION IF NOT EXISTS hypopg;\nCREATE TABLE a (id int);")
    assert r.verified and "CREATE EXTENSION" in r.normalized_statements[0]["statement"]
    assert all("EXTENSION" not in s.upper() for s in r.apply_statements)


async def test_failed_verification_never_provisions():
    m = SandboxManager(Database())            # registry pool not even opened
    with pytest.raises(QuerySentinelError) as ei:
        await m.create("sess-aaaaaaaa", "CREATE TABLE a (id int);\nINSERT INTO a VALUES (1);")
    assert ei.value.category == ErrorCategory.UPLOAD_REJECTED
    assert ei.value.details["verification"]["verified"] is False
    assert m.provision_calls == 0


# ----------------------------------------------------------------------------- live sandbox-pg
async def _sbx_dbs() -> set[str]:
    s = get_settings()
    c = await asyncpg.connect(**s.sandbox_admin_dsn_kwargs)
    try:
        return {r[0] for r in await c.fetch("SELECT datname FROM pg_database WHERE datname LIKE 'sbx\\_%'")}
    finally:
        await c.close()


@pytest.fixture
async def mgr():
    if not admin_db_available():
        pytest.skip("main DB not reachable (NOT RUN)")
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    db = Database()
    assert await db.connect()
    m = SandboxManager(db)
    yield m
    await db.close()


@requires_sandbox_pg
async def test_rejected_upload_creates_no_objects(mgr):
    before = await _sbx_dbs()
    with pytest.raises(QuerySentinelError):
        await mgr.create("sess-" + uuid.uuid4().hex[:10], "CREATE TABLE a (id int);\nCOPY a FROM '/x';")
    assert await _sbx_dbs() == before and mgr.provision_calls == 0


@requires_sandbox_pg
async def test_provision_hypopg_isolation_synthetic_and_cleanup(mgr):
    s = get_settings()
    sess_a, sess_b = "sess-" + uuid.uuid4().hex[:10], "sess-" + uuid.uuid4().hex[:10]
    a = await mgr.create(sess_a, SHOP, name="shop A", scale="small")
    b = await mgr.create(sess_b, "CREATE TABLE t (id int primary key, v text);", name="B", scale="none")
    try:
        assert a["status"] == "ready" and a["synthetic"] and "SYNTHETIC DATA" in a["provenance"]
        syn = a["synthetic_data"]["tables"]
        assert syn["public.customers"] == s.synthetic_rows_small and syn["public.orders"] == s.synthetic_rows_small
        assert 0 < syn["public.order_lines"] <= s.synthetic_rows_small      # composite PK duplicates skipped
        sid = a["data_source_id"]
        db = mgr.db_factory({"id": sid, "db_name": sid, "role_name": mgr.role_name(sid)})
        assert await db.connect()
        try:
            async with db.connection() as conn:
                # FK honoured: every order references an existing customer
                assert await conn.fetchval("SELECT count(*) FROM orders o LEFT JOIN customers c ON "
                                           "c.id = o.customer_id WHERE c.id IS NULL") == 0
                # HypoPG usable by the sandbox role (create + reset a hypothetical index)
                await conn.fetch("SELECT * FROM hypopg_create_index('CREATE INDEX ON orders (ordered_at)')")
                assert await conn.fetchval("SELECT count(*) FROM hypopg_list_indexes") == 1
                await conn.execute("SELECT hypopg_reset()")
                assert await conn.fetchval("SELECT rolsuper OR rolcreatedb OR rolcreaterole FROM pg_roles "
                                           "WHERE rolname = current_user") is False
        finally:
            await db.close()
        # request isolation: role A cannot reach B's database nor the main database
        pw_a = mgr.role_password(sid)
        with pytest.raises(asyncpg.PostgresError):
            c = await asyncpg.connect(host=s.sandbox_pg_host, port=s.sandbox_pg_port, database=b["data_source_id"],
                                      user=mgr.role_name(sid), password=pw_a)
            await c.close()
        with pytest.raises((asyncpg.PostgresError, OSError)):
            c = await asyncpg.connect(host=s.postgres_host, port=s.postgres_port, database=s.postgres_db,
                                      user=mgr.role_name(sid), password=pw_a)
            await c.close()
        # ownership
        with pytest.raises(QuerySentinelError) as ei:
            await mgr.get(sid, sess_b)
        assert ei.value.category == ErrorCategory.SOURCE_FORBIDDEN
        ext = await mgr.extend(sid, sess_a)
        assert ext["extensions_used"] == 1
    finally:
        await mgr.delete(a["data_source_id"], sess_a)
        await mgr.delete(b["data_source_id"], sess_b)
    dbs = await _sbx_dbs()
    assert a["data_source_id"] not in dbs and b["data_source_id"] not in dbs
    # idempotent cleanup
    await mgr._drop(a["data_source_id"])


@requires_sandbox_pg
async def test_synthetic_scales_row_counts(mgr):
    sess = "sess-" + uuid.uuid4().hex[:10]
    ddl = "CREATE TABLE p (id int primary key);\nCREATE TABLE c (id bigint primary key, p_id int NOT NULL " \
          "REFERENCES p(id), amount numeric(8,2) CHECK (amount > 0));"
    got = {}
    for scale in ("small", "medium"):
        r = await mgr.create(sess, ddl, scale=scale)
        got[scale] = r["synthetic_data"]["tables"]
        await mgr.delete(r["data_source_id"], sess)
    s = get_settings()
    assert got["small"] == {"public.p": s.synthetic_rows_small, "public.c": s.synthetic_rows_small}
    assert got["medium"] == {"public.p": s.synthetic_rows_medium, "public.c": s.synthetic_rows_medium}


@requires_sandbox_pg
async def test_ttl_reaper_and_startup_sweep(mgr):
    s = get_settings()
    sess = "sess-" + uuid.uuid4().hex[:10]
    r = await mgr.create(sess, "CREATE TABLE t (id int);", scale="none", ttl_minutes=2 / 60)   # 2 seconds
    sid = r["data_source_id"]
    await asyncio.sleep(2.5)
    reaped = await mgr.reap_expired()
    assert sid in reaped and sid not in await _sbx_dbs()
    st = await mgr.main_db.fetchval("SELECT status FROM qs_meta.data_sources WHERE id = $1", sid)
    assert st == "expired"
    assert sid not in await mgr.reap_expired()            # idempotent
    # simulated orphan: database + role with no registry row
    orphan = "sbx_" + uuid.uuid4().hex[:12]
    c = await asyncpg.connect(**s.sandbox_admin_dsn_kwargs)
    try:
        await c.execute(f'CREATE ROLE "{orphan}_r" LOGIN')
        await c.execute(f'CREATE DATABASE "{orphan}"')
    finally:
        await c.close()
    out = await mgr.startup_sweep()
    assert orphan in out["dropped_orphans"] and orphan not in await _sbx_dbs()
    # registry row whose database is gone gets fixed
    ghost = "sbx_" + uuid.uuid4().hex[:12]
    await mgr.main_db.execute_write("INSERT INTO qs_meta.data_sources (id, kind, name, status, db_name, role_name, "
                                    "expires_at) VALUES ($1,'sandbox','g','ready',$1,$2, now() + interval '1 hour')",
                                    ghost, ghost + "_r")
    out = await mgr.startup_sweep()
    assert ghost in out["fixed_registry"]
