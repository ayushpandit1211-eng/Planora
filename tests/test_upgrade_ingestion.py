"""P2 dynamic ingestion + additive schema evolution (real PostgreSQL for DB-touching tests)."""
from __future__ import annotations

import json
import uuid

import pytest

from app.dataset_ingestion import (DatasetIngestionService, analyze_file, dedupe, normalize_identifier)
from app.db import Database
from app.errors import ErrorCategory, QuerySentinelError
from app.postgres_instrumentation import PostgresInstrumentation
from app.schema_inspector import SchemaInspector
from tests.upgrade_helpers import admin_exec, requires_admin_db


def _file(tmp_path, name, content, enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(content.encode(enc) if isinstance(content, str) else content)
    return p


# ----------------------------------------------------------------------------- pure (no DB)
def test_identifier_normalisation():
    assert normalize_identifier("Order Date") == "order_date"
    assert normalize_identifier("CustomerID") == "customer_id"
    assert normalize_identifier("1st value") == "c_1st_value"
    assert normalize_identifier("select") == "select_col"
    assert normalize_identifier("_qs_ingest_id") == "u_qs_ingest_id"
    assert len(normalize_identifier("x" * 100).encode()) <= 63
    assert dedupe(["a", "a", "a"]) == ["a", "a_2", "a_3"]


def test_csv_type_inference_and_edge_cases(tmp_path):
    csv = ("id;zip;amount;price;is_active;flag01;created;seen_at;ref;note\n"
           "1;00123;1,200.50;$5;1;0;2026-01-05;2026-01-05T10:00:00Z;" + str(uuid.uuid4()) + ";hello\n"
           "2;04567;3,000.00;$7;0;1;2026-02-11;2026-02-11T11:30:00+05:30;" + str(uuid.uuid4()) + ";NA\n"
           "3;00999;10.00;$9;1;1;2026-03-01;2026-03-01T00:00:00Z;" + str(uuid.uuid4()) + ";\n")
    p = _file(tmp_path, "edge.csv", "﻿" + csv)
    a = analyze_file(p, "edge.csv")
    assert a.delimiter == ";" and a.encoding == "utf-8-sig"
    t = a.column_types()
    assert t["id"] == "integer"
    assert t["zip"] == "text"                              # leading zeros stay text
    assert t["amount"] == "text"                           # mixed "1,200.50" and "10.00" -> inconsistent -> text
    assert t["price"] == "numeric"                         # consistent currency -> cleaned with warning
    assert t["is_active"] == "boolean"                     # 1/0 + boolean-looking name
    assert t["flag01"] == "boolean"
    assert t["created"] == "date"
    assert t["seen_at"] == "timestamptz"
    assert t["ref"] == "uuid"
    assert t["note"] == "text"
    assert any("currency" in w for w in a.warnings)


def test_latin1_ragged_and_no_header(tmp_path):
    p = _file(tmp_path, "l1.csv", "name,city\nJosé,Zürich\nAna,Köln,extra\nBo\n", enc="latin-1")
    a = analyze_file(p, "l1.csv")
    assert a.encoding == "latin-1" and a.rows_read == 3
    assert a.ragged_rows == [3]                           # extra value -> rejected; short row -> NULL padded
    p2 = _file(tmp_path, "nh.csv", "1,2\n3,4\n")
    a2 = analyze_file(p2, "nh.csv")
    assert [c.name for c in a2.columns] == ["column_1", "column_2"]


def test_json_variants(tmp_path):
    arr = [{"id": 1, "user": {"name": "a", "tags": ["x"]}, "meta": {"deep": {"k": 1}}}, {"id": 2, "extra": True}]
    for name, content in [("a.json", json.dumps(arr)),
                          ("b.jsonl", "\n".join(json.dumps(r) for r in arr)),
                          ("c.json", json.dumps({"records": arr})),
                          ("d.json", "\n".join(json.dumps(r) for r in arr))]:
        a = analyze_file(_file(tmp_path, name, content), name)
        t = a.column_types()
        assert t["id"] == "integer" and t["user__name"] == "text", name
        assert t["user__tags"] == "jsonb" and t["meta__deep"] == "jsonb", name
        assert "extra" in t and a.columns[[c.name for c in a.columns].index("extra")].nulls == 1, name


def test_sql_and_unknown_extensions_rejected(tmp_path):
    for name in ("x.sql", "x.xlsx"):
        with pytest.raises(QuerySentinelError) as ei:
            analyze_file(_file(tmp_path, name, "select 1"), name)
        assert ei.value.category == ErrorCategory.UPLOAD_REJECTED


# ----------------------------------------------------------------------------- live
@pytest.fixture
def svc():
    s = DatasetIngestionService()
    yield s


def _cleanup(ds: str):
    rows = admin_exec("SELECT data_source_id FROM qs_meta.datasets WHERE name = %s", ds) or []
    for (sid,) in rows:
        admin_exec("DELETE FROM qs_meta.dataset_columns WHERE table_id IN (SELECT t.id FROM qs_meta.dataset_tables t "
                   "JOIN qs_meta.datasets d ON d.id = t.dataset_id WHERE d.data_source_id = %s)", sid)
        admin_exec("DELETE FROM qs_meta.dataset_tables WHERE dataset_id IN (SELECT id FROM qs_meta.datasets "
                   "WHERE data_source_id = %s)", sid)
        admin_exec("DELETE FROM qs_meta.datasets WHERE data_source_id = %s", sid)
        admin_exec("DELETE FROM qs_meta.schema_versions WHERE data_source_id = %s", sid)
        admin_exec("DELETE FROM qs_meta.ingestion_runs WHERE data_source_id = %s", sid)
        admin_exec("DELETE FROM qs_meta.data_sources WHERE id = %s", sid)
    for (t,) in admin_exec("SELECT tablename FROM pg_tables WHERE schemaname='qs_dynamic' AND tablename LIKE %s",
                           ds + "\\_\\_%") or []:
        admin_exec(f'DROP TABLE qs_dynamic."{t}"')     # test cleanup only (no product code path drops tables)


@requires_admin_db
async def test_create_append_evolve_conflict_cycle(tmp_path, svc):
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    ds = "p2test" + uuid.uuid4().hex[:6]
    _cleanup(ds)
    sess = "sess-" + uuid.uuid4().hex[:12]
    try:
        c1 = "region,amount,sold_on\nnorth,10.5,2026-01-01\nsouth,20,2026-01-02\n"
        f1 = _file(tmp_path, "sales.csv", c1)
        pv = await svc.preview(f1, "sales.csv", dataset_name=ds, session_id=sess)
        assert pv["plan"]["action"] == "create_table"
        assert {c["name"]: c["inferred_type"] for c in pv["columns"]} == {
            "region": "text", "amount": "numeric", "sold_on": "date"}
        r1 = await svc.commit(pv["preview_id"], session_id=sess)
        assert r1["rows_loaded"] == 2 and r1["schema_version"] == 1
        sid = r1["data_source_id"]
        tbl = r1["table"].split(".")[1]
        assert admin_exec(f'SELECT count(*) FROM qs_dynamic."{tbl}"')[0][0] == 2

        # duplicate file needs explicit flag
        pv_dup = await svc.preview(_file(tmp_path, "sales.csv", c1), "sales.csv", dataset_name=ds,
                                   session_id=sess)
        with pytest.raises(QuerySentinelError) as ei:
            await svc.commit(pv_dup["preview_id"], session_id=sess)
        assert ei.value.category == ErrorCategory.UPLOAD_REJECTED

        # compatible re-upload appends and records a run (integers into numeric column)
        f2 = _file(tmp_path, "sales.csv", "region,amount,sold_on\neast,7,2026-01-03\n")
        r2 = await svc.commit((await svc.preview(f2, "sales.csv", dataset_name=ds, session_id=sess))["preview_id"],
                              session_id=sess)
        assert r2["data_source_id"] == sid and r2["schema_version"] == 1 and r2["schema_changes"] == []
        assert admin_exec(f'SELECT count(*) FROM qs_dynamic."{tbl}"')[0][0] == 3
        assert admin_exec("SELECT count(*) FROM qs_meta.ingestion_runs WHERE data_source_id=%s", sid)[0][0] == 2

        # new column -> additive ALTER + schema_version bump; missing column -> NULL
        f3 = _file(tmp_path, "sales.csv", "region,sold_on,channel\nwest,2026-01-04,web\n")
        pv3 = await svc.preview(f3, "sales.csv", dataset_name=ds, session_id=sess)
        assert pv3["plan"]["new_columns"] == ["channel"] and pv3["plan"]["missing_columns_filled_null"] == ["amount"]
        r3 = await svc.commit(pv3["preview_id"], session_id=sess)
        assert r3["schema_version"] == 2 and r3["schema_changes"][0]["change"] == "add_column"
        assert admin_exec(f'SELECT amount FROM qs_dynamic."{tbl}" WHERE region = %s', "west")[0][0] is None

        # incompatible type -> whole ingestion rejected, precise report, zero partial writes
        f4 = _file(tmp_path, "sales.csv", "region,amount,sold_on,newcol\nx,abc,2026-01-05,1\ny,12,not-a-date,2\n")
        pv4 = await svc.preview(f4, "sales.csv", dataset_name=ds, session_id=sess)
        assert pv4["plan"]["conflicts"]
        with pytest.raises(QuerySentinelError) as ei:
            await svc.commit(pv4["preview_id"], session_id=sess)
        assert ei.value.category == ErrorCategory.SCHEMA_CONFLICT
        cols = {c["column"]: c for c in ei.value.details["conflicts"]}
        assert cols["amount"]["sample_values"] == ["abc"] and cols["amount"]["row_numbers"] == [2]
        assert admin_exec(f'SELECT count(*) FROM qs_dynamic."{tbl}"')[0][0] == 4
        assert admin_exec("SELECT count(*) FROM information_schema.columns WHERE table_schema='qs_dynamic' "
                          "AND table_name=%s AND column_name='newcol'", tbl)[0][0] == 0

        # other session cannot commit someone else's preview
        pv5 = await svc.preview(_file(tmp_path, "sales.csv", "region\nz\n"), "sales.csv", dataset_name=ds,
                                session_id=sess)
        with pytest.raises(QuerySentinelError) as ei:
            await svc.commit(pv5["preview_id"], session_id="someone-else-123")
        assert ei.value.category == ErrorCategory.SOURCE_FORBIDDEN

        # SchemaInspector shows the table and columns (hidden lineage column excluded)
        db = Database(server_settings={"search_path": "qs_dynamic"})
        assert await db.connect()
        try:
            meta = await SchemaInspector(PostgresInstrumentation(db, "qs_dynamic")).inspect([tbl])
            assert tbl in meta.tables
            names = [c.name for c in meta.tables[tbl].columns]
            assert names == ["region", "amount", "sold_on", "channel"]
            rows = await db.fetch(f'SELECT count(*) AS n FROM "{tbl}"')
            assert rows[0]["n"] == 4
        finally:
            await db.close()
    finally:
        _cleanup(ds)


@requires_admin_db
async def test_same_dataset_name_two_sessions_never_share_a_table(tmp_path, svc):
    from app.meta_migration import run_meta_migration
    await run_meta_migration()
    ds = "p2iso" + uuid.uuid4().hex[:6]
    a, b = "sess-" + uuid.uuid4().hex[:10], "sess-" + uuid.uuid4().hex[:10]
    try:
        ra = await svc.commit((await svc.preview(_file(tmp_path, "t.csv", "k,v\nx,1\n"), "t.csv", ds, None, a))
                              ["preview_id"], session_id=a)
        rb = await svc.commit((await svc.preview(_file(tmp_path, "t.csv", "k,v\ny,2\n"), "t.csv", ds, None, b))
                              ["preview_id"], session_id=b)
        assert ra["table"] != rb["table"] and ra["data_source_id"] != rb["data_source_id"]
        ta, tb = ra["table"].split(".")[1], rb["table"].split(".")[1]
        assert admin_exec(f'SELECT k FROM qs_dynamic."{ta}"') == [("x",)]
        assert admin_exec(f'SELECT k FROM qs_dynamic."{tb}"') == [("y",)]
    finally:
        _cleanup(ds)
        for (t,) in admin_exec("SELECT tablename FROM pg_tables WHERE schemaname='qs_dynamic' AND tablename LIKE %s",
                               ds + "%") or []:
            admin_exec(f'DROP TABLE qs_dynamic."{t}"')
