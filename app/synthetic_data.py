"""Synthetic data generation inside a BYOD sandbox (Final Upgrade P3).

Reads the sandbox catalog (columns, types, nullability, defaults, identity/generated columns, unique/primary keys,
foreign keys), orders tables by foreign-key dependencies, and fills each table with generated rows using
server-side `generate_series` (no row data travels through the backend). Best effort: unique violations are
skipped with ON CONFLICT DO NOTHING, and any constraint that cannot be satisfied is reported, never hidden.
All SQL is composed by code from catalog identifiers (quoted); nothing from the upload is executed here.
Results computed on this data are [SYNTHETIC DATA] and illustrative only.
"""
from __future__ import annotations

from typing import Any

import asyncpg

from app.provenance import Provenance

SQL_COLS = """
SELECT c.relname AS tbl, n.nspname AS sch, a.attname AS col, a.attnum,
       format_type(a.atttypid, a.atttypmod) AS fmt, t.typname, t.typtype::text AS typtype, a.attnotnull AS notnull,
       a.atthasdef AS hasdef, a.attidentity <> '' AS is_identity, a.attgenerated <> '' AS is_generated,
       pg_get_expr(d.adbin, d.adrelid) AS defexpr, a.atttypmod
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
WHERE c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
  AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_%'
ORDER BY n.nspname, c.relname, a.attnum
"""
SQL_CONS = """
SELECT c.relname AS tbl, n.nspname AS sch, con.contype::text AS contype, con.conname,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) k JOIN pg_attribute a ON a.attrelid = con.conrelid
             AND a.attnum = k) AS cols,
       rc.relname AS ref_tbl, rn.nspname AS ref_sch,
       ARRAY(SELECT a.attname FROM unnest(con.confkey) k JOIN pg_attribute a ON a.attrelid = con.confrelid
             AND a.attnum = k) AS ref_cols
FROM pg_constraint con JOIN pg_class c ON c.oid = con.conrelid JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_class rc ON rc.oid = con.confrelid LEFT JOIN pg_namespace rn ON rn.oid = rc.relnamespace
WHERE con.contype IN ('p', 'u', 'f', 'c') AND n.nspname NOT IN ('pg_catalog', 'information_schema')
"""
SQL_UNIQUE_IDX = """
SELECT c.relname AS tbl, n.nspname AS sch,
       ARRAY(SELECT a.attname FROM unnest(ix.indkey) k JOIN pg_attribute a ON a.attrelid = ix.indrelid
             AND a.attnum = k) AS cols
FROM pg_index ix JOIN pg_class c ON c.oid = ix.indrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE ix.indisunique AND n.nspname NOT IN ('pg_catalog', 'information_schema')
"""


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _expr(col: dict, unique: bool, fk: tuple[str, str] | None) -> str | None:
    """SQL expression for one column given row number `g`; None = leave to DEFAULT."""
    name, tn, fmt = col["col"], col["typname"], col["fmt"]
    if fk is not None:
        arr = fk[0]
        return f"{arr}[1 + floor(random() * greatest(array_length({arr}, 1), 1))::int]"
    if col["typtype"] == "e":
        t = fmt
        return f"(enum_range(NULL::{t}))[1 + floor(random() * array_length(enum_range(NULL::{t}), 1))::int]"
    if tn in ("int2", "int4", "int8", "oid"):
        if unique:
            return "g" if tn != "int2" else "(g % 32000)::int2"
        return "(1 + floor(random() * 1000))::int" if tn != "int2" else "(1 + floor(random() * 100))::int2"
    if tn in ("numeric", "float4", "float8", "money"):
        if unique:
            return f"(g + 0.5)::{fmt}" if tn != "money" else "(g + 0.5)::numeric::money"
        return f"round((1 + random() * 999)::numeric, 2)::{fmt}" if tn != "money" else \
            "round((1 + random() * 999)::numeric, 2)::money"
    if tn == "bool":
        return "random() < 0.5"
    if tn == "date":
        return "date '2024-01-01' + (random() * 730)::int"
    if tn in ("timestamp", "timestamptz"):
        return f"(timestamp '2024-01-01' + random() * interval '730 days')::{fmt}"
    if tn == "time":
        return "time '00:00' + random() * interval '24 hours'"
    if tn == "interval":
        return "random() * interval '30 days'"
    if tn == "uuid":
        return "gen_random_uuid()"
    if tn in ("json", "jsonb"):
        return f"jsonb_build_object('n', g)::{tn}"
    if tn in ("text", "varchar", "bpchar", "name", "citext"):
        maxlen = col["atttypmod"] - 4 if col["atttypmod"] and col["atttypmod"] > 4 else 0
        lname = name.lower()
        if "email" in lname:
            base = "'user' || g || '@example.test'"
        elif unique:
            base = f"'{lname[:12]}_' || g"
        else:
            base = f"'{lname[:12]}_' || (1 + floor(random() * 50))::int"
        return f"left({base}, {maxlen})" if maxlen else base
    if tn == "bytea":
        return "decode(md5(g::text), 'hex')"
    if tn == "inet":
        return "('10.0.' || (g % 250) || '.' || (g % 250))::inet"
    return None


async def generate(conn: asyncpg.Connection, rows: int) -> dict[str, Any]:
    cols = [dict(r) for r in await conn.fetch(SQL_COLS)]
    cons = [dict(r) for r in await conn.fetch(SQL_CONS)]
    uidx = [dict(r) for r in await conn.fetch(SQL_UNIQUE_IDX)]
    tables: dict[tuple[str, str], list[dict]] = {}
    for c in cols:
        tables.setdefault((c["sch"], c["tbl"]), []).append(c)
    unique_cols: dict[tuple[str, str], set[str]] = {}
    for u in [c for c in cons if c["contype"] in ("p", "u")] + uidx:
        if len(u["cols"]) == 1:
            unique_cols.setdefault((u["sch"], u["tbl"]), set()).add(u["cols"][0])
    fks = [c for c in cons if c["contype"] == "f"]
    deps = {t: {(f["ref_sch"], f["ref_tbl"]) for f in fks if (f["sch"], f["tbl"]) == t
                and (f["ref_sch"], f["ref_tbl"]) != t} for t in tables}
    order: list[tuple[str, str]] = []
    pending = dict(deps)
    while pending:
        ready = sorted(t for t, d in pending.items() if not (d - set(order)))
        if not ready:                      # cycle: break it deterministically and report
            ready = [sorted(pending)[0]]
        for t in ready:
            order.append(t)
            pending.pop(t)
    report: dict[str, Any] = {"rows_requested_per_table": rows, "tables": {}, "unsatisfied": [],
                              "provenance": [Provenance.SYNTHETIC_DATA.value]}
    for t in order:
        sch, tbl = t
        tfks = [f for f in fks if (f["sch"], f["tbl"]) == t]
        fk_map: dict[str, tuple[str, str]] = {}
        ctes = []
        skip_fk_cols: set[str] = set()
        for i, f in enumerate(tfks):
            if len(f["cols"]) != 1:
                report["unsatisfied"].append({"table": f"{sch}.{tbl}", "constraint": f["conname"],
                                              "reason": "composite foreign key: rows generated only where the "
                                                        "columns are nullable"})
                skip_fk_cols.update(f["cols"])
                continue
            if (f["ref_sch"], f["ref_tbl"]) == t:
                skip_fk_cols.add(f["cols"][0])
                report["unsatisfied"].append({"table": f"{sch}.{tbl}", "constraint": f["conname"],
                                              "reason": "self-reference left NULL"})
                continue
            alias = f"p{i}"
            ctes.append(f"{alias} AS (SELECT array_agg(v) AS a FROM (SELECT {qi(f['ref_cols'][0])} AS v FROM "
                        f"{qi(f['ref_sch'])}.{qi(f['ref_tbl'])} LIMIT 100000) s)")
            fk_map[f["cols"][0]] = (f"{alias}.a", alias)
        target, exprs = [], []
        skipped_required = []
        for c in tables[t]:
            if c["is_generated"] or c["is_identity"]:
                continue
            if c["col"] in skip_fk_cols:
                if c["notnull"]:
                    skipped_required.append(c["col"])
                continue
            if c["hasdef"] and (c["defexpr"] or "").startswith("nextval("):
                continue                           # serial: let the sequence fill it
            e = _expr(c, c["col"] in unique_cols.get(t, set()), fk_map.get(c["col"]))
            if e is None:
                if c["notnull"] and not c["hasdef"]:
                    skipped_required.append(c["col"])
                continue
            target.append(qi(c["col"]))
            exprs.append(e)
        if skipped_required:
            report["unsatisfied"].append({"table": f"{sch}.{tbl}", "constraint": "NOT NULL",
                                          "reason": f"no generator for required column(s) {skipped_required}"})
            report["tables"][f"{sch}.{tbl}"] = 0
            continue
        froms = ", ".join(["generate_series(1, $1) AS g"] + [a for _, a in fk_map.values()])
        sql = (("WITH " + ", ".join(ctes) + " ") if ctes else "") + \
            (f"INSERT INTO {qi(sch)}.{qi(tbl)} ({', '.join(target)}) SELECT {', '.join(exprs)} FROM {froms} "
             if target else f"INSERT INTO {qi(sch)}.{qi(tbl)} SELECT FROM generate_series(1, $1) g ") + \
            "ON CONFLICT DO NOTHING"
        try:
            async with conn.transaction():
                status = await conn.execute(sql, rows)
            n = int(status.split()[-1])
        except asyncpg.PostgresError as exc:
            report["unsatisfied"].append({"table": f"{sch}.{tbl}", "constraint": getattr(exc, "constraint_name", None)
                                          or exc.sqlstate, "reason": f"generation failed ({exc.sqlstate}); "
                                                                     "table left empty"})
            n = 0
        report["tables"][f"{sch}.{tbl}"] = n
        if 0 < n < rows:
            report["unsatisfied"].append({"table": f"{sch}.{tbl}", "constraint": "UNIQUE/PRIMARY KEY",
                                          "reason": f"{rows - n} generated rows skipped as duplicates"})
    for sch, tbl in order:
        await conn.execute(f"ANALYZE {qi(sch)}.{qi(tbl)}")
    return report
