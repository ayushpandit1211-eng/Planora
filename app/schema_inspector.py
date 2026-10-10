"""Schema inspector (Section 17).

Collects ONLY safe structural metadata: names, types, nullability, keys, indexes, estimated
row counts, sizes and selected planner statistics. Row contents are never read. Two views:

* `SchemaMetadata` with real names - trusted backend only (validator, sandbox, migrations)
* `anonymize()` output - pseudonymised names, safe for agents/LLM payloads
"""
from __future__ import annotations

from typing import Any

from app.contracts import ColumnMeta, ForeignKeyMeta, IndexMeta, SchemaMetadata, TableMeta
from app.masking_vault import DataObfuscator
from app.postgres_instrumentation import PostgresInstrumentation


def table_size_bucket(rows: float | None) -> str:
    rows = rows or 0
    if rows < 10_000:
        return "small(<10k)"
    if rows < 1_000_000:
        return "medium(10k-1M)"
    if rows < 10_000_000:
        return "large(1M-10M)"
    return "xlarge(>10M)"


def write_frequency_bucket(writes: float | None) -> str:
    w = writes or 0
    if w < 1_000:
        return "low"
    if w < 100_000:
        return "medium"
    return "high"


def estimated_distinct(col: ColumnMeta | None, rows: float) -> float | None:
    """pg_stats n_distinct: positive = absolute count, negative = fraction of rows."""
    if col is None or col.n_distinct is None:
        return None
    nd = col.n_distinct
    return abs(nd) * rows if nd < 0 else nd


class SchemaInspector:
    def __init__(self, instrumentation: PostgresInstrumentation):
        self.instr = instrumentation

    async def inspect(self, tables: list[str] | None = None) -> SchemaMetadata:
        names = sorted(set(tables)) if tables else None
        trows = await self.instr.tables(names)
        crows = await self.instr.columns(names)
        irows = await self.instr.indexes(names)
        frows = await self.instr.foreign_keys(names)
        pks = await self.instr.primary_keys()
        return build_schema_metadata(trows, crows, irows, frows, pks)


class SchemaCache:
    """Final Upgrade P1/P2: per-source cache keyed by (data_source_id, schema_version) with invalidation.
    The original `SchemaInspector.inspect()` is unchanged and uncached."""

    def __init__(self) -> None:
        self._d: dict[tuple[str, int], SchemaMetadata] = {}

    async def get(self, source_id: str, version: int, instr: PostgresInstrumentation,
                  tables: list[str] | None = None) -> SchemaMetadata:
        key = (source_id, version)
        if key not in self._d:
            meta = await SchemaInspector(instr).inspect(tables)
            self._d[key] = meta
        meta = self._d[key]
        if tables:
            keep = set(tables)
            return meta.model_copy(update={"tables": {k: v for k, v in meta.tables.items() if k in keep}}) \
                if hasattr(meta, "tables") and isinstance(meta.tables, dict) else meta
        return meta

    def invalidate(self, source_id: str) -> None:
        for k in [k for k in self._d if k[0] == source_id]:
            self._d.pop(k, None)


schema_cache = SchemaCache()


def build_schema_metadata(trows: list[dict], crows: list[dict], irows: list[dict],
                          frows: list[dict], pks: dict[str, list[str]]) -> SchemaMetadata:
    meta = SchemaMetadata()
    cols_by_table: dict[str, list[ColumnMeta]] = {}
    for c in crows:
        cols_by_table.setdefault(c["table_name"], []).append(ColumnMeta(
            name=c["column_name"], data_type=c["data_type"], nullable=bool(c["nullable"]),
            n_distinct=c.get("n_distinct"), null_frac=c.get("null_frac"), correlation=c.get("correlation"),
            top_value_freq=c.get("top_value_freq")))
    for t in trows:
        name = t["table_name"]
        meta.tables[name] = TableMeta(
            schema_name=t.get("schema_name", "public"), name=name, columns=cols_by_table.get(name, []),
            primary_key=pks.get(name, []), estimated_rows=max(float(t.get("estimated_rows") or 0), 0.0),
            table_size_bytes=t.get("table_size_bytes"), total_size_bytes=t.get("total_size_bytes"),
            n_tup_ins=t.get("n_tup_ins"), n_tup_upd=t.get("n_tup_upd"), n_tup_del=t.get("n_tup_del"),
            n_tup_hot_upd=t.get("n_tup_hot_upd"), seq_scan=t.get("seq_scan"), idx_scan=t.get("idx_scan"),
            last_analyze=t.get("last_analyze"))
    for i in irows:
        if i.get("has_expression"):
            kind = "expression"
        else:
            kind = i.get("am") or "btree"
        meta.indexes.append(IndexMeta(
            name=i["index_name"], table=i["table_name"], columns=list(i.get("key_columns") or []),
            is_unique=bool(i.get("is_unique")), is_primary=bool(i.get("is_primary")),
            is_partial=bool(i.get("is_partial")), include_columns=list(i.get("include_columns") or []),
            size_bytes=i.get("size_bytes"), idx_scan=i.get("idx_scan"), definition_kind=kind))
    for f in frows:
        cols = list(f["columns"])
        supported = any(ix.table == f["table_name"] and ix.columns[:len(cols)] == cols for ix in meta.indexes)
        meta.foreign_keys.append(ForeignKeyMeta(table=f["table_name"], columns=cols, ref_table=f["ref_table"],
                                                ref_columns=list(f["ref_columns"]), has_supporting_index=supported))
    return meta


def pseudonymize_metadata(meta: SchemaMetadata, obfuscator: DataObfuscator, request_id: str) -> SchemaMetadata:
    """Same structure as `meta` with every table/column/index name replaced by its pseudonym.
    This is the schema view the agents work with."""
    names: dict[str, str] = {}
    for t in meta.tables.values():
        names[t.name] = "table"
        for c in t.columns:
            names.setdefault(c.name, "column")
    for ix in meta.indexes:
        names.setdefault(ix.name, "index")
    m = obfuscator.pseudonymize_many(request_id, names)
    out = SchemaMetadata()
    for t in meta.tables.values():
        out.tables[m[t.name]] = t.model_copy(update={
            "name": m[t.name], "schema_name": "public",
            "columns": [c.model_copy(update={"name": m[c.name]}) for c in t.columns],
            "primary_key": [m.get(c, c) for c in t.primary_key]})
    out.indexes = [ix.model_copy(update={"name": m[ix.name], "table": m.get(ix.table, ix.table),
                                         "columns": [m.get(c, c) for c in ix.columns],
                                         "include_columns": [m.get(c, c) for c in ix.include_columns]})
                   for ix in meta.indexes if ix.table in meta.tables]
    out.foreign_keys = [fk.model_copy(update={"table": m.get(fk.table, fk.table), "ref_table": m.get(fk.ref_table, fk.ref_table),
                                              "columns": [m.get(c, c) for c in fk.columns],
                                              "ref_columns": [m.get(c, c) for c in fk.ref_columns]})
                        for fk in meta.foreign_keys if fk.table in meta.tables]
    return out


def anonymize(meta: SchemaMetadata, obfuscator: DataObfuscator, request_id: str) -> dict[str, Any]:
    """Pseudonymised, LLM-safe view of the schema metadata."""
    names: dict[str, str] = {}
    for t in meta.tables.values():
        names[t.name] = "table"
        for c in t.columns:
            names.setdefault(c.name, "column")
    for ix in meta.indexes:
        names.setdefault(ix.name, "index")
    m = obfuscator.pseudonymize_many(request_id, names)
    out_tables = {}
    for t in meta.tables.values():
        out_tables[m[t.name]] = {
            "columns": [{"name": m[c.name], "type": c.data_type, "nullable": c.nullable,
                         "n_distinct": c.n_distinct, "null_frac": c.null_frac,
                         "correlation": None if c.correlation is None else round(c.correlation, 3),
                         "top_value_freq": None if c.top_value_freq is None else round(c.top_value_freq, 4)}
                        for c in t.columns],
            "primary_key": [m[c] for c in t.primary_key if c in m],
            "estimated_rows": t.estimated_rows,
            "table_size_bucket": table_size_bucket(t.estimated_rows),
            "table_size_mb": None if t.table_size_bytes is None else round(t.table_size_bytes / 1048576, 1),
            "write_frequency_bucket": write_frequency_bucket((t.n_tup_ins or 0) + (t.n_tup_upd or 0) + (t.n_tup_del or 0)),
        }
    return {
        "tables": out_tables,
        "indexes": [{"name": m[i.name], "table": m.get(i.table, i.table), "columns": [m.get(c, c) for c in i.columns],
                     "include": [m.get(c, c) for c in i.include_columns], "unique": i.is_unique,
                     "primary": i.is_primary, "partial": i.is_partial, "kind": i.definition_kind,
                     "size_mb": None if i.size_bytes is None else round(i.size_bytes / 1048576, 2),
                     "idx_scan": i.idx_scan} for i in meta.indexes if i.table in meta.tables],
        "foreign_keys": [{"table": m.get(f.table, f.table), "columns": [m.get(c, c) for c in f.columns],
                          "ref_table": m.get(f.ref_table, f.ref_table),
                          "ref_columns": [m.get(c, c) for c in f.ref_columns],
                          "has_supporting_index": f.has_supporting_index} for f in meta.foreign_keys
                         if f.table in meta.tables],
    }
