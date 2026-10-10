"""Recommendation translation (Section 29) and validation (Section 30).

LLM/agent output   : CREATE INDEX ON tbl_a31f7c (col_c81d44, col_91ab23)
private mapping    : tbl_a31f7c -> orders, col_c81d44 -> tenant_id, col_91ab23 -> order_date
sandbox DDL        : CREATE INDEX ON public.orders USING btree (tenant_id, order_date)

Validation before HypoPG: identifier existence, valid index syntax, allowed DDL type (plain
CREATE INDEX only), no destructive operations, duplicate/redundant detection, safe identifier
quoting (DDL is REBUILT from validated parts, never passed through), no injection (single
parsed statement, whitelisted predicate node types), schema compatibility. The reverse
mapping never leaves the trusted backend.
"""
from __future__ import annotations

import re

from pglast import ast, parse_sql
from pglast.stream import RawStream

from app.contracts import IndexCandidate, SchemaMetadata, ValidationOutcome
from app.errors import QuerySentinelError
from app.index_hygiene import is_redundant
from app.masking_vault import DataObfuscator

MAX_INDEX_COLUMNS = 5
ALLOWED_PREDICATE_NODES = (ast.A_Expr, ast.BoolExpr, ast.ColumnRef, ast.A_Const, ast.TypeCast, ast.NullTest,
                           ast.String, ast.Integer, ast.Float, ast.Boolean, ast.TypeName)


def quote_ident(name: str) -> str:
    if re.fullmatch(r"[a-z_][a-z0-9_]*", name) and name not in {"user", "order", "group", "select", "table"}:
        return name
    return '"' + name.replace('"', '""') + '"'


def _predicate_ok(node, columns: set[str]) -> list[str]:
    errors: list[str] = []

    def walk(n):
        if n is None:
            return
        if isinstance(n, tuple):
            for x in n:
                walk(x)
            return
        if not isinstance(n, ALLOWED_PREDICATE_NODES):
            errors.append(f"partial-index predicate contains unsupported node {type(n).__name__}")
            return
        if isinstance(n, ast.ColumnRef):
            parts = [getattr(f, "sval", None) for f in n.fields]
            if len(parts) != 1 or parts[0] not in columns:
                errors.append("partial-index predicate references an unknown column")
            return
        for slot in n.__slots__:
            child = getattr(n, slot, None)
            if isinstance(child, (ast.Node, tuple)):
                walk(child)
    walk(node)
    return errors


class RecommendationValidator:
    def __init__(self, obfuscator: DataObfuscator, real_schema: SchemaMetadata, schema_name: str = "public"):
        self.obf = obfuscator
        self.schema = real_schema
        self.schema_name = schema_name        # Final Upgrade P5: target schema of the active data source

    def validate(self, cand: IndexCandidate, request_id: str) -> ValidationOutcome:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            real = self.obf.translate_llm_ddl_to_real(cand.ddl, request_id)
        except QuerySentinelError as exc:
            return ValidationOutcome(candidate=cand, valid=False, errors=[f"{exc.category.value}: {exc.message}"])
        stmt = parse_sql(real)[0].stmt
        if not isinstance(stmt, ast.IndexStmt):
            return ValidationOutcome(candidate=cand, valid=False, errors=["only CREATE INDEX is allowed"])
        if stmt.unique:
            errors.append("UNIQUE indexes change constraints and are not allowed as recommendations")
        if stmt.concurrent:
            warnings.append("CONCURRENTLY stripped for the sandbox; added back in the migration script")
        am = (stmt.accessMethod or "btree").lower()
        if am != "btree":
            errors.append(f"access method {am} not supported by the validator")
        table = stmt.relation.relname
        if stmt.relation.schemaname not in (None, self.schema_name):
            errors.append(f"only {self.schema_name}-schema tables may be targeted")
        tmeta = self.schema.tables.get(table)
        if tmeta is None:
            errors.append("target table does not exist in the inspected schema")
            return ValidationOutcome(candidate=cand, valid=False, errors=errors)
        colnames = {c.name for c in tmeta.columns}
        keys: list[str] = []
        for p in stmt.indexParams or ():
            if p.name is None or p.expr is not None:
                errors.append("expression indexes are not accepted from recommendations")
                continue
            if p.name not in colnames:
                errors.append("index column does not exist on the target table")
            keys.append(p.name)
        include = [p.name for p in (stmt.indexIncludingParams or ()) if getattr(p, "name", None)]
        for c in include:
            if c not in colnames:
                errors.append("INCLUDE column does not exist on the target table")
        if not keys:
            errors.append("index has no key columns")
        if len(keys) > MAX_INDEX_COLUMNS:
            errors.append(f"more than {MAX_INDEX_COLUMNS} key columns")
        if len(set(keys)) != len(keys):
            errors.append("duplicate key column")
        where_sql = None
        if stmt.whereClause is not None:
            errors += _predicate_ok(stmt.whereClause, colnames)
            where_sql = RawStream()(stmt.whereClause)
        red, by = is_redundant(keys, table, self.schema, include, where_sql)
        if red:
            errors.append(f"redundant: existing index {by} already provides this access path")
        for ix in self.schema.indexes_for(table):
            if ix.columns == keys and ix.include_columns == include and not ix.is_partial and where_sql is None:
                errors.append(f"duplicate of existing index {ix.name}")
                break
        if errors:
            return ValidationOutcome(candidate=cand, valid=False, errors=sorted(set(errors)), warnings=warnings)
        canonical = (f"CREATE INDEX ON {quote_ident(self.schema_name) if self.schema_name != 'public' else 'public'}"
                     f".{quote_ident(table)} USING btree "
                     f"({', '.join(quote_ident(k) for k in keys)})")
        if include:
            canonical += f" INCLUDE ({', '.join(quote_ident(c) for c in include)})"
        if where_sql:
            canonical += f" WHERE {where_sql}"
        # final re-parse of the rebuilt statement (defence in depth)
        parsed = parse_sql(canonical)
        if len(parsed) != 1 or not isinstance(parsed[0].stmt, ast.IndexStmt):
            return ValidationOutcome(candidate=cand, valid=False, errors=["rebuilt DDL failed to parse"])
        return ValidationOutcome(candidate=cand, valid=True, real_ddl=canonical, warnings=warnings)

def real_index_parts(real_ddl: str) -> dict:
    """Structured view of a validated canonical DDL (trusted side: migrations / blast radius)."""
    stmt = parse_sql(real_ddl)[0].stmt
    return {"table": stmt.relation.relname,
            "columns": [p.name for p in stmt.indexParams],
            "include": [p.name for p in (stmt.indexIncludingParams or ())],
            "where": RawStream()(stmt.whereClause) if stmt.whereClause is not None else None}
