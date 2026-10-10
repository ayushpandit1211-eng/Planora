"""Strict BYOD schema verifier (Final Upgrade P3).

Runs BEFORE anything is provisioned. Parses the uploaded file with the real PostgreSQL parser (pglast) and
validates syntax, supported constructs, data types, extensions, object names, dependency order, duplicates and
forbidden operations. Nothing is silently stripped: harmless pg_dump preamble (SET, set_config(search_path),
OWNER TO, psql \\restrict lines, the system-provided hypopg extension) is removed only with an entry in
`normalized_statements`. Only `verified=True` may proceed to provisioning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pglast import ast, parse_sql, visitors
from pglast.enums import AlterTableType, ConstrType, ObjectType
from pglast.parser import ParseError

from app.security import DANGEROUS_FUNCTIONS

ALLOWED_TYPES = {
    "int2", "int4", "int8", "smallint", "integer", "bigint", "int", "serial", "bigserial", "smallserial",
    "serial4", "serial8", "serial2", "numeric", "decimal", "float4", "float8", "real", "double precision",
    "money", "bool", "boolean", "text", "varchar", "bpchar", "char", "character", "name", "citext",
    "date", "time", "timetz", "timestamp", "timestamptz", "interval", "uuid", "json", "jsonb", "bytea",
    "inet", "cidr", "macaddr", "macaddr8", "xml", "tsvector", "tsquery", "bit", "varbit", "oid",
    "int4range", "int8range", "numrange", "daterange", "tsrange", "tstzrange", "point", "line", "lseg",
    "box", "path", "polygon", "circle",
}
SYSTEM_EXTENSIONS = {"hypopg"}        # provided by the system; never installed from an upload
ALLOWED_INDEX_AM = {None, "btree", "hash", "gin", "gist", "brin", "spgist"}
ALLOWED_ALTER = {AlterTableType.AT_AddConstraint, AlterTableType.AT_AddColumn, AlterTableType.AT_ColumnDefault,
                 AlterTableType.AT_AddIndexConstraint}
FORBIDDEN_STMT = {
    "InsertStmt": "INSERT (data-bearing) is not accepted; upload schema only and use synthetic data or the CSV/JSON "
                  "loader",
    "CopyStmt": "COPY (any form) is not accepted",
    "UpdateStmt": "UPDATE is not accepted", "DeleteStmt": "DELETE is not accepted",
    "MergeStmt": "MERGE is not accepted", "TruncateStmt": "TRUNCATE is not accepted",
    "DropStmt": "DROP is not accepted", "CreateFunctionStmt": "CREATE FUNCTION/PROCEDURE is not accepted",
    "CreateTrigStmt": "CREATE TRIGGER is not accepted", "RuleStmt": "CREATE RULE is not accepted",
    "CreateEventTrigStmt": "CREATE EVENT TRIGGER is not accepted", "DoStmt": "DO blocks are not accepted",
    "CreateRoleStmt": "role management is not accepted", "AlterRoleStmt": "role management is not accepted",
    "DropRoleStmt": "role management is not accepted", "AlterSystemStmt": "ALTER SYSTEM is not accepted",
    "CreatedbStmt": "CREATE DATABASE is not accepted", "GrantStmt": "GRANT/REVOKE is not accepted",
    "GrantRoleStmt": "GRANT/REVOKE is not accepted", "CreateFdwStmt": "foreign data wrappers are not accepted",
    "CreateForeignServerStmt": "foreign servers are not accepted",
    "CreateForeignTableStmt": "foreign tables are not accepted",
    "CreateUserMappingStmt": "user mappings are not accepted",
    "ImportForeignSchemaStmt": "IMPORT FOREIGN SCHEMA is not accepted", "CallStmt": "CALL is not accepted",
    "LoadStmt": "LOAD is not accepted", "AlterDefaultPrivilegesStmt": "privilege changes are not accepted",
    "CreatePLangStmt": "procedural languages are not accepted", "CreateCastStmt": "CREATE CAST is not accepted",
    "CreateOpClassStmt": "operator classes are not accepted", "VacuumStmt": "VACUUM/ANALYZE is not accepted",
    "ExplainStmt": "EXPLAIN is not accepted in a schema file", "TransactionStmt": None,  # BEGIN/COMMIT: normalized
}
DIALECT_HINTS = [
    (re.compile(r"`[^`]+`"), "looks like MySQL (backtick identifiers)"),
    (re.compile(r"\bAUTO_INCREMENT\b", re.I), "looks like MySQL (AUTO_INCREMENT)"),
    (re.compile(r"\)\s*ENGINE\s*=", re.I), "looks like MySQL (ENGINE=)"),
    (re.compile(r"\bIDENTITY\s*\(\s*\d+\s*,\s*\d+\s*\)", re.I), "looks like SQL Server (IDENTITY(1,1))"),
    (re.compile(r"^\s*GO\s*$", re.I | re.M), "looks like SQL Server (GO batch separator)"),
    (re.compile(r"\bNVARCHAR2?\b|\bVARCHAR2\b", re.I), "looks like SQL Server / Oracle (NVARCHAR / VARCHAR2)"),
    (re.compile(r"\bAUTOINCREMENT\b", re.I), "looks like SQLite (AUTOINCREMENT)"),
]
HOW_TO_FIX = ("Produce a clean schema-only file with: pg_dump --schema-only --no-owner --no-privileges "
              "--no-comments -d <db> > schema.sql")
# quoted identifiers ("Customer Data") are valid PostgreSQL names: any printable text up to 63 bytes
_NAME_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,63}$")
_SAFE_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass
class VerificationReport:
    verified: bool = False
    schema_objects: int = 0
    rejected_statements: list[dict[str, Any]] = field(default_factory=list)
    unsupported_features: list[dict[str, Any]] = field(default_factory=list)
    normalized_statements: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    object_counts: dict[str, int] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)
    apply_statements: list[str] = field(default_factory=list)   # trusted side only; never returned to clients

    def to_dict(self) -> dict[str, Any]:
        return {"verified": self.verified, "schema_objects": self.schema_objects,
                "object_counts": self.object_counts, "tables": self.tables,
                "rejected_statements": self.rejected_statements, "unsupported_features": self.unsupported_features,
                "normalized_statements": self.normalized_statements, "warnings": self.warnings,
                "how_to_fix": HOW_TO_FIX if not self.verified or self.normalized_statements else None}


class _FuncScan(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.bad: list[str] = []
        self.subqueries = 0

    def visit_FuncCall(self, ancestors, node):
        parts = [getattr(p, "sval", "") for p in (node.funcname or ())]
        name = (parts[-1] if parts else "").lower()
        if name in DANGEROUS_FUNCTIONS - {"nextval", "setval"} or name.startswith(("dblink", "lo_", "pg_read",
                                                                                   "pg_ls", "pg_file")):
            self.bad.append(name)

    def visit_SubLink(self, ancestors, node):
        self.subqueries += 1


def _line_of(text: str, offset: int) -> int:
    b = text.encode("utf-8")
    return b[:max(offset, 0)].decode("utf-8", "ignore").count("\n") + 1


def _type_name(tn: ast.TypeName) -> str:
    names = [n.sval for n in tn.names]
    return names[-1].lower() if names else ""


def verify_schema(text: str, max_statements: int = 5000, max_bytes: int | None = None) -> VerificationReport:
    rep = VerificationReport()
    if max_bytes is not None and len(text.encode("utf-8")) > max_bytes:
        rep.rejected_statements.append({"line": 1, "statement_type": "file", "reason": "file exceeds size limit"})
        return rep
    if "\x00" in text:
        rep.rejected_statements.append({"line": 1, "statement_type": "file", "reason": "NUL byte in file"})
        return rep
    # ---- psql meta-commands (backslash lines) are not SQL: \restrict/\unrestrict (pg_dump) are normalized
    lines = text.split("\n")
    clean_lines = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("\\"):
            cmd = s.split()[0]
            if cmd in ("\\restrict", "\\unrestrict"):
                rep.normalized_statements.append({"line": i, "statement": cmd,
                                                  "reason": "pg_dump psql guard line removed"})
            else:
                rep.rejected_statements.append({"line": i, "statement_type": "psql meta-command",
                                                "reason": f"psql meta-command '{cmd}' is not SQL and is not "
                                                          "accepted"})
            clean_lines.append("")          # keep line numbering stable
        else:
            clean_lines.append(ln)
    sql = "\n".join(clean_lines)
    for pat, msg in DIALECT_HINTS:
        m = pat.search(sql)
        if m:
            rep.rejected_statements.append({"line": _line_of(sql, m.start()), "statement_type": "dialect",
                                            "reason": f"this file {msg}; only PostgreSQL DDL is supported"})
            return rep
    try:
        stmts = parse_sql(sql)
    except ParseError as exc:
        msg, pos = (exc.args + (None,))[:2]
        rep.rejected_statements.append({"line": _line_of(sql, pos or 0), "statement_type": "syntax",
                                        "reason": f"PostgreSQL parser: {msg}"})
        return rep
    if len(stmts) > max_statements:
        rep.rejected_statements.append({"line": 1, "statement_type": "file",
                                        "reason": f"{len(stmts)} statements exceed the limit of {max_statements}"})
        return rep
    if not stmts:
        rep.rejected_statements.append({"line": 1, "statement_type": "file", "reason": "no SQL statements found"})
        return rep

    schemas = {"public"}
    tables: dict[tuple[str, str], int] = {}
    objects: dict[tuple[str, str, str], int] = {}
    types_defined: set[str] = set()
    sequences: set[str] = set()
    fk_refs: list[tuple[int, str, str, int]] = []    # (stmt index, schema, table, line)
    counts: dict[str, int] = {}
    created_order: dict[tuple[str, str], int] = {}

    def ref_ok(rv: ast.RangeVar, line: int, what: str) -> tuple[str, str] | None:
        sch = (rv.schemaname or "public").lower() if rv.schemaname else "public"
        name = rv.relname or ""
        if sch not in schemas:
            reject(line, what, f"schema '{sch}' is not created in this file (only public or schemas created "
                               "here are allowed)")
            return None
        if not _NAME_RE.match(name) or len(name.encode()) > 63 or name.lower().startswith("pg_"):
            reject(line, what, f"invalid object name '{name[:40]}'")
            return None
        return sch, name

    def reject(line, stype, reason):
        rep.rejected_statements.append({"line": line, "statement_type": stype, "reason": reason})

    def unsupported(line, feature, reason):
        rep.unsupported_features.append({"line": line, "feature": feature, "reason": reason})

    def normalize(line, stype, snippet, reason):
        rep.normalized_statements.append({"line": line, "statement": f"{stype}: {snippet[:80]}", "reason": reason})

    def scan_funcs(node, line, stype) -> bool:
        v = _FuncScan()
        v(node)
        if v.bad:
            reject(line, stype, f"forbidden function(s): {', '.join(sorted(set(v.bad)))}")
            return False
        return True

    def bump(k):
        counts[k] = counts.get(k, 0) + 1

    for idx, raw in enumerate(stmts):
        st = raw.stmt
        stype = type(st).__name__
        start = raw.stmt_location or 0
        seg = sql.encode("utf-8")[start: start + raw.stmt_len].decode("utf-8", "ignore") if raw.stmt_len else \
            sql.encode("utf-8")[start:].decode("utf-8", "ignore")
        # stmt_location points at the start of the statement's leading whitespace/comments; find first SQL line
        lead = len(seg) - len(seg.lstrip())
        line = _line_of(sql, start) + seg[:lead].count("\n")
        snippet = " ".join(seg.split())[:80]
        n_before = len(rep.rejected_statements) + len(rep.unsupported_features)

        if stype in FORBIDDEN_STMT:
            if FORBIDDEN_STMT[stype] is None:
                normalize(line, stype, snippet, "transaction control removed (provisioning runs in its own "
                                                "transaction)")
            else:
                reject(line, stype, FORBIDDEN_STMT[stype])
            continue
        if stype == "VariableSetStmt":
            normalize(line, "SET", snippet, "pg_dump session preamble removed")
            continue
        if stype == "SelectStmt":
            v = _FuncScan()
            v(st)
            fn = [getattr(p, "sval", "") for t in (st.targetList or ())
                  for p in (getattr(getattr(t, "val", None), "funcname", None) or ())]
            if fn and fn[-1] == "set_config" and st.fromClause is None and "search_path" in seg:
                normalize(line, "SELECT set_config", snippet, "pg_dump search_path preamble removed")
            else:
                reject(line, stype, "SELECT statements are not accepted in a schema file (no data-bearing or "
                                    "function-calling statements)")
            continue
        if stype == "CreateExtensionStmt":
            if (st.extname or "").lower() in SYSTEM_EXTENSIONS:
                normalize(line, "CREATE EXTENSION", snippet, "HypoPG is enabled by the system; not installed from "
                                                            "the upload")
            else:
                unsupported(line, f"extension {st.extname}", "extensions cannot be installed from an upload")
            continue
        if stype == "CommentStmt":
            if st.objtype == ObjectType.OBJECT_EXTENSION:
                normalize(line, "COMMENT ON EXTENSION", snippet, "extension comments removed")
                continue
            rep.apply_statements.append(seg.strip())
            bump("comments")
            continue
        if stype == "CreateSchemaStmt":
            name = (st.schemaname or "").lower()
            if st.authrole is not None or st.schemaElts:
                reject(line, stype, "CREATE SCHEMA with AUTHORIZATION or embedded elements is not accepted")
            elif not _SAFE_SCHEMA.match(name) or name.startswith("pg_") or name in ("information_schema",
                                                                                    "public"):
                reject(line, stype, f"schema name '{name[:40]}' is not allowed")
            else:
                schemas.add(name)
                rep.apply_statements.append(seg.strip())
                bump("schemas")
            continue
        if stype == "CreateStmt":
            key = ref_ok(st.relation, line, "CREATE TABLE")
            if key is None:
                continue
            if key in tables:
                reject(line, "CREATE TABLE", f"duplicate definition of table {key[0]}.{key[1]}")
                continue
            if st.inhRelations:
                unsupported(line, "table inheritance", "INHERITS / PARTITION OF are not supported")
                continue
            ok = scan_funcs(st, line, "CREATE TABLE")
            for el in st.tableElts or ():
                if isinstance(el, ast.ColumnDef):
                    tname = _type_name(el.typeName)
                    if tname not in ALLOWED_TYPES and tname not in types_defined:
                        unsupported(line, f"type {tname}", f"column '{el.colname}': data type '{tname}' is not "
                                                            "supported (or its CREATE TYPE comes later)")
                        ok = False
                    for c in el.constraints or ():
                        if c.contype == ConstrType.CONSTR_FOREIGN and c.pktable is not None:
                            fk_refs.append((idx, (c.pktable.schemaname or "public").lower(), c.pktable.relname,
                                            line))
                elif isinstance(el, ast.Constraint) and el.contype == ConstrType.CONSTR_FOREIGN:
                    fk_refs.append((idx, (el.pktable.schemaname or "public").lower(), el.pktable.relname, line))
            if ok:
                tables[key] = idx
                created_order[key] = idx
                rep.tables.append(f"{key[0]}.{key[1]}")
                rep.apply_statements.append(seg.strip())
                bump("tables")
            continue
        if stype == "IndexStmt":
            key = ref_ok(st.relation, line, "CREATE INDEX")
            if key is None:
                continue
            if key not in tables:
                reject(line, "CREATE INDEX", f"index on unknown table {key[0]}.{key[1]} (create the table first)")
                continue
            if (st.accessMethod or None) not in ALLOWED_INDEX_AM:
                unsupported(line, f"index method {st.accessMethod}", "unsupported index access method")
                continue
            if st.idxname:
                ok_key = (key[0], "index", st.idxname)
                if ok_key in objects:
                    reject(line, "CREATE INDEX", f"duplicate index name {st.idxname}")
                    continue
                objects[ok_key] = idx
            if scan_funcs(st, line, "CREATE INDEX"):
                rep.apply_statements.append(seg.strip())
                bump("indexes")
            continue
        if stype == "CreateSeqStmt":
            key = ref_ok(st.sequence, line, "CREATE SEQUENCE")
            if key is None:
                continue
            if key[1] in sequences:
                reject(line, "CREATE SEQUENCE", f"duplicate sequence {key[1]}")
                continue
            sequences.add(key[1])
            rep.apply_statements.append(seg.strip())
            bump("sequences")
            continue
        if stype == "AlterSeqStmt":
            opts = [o.defname for o in (st.options or ())]
            if opts and all(o == "owned_by" for o in opts):
                rep.apply_statements.append(seg.strip())
                bump("sequence_ownership")
            else:
                reject(line, "ALTER SEQUENCE", "only ALTER SEQUENCE ... OWNED BY is accepted")
            continue
        if stype == "CreateEnumStmt":
            tname = [n.sval for n in st.typeName][-1].lower()
            if tname in types_defined:
                reject(line, "CREATE TYPE", f"duplicate type {tname}")
                continue
            types_defined.add(tname)
            rep.apply_statements.append(seg.strip())
            bump("types")
            continue
        if stype == "CompositeTypeStmt":
            tname = (st.typevar.relname or "").lower()
            if tname in types_defined:
                reject(line, "CREATE TYPE", f"duplicate type {tname}")
                continue
            bad = [_type_name(c.typeName) for c in st.coldeflist or ()
                   if _type_name(c.typeName) not in ALLOWED_TYPES | types_defined]
            if bad:
                unsupported(line, f"type {bad[0]}", "composite type uses an unsupported data type")
                continue
            types_defined.add(tname)
            rep.apply_statements.append(seg.strip())
            bump("types")
            continue
        if stype == "DefineStmt":
            unsupported(line, "CREATE TYPE/AGGREGATE/OPERATOR", "only ENUM and composite types are supported")
            continue
        if stype == "AlterTableStmt":
            if st.objtype not in (ObjectType.OBJECT_TABLE,):
                reject(line, "ALTER", "only ALTER TABLE is accepted")
                continue
            subs = [c.subtype for c in st.cmds or ()]
            if subs and all(s == AlterTableType.AT_ChangeOwner for s in subs):
                normalize(line, "ALTER ... OWNER TO", snippet, "ownership clause removed (sandbox role owns "
                                                              "everything)")
                continue
            key = ref_ok(st.relation, line, "ALTER TABLE")
            if key is None:
                continue
            if key not in tables:
                reject(line, "ALTER TABLE", f"ALTER of unknown table {key[0]}.{key[1]}")
                continue
            not_ok = [s.name for s in subs if s not in ALLOWED_ALTER]
            if not_ok:
                reject(line, "ALTER TABLE", f"ALTER TABLE {', '.join(not_ok)} is not accepted (only ADD COLUMN, "
                                            "ADD CONSTRAINT, SET DEFAULT)")
                continue
            for c in st.cmds:
                d = getattr(c, "def_", None)
                if isinstance(d, ast.Constraint) and d.contype == ConstrType.CONSTR_FOREIGN and d.pktable:
                    fk_refs.append((idx, (d.pktable.schemaname or "public").lower(), d.pktable.relname, line))
            if scan_funcs(st, line, "ALTER TABLE"):
                rep.apply_statements.append(seg.strip())
                bump("alter_table")
            continue
        if stype == "ViewStmt":
            key = ref_ok(st.view, line, "CREATE VIEW")
            if key is None:
                continue
            from app.security import _SafetyVisitor
            if not isinstance(st.query, ast.SelectStmt):
                reject(line, "CREATE VIEW", "view body must be a single SELECT")
                continue
            sv = _SafetyVisitor()
            sv(st.query)
            if sv.violations:
                reject(line, "CREATE VIEW", "view body rejected by the SQL safety gate: "
                                            + "; ".join(sorted(set(sv.violations))))
                continue
            rep.apply_statements.append(seg.strip())
            bump("views")
            continue
        if len(rep.rejected_statements) + len(rep.unsupported_features) == n_before:
            reject(line, stype, f"{stype} is not a supported schema construct")

    for idx, sch, tbl, line in fk_refs:
        key = (sch, tbl)
        if key not in tables:
            rep.rejected_statements.append({"line": line, "statement_type": "FOREIGN KEY",
                                            "reason": f"references unknown table {sch}.{tbl}"})
        elif tables[key] > idx:
            rep.rejected_statements.append({"line": line, "statement_type": "FOREIGN KEY",
                                            "reason": f"references {sch}.{tbl} before it is created (dependency "
                                                      "order)"})
    rep.object_counts = counts
    rep.schema_objects = sum(counts.values())
    if not counts.get("tables"):
        rep.warnings.append("no tables were defined")
    rep.verified = not rep.rejected_statements and not rep.unsupported_features and bool(counts.get("tables"))
    if not rep.verified:
        rep.apply_statements = []
    return rep
