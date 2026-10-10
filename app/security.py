"""Security gateway (Section 11).

Only a single, read-only SELECT statement may enter the analysis path. User SQL is never
executed directly: the only command built from it is

    EXPLAIN (FORMAT JSON, VERBOSE TRUE, BUFFERS TRUE) <validated query>

run inside a READ ONLY transaction with a statement timeout by app/db.py, as the
least-privilege role `qs_app`. EXPLAIN ANALYZE is never used on this path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pglast import ast, parse_sql, visitors
from pglast.parser import ParseError

from app.config import get_settings
from app.errors import ErrorCategory, SecurityRejection

EXPLAIN_PREFIX = "EXPLAIN (FORMAT JSON, VERBOSE TRUE, BUFFERS TRUE) "
# PostgreSQL 16+: plans a parameterised ($n) statement without values (pg_stat_statements text)
EXPLAIN_GENERIC_PREFIX = "EXPLAIN (FORMAT JSON, VERBOSE TRUE, BUFFERS TRUE, GENERIC_PLAN TRUE) "


def explain_command(sql: str, generic: bool = False) -> str:
    return (EXPLAIN_GENERIC_PREFIX if generic else EXPLAIN_PREFIX) + sql

# Functions with side effects, filesystem / network / process access or that disclose
# server configuration. Matched on the unqualified, lower-cased function name.
DANGEROUS_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "pg_ls_logdir",
    "pg_ls_waldir", "pg_ls_tmpdir", "pg_ls_archive_statusdir",
    "lo_import", "lo_export", "lo_get", "lo_put", "lo_from_bytea", "lo_create", "lo_unlink",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile",
    "pg_switch_wal", "pg_create_restore_point", "pg_promote", "pg_log_backend_memory_contexts",
    "set_config", "current_setting", "pg_settings",
    "dblink", "dblink_exec", "dblink_connect", "dblink_send_query",
    "query_to_xml", "query_to_xml_and_xmlschema", "table_to_xml", "cursor_to_xml",
    "nextval", "setval",
    "pg_advisory_lock", "pg_advisory_xact_lock", "pg_try_advisory_lock",
    "pg_advisory_lock_shared", "pg_try_advisory_xact_lock",
    "pg_file_write", "pg_file_rename", "pg_file_unlink",
    "hypopg_create_index", "hypopg_drop_index", "hypopg_reset", "hypopg_hide_index",
    "pg_stat_reset", "pg_stat_statements_reset",
    "txid_current", "pg_current_xact_id", "pg_notify",
    "pg_create_logical_replication_slot", "pg_drop_replication_slot",
}

# Catalog relations that expose credentials or raw large objects.
FORBIDDEN_RELATIONS = {"pg_authid", "pg_shadow", "pg_user_mappings", "pg_largeobject",
                       "pg_largeobject_metadata", "pg_hba_file_rules", "pg_file_settings",
                       "pg_settings", "pg_roles", "pg_user"}

_UTILITY_CATEGORIES = {
    "CopyStmt": "filesystem COPY is not allowed",
    "VariableSetStmt": "SET commands are not allowed",
    "AlterSystemStmt": "ALTER SYSTEM is not allowed",
    "VacuumStmt": "VACUUM/ANALYZE administrative commands are not allowed",
    "ExplainStmt": "submit the bare SELECT; Planora builds its own EXPLAIN",
    "TransactionStmt": "transaction control is not allowed",
    "DoStmt": "anonymous code blocks are not allowed",
    "CallStmt": "procedure calls are not allowed",
    "GrantStmt": "privilege changes are not allowed",
    "GrantRoleStmt": "privilege changes are not allowed",
    "CreateRoleStmt": "role management is not allowed",
    "AlterRoleStmt": "role management is not allowed",
    "DropRoleStmt": "role management is not allowed",
    "LoadStmt": "LOAD is not allowed",
    "CheckPointStmt": "CHECKPOINT is not allowed",
    "ReindexStmt": "REINDEX is not allowed",
    "ClusterStmt": "CLUSTER is not allowed",
    "LockStmt": "LOCK is not allowed",
    "NotifyStmt": "NOTIFY is not allowed",
    "InsertStmt": "write statements are not analysed on this path",
    "UpdateStmt": "write statements are not analysed on this path",
    "DeleteStmt": "write statements are not analysed on this path",
    "MergeStmt": "write statements are not analysed on this path",
    "TruncateStmt": "TRUNCATE is not allowed",
}

_DDL_PREFIXES = ("Create", "Alter", "Drop", "Rename", "Comment", "Define", "Index", "View",
                 "Rule", "Trigger", "Sec", "Import")


@dataclass
class ValidatedQuery:
    sql: str                      # stripped, single statement (still raw - trusted side only)
    statement_type: str
    referenced_relations: list[str] = field(default_factory=list)
    function_calls: list[str] = field(default_factory=list)
    length: int = 0
    has_params: bool = False      # $n placeholders -> EXPLAIN uses GENERIC_PLAN

    @property
    def explain_sql(self) -> str:
        return explain_command(self.sql, self.has_params)


class _SafetyVisitor(visitors.Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.relations: list[str] = []
        self.functions: list[str] = []
        self.violations: list[str] = []
        self.params = 0

    def visit_ParamRef(self, ancestors, node):
        self.params += 1

    def visit_RangeVar(self, ancestors, node: ast.RangeVar):
        name = (node.relname or "").lower()
        self.relations.append(name)
        if name in FORBIDDEN_RELATIONS:
            self.violations.append(f"access to restricted catalog relation '{name}'")

    def visit_FuncCall(self, ancestors, node: ast.FuncCall):
        parts = [getattr(p, "sval", "") for p in (node.funcname or ())]
        fname = (parts[-1] if parts else "").lower()
        self.functions.append(fname)
        if fname in DANGEROUS_FUNCTIONS or fname.startswith("dblink") or fname.startswith("lo_"):
            self.violations.append(f"dangerous function '{fname}'")

    def visit_SelectStmt(self, ancestors, node: ast.SelectStmt):
        if node.intoClause is not None:
            self.violations.append("SELECT INTO creates a table and is not allowed")
        if node.lockingClause:
            self.violations.append("row-locking clauses (FOR UPDATE/SHARE) are not allowed")

    def visit_CommonTableExpr(self, ancestors, node: ast.CommonTableExpr):
        if not isinstance(node.ctequery, ast.SelectStmt):
            self.violations.append("data-modifying CTEs are not allowed")


class SecurityGateway:
    def __init__(self, max_query_length: int | None = None):
        self.max_query_length = max_query_length or get_settings().max_query_length

    def validate(self, sql: str) -> ValidatedQuery:
        if sql is None or not str(sql).strip():
            raise SecurityRejection(ErrorCategory.INVALID_SQL, "empty query", stage="security")
        sql = str(sql)
        if len(sql) > self.max_query_length:
            raise SecurityRejection(
                ErrorCategory.UNSAFE_SQL,
                f"query length {len(sql)} exceeds limit {self.max_query_length}",
                stage="security", details={"rule": "max_query_length"})
        if "\x00" in sql:
            raise SecurityRejection(ErrorCategory.UNSAFE_SQL, "NUL byte in query", stage="security")
        try:
            stmts = parse_sql(sql)
        except ParseError as exc:
            # pglast error text can quote the input around the error position -> keep only position
            pos = getattr(exc, "location", None)
            raise SecurityRejection(ErrorCategory.INVALID_SQL, "SQL could not be parsed",
                                    stage="security", details={"position": pos}) from None
        if len(stmts) != 1:
            raise SecurityRejection(ErrorCategory.UNSAFE_SQL,
                                    f"exactly one statement is allowed, found {len(stmts)}",
                                    stage="security", details={"rule": "single_statement"})
        stmt = stmts[0].stmt
        stype = type(stmt).__name__
        if stype in _UTILITY_CATEGORIES:
            cat = ErrorCategory.UNSAFE_SQL
            raise SecurityRejection(cat, f"{stype}: {_UTILITY_CATEGORIES[stype]}", stage="security",
                                    details={"rule": "statement_type", "statement_type": stype})
        if stype != "SelectStmt":
            reason = "DDL is not allowed on analysis paths" if stype.startswith(_DDL_PREFIXES) \
                else "only SELECT statements can be analysed"
            raise SecurityRejection(ErrorCategory.UNSAFE_SQL, f"{stype}: {reason}", stage="security",
                                    details={"rule": "statement_type", "statement_type": stype})
        visitor = _SafetyVisitor()
        visitor(stmts[0])
        if visitor.violations:
            raise SecurityRejection(ErrorCategory.UNSAFE_SQL, "; ".join(sorted(set(visitor.violations))),
                                    stage="security", details={"rule": "content"})
        clean = sql.strip().rstrip(";").strip()
        return ValidatedQuery(sql=clean, statement_type=stype,
                              referenced_relations=sorted(set(visitor.relations)),
                              function_calls=sorted(set(visitor.functions)), length=len(sql),
                              has_params=visitor.params > 0)

    @staticmethod
    def classify_statement(sql: str) -> str | None:
        """Statement type for already-normalised workload SQL (used by blast-radius); None if unparsable."""
        try:
            stmts = parse_sql(sql)
        except ParseError:
            return None
        return type(stmts[0].stmt).__name__ if stmts else None
