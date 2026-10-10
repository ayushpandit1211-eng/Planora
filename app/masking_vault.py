"""Privacy vault and SQL obfuscation (Sections 12-15).

Literal behaviour   : 'amit@example.com' -> $1, 5000 -> $2, '2026-10-01' -> $3   (pglast AST)
Identifier behaviour: orders -> tbl_a31f7c, tenant_id -> col_c81d44            (HMAC-SHA256)

HMAC is NOT reversible. Reverse translation is possible only through the private,
request-scoped mapping vault kept on the trusted backend. The reverse mapping is never
returned by the API, never logged and never placed in an LLM payload.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from pglast import ast, fingerprint, parse_sql, visitors
from pglast.parser import ParseError
from pglast.stream import RawStream

from app.config import get_settings
from app.contracts import PrivacyAudit, PrivacyResult
from app.errors import ErrorCategory, PrivacyViolation
from app.pii import PIIDetector, encode_bitmask

PSEUDONYM_RE = re.compile(r"\b(tbl|col|als|sch|idx)_[0-9a-f]{6,16}\b")
PARAM_RE = re.compile(r"\$(\d+)\b")
_TEMP_PARAM_BASE = 100000
PREFIX = {"table": "tbl", "column": "col", "alias": "als", "schema": "sch", "index": "idx"}
UNMASKED_SCHEMAS = {"pg_catalog", "information_schema"}


# =========================================================================== vault backends
class VaultBackend(Protocol):
    def put(self, request_id: str, data: dict) -> None: ...
    def get(self, request_id: str) -> dict | None: ...
    def delete(self, request_id: str) -> None: ...


class InMemoryVaultBackend:
    """Request-scoped in-process store with TTL expiry (default for the local prototype)."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._data: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _gc(self) -> None:
        now = time.time()
        for k in [k for k, (ts, _) in self._data.items() if now - ts > self.ttl]:
            self._data.pop(k, None)

    def put(self, request_id: str, data: dict) -> None:
        with self._lock:
            self._gc()
            self._data[request_id] = (time.time(), data)

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            self._gc()
            item = self._data.get(request_id)
            return item[1] if item else None

    def delete(self, request_id: str) -> None:
        with self._lock:
            self._data.pop(request_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def build_vault_backend() -> VaultBackend:
    s = get_settings()
    return InMemoryVaultBackend(s.vault_ttl_seconds)


# =========================================================================== pseudonymizer
class Pseudonymizer:
    def __init__(self, key: bytes):
        self._key = key

    def token(self, name: str, kind: str, length: int = 6) -> str:
        digest = hmac.new(self._key, name.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{PREFIX[kind]}_{digest[:length]}"


@dataclass
class RequestVault:
    """Private per-request mapping. Lives only on the trusted backend."""
    request_id: str
    forward: dict[str, str] = field(default_factory=dict)        # original -> pseudonym
    reverse: dict[str, str] = field(default_factory=dict)        # pseudonym -> original
    kinds: dict[str, str] = field(default_factory=dict)          # pseudonym -> kind
    literals: dict[str, str] = field(default_factory=dict)       # "$n" -> SQL literal text
    literal_values: list[str] = field(default_factory=list)      # raw values (guard checks only)

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "forward": self.forward, "reverse": self.reverse,
                "kinds": self.kinds, "literals": self.literals, "literal_values": self.literal_values}

    @classmethod
    def from_dict(cls, d: dict) -> "RequestVault":
        return cls(request_id=d["request_id"], forward=d.get("forward", {}), reverse=d.get("reverse", {}),
                   kinds=d.get("kinds", {}), literals=d.get("literals", {}),
                   literal_values=d.get("literal_values", []))


def _sql_literal(node: ast.A_Const) -> tuple[str, str]:
    """Return (sql_text, raw_value) for an A_Const."""
    val = node.val
    if isinstance(val, ast.String):
        raw = val.sval
        return "'" + raw.replace("'", "''") + "'", raw
    if isinstance(val, ast.Integer):
        return str(val.ival), str(val.ival)
    if isinstance(val, ast.Float):
        return str(val.fval), str(val.fval)
    if isinstance(val, ast.Boolean):
        return ("true" if val.boolval else "false"), str(val.boolval)
    if isinstance(val, ast.BitString):
        return f"B'{val.bsval.lstrip('b')}'", val.bsval
    raise ValueError("unsupported constant")


def _is_ordinal_position(ancestors) -> bool:
    if isinstance(ancestors.node, ast.SortBy) and ancestors.member == "node":
        return True
    parent = ancestors.parent
    return parent is not None and parent.member == "groupClause" and isinstance(ancestors.node, tuple)


class _LiteralCollector(visitors.Visitor):
    """Replace literal constants with temporary ParamRefs ($100000+k)."""

    def __init__(self):
        super().__init__()
        self.items: list[tuple[int, str, str]] = []   # temp_number, sql_text, raw

    def visit_A_Const(self, ancestors, node: ast.A_Const):
        if node.isnull:
            return None
        if isinstance(node.val, ast.Integer) and _is_ordinal_position(ancestors):
            return None   # GROUP BY 1 / ORDER BY 2 are structural ordinals, not data values
        sql_text, raw = _sql_literal(node)
        n = _TEMP_PARAM_BASE + len(self.items)
        self.items.append((n, sql_text, raw))
        return ast.ParamRef(number=n)

    def visit_ParamRef(self, ancestors, node: ast.ParamRef):
        if node.number >= _TEMP_PARAM_BASE:
            return None
        n = _TEMP_PARAM_BASE + len(self.items)
        self.items.append((n, f"${node.number}", f"${node.number}"))  # pre-normalised input param
        return ast.ParamRef(number=n)


class _NameCollector(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.tables: set[str] = set()
        self.aliases: set[str] = set()
        self.schemas: set[str] = set()

    def visit_RangeVar(self, anc, node):
        if node.relname:
            self.tables.add(node.relname)
        if node.schemaname:
            self.schemas.add(node.schemaname)

    def visit_CommonTableExpr(self, anc, node):
        self.tables.add(node.ctename)

    def visit_Alias(self, anc, node):
        self.aliases.add(node.aliasname)

    def visit_ResTarget(self, anc, node):
        if node.name:
            self.aliases.add(node.name)


class _IdentifierRewriter(visitors.Visitor):
    def __init__(self, resolve):
        super().__init__()
        self.resolve = resolve   # (name, default_kind) -> pseudonym

    def visit_RangeVar(self, anc, node):
        if node.schemaname and node.schemaname not in UNMASKED_SCHEMAS:
            node.schemaname = self.resolve(node.schemaname, "schema")
        if node.relname and node.schemaname not in UNMASKED_SCHEMAS:
            node.relname = self.resolve(node.relname, "table")

    def visit_CommonTableExpr(self, anc, node):
        node.ctename = self.resolve(node.ctename, "table")
        if node.aliascolnames:
            node.aliascolnames = tuple(ast.String(sval=self.resolve(s.sval, "column")) for s in node.aliascolnames)

    def visit_Alias(self, anc, node):
        node.aliasname = self.resolve(node.aliasname, "alias")
        if node.colnames:
            node.colnames = tuple(ast.String(sval=self.resolve(s.sval, "column")) for s in node.colnames)

    def visit_ResTarget(self, anc, node):
        if node.name:
            # in UPDATE SET targets the name is a column; in SELECT lists it is an output alias
            node.name = self.resolve(node.name, "alias")

    def visit_ColumnRef(self, anc, node):
        fields = list(node.fields or ())
        strs = [f for f in fields if isinstance(f, ast.String)]
        if not strs:
            return None
        new_fields = []
        n = len(fields)
        for i, f in enumerate(fields):
            if not isinstance(f, ast.String):
                new_fields.append(f)
                continue
            if i == n - 1:
                kind = "column"
            elif n == 3 and i == 0:
                kind = "schema"
            else:
                kind = "qualifier"
            new_fields.append(ast.String(sval=self.resolve(f.sval, kind)))
        node.fields = tuple(new_fields)

    def visit_JoinExpr(self, anc, node):
        if node.usingClause:
            node.usingClause = tuple(ast.String(sval=self.resolve(s.sval, "column")) for s in node.usingClause)

    def visit_IndexElem(self, anc, node):
        if node.name:
            node.name = self.resolve(node.name, "column")

    def visit_IndexStmt(self, anc, node):
        if node.idxname:
            node.idxname = self.resolve(node.idxname, "index")


# =========================================================================== obfuscator
class DataObfuscator:
    def __init__(self, backend: VaultBackend | None = None, hmac_key: bytes | None = None,
                 known_tables: Iterable[str] | None = None):
        self.backend = backend or build_vault_backend()
        self.pseudo = Pseudonymizer(hmac_key or get_settings().hmac_key_bytes())
        self.detector = PIIDetector()
        self.known_tables = set(known_tables or ())

    # ------------------------------------------------------------- vault helpers
    def _load(self, request_id: str) -> RequestVault:
        data = self.backend.get(request_id)
        return RequestVault.from_dict(data) if data else RequestVault(request_id=request_id)

    def _save(self, vault: RequestVault) -> None:
        self.backend.put(vault.request_id, vault.to_dict())

    def get_vault(self, request_id: str) -> RequestVault | None:
        """Trusted-side access only (translation, sandbox, guard)."""
        data = self.backend.get(request_id)
        return RequestVault.from_dict(data) if data else None

    def _register(self, vault: RequestVault, name: str, kind: str) -> str:
        if name in vault.forward:
            return vault.forward[name]
        length = 6
        while True:
            token = self.pseudo.token(name, kind, length)
            owner = vault.reverse.get(token)
            if owner is None or owner == name:
                break
            length += 2   # collision inside this request -> lengthen deterministically
        vault.forward[name] = token
        vault.reverse[token] = name
        vault.kinds[token] = kind
        return token

    def pseudonymize_identifier(self, request_id: str, name: str, kind: str) -> str:
        """Register/lookup a pseudonym (used by schema inspector and plan sanitizer)."""
        vault = self._load(request_id)
        token = self._register(vault, name, kind)
        self._save(vault)
        return token

    def pseudonymize_many(self, request_id: str, names: dict[str, str]) -> dict[str, str]:
        vault = self._load(request_id)
        out = {n: self._register(vault, n, k) for n, k in names.items()}
        self._save(vault)
        return out

    # ------------------------------------------------------------- mask
    def mask_sql(self, raw_sql: str, request_id: str) -> dict:
        scan = self.detector.scan(raw_sql)
        vault = self._load(request_id)
        try:
            stmts = parse_sql(raw_sql)
            if not stmts:
                raise ValueError("empty")
            result = self._mask_ast(raw_sql, stmts, vault, scan)
        except (ParseError, ValueError, AttributeError, TypeError):
            # AttributeError/TypeError: pglast cannot deparse a parameterised utility node
            result = self._mask_fallback(raw_sql, vault, scan)
        self._save(vault)
        return result.model_dump()

    def _mask_ast(self, raw_sql: str, stmts, vault: RequestVault, scan) -> PrivacyResult:
        names = _NameCollector()
        for s in stmts:
            names(s)
        tables = names.tables | self.known_tables
        aliases = names.aliases - tables

        def resolve(name: str, kind: str) -> str:
            if kind == "qualifier":
                kind = "table" if name in tables else "alias"
            elif kind == "column":
                if name in aliases and name not in vault.forward:
                    kind = "alias"   # ORDER BY / GROUP BY on an output alias
                elif name in vault.forward:
                    return vault.forward[name]
            elif kind == "alias" and name in tables:
                kind = "table"
            return self._register(vault, name, kind)

        collector = _LiteralCollector()
        rewriter = _IdentifierRewriter(resolve)
        parts = []
        for s in stmts:
            collector(s)
            rewriter(s)
            parts.append(RawStream()(s))
        deparsed = "; ".join(parts)

        # renumber temp params in TEXTUAL order -> $1..$n
        temp = {n: (sql_text, raw) for n, sql_text, raw in collector.items}
        order: dict[int, int] = {}

        def _renum(m: re.Match) -> str:
            t = int(m.group(1))
            if t < _TEMP_PARAM_BASE:
                return m.group(0)
            if t not in order:
                order[t] = len(order) + 1
            return f"${order[t]}"

        masked = PARAM_RE.sub(_renum, deparsed)
        current_literals: list[str] = []
        for t, n in order.items():
            sql_text, raw = temp[t]
            vault.literals[f"${n}"] = sql_text
            if not raw.startswith("$"):
                vault.literal_values.append(raw)
                current_literals.append(raw)

        bitmask, flags = encode_bitmask(scan, stmts[0])
        audit = self._audit_masked(masked, current_literals, scan)
        return PrivacyResult(
            masked_sql=masked,
            ast_signature=fingerprint(raw_sql),
            literal_count=len(order),
            mapped_tables={k: "table" for k, v in vault.kinds.items() if v == "table"},
            mapped_columns={k: "column" for k, v in vault.kinds.items() if v == "column"},
            fallback_used=False,
            quarantined=not audit.safe_for_llm,
            privacy_bitmask=bitmask,
            privacy_flags=flags,
            privacy_audit=audit,
        )

    _FALLBACK_STRING = re.compile(r"(?:[EeBbXxNn])?'(?:[^']|'')*'")
    _FALLBACK_NUMBER = re.compile(r"(?<![\w$])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)")
    _FALLBACK_UNSAFE = re.compile(r"\$[A-Za-z_]*\$|\"|--|/\*")

    def _mask_fallback(self, raw_sql: str, vault: RequestVault, scan) -> PrivacyResult:
        """Controlled fallback: only plain quoted strings / numbers and KNOWN identifiers.

        Dollar quoting, quoted identifiers and comments cannot be anonymised safely without an
        AST, so the request is quarantined instead.
        """
        if self._FALLBACK_UNSAFE.search(raw_sql):
            return self._quarantine(raw_sql, scan, "constructs that cannot be anonymised without an AST")
        counter = [0]
        current_literals: list[str] = []

        def lit(m: re.Match) -> str:
            counter[0] += 1
            key = f"${counter[0]}"
            vault.literals[key] = m.group(0)
            vault.literal_values.append(m.group(0).strip("'"))
            current_literals.append(m.group(0).strip("'"))
            return key

        masked = self._FALLBACK_STRING.sub(lit, raw_sql)
        masked = self._FALLBACK_NUMBER.sub(lit, masked)
        known = sorted(self.known_tables, key=len, reverse=True)
        for name in known:
            masked = re.sub(rf"\b{re.escape(name)}\b", self._register(vault, name, "table"), masked)
        audit = self._audit_masked(masked, current_literals, scan)
        if not audit.safe_for_llm:
            return self._quarantine(raw_sql, scan, "fallback output failed privacy verification")
        bitmask, flags = encode_bitmask(scan, None)
        return PrivacyResult(masked_sql=masked, ast_signature="fallback:" + hashlib.sha256(masked.encode()).hexdigest()[:16],
                             literal_count=counter[0], fallback_used=True, privacy_bitmask=bitmask,
                             privacy_flags=flags, privacy_audit=audit,
                             mapped_tables={k: "table" for k, v in vault.kinds.items() if v == "table"})

    def _quarantine(self, raw_sql: str, scan, reason: str) -> PrivacyResult:
        audit = PrivacyAudit(pii_detected=scan.pii_detected, pii_types=sorted(scan.pii_types),
                             literal_leak_detected=True, secret_leak_detected=scan.secret_detected,
                             safe_for_llm=False)
        bitmask, flags = encode_bitmask(scan, None)
        return PrivacyResult(masked_sql="", ast_signature="quarantined", literal_count=0,
                             fallback_used=True, quarantined=True, quarantine_reason=reason,
                             privacy_bitmask=bitmask, privacy_flags=flags, privacy_audit=audit)

    def _audit_masked(self, masked: str, literals: list[str], raw_scan) -> PrivacyAudit:
        post = self.detector.scan(masked)
        literal_leak = any(len(v) >= 3 and v in masked for v in literals
                           if not v.isdigit() or len(v) >= 5)
        secret_leak = post.secret_detected
        safe = not (post.pii_detected or literal_leak or secret_leak)
        return PrivacyAudit(pii_detected=raw_scan.pii_detected or raw_scan.secret_detected,
                            pii_types=sorted(set(raw_scan.pii_types) | set(raw_scan.secret_types)),
                            literal_leak_detected=literal_leak, secret_leak_detected=secret_leak,
                            safe_for_llm=safe)

    # ------------------------------------------------------------- reverse translation
    def _translate(self, masked: str, request_id: str, restore_literals: bool) -> str:
        vault = self.get_vault(request_id)
        if vault is None:
            raise PrivacyViolation(ErrorCategory.NOT_FOUND, "no vault for request (expired or cleared)",
                                   stage="translation")
        unknown = sorted({m.group(0) for m in PSEUDONYM_RE.finditer(masked)} - set(vault.reverse))
        if unknown:
            raise PrivacyViolation(ErrorCategory.UNKNOWN_IDENTIFIER,
                                   f"{len(unknown)} identifier(s) not present in the anonymised schema",
                                   stage="translation", details={"unknown_pseudonyms": unknown})

        def ident(m: re.Match) -> str:
            original = vault.reverse[m.group(0)]
            if re.fullmatch(r"[a-z_][a-z0-9_$]*", original):
                return original
            return '"' + original.replace('"', '""') + '"'

        out = PSEUDONYM_RE.sub(ident, masked)
        if restore_literals:
            missing = [m.group(0) for m in PARAM_RE.finditer(out) if m.group(0) not in vault.literals]
            if missing:
                raise PrivacyViolation(ErrorCategory.SCHEMA_MISMATCH,
                                       "statement references parameters that do not exist in the original query",
                                       stage="translation", details={"params": sorted(set(missing))})
            out = PARAM_RE.sub(lambda m: vault.literals[m.group(0)], out)
        return out

    def translate_llm_sql_to_real(self, masked_sql: str, request_id: str) -> str:
        real = self._translate(masked_sql, request_id, restore_literals=True)
        try:
            stmts = parse_sql(real)
        except ParseError:
            raise PrivacyViolation(ErrorCategory.INVALID_SQL, "translated SQL does not parse",
                                   stage="translation") from None
        if len(stmts) != 1 or not isinstance(stmts[0].stmt, ast.SelectStmt):
            raise PrivacyViolation(ErrorCategory.UNSAFE_SQL, "translated SQL must be a single SELECT",
                                   stage="translation")
        return real

    def translate_llm_ddl_to_real(self, llm_ddl: str, request_id: str) -> str:
        real = self._translate(llm_ddl, request_id, restore_literals=True)
        try:
            stmts = parse_sql(real)
        except ParseError:
            raise PrivacyViolation(ErrorCategory.INVALID_DDL, "translated DDL does not parse",
                                   stage="translation") from None
        if len(stmts) != 1 or not isinstance(stmts[0].stmt, ast.IndexStmt):
            raise PrivacyViolation(ErrorCategory.INVALID_DDL, "only a single CREATE INDEX is allowed",
                                   stage="translation")
        return real

    def to_masked_identifier(self, request_id: str, original: str) -> str | None:
        vault = self.get_vault(request_id)
        return vault.forward.get(original) if vault else None

    def original_identifier(self, request_id: str, pseudonym: str) -> str | None:
        """Trusted-side lookup. Never call when building LLM payloads."""
        vault = self.get_vault(request_id)
        return vault.reverse.get(pseudonym) if vault else None

    def clear_vault(self, request_id: str) -> None:
        self.backend.delete(request_id)
