"""PII pre-filter (Section 12) and privacy bitmask encoder (Section 13).

Detection reports TYPES and COUNTS only - never the matched values. A query containing
an email or UUID literal is NOT rejected; it is routed through AST parameterisation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntFlag

from pglast import visitors

# ------------------------------------------------------------------ patterns
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "phone": re.compile(r"(?<![\w$.:])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?)?\d{2,5}(?:[\s.-]\d{2,5}){1,3}(?![\w.:])"
                        r"|(?<![\w$])\+\d{10,14}(?!\w)"),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "credit_card_like": re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])"),
}

SECRET_PATTERNS: dict[str, re.Pattern] = {
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"),
    "api_key": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}"
                          r"|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}"
                          r"|glpat-[A-Za-z0-9_-]{20,})\b"),
    "password_assignment": re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\b\s*(?:=|:|:=)\s*['\"]?[^\s'\",;)]{3,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "connection_string": re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:@/]+:[^\s@/]+@"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    "generic_hex_secret": re.compile(r"\b[0-9a-f]{40,}\b"),
}

PII_TYPES = tuple(PII_PATTERNS)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
SECRET_TYPES = tuple(SECRET_PATTERNS)


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 13:
        return False
    total, alt = 0, False
    for d in reversed(nums):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class PIIScanResult:
    pii_types: dict[str, int] = field(default_factory=dict)
    secret_types: dict[str, int] = field(default_factory=dict)

    @property
    def pii_detected(self) -> bool:
        return bool(self.pii_types)

    @property
    def secret_detected(self) -> bool:
        return bool(self.secret_types)

    def as_dict(self) -> dict:
        return {"pii_detected": self.pii_detected, "pii_types": sorted(self.pii_types),
                "secret_detected": self.secret_detected, "secret_types": sorted(self.secret_types),
                "counts": {**self.pii_types, **self.secret_types}}


class PIIDetector:
    """Regex detector for obvious PII and secret patterns."""

    def scan(self, text: str) -> PIIScanResult:
        res = PIIScanResult()
        if not text:
            return res
        for name, pat in PII_PATTERNS.items():
            matches = pat.findall(text)
            if name == "credit_card_like":
                matches = [m for m in matches if _luhn_ok(m)]
            if name == "phone":
                # 9-15 digits, and not an ISO date / date-time fragment (2026-10-01 12)
                matches = [m for m in matches if 9 <= sum(c.isdigit() for c in m) <= 15
                           and not _ISO_DATE.match(m.strip())]
            if matches:
                res.pii_types[name] = len(matches)
        for name, pat in SECRET_PATTERNS.items():
            matches = pat.findall(text)
            if matches:
                res.secret_types[name] = len(matches)
        return res


# ------------------------------------------------------------------ bitmask
class PrivacyFlag(IntFlag):
    CONTAINS_EMAIL_LITERAL = 1 << 0
    CONTAINS_UUID_LITERAL = 1 << 1
    CONTAINS_PHONE_PATTERN = 1 << 2
    CONTAINS_JOIN = 1 << 3
    CONTAINS_SORT = 1 << 4
    CONTAINS_AGGREGATION = 1 << 5
    CONTAINS_SUBQUERY = 1 << 6
    CONTAINS_SEQUENTIAL_SCAN = 1 << 7
    CONTAINS_INDEX_SCAN = 1 << 8
    CONTAINS_PARTITION_CANDIDATE = 1 << 9
    CONTAINS_SECRET_PATTERN = 1 << 10
    CONTAINS_LIMIT = 1 << 11
    CONTAINS_CTE = 1 << 12
    CONTAINS_WINDOW = 1 << 13
    CONTAINS_DATE_RANGE = 1 << 14


AGG_FUNCS = {"count", "sum", "avg", "min", "max", "array_agg", "string_agg", "bool_and", "bool_or",
             "stddev", "variance", "percentile_cont", "percentile_disc", "json_agg", "jsonb_agg"}


class _StructVisitor(visitors.Visitor):
    def __init__(self):
        super().__init__()
        self.join = self.sort = self.agg = self.subquery = self.limit = self.cte = self.window = False
        self.range_ops = False
        self.selects = 0

    def visit_JoinExpr(self, anc, node):
        self.join = True

    def visit_SelectStmt(self, anc, node):
        self.selects += 1
        if node.sortClause:
            self.sort = True
        if node.groupClause:
            self.agg = True
        if node.limitCount is not None:
            self.limit = True
        if node.withClause is not None:
            self.cte = True
        if node.fromClause and len(node.fromClause) > 1:
            self.join = True

    def visit_SubLink(self, anc, node):
        self.subquery = True

    def visit_RangeSubselect(self, anc, node):
        self.subquery = True

    def visit_FuncCall(self, anc, node):
        fname = getattr((node.funcname or [None])[-1], "sval", "") or ""
        if fname.lower() in AGG_FUNCS:
            self.agg = True
        if node.over is not None:
            self.window = True

    def visit_A_Expr(self, anc, node):
        ops = [getattr(n, "sval", "") for n in (node.name or ())]
        if any(o in ("<", ">", "<=", ">=") for o in ops) or str(node.kind).endswith("BETWEEN"):
            self.range_ops = True


def encode_bitmask(raw_scan: PIIScanResult, parsed_stmt=None, plan_histogram: dict[str, int] | None = None,
                   partition_candidate: bool = False) -> tuple[int, dict[str, bool]]:
    """Encode sensitive/structural attributes as bit fields (Section 13).

    This is an abstraction for downstream models; it is NOT a substitute for literal
    removal, which is performed separately by AST parameterisation.
    """
    flags = PrivacyFlag(0)
    if raw_scan.pii_types.get("email"):
        flags |= PrivacyFlag.CONTAINS_EMAIL_LITERAL
    if raw_scan.pii_types.get("uuid"):
        flags |= PrivacyFlag.CONTAINS_UUID_LITERAL
    if raw_scan.pii_types.get("phone"):
        flags |= PrivacyFlag.CONTAINS_PHONE_PATTERN
    if raw_scan.secret_detected:
        flags |= PrivacyFlag.CONTAINS_SECRET_PATTERN
    if parsed_stmt is not None:
        v = _StructVisitor()
        v(parsed_stmt)
        if v.join:
            flags |= PrivacyFlag.CONTAINS_JOIN
        if v.sort:
            flags |= PrivacyFlag.CONTAINS_SORT
        if v.agg:
            flags |= PrivacyFlag.CONTAINS_AGGREGATION
        if v.subquery or v.selects > 1:
            flags |= PrivacyFlag.CONTAINS_SUBQUERY
        if v.limit:
            flags |= PrivacyFlag.CONTAINS_LIMIT
        if v.cte:
            flags |= PrivacyFlag.CONTAINS_CTE
        if v.window:
            flags |= PrivacyFlag.CONTAINS_WINDOW
        if v.range_ops:
            flags |= PrivacyFlag.CONTAINS_DATE_RANGE
    if plan_histogram:
        if plan_histogram.get("Seq Scan", 0):
            flags |= PrivacyFlag.CONTAINS_SEQUENTIAL_SCAN
        if any(plan_histogram.get(k, 0) for k in ("Index Scan", "Index Only Scan", "Bitmap Index Scan")):
            flags |= PrivacyFlag.CONTAINS_INDEX_SCAN
    if partition_candidate:
        flags |= PrivacyFlag.CONTAINS_PARTITION_CANDIDATE
    named = {f.name.lower(): bool(flags & f) for f in PrivacyFlag}
    return int(flags), named


def decode_bitmask(value: int) -> list[str]:
    return [f.name.lower() for f in PrivacyFlag if value & f]
