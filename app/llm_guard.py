"""Zero-leak outgoing LLM payload guard (Sections 4.1 and 15).

Every payload leaving the trusted backend for an external LLM passes `inspect()`. If any
forbidden content is present the request is BLOCKED, an audit event is written with the
request ID, and a structured PrivacyViolation is raised.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.audit import AuditLogger, audit_logger
from app.errors import ErrorCategory, PrivacyViolation
from app.masking_vault import DataObfuscator
from app.pii import PIIDetector

# Words that also occur in normal English / SQL keywords: an original identifier equal to one
# of these is still checked, but only as an exact JSON string value (not free-text match).
_COMMON_WORDS = {"name", "status", "id", "date", "type", "value", "key", "count", "time", "data"}


def _string_values(obj: Any):
    """Yield every string VALUE in a nested payload (dict keys are fixed schema, not data)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _string_values(v)


@dataclass
class GuardReport:
    allowed: bool
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "violations": self.violations}


class OutgoingPayloadGuard:
    def __init__(self, obfuscator: DataObfuscator, audit: AuditLogger | None = None):
        self.obfuscator = obfuscator
        self.audit = audit or audit_logger
        self.detector = PIIDetector()

    def check(self, request_id: str, template: str, data: Any,
              extra_forbidden: Iterable[str] = ()) -> GuardReport:
        data_text = json.dumps(data, default=str, sort_keys=True)
        full_text = template + "\n" + data_text
        violations: list[str] = []

        scan = self.detector.scan(full_text)
        for t in scan.pii_types:
            violations.append(f"pii:{t}")
        for t in scan.secret_types:
            violations.append(f"secret:{t}")

        vault = self.obfuscator.get_vault(request_id)
        if vault is not None:
            for raw in vault.literal_values:
                if raw.startswith("$"):
                    continue
                if (len(raw) >= 4 or (raw.isdigit() and len(raw) >= 3)) and raw in full_text:
                    violations.append("raw_literal")
                    break
            originals = [o for o in vault.reverse.values() if len(o) >= 3]
            if originals:
                common = {o for o in originals if o.lower() in _COMMON_WORDS}
                pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(
                    re.escape(o) for o in sorted(set(originals) - common, key=len, reverse=True)) + r")(?![A-Za-z0-9_])") \
                    if set(originals) - common else None
                for value in _string_values(data):
                    # common English words (name, status, ...) only count as an exact value match
                    if value in common or (pattern is not None and pattern.search(value)):
                        violations.append("reverse_map_identifier")
                        break
        for value in extra_forbidden:
            if value and len(str(value)) >= 3 and str(value) in full_text:
                violations.append("raw_result_value")
                break
        return GuardReport(allowed=not violations, violations=sorted(set(violations)))

    def inspect(self, request_id: str, template: str, data: Any,
                extra_forbidden: Iterable[str] = ()) -> None:
        report = self.check(request_id, template, data, extra_forbidden)
        if report.allowed:
            return
        self.audit.record(request_id, "PRIVACY_BLOCK", stage="llm_guard", severity="security",
                          details={"violations": report.violations})
        category = ErrorCategory.SECRET_LEAK if any(v.startswith("secret") for v in report.violations) \
            else ErrorCategory.PII_LEAK
        raise PrivacyViolation(category, "outgoing LLM payload blocked by privacy guard",
                               stage="llm_guard", details=report.as_dict())
