"""Structured JSON logging (Section 56).

Log records carry request_id / query_fingerprint / stage / duration / model mode /
sandbox status / error category. A redaction filter scrubs quoted literals, emails,
UUIDs, tokens and DSN passwords from every message as a last line of defence; callers
must still never pass raw SQL to the logger.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any

_STRUCT_FIELDS = ("request_id", "query_fingerprint", "stage", "duration_ms", "model_mode",
                  "sandbox_status", "error_category", "event", "trace_id", "data_source_id")

_REDACTIONS = [
    (re.compile(r"postgres(?:ql)?://[^\s]+", re.I), "postgresql://<redacted>"),
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[=:]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"'(?:[^']|'')*'"), "'?'"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), "<jwt>"),
    (re.compile(r"\b(?:sk|pk|AKIA|AIza|ghp|xox[bap])[-_A-Za-z0-9]{12,}\b"), "<api-key>"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactingJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for field in _STRUCT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = redact(str(value)) if isinstance(value, str) else value
        if record.exc_info:
            # exception type only - tracebacks can contain SQL text / secrets
            payload["exception"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingJsonFormatter())
    root = logging.getLogger("querysentinel")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"querysentinel.{name}")
