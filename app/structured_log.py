"""Structured logging helper for the upgrade paths: every record carries trace_id and data_source_id.

Never pass secrets, uploaded file contents or result rows as fields (the formatter also redacts).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.logging_setup import get_logger


def new_trace_id() -> str:
    return f"tr-{uuid.uuid4().hex[:16]}"


class TraceLogger:
    def __init__(self, name: str, trace_id: str | None = None, data_source_id: str | None = None):
        self._log = get_logger(name)
        self.trace_id = trace_id or new_trace_id()
        self.data_source_id = data_source_id or "demo"

    def _emit(self, level: int, msg: str, **fields: Any) -> None:
        self._log.log(level, msg, extra={"trace_id": self.trace_id, "data_source_id": self.data_source_id,
                                         **{k: v for k, v in fields.items() if k in
                                            ("stage", "duration_ms", "error_category", "event", "request_id")}})

    def info(self, msg: str, **f: Any) -> None:
        self._emit(logging.INFO, msg, **f)

    def warning(self, msg: str, **f: Any) -> None:
        self._emit(logging.WARNING, msg, **f)

    def error(self, msg: str, **f: Any) -> None:
        self._emit(logging.ERROR, msg, **f)
