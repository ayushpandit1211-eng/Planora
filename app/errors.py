"""Structured error taxonomy (Section 55). Messages never include raw SQL, literals or secrets."""
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    INVALID_SQL = "invalid_sql"
    UNSAFE_SQL = "unsafe_sql"
    PII_LEAK = "pii_leak"
    SECRET_LEAK = "secret_leak"
    AST_FAILURE = "ast_failure"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    LLM_TIMEOUT = "llm_timeout"
    LLM_MALFORMED = "llm_malformed_json"
    LLM_UNAVAILABLE = "llm_unavailable"
    HYPOPG_UNAVAILABLE = "hypopg_unavailable"
    PG_STAT_UNAVAILABLE = "pg_stat_statements_unavailable"
    GNN_UNAVAILABLE = "gnn_model_unavailable"
    CORRUPT_CHECKPOINT = "corrupted_model_checkpoint"
    RL_UNAVAILABLE = "rl_model_unavailable"
    DB_UNAVAILABLE = "database_unavailable"
    INVALID_DDL = "invalid_ddl"
    SANDBOX_FAILURE = "sandbox_failure"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    SSE_FAILURE = "sse_failure"
    CAPABILITY_DENIED = "capability_denied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    QUARANTINED = "quarantined"
    NOT_FOUND = "not_found"
    INTERNAL = "internal_error"
    # Final upgrade (machine-readable codes, Section P13)
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_FORBIDDEN = "source_forbidden"
    SANDBOX_EXPIRED = "sandbox_expired"
    QUOTA_EXCEEDED = "quota_exceeded"
    UPLOAD_REJECTED = "upload_rejected"
    SCHEMA_CONFLICT = "schema_conflict"
    SQL_REJECTED = "sql_rejected"
    PRIVACY_BUDGET_EXHAUSTED = "privacy_budget_exhausted"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    FEATURE_DISABLED = "feature_disabled"


HTTP_STATUS = {
    ErrorCategory.INVALID_SQL: 400,
    ErrorCategory.UNSAFE_SQL: 400,
    ErrorCategory.PII_LEAK: 422,
    ErrorCategory.SECRET_LEAK: 422,
    ErrorCategory.AST_FAILURE: 422,
    ErrorCategory.QUARANTINED: 422,
    ErrorCategory.SCHEMA_MISMATCH: 422,
    ErrorCategory.UNKNOWN_IDENTIFIER: 422,
    ErrorCategory.INVALID_DDL: 422,
    ErrorCategory.CAPABILITY_DENIED: 403,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.DB_UNAVAILABLE: 503,
    ErrorCategory.SANDBOX_UNAVAILABLE: 503,
    ErrorCategory.HYPOPG_UNAVAILABLE: 503,
    ErrorCategory.PG_STAT_UNAVAILABLE: 503,
    ErrorCategory.LLM_TIMEOUT: 504,
    ErrorCategory.SOURCE_NOT_FOUND: 404,
    ErrorCategory.SOURCE_FORBIDDEN: 403,
    ErrorCategory.SANDBOX_EXPIRED: 410,
    ErrorCategory.QUOTA_EXCEEDED: 429,
    ErrorCategory.UPLOAD_REJECTED: 422,
    ErrorCategory.SCHEMA_CONFLICT: 409,
    ErrorCategory.SQL_REJECTED: 422,
    ErrorCategory.PRIVACY_BUDGET_EXHAUSTED: 429,
    ErrorCategory.SEARCH_BUDGET_EXHAUSTED: 429,
    ErrorCategory.FEATURE_DISABLED: 404,
}


class QuerySentinelError(Exception):
    """Base error. `message` must be safe to show to users and to log."""

    def __init__(self, category: ErrorCategory, message: str, *, stage: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.stage = stage
        self.details = details or {}

    @property
    def http_status(self) -> int:
        return HTTP_STATUS.get(self.category, 500)

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "error": self.category.value,
            "message": self.message,
            "stage": self.stage,
            "details": self.details,
        }
        if request_id:
            out["request_id"] = request_id
        return out


class SecurityRejection(QuerySentinelError):
    pass


class PrivacyViolation(QuerySentinelError):
    pass


class CapabilityDenied(QuerySentinelError):
    def __init__(self, message: str, **kw):
        super().__init__(ErrorCategory.CAPABILITY_DENIED, message, **kw)


class BudgetExhausted(QuerySentinelError):
    def __init__(self, message: str, **kw):
        super().__init__(ErrorCategory.BUDGET_EXHAUSTED, message, **kw)
