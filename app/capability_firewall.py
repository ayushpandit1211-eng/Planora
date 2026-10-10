"""Agent capability firewall (Addendum R10).

Every agent/tool operation maps to an explicit capability. No agent receives a production-write
capability. Denied operations raise CapabilityDenied and produce an audit event. The same
boundary is enforced in PostgreSQL: the application role `qs_app` has no INSERT/UPDATE/DELETE/DDL
privilege on business tables (scripts/init_db.sql), so even an allowed code path cannot write.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from app.audit import AuditLogger, audit_logger
from app.errors import CapabilityDenied


class Capability(str, Enum):
    READ_SANITIZED_METADATA = "READ_SANITIZED_METADATA"
    READ_SCHEMA_METADATA = "READ_SCHEMA_METADATA"
    READ_PLAN = "READ_PLAN"
    READ_SANITIZED_STATS = "READ_SANITIZED_STATS"
    PROPOSE_SQL = "PROPOSE_SQL"
    PROPOSE_SCHEMA_ADVISORY = "PROPOSE_SCHEMA_ADVISORY"
    PROPOSE_INDEX = "PROPOSE_INDEX"
    RUN_HYPOPG = "RUN_HYPOPG"
    GENERATE_REPORT = "GENERATE_REPORT"
    GENERATE_MIGRATION = "GENERATE_MIGRATION"
    CALL_LLM = "CALL_LLM"
    # Defined so that it can be explicitly REFUSED - never granted to any agent.
    PRODUCTION_WRITE = "PRODUCTION_WRITE"


AGENT_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "sql_rewriter": frozenset({Capability.READ_SANITIZED_METADATA, Capability.PROPOSE_SQL, Capability.CALL_LLM}),
    "schema_expert": frozenset({Capability.READ_SCHEMA_METADATA, Capability.PROPOSE_SCHEMA_ADVISORY,
                                Capability.CALL_LLM}),
    "index_strategist": frozenset({Capability.READ_SCHEMA_METADATA, Capability.PROPOSE_INDEX, Capability.CALL_LLM}),
    "supervisor": frozenset({Capability.READ_PLAN, Capability.RUN_HYPOPG, Capability.READ_SANITIZED_STATS,
                             Capability.GENERATE_REPORT}),
    "deployment_generator": frozenset({Capability.GENERATE_MIGRATION}),
    "rl_optimizer": frozenset({Capability.READ_SANITIZED_STATS}),
    "nl_assistant": frozenset({Capability.READ_SANITIZED_METADATA, Capability.CALL_LLM}),
}

assert all(Capability.PRODUCTION_WRITE not in caps for caps in AGENT_CAPABILITIES.values())


class CapabilityFirewall:
    def __init__(self, audit: AuditLogger | None = None):
        self.audit = audit or audit_logger
        self.decisions: list[dict] = []

    def allowed(self, agent: str, capability: Capability) -> bool:
        return capability in AGENT_CAPABILITIES.get(agent, frozenset())

    def require(self, request_id: str, agent: str, capability: Capability, operation: str = "") -> None:
        ok = self.allowed(agent, capability)
        self.decisions.append({"agent": agent, "capability": capability.value, "operation": operation,
                               "allowed": ok})
        if not ok:
            self.audit.record(request_id, "CAPABILITY_DENIED", stage=agent, severity="security",
                              details={"agent": agent, "capability": capability.value, "operation": operation})
            raise CapabilityDenied(f"agent '{agent}' lacks capability {capability.value}", stage=agent,
                                   details={"agent": agent, "capability": capability.value})

    def guard(self, request_id: str, agent: str, capability: Capability, operation: str = ""):
        """Decorator-style helper: firewall.guard(...)(fn)(*args)."""
        def deco(fn: Callable):
            def wrapper(*a, **kw):
                self.require(request_id, agent, capability, operation or fn.__name__)
                return fn(*a, **kw)
            return wrapper
        return deco
