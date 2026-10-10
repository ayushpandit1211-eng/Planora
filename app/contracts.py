"""Cross-module contracts (Execution Controller section I).

Every structure that crosses a module boundary (API, dashboard, agents, tests) is defined
here once. Field names are mirrored in .build_state/CONTRACTS.md and docs/api.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Result provenance (Section 4.3 / R11)
# ---------------------------------------------------------------------------
class EvidenceKind(str, Enum):
    OBSERVED = "observed"                 # FACT: directly observed PG metadata / plan
    INFERRED = "inferred"                 # rule / analytic inference
    SIMULATED = "simulated"               # HypoPG hypothetical / planner simulation
    ESTIMATED = "estimated"               # heuristic estimate (storage/write impact)
    BENCHMARK_MEASURED = "benchmark-measured"  # real execution timing


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
class PrivacyAudit(BaseModel):
    pii_detected: bool = False
    pii_types: list[str] = Field(default_factory=list)
    literal_leak_detected: bool = False
    secret_leak_detected: bool = False
    safe_for_llm: bool = True


class PrivacyResult(BaseModel):
    """Return contract of DataObfuscator.mask_sql (Section 14)."""
    masked_sql: str
    ast_signature: str
    literal_count: int = 0
    mapped_tables: dict[str, str] = Field(default_factory=dict)   # pseudonym -> kind marker only (no originals)
    mapped_columns: dict[str, str] = Field(default_factory=dict)
    fallback_used: bool = False
    quarantined: bool = False
    quarantine_reason: str | None = None
    privacy_bitmask: int = 0
    privacy_flags: dict[str, bool] = Field(default_factory=dict)
    privacy_audit: PrivacyAudit = Field(default_factory=PrivacyAudit)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class ColumnMeta(BaseModel):
    name: str
    data_type: str
    nullable: bool
    n_distinct: float | None = None
    null_frac: float | None = None
    correlation: float | None = None
    top_value_freq: float | None = None    # pg_stats.most_common_freqs[1]: a FREQUENCY, never the value


class IndexMeta(BaseModel):
    name: str
    table: str
    columns: list[str]
    is_unique: bool = False
    is_primary: bool = False
    is_partial: bool = False
    include_columns: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    idx_scan: int | None = None
    definition_kind: str = "btree"


class ForeignKeyMeta(BaseModel):
    table: str
    columns: list[str]
    ref_table: str
    ref_columns: list[str]
    has_supporting_index: bool = False


class TableMeta(BaseModel):
    schema_name: str = "public"
    name: str
    columns: list[ColumnMeta]
    primary_key: list[str] = Field(default_factory=list)
    estimated_rows: float = 0
    table_size_bytes: int | None = None
    total_size_bytes: int | None = None
    n_tup_ins: int | None = None
    n_tup_upd: int | None = None
    n_tup_del: int | None = None
    n_tup_hot_upd: int | None = None
    seq_scan: int | None = None
    idx_scan: int | None = None
    last_analyze: str | None = None


class SchemaMetadata(BaseModel):
    tables: dict[str, TableMeta] = Field(default_factory=dict)
    indexes: list[IndexMeta] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyMeta] = Field(default_factory=list)

    def indexes_for(self, table: str) -> list[IndexMeta]:
        return [i for i in self.indexes if i.table == table]

    def has_column(self, table: str, column: str) -> bool:
        t = self.tables.get(table)
        return bool(t and any(c.name == column for c in t.columns))

    def column(self, table: str, column: str) -> ColumnMeta | None:
        t = self.tables.get(table)
        if not t:
            return None
        return next((c for c in t.columns if c.name == column), None)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
class PlanNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_id: int
    parent_id: int | None = None
    depth: int = 0
    node_type: str
    parent_relationship: str | None = None
    startup_cost: float = 0.0
    total_cost: float = 0.0
    plan_rows: float = 0.0
    plan_width: int = 0
    exclusive_cost: float = 0.0
    estimated_loops: float = 1.0          # planner-implied executions (SubPlan / NL inner side)
    effective_cost: float = 0.0           # exclusive_cost * estimated_loops
    relation_name: str | None = None
    alias: str | None = None
    index_name: str | None = None
    filter: str | None = None
    index_cond: str | None = None
    recheck_cond: str | None = None
    join_type: str | None = None
    hash_cond: str | None = None
    merge_cond: str | None = None
    join_filter: str | None = None
    sort_key: list[str] = Field(default_factory=list)
    group_key: list[str] = Field(default_factory=list)
    strategy: str | None = None
    shared_hit_blocks: int | None = None
    shared_read_blocks: int | None = None
    actual_rows: float | None = None
    actual_loops: float | None = None
    actual_total_time: float | None = None
    subplan_name: str | None = None
    parallel_aware: bool = False


class PlanAnalysis(BaseModel):
    plan_fingerprint: str
    plan_shape: str
    total_cost: float
    startup_cost: float
    plan_rows: float
    node_count: int
    nodes: list[PlanNode]
    edges: list[tuple[int, int]]
    operator_histogram: dict[str, int]
    highest_cost_node: dict[str, Any] | None = None
    seq_scans: list[dict[str, Any]] = Field(default_factory=list)
    has_actuals: bool = False
    sanitized: bool = True
    planning_shared_hit_blocks: int | None = None


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------
BOTTLENECK_CLASSES = {
    0: "Unindexed Sequential Scan",
    1: "High-Cost Nested Loop / Join",
    2: "Heavy Sort / Aggregation",
    3: "Efficient / No Major Bottleneck",
}


class GNNDiagnosis(BaseModel):
    primary_bottleneck_type: str
    bottleneck_class_id: int
    confidence: float | str | None = None   # NN probability (trained) or rule label (fallback)
    class_probabilities: dict[str, float] | None = None
    highest_cost_node: dict[str, Any] = Field(default_factory=dict)
    natural_language_summary: str
    graph_nodes: list[dict[str, Any]] = Field(default_factory=list)
    graph_edges: list[list[int]] = Field(default_factory=list)
    model_mode: Literal["trained", "analytic_fallback"]
    evidence: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    attention: dict[str, Any] | None = None      # Final Upgrade P8 (only when QS_FEATURE_GATV2 is on)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
class IndexCandidate(BaseModel):
    """Index candidate expressed in PSEUDONYMIZED identifiers."""
    table: str
    columns: list[str]
    kind: Literal["single", "composite", "partial", "covering"] = "single"
    include_columns: list[str] = Field(default_factory=list)
    where_clause: str | None = None          # masked predicate (params $n)
    ddl: str = ""
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: str = "index_strategist"
    signature: str = ""                      # abstract RL signature (no identifiers)
    roles: list[str] = Field(default_factory=list)  # eq/range/join/order/group per column


class PartitionAdvice(BaseModel):
    strategy_type: Literal["range", "hash", "list"]
    table: str
    candidate_key: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    expected_benefit: str
    operational_risks: list[str] = Field(default_factory=list)
    validation_required: str = "Advisory - Requires isolated design/benchmark validation"
    validated: bool = False
    label: str = "Advisory - Requires isolated design/benchmark validation"


class ShardingAdvice(BaseModel):
    strategy_type: str = "tenant_sharding"
    candidate_key: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    expected_benefit: str
    operational_risks: list[str] = Field(default_factory=list)
    validation_required: str = "Advisory - Requires isolated design/benchmark validation"
    validated: bool = False
    label: str = "Advisory - Requires isolated design/benchmark validation"


class ProposedIndexLLM(BaseModel):
    ddl: str
    reason: str = ""
    columns: list[str] = Field(default_factory=list)


class LLMRecommendation(BaseModel):
    """Section 28 - LLM output schema; validated before any use."""
    model_config = ConfigDict(extra="forbid")
    rewritten_sql: str | None = None
    proposed_indexes: list[ProposedIndexLLM] = Field(default_factory=list)
    partitioning_recommendation: dict[str, Any] | None = None
    sharding_recommendation: dict[str, Any] | None = None
    technical_reasoning: str = ""
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be within [0,1]")
        return v


class RewriteProposal(BaseModel):
    rewritten_masked_sql: str
    rules_applied: list[str]
    reasoning: str
    semantic_diff: dict[str, Any] = Field(default_factory=dict)
    semantics_verified: bool = False
    requires_human_review: bool = True
    source: str = "deterministic_rules"


class AgentRecommendation(BaseModel):
    rewritten_sql: str | None = None
    rewrite: RewriteProposal | None = None
    proposed_indexes: list[IndexCandidate] = Field(default_factory=list)
    partitioning_recommendation: PartitionAdvice | None = None
    sharding_recommendation: ShardingAdvice | None = None
    extended_statistics: list[dict[str, Any]] = Field(default_factory=list)
    technical_reasoning: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float | str | None = None
    llm: dict[str, Any] = Field(default_factory=dict)


class RLRecommendation(BaseModel):
    policy_mode: Literal["trained_policy", "analytic_fallback"]
    state_key: str
    high_level_action: str
    low_level_action: str | None = None
    selected_candidate: dict[str, Any] | None = None
    q_values: dict[str, float] = Field(default_factory=dict)
    valid_actions: list[str] = Field(default_factory=list)
    masked_actions: dict[str, str] = Field(default_factory=dict)
    explanation: str = ""
    retraining_recommended: bool = False


class ValidationOutcome(BaseModel):
    candidate: IndexCandidate
    valid: bool
    real_ddl: str | None = None   # trusted-side only; never sent to LLM
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    before_total_cost: float | None
    after_total_cost: float | None
    planner_cost_reduction_pct: float | None
    virtual_indexes_used: list[str] = Field(default_factory=list)
    virtual_indexes_created: list[dict[str, Any]] = Field(default_factory=list)
    estimated_storage_impact_mb: float | None = None
    estimated_write_overhead: dict[str, Any] | None = None
    baseline_plan_fingerprint: str | None = None
    optimized_plan_fingerprint: str | None = None
    plan_changed: bool = False
    rewrite_before_cost: float | None = None
    rewrite_after_cost: float | None = None
    rewrite_cost_reduction_pct: float | None = None
    rewrite_with_indexes_cost: float | None = None
    combination_note: str | None = None
    per_candidate: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["completed", "skipped", "failed", "unavailable"] = "completed"
    cleanup_ok: bool = True
    error: str | None = None
    metric_label: str = "PLANNER-ESTIMATED COST REDUCTION (not a runtime speedup)"
    evidence_kind: EvidenceKind = EvidenceKind.SIMULATED


class BlastRadiusResult(BaseModel):
    target_table: str
    write_queries_analyzed: list[dict[str, Any]] = Field(default_factory=list)
    write_frequency_summary: dict[str, Any] = Field(default_factory=dict)
    estimated_write_impact: dict[str, Any] = Field(default_factory=dict)
    estimated_storage_impact_mb: float | None = None
    risk_level: Literal["low", "medium", "high"] = "low"
    methodology: str = ""
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class DeploymentPlan(BaseModel):
    migration_id: str | None = None
    migration_sql: str = ""
    rollback_sql: str = ""
    requires_human_review: bool = True
    generated_at: str = Field(default_factory=utcnow_iso)
    rationale: str = ""
    query_fingerprint: str | None = None
    approval_status: str = "PENDING_REVIEW"
    advisory_notes: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    step: Literal["OBSERVATION", "EVIDENCE", "INTERPRETATION", "RECOMMENDATION",
                  "SANDBOX_RESULT", "BLAST_RADIUS_RESULT", "DEPLOYMENT_RISK"]
    source_type: str
    kind: EvidenceKind
    statement: str
    data: dict[str, Any] = Field(default_factory=dict)
    # Final Upgrade P11 (additive): machine-readable type, provenance tags and source artifact reference
    evidence_type: str | None = None
    provenance: list[str] = Field(default_factory=list)
    artifact_id: str | None = None


class AuditEvent(BaseModel):
    request_id: str
    event_type: str
    stage: str | None = None
    severity: Literal["info", "warning", "error", "security"] = "info"
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow_iso)


class StreamEvent(BaseModel):
    event: str
    timestamp: str = Field(default_factory=utcnow_iso)
    request_id: str
    stage: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class FinalOptimizationReport(BaseModel):
    """Section 67 report contract (+R9 budget, R11 evidence chain, model modes)."""
    request_id: str
    query_fingerprint: str
    privacy_audit: dict[str, Any]
    diagnosis: dict[str, Any]
    optimization: dict[str, Any]
    validation: dict[str, Any]
    blast_radius: dict[str, Any] | list[dict[str, Any]] | None
    deployment: dict[str, Any]
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)
    model_modes: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    workload_context: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    structural_simulation: dict[str, Any] = Field(default_factory=dict)
    zero_exposure_proof: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=utcnow_iso)
