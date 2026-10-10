"""Partitioning / sharding advisory layer (Sections 32 and 54).

Evidence used: table size, date-range access patterns, equality access on a tenant-like key,
distribution skew (pg_stats most_common_freqs[1] - a frequency, never a value), cardinality,
query frequency and workload isolation. Output is ALWAYS labelled
"Advisory - Requires isolated design/benchmark validation": HypoPG cannot simulate
partitioning or sharding, and production tables are never partitioned automatically.
"""
from __future__ import annotations

from typing import Any

from app.contracts import PartitionAdvice, SchemaMetadata, ShardingAdvice
from app.predicate_intelligence import AccessPattern
from app.schema_inspector import estimated_distinct

RANGE_MIN_ROWS = 1_000_000
HASH_MIN_ROWS = 1_000_000
SKEW_THRESHOLD = 0.05            # top value holds >= 5% of rows
TENANT_KEY_MAX_DISTINCT = 100_000
LIST_MAX_DISTINCT = 12


class PartitionAdvisor:
    def __init__(self, schema: SchemaMetadata):
        """`schema` = pseudonymised metadata."""
        self.schema = schema

    def advise(self, ap: AccessPattern, workload_weight: float = 1.0) -> dict[str, Any]:
        partition: PartitionAdvice | None = None
        sharding: ShardingAdvice | None = None
        considered: list[dict[str, Any]] = []
        per_table = ap.per_table()
        for table, roles in per_table.items():
            meta = self.schema.tables.get(table)
            if meta is None:
                continue
            rows = meta.estimated_rows
            size_mb = round((meta.table_size_bytes or 0) / 1048576, 1)
            # --- range partitioning on a date/timestamp column used in range predicates
            for col in roles.get("range", []) + [c for c in roles.get("non_sargable", [])]:
                cm = self.schema.column(table, col)
                if cm is None or not cm.data_type.startswith(("timestamp", "date")):
                    continue
                ok = rows >= RANGE_MIN_ROWS
                considered.append({"strategy": "range", "table": table, "key": col, "eligible": ok,
                                   "rows": rows})
                if ok and partition is None:
                    corr = cm.correlation
                    partition = PartitionAdvice(
                        strategy_type="range", table=table, candidate_key=col,
                        rationale=(f"{table} has ~{rows:,.0f} rows ({size_mb} MB) and the query filters {col} by range; "
                                   "monthly range partitions would let the planner prune partitions outside the window."),
                        evidence=[f"estimated rows = {rows:,.0f}", f"table size = {size_mb} MB",
                                  f"range predicate on {col} ({cm.data_type})",
                                  f"physical correlation of {col} = {corr if corr is None else round(corr, 3)}",
                                  f"workload frequency weight = {workload_weight:.2f}"],
                        expected_benefit="partition pruning for date-window queries; cheaper retention (DROP/DETACH old partitions)",
                        operational_risks=["primary key / unique constraints must include the partition key",
                                           "queries without a date predicate scan all partitions",
                                           "migration requires a rebuild or online copy (pg_partman / logical replication)",
                                           "foreign keys referencing the table need review"])
            # --- hash partitioning / sharding on a skewed tenant-like equality key
            for col in roles.get("eq", []) + roles.get("join", []):
                cm = self.schema.column(table, col)
                if cm is None:
                    continue
                nd = estimated_distinct(cm, rows) or 0
                skew = cm.top_value_freq or 0.0
                tenant_like = 1 < nd <= TENANT_KEY_MAX_DISTINCT and cm.data_type in ("uuid", "integer", "bigint", "text")
                if not tenant_like or rows < HASH_MIN_ROWS:
                    continue
                considered.append({"strategy": "hash/shard", "table": table, "key": col, "n_distinct": nd,
                                   "top_value_freq": skew})
                if partition is None:
                    partition = PartitionAdvice(
                        strategy_type="hash", table=table, candidate_key=col,
                        rationale=(f"{table} (~{rows:,.0f} rows) is accessed by equality on {col} "
                                   f"(~{nd:,.0f} distinct values); hash partitions isolate tenants' data."),
                        evidence=[f"estimated rows = {rows:,.0f}", f"n_distinct({col}) ~ {nd:,.0f}",
                                  f"top value frequency = {skew:.1%}"],
                        expected_benefit="smaller per-partition indexes and vacuum work; tenant locality",
                        operational_risks=["hash partitioning does not prune range queries",
                                           "hot tenants still concentrate in one partition (skew)",
                                           "constraints must include the partition key"])
                if sharding is None:
                    sharding = ShardingAdvice(
                        candidate_key=col,
                        rationale=(f"Equality access on {col} with skewed distribution (top value holds {skew:.1%} "
                                   "of rows) suggests tenant-based distribution, isolating hot tenants."),
                        evidence=[f"n_distinct({col}) ~ {nd:,.0f}", f"top value frequency = {skew:.1%}",
                                  f"table rows = {rows:,.0f}"],
                        expected_benefit="workload isolation for hot tenants; horizontal scale-out of tenant data",
                        operational_risks=["cross-tenant queries become distributed queries",
                                           "rebalancing hot tenants requires tooling (e.g. Citus tenant isolation)",
                                           "application must route by the distribution key"])
            # --- list partitioning on a very low-cardinality equality column (where appropriate)
            for col in roles.get("eq", []):
                cm = self.schema.column(table, col)
                nd = estimated_distinct(cm, rows) if cm else None
                if cm and nd and 1 < nd <= LIST_MAX_DISTINCT and rows >= RANGE_MIN_ROWS and partition is None:
                    partition = PartitionAdvice(
                        strategy_type="list", table=table, candidate_key=col,
                        rationale=f"{col} has only ~{nd:.0f} distinct values and is always filtered by equality.",
                        evidence=[f"n_distinct({col}) ~ {nd:.0f}", f"estimated rows = {rows:,.0f}"],
                        expected_benefit="pruning to the partitions of the requested value(s)",
                        operational_risks=["value changes move rows between partitions (UPDATE cost)",
                                           "uneven partition sizes when one value dominates"])
        return {"partitioning": partition, "sharding": sharding, "considered": considered,
                "label": "Advisory - Requires isolated design/benchmark validation"}
