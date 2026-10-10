"""RL state encoding (Section 21 RL STATE).

The state is built ONLY from anonymised structural features and discretised into a compact
key for tabular Q-learning. No raw values, no identifiers (the query fingerprint is kept as
metadata for workload weighting but is NOT part of the discretised key, which lets the policy
generalise across queries with the same structure).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _bucket_log10(x: float | None, edges=(2, 3, 4, 5, 6, 7)) -> int:
    if not x or x <= 0:
        return 0
    v = math.log10(x)
    return sum(1 for e in edges if v >= e)


@dataclass
class RLState:
    query_fingerprint: str = ""
    bottleneck_class: int = 3                # GNN / fallback class id
    cost_bucket: int = 0                     # log10 planner cost bucket
    cardinality_bucket: int = 0              # log10 estimated rows bucket
    has_seq_scan: int = 0
    has_join: int = 0
    has_sort_agg: int = 0
    has_subplan: int = 0
    n_eq_filters: int = 0                    # capped 0..2
    has_range_filter: int = 0
    has_non_sargable: int = 0
    existing_index_coverage: int = 0         # 0 none, 1 partial prefix, 2 full
    table_size_bucket: int = 0               # 0 small .. 3 xlarge
    write_frequency_bucket: int = 0          # 0 low 1 medium 2 high
    partition_suitable: int = 0
    indexes_added: int = 0                   # within an episode (0..3)
    operator_histogram: dict[str, int] = field(default_factory=dict)

    KEY_FIELDS = ("bottleneck_class", "cost_bucket", "cardinality_bucket", "has_seq_scan", "has_join",
                  "has_sort_agg", "has_subplan", "n_eq_filters", "has_range_filter", "has_non_sargable",
                  "existing_index_coverage", "table_size_bucket", "write_frequency_bucket",
                  "partition_suitable", "indexes_added")

    KEY_CODES = {"bottleneck_class": "bc", "cost_bucket": "cb", "cardinality_bucket": "cr", "has_seq_scan": "ss",
                 "has_join": "jn", "has_sort_agg": "sa", "has_subplan": "sp", "n_eq_filters": "eq",
                 "has_range_filter": "rg", "has_non_sargable": "ns", "existing_index_coverage": "ix",
                 "table_size_bucket": "ts", "write_frequency_bucket": "wf", "partition_suitable": "pt",
                 "indexes_added": "ia"}

    def key(self) -> str:
        """e.g. 'bc0|cb4|cr1|ss1|jn0|...' - unique 2-letter code per discretised feature."""
        return "|".join(f"{self.KEY_CODES[f]}{getattr(self, f)}" for f in self.KEY_FIELDS)

    def coarse_key(self) -> str:
        """Back-off key for unseen states: the features most predictive of action value."""
        return f"b{self.bottleneck_class}|q{self.has_seq_scan}{self.has_subplan}|e{min(self.n_eq_filters,1)}{self.has_range_filter}|c{self.existing_index_coverage}|i{self.indexes_added}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key()
        return d


SIZE_BUCKETS = {"small(<10k)": 0, "medium(10k-1M)": 1, "large(1M-10M)": 2, "xlarge(>10M)": 3}
WRITE_BUCKETS = {"low": 0, "medium": 1, "high": 2}


def encode_state(*, query_fingerprint: str, bottleneck_class: int, plan_total_cost: float, plan_rows: float,
                 operator_histogram: dict[str, int], per_table_access: dict[str, dict[str, list[str]]],
                 target_table: str | None, existing_index_coverage: int, table_size_bucket: str | None,
                 write_frequency_bucket: str | None, partition_suitable: bool, non_sargable: bool,
                 has_subplan: bool, indexes_added: int = 0) -> RLState:
    hist = operator_histogram or {}
    acc = (per_table_access or {}).get(target_table or "", {})
    n_eq = len(acc.get("eq", [])) + len(acc.get("in", []))
    return RLState(
        query_fingerprint=query_fingerprint,
        bottleneck_class=int(bottleneck_class),
        cost_bucket=_bucket_log10(plan_total_cost),
        cardinality_bucket=_bucket_log10(plan_rows, edges=(1, 2, 3, 4, 5, 6)),
        has_seq_scan=int(hist.get("Seq Scan", 0) > 0),
        has_join=int(any(hist.get(k, 0) for k in ("Nested Loop", "Hash Join", "Merge Join"))),
        has_sort_agg=int(any(hist.get(k, 0) for k in ("Sort", "Aggregate", "Incremental Sort"))),
        has_subplan=int(bool(has_subplan)),
        n_eq_filters=min(n_eq, 2),
        has_range_filter=int(bool(acc.get("range"))),
        has_non_sargable=int(bool(non_sargable)),
        existing_index_coverage=int(existing_index_coverage),
        table_size_bucket=SIZE_BUCKETS.get(table_size_bucket or "", 0),
        write_frequency_bucket=WRITE_BUCKETS.get(write_frequency_bucket or "", 0),
        partition_suitable=int(bool(partition_suitable)),
        indexes_added=min(indexes_added, 3),
        operator_histogram=dict(hist),
    )
