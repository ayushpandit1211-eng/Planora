"""Schema Expert agent (Section 24).

Analyses existing indexes, missing access paths, estimated row counts, key relationships,
statistics freshness, table/index size and partitioning suitability. Returns evidence-backed
reasoning derived ONLY from pseudonymised metadata.
"""
from __future__ import annotations

from typing import Any

from app.cardinality_analyzer import analyze_cardinality
from app.contracts import PlanAnalysis, SchemaMetadata
from app.index_hygiene import analyze_hygiene
from app.partition_advisor import PartitionAdvisor
from app.predicate_intelligence import AccessPattern


class SchemaExpert:
    def __init__(self, schema: SchemaMetadata):
        self.schema = schema   # pseudonymised

    def analyze(self, ap: AccessPattern, plan: PlanAnalysis | None = None,
                workload_weight: float = 1.0) -> dict[str, Any]:
        findings: list[str] = []
        per_table = ap.per_table()
        missing_paths = []
        tables_out = {}
        for table in ap.tables:
            meta = self.schema.tables.get(table)
            if meta is None:
                continue
            idx = self.schema.indexes_for(table)
            leading = {i.columns[0] for i in idx if i.columns}
            roles = per_table.get(table, {})
            for role in ("eq", "range", "join", "in"):
                for col in roles.get(role, []):
                    if col not in leading:
                        missing_paths.append({"table": table, "column": col, "role": role,
                                              "reason": "no index has this column as its leading key"})
            tables_out[table] = {
                "estimated_rows": meta.estimated_rows,
                "table_size_mb": round((meta.table_size_bytes or 0) / 1048576, 1),
                "index_count": len(idx),
                "indexes": [{"name": i.name, "columns": i.columns, "size_mb": round((i.size_bytes or 0) / 1048576, 2),
                             "idx_scan": i.idx_scan} for i in idx],
                "writes_since_stats_reset": (meta.n_tup_ins or 0) + (meta.n_tup_upd or 0) + (meta.n_tup_del or 0),
                "hot_update_ratio": round((meta.n_tup_hot_upd or 0) / meta.n_tup_upd, 3) if meta.n_tup_upd else None,
                "last_analyze": meta.last_analyze,
            }
            if meta.last_analyze is None:
                findings.append(f"{table}: no ANALYZE recorded - planner statistics may be stale")
        fk_rel = [fk.model_dump() for fk in self.schema.foreign_keys if fk.table in ap.tables or fk.ref_table in ap.tables]
        for fk in fk_rel:
            if not fk["has_supporting_index"]:
                findings.append(f"foreign key {fk['table']}({', '.join(fk['columns'])}) -> {fk['ref_table']} has no supporting index")
        for mp in missing_paths:
            findings.append(f"{mp['table']}.{mp['column']} ({mp['role']}) has no leading index")
        advisory = PartitionAdvisor(self.schema).advise(ap, workload_weight)
        card = analyze_cardinality(plan, {t: r.get("eq", []) + r.get("range", []) for t, r in per_table.items()}) \
            if plan is not None else {"available": False}
        hygiene = analyze_hygiene(self.schema)
        return {
            "tables": tables_out,
            "key_relationships": fk_rel,
            "missing_access_paths": missing_paths,
            "partitioning_recommendation": advisory["partitioning"].model_dump() if advisory["partitioning"] else None,
            "sharding_recommendation": advisory["sharding"].model_dump() if advisory["sharding"] else None,
            "partitioning_considered": advisory["considered"],
            "cardinality_analysis": card,
            "extended_statistics": card.get("extended_statistics_candidates", []) if card.get("available") else [],
            "index_hygiene": hygiene,
            "findings": findings,
            "evidence_kind": "observed",
        }
