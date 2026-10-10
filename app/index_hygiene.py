"""Index hygiene (Addendum R4) - a SUPPORTING capability.

Detects, when the evidence permits: missing index candidates (FKs without a supporting index),
duplicate/redundant indexes (identical or left-prefix-covered btree definitions) and unused-index
candidates (idx_scan = 0 since the last statistics reset). It is report-only: it never drops
anything and never overrides the primary objective of fixing the active performance problem.
"""
from __future__ import annotations

from typing import Any

from app.contracts import IndexMeta, SchemaMetadata


def _plain_btree(ix: IndexMeta) -> bool:
    return ix.definition_kind == "btree" and not ix.is_partial


def is_redundant(columns: list[str], table: str, schema: SchemaMetadata, include: list[str] | None = None,
                 where: str | None = None) -> tuple[bool, str | None]:
    """Is a proposed btree (table, columns) already served by an existing index?
    Redundant when an existing plain btree has these columns as its leading prefix (and covers
    the INCLUDE columns). Partial candidates are only redundant against an identical partial."""
    include = include or []
    for ix in schema.indexes_for(table):
        if where:
            continue  # we cannot compare partial predicates safely -> never call it redundant
        if not _plain_btree(ix):
            continue
        if ix.columns[:len(columns)] == columns:
            covered = set(ix.columns) | set(ix.include_columns)
            if all(c in covered for c in include):
                return True, ix.name
    return False, None


def analyze_hygiene(schema: SchemaMetadata) -> dict[str, Any]:
    duplicates, redundant, unused, missing_fk = [], [], [], []
    by_table: dict[str, list[IndexMeta]] = {}
    for ix in schema.indexes:
        by_table.setdefault(ix.table, []).append(ix)
    for table, ixs in by_table.items():
        for i, a in enumerate(ixs):
            for b in ixs[i + 1:]:
                if not (_plain_btree(a) and _plain_btree(b)):
                    continue
                if a.columns == b.columns and a.include_columns == b.include_columns:
                    duplicates.append({"table": table, "indexes": [a.name, b.name], "columns": a.columns})
                elif b.columns[:len(a.columns)] == a.columns and not a.is_unique:
                    redundant.append({"table": table, "redundant": a.name, "covered_by": b.name,
                                      "reason": "left prefix of a wider index"})
                elif a.columns[:len(b.columns)] == b.columns and not b.is_unique:
                    redundant.append({"table": table, "redundant": b.name, "covered_by": a.name,
                                      "reason": "left prefix of a wider index"})
            if a.idx_scan == 0 and not a.is_primary and not a.is_unique:
                unused.append({"table": table, "index": a.name, "size_bytes": a.size_bytes,
                               "note": "idx_scan = 0 since statistics reset; confirm over a full business cycle"})
    for fk in schema.foreign_keys:
        if not fk.has_supporting_index:
            t = schema.tables.get(fk.table)
            missing_fk.append({"table": fk.table, "columns": fk.columns, "references": fk.ref_table,
                               "estimated_rows": t.estimated_rows if t else None,
                               "reason": "foreign key without a supporting index (joins / cascades scan the table)"})
    return {"duplicate_indexes": duplicates, "redundant_indexes": redundant,
            "unused_index_candidates": unused, "missing_fk_indexes": missing_fk,
            "policy": "report-only; Planora never drops indexes automatically",
            "evidence_kind": "observed"}
