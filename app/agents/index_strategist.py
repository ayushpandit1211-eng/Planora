"""Index Strategist agent (Section 25 + R3).

Works ONLY in pseudonymised identifier space: the masked SQL's access pattern and the
pseudonymised schema metadata. Generates bounded single / composite / partial / covering
index candidates from predicate, join, ordering and grouping evidence, ordered by
selectivity (pg_stats n_distinct). Never invents identifiers: every column is checked against
the pseudonymised schema. Redundant candidates are kept but flagged (RL masking removes them).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.contracts import IndexCandidate, SchemaMetadata
from app.index_hygiene import is_redundant
from app.predicate_intelligence import AccessPattern
from app.schema_inspector import estimated_distinct

MAX_KEY_COLUMNS = 3
MAX_INCLUDE_COLUMNS = 3
LOW_CARDINALITY = 20          # distinct values at/below which a column suits a partial-index predicate
MIN_TABLE_ROWS = 10_000       # below this an index rarely changes the plan


@dataclass
class _Cols:
    eq: list[str]
    rng: list[str]
    join: list[str]
    order: list[tuple[str, str]]
    group: list[str]
    select: list[str]
    eq_params: dict[str, str]


def _ddl(c: IndexCandidate) -> str:
    inc = f" INCLUDE ({', '.join(c.include_columns)})" if c.include_columns else ""
    where = f" WHERE {c.where_clause}" if c.where_clause else ""
    return f"CREATE INDEX ON {c.table} ({', '.join(c.columns)}){inc}{where}"


def _signature(kind: str, roles: list[str], include: bool, partial: bool) -> str:
    sig = f"{kind}:{'+'.join(roles)}"
    if include:
        sig += "+inc"
    if partial:
        sig += "|partial"
    return sig


class IndexStrategist:
    def __init__(self, schema: SchemaMetadata):
        """`schema` must be the PSEUDONYMISED metadata (schema_inspector.pseudonymize_metadata)."""
        self.schema = schema

    def _distinct(self, table: str, col: str) -> float:
        t = self.schema.tables[table]
        nd = estimated_distinct(self.schema.column(table, col), t.estimated_rows)
        return nd if nd is not None else 0.0

    def _collect(self, ap: AccessPattern, table: str) -> _Cols:
        eq, rng, join, order, group, select = [], [], [], [], [], []
        for u in ap.uses:
            if u.table != table or not self.schema.has_column(table, u.column):
                continue
            target = {"eq": eq, "in": eq, "range": rng, "join": join, "group": group,
                      "select": select, "in_subquery": join}.get(u.role)
            if u.role in ("eq", "in", "range") and u.in_or:
                continue   # OR-ed predicates cannot be used as index prefixes safely
            if target is not None and u.column not in target:
                target.append(u.column)
            if u.role == "order" and u.column not in [o[0] for o in order]:
                order.append((u.column, u.direction or "asc"))
        return _Cols(eq, rng, join, order, group, select, ap.eq_params(table))

    def propose(self, ap: AccessPattern, scanned_tables: dict[str, dict[str, Any]] | None = None,
                max_candidates: int = 8) -> list[IndexCandidate]:
        """`scanned_tables`: pseudonymised table -> plan evidence (e.g. seq-scan cost share)."""
        scanned_tables = scanned_tables or {}
        out: list[IndexCandidate] = []
        seen: set[tuple] = set()

        def add(table: str, cols: list[str], roles: list[str], kind: str, reason: str, evidence: list[str],
                include: list[str] | None = None, where: str | None = None) -> None:
            cols = cols[:MAX_KEY_COLUMNS]
            roles = roles[:len(cols)]
            include = [c for c in (include or []) if c not in cols][:MAX_INCLUDE_COLUMNS]
            if not cols or any(not self.schema.has_column(table, c) for c in cols + include):
                return
            key = (table, tuple(cols), tuple(include), where)
            if key in seen:
                return
            seen.add(key)
            red, by = is_redundant(cols, table, self.schema, include, where)
            cand = IndexCandidate(table=table, columns=cols, kind=kind, include_columns=include,
                                  where_clause=where, reason=reason,
                                  evidence=evidence + ([f"already served by existing index {by}"] if red else []),
                                  roles=roles, signature=_signature(kind, roles, bool(include), bool(where)))
            cand.ddl = _ddl(cand)
            if red:
                cand.source = "index_strategist:redundant"
            out.append(cand)

        for table in ap.tables:
            if table not in self.schema.tables:
                continue
            meta = self.schema.tables[table]
            c = self._collect(ap, table)
            plan_ev = scanned_tables.get(table, {})
            ev_base = [f"table rows (estimate) = {meta.estimated_rows:.0f}"]
            if plan_ev:
                ev_base.append(f"plan: {plan_ev.get('node_type', 'scan')} cost share "
                               f"{(plan_ev.get('cost_share') or 0):.0%}"
                               + (f", executed ~{plan_ev['loops']:.0f}x" if plan_ev.get("loops", 1) > 1 else ""))
            if meta.estimated_rows < MIN_TABLE_ROWS and not plan_ev:
                continue
            eq_sorted = sorted(c.eq, key=lambda col: -self._distinct(table, col))
            ev_sel = [f"n_distinct({col}) ~ {self._distinct(table, col):.0f}" for col in eq_sorted]

            # 1. join columns of a scanned inner table (FK / correlated subquery access path)
            for jc in c.join:
                if plan_ev or not any(i.columns[:1] == [jc] for i in self.schema.indexes_for(table)):
                    add(table, [jc], ["join"], "single", "join/correlation column without a leading index",
                        ev_base + [f"{jc} used as join key"])
            # 2. composite: equality columns by selectivity, then range column (or order column)
            if eq_sorted:
                tail_roles: list[str] = []
                tail: list[str] = []
                if c.rng:
                    tail, tail_roles = [c.rng[0]], ["range"]
                elif c.order:
                    tail, tail_roles = [c.order[0][0]], ["order"]
                keys = eq_sorted[:2] + [t for t in tail if t not in eq_sorted[:2]]
                roles = ["eq"] * len(eq_sorted[:2]) + tail_roles[:len(keys) - len(eq_sorted[:2])]
                if len(keys) >= 2:
                    add(table, keys, roles, "composite",
                        "equality predicate column(s) first (most selective first), then range/order column",
                        ev_base + ev_sel + [f"range/order column {tail[0]}" if tail else ""])
                    # covering variant when the query reads few additional columns
                    extra = [s for s in c.select + c.group if s not in keys]
                    if 0 < len(extra) <= MAX_INCLUDE_COLUMNS:
                        add(table, keys, roles, "covering", "composite + INCLUDE to allow index-only access",
                            ev_base + [f"query reads {len(extra)} additional column(s)"], include=extra)
                add(table, [eq_sorted[0]], ["eq"], "single", "most selective equality predicate column",
                    ev_base + ev_sel[:1])
            # 3. equality + join for the inner side of a join (e.g. filter then join back)
            if c.eq and c.join:
                add(table, [eq_sorted[0], c.join[0]], ["eq", "join"], "composite",
                    "equality filter then join key (inner side of join)", ev_base + ev_sel[:1])
            # 4. range-only access
            if c.rng and not c.eq:
                add(table, [c.rng[0]], ["range"], "single", "range predicate column", ev_base)
            # 5. partial index: low-cardinality equality column used as a predicate
            low = [col for col in c.eq if 0 < self._distinct(table, col) <= LOW_CARDINALITY and col in c.eq_params]
            key_cols = [k for k in (c.rng[:1] or [o[0] for o in c.order[:1]] or [j for j in c.join[:1]]) if k not in low]
            if low and key_cols:
                add(table, key_cols, ["range" if c.rng else "order" if c.order else "join"], "partial",
                    "low-cardinality predicate moved into the index WHERE clause",
                    ev_base + [f"{low[0]} has ~{self._distinct(table, low[0]):.0f} distinct values"],
                    where=f"{low[0]} = {c.eq_params[low[0]]}")
            # 6. group-by support
            if c.group and not c.eq and not c.rng:
                add(table, c.group[:2], ["group"] * len(c.group[:2]), "composite" if len(c.group) > 1 else "single",
                    "grouping columns", ev_base)

        def rank(cand: IndexCandidate) -> tuple:
            plan_bonus = 0 if cand.table in scanned_tables else 1
            kind_order = {"composite": 0, "single": 1, "covering": 2, "partial": 3}[cand.kind]
            return (cand.source.endswith("redundant"), plan_bonus, kind_order, -len(cand.columns))

        out.sort(key=rank)
        return out[:max_candidates]
