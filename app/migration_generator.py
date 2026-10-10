"""Staging migration + rollback generator (Sections 50-51) and the human approval boundary (4.2).

Generates `CREATE INDEX CONCURRENTLY` migrations and `DROP INDEX CONCURRENTLY` rollbacks with
deterministic index naming, safe identifier quoting, comments, source query fingerprint,
rationale, migration identifier and timestamp. Scripts are for STAGING REVIEW. Planora
never executes them: there is no code path that runs a migration, and the database role has
no DDL privilege on business tables.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from app.contracts import DeploymentPlan, PartitionAdvice, ShardingAdvice
from app.recommendation_mapper import quote_ident, real_index_parts

MAX_IDENT = 63


def deterministic_index_name(table: str, columns: list[str], include: list[str] | None = None,
                             where: str | None = None) -> str:
    sig = f"{table}|{','.join(columns)}|{','.join(include or [])}|{where or ''}"
    h = hashlib.sha256(sig.encode()).hexdigest()[:8]
    base = f"qs_idx_{table}_{'_'.join(columns)}"
    room = MAX_IDENT - len(h) - 1
    return f"{base[:room]}_{h}"


def migration_id(fingerprint: str, ddls: list[str], ts: datetime) -> str:
    h = hashlib.sha256((fingerprint + "|".join(ddls)).encode()).hexdigest()[:10]
    return f"QSM-{ts:%Y%m%d%H%M%S}-{h}"


def _comment(text: str) -> str:
    return "\n".join("-- " + line for line in text.replace("\r", "").splitlines())


class MigrationGenerator:
    def generate(self, *, request_id: str, query_fingerprint: str, validated_ddls: list[dict],
                 partition: PartitionAdvice | dict | None = None, sharding: ShardingAdvice | dict | None = None,
                 rewrite_sql_masked: str | None = None, now: datetime | None = None,
                 schema: str = "public") -> DeploymentPlan:
        """`validated_ddls`: [{"real_ddl": canonical DDL, "rationale": str, "planner_cost_reduction_pct": float|None}]"""
        ts = now or datetime.now(timezone.utc)
        mid = migration_id(query_fingerprint, [d["real_ddl"] for d in validated_ddls], ts)
        header = _comment(
            f"Planora staging migration {mid}\n"
            f"generated_at: {ts.isoformat(timespec='seconds')}\n"
            f"request_id: {request_id}\n"
            f"source query fingerprint: {query_fingerprint}\n"
            "STATUS: PENDING HUMAN REVIEW - for STAGING. Planora never executes this script.\n"
            "CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction block: run statements one at a\n"
            "time (e.g. psql without --single-transaction) and check for INVALID indexes afterwards.")
        mig, rb, notes = [header, "SET lock_timeout = '5s';", "SET statement_timeout = 0;"], [], []
        rb.append(_comment(f"Emergency rollback for {mid}\nDrops ONLY the indexes created by this migration."))
        for d in validated_ddls:
            parts = real_index_parts(d["real_ddl"])
            name = deterministic_index_name(parts["table"], parts["columns"], parts["include"], parts["where"])
            cols = ", ".join(quote_ident(c) for c in parts["columns"])
            sch = "public" if schema == "public" else quote_ident(schema)
            stmt = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)}\n    ON {sch}.{quote_ident(parts['table'])} USING btree ({cols})"
            if parts["include"]:
                stmt += f"\n    INCLUDE ({', '.join(quote_ident(c) for c in parts['include'])})"
            if parts["where"]:
                stmt += f"\n    WHERE {parts['where']}"
            pct = d.get("planner_cost_reduction_pct")
            mig.append(_comment(f"rationale: {d.get('rationale', '')}\n"
                                f"planner-estimated cost reduction (HypoPG, NOT runtime): "
                                f"{'n/a' if pct is None else str(pct) + ' %'}"))
            mig.append(stmt + ";")
            rb.append(f"DROP INDEX CONCURRENTLY IF EXISTS {sch}.{quote_ident(name)};")
        if not validated_ddls:
            mig.append("-- no validated index change: nothing to migrate")
            rb.append("-- nothing to roll back")
        if validated_ddls:
            mig.append("-- post-deployment check (run after each CREATE INDEX CONCURRENTLY):\n"
                       "-- SELECT indexrelid::regclass FROM pg_index WHERE NOT indisvalid;")
        mig.append("ANALYZE;" if validated_ddls else "")
        for adv, kind in ((partition, "partitioning"), (sharding, "sharding")):
            if adv:
                a = adv if isinstance(adv, dict) else adv.model_dump()
                notes.append(f"{kind} ({a.get('strategy_type')} on {a.get('candidate_key')}): "
                             f"{a.get('label', 'Advisory - Requires isolated design/benchmark validation')} - "
                             "NOT included in the migration script.")
        if rewrite_sql_masked:
            notes.append("SQL rewrite is an application/query change, not DDL - review the rewritten query; "
                         "it is NOT part of this migration.")
        rationale = "; ".join(d.get("rationale", "") for d in validated_ddls) or "no index change"
        return DeploymentPlan(migration_id=mid, migration_sql="\n".join(x for x in mig if x).strip() + "\n",
                              rollback_sql="\n".join(rb).strip() + "\n", requires_human_review=True,
                              generated_at=ts.isoformat(timespec="seconds"), rationale=rationale,
                              query_fingerprint=query_fingerprint, approval_status="PENDING_REVIEW",
                              advisory_notes=notes)
