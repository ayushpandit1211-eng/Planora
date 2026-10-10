"""Slow-query and historical plan ingestion engine (Section 16).

raw source -> PII pre-filter -> AST processing -> literal parameterisation ->
structural pseudonymisation -> structural feature extraction -> privacy verification ->
safe historical record

Supports JSON and CSV demonstration formats and a collector that derives candidate
slow-query metadata from pg_stat_statements. Raw result rows are never ingested and raw
logs are never forwarded to an LLM.
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from pglast import fingerprint as pg_fingerprint

from app.contracts import SchemaMetadata
from app.llm_guard import OutgoingPayloadGuard
from app.masking_vault import DataObfuscator
from app.pii import PIIDetector
from app.plan_parser import parse_explain, sanitize_plan
from app.predicate_intelligence import PredicateAnalyzer
from app.schema_inspector import table_size_bucket, write_frequency_bucket
from app.security import SecurityGateway
from app.workload_store import WorkloadRecord, WorkloadStore, hour_bucket

QUERY_KEYS = ("query", "sql", "statement")


@dataclass
class IngestionReport:
    batch_id: str
    accepted: int = 0
    rejected: int = 0
    quarantined: int = 0
    pii_records: int = 0
    plans_parsed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    records: list[WorkloadRecord] = field(default_factory=list)

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        out = {"batch_id": self.batch_id, "accepted": self.accepted, "rejected": self.rejected,
               "quarantined": self.quarantined, "records_with_pii_removed": self.pii_records,
               "plans_parsed": self.plans_parsed, "errors": self.errors[:50]}
        if include_records:
            out["records"] = [r.model_dump() for r in self.records[:200]]
        return out


def _num(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class IngestionEngine:
    def __init__(self, obfuscator: DataObfuscator, store: WorkloadStore, guard: OutgoingPayloadGuard,
                 schema: SchemaMetadata | None = None):
        self.obf = obfuscator
        self.store = store
        self.guard = guard
        self.schema = schema
        self.detector = PIIDetector()

    # ----------------------------------------------------------------- parsers
    @staticmethod
    def parse_json(payload: str | bytes | list | dict) -> list[dict]:
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        if isinstance(data, dict):
            data = data.get("records") or data.get("queries") or data.get("plans") or [data]
        if not isinstance(data, list):
            raise ValueError("JSON ingestion expects a list of records")
        return [d for d in data if isinstance(d, dict)]

    @staticmethod
    def parse_csv(payload: str | bytes) -> list[dict]:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        rows = list(csv.DictReader(io.StringIO(text)))
        for r in rows:
            pj = r.get("plan_json") or r.get("plan")
            if pj:
                try:
                    r["plan"] = json.loads(pj)
                except json.JSONDecodeError:
                    r["plan"] = None
        return rows

    # ----------------------------------------------------------------- core
    def _column_index_masked(self, batch_id: str) -> dict[str, set[str]]:
        if not self.schema:
            return {}
        out: dict[str, set[str]] = {}
        for t, meta in self.schema.tables.items():
            mt = self.obf.pseudonymize_identifier(batch_id, t, "table")
            out[mt] = {self.obf.pseudonymize_identifier(batch_id, c.name, "column") for c in meta.columns}
        return out

    def ingest(self, records: list[dict], source: str) -> IngestionReport:
        batch_id = f"ingest-{uuid.uuid4().hex[:12]}"
        report = IngestionReport(batch_id=batch_id)
        col_index = self._column_index_masked(batch_id)
        analyzer = PredicateAnalyzer(column_index=col_index)
        try:
            for i, raw in enumerate(records):
                rid = f"{batch_id}-{i}"   # per-record vault: literal checks never cross records
                try:
                    rec = self._one(raw, rid, analyzer, source, report)
                except Exception as exc:  # noqa: BLE001 - one bad record must not abort the batch
                    report.rejected += 1
                    report.errors.append({"index": i, "error": type(exc).__name__})
                    continue
                finally:
                    self.obf.clear_vault(rid)
                if rec is not None:
                    report.records.append(rec)
                    report.accepted += 1
            self.store.extend(report.records)
        finally:
            self.obf.clear_vault(batch_id)    # the reverse map never outlives the batch
        return report

    def _one(self, raw: dict, batch_id: str, analyzer: PredicateAnalyzer, source: str,
             report: IngestionReport) -> WorkloadRecord | None:
        sql = next((raw[k] for k in QUERY_KEYS if raw.get(k)), None)
        if not sql:
            report.rejected += 1
            report.errors.append({"error": "missing query field"})
            return None
        scan = self.detector.scan(sql)
        if scan.pii_detected or scan.secret_detected:
            report.pii_records += 1
        stype = SecurityGateway.classify_statement(sql) or "Unknown"
        masked = self.obf.mask_sql(sql, batch_id)
        if masked["quarantined"] or not masked["privacy_audit"]["safe_for_llm"]:
            report.quarantined += 1
            return None
        try:
            qfp = pg_fingerprint(sql)
        except Exception:  # noqa: BLE001
            qfp = masked["ast_signature"]
        ap = analyzer.analyze(masked["masked_sql"])
        calls = int(_num(raw.get("calls")) or 1)
        rec = WorkloadRecord(
            query_fingerprint=qfp,
            timestamp_bucket=hour_bucket(raw.get("timestamp") or raw.get("ts")),
            call_count=max(calls, 1),
            mean_exec_time_ms=_num(raw.get("mean_time_ms") or raw.get("mean_exec_time")),
            anonymized_sql=masked["masked_sql"],
            access_features=ap.to_dict(),
            join_features=ap.joins,
            sort_features=[f"{u.table}.{u.column}:{u.direction}" for u in ap.uses if u.role == "order"],
            statement_type=stype,
            source=source,
        )
        plan_doc = raw.get("plan")
        if isinstance(plan_doc, (dict, list)):
            try:
                pa = sanitize_plan(parse_explain(plan_doc), self.obf, batch_id)
                report.plans_parsed += 1
                rec.plan_fingerprint = pa.plan_fingerprint
                rec.plan_shape = pa.plan_shape
                rec.estimated_cost = pa.total_cost
                rec.estimated_rows = pa.plan_rows
                rec.operator_histogram = pa.operator_histogram
                root = pa.nodes[0]
                rec.actual_rows = root.actual_rows
                rec.index_features = sorted({n.index_name for n in pa.nodes if n.index_name})
                if pa.has_actuals:
                    rec.access_features["cardinality_pairs"] = [
                        {"node_id": n.node_id, "node_type": n.node_type, "plan_rows": n.plan_rows,
                         "actual_rows": n.actual_rows, "actual_loops": n.actual_loops}
                        for n in pa.nodes if n.actual_rows is not None]
            except (ValueError, KeyError, TypeError):
                report.errors.append({"error": "plan could not be parsed"})
        if self.schema:
            touched = [t for t in self.schema.tables if self.obf.to_masked_identifier(batch_id, t) in ap.tables]
            if touched:
                biggest = max(touched, key=lambda t: self.schema.tables[t].estimated_rows)
                tm = self.schema.tables[biggest]
                rec.table_size_bucket = table_size_bucket(tm.estimated_rows)
                rec.write_frequency_bucket = write_frequency_bucket(
                    (tm.n_tup_ins or 0) + (tm.n_tup_upd or 0) + (tm.n_tup_del or 0))
        # privacy verification of the stored record
        # (timestamp_bucket is coarse derived metadata and excluded from the literal check)
        report_check = self.guard.check(batch_id, "", rec.model_dump(exclude={"timestamp_bucket"}))
        if not report_check.allowed:
            report.quarantined += 1
            return None
        return rec

    def ingest_json(self, payload, source: str = "json_log") -> IngestionReport:
        return self.ingest(self.parse_json(payload), source)

    def ingest_csv(self, payload, source: str = "csv_log") -> IngestionReport:
        return self.ingest(self.parse_csv(payload), source)


class PgStatStatementsCollector:
    """Derives candidate slow-query metadata from pg_stat_statements.

    pg_stat_statements already stores normalised text ($n placeholders); the text still
    passes through the same masking/verification pipeline before storage.
    """

    def __init__(self, instrumentation, engine: IngestionEngine):
        self.instr = instrumentation
        self.engine = engine

    async def collect(self, limit: int = 50, min_mean_ms: float = 0.0, include_writes: bool = True) -> IngestionReport:
        rows = await self.instr.statements(limit * 4)
        recs = []
        for r in rows:
            q = r["query"] or ""
            st = SecurityGateway.classify_statement(q)
            if st not in ("SelectStmt", "InsertStmt", "UpdateStmt", "DeleteStmt"):
                continue   # DDL / utility / maintenance statements are not workload queries
            if st != "SelectStmt" and not include_writes:
                continue
            if any(x in q.lower() for x in ("pg_catalog", "pg_stat", "hypopg", "information_schema", "explain")):
                continue
            if (r["mean_exec_time"] or 0) < min_mean_ms:
                continue
            recs.append({"query": q, "calls": r["calls"], "mean_time_ms": r["mean_exec_time"]})
            if len(recs) >= limit:
                break
        return self.engine.ingest(recs, "pg_stat_statements")
