# Planora - API Reference

Base URL `http://localhost:8000`. Interactive OpenAPI: `/docs`. If `QS_API_KEY` is set, send
`X-API-Key: <key>` on every endpoint except `/api/v1/health` and `/api/v1/examples`.

**Error behaviour (all endpoints):** structured JSON `{"error": <category>, "message": <safe text>,
"stage": <stage|null>, "details": {...}, "request_id": <id|null>}`. Categories include
`invalid_sql`, `unsafe_sql`, `pii_leak`, `secret_leak`, `ast_failure`, `quarantined`,
`schema_mismatch`, `unknown_identifier`, `invalid_ddl`, `llm_timeout`, `llm_malformed_json`,
`llm_unavailable`, `hypopg_unavailable`, `pg_stat_statements_unavailable`,
`gnn_model_unavailable`, `corrupted_model_checkpoint`, `rl_model_unavailable`,
`database_unavailable`, `sandbox_failure`, `capability_denied`, `budget_exhausted`, `not_found`,
`internal_error`. Stack traces are never returned. HTTP: 400 unsafe/invalid, 401 bad key, 404
unknown id, 422 privacy/validation, 503 dependency unavailable, 500 internal.

**Privacy notes (all endpoints):** responses never contain the reverse pseudonym mapping.
`/optimize` responses intentionally contain real identifiers in `deployment.migration_sql`,
`deployment.rollback_sql` and `deployment.rewritten_sql_for_review` (labelled INTERNAL) because the
DBA needs executable scripts; every other section is anonymised.

---

## GET /api/v1/health
Response `200` (or `503` when PostgreSQL is down):
```json
{"status": "ok", "postgresql": true, "hypopg": true, "pg_stat_statements": true}
```

## GET /api/v1/status
Component status for the dashboard: `postgresql`, `hypopg`, `pg_stat_statements`,
`privacy_vault`, `llm` (provider/model/enabled), `gnn` (mode, held-out accuracy), `rl` (mode,
evaluation), `langgraph`, `workload_store`, redacted `settings`.

## POST /api/v1/mask
Request `{"sql": "SELECT * FROM orders WHERE customer_email = 'amit@example.com'", "request_id": null}`
Response:
```json
{"request_id": "qs-…", "masked_sql": "SELECT * FROM tbl_a241e7 WHERE col_df358b = $1",
 "ast_signature": "3ab240e40856d01b", "literal_count": 1,
 "mapped_tables": {"tbl_a241e7": "table"}, "mapped_columns": {"col_df358b": "column"},
 "fallback_used": false, "quarantined": false, "quarantine_reason": null,
 "privacy_bitmask": 1, "privacy_flags": {"contains_email_literal": true, "...": false},
 "privacy_audit": {"pii_detected": true, "pii_types": ["email"], "literal_leak_detected": false,
                   "secret_leak_detected": false, "safe_for_llm": true},
 "note": "reverse mapping is kept server-side and is never returned"}
```

## POST /api/v1/analyze
Request `{"sql": "<single SELECT>"}`. Runs security → privacy → schema → EXPLAIN → graph → GNN.
Response: `request_id, privacy_audit, schema_analysis (anonymised), plan_analysis (sanitised nodes,
edges, operator_histogram, highest_cost_node, seq_scans, plan_fingerprint, plan_shape, tree_text),
graph {num_nodes, num_edges, feature_dim}, gnn_analysis, access_pattern, workload_context,
regression`.

## POST /api/v1/optimize  (synchronous)  ·  POST /api/v1/optimize/async
Request `{"sql": "<single SELECT>", "request_id": "optional-6-64-chars"}`.
Async response: `{"request_id": "...", "stream_url": "/api/v1/stream/<id>", "report_url": "/api/v1/report/<id>"}`.
Sync / report response (Section 39 contract + dashboard fields):
```json
{
 "request_id": "qs-…", "status": "completed",
 "privacy_audit": {"masked_sql": "...", "literal_count": 2, "privacy_audit": {"safe_for_llm": true, "...": "..."}},
 "schema_analysis": {"anonymized_schema": {"tables": {...}, "indexes": [...], "foreign_keys": [...]},
                     "schema_expert": {"findings": [...], "missing_access_paths": [...], "cardinality_analysis": {...}}},
 "plan_analysis": {"total_cost": 319578254.73, "highest_cost_node": {...}, "tree_text": "..."},
 "gnn_analysis": {"primary_bottleneck_type": "...", "confidence": 0.99, "model_mode": "trained",
                  "highest_cost_node": {...}, "natural_language_summary": "...", "graph_nodes": [...], "graph_edges": [...]},
 "agent_recommendation": {"rewritten_sql": "...", "proposed_indexes": [...], "partitioning_recommendation": {...},
                          "sharding_recommendation": {...}, "technical_reasoning": [...], "assumptions": [...],
                          "risks": [...], "confidence": 0.99, "llm": {...}},
 "rl_recommendation": {"policy_mode": "trained_policy", "high_level_action": "REWRITE_SQL", "low_level_action": "...",
                       "q_values": {...}, "valid_actions": [...], "masked_actions": {...}, "explanation": "...", "drift": {...}},
 "sandbox_simulation": {"before_total_cost": 319578254.73, "after_total_cost": 49745.28,
                        "planner_cost_reduction_pct": 99.98, "virtual_indexes_used": ["CREATE INDEX ON tbl_… (col_…)"],
                        "estimated_storage_impact_mb": 119.59, "estimated_write_overhead": {...},
                        "metric_label": "PLANNER-ESTIMATED COST REDUCTION (not a runtime speedup)", "cleanup_ok": true},
 "blast_radius": {"results": [{"target_table": "tbl_…", "risk_level": "medium", "write_queries_analyzed": [...],
                  "write_frequency_summary": {...}, "estimated_write_impact": {...}, "methodology": "..."}], "overall_risk": "medium"},
 "deployment": {"migration_sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS ...", "rollback_sql": "DROP INDEX CONCURRENTLY IF EXISTS ...",
                "requires_human_review": true, "migration_id": "QSM-…", "approval_status": "PENDING_REVIEW", "...": "..."},
 "limitations": ["Planner-estimated cost reductions come from EXPLAIN/HypoPG and are NOT runtime speedups.", "..."],
 "final_report": {"query_fingerprint": "...", "privacy_audit": {}, "diagnosis": {"primary_bottleneck": "...", "evidence": []},
                  "optimization": {"rewritten_sql": "...", "proposed_indexes": [], "partitioning_recommendation": null,
                                   "sharding_recommendation": null, "rl_action": null},
                  "validation": {"hypopg_result": {}, "planner_cost_reduction_pct": null, "measured_execution_time_reduction_pct": null},
                  "blast_radius": [], "deployment": {"migration_sql": "...", "rollback_sql": "...", "requires_human_review": true},
                  "evidence_chain": [], "explanation": {}, "model_modes": {}, "budget": {}, "workload_context": {}, "limitations": []},
 "access_pattern": {...}, "workload_context": {...}, "agent_status": {...}, "agent_logs": [...]
}
```
Rejected: HTTP 400 `{"request_id": "...", "status": "rejected", "error": {"error": "unsafe_sql", "message": "DropStmt: DDL is not allowed on analysis paths", ...}}`.

## GET /api/v1/report/{request_id}
`200` final report · `202 {"status": "running"}` · `404` unknown id.

## GET /api/v1/stream/{request_id}  (Server-Sent Events)
`Content-Type: text/event-stream`. Each message:
```
event: SANDBOX_COMPLETE
data: {"event": "SANDBOX_COMPLETE", "timestamp": "2026-10-03T09:22:51.123+00:00", "request_id": "qs-…",
       "stage": "sandbox", "message": "HypoPG simulation completed: planner-estimated cost reduction 99.98%", "data": {"status": "completed"}}
```
History is replayed for late subscribers; `HEARTBEAT` every 10 s; the stream ends after
`FINAL_REPORT_READY`, `ERROR` or `SECURITY_REJECTED`. Messages are redacted (no SQL/literals).

## POST /api/v1/ingest/logs  ·  POST /api/v1/ingest/plans
Request (one of):
```json
{"format": "json", "content": "<JSON text: list or {\"records\": [...]}>"}
{"format": "csv",  "content": "query,calls,mean_time_ms,timestamp,plan_json\n..."}
{"records": [{"query": "SELECT ...", "calls": 10, "mean_time_ms": 4.2, "timestamp": "2026-09-01T10:00:00", "plan": {"Plan": {...}}}]}
```
Response: `batch_id, accepted, rejected, quarantined, records_with_pii_removed, plans_parsed,
errors, records (anonymised WorkloadRecord list), persisted_to_optimizer_meta`.

## POST /api/v1/ingest/pg_stat_statements
Request `{"limit": 50, "min_mean_ms": 0}` → same response shape (source `pg_stat_statements`).

## GET /api/v1/workload
`summary`, `top_fingerprints`, `regressions` (PLAN_REGRESSION / PLAN_IMPROVEMENT / PLAN_CHANGE with
evidence), `drift` (Jensen-Shannon scores per dimension, threshold, status).

## GET /api/v1/hygiene
Report-only: `duplicate_indexes`, `redundant_indexes`, `unused_index_candidates`, `missing_fk_indexes`.

## POST /api/v1/benchmark
Request `{"demo_query_index": 1, "variants": null, "measure": false}` or `{"sql": "...", ...}`.
Response: `query_fingerprint, environment {postgresql_version, settings, dataset, gnn_mode,
rl_mode}, baseline_plan_fingerprint, baseline_planner_cost, results[{variant, note, indexes,
indexes_used_by_planner, rewrite, planner_cost, plan_fingerprint, planner_cost_reduction_pct,
write_risk_assessed, validated_by_hypopg}], measurement {measured: false, reason} |
{measured: true, before_ms_median, after_ms_median, measured_execution_time_reduction_pct, ...},
metric_separation, caveat`.

## POST /api/v1/nl/ask
Request `{"question": "Why is the weekly sales reporting dashboard timing out?"}`. Response:
`question_intent {intents, time_granularity, matched_tables, matched_columns}`, `candidates
[{candidate_id, anonymized_sql, score, calls, mean_exec_time_ms, total_exec_time_ms, evidence[]}]`,
`similar_incidents` (R13), `answer`, `source`, `boundary`. No SQL is generated or executed.

## POST /api/v1/nl/analyze
Request `{"candidate_id": "cand-0123456789"}` → same as `/optimize/async` (candidate text resolved
server-side; never returned raw).

## GET /api/v1/audit/{request_id}
`{"request_id": "...", "events": [{"event_type", "stage", "severity", "details", "timestamp"}]}`.

## GET /api/v1/metrics  ·  GET /api/v1/metrics/prometheus
Counters and stage-duration percentiles (JSON or Prometheus text). No sensitive values.

## GET /api/v1/examples
Demonstration scenarios A-E with SQL.
