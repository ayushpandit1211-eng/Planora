# Planora - Architecture

## 1. Component architecture

| Layer | Components | Trust level |
|---|---|---|
| Presentation | `dashboard/app.py`, `streaming_ui.py`, `components.py`, `plots.py` (Streamlit) | Trusted DBA UI - talks only to the API |
| API | `app/api.py` (FastAPI, SSE), `app/events.py`, `app/audit.py`, `app/metrics.py` | Trusted backend |
| Orchestration | `app/agents/supervisor.py` (LangGraph `StateGraph`), `app/agents/state.py`, `app/pipeline.py` (service container + trusted store) | Trusted backend |
| Privacy boundary | `security.py`, `pii.py`, `masking_vault.py`, `llm_guard.py`, `capability_firewall.py` | Trusted backend - the boundary itself |
| Analysis (anonymised) | `predicate_intelligence.py`, `plan_parser.sanitize_plan`, `gnn/*`, `rl/*`, `agents/sql_rewriter.py`, `agents/schema_expert.py`, `agents/index_strategist.py`, `partition_advisor.py`, `cardinality_analyzer.py`, `sql_semantic_diff.py`, `evidence_engine.py` | Works only on pseudonymised data |
| Trusted execution | `recommendation_mapper.py` (translation + validation), `sandbox.py` (HypoPG), `blast_radius.py`, `migration_generator.py`, `benchmark.py` | Real identifiers; never sent to an LLM |
| PostgreSQL instrumentation | `db.py` (asyncpg pool), `postgres_instrumentation.py`, `schema_inspector.py` | Least-privilege role `qs_app` |
| Workload intelligence | `ingestion.py`, `workload_store.py`, `regression_detector.py`, `workload_drift.py`, `index_hygiene.py` | Anonymised records only |
| External (optional) | `llm.py` - Gemini / OpenAI-compatible over HTTPS | Untrusted; receives guarded, anonymised payloads only |

## 2. Sequence flow (`POST /api/v1/optimize/async`)

```
Dashboard        API            Pipeline (LangGraph)                     PostgreSQL (qs_app)       LLM (optional)
   |  POST async   |                      |                                       |                      |
   |-------------->| create task -------->| security_validation                   |                      |
   |  {request_id} |                      | privacy_masking (vault: rid)          |                      |
   |<--------------|                      | schema_inspection ------------------->| catalog queries      |
   |  GET stream   |                      | explain ----------------------------->| EXPLAIN JSON (RO tx) |
   |-------------->| replay + live events | plan_to_graph -> gnn_analysis         |                      |
   |<== SSE =======|<---- publish --------| supervisor -> fan-out experts --------(guarded payload)------>|
   |               |                      | supervisor_merge -> rl_optimizer      |                      |
   |               |                      | recommendation_validator (translate)  |                      |
   |               |                      | hypopg_sandbox ---------------------->| hypopg_* + EXPLAIN   |
   |               |                      | blast_radius ------------------------>| pg_stat_statements   |
   |               |                      | final_report (migration + evidence)   |                      |
   |               |                      |   persist optimizer_meta ------------>| INSERT optimizer_meta|
   |  GET report   |<---- report ---------| vault cleared, trusted store dropped  |                      |
```

## 3. LangGraph state transitions

Nodes: `security_validation → privacy_masking → schema_inspection → explain → plan_to_graph →
gnn_analysis → supervisor → {sql_rewriter | schema_expert | index_strategist} → supervisor_merge →
rl_optimizer → recommendation_validator → hypopg_sandbox → blast_radius → final_report → END`.

Conditional edges:

| From | Condition | To |
|---|---|---|
| security / privacy / schema / explain / gnn | `state.error` set | `final_report` (error report, nothing else runs) |
| supervisor | `route` list (always `schema_expert`; `sql_rewriter` when a rewrite operator applies or an LLM is enabled for a bottleneck; `index_strategist` unless the plan is efficient) | parallel fan-out |
| recommendation_validator | no valid index and no translated rewrite | `final_report` (sandbox + blast radius `SKIPPED`) |
| hypopg_sandbox | no virtual index used by the planner | `final_report` (blast radius `SKIPPED`) |

Each node is wrapped to: skip when an earlier node failed, consume the agent-step budget, catch
errors into a structured `state.error`, record durations, and set the agent status
(`QUEUED/RUNNING/COMPLETED/SKIPPED/REJECTED/ERROR`). `agent_logs` and `agent_status` use LangGraph
reducers so parallel branches merge safely.

## 4. Privacy flow

```
raw SQL (trusted) --PII scan--> AST parse (pglast)
   literals  -> ParamRef $n (textual order)        vault.literals[$n] = original SQL literal
   tables    -> tbl_<hmac6>, columns -> col_<hmac6>, aliases -> als_<hmac6>   vault.reverse
   bitmask   <- PII flags + structural flags (+ plan flags)
masked SQL --audit (PII / literal / secret in output?)--> safe_for_llm | quarantine
LLM payload = template(privacy contract) + {masked SQL, anonymised schema, sanitised plan, GNN summary}
          --OutgoingPayloadGuard--> blocked (audit PRIVACY_BLOCK) | sent
LLM reply --Pydantic LLMRecommendation--> identifiers checked --> translate_llm_*_to_real (vault)
          --> RecommendationValidator rebuilds canonical DDL --> sandbox
end of request: vault.clear(request_id)
```

## 5. GNN flow

`EXPLAIN JSON → parse_explain (per-loop cost attribution) → sanitize_plan → plan_to_graph
(Data(x[N,32], edge_index[2,2E])) → PlanGCN → softmax → class + probability` and in parallel
`fallback_analytics.classify → rule + rule-confidence label + evidence`. If no valid checkpoint:
fallback only, `model_mode="analytic_fallback"`. If the trained class disagrees with the rules the
summary states the disagreement.

## 6. RL environment

```
EpisodeContext(state, masked low-level actions, evaluator, frequency_weight)
reset(): hypopg_reset, baseline EXPLAIN cost
step(a): ADD_INDEX  -> hypopg_create_index(real DDL) ; EXPLAIN current query
         REWRITE_SQL-> EXPLAIN rewritten real SQL (becomes current query)
         PARTITION / SHARD_ADVISORY -> not physically validated (no benefit, risk penalty)
         NO_CHANGE -> terminal
reward = w_freq * Δcost/baseline - storage - write - redundancy - risk
next state: indexes_added+1, new cost bucket, coverage updated ; done at NO_CHANGE / 3 steps / no actions
```
Inference (`RLPolicy.recommend`) uses the same state encoder and masking on the live request and
never steps the environment; training (`scripts/train_rl.py`) uses `SyncHypoPGEvaluator`.

## 7. Sandbox lifecycle

1. `hypopg_reset()` (clean backend), 2. baseline EXPLAIN, 3. per-candidate create → EXPLAIN →
reset, 4. combined experiment with candidates that were used **and** improved cost ≥ 1 %,
5. rewrite EXPLAIN (alone and with the indexes), 6. sizes via `hypopg_relation_size`,
7. `hypopg_reset()` in `finally`. All inside `BEGIN READ ONLY` on one pooled connection, rolled back.

## 8. Blast-radius analysis

pg_stat_statements → parse writes (pglast) → match the index table → rank by calls → top 3 →
per-operation impact (INSERT/UPDATE touching indexed columns/DELETE) → relative maintenance
increase, HOT-update loss, storage ratio → risk band. Labelled ESTIMATED.

## 9. Audit trail

`AuditLogger.record()` sanitises details, keeps a bounded in-memory history per request and
persists asynchronously to `optimizer_meta.audit_events` (failures never break the pipeline).
Recorded: request received, security scan / rejection, PII detection, masking, schema inspection,
EXPLAIN, graph conversion, GNN mode, agent routing, LLM call, RL recommendation, sandbox
experiment, blast radius, final recommendation, migration generation, privacy blocks, capability
denials, budget exhaustion, plan regression, workload drift, ingestion, benchmark, errors.

## 10. SSE event flow

`EventBus.publish()` validates the event name, redacts the message/data (PII/secret filter), appends
to the request history and pushes to subscriber queues. `GET /api/v1/stream/{id}` replays history
then streams live events, emits `HEARTBEAT` every 10 s, and ends at `FINAL_REPORT_READY`, `ERROR`
or `SECURITY_REJECTED`. The dashboard starts the async run first and then subscribes - replay
guarantees no event is lost.

## 11. Deployment boundary

```
AI recommendation -> validation -> sandbox experiment -> blast-radius analysis -> human review
-> generated staging migration (CREATE INDEX CONCURRENTLY) -> separate DBA execution
```
There is no code path that executes a migration; `qs_app` has no DDL privilege on business
tables; the capability firewall defines no production-write capability for any agent.

## 12. Data stores

| Store | Content | Lifetime |
|---|---|---|
| Vault (in-memory) | reverse identifier map, literal map | one request (TTL 30 min max) |
| `Services.trusted` | raw SQL, real schema/plan, real DDL, budget | one request |
| `optimizer_meta.optimization_runs` | fingerprint, anonymised SQL, plan fingerprint, recommendation summary, sandbox / blast metrics, model modes | persistent |
| `optimizer_meta.query_history` | anonymised workload records | persistent |
| `optimizer_meta.audit_events` | sanitised audit events | persistent |
| `optimizer_meta.migrations` | generated migration/rollback text, `PENDING_REVIEW` | persistent |
