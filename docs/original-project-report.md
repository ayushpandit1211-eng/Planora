# QuerySentinel

**Privacy-Aware AI-Assisted PostgreSQL Performance Optimization & DBA Intelligence Platform**

Code Utsava X.0 · NIT Raipur · Author: **Amit Singh Bisht**

> **QuerySentinel is an AI-assisted DBA decision-support prototype. It is NOT an autonomous
> production deployment engine.** It recommends, simulates and explains. Every change is a
> staging migration that a human must review and execute separately. Nothing is ever deployed
> automatically.

---

## Contents

1. [Problem context](#1-problem-context)
2. [Product overview](#2-product-overview)
3. [Architecture and end-to-end flow](#3-architecture-and-end-to-end-flow)
4. [Problem Statement Traceability Matrix](#4-problem-statement-traceability-matrix)
5. [Privacy architecture](#5-privacy-architecture)
6. [Security architecture](#6-security-architecture)
7. [Ingestion pipeline](#7-ingestion-pipeline)
8. [GNN design](#8-gnn-design)
9. [RL design](#9-rl-design)
10. [LangGraph design](#10-langgraph-design)
11. [HypoPG methodology](#11-hypopg-methodology)
12. [Blast-radius methodology](#12-blast-radius-methodology)
13. [Partitioning / sharding advisory methodology](#13-partitioning--sharding-advisory-methodology)
14. [API and SSE documentation](#14-api-and-sse-documentation)
15. [Dashboard walkthrough](#15-dashboard-walkthrough)
16. [Docker setup](#16-docker-setup)
17. [Environment variables](#17-environment-variables)
18. [Seed data: lightweight vs full mode](#18-seed-data-lightweight-vs-full-mode)
19. [Training commands](#19-training-commands)
20. [Testing commands](#20-testing-commands)
21. [Example queries](#21-example-queries)
22. [Benchmark instructions](#22-benchmark-instructions)
23. [Staging deployment and rollback workflow](#23-staging-deployment-and-rollback-workflow)
24. [Verified results in the build environment](#24-verified-results-in-the-build-environment)
25. [Limitations](#25-limitations)
26. [Future work](#26-future-work)
27. [Source-PS ambiguity note](#27-source-ps-ambiguity-note)

---

## 1. Problem context

Slow queries in production PostgreSQL databases are usually diagnosed by hand. A DBA reads
`EXPLAIN` output, guesses an index, and hopes it does not slow down writes. Sending production
SQL to an AI model is risky, because queries contain customer emails, IDs, amounts and
internal schema names.

Problem Statement 4 asks for AI and reinforcement-learning optimisation of database workloads with:

- secure anonymised ingestion
- graph-neural-network execution-tree analytics
- a simulation sandbox
- an interactive DBA dashboard

## 2. Product overview

QuerySentinel accepts three kinds of input:

- **a slow SQL query**
- **a natural-language DBA question**, such as *"Why is the weekly sales reporting dashboard timing out?"*
- **anonymised historical slow-query logs / execution plans**, in JSON or CSV

It then answers six questions with evidence:

1. *Why is this query slow?* Plan parsing plus GNN or rule-based diagnosis.
2. *What optimisation strategies are candidates?* Index strategist, SQL rewriter and schema expert.
3. *Which strategies are supported by plan/schema evidence?* Evidence chain and the validator.
4. *What does PostgreSQL's planner estimate if the proposed index exists?* HypoPG sandbox.
5. *What storage and write-workload impact could it have?* Blast-radius analysis.
6. *What should a DBA review before staging deployment?* Migration, rollback and limitations.

## 3. Architecture and end-to-end flow

```
USER / DBA ──► slow SQL │ NL question │ historical logs │ historical plans
                 │
                 ▼
 SECURITY GATEWAY      single read-only SELECT, no DDL/COPY/admin/dangerous functions, length limit
 PII PRE-FILTER        email / UUID / phone / JWT / API keys / passwords / connection strings
 PRIVACY VAULT         pglast AST parameterisation ($n) + HMAC-SHA256 identifier pseudonyms
                       + privacy bitmask; private reverse map kept server-side only
 SCHEMA INSPECTOR      catalog metadata only (no rows), pseudonymised before AI use
 POSTGRES EXPLAIN      EXPLAIN (FORMAT JSON, VERBOSE, BUFFERS) in a READ ONLY txn, restricted role
 PLAN PARSER           plan tree, per-loop cost attribution (SubPlan / nested loop / LIMIT)
 PLAN -> GRAPH         PyTorch Geometric Data(x, edge_index)
 GNN / FALLBACK        GCN bottleneck classifier (trained) or deterministic rules (fallback)
 LANGGRAPH SUPERVISOR ─┬─ SQL Rewriter ─┐
                       ├─ Schema Expert ┤  (conditional fan-out, then merge)
                       └─ Index Strategist
 RL OPTIMIZER          hierarchical tabular Q-learning with invalid-action masking
 RECOMMENDATION VALIDATOR  translate pseudonyms -> real names, rebuild + validate DDL
 HYPOPG SANDBOX        hypothetical indexes, EXPLAIN before/after, hypopg_reset() always
 BLAST RADIUS          pg_stat_statements write workload + write/storage estimates
 FINAL EVIDENCE REPORT evidence chain (observed / inferred / simulated / estimated / measured)
 MIGRATION + ROLLBACK  CREATE/DROP INDEX CONCURRENTLY, human review required
 STREAMLIT DASHBOARD + SSE LOGS
```

Detailed component, sequence and state-transition documentation is in
[`docs/architecture.md`](docs/architecture.md).

### Repository structure

```
db-opt-platform/
├── .env.example  .gitignore  Dockerfile  docker-compose.yml  requirements.txt  pytest.ini  README.md
├── .build_state/          MASTER_STATE.md FILE_MANIFEST.json CONTRACTS.md TRACEABILITY.md DECISIONS.md VERIFY.md
├── .streamlit/config.toml dark theme
├── docker/db/             PostgreSQL 16 + HypoPG image, app-role init script
├── docs/                  architecture.md security.md api.md problem_statement_traceability.md screenshots/
├── scripts/               init_db.sql create_extensions.sql seed_data.py generate_workload.py
│                          train_gnn.py train_rl.py run_benchmark.py entrypoint.sh
├── app/
│   ├── config.py db.py security.py pii.py masking_vault.py llm_guard.py llm.py
│   ├── ingestion.py schema_inspector.py plan_parser.py postgres_instrumentation.py
│   ├── recommendation_mapper.py migration_generator.py sandbox.py blast_radius.py partition_advisor.py
│   ├── audit.py events.py metrics.py contracts.py errors.py logging_setup.py pipeline.py api.py
│   ├── workload_store.py regression_detector.py predicate_intelligence.py index_hygiene.py
│   ├── cardinality_analyzer.py sql_semantic_diff.py workload_drift.py optimization_budget.py
│   ├── capability_firewall.py evidence_engine.py benchmark.py          (Addendum R1-R12)
│   ├── gnn/    mapper.py model.py training.py fallback_analytics.py analyzer.py
│   ├── rl/     state_encoder.py environment.py agent.py policy.py trainer.py fallback_policy.py episode_builder.py
│   └── agents/ state.py supervisor.py sql_rewriter.py schema_expert.py index_strategist.py nl_intent.py
├── dashboard/   app.py streaming_ui.py components.py plots.py
├── examples/    slow_queries.sql sample_history.json sample_history.csv
├── models/      gnn_bottleneck.pt gnn_metrics.json rl_policy.json rl_metrics.json  (produced by the training scripts)
└── tests/       15 test modules + fixtures captured from the demo database
```

## 4. Problem Statement Traceability Matrix

| PS requirement | Implementation module(s) | Evidence / demo | Limitation |
|---|---|---|---|
| Database optimisation | `agents/index_strategist.py`, `agents/sql_rewriter.py`, `sandbox.py` | Scenario C: HypoPG planner cost 319,578,254 → 49,745 | Planner estimates, not runtime |
| Artificial intelligence | `agents/supervisor.py` (LangGraph), `llm.py` | 9-node graph executes for every request | LLM optional; deterministic agents by default |
| Data privacy | `masking_vault.py`, `pii.py`, `llm_guard.py` | Tests: no literal / PII / real identifier reaches the LLM payload | Regex PII detection is not exhaustive |
| Reinforcement learning | `rl/*`, `scripts/train_rl.py` | Unseen templates: reward 0.77 vs 0.51 for the rule baseline | Small demo workload |
| Graph neural networks | `gnn/*`, `scripts/train_gnn.py` | Held-out 100 %, unseen plan shapes 83.5 % | Synthetic training labels |
| Secure anonymised ingestion | `ingestion.py`, `workload_store.py` | 48/48 sample records ingested, 36 with PII anonymised | JSON/CSV demo formats |
| Composite-index recommendation | `agents/index_strategist.py`, `predicate_intelligence.py` | Scenario B → `(tenant_id, order_date)` | Max 3 key columns |
| Partitioning/sharding recommendation | `partition_advisor.py`, `structural_sim.py` | Range on `order_date` simulated on a temp copy: 1 of 36 partitions scanned, −94 % cost | Sample-based simulation; deployment still advisory |
| Automated SQL rewriting | `agents/sql_rewriter.py`, `sql_semantic_diff.py` | Correlated COUNT → LEFT JOIN; date cast → range | 4 rewrite operators |
| Execution-tree explainability | `plan_parser.py`, `gnn/fallback_analytics.py`, `evidence_engine.py` | Evidence chain + plan DAG in dashboard | — |
| PostgreSQL simulation via extensions | `postgres_instrumentation.py`, `sandbox.py`, `structural_sim.py` | HypoPG + pg_stat_statements + EXPLAIN JSON; write latency in ms timed on temp copies (e.g. +1.1 ms per 1,000 inserts) | No native C planner hook; temp-copy timing is order-of-magnitude |
| Interactive DBA dashboard | `dashboard/*` | Plain-English answer card, demo mode (5 core tabs) or 9 tabs, SSE live log, real status bar | Streamlit prototype UI |
| Evidence-backed optimisation | `evidence_engine.py`, `contracts.py` | Every item labelled observed/inferred/simulated/estimated/measured | — |
| Zero raw-data exposure to AI | `llm_guard.py`, `capability_firewall.py`, `agents/supervisor._zero_exposure_proof` | Every report re-checks the exact AI payload and states "Zero raw data was exposed" with counts (100 % of literals masked) | Guard is pattern-based |
| Safe pre-deployment experimentation | `sandbox.py`, `benchmark.py`, `migration_generator.py` | Read-only txn, `hypopg_reset()` in `finally`, rolled-back benchmark | Measured mode needs a disposable DB |

A per-file table is in [`docs/problem_statement_traceability.md`](docs/problem_statement_traceability.md).

## 5. Privacy architecture

- **AST parameterisation (pglast):** literals become `$1, $2, …` in textual order.
  For example, `'amit@example.com' → $1`, `5000 → $2`, `'2026-10-01' → $3`. GROUP BY / ORDER BY
  ordinals are kept, because they are structure, not data.
- **Structural pseudonymisation:** `orders → tbl_xxxxxx` and `tenant_id → col_xxxxxx`, using
  HMAC-SHA256 with `PRIVACY_HMAC_KEY`. Pseudonyms are deterministic, so workload history stays
  comparable. HMAC is **not reversible**.
- **Private vault:** the reverse map (pseudonym → original) and the literal map (`$n` → value)
  live in a request-scoped in-memory store. They are cleared when the request
  ends and never returned by the API, logged, or sent to an LLM.
- **Bitmask abstraction:** structural and sensitive flags are stored as integer bit fields, such as
  `contains_email_literal` and `contains_join`. This is used *in addition to* literal removal, never
  instead of it.
- **Outgoing payload guard:** every LLM payload is scanned for PII, secrets, raw literals of the
  request, real identifiers from the reverse map, and raw result values. Any hit **blocks** the call,
  writes a `PRIVACY_BLOCK` audit event, and returns a structured error.
- **Controlled fallback:** if AST parsing fails, only plain literals and known identifiers are
  replaced. Dollar-quoting, quoted identifiers and comments cause **quarantine**.

## 6. Security architecture

- The gateway accepts exactly one `SelectStmt`. It rejects DDL, DML, `COPY`, `SET`,
  `ALTER SYSTEM`, `VACUUM`, `SELECT INTO`, `FOR UPDATE`, data-modifying CTEs, and dangerous
  functions such as `pg_read_file`, `pg_sleep`, `dblink`, `lo_*`, `set_config` and `nextval`.
  It also rejects credential catalogs and over-long queries.
- User SQL is never executed. Only `EXPLAIN (FORMAT JSON, VERBOSE, BUFFERS)` runs, inside
  `BEGIN READ ONLY` with `SET LOCAL statement_timeout`. `EXPLAIN ANALYZE` exists only in the opt-in
  benchmark, against a disposable DB.
- **Least privilege in PostgreSQL itself.** Role `qs_app` has `SELECT` on the business tables and
  writes only to `optimizer_meta`. It also has `pg_read_all_stats`, and can use HypoPG. This was
  verified by a live test: INSERT and CREATE INDEX are denied.
- **Capability firewall (R10):** each agent maps to explicit capabilities. No agent has a
  production-write capability, and denials are audited.
- Errors are structured, and stack traces are never returned. JSON logs are redacted for
  literals, emails, UUIDs, tokens and DSNs.

Full threat model: [`docs/security.md`](docs/security.md).

## 7. Ingestion pipeline

`raw source → PII pre-filter → AST processing → literal parameterisation → structural
pseudonymisation → structural feature extraction → privacy verification → safe historical record`

Each stored record keeps only structural, anonymised fields:

- `query_fingerprint` (pglast fingerprint) and `timestamp_bucket` (hour)
- `plan_fingerprint` (computed over pseudonyms) and `operator_histogram`
- `estimated_cost`, `estimated_rows` and actual rows when present
- `join_features`, `sort_features`, `index_features`
- `table_size_bucket`, `write_frequency_bucket`, and the anonymised SQL

Each record uses its own vault, which is cleared immediately. Raw result rows are never ingested.
Records are persisted to `optimizer_meta.query_history`. A collector derives candidates from
`pg_stat_statements` (`POST /api/v1/ingest/pg_stat_statements`).

## 8. GNN design

- **Graph.** Nodes are plan operators; edges are parent↔child links in both directions.
  Each node has 32 features:
  - operator one-hot using the spec mapping (`Seq Scan`=0 … `Materialize`=11, other=12)
  - operator-family one-hot
  - log1p of startup cost, total cost, rows, width, exclusive cost and loops
  - effective cost share, depth, filter / index-condition / parallel / subplan flags
  - top-cost indicator, gap to the max share, and log plan cost
- **Model.** `GCNConv(32→32) → ReLU → GCNConv(32→16) → ReLU → global_max_pool → Linear(16→4)`.
  Classes: Unindexed Sequential Scan, High-Cost Nested Loop / Join, Heavy Sort / Aggregation,
  Efficient / No Major Bottleneck.
- **Cost attribution.** Children of a correlated `SubPlan` and the inner side of a `Nested Loop`
  are charged per outer row. Children of a `LIMIT` are charged only for the fraction consumed.
  This is why scenario C's 319 M planner cost is attributed to the inner `Seq Scan` that runs
  about 4,668 times.
- **Training.** `scripts/train_gnn.py` uses a synthetic, PostgreSQL-shaped plan generator. The
  scenario defines the label, and cheap distractor operators are added. Evaluation uses a held-out
  seed and **unseen plan-shape variants**.
- **Two explicit modes.** `trained` reports a softmax probability. `analytic_fallback` uses
  deterministic rules and reports a rule label (`rule:high`), never a fake probability. A corrupt
  or missing checkpoint switches to fallback, and the API says so.

## 9. RL design

- **State:** discretised, identifier-free features. These are the bottleneck class, cost and
  cardinality buckets, operator flags, eq/range filter counts, non-sargable and subplan flags,
  existing-index coverage, table-size and write-frequency buckets, partition suitability, and
  indexes added so far. Example key: `bc0|cb4|cr1|ss1|…`.
- **Hierarchical actions (R7).** High-level actions are `ADD_INDEX | REWRITE_SQL | PARTITION |
  SHARD_ADVISORY | NO_CHANGE`. Each has low-level abstract signatures, for example
  `composite:eq+range`, `single:join`, `partial:range|partial`, `covering:eq+range+inc` or
  `correlated_count_subquery_to_left_join`.
- **Invalid-action masking.** Actions come only from schema and workload evidence. Nonexistent
  identifiers, redundant or duplicate indexes, and safety-policy violations are removed before
  the policy sees the action set.
- **Algorithm.** Tabular Q-learning (α 0.2, γ 0.9, ε-greedy with decay 0.993 down to 0.05), with
  multi-step episodes of up to 3 actions where hypothetical indexes accumulate. For unseen states
  it backs off to a coarse key, then to state-marginal values (reported). It supports `train()`,
  `predict()` (`select`), `update()`, `save_policy()` and `load_policy()`.
- **Reward:**
  `w_freq · normalized_planner_cost_reduction − storage − write − redundancy − risk`, where
  `w_freq = 1 + log10(1 + workload calls)`. Partition and shard actions get no validated benefit,
  only a risk penalty. This reward reflects planner-estimated benefit, not real user value.
- **Training environment.** Real HypoPG and EXPLAIN on the demo DB. Training uses 10 workload
  templates; evaluation uses 5 held-out templates. Stopping happens at max episodes or when the
  moving-average reward stabilises.
- **Fallback.** Without a policy file, `policy_mode = "analytic_fallback"` uses deterministic rules,
  and it is never labelled as trained RL inference.

## 10. LangGraph design

The pipeline is a real compiled `StateGraph`, and every request executes its state transitions
(`app/agents/supervisor.py`). Its state follows Section 22 (`request_id`, `masked_sql`,
`gnn_diagnosis`, `schema_stats`, `proposed_sql`, `proposed_indexes`,
`partitioning_recommendation`, `rl_recommendation`, `sandbox_results`, `blast_radius`,
`agent_logs`, `final_report`), plus status, routing and error keys. Trusted data never enters
the state.

Conditional routing:

- the SQL Rewriter is skipped when no rewrite applies
- the Index Strategist is skipped for efficient plans
- HypoPG is skipped when nothing valid remains
- blast radius runs only for indexes the planner actually used
- unsafe or invalid recommendations are rejected
- high-risk results are flagged, and every result requires human review

## 11. HypoPG methodology

The sandbox runs everything on one connection inside a READ ONLY transaction:

1. baseline `EXPLAIN`
2. create each validated candidate alone with `SELECT * FROM hypopg_create_index($1)`, then `EXPLAIN`
   again (per-candidate attribution)
3. keep only candidates the planner used **and** that cut cost by ≥ 1 %
4. combined experiment
5. EXPLAIN of the rewritten SQL, alone and together with the indexes
6. capture `hypopg_relation_size`
7. `hypopg_reset()` in a `finally` block

The cost-reduction metric is `((before − after) / before) × 100`, with zero and null handled.
It is labelled **PLANNER-ESTIMATED COST REDUCTION (not a runtime speedup)**. If combining the
rewrite with the indexes is estimated worse than the best single option, the report says to
deploy one or the other.

## 12. Blast-radius methodology

`pg_stat_statements` provides candidate INSERT/UPDATE/DELETE statements. They are parsed with
pglast, matched to the index's table, ranked by calls, and the top three are analysed. Per
operation:

- an INSERT adds one index entry per row
- an UPDATE adds index work only if it modifies an indexed column, and then also loses HOT eligibility
- a DELETE leaves dead entries

The relative maintenance increase is 1 / (existing indexes). Storage comes from HypoPG. The risk
band (low / medium / high) combines write share, indexed-column updates and index-size ratio;
the ratio only counts for indexes of at least 10 MB or 50 MB. These values are labelled
**ESTIMATED**.

**Write latency in milliseconds** (`app/structural_sim.py`). A temporary copy of the target table
is built from ~20,000 sampled rows plus all its existing indexes (unique ones rebuilt as
non-unique). INSERTs of 4,000 rows are timed (median of 7, each rolled back to a savepoint):
baseline, with the proposed index, then baseline again (A-B-A) to cancel timing drift. The result is reported as ms per 1,000 inserted rows
and per single-row write, labelled **SIMULATED**. Everything runs in one transaction that is
always rolled back. The app role cannot write business tables, so only temp objects are touched.
A small copy has a shallower B-tree than the full table, so treat the number as an
order-of-magnitude estimate.

## 13. Partitioning / sharding advisory methodology

Evidence used: table size, range predicates on date/timestamp columns, physical correlation,
equality access on tenant-like keys, `n_distinct`, and skew. Skew comes from
`pg_stats.most_common_freqs[1]`, which is a frequency and never a value.

Output covers range, hash, and list partitioning (where appropriate) plus tenant-sharding advice.

**Partition simulation** (`app/structural_sim.py`). A temporary sample copy (60,000 rows) and a
temporary partitioned copy (monthly range partitions, or 8 hash partitions) are both ANALYZEd.
A representative query (latest one-month window, or one key value) is EXPLAINed on each. The
report shows partitions scanned vs total and the cost on each copy. Example: scenario B scans
1 of 36 monthly partitions, −94 % planner cost on the sample. The transaction is rolled back.
Each piece of advice includes the strategy type, candidate key, rationale, expected benefit,
operational risks and validation required. It is always labelled
**"Advisory - Requires isolated design/benchmark validation"**, is never included in migration
scripts, and production tables are never partitioned automatically.

## 14. API and SSE documentation

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | DB / HypoPG / pg_stat_statements health |
| GET | `/api/v1/status` | Dashboard status bar (all components + model modes) |
| POST | `/api/v1/mask` | Anonymise SQL (reverse map never returned) |
| POST | `/api/v1/analyze` | Security → privacy → schema → EXPLAIN → graph → GNN |
| POST | `/api/v1/optimize` | Full pipeline, synchronous; Section 39 response contract |
| POST | `/api/v1/optimize/async` | Start the pipeline, get `request_id` + stream URL |
| GET | `/api/v1/report/{request_id}` | Final report (202 while running) |
| GET | `/api/v1/stream/{request_id}` | **SSE** live events (`text/event-stream`) |
| POST | `/api/v1/ingest/logs` · `/api/v1/ingest/plans` | JSON/CSV ingestion |
| POST | `/api/v1/ingest/pg_stat_statements` | Collector |
| GET | `/api/v1/workload` | Workload summary, plan regressions (R2), drift (R8) |
| GET | `/api/v1/hygiene` | Index hygiene report (R4) |
| POST | `/api/v1/benchmark` | Benchmark + ablation lab (R12) |
| POST | `/api/v1/nl/ask` · `/api/v1/nl/analyze` | Natural-language DBA assistant |
| GET | `/api/v1/audit/{request_id}` | Audit events |
| GET | `/api/v1/examples` | Demo scenarios |

Every SSE event has the shape `{"event", "timestamp", "request_id", "stage", "message", "data"}`.
The events are:

- `REQUEST_RECEIVED`, `SECURITY_SCAN_STARTED`/`_COMPLETE`, `PII_SCAN_COMPLETE`
- `MASKING_STARTED`/`_COMPLETE`, `SCHEMA_INSPECTION_COMPLETE`, `EXPLAIN_STARTED`/`_COMPLETE`
- `GRAPH_CONVERSION_COMPLETE`, `GNN_ANALYSIS_COMPLETE`, `SUPERVISOR_STARTED`
- `SQL_AGENT_*`, `SCHEMA_AGENT_*`, `INDEX_AGENT_*`, `RL_ANALYSIS_*`, `SANDBOX_*`, `BLAST_RADIUS_*`
- `FINAL_REPORT_READY`, `ERROR`
- extras: `AGENT_SKIPPED`, `VALIDATION_COMPLETE`, `MIGRATION_GENERATED`, `PLAN_REGRESSION`,
  `WORKLOAD_DRIFT`, `SEARCH_BUDGET_EXHAUSTED`, `PRIVACY_BLOCKED`, `SECURITY_REJECTED`

Messages are redacted, and raw SQL and literals never appear. Full schemas and examples are in
[`docs/api.md`](docs/api.md).

```bash
curl -s localhost:8000/api/v1/health
curl -s -X POST localhost:8000/api/v1/optimize -H 'content-type: application/json' \
     -d '{"sql":"SELECT count(*) FROM orders o WHERE o.status = '\''pending'\'' AND o.order_date >= '\''2026-09-01'\''"}'
RID=$(curl -s -X POST localhost:8000/api/v1/optimize/async -H 'content-type: application/json' \
      -d '{"sql":"SELECT o.order_id FROM orders o WHERE o.customer_email = '\''customer1@example.com'\''"}' | python -c 'import sys,json;print(json.load(sys.stdin)["request_id"])')
curl -N localhost:8000/api/v1/stream/$RID
curl -s localhost:8000/api/v1/report/$RID
```

## 15. Dashboard walkthrough

The dashboard (`http://localhost:8501`) uses a wide, dark layout. The **status bar** shows real
backend health for PostgreSQL, HypoPG, pg_stat_statements, the Privacy Vault, the LLM, the GNN
mode, the RL mode and LangGraph.

1. **Ask the DBA assistant** in plain English. The assistant ranks *existing* workload statements
   and cites its evidence. Click *Analyze this candidate*.
2. Or **load a demo scenario** from the sidebar, or paste SQL, then click *Run optimisation
   pipeline*. The live log and agent tree update from the SSE stream.
3. Explore the tabs. **Demo mode** (sidebar toggle, ON by default) shows only the 5 core tabs: Privacy Audit, Execution Plan / GNN, Optimization, HypoPG, Deployment Review. Turn it OFF to see all tabs.

| Tab | What it shows |
|---|---|
| **Privacy Audit** | Raw SQL (INTERNAL / SENSITIVE) vs anonymised SQL (SAFE FOR MODEL INGESTION), PII, literal count, fingerprint, pseudonym count, fallback status, payload check, bitmask |
| **Execution Plan / GNN** | Plan DAG with the bottleneck in red, operator cost chart, class probabilities or the fallback label, evidence, plan tree |
| **LangGraph Agents** | Supervisor tree with QUEUED / RUNNING / COMPLETED / SKIPPED / REJECTED / ERROR, the agent log, and the live SSE event stream (expander) |
| **Optimization** | Original vs rewritten SQL with a diff and semantic-diff verdict, proposed indexes, partition and shard advisories, RL action with Q-values and masked actions, reasoning, assumptions, risks, uncertainty |
| **HypoPG** | Baseline vs optimised planner cost, planner-estimated reduction, virtual indexes, storage, write overhead, blast radius, risk |
| **Deployment Review** | Staging migration (`CREATE INDEX CONCURRENTLY`), emergency rollback (`DROP INDEX CONCURRENTLY`), rationale, fingerprint, migration ID, timestamp, human-review banner, downloads |
| **Evidence Chain** | The Section 68 explanation and the R11 chain with evidence kinds |
| **Workload Intelligence** | Ingest files, collect from pg_stat_statements, regressions, drift chart, index hygiene |
| **Benchmark Lab** | Modes and ablations, plus an optional measured mode |

Screenshots: `docs/screenshots/`.

## 16. Docker setup

```bash
cp .env.example .env            # set POSTGRES_PASSWORD, QS_APP_PASSWORD, PRIVACY_HMAC_KEY
docker compose up --build       # db (PG16 + HypoPG + pg_stat_statements) -> backend -> dashboard
# backend   : http://localhost:8000/docs
# dashboard : http://localhost:8501
docker compose down             # add -v to drop the database volume
```

How the services start:

- `db` is built from `docker/db/Dockerfile` (`postgres:16-bookworm` + `postgresql-16-hypopg`).
  It starts with `shared_preload_libraries=pg_stat_statements`, has a health check, and runs the
  init scripts: extensions, schema, roles, and the app-role password from the environment.
- `backend` waits for the database to be healthy, then (`SEED_ON_START=true`) seeds the
  lightweight dataset and a demo workload. It then runs
  `uvicorn app.api:app --host 0.0.0.0 --port 8000` as a non-root user.
- `dashboard` waits for the backend to be healthy and runs
  `streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501`.
- No secret is hard-coded; compose refuses to start without the passwords.

**Running without Docker** (as verified in the build environment):

```bash
pip install -r requirements.txt
# PostgreSQL 16 with: shared_preload_libraries = 'pg_stat_statements'; apt install postgresql-16-hypopg
createdb querysentinel
export POSTGRES_HOST=localhost POSTGRES_USER=postgres QS_APP_PASSWORD=change-me
python scripts/seed_data.py              # lightweight; add --full for 1M/3M rows
python scripts/generate_workload.py      # pg_stat_statements evidence (writes are rolled back)
uvicorn app.api:app --host 0.0.0.0 --port 8000
BACKEND_URL=http://localhost:8000 streamlit run dashboard/app.py --server.port 8501
```

## 17. Environment variables

All variables are documented in [`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `db/5432/querysentinel/postgres/–` | Superuser; used only by the db container and seed scripts |
| `QS_APP_USER/QS_APP_PASSWORD` | `qs_app/–` | Least-privilege application role |
| `PRIVACY_HMAC_KEY` | random per process | HMAC key for pseudonyms; set it for stable pseudonyms |
| `MAX_QUERY_LENGTH`, `STATEMENT_TIMEOUT_MS` | 20000, 15000 | Gateway limits |
| `QS_API_KEY` | empty | Optional `X-API-Key` for the API |
| `LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL` | `none` | `none`, `gemini`, `openai` (any OpenAI-compatible endpoint) or `mock` |
| `GNN_CHECKPOINT_PATH`, `RL_POLICY_PATH` | `models/...` | Model artefacts |
| `BENCHMARK_ENABLED`, `BENCHMARK_ADMIN_DSN`, `BENCHMARK_REPETITIONS` | `false`, –, 3 | Measured mode (disposable DB only) |
| `BUDGET_MAX_*` | 8/10/20/2/50 | Optimisation budget (R9) |
| `DRIFT_JS_THRESHOLD` | 0.2 | Workload drift threshold (R8) |
| `SEED_TENANTS/SEED_ORDERS/SEED_ORDER_ITEMS/SEED_RANDOM_SEED` | 100/100000/300000/42 | Seed size |

## 18. Seed data: lightweight vs full mode

| Mode | Command | Size | Time in the build environment |
|---|---|---|---|
| Lightweight (default) | `python scripts/seed_data.py` | 100 tenants · 100,000 orders · 300,000 items | ~7 s |
| Full demonstration | `python scripts/seed_data.py --full` | 100 · 1,000,000 · 3,000,000 | 68 s (measured) |

The seed data is fully configurable:

- size overrides: `SEED_TENANTS`, `SEED_ORDERS`, `SEED_ORDER_ITEMS`
- `--reset` regenerates the data; `--if-empty` skips seeding when data already exists
- loading uses `COPY` batches, so millions of rows are never held in RAM
- the seed is deterministic (42)
- the data has realistic status, amount, date and Zipf product distributions
- 80 % of orders belong to 5 hot tenants
- `ANALYZE` runs at the end

The index landscape is deliberately mixed. `orders(status)` and `orders(order_date)` exist.
`orders(tenant_id, order_date)`, `orders(status, order_date)`, `order_items(order_id, product_sku)`
and both FK indexes are intentionally missing.

## 19. Training commands

```bash
python scripts/train_gnn.py --real-plans      # -> models/gnn_bottleneck.pt + gnn_metrics.json  (~10 s CPU)
python scripts/train_rl.py                    # -> models/rl_policy.json + rl_metrics.json   (needs the demo DB + HypoPG, ~5 s)
```

Trained artefacts produced by these scripts in the build environment are included in `models/`.
Delete them to see the explicit fallback modes.

## 20. Testing commands

```bash
python -m compileall app dashboard scripts tests
pytest -q                              # 141 tests; LLM calls are mocked
pytest -q -m "not integration"         # offline only (137 tests, fixtures captured from the demo DB)
docker compose config --quiet          # compose validation
```

The 4 integration tests run automatically when a seeded PostgreSQL with HypoPG is reachable.
They check extension health, least-privilege enforcement, a real HypoPG experiment with cleanup,
and the live end-to-end pipeline.

## 21. Example queries

See [`examples/slow_queries.sql`](examples/slow_queries.sql). For each scenario it lists the
expected evidence, index strategy, rewrite, sandbox method, blast-radius check and deployment
review:

- **A:** unindexed multi-table join
- **B:** date-range aggregation ("weekly sales reporting")
- **C:** nested (correlated) subquery
- **D:** non-sargable date predicate with a PII literal
- **E:** an efficient control query

`examples/sample_history.json` and `examples/sample_history.csv` contain 48 historical records
with real EXPLAIN (ANALYZE) plans captured from the demo DB. They span two time windows, and a
temporary index existed only in the first window. This gives the regression detector a genuine
plan change to find.

## 22. Benchmark instructions

```bash
python scripts/run_benchmark.py                       # planner-estimate mode, all scenarios, all modes + ablations
BENCHMARK_ENABLED=true BENCHMARK_ADMIN_DSN=postgresql://user:pass@host/disposable_copy \
  python scripts/run_benchmark.py --measure --query 1 # real EXPLAIN ANALYZE timings
```

- **Modes:** `baseline`, `index_only`, `rewrite_only`, `gnn_assisted`, `rl_assisted`, `full`
- **Ablations:** `without_gnn`, `without_rl`, `without_workload_context`, `without_hypopg`,
  `without_predicate_intelligence`, `without_blast_radius`
- **Report contents:** PostgreSQL version, dataset size, relevant settings, query fingerprint,
  baseline and optimised plan fingerprints, planner cost delta, and the measured runtime delta
  when measured
- **Measured mode** builds the real indexes inside a transaction that is always rolled back, and
  times the query with `EXPLAIN ANALYZE` (median of N runs).

> **Results depend on PostgreSQL version, statistics, hardware, data distribution,
> configuration and cache state.** No improvement target is hard-coded. The PS target of
> > 50 % is reported only when a measurement shows it.

## 23. Staging deployment and rollback workflow

1. Review the **Deployment Review** tab: rationale, evidence chain, blast radius and limitations.
2. Download `<migration_id>_migration.sql`. Apply it to **staging** with `psql` outside a
   transaction block, because CONCURRENTLY cannot run inside one. Then check for invalid indexes
   using the query included in the script.
3. Run the measured benchmark and your own write-load test on staging.
4. Promote it through your normal change management, executed by a DBA.
5. **Rollback:** run `<migration_id>_rollback.sql`. It uses `DROP INDEX CONCURRENTLY IF EXISTS`
   and drops only the indexes that migration created.

Partition and shard advice and SQL rewrites are never part of a migration script. They need
design work and application changes.

## 24. Verified results in the build environment

These come from PostgreSQL 16.15 with HypoPG 1.4.0, on the full dataset (1M orders, 3M items),
CPU only. They are planner estimates unless marked *measured*.

| Scenario | Diagnosis | Recommendation | Planner cost (HypoPG) | Measured (opt-in) |
|---|---|---|---|---|
| A join | Seq Scan on `order_items` | `order_items(product_sku)` + `orders(tenant_id, order_date)` | 65,628 → 17,859 (−72.8 %) | — |
| B weekly report | Bitmap heap scan with tenant filter | `orders(tenant_id, order_date)` | 23,103 → 18,177 (−21.3 %) | 57.3 ms → 30.0 ms median (−47.6 %) |
| C correlated subquery | Inner Seq Scan ×~4,668 | FK index `order_items(order_id)`; LEFT JOIN rewrite | 319,578,255 → 49,745 (−99.98 %) | not run (baseline too slow for the sandbox) |
| D cast + email | Seq Scan, non-sargable cast | `orders(customer_email)`; range rewrite | 22,036 → 20 (−99.9 %) | 44.3 ms → 0.009 ms (−99.98 %) |
| E point lookup | Efficient | No change | — | — |

- **GNN:** held-out synthetic plans 100 %. Never-seen plan-shape variants 83.5 %. Agreement with
  the analytic rules on the 5 live plans: 4/5. On the correlated subquery the GNN says "join" and
  the rules say "scan"; the dashboard shows both and flags the result as uncertain.
- **RL** (mean episode reward, higher is better):
  - seen templates: trained 0.599 vs rule fallback 0.512
  - unseen templates: trained 0.767 vs rule fallback 0.511
  - recommendation validity is 100 % for both

### Example answer (scenario B, shown in the dashboard's green Answer box)

> Adding a composite index on orders(tenant_id, order_date) is estimated to cut query cost by 20.7%,
> with only +1.09 ms simulated write latency per 1,000 inserted rows (~0.0011 ms per single-row
> write) and +149 MB of index storage. Optionally, partitioning orders by month on order_date would
> let date-window queries scan 1 of 36 partitions (-94.2% cost on a simulated sample copy). Zero raw
> data was exposed during this analysis: 4/4 literal values parameterised, 5/5 table/column names
> replaced by HMAC hashes, and the outgoing AI payload passed the leak check.

## 25. Limitations

- **Planner estimates are not runtime.** HypoPG and EXPLAIN costs depend on statistics. Runtime
  claims need the measured benchmark on representative hardware and data.
- **The GNN is trained on synthetic labelled plans.** No public labelled corpus was available
  offline, so the GNN mostly learns cost-share structure. Real-plan performance is reported as
  agreement with the rules, not as accuracy.
- **The RL workload is small.** It uses 15 templates on one demo schema. This is not evidence of
  scientific generalisation, and the reward is planner-cost based.
- **The rewriter has 4 operators.** The semantic diff checks structure; it does not prove
  equivalence. All rewrites require human review.
- **PII detection is pattern-based.** Free-text identifiers in unusual formats may be missed. AST
  parameterisation still removes *all* literals, whether or not PII is detected.
- **Generic plans can't see skew.** Parameterised (`$n`) workload statements are analysed with
  `EXPLAIN (GENERIC_PLAN)`, so data skew such as hot tenants is invisible. Re-run with literal
  values for a value-specific diagnosis.
- **No native C-level planner hook.** Instrumentation is extension and API based (HypoPG,
  pg_stat_statements, EXPLAIN, catalogs).
- **Partitioning is simulated on a sample copy, not the full table.** Sharding is advisory only.
  Neither is ever applied automatically.
- **Write latency is timed on a small temp copy.** It shows the order of magnitude, not the exact
  production number.
- **The default vault is in-memory, per process.** Run one backend worker (multi-worker would need a shared vault store).
- **Docker images were not built in the build sandbox,** because the registry was blocked by the
  network proxy. `docker compose config` validates. Everything else was verified against a native
  PostgreSQL 16 + HypoPG install; see `.build_state/VERIFY.md`.
- **The LLM integration is optional.** It is tested with a mocked provider only; no real LLM call
  was made in the build environment.

## 26. Future work

- A labelled real-world plan corpus for GNN training, and calibration of confidence scores.
- Workload-level RL (multi-query index selection under a storage budget) with a larger template set.
- More rewrite operators with formal equivalence checks.
- Full-size partition experiments in disposable clones (pg_partman), Citus-based shard simulation.
- Incident retrieval with learned plan embeddings (R13 is currently heuristic similarity).
- Role-based access control and SSO for the dashboard.

## 27. Source-PS ambiguity note

The supplied Problem Statement 4 text contains a malformed line about *live camera frames of
passing billets*. It is structurally inconsistent with the database-optimisation requirements.
**No computer-vision subsystem was built.** Any future industrial-input integration should be
added as an explicit extension point (for example, a new ingestion source producing anonymised
workload records). It should not become a core feature until the official PS confirms it.

---

## 28. Final Upgrade: use your own data

Two data sources, picked in the sidebar ("Data source"):

| Source | Where the data lives | Lifetime |
|---|---|---|
| **Demo Database** | the existing `public` schema (never modified) | permanent |
| **Upload Dataset (CSV, JSON)** | main PostgreSQL, schema `qs_dynamic`, table named after the dataset (e.g. `shop`) | until deleted |

How each works, in short:

- **Upload a dataset** → *Preview* shows detected format, column types and warnings (nothing written) → *Commit*
  loads it in one transaction. Re-uploads only ever ADD nullable columns; an incompatible file is rejected with the
  bad values and row numbers, and nothing is written.
- **Delete a dataset** → in *Upload Dataset* mode, *Your uploaded datasets* lists every upload; select one or more,
  confirm, and *Delete selected* drops their tables and registry rows (`DELETE /api/v1/data/datasets/{id}`). The active
  dataset can also be deleted from the source panel. The demo database can never be deleted.
- **Ask Your Database** → a plain-English question becomes ONE SELECT built from the real columns, passes a
  10-stage safety gate (parse, SELECT-only, catalogs, relation and column checks, function allowlist, EXPLAIN cost
  cap, row cap, read-only restricted role), optionally goes through the existing optimizer (GNN, RL, HypoPG), runs,
  and is answered. Works without an LLM key (template generator, labelled ANALYTIC FALLBACK); with
  `LLM_PROVIDER` set, the LLM writes the SQL and failed validations are fed back for repair.

Every number carries a provenance badge: POSTGRES OBSERVED, PLANNER ESTIMATE, HYPOPG SIMULATION, MEASURED SANDBOX,
ANALYTIC FALLBACK, TRAINED GNN, RL POLICY, LLM RECOMMENDATION, HEURISTIC ESTIMATE, DP-PROTECTED.

Optional features (environment flags, default off): `QS_FEATURE_GATV2` (GATv2 attention overlay on the plan
graph), `QS_FEATURE_INDEX_PORTFOLIO` (duplicate/redundant/unused index review), `QS_FEATURE_DP` (Laplace noise on
numeric aggregates sent to AI, with an epsilon budget), `QS_FEATURE_RL_UPGRADES` (curriculum + prioritized replay
options for RL training). All variables are listed in `.env.example`; API contracts in
`.build_state/API_CONTRACTS.md`; the full upgrade record in `.build_state/FINAL_REPORT.md`.

Honest limits: the session id is
ownership, not authentication; NL-to-SQL correctness depends on the generator while the validator only guarantees
safety; type inference is heuristic; index advice is planner-estimate based; GATv2 and RL are trained on synthetic
labels; DP covers only the specific aggregate releases under the stated assumptions.
