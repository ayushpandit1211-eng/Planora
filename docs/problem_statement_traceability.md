# Problem Statement 4 - Traceability Matrix

Status legend: **Verified** = implemented and exercised by automated tests and/or a live run in
the build environment (see `.build_state/VERIFY.md`); **Verified (advisory)** = implemented,
output intentionally advisory.

| # | PS area | Requirement | Module / file | Demo evidence | Test(s) | Status | Limitation |
|---|---|---|---|---|---|---|---|
| 1 | Data Obfuscation & Ingestion Engine | Literal parameterisation via AST | `app/masking_vault.py` | `/api/v1/mask`; Privacy Audit tab | `test_masking.py::test_literals_parameterized` | Verified | Non-AST fallback limited to plain literals |
| 2 | Data Obfuscation & Ingestion Engine | Structural pseudonymisation (HMAC) + private vault | `app/masking_vault.py` | `tbl_xxxxxx / col_xxxxxx` in every anonymised view | `test_masking.py` (determinism, reverse translation, isolation) | Verified | Deterministic per key |
| 3 | Data Obfuscation & Ingestion Engine | Attribute bitmasking / structural metadata hashing | `app/pii.py` (PrivacyFlag), `plan_parser.plan_fingerprint` | `privacy_bitmask` in reports | `test_masking.py::test_bitmask_flags` | Verified | - |
| 4 | Data Obfuscation & Ingestion Engine | Slow-query logs & historical plans ingestion (JSON/CSV, pg_stat_statements) | `app/ingestion.py`, `app/workload_store.py` | 48/48 sample records, 36 PII-anonymised | `test_ingestion.py` | Verified | Demo formats only |
| 5 | AI & RL Optimization Agent | LangGraph multi-agent (Supervisor, SQL, Schema, Index) | `app/agents/*` | Agents tab: real node statuses | `test_agents.py` | Verified | - |
| 6 | AI & RL Optimization Agent | Reinforcement learning (not heuristics labelled RL) | `app/rl/*`, `scripts/train_rl.py` | Trained Q-policy, reward 0.77 vs 0.51 on unseen templates | `test_rl.py` | Verified | Small template set |
| 7 | AI & RL Optimization Agent | Composite-index recommendation | `app/agents/index_strategist.py`, `app/predicate_intelligence.py` | Scenario B → `(tenant_id, order_date)` | `test_agents.py::test_index_strategist_uses_only_known_identifiers` | Verified | ≤3 key columns |
| 8 | AI & RL Optimization Agent | Partitioning / sharding strategies | `app/partition_advisor.py` | Range on `order_date`: 1/36 partitions scanned on a temp partitioned copy | `test_research_modules.py::test_partition_advice_is_advisory`, `test_structural_sim.py` | Verified (simulated) | Sample-based; sharding advisory |
| 9 | AI & RL Optimization Agent | Automated SQL rewriting | `app/agents/sql_rewriter.py`, `app/sql_semantic_diff.py` | Scenario C/D rewrites, semantic diff | `test_agents.py`, `test_research_modules.py::test_semantic_diff_detects_changes` | Verified | 4 operators; human review |
| 10 | Execution Tree Analytics Module | Plan → graph, GNN bottleneck detection | `app/plan_parser.py`, `app/gnn/*` | Plan DAG with highlighted bottleneck | `test_plan_parser.py`, `test_gnn.py` | Verified | Synthetic training labels |
| 11 | Execution Tree Analytics Module | Deterministic fallback, explicit model mode | `app/gnn/fallback_analytics.py`, `app/gnn/analyzer.py` | `model_mode` shown everywhere | `test_gnn.py::test_fallback_mode_when_no_checkpoint` | Verified | - |
| 12 | Simulation Sandbox | HypoPG hypothetical indexes, cost comparison, cleanup | `app/sandbox.py` | Scenario C planner cost −99.98 % | `test_sandbox.py`, `test_integration.py::test_real_hypopg_sandbox_and_cleanup` | Verified | Planner estimate |
| 13 | Simulation Sandbox | PostgreSQL instrumentation via extensions | `app/postgres_instrumentation.py` | Status bar health | `test_integration.py::test_extensions_operational` | Verified | No native C hook |
| 14 | Simulation Sandbox | Blast radius / write overhead | `app/blast_radius.py`, `app/structural_sim.py` | Risk level + write latency in ms (timed INSERTs on a temp copy) | `test_sandbox.py::test_blast_radius_ranking_and_risk`, `test_structural_sim.py` | Verified | Temp-copy timing = order of magnitude |
| 15 | Interactive Dashboard | Streamlit DBA UI, NL input, live log | `dashboard/*` | Screenshots `docs/screenshots/` | Browser smoke test (VERIFY.md) | Verified | Prototype UI |
| 16 | Interactive Dashboard | Natural-language DBA assistant | `app/agents/nl_intent.py` | "weekly sales dashboard" → date_trunc report query | `/api/v1/nl/ask` live run (VERIFY.md) | Verified | Ranks existing workload only |
| 17 | Data Protection & Compliance | Zero raw-data exposure to AI | `app/llm_guard.py`, `app/capability_firewall.py` | PRIVACY_BLOCK audit events; per-report `zero_exposure_proof` + "Zero raw data was exposed" sentence | `test_masking.py::test_guard_blocks_and_audits`, `test_agents.py::test_mocked_llm_payload_is_anonymised_and_validated`, `test_structural_sim.py::test_live_report_has_answer_and_zero_exposure_proof` | Verified | Pattern-based detection |
| 18 | Data Protection & Compliance | Audit trail, least privilege | `app/audit.py`, `scripts/init_db.sql` | `optimizer_meta.audit_events` | `test_core.py`, `test_integration.py::test_least_privilege_role_cannot_write_or_ddl` | Verified | - |
| 19 | Operational Efficiency | Migration + rollback, human approval boundary | `app/migration_generator.py` | Deployment Review tab | `test_sandbox.py::test_migration_and_rollback_generation` | Verified | Staging only |
| 20 | Operational Efficiency | Optimization budget, metrics | `app/optimization_budget.py`, `app/metrics.py` | `budget` in final report | `test_core.py` | Verified | - |
| 21 | Explainability & Trust | Evidence-backed explanation & evidence chain | `app/evidence_engine.py`, `agents/supervisor._evidence_explanation` | Evidence Chain tab | `test_research_modules.py::test_evidence_chain_kinds` | Verified | - |
| 22 | Optimization Efficacy | >50 % improvement target - measured, never hard-coded | `app/benchmark.py`, `scripts/run_benchmark.py` | Measured −47.6 % (B), −99.98 % (D) runtime in build env | `test_api.py::test_optimize_contract` (measured null by default) | Verified | Environment-dependent |
| 23 | Robustness / Unseen Pattern Evaluation | Held-out evaluation for GNN and RL | `app/gnn/training.py`, `scripts/train_rl.py` | GNN unseen variants 83.5 %; RL unseen templates | `models/*_metrics.json` | Verified | Demo scale |
| 24 | Workload intelligence (R1/R2/R8) | Historical context, plan regression, drift | `workload_store.py`, `regression_detector.py`, `workload_drift.py` | Regressions found in sample history | `test_research_modules.py` | Verified | - |
| 25 | Source fidelity | Camera/billet line not implemented | README §27 | - | - | Documented | Awaiting official PS confirmation |
