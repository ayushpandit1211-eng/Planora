# Planora

**Privacy-Aware PostgreSQL Optimization**

A PostgreSQL query-analysis dashboard that explains execution plans, checks privacy, and evaluates index recommendations with HypoPG before a person reviews the generated migration.

Built with FastAPI, Streamlit, PostgreSQL 16, PyTorch/PyTorch Geometric, LangGraph, and Plotly. The forest-glass interface keeps the existing application workflows and trained GNN/RL artifacts.

## What works

- Analyze read-only SQL with a security gate, literal masking, and structural pseudonyms.
- Inspect query-plan graphs with the supplied GNN checkpoint and RL policy.
- Compare hypothetical indexes using real PostgreSQL planner estimates; no physical index is created by the main optimization pipeline.
- Review blast-radius estimates, migration SQL, and rollback SQL.
- Upload CSV/JSON datasets, inspect inferred schemas, ask supported natural-language questions, and run validated SQL.
- Review workload statistics and audit evidence. Optional LLM access is disabled by default; the local deterministic path works without an API key.

**Verification, 9 October 2026:** 272 tests passed, zero skipped, against native PostgreSQL 16.15 + HypoPG 1.4.3 and a separate disposable sandbox server. See [verification](docs/VERIFICATION.md). GitHub Actions is configured; its remote run and the Docker image build have not yet been verified.

## Deploy for trusted reviewers

Use the separate [production deployment guide](docs/DEPLOYMENT.md) for HTTPS, login, private networking, backups and release checks. The development Compose file below is for local use.

## Run with Docker

Requires Docker Engine/Compose. Docker Desktop on Windows needs working virtualization; it is not required for the native setup below.

```bash
python scripts/setup_env.py
docker compose up --build -d
```

The setup script creates random local database passwords and a privacy HMAC key, and preserves an existing `.env`. First startup installs dependencies and seeds the demo database.

Open **http://localhost:8501** (dashboard) and **http://localhost:8000/docs** (API).

```bash
python scripts/check_ready.py
docker compose logs --tail=80 backend
docker compose down
```

`check_ready.py` must show PASS for PostgreSQL, HypoPG, pg_stat_statements, privacy vault, LangGraph, and both trained models. LLM disabled is expected. `docker compose down` retains the data volume.

## Run without Docker

See [native setup](docs/NATIVE_SETUP.md). Use PostgreSQL 16 with matching HypoPG binaries and preload `pg_stat_statements`. After preparing the database and `.env`:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python scripts/run_local.py
```

The launcher runs both services on loopback and stops them with Ctrl+C. It does not install or start PostgreSQL. Native Windows HypoPG must match the PostgreSQL server ABI; do not install an arbitrary DLL.

## Five-minute demonstration

1. Select **Demo Database**, select scenario **C - nested (correlated) subquery**, then **Load scenario**.
2. Click **Run optimisation pipeline**. Wait for the final report.
3. Open **Privacy Audit**, **Execution Plan / GNN**, **Simulation (HypoPG)**, and **Deployment Review**.
4. Explain that planner-cost reduction is an estimate, not measured latency improvement.
5. Upload [railway_demo.csv](examples/datasets/railway_demo.csv), preview and commit it as `railway_demo`. The 10,000 rows are entirely synthetic, not IRCTC or government records. Run the example query in [dataset notes](examples/datasets/README.md).

Turn off **Demo mode (5 core tabs)** to expose all nine report views. Plain-English support without an LLM uses supported templates; use explicit read-only SQL for unsupported questions.

## Data and models

The seed script creates 100 tenants, 100,000 orders, and 300,000 order items by default, using a deterministic seed and fictional emails. Real passenger or production records are not bundled. The original trained files under `models/` were not retrained or modified for this release.

```bash
# Run only against a disposable configured database; reads POSTGRES_* and QS_APP_PASSWORD.
python scripts/seed_data.py --if-empty
python -m pytest -q
```

Without PostgreSQL, integration tests skip. The CI integration job provisions two disposable servers and rejects unexpected skips. BYOD schema-sandbox/benchmark modules are tested library functionality; the current dashboard exposes dataset upload, not a separate BYOD schema-upload screen.

## Repository and publication

The `.gitignore` and `.dockerignore` exclude `.env`, local runtimes, logs, and caches. `.env.example` contains placeholders. Run `python scripts/setup_env.py` locally instead of committing real credentials.

Use this folder as the repository root, not the parent workspace or an older project ZIP. Review `git status` before committing. See [release and resume notes](docs/RELEASE_NOTES.md).

This is a local research/demo prototype, not a production service with multi-user authentication. Session IDs provide ownership separation, not authentication. Docker ports bind to loopback. Public hosting needs a separate deployment/security review. Core optimization recommendations require human review; the existing optional uploaded-dataset apply-index action is a real mutation when explicitly used.

## Attribution

The supplied project report credits **Amit Singh Bisht**, Code Utsava X.0, NIT Raipur. That attribution is retained in [the original report](docs/original-project-report.md). The report is archival and includes older verification claims; use the current verification document for this release. No new ownership claim or software license has been invented. Confirm contributor attribution and redistribution rights before publishing.
