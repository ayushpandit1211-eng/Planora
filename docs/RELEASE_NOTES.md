# Release and resume notes

## Confirmed fixes

- Restored missing sandbox configuration and test helpers that prevented test collection.
- Restored sandbox source resolution, connection-pool teardown, and synthetic-data provenance used by the supplied sandbox/benchmark modules.
- Fixed the integration test's zero-budget override to remain effective when the canonical environment variable is already configured; the exhaustion assertion remains unchanged.
- Added safe environment generation, an explicit readiness check, a two-process local launcher, secret/cache exclusions, loopback-only Docker ports, and offline/integration CI jobs.
- Preserved all supplied model artifacts and the existing forest-glass frontend.

## Resume wording

Use only statements matching your own contribution; the supplied project has existing contributor attribution.

Project description: **Planora — a privacy-aware PostgreSQL query optimization dashboard using FastAPI, Streamlit, PyTorch Geometric, LangGraph, and HypoPG.**

Possible bullets for work you personally performed:

- Integrated query-plan analysis, privacy masking, and hypothetical-index evaluation into an interactive PostgreSQL dashboard.
- Validated optimizer, dataset ingestion, query-safety, and sandbox-cleanup workflows with 272 passing tests against live PostgreSQL and isolated sandbox databases.
- Prepared a reproducible local demonstration with 400,100 synthetic relational records and a separate 10,000-row synthetic railway dataset.

Do not describe planner-cost reduction as measured query speedup. Do not claim IRCTC integration, government data access, production deployment, or sole authorship without evidence. Docker and GitHub Actions are configured but were not executed on a working Docker host during this Windows verification.

## Before a GitHub push

Use this folder as a standalone repository. Inspect the files and staged diff, confirm collaborator attribution/license rights, then add the intended remote and push. No remote repository has been created or modified by this preparation. Do not upload an older ZIP containing `.env` or local database files.

## Planora naming update — 10 October 2026

The active product, dashboard/browser title, API title, descriptive text, launcher, source folder, deployment project names and release package use Planora. The existing database name `querysentinel`, `QS_*` environment variables, `qs_*` roles/schemas, exception classes and telemetry identifiers are compatibility names; they are retained to preserve data and integrations. Trained model artifacts and the original attributed project report remain unchanged. Historical workspace copies and archives retain their original names. The source root is `output/Planora`; use `Start-Planora.ps1` in the E: workspace.
