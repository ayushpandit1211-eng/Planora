# Native setup (without Docker)

The verified local environment used Windows x64, Python from the existing workspace virtual environment, PostgreSQL 16.15, HypoPG 1.4.3, and preloaded pg_stat_statements. Runtime binaries and data are deliberately excluded from the GitHub source package.

1. Install Python 3.12 or 3.13 and PostgreSQL 16. Install a HypoPG build compatible with that exact server distribution and architecture. On Linux, use the PostgreSQL package repository's `postgresql-16-hypopg` package. On Windows, vendor-compatible binaries or a compatible native build are required.
2. In PostgreSQL configuration, set `shared_preload_libraries = 'pg_stat_statements'` and restart the database server. Bind the demo server to localhost.
3. Create an empty database named `querysentinel` as the local PostgreSQL administrator.
4. Run `python scripts/setup_env.py`. Edit `.env`: use `POSTGRES_HOST=127.0.0.1`, the actual port, database name, and administrator credentials. Keep a separate `QS_APP_PASSWORD` for the restricted app role. Do not commit `.env`.
5. Create a virtual environment and install `requirements.txt`. Seed using the wrapper below, which loads `.env` without exposing passwords on the command line:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from scripts.seed_data import main; raise SystemExit(main())" --if-empty
```

The seed script creates the required extensions, demo schema, permissions, and synthetic records. `--if-empty` preserves existing seeded data. A missing HypoPG extension is a setup failure; do not silently omit it.

6. Run `python scripts/run_local.py`. Open http://localhost:8501. In a second terminal, run `python scripts/check_ready.py`.

## Tests with a real database

Use disposable databases, never production. Load the same `.env` into the test process and override `QS_SESSION_MODE=per_session` for the ownership tests. A separate PostgreSQL instance is required for the optional sandbox tests. Configure `SANDBOX_PG_HOST`, `SANDBOX_PG_PORT`, `SANDBOX_PG_USER`, `SANDBOX_PG_PASSWORD`, and `SANDBOX_SECRET` there. Empty sandbox host means sandbox tests are skipped; it does not fall back to the main database.

The sandbox instance must have HypoPG installed. Tests create and drop `sbx_*` databases/roles and exercise TTL cleanup. The default synthetic sizes are 1,000 / 10,000 / 100,000 rows per table. These settings restore the supplied sandbox modules; they do not add a new dashboard route.

## Windows build used during verification

HypoPG was built from the official 1.4.3 source with Zig 0.16.0 and the PostgreSQL 16.15 headers/import library. The compiler's default MinGW exception buffer differs from an MSVC-built PostgreSQL server, so a private build-header override selected the server-compatible `jmp_buf` path. This does not modify project/model code. That DLL was tested in isolated databases before being loaded by the local demo. It is not included as a portable binary or claimed compatible with other PostgreSQL distributions.

Official source: https://github.com/HypoPG/hypopg/releases/tag/1.4.3
Installation documentation: https://hypopg.readthedocs.io/en/rel1_stable/installation.html
