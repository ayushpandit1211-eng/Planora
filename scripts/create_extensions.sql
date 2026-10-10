-- Extensions required by Planora (Section 6).
-- pg_stat_statements ALSO requires `shared_preload_libraries = 'pg_stat_statements'`
-- at server start (set in docker-compose.yml `command:`). CREATE EXTENSION alone is
-- not sufficient; app/postgres_instrumentation.py verifies the view is queryable.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS hypopg;
