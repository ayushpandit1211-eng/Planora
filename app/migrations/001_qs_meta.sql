-- Planora Final Upgrade - additive, idempotent metadata migration (run after init_db.sql).
-- Never modifies the demo schema. Safe to run any number of times.
CREATE SCHEMA IF NOT EXISTS qs_meta;
CREATE SCHEMA IF NOT EXISTS qs_dynamic;

CREATE TABLE IF NOT EXISTS qs_meta.data_sources (
    id               text PRIMARY KEY,
    kind             text NOT NULL CHECK (kind IN ('demo', 'dynamic', 'sandbox')),
    name             text NOT NULL,
    owner_session_id text,
    status           text NOT NULL DEFAULT 'ready'
                     CHECK (status IN ('creating', 'ready', 'failed', 'expired', 'deleted')),
    db_name          text,
    schema_name      text,
    role_name        text,
    schema_version   integer NOT NULL DEFAULT 1,
    created_at       timestamptz NOT NULL DEFAULT now(),
    expires_at       timestamptz,
    last_accessed_at timestamptz,
    extensions_used  integer NOT NULL DEFAULT 0,
    config           jsonb NOT NULL DEFAULT '{}'::jsonb      -- never contains passwords
);
CREATE INDEX IF NOT EXISTS data_sources_owner_idx ON qs_meta.data_sources (owner_session_id, status);

CREATE TABLE IF NOT EXISTS qs_meta.datasets (
    id             text PRIMARY KEY,
    data_source_id text NOT NULL REFERENCES qs_meta.data_sources(id),
    name           text NOT NULL,
    original_name  text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qs_meta.dataset_tables (
    id             bigserial PRIMARY KEY,
    dataset_id     text NOT NULL REFERENCES qs_meta.datasets(id),
    schema_name    text NOT NULL,
    table_name     text NOT NULL,
    original_name  text,
    row_count      bigint NOT NULL DEFAULT 0,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (schema_name, table_name)
);

CREATE TABLE IF NOT EXISTS qs_meta.dataset_columns (
    id               bigserial PRIMARY KEY,
    table_id         bigint NOT NULL REFERENCES qs_meta.dataset_tables(id),
    column_name      text NOT NULL,
    original_name    text,
    data_type        text NOT NULL,
    nullable         boolean NOT NULL DEFAULT true,
    added_in_version integer NOT NULL DEFAULT 1,
    stats            jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (table_id, column_name)
);

CREATE TABLE IF NOT EXISTS qs_meta.schema_versions (
    id             bigserial PRIMARY KEY,
    data_source_id text NOT NULL REFERENCES qs_meta.data_sources(id),
    version        integer NOT NULL,
    diff           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (data_source_id, version)
);

CREATE TABLE IF NOT EXISTS qs_meta.ingestion_runs (
    id             text PRIMARY KEY,
    data_source_id text REFERENCES qs_meta.data_sources(id),
    dataset_id     text,
    filename       text,
    sha256         text,
    size_bytes     bigint,
    status         text NOT NULL,
    rows_read      bigint NOT NULL DEFAULT 0,
    rows_loaded    bigint NOT NULL DEFAULT 0,
    rows_rejected  bigint NOT NULL DEFAULT 0,
    schema_changes jsonb NOT NULL DEFAULT '[]'::jsonb,
    warnings       jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingestion_runs_sha_idx ON qs_meta.ingestion_runs (sha256);

CREATE TABLE IF NOT EXISTS qs_meta.audit_log (
    id             bigserial PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    data_source_id text,
    session_id     text,
    event_type     text NOT NULL,
    details        jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS qs_meta.query_runs (
    id             text PRIMARY KEY,
    data_source_id text NOT NULL,
    session_id     text,
    question_hash  text,
    status         text NOT NULL,
    stage_failed   text,
    sql_fingerprint text,
    generator      text,
    component_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Workload namespace (additive; existing rows backfilled with the neutral default 'demo')
ALTER TABLE optimizer_meta.query_history ADD COLUMN IF NOT EXISTS data_source_id text NOT NULL DEFAULT 'demo';

-- Restricted NL-query role: NOLOGIN, used via SET LOCAL ROLE inside READ ONLY transactions.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qs_nlq_reader') THEN
        CREATE ROLE qs_nlq_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END $$;
GRANT USAGE ON SCHEMA public TO qs_nlq_reader;
GRANT SELECT ON public.tenants, public.orders, public.order_items TO qs_nlq_reader;
GRANT USAGE ON SCHEMA qs_dynamic TO qs_nlq_reader;
REVOKE CREATE ON SCHEMA qs_dynamic FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qs_app') THEN
        EXECUTE 'GRANT qs_nlq_reader TO qs_app';
        EXECUTE 'GRANT USAGE ON SCHEMA qs_meta, qs_dynamic TO qs_app';
        EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA qs_meta TO qs_app';
        EXECUTE 'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA qs_meta TO qs_app';
        EXECUTE 'GRANT SELECT ON ALL TABLES IN SCHEMA qs_dynamic TO qs_app';
        EXECUTE 'GRANT DELETE ON optimizer_meta.query_history TO qs_app';
        EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA qs_dynamic GRANT SELECT ON TABLES TO qs_app';
    END IF;
END $$;
GRANT SELECT ON ALL TABLES IN SCHEMA qs_dynamic TO qs_nlq_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA qs_dynamic GRANT SELECT ON TABLES TO qs_nlq_reader;

INSERT INTO qs_meta.data_sources (id, kind, name, status, schema_name, role_name)
VALUES ('demo', 'demo', 'Demo Database', 'ready', 'public', 'qs_app')
ON CONFLICT (id) DO NOTHING;
