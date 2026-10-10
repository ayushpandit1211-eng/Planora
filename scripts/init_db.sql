-- Planora database initialisation (run as superuser on an empty database).
-- Executed automatically by the db container via /docker-entrypoint-initdb.d,
-- or manually:  psql -U postgres -d querysentinel -f scripts/init_db.sql
-- NOTE: extensions are created in scripts/create_extensions.sql (runs first).

-- ---------------------------------------------------------------------------
-- 1. Demonstration business schema (Section 8)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id  UUID PRIMARY KEY,
    name       TEXT NOT NULL,
    tier       TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id       UUID PRIMARY KEY,
    tenant_id      UUID NOT NULL REFERENCES tenants (tenant_id),
    customer_email TEXT,
    total_amount   NUMERIC(12,2) NOT NULL,
    status         TEXT NOT NULL,
    order_date     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id     UUID PRIMARY KEY,
    order_id    UUID NOT NULL REFERENCES orders (order_id),
    product_sku TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    unit_price  NUMERIC(10,2) NOT NULL
);

-- ---------------------------------------------------------------------------
-- 2. Realistic index landscape (Section 9)
--    Present : orders(status), orders(order_date)
--    Missing ON PURPOSE : orders(tenant_id, order_date), orders(status, order_date),
--                         order_items(order_id, product_sku), FK index order_items(order_id),
--                         FK index orders(tenant_id)
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders (order_date);
CREATE INDEX IF NOT EXISTS idx_tenants_tier      ON tenants (tier);

-- ---------------------------------------------------------------------------
-- 3. Application metadata schema (Section 7) - anonymized / auditable data only
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS optimizer_meta;

CREATE TABLE IF NOT EXISTS optimizer_meta.optimization_runs (
    request_id          TEXT PRIMARY KEY,
    query_fingerprint   TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    processing_status   TEXT NOT NULL,
    anonymized_sql      TEXT,
    plan_fingerprint    TEXT,
    recommendation_summary JSONB,
    sandbox_metrics     JSONB,
    blast_radius_metrics JSONB,
    model_modes         JSONB,
    error_category      TEXT
);

CREATE TABLE IF NOT EXISTS optimizer_meta.query_history (
    id                  BIGSERIAL PRIMARY KEY,
    query_fingerprint   TEXT NOT NULL,
    plan_fingerprint    TEXT,
    plan_shape          TEXT,
    timestamp_bucket    TIMESTAMPTZ,
    call_count          BIGINT,
    mean_exec_time_ms   DOUBLE PRECISION,
    estimated_cost      DOUBLE PRECISION,
    estimated_rows      DOUBLE PRECISION,
    actual_rows         DOUBLE PRECISION,
    operator_histogram  JSONB,
    access_features     JSONB,
    table_size_bucket   TEXT,
    write_frequency_bucket TEXT,
    anonymized_sql      TEXT,
    source              TEXT,
    recommendation_outcome TEXT,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_qh_fingerprint ON optimizer_meta.query_history (query_fingerprint, timestamp_bucket);

CREATE TABLE IF NOT EXISTS optimizer_meta.audit_events (
    id          BIGSERIAL PRIMARY KEY,
    request_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    stage       TEXT,
    severity    TEXT NOT NULL DEFAULT 'info',
    details     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_request ON optimizer_meta.audit_events (request_id, created_at);

CREATE TABLE IF NOT EXISTS optimizer_meta.migrations (
    migration_id      TEXT PRIMARY KEY,
    request_id        TEXT NOT NULL,
    query_fingerprint TEXT,
    migration_sql     TEXT NOT NULL,
    rollback_sql      TEXT NOT NULL,
    rationale         TEXT,
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,
    approval_status   TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 4. Least-privilege application role (Section 11 / R10)
--    Password is set by scripts/create_app_role.sh (docker) or seed_data.py
--    from QS_APP_PASSWORD; never hardcoded here.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qs_app') THEN
        CREATE ROLE qs_app LOGIN;
    END IF;
END$$;

ALTER ROLE qs_app SET statement_timeout = '15s';
ALTER ROLE qs_app SET idle_in_transaction_session_timeout = '60s';

REVOKE ALL ON SCHEMA public FROM qs_app;
GRANT USAGE ON SCHEMA public TO qs_app;
-- read-only on business tables: required for EXPLAIN; NO insert/update/delete/ddl
GRANT SELECT ON tenants, orders, order_items TO qs_app;

GRANT USAGE ON SCHEMA optimizer_meta TO qs_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA optimizer_meta TO qs_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA optimizer_meta TO qs_app;

-- pg_stat_statements query text of other roles is only visible with pg_read_all_stats
GRANT pg_read_all_stats TO qs_app;

-- HypoPG: hypothetical-index functions are executable by PUBLIC; the listing view
-- needs an explicit grant for the restricted role.
GRANT SELECT ON hypopg_list_indexes TO qs_app;
