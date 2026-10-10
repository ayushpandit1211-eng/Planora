# PostgreSQL 16 + HypoPG + pg_stat_statements (contrib, preloaded via compose command)
FROM postgres:16-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-hypopg \
 && rm -rf /var/lib/apt/lists/*
COPY scripts/create_extensions.sql /docker-entrypoint-initdb.d/01_create_extensions.sql
COPY scripts/init_db.sql           /docker-entrypoint-initdb.d/02_init_db.sql
COPY docker/db/03_app_role.sh      /docker-entrypoint-initdb.d/03_app_role.sh
