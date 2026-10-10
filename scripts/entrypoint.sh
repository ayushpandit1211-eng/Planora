#!/bin/sh
# Container entrypoint: wait for PostgreSQL, optionally seed (backend only), then exec CMD.
set -e

if [ -n "${POSTGRES_HOST:-}" ]; then
  echo "[entrypoint] waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT:-5432} ..."
  i=0
  until pg_isready -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; do
    i=$((i+1))
    if [ "$i" -ge 60 ]; then echo "[entrypoint] database not ready after 120s" >&2; exit 1; fi
    sleep 2
  done
fi

if [ "${QS_ROLE:-}" = "backend" ] && [ "${SEED_ON_START:-false}" = "true" ]; then
  python scripts/seed_data.py --if-empty
  python scripts/generate_workload.py --if-empty || echo "[entrypoint] workload generation skipped"
fi

exec "$@"
