#!/bin/bash
# Sets the restricted application role password from the environment (never hardcoded).
set -euo pipefail
if [ -n "${QS_APP_PASSWORD:-}" ]; then
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
       -v pw="$QS_APP_PASSWORD" <<'EOSQL'
ALTER ROLE qs_app PASSWORD :'pw';
EOSQL
fi
