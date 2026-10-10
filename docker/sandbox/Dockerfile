# Planora BYOD sandbox server (Final Upgrade P14): PostgreSQL 16 + HypoPG, NO demo init scripts.
# Reachable only on the internal `sandbox_net` network; one database + one role per sandbox, TTL-reaped.
FROM postgres:16-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-hypopg \
 && rm -rf /var/lib/apt/lists/*
