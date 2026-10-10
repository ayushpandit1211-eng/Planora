# Deploy Planora as a private review/demo service

This deployment targets one Linux x64 Docker host and a DNS hostname. It preserves the existing features and models. It is a private demonstration for the owner and trusted reviewers, not a public multi-tenant SaaS platform. Session IDs are not user authentication; shared access means reviewers belong to the same trust boundary.

## What the deployment provides

- Caddy gates every browser request with a hashed-password login and provisions HTTPS for the configured domain.
- Only ports 80/443 are published. PostgreSQL, the backend API, and Streamlit have no host port mappings.
- The backend requires a generated API key; the dashboard stores that key server-side and does not expose an editable backend destination or key widget in production mode.
- Named volumes retain the database and TLS certificates across container recreation. Services restart after host reboot when Docker starts.
- Readiness requires PostgreSQL, HypoPG, preloaded pg_stat_statements, the privacy vault, LangGraph, and both supplied trained models.
- One backend worker is intentional: pipeline events, reports, and privacy-vault state are in memory. Do not scale workers or replicas without adding shared state.

The existing ingestion/migration functionality needs backend administrator database credentials. They are restricted to the private deployment and must never point at a production business database. Use synthetic/demo data here. The optional LLM remains disabled; open internet egress is not provided to the backend's internal networks. All original local-mode features remain available.

## First deployment

1. Provision your chosen Linux host with Docker Engine and the Compose plugin. Point the domain's A/AAAA records at it. Allow inbound TCP 80/443; restrict SSH to your administrator access. Do not open 5432, 8000 or 8501.
2. Copy or clone this repository to the host. Use `output/Planora` as the source repository root, not the entire parent workspace. Do not upload local `.env`, databases, backups or the Python virtual environment.
3. Create a temporary operator Python virtual environment and install the deployment tool dependency:

```bash
python3 -m venv .operator-venv
.operator-venv/bin/pip install -r requirements-deploy.txt
.operator-venv/bin/python scripts/configure_production.py --domain demo.YOUR-DOMAIN.com --email YOUR-EMAIL --user reviewer
```

Enter a unique 16-72-byte demo login password when prompted. The tool hashes it and independently generates database/API/HMAC secrets in `.env.production`, with restrictive Unix file permissions. It does not print secrets or replace an existing environment file. Store the login password in your password manager; back up the environment file privately.

4. Validate and start:

```bash
docker compose --env-file .env.production -f compose.production.yml config --quiet
docker compose --env-file .env.production -f compose.production.yml up --build -d --wait --wait-timeout 600
docker compose --env-file .env.production -f compose.production.yml exec -T backend python scripts/check_ready.py
```

Do not run ordinary `docker compose config` in shared logs because resolved configuration contains secrets. Image build and initial data seeding can take several minutes. Persistent volumes seed only when empty.

5. Open `https://YOUR-DOMAIN` and log in. Confirm unauthenticated requests receive 401 and HTTPS has a trusted certificate. Run scenario C, inspect HypoPG results, upload the synthetic railway CSV, execute a supported question, and test browser refresh. Confirm a new unauthorized browser cannot view the dashboard. Check the WebSocket stays connected and server logs have no errors.
6. Verify persistence after a container restart and complete a backup/restore drill below. A passing local suite alone does not prove this remote deployment works.

## Backups and restore checks

```bash
python scripts/backup_database.py
python scripts/verify_backup.py backups/NAME-OF-BACKUP.dump
```

The restore verifier uses a fresh randomly named database, checks a restored table, and removes only that temporary database. It never restores over the active database. Schedule backup_database.py using the host scheduler, retain an appropriate number of backups, and copy them to private off-host storage. The source package does not automatically configure an external storage account or retention policy.

Keep `.env.production` and the matching source revision alongside the encrypted/off-host backups. On a fresh host first initialize the same database roles with the saved environment. For disaster recovery, stop backend/dashboard, restore the selected dump to a new database using `pg_restore`, inspect it, then change the database configuration deliberately. Do not use `down -v` on the live host: it deletes persistent volumes.

## Updates and rollback

Record the deployed Git commit, run a backup, and require both GitHub workflows to pass before upgrading. Keep the previous source revision available. Rebuild/recreate with the same `.env.production` and volumes. If an update fails, return to the previous source revision and rebuild; schema compatibility must be checked before rollback. Restore database state only as a deliberate recovery operation after inspecting the backup. Passwords in an existing PostgreSQL volume do not automatically rotate when you edit the environment file.

## CI gates

`tests.yml` runs offline and live PostgreSQL tests using separate main/sandbox instances and rejects skipped integration tests. `containers.yml` builds the actual images, starts the deployment stack, verifies gateway authentication, checks readiness/private ports, restores a backup, and checks restart persistence. Its loopback HTTP setup is a disposable CI-only substitute for public DNS/TLS; a real host must still pass the HTTPS/browser checks above.

## Current verification boundary

See VERIFICATION.md. Docker is unavailable on the current Windows host because virtualization is unavailable. Container execution, remote GitHub Actions, DNS, public certificates, and a live host cannot be marked verified until an actual repository and deployment host are provided. Do not claim the application is publicly deployed based on these files alone.

References: [Caddy authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth), [automatic HTTPS requirements](https://caddyserver.com/docs/automatic-https), [WebSocket proxying](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).
