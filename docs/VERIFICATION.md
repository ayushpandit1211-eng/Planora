# Verification — 9 October 2026

## Passed locally on Windows (native PostgreSQL)

- Full suite after deployment changes: **272 passed, 0 skipped** in 36.17 seconds.
- PostgreSQL 16.15 / HypoPG 1.4.3; separate disposable main and sandbox test instances.
- Production API rejects missing/incorrect keys, refuses missing startup secrets, and reports unavailable dependencies as HTTP 503. Eight added regression cases cover these behaviors.
- Real production-mode API run: authenticated readiness HTTP 200; optimization pipeline completed using the existing models and database.
- Streamlit production-mode render: no exceptions; backend URL and API-key widgets absent.
- Official Caddy 2.10.2 Windows archive checked against its published SHA-512 checksum; production Caddyfile validation passed.
- Live local gateway smoke check: absent/incorrect credentials HTTP 401; correct credentials proxied dashboard health with HTTP 200. This used loopback HTTP test traffic, not a public TLS certificate.
- Production Compose schema and environment interpolation validated. No published database, API or dashboard ports; dashboard does not receive the database administrator password.
- Native PostgreSQL custom-format backup restored into a fresh disposable database: 100 tenants, 100,000 orders, 300,000 order items verified; temporary restore database removed.
- All five supplied model files unchanged by SHA-256 comparison. See model-hashes.json.

## Not yet verified

- Actual Docker image build/container execution: local Docker engine is unavailable because virtualization is unavailable.
- GitHub Actions workflows: files are prepared but no destination repository/run is available yet.
- Public server, domain/DNS, ACME certificate issuance, remote browser/WebSocket session, host restart, scheduled/off-host backups and remote resource capacity.

The source is prepared for a private trusted-reviewer deployment. These remaining gates must pass on the chosen host before calling it fully deployed. It is not an anonymous public multi-tenant service.

Raw verification logs and JSON evidence are saved outside the public source package in the E: workspace's output/release-readiness folder. No credential files or database dumps are bundled in the GitHub ZIP.


## Planora rename verification — 10 October 2026

All 272 live tests passed (zero skipped) after renaming. The local dashboard renders Planora, and the API OpenAPI title is Planora API. All five trained model/artifact files match the original baseline by SHA-256. Compatibility identifiers and attributed historical material are intentionally preserved. Remote CI and public hosting remain pending.
