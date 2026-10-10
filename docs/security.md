# Planora - Security & Privacy

Planora is a prototype. This document states exactly what is protected and what remains
outside its guarantees. **It does not claim absolute security.**

## 1. Threat model

| Asset | Threat | Mitigation |
|---|---|---|
| Production data values (emails, UUIDs, amounts, names) | Disclosure to an external LLM | AST parameterisation removes every literal; PII/secret scan; outgoing payload guard blocks raw literals, PII, secrets and real identifiers; LLM disabled by default |
| Schema names (business-sensitive) | Disclosure to an external LLM | HMAC-SHA256 pseudonyms; reverse map server-side only; guard checks for real identifiers |
| Database integrity | Malicious/accidental writes via submitted SQL | Single `SelectStmt` gateway; READ ONLY transaction; least-privilege role without write/DDL rights; no migration execution path |
| Database availability | Expensive queries / long EXPLAIN | `statement_timeout` (session + `SET LOCAL`); EXPLAIN never ANALYZE on the analysis path; query length limit; optimisation budget |
| Server filesystem / OS | `COPY`, `pg_read_file`, `lo_import`, `dblink` | Statement-type and function denylist; role is not superuser |
| Credentials | Leakage via logs, errors, API, dashboard | Secrets only from env (`SecretStr`); redacting JSON log formatter; structured errors without stack traces; `/status` returns a redacted settings view |
| LLM output | Prompt-injected / hallucinated DDL or SQL | Pydantic schema validation; unknown pseudonyms rejected; validator rebuilds DDL from whitelisted parts; semantic diff on rewrites; sandbox only via HypoPG; human review |
| Agents | Capability escalation | Capability firewall with explicit per-agent capabilities, no production-write capability, audited denials |

## 2. Trust boundaries

1. **Browser ↔ API** - the dashboard holds the user's own raw SQL in its session; the API
   never returns the reverse mapping. Optional `X-API-Key` (`QS_API_KEY`). Deploy behind TLS.
2. **API/pipeline (trusted) ↔ LLM (untrusted)** - only anonymised SQL, anonymised schema
   metadata, sanitised plan structure, workload-frequency summaries, GNN diagnosis and RL state
   features may cross, and only after `OutgoingPayloadGuard.inspect()`.
3. **Pipeline ↔ PostgreSQL** - least-privilege role `qs_app`, read-only transactions.
4. **Graph state ↔ trusted store** - LangGraph state contains anonymised data only; raw SQL,
   real schema and real DDL live in `Services.trusted[request_id]` and are dropped at the end.

## 3. LLM boundary

Every prompt carries the privacy contract: *"You receive anonymized SQL and anonymized metadata
only. Do not request raw values. Do not request secrets. Do not request credentials. Do not invent
schema. Do not claim measured performance without measurement. Separate observations,
assumptions, recommendations, and risks."* The provider is environment-configurable
(`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`); API keys are sent in headers, never
URLs. Replies must validate against `LLMRecommendation`; malformed JSON, timeouts and HTTP errors
become structured `llm_*` error categories and the deterministic agents are used.

## 4. PII handling

Detected (types and counts only, never values): email, UUID, phone, IPv4, Luhn-valid card-like
numbers; secrets: JWT, API keys (OpenAI/Stripe/AWS/Google/GitHub/Slack/GitLab patterns), password
assignments, private keys, credentialed connection strings, bearer tokens, long hex secrets.
A query is **not** rejected merely because it contains a UUID/email literal - it is parameterised.
Quarantine happens only when safe anonymisation cannot be established (AST failure plus unsafe
constructs, or residual PII/literals after masking).

## 5. Secret handling

`.env` (git-ignored) or the environment; pydantic `SecretStr`; never logged; compose refuses to
start without `POSTGRES_PASSWORD`/`QS_APP_PASSWORD`; the app-role password is set by an init
script from the environment; `PRIVACY_HMAC_KEY` should be set in any shared deployment (otherwise
a random per-process key is used).

## 6. SQL validation

See `app/security.py`: exactly one statement; `SelectStmt` only (read-only CTEs allowed); rejects
DDL, DML, `COPY`, `SET`, `ALTER SYSTEM`, `VACUUM/ANALYZE`, `EXPLAIN` from the user, transaction
control, `DO`, `CALL`, role/privilege commands, `LOCK`, `NOTIFY`, `SELECT INTO`, row locking,
data-modifying CTEs, credential catalogs (`pg_authid`, `pg_shadow`, ...), and dangerous functions
(file/large-object access, sleep, backend signalling, `set_config`, `current_setting`, `dblink`,
`nextval/setval`, advisory locks, `hypopg_*`, stats reset, ...). Parse errors never echo the input.

## 7. Least-privilege database role

`scripts/init_db.sql`: `qs_app` LOGIN with `statement_timeout=15s`,
`idle_in_transaction_session_timeout=60s`; `USAGE` on `public`; `SELECT` only on `tenants`,
`orders`, `order_items`; `SELECT/INSERT/UPDATE` on `optimizer_meta`; `pg_read_all_stats` (to read
other roles' `pg_stat_statements` text); `SELECT` on `hypopg_list_indexes`. Verified by
`tests/test_integration.py`: INSERT into a business table and CREATE INDEX are denied, writes in a
read-only transaction are denied. Note: `pg_stats.most_common_vals`/`histogram_bounds` (raw data)
are never selected by the application.

## 8. Logging policy

Structured JSON logs with `request_id`, `query_fingerprint`, `stage`, `duration_ms`, `model_mode`,
`sandbox_status`, `error_category`. Never logged: raw literals, credentials, API keys, full
sensitive SQL. A redaction filter (quoted literals, emails, UUIDs, JWTs, API keys, DSNs,
`password=`) is applied to every message as a last line of defence; exception logs record the
exception type only.

## 9. Vault isolation

Request-scoped mappings keyed by request ID with TTL; per-record vaults for ingestion; cleared in
`finally` blocks; in-memory backend (single backend worker). The reverse map is never part of any API response, event, log or LLM payload.

## 10. Model risks

- GNN trained on synthetic plans - may misclassify unusual real plans; disagreement with rules is
  surfaced. RL trained on a small template set - recommendations are always validated by HypoPG
  and require review. LLM output may be wrong or adversarial - it is treated as untrusted input.
- Planner estimates can be wrong (stale statistics, correlated predicates); runtime must be
  measured in staging.

## 11. Residual risk (outside the prototype's guarantees)

- Pattern-based PII detection can miss unusual formats in identifiers or comments (literals are
  still removed by AST parameterisation; comments are dropped by deparsing).
- Pseudonyms are deterministic per key: an attacker who knows the key and candidate names can
  confirm guesses. Protect `PRIVACY_HMAC_KEY`.
- Query *structure* (joins, predicate shapes, table counts) is visible to the LLM by design.
- EXPLAIN can still consume planner CPU for pathological queries within the timeout.
- The dashboard has no user authentication beyond the optional API key; run it on a trusted network.
- `pg_read_all_stats` lets `qs_app` read all normalised statement texts in `pg_stat_statements`
  (no literals, but structure of other workloads).
- Measured benchmark mode builds real indexes in a disposable database; pointing
  `BENCHMARK_ADMIN_DSN` at production would be unsafe (documented, not technically prevented).
