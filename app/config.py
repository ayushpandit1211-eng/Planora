"""Central configuration (pydantic-settings). All secrets come from the environment."""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "querysentinel"
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("")
    qs_app_user: str = "qs_app"
    qs_app_password: SecretStr = SecretStr("")
    db_pool_min_size: int = 1
    db_pool_max_size: int = 8
    db_connect_timeout_s: float = 10.0
    db_connect_retries: int = 5

    # Privacy
    privacy_hmac_key: SecretStr = SecretStr("")
    vault_ttl_seconds: int = 1800

    # Security gateway
    max_query_length: int = 20000
    statement_timeout_ms: int = 15000
    qs_api_key: SecretStr = SecretStr("")
    qs_deployment_mode: Literal["local", "production"] = "local"

    # LLM
    llm_provider: str = "none"  # none | gemini | openai | mock
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = ""
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1

    # Models
    gnn_checkpoint_path: str = "models/gnn_bottleneck.pt"
    rl_policy_path: str = "models/rl_policy.json"
    gnn_gat_checkpoint_path: str = "models/gnn_gatv2.pt"

    # Benchmark
    benchmark_enabled: bool = False
    benchmark_repetitions: int = 3
    # DSN of a DISPOSABLE copy of the database with DDL rights (measured mode only)
    benchmark_admin_dsn: SecretStr = SecretStr("")

    # Optimization budget (R9)
    budget_max_index_candidates: int = Field(8, validation_alias=AliasChoices("budget_max_index_candidates", "max_index_candidates"))
    budget_max_hypopg_experiments: int = Field(10, validation_alias=AliasChoices("budget_max_hypopg_experiments", "max_hypopg_experiments"))
    budget_max_agent_steps: int = Field(20, validation_alias=AliasChoices("budget_max_agent_steps", "max_agent_steps"))
    budget_max_llm_retries: int = Field(2, validation_alias=AliasChoices("budget_max_llm_retries", "max_llm_retries"))
    budget_max_workload_queries: int = Field(50, validation_alias=AliasChoices("budget_max_workload_queries", "max_historical_queries"))

    # Drift (R8)
    drift_js_threshold: float = 0.2

    # ------------------------------------------------------------------ Final upgrade (Section 10)
    # Feature flags. Additive routes default on; behaviour-changing features default off until
    # their phase gate passed (see .build_state/DECISIONS.md U-series).
    qs_feature_ingestion: bool = True
    qs_feature_nlq: bool = True
    qs_feature_gatv2: bool = False
    qs_feature_rl_upgrades: bool = False
    qs_feature_dp: bool = False
    qs_feature_index_portfolio: bool = False
    qs_metadata_schema: str = "qs_meta"
    # per_session (default): datasets are visible only to the browser session that created them.
    # single_user: one local user - every dashboard session sees every dataset (docker-compose default).
    qs_session_mode: str = "per_session"
    qs_run_meta_migration: bool = True
    budget_max_sandbox_benchmarks: int = Field(3, validation_alias=AliasChoices(
        "budget_max_sandbox_benchmarks", "max_sandbox_benchmarks"))

    # Optional isolated schema sandbox. Empty host means unavailable, never the main DB.
    sandbox_pg_host: str = ""
    sandbox_pg_port: int = 5432
    sandbox_pg_user: str = "postgres"
    sandbox_pg_password: SecretStr = SecretStr("")
    sandbox_secret: SecretStr = SecretStr("")
    sandbox_ttl_minutes: float = Field(60.0, gt=0)
    sandbox_cleanup_interval_seconds: float = Field(30.0, gt=0)
    sandbox_connection_limit: int = Field(5, ge=1)
    sandbox_max_active_per_session: int = Field(2, ge=1)
    sandbox_max_active_global: int = Field(10, ge=1)
    sandbox_max_db_size_mb: int = Field(512, ge=1)
    sandbox_max_extensions: int = Field(2, ge=0)
    sandbox_max_statements: int = Field(500, ge=1)
    sandbox_max_upload_mb: float = Field(10.0, gt=0)
    sandbox_statement_timeout_ms: int = Field(15000, ge=1)
    synthetic_rows_small: int = Field(1000, ge=1)
    synthetic_rows_medium: int = Field(10000, ge=1)
    synthetic_rows_large: int = Field(100000, ge=1)

    # Dynamic dataset ingestion
    ingest_target_schema: str = "qs_dynamic"
    ingest_max_upload_mb: float = 100.0
    ingest_sample_rows: int = 10000
    ingest_max_reject_ratio: float = 0.01
    ingest_allow_widening: bool = False
    ingest_null_tokens: str = ",NULL,null,NA,N/A,n/a,None,none,-,nan,NaN"

    # NL-to-SQL
    nlq_max_retries: int = 2
    nlq_max_rows: int = 1000
    nlq_statement_timeout_ms: int = 10000
    nlq_max_plan_cost: float = 5.0e7
    nlq_allowed_functions: str = ""          # extra allow-listed function names (comma separated)
    nlq_schema_context_max_tables: int = 15
    nlq_sample_values: bool = True

    # Differential privacy (AI-facing aggregates only)
    dp_epsilon_per_release: float = 0.5
    dp_epsilon_budget_total: float = 5.0
    dp_budget_scope: str = "session"          # session | global

    backend_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def app_dsn_kwargs(self) -> dict:
        """Connection kwargs for the restricted application role (never logged)."""
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "database": self.postgres_db,
            "user": self.qs_app_user,
            "password": self.qs_app_password.get_secret_value() or None,
        }

    @property
    def sandbox_admin_dsn_kwargs(self) -> dict:
        return {"host": self.sandbox_pg_host, "port": self.sandbox_pg_port,
                "database": "postgres", "user": self.sandbox_pg_user,
                "password": self.sandbox_pg_password.get_secret_value() or None}

    @property
    def single_user(self) -> bool:
        return self.qs_session_mode.strip().lower() == "single_user"

    def null_tokens(self) -> set[str]:
        return {t.strip() for t in self.ingest_null_tokens.split(",")} | {""}

    @property
    def admin_dsn_kwargs(self) -> dict:
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "database": self.postgres_db,
            "user": self.postgres_user,
            "password": self.postgres_password.get_secret_value() or None,
        }

    def hmac_key_bytes(self) -> bytes:
        key = self.privacy_hmac_key.get_secret_value()
        if key:
            return key.encode("utf-8")
        return _ephemeral_key()

    def redacted(self) -> dict:
        """Settings view safe to show in health/status output."""
        return {
            "postgres_host": self.postgres_host,
            "postgres_db": self.postgres_db,
            "app_role": self.qs_app_user,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model or None,
            "llm_api_key_configured": bool(self.llm_api_key.get_secret_value()),
            "vault_backend": "memory",
            "hmac_key_configured": bool(self.privacy_hmac_key.get_secret_value()),
            "benchmark_enabled": self.benchmark_enabled,
        }


@lru_cache(maxsize=1)
def _ephemeral_key() -> bytes:
    return secrets.token_bytes(32)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


os.environ.setdefault("PYTHONHASHSEED", "0")
