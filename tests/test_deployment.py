"""Production startup and authenticated readiness must fail closed."""
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

from app.api import create_app
from app.config import reset_settings_cache


@pytest.fixture
def production_env(monkeypatch):
    monkeypatch.setenv("QS_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("QS_API_KEY", "a" * 40)
    monkeypatch.setenv("PRIVACY_HMAC_KEY", "h" * 40)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.mark.parametrize("key", ["QS_API_KEY", "PRIVACY_HMAC_KEY"])
def test_production_rejects_missing_secrets(production_env, monkeypatch, fake_services, key):
    monkeypatch.setenv(key, "")
    reset_settings_cache()
    with pytest.raises(ValueError, match=key):
        create_app(fake_services, connect=False)


def healthy():
    data = {name: {"ok": True} for name in
            ("postgresql", "hypopg", "privacy_vault", "langgraph")}
    return {**data, "pg_stat_statements": {"ok": True, "preloaded": True},
            "gnn": {"mode": "trained"}, "rl": {"mode": "trained_policy"}}


def test_readiness_requires_auth(production_env, fake_services):
    fake_services.status = AsyncMock(return_value=healthy())
    with TestClient(create_app(fake_services, connect=False)) as client:
        assert client.get("/api/v1/ready").status_code == 401
        assert client.get("/api/v1/ready", headers={"x-api-key": "wrong"}).status_code == 401
        assert client.get("/api/v1/ready", headers={"x-api-key": "a" * 40}).status_code == 200


@pytest.mark.parametrize("component,field,value", [
    ("postgresql", "ok", False), ("hypopg", "ok", False),
    ("pg_stat_statements", "preloaded", False), ("gnn", "mode", "analytic_fallback"),
    ("rl", "mode", "analytic_fallback"),
])
def test_readiness_rejects_degraded_core(production_env, fake_services, component, field, value):
    status = healthy()
    status[component][field] = value
    fake_services.status = AsyncMock(return_value=status)
    with TestClient(create_app(fake_services, connect=False)) as client:
        result = client.get("/api/v1/ready", headers={"x-api-key": "a" * 40})
        assert result.status_code == 503
        assert result.json()["checks"][component] is False
