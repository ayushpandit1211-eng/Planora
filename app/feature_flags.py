"""Feature flags for the Final Upgrade (Section 10). Read from Settings; reported by /api/v1/upgrade/status."""
from __future__ import annotations

from typing import Any

from app.config import get_settings

FLAG_FIELDS = {
    "ingestion": "qs_feature_ingestion",
    "nlq": "qs_feature_nlq",
    "gatv2": "qs_feature_gatv2",
    "rl_upgrades": "qs_feature_rl_upgrades",
    "dp": "qs_feature_dp",
    "index_portfolio": "qs_feature_index_portfolio",
}


def enabled(name: str) -> bool:
    return bool(getattr(get_settings(), FLAG_FIELDS[name]))


def snapshot() -> dict[str, Any]:
    s = get_settings()
    return {name: bool(getattr(s, field)) for name, field in FLAG_FIELDS.items()}
