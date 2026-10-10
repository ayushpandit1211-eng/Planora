"""Metric provenance (Final Upgrade Section 7).

Every metric, label or recommendation produced by the new paths carries one or more provenance tags.
The tag travels with the value (`{"value": ..., "provenance": [...]}`) and the dashboard renders it as a badge.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class Provenance(str, Enum):
    POSTGRES_OBSERVED = "POSTGRES OBSERVED"
    PLANNER_ESTIMATE = "PLANNER ESTIMATE"
    HYPOPG_SIMULATION = "HYPOPG SIMULATION"
    MEASURED_SANDBOX = "MEASURED SANDBOX"
    ANALYTIC_FALLBACK = "ANALYTIC FALLBACK"
    TRAINED_GNN = "TRAINED GNN"
    RL_POLICY = "RL POLICY"
    LLM_RECOMMENDATION = "LLM RECOMMENDATION"
    HEURISTIC_ESTIMATE = "HEURISTIC ESTIMATE"
    SYNTHETIC_DATA = "SYNTHETIC DATA"
    DP_PROTECTED = "DP-PROTECTED"


ALL_TAGS = tuple(p.value for p in Provenance)


def _norm(tags: Iterable[Provenance | str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        v = t.value if isinstance(t, Provenance) else str(t).strip("[] ")
        if v not in ALL_TAGS:
            raise ValueError(f"unknown provenance tag: {v!r}")
        if v not in out:
            out.append(v)
    return out


def tagged(value: Any, *tags: Provenance | str, **extra: Any) -> dict[str, Any]:
    """Attach provenance to a value: {"value": v, "provenance": ["PLANNER ESTIMATE", ...], **extra}."""
    if not tags:
        raise ValueError("at least one provenance tag is required")
    return {"value": value, "provenance": _norm(tags), **extra}


def badge(tag: Provenance | str) -> str:
    """Text form used in sentences and the UI: '[PLANNER ESTIMATE]'."""
    return f"[{_norm([tag])[0]}]"


def has_provenance(item: Any) -> bool:
    return isinstance(item, dict) and bool(item.get("provenance")) and all(
        t in ALL_TAGS for t in item["provenance"])
