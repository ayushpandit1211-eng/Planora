"""Differential privacy for AI-facing numeric aggregates (Final Upgrade P10, QS_FEATURE_DP).

Mechanism: Laplace, M(D) = clip(f(D), lower, upper) + Laplace(0, sensitivity / epsilon), then post-processing
(rounding, clamp >= lower) which does not weaken the guarantee.
Neighbouring datasets: differ by adding or removing ONE row (row-count style releases, sensitivity 1) or ONE
query execution (workload counts, sensitivity 1). Values are clipped to [lower, upper] first, so a release with
declared sensitivity s is valid only under these assumptions.
Composition: sequential, epsilon_total = sum(epsilon_i) per budget scope (session or global).
Claim: epsilon-DP for these specific aggregate releases under the stated assumptions - NOT a guarantee for the
whole system. Structural facts (plan shapes, pseudonymised identifiers) are never noised. Trusted local validation
(planner, HypoPG, sandbox) always uses exact values; only the AI-facing payload is noised.
"""
from __future__ import annotations

import math
import random
import secrets
import threading
from typing import Any

from app.config import get_settings

_SYS = secrets.SystemRandom()          # cryptographically sound source
_TEST_RNG: random.Random | None = None


def set_test_rng(rng: random.Random | None) -> None:
    """Seeding is allowed in tests only."""
    global _TEST_RNG
    _TEST_RNG = rng


def _uniform() -> float:
    return (_TEST_RNG or _SYS).random()


def laplace(scale: float) -> float:
    u = _uniform() - 0.5
    while u == -0.5:                    # avoid log(0)
        u = _uniform() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


class BudgetExhaustedError(Exception):
    pass


class DPAccountant:
    def __init__(self, total: float | None = None):
        self._total = total
        self._spent: dict[str, float] = {}
        self._releases: dict[str, int] = {}
        self._exhausted_logged: set[str] = set()
        self._lock = threading.Lock()
        self.audit_hook = None             # callable(scope, snapshot) for PRIVACY_BUDGET_EXHAUSTED

    @property
    def total(self) -> float:
        return float(self._total if self._total is not None else get_settings().dp_epsilon_budget_total)

    def charge(self, scope: str, eps: float) -> None:
        with self._lock:
            spent = self._spent.get(scope, 0.0)
            if spent + eps > self.total + 1e-12:
                first = scope not in self._exhausted_logged
                self._exhausted_logged.add(scope)
                if first and self.audit_hook:
                    self.audit_hook(scope, self.snapshot(scope, locked=True))
                raise BudgetExhaustedError(scope)
            self._spent[scope] = spent + eps
            self._releases[scope] = self._releases.get(scope, 0) + 1

    def exhausted(self, scope: str) -> bool:
        return scope in self._exhausted_logged

    def snapshot(self, scope: str = "global", locked: bool = False) -> dict[str, Any]:
        spent = self._spent.get(scope, 0.0)
        return {"scope": scope, "epsilon_spent": round(spent, 6), "epsilon_total": self.total,
                "epsilon_remaining": round(max(self.total - spent, 0.0), 6), "releases": self._releases.get(scope, 0),
                "exhausted": scope in self._exhausted_logged,
                "composition": "sequential (sum of epsilons)",
                "claim": "epsilon-DP for these specific aggregate releases under the stated assumptions"}


accountant = DPAccountant()


def validate_params(epsilon: float, sensitivity: float, lower: float, upper: float | None) -> None:
    if not (isinstance(epsilon, (int, float)) and epsilon > 0 and math.isfinite(epsilon)):
        raise ValueError("epsilon must be a positive finite number")
    if not (isinstance(sensitivity, (int, float)) and sensitivity > 0 and math.isfinite(sensitivity)):
        raise ValueError("sensitivity must be a positive finite number")
    if upper is not None and upper < lower:
        raise ValueError("upper bound must be >= lower bound")


def dp_release(name: str, value: float, *, sensitivity: float = 1.0, lower: float = 0.0,
               upper: float | None = None, scope: str = "global", epsilon: float | None = None,
               accountant: DPAccountant | None = None) -> float | None:
    """Noisy release of one bounded numeric aggregate. Returns None (send NO aggregate) once the budget is
    exhausted - the caller then falls back to structural / AST-masked information only."""
    eps = float(epsilon if epsilon is not None else get_settings().dp_epsilon_per_release)
    validate_params(eps, sensitivity, lower, upper)
    acc = accountant or globals()["accountant"]
    try:
        acc.charge(scope, eps)
    except BudgetExhaustedError:
        return None
    clipped = min(max(float(value), lower), upper if upper is not None else float("inf"))
    noisy = clipped + laplace(sensitivity / eps)
    return float(max(round(noisy), lower))        # post-processing


def protect_schema_payload(schema: dict[str, Any] | None, scope: str = "global",
                           acc: DPAccountant | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """AI-facing copy of the anonymised schema with numeric COUNT-like aggregates DP-protected.
    Structure (names, types, keys, buckets) is unchanged. Returns (payload, report)."""
    import copy
    if not schema:
        return schema, {"released": 0}
    out = copy.deepcopy(schema)
    n = 0
    for t in (out.get("tables") or {}).values():
        t["estimated_rows"] = dp_release("estimated_rows", t.get("estimated_rows") or 0, upper=1e12, scope=scope,
                                         accountant=acc)
        n += 1
        for c in t.get("columns", []):
            nd = c.get("n_distinct")
            if isinstance(nd, (int, float)) and nd > 0:          # absolute distinct count (negative = ratio)
                c["n_distinct"] = dp_release("n_distinct", nd, upper=1e12, scope=scope, accountant=acc)
                n += 1
    for ix in out.get("indexes", []):
        if ix.get("idx_scan") is not None:
            ix["idx_scan"] = dp_release("idx_scan", ix["idx_scan"], upper=1e12, scope=scope, accountant=acc)
            n += 1
    snap = (acc or accountant).snapshot(scope)
    return out, {"released": n, "provenance": ["DP-PROTECTED"], **snap}
