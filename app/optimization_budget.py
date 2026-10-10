"""Optimization budget controller (Addendum R9).

Bounds the search: index candidates, HypoPG experiments, agent steps, LLM retries and workload
queries per optimisation run. When a budget is exhausted a SEARCH_BUDGET_EXHAUSTED event is
emitted and the report states that exploration was bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import get_settings

BUDGET_KEYS = ("index_candidates", "hypopg_experiments", "agent_steps", "llm_retries", "workload_queries",
               "sandbox_benchmarks")


@dataclass
class OptimizationBudget:
    limits: dict[str, int]
    used: dict[str, int] = field(default_factory=lambda: {k: 0 for k in BUDGET_KEYS})
    exhausted: set[str] = field(default_factory=set)
    on_exhausted: Callable[[str, dict[str, Any]], None] | None = None

    @classmethod
    def from_settings(cls, on_exhausted=None) -> "OptimizationBudget":
        s = get_settings()
        return cls(limits={"index_candidates": s.budget_max_index_candidates,
                           "hypopg_experiments": s.budget_max_hypopg_experiments,
                           "agent_steps": s.budget_max_agent_steps,
                           "llm_retries": s.budget_max_llm_retries,
                           "workload_queries": s.budget_max_workload_queries,
                           "sandbox_benchmarks": s.budget_max_sandbox_benchmarks},
                   on_exhausted=on_exhausted)

    def __post_init__(self) -> None:
        # budgets built with an older key set (e.g. without sandbox_benchmarks) stay valid
        for k in BUDGET_KEYS:
            self.used.setdefault(k, 0)

    def remaining(self, key: str) -> int:
        return max(self.limits.get(key, 0) - self.used[key], 0)

    def consume(self, key: str, n: int = 1) -> bool:
        """Consume `n` units; returns False (and records exhaustion) if not enough remain."""
        if self.used[key] + n > self.limits.get(key, 0):
            if key not in self.exhausted:
                self.exhausted.add(key)
                if self.on_exhausted:
                    self.on_exhausted(key, self.snapshot())
            return False
        self.used[key] += n
        return True

    def take(self, key: str, items: list) -> list:
        """Return the prefix of `items` that fits in the remaining budget."""
        room = self.remaining(key)
        if len(items) > room:
            self.consume(key, len(items))   # triggers exhaustion record
            self.used[key] = self.limits.get(key, 0)
            return items[:room]
        self.used[key] += len(items)
        return items

    def snapshot(self) -> dict[str, Any]:
        return {"limits": dict(self.limits), "used": dict(self.used),
                "remaining": {k: self.remaining(k) for k in BUDGET_KEYS},
                "exhausted": sorted(self.exhausted),
                "statement": ("Exploration was bounded by the optimization budget: "
                              + ", ".join(sorted(self.exhausted))) if self.exhausted else
                             "All exploration completed within budget."}
