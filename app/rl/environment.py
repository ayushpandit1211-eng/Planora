"""RL environment: state -> action -> sandbox evaluation -> next state + reward (Section 21, R7).

Hierarchical action space
  HIGH: ADD_INDEX | REWRITE_SQL | PARTITION | SHARD_ADVISORY | NO_CHANGE
  LOW : ADD_INDEX      -> a validated index candidate (single/composite/partial/covering)
        REWRITE_SQL    -> a safe rewrite operator exposed by the SQL rewriter
        PARTITION      -> a validated range/hash partition advisory
        SHARD_ADVISORY -> a shard-key advisory
Invalid-action masking (R7): candidates come only from anonymised schema/workload evidence;
nonexistent identifiers, duplicate/redundant indexes and safety-policy violations are removed
BEFORE the policy sees the action set.

Reward (documented formulation, R7):
  reward = w_freq * normalized_planner_cost_reduction
           - storage_penalty - write_penalty - redundancy_penalty - risk_penalty
  normalized_planner_cost_reduction = (cost_before_step - cost_after_step) / baseline_cost
  w_freq = 1 + log10(1 + workload calls for the query fingerprint)   (1.0 without history)
  storage_penalty    = 0.10 * min(1, estimated_index_bytes / table_bytes)
  write_penalty      = 0.04 * index_columns * write_factor  (low 0.25, medium 0.6, high 1.0)
  redundancy_penalty = 0.30 if redundant, 0.02 if the action changed planner cost by < 1 %
  risk_penalty       = PARTITION 0.05, SHARD_ADVISORY 0.08, rewrite 0.02 (verified) / 0.20 (unverified)
PARTITION / SHARD_ADVISORY receive NO cost benefit because HypoPG cannot validate them; they are
advisory only. The reward measures planner-estimated benefit, not real user value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rl.state_encoder import RLState

HIGH_ACTIONS = ["ADD_INDEX", "REWRITE_SQL", "PARTITION", "SHARD_ADVISORY", "NO_CHANGE"]
WRITE_FACTOR = {0: 0.25, 1: 0.6, 2: 1.0}
MAX_STEPS = 3


@dataclass
class LowAction:
    high: str
    signature: str                     # abstract, identifier-free (used as Q key)
    payload: dict[str, Any] = field(default_factory=dict)   # candidate details (pseudonymised)
    redundant: bool = False
    verified: bool = True              # rewrite semantics verified by the semantic diff


@dataclass
class StepOutcome:
    cost_before: float | None
    cost_after: float | None
    storage_bytes: float | None = None
    table_bytes: float | None = None
    n_columns: int = 0
    evaluated: bool = True
    note: str = ""


class Evaluator(Protocol):
    """Sandbox used by the environment. Implementations: HypoPG (live) or a test double."""
    def baseline_cost(self) -> float: ...
    def apply(self, action: LowAction) -> StepOutcome: ...
    def reset(self) -> None: ...


@dataclass
class EpisodeContext:
    state: RLState
    actions: list[LowAction]
    evaluator: Evaluator
    frequency_weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


def mask_actions(actions: list[LowAction], taken: set[str]) -> tuple[dict[str, list[LowAction]], dict[str, str]]:
    """Group valid low-level actions by high-level action; report why others were masked."""
    valid: dict[str, list[LowAction]] = {h: [] for h in HIGH_ACTIONS}
    masked: dict[str, str] = {}
    for a in actions:
        key = f"{a.high}:{a.signature}:{a.payload.get('id', '')}"
        if a.redundant:
            masked[key] = "redundant with an existing index"
            continue
        if key in taken:
            masked[key] = "already applied in this episode"
            continue
        valid[a.high].append(a)
    valid["NO_CHANGE"] = [LowAction("NO_CHANGE", "none")]
    for h in HIGH_ACTIONS:
        if not valid[h]:
            masked.setdefault(h, "no valid candidate for this action")
    return {h: v for h, v in valid.items() if v}, masked


def compute_reward(action: LowAction, outcome: StepOutcome, baseline: float, state: RLState,
                   frequency_weight: float) -> tuple[float, dict[str, float]]:
    parts = {"benefit": 0.0, "storage_penalty": 0.0, "write_penalty": 0.0,
             "redundancy_penalty": 0.0, "risk_penalty": 0.0}
    if action.high == "NO_CHANGE":
        return 0.0, parts
    if action.redundant:
        parts["redundancy_penalty"] = 0.30
    if action.high in ("ADD_INDEX", "REWRITE_SQL") and outcome.evaluated and baseline > 0 \
            and outcome.cost_before is not None and outcome.cost_after is not None:
        delta = (outcome.cost_before - outcome.cost_after) / baseline
        parts["benefit"] = frequency_weight * max(min(delta, 1.0), -1.0)
        if abs(outcome.cost_before - outcome.cost_after) < 0.01 * max(outcome.cost_before, 1e-9):
            parts["redundancy_penalty"] += 0.02
    if action.high == "ADD_INDEX":
        if outcome.storage_bytes and outcome.table_bytes:
            parts["storage_penalty"] = 0.10 * min(1.0, outcome.storage_bytes / outcome.table_bytes)
        parts["write_penalty"] = 0.04 * max(outcome.n_columns, 1) * WRITE_FACTOR.get(state.write_frequency_bucket, 0.25)
    elif action.high == "REWRITE_SQL":
        parts["risk_penalty"] = 0.02 if action.verified else 0.20
    elif action.high == "PARTITION":
        parts["risk_penalty"] = 0.05
    elif action.high == "SHARD_ADVISORY":
        parts["risk_penalty"] = 0.08
    reward = parts["benefit"] - parts["storage_penalty"] - parts["write_penalty"] \
        - parts["redundancy_penalty"] - parts["risk_penalty"]
    return round(reward, 6), {k: round(v, 6) for k, v in parts.items()}


class OptimizationEnv:
    def __init__(self, ctx: EpisodeContext, max_steps: int = MAX_STEPS):
        self.ctx = ctx
        self.max_steps = max_steps
        self.state = ctx.state
        self.steps = 0
        self.taken: set[str] = set()
        self.baseline = 0.0
        self.current_cost = 0.0
        self.history: list[dict[str, Any]] = []

    def reset(self) -> RLState:
        self.ctx.evaluator.reset()
        self.state = RLState(**{**self.ctx.state.__dict__})
        self.steps = 0
        self.taken.clear()
        self.baseline = self.ctx.evaluator.baseline_cost()
        self.current_cost = self.baseline
        self.history = []
        return self.state

    def valid_actions(self) -> tuple[dict[str, list[LowAction]], dict[str, str]]:
        return mask_actions(self.ctx.actions, self.taken)

    def step(self, action: LowAction) -> tuple[RLState, float, bool, dict[str, Any]]:
        self.steps += 1
        if action.high == "NO_CHANGE":
            outcome = StepOutcome(self.current_cost, self.current_cost, evaluated=False)
        else:
            outcome = self.ctx.evaluator.apply(action)
            self.taken.add(f"{action.high}:{action.signature}:{action.payload.get('id', '')}")
        reward, parts = compute_reward(action, outcome, self.baseline, self.state, self.ctx.frequency_weight)
        if action.high in ("ADD_INDEX", "REWRITE_SQL") and outcome.cost_after is not None:
            self.current_cost = outcome.cost_after
        nxt = RLState(**{**self.state.__dict__})
        if action.high == "ADD_INDEX":
            nxt.indexes_added = min(nxt.indexes_added + 1, 3)
            if outcome.cost_after is not None:
                from app.rl.state_encoder import _bucket_log10
                nxt.cost_bucket = _bucket_log10(outcome.cost_after)
            nxt.existing_index_coverage = max(nxt.existing_index_coverage, 1)
        elif action.high == "REWRITE_SQL":
            nxt.has_non_sargable = 0
            nxt.has_subplan = 0 if "subquery" in action.signature else nxt.has_subplan
        valid_next, _ = mask_actions(self.ctx.actions, self.taken)
        done = action.high == "NO_CHANGE" or self.steps >= self.max_steps or list(valid_next) == ["NO_CHANGE"]
        info = {"reward_parts": parts, "cost_before": outcome.cost_before, "cost_after": outcome.cost_after,
                "evaluated": outcome.evaluated, "note": outcome.note,
                "planner_cost_reduction_pct": None if not outcome.cost_before or outcome.cost_after is None
                else round((outcome.cost_before - outcome.cost_after) / outcome.cost_before * 100, 2)}
        self.history.append({"action": f"{action.high}:{action.signature}", "reward": reward, **info})
        self.state = nxt
        return nxt, reward, done, info

    def cumulative_reduction_pct(self) -> float | None:
        if not self.baseline:
            return None
        return round((self.baseline - self.current_cost) / self.baseline * 100, 2)


def frequency_weight_from_calls(calls: int | None) -> float:
    return 1.0 + math.log10(1 + max(calls or 0, 0))
