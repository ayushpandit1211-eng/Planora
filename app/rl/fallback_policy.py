"""Deterministic fallback policy (Section 21 FALLBACK). policy_mode = "analytic_fallback".

Used when no trained policy exists. It ranks the SAME masked action set with transparent
rules; it is never reported as trained RL inference.
"""
from __future__ import annotations

from app.contracts import RLRecommendation
from app.rl.environment import LowAction
from app.rl.state_encoder import RLState

INDEX_KIND_PRIORITY = {"composite": 0, "covering": 1, "single": 2, "partial": 3}


def _index_rank(a: LowAction) -> tuple:
    roles = a.payload.get("roles", [])
    eq_first = 0 if roles and roles[0] in ("eq", "join", "in") else 1
    return (INDEX_KIND_PRIORITY.get(a.payload.get("kind", "single"), 9), eq_first, len(roles))


def choose(state: RLState, valid: dict[str, list[LowAction]]) -> tuple[LowAction, str]:
    if state.bottleneck_class == 3 and state.cost_bucket <= 2:
        return LowAction("NO_CHANGE", "none"), "plan is already efficient (bottleneck class 3, low cost)"
    if "REWRITE_SQL" in valid and (state.has_non_sargable or state.has_subplan):
        cands = [a for a in valid["REWRITE_SQL"] if a.verified] or valid["REWRITE_SQL"]
        return cands[0], "non-sargable predicate or correlated subquery: structural rewrite first"
    if "ADD_INDEX" in valid and state.bottleneck_class in (0, 1, 2):
        best = sorted(valid["ADD_INDEX"], key=_index_rank)[0]
        return best, "scan/join/sort bottleneck with an evidence-backed index candidate"
    if "PARTITION" in valid and state.partition_suitable:
        return valid["PARTITION"][0], "large table with range access pattern: partition advisory"
    return LowAction("NO_CHANGE", "none"), "no evidence-backed structural change"


def recommend(state: RLState, valid: dict[str, list[LowAction]], masked: dict[str, str],
              reason: str) -> RLRecommendation:
    action, why = choose(state, valid)
    return RLRecommendation(
        policy_mode="analytic_fallback", state_key=state.key(), high_level_action=action.high,
        low_level_action=action.signature if action.high != "NO_CHANGE" else None,
        selected_candidate=action.payload or None, q_values={},
        valid_actions=[f"{h}:{a.signature}" for h, lst in valid.items() for a in lst],
        masked_actions=masked,
        explanation=f"Deterministic fallback policy (no trained RL policy: {reason}). Rule: {why}.",
    )
