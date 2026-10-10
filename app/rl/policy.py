"""RL policy facade used by the pipeline: trained policy if available, else deterministic
fallback - and the mode is always reported (Section 69)."""
from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.contracts import RLRecommendation
from app.errors import QuerySentinelError
from app.rl import fallback_policy
from app.rl.agent import QLearningAgent
from app.rl.environment import LowAction, mask_actions
from app.rl.state_encoder import RLState


class RLPolicy:
    def __init__(self, policy_path: str | None = None):
        s = get_settings()
        self.path = policy_path or str(s.resolve_path(s.rl_policy_path))
        self.agent: QLearningAgent | None = None
        self.load_error: str | None = None
        try:
            self.agent = QLearningAgent.load_policy(self.path)
        except QuerySentinelError as exc:
            self.load_error = f"{exc.category.value}: {exc.message}"

    @property
    def mode(self) -> str:
        return "trained_policy" if self.agent is not None else "analytic_fallback"

    def status(self) -> dict[str, Any]:
        meta = self.agent.meta if self.agent else {}
        return {"mode": self.mode, "policy_path": self.path, "error": self.load_error,
                "trained_at": meta.get("saved_at"), "evaluation": meta.get("evaluation_summary")}

    def recommend(self, state: RLState, actions: list[LowAction],
                  drift_detected: bool = False) -> RLRecommendation:
        valid, masked = mask_actions(actions, set())
        if self.agent is None:
            rec = fallback_policy.recommend(state, valid, masked, self.load_error or "no policy")
            rec.retraining_recommended = drift_detected
            return rec
        action, info = self.agent.select(state, valid, explore=False)
        expl = (f"Trained hierarchical Q-learning policy chose {action.high}"
                + (f" / {action.signature}" if action.high != "NO_CHANGE" else "")
                + f" (value source: {info['value_source']}; Q values are learned planner-cost-based "
                  "returns, not runtime guarantees).")
        if info["value_source"] != "exact_state":
            expl += " This exact state was not seen during training; values were backed off."
        if drift_detected:
            expl += " WORKLOAD_DRIFT detected: policy re-evaluation/retraining is recommended."
        return RLRecommendation(
            policy_mode="trained_policy", state_key=state.key(), high_level_action=action.high,
            low_level_action=action.signature if action.high != "NO_CHANGE" else None,
            selected_candidate=action.payload or None,
            q_values={**{f"HIGH:{k}": round(v, 4) for k, v in info["q_high"].items()},
                      **{f"LOW:{k}": round(v, 4) for k, v in info["q_low"].items()}},
            valid_actions=[f"{h}:{a.signature}" for h, lst in valid.items() for a in lst],
            masked_actions=masked, explanation=expl, retraining_recommended=drift_detected)
