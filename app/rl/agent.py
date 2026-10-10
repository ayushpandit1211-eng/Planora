"""Hierarchical tabular Q-learning agent (Section 21 RL ALGORITHM, Addendum R7).

State representation : discretised RLState.key() (anonymised structural features)
Action representation: Q_high[state][HIGH] and Q_low[state|HIGH][signature]
Exploration          : epsilon-greedy, epsilon decays multiplicatively per episode
Learning rate (alpha), discount (gamma): constructor arguments (defaults 0.2 / 0.9)
Update               : Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
Generalisation       : for states never visited in training, values back off to the
                       coarse-key table and then to state-marginal action values (reported).
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.errors import ErrorCategory, QuerySentinelError
from app.rl.environment import HIGH_ACTIONS, LowAction
from app.rl.state_encoder import RLState

POLICY_VERSION = 1


def _dd():
    return defaultdict(float)


ACTION_SCHEMA_VERSION = 1      # HIGH_ACTIONS list of this code version
STATE_ENCODING_VERSION = 1     # app/rl/state_encoder.py RLState.key()


class QLearningAgent:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.97, seed: int = 7):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = random.Random(seed)
        self.q_high: dict[str, dict[str, float]] = defaultdict(_dd)
        self.q_low: dict[str, dict[str, float]] = defaultdict(_dd)
        self.q_high_coarse: dict[str, dict[str, float]] = defaultdict(_dd)
        self.q_low_coarse: dict[str, dict[str, float]] = defaultdict(_dd)
        self.visits: dict[str, int] = defaultdict(int)
        self.marginal_high: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        self.marginal_low: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        self.meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ value lookup
    def high_values(self, state: RLState, valid: list[str]) -> tuple[dict[str, float], str]:
        k = state.key()
        if self.visits.get(k):
            return {a: self.q_high[k].get(a, 0.0) for a in valid}, "exact_state"
        ck = state.coarse_key()
        if ck in self.q_high_coarse and self.q_high_coarse[ck]:
            return {a: self.q_high_coarse[ck].get(a, 0.0) for a in valid}, "coarse_state_backoff"
        return {a: (self.marginal_high[a][0] / self.marginal_high[a][1]) if self.marginal_high[a][1] else 0.0
                for a in valid}, "state_marginal_backoff"

    def low_values(self, state: RLState, high: str, sigs: list[str]) -> dict[str, float]:
        k = f"{state.key()}#{high}"
        if k in self.q_low and self.q_low[k]:
            return {s: self.q_low[k].get(s, 0.0) for s in sigs}
        ck = f"{state.coarse_key()}#{high}"
        if ck in self.q_low_coarse and self.q_low_coarse[ck]:
            return {s: self.q_low_coarse[ck].get(s, 0.0) for s in sigs}
        return {s: (self.marginal_low[f"{high}#{s}"][0] / self.marginal_low[f"{high}#{s}"][1])
                if self.marginal_low[f"{high}#{s}"][1] else 0.0 for s in sigs}

    # ------------------------------------------------------------------ acting
    def select(self, state: RLState, valid: dict[str, list[LowAction]], explore: bool) -> tuple[LowAction, dict[str, Any]]:
        highs = list(valid)
        hv, source = self.high_values(state, highs)
        if explore and self.rng.random() < self.epsilon:
            high = self.rng.choice(highs)
        else:
            best = max(hv.values())
            high = sorted([a for a in highs if hv[a] == best], key=HIGH_ACTIONS.index)[0]
        lows = valid[high]
        lv = self.low_values(state, high, [a.signature for a in lows])
        if explore and self.rng.random() < self.epsilon:
            low = self.rng.choice(lows)
        else:
            low = max(lows, key=lambda a: (lv.get(a.signature, 0.0), -lows.index(a)))
        return low, {"q_high": hv, "q_low": lv, "value_source": source}

    # ------------------------------------------------------------------ learning
    def update(self, s: RLState, action: LowAction, reward: float, s_next: RLState,
               next_valid: list[str], done: bool, weight: float = 1.0) -> float:
        """Q-learning update. Returns the TD error of the high-level Q value (Final Upgrade P9: used as replay
        priority). `weight` = importance-sampling weight for prioritized replay (1.0 = original behaviour)."""
        k, ck = s.key(), s.coarse_key()
        td_error = 0.0
        self.visits[k] += 1
        if done or not next_valid:
            future = 0.0
        else:
            nk = s_next.key()
            future = max(self.q_high[nk].get(a, 0.0) for a in next_valid) if nk in self.q_high else 0.0
        target = reward + self.gamma * future
        lr = self.alpha * weight
        for i, (table, key) in enumerate(((self.q_high, k), (self.q_high_coarse, ck))):
            q = table[key][action.high]
            if i == 0:
                td_error = target - q
            table[key][action.high] = q + lr * (target - q)
        for table, key in ((self.q_low, f"{k}#{action.high}"), (self.q_low_coarse, f"{ck}#{action.high}")):
            q = table[key][action.signature]
            table[key][action.signature] = q + lr * (target - q)
        mh = self.marginal_high[action.high]
        mh[0] += reward
        mh[1] += 1
        ml = self.marginal_low[f"{action.high}#{action.signature}"]
        ml[0] += reward
        ml[1] += 1
        return td_error

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ persistence
    def save_policy(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": POLICY_VERSION, "algorithm": "hierarchical tabular Q-learning",
            "alpha": self.alpha, "gamma": self.gamma, "epsilon_final": self.epsilon,
            "high_actions": HIGH_ACTIONS,
            "q_high": {k: dict(v) for k, v in self.q_high.items()},
            "q_low": {k: dict(v) for k, v in self.q_low.items()},
            "q_high_coarse": {k: dict(v) for k, v in self.q_high_coarse.items()},
            "q_low_coarse": {k: dict(v) for k, v in self.q_low_coarse.items()},
            "visits": dict(self.visits),
            "marginal_high": {k: list(v) for k, v in self.marginal_high.items()},
            "marginal_low": {k: list(v) for k, v in self.marginal_low.items()},
            "meta": {**(meta or {}), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(path)

    @classmethod
    def load_policy(cls, path: str | Path) -> "QLearningAgent":
        path = Path(path)
        if not path.exists():
            raise QuerySentinelError(ErrorCategory.RL_UNAVAILABLE, "RL policy file not found", stage="rl")
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, "RL policy file unreadable", stage="rl") from None
        if data.get("version") != POLICY_VERSION or not isinstance(data.get("q_high"), dict) or not data["q_high"]:
            raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, "RL policy incompatible or empty", stage="rl")
        agent = cls(alpha=data.get("alpha", 0.2), gamma=data.get("gamma", 0.9), epsilon=0.0)
        for name in ("q_high", "q_low", "q_high_coarse", "q_low_coarse"):
            table = getattr(agent, name)
            for k, v in data.get(name, {}).items():
                for a, q in v.items():
                    if not isinstance(q, (int, float)):
                        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, "RL policy has non-numeric values", stage="rl")
                    table[k][a] = float(q)
        agent.visits.update({k: int(v) for k, v in data.get("visits", {}).items()})
        for k, v in data.get("marginal_high", {}).items():
            agent.marginal_high[k] = [float(v[0]), int(v[1])]
        for k, v in data.get("marginal_low", {}).items():
            agent.marginal_low[k] = [float(v[0]), int(v[1])]
        agent.meta = data.get("meta", {})
        # Final Upgrade P9: never load a policy whose action schema / state encoding differs from this code
        stored_actions = data.get("high_actions")
        if stored_actions is not None and list(stored_actions) != list(HIGH_ACTIONS):
            raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, "RL policy action schema incompatible",
                                     stage="rl")
        for key, cur in (("action_schema_version", ACTION_SCHEMA_VERSION),
                         ("state_encoding_version", STATE_ENCODING_VERSION)):
            if key in agent.meta and agent.meta[key] != cur:
                raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, f"RL policy {key} incompatible",
                                         stage="rl")
        return agent
