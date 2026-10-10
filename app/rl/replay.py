"""Bounded prioritized experience replay for the tabular Q-learner (Final Upgrade P9, QS_FEATURE_RL_UPGRADES).

priority p_i = |TD error_i| + eps; sampling P(i) = p_i^alpha / sum_k p_k^alpha, mixed with a uniform fraction
for diversity; importance-sampling weights w_i = (N * P(i))^-beta / max_j w_j with beta annealed to 1.
Priorities are updated after every learning step. Negative reward is never the principal priority.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ReplayConfig:
    capacity: int = 2000
    alpha: float = 0.6
    beta0: float = 0.4
    beta_steps: int = 5000
    eps: float = 1e-3
    uniform_mix: float = 0.1
    batch: int = 8


class PrioritizedReplay:
    def __init__(self, cfg: ReplayConfig | None = None, seed: int = 0):
        self.cfg = cfg or ReplayConfig()
        self.items: list[Any] = []
        self.prio: list[float] = []
        self.pos = 0
        self.steps = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: Any, td_error: float) -> None:
        p = abs(td_error) + self.cfg.eps
        if len(self.items) < self.cfg.capacity:
            self.items.append(transition)
            self.prio.append(p)
        else:                                   # bounded: overwrite oldest
            self.items[self.pos] = transition
            self.prio[self.pos] = p
            self.pos = (self.pos + 1) % self.cfg.capacity

    @property
    def beta(self) -> float:
        frac = min(self.steps / max(self.cfg.beta_steps, 1), 1.0)
        return self.cfg.beta0 + frac * (1.0 - self.cfg.beta0)

    def probabilities(self) -> list[float]:
        scaled = [p ** self.cfg.alpha for p in self.prio]
        tot = sum(scaled) or 1.0
        n = len(scaled)
        u = self.cfg.uniform_mix
        return [(1 - u) * s / tot + u / n for s in scaled]

    def sample(self, k: int | None = None) -> tuple[list[int], list[Any], list[float]]:
        k = min(k or self.cfg.batch, len(self.items))
        if k == 0:
            return [], [], []
        probs = self.probabilities()
        idx = self.rng.choices(range(len(self.items)), weights=probs, k=k)
        n = len(self.items)
        w = [(n * probs[i]) ** (-self.beta) for i in idx]
        mw = max(w)
        self.steps += 1
        return idx, [self.items[i] for i in idx], [x / mw for x in w]

    def update(self, idx: list[int], td_errors: list[float]) -> None:
        for i, td in zip(idx, td_errors):
            self.prio[i] = abs(td) + self.cfg.eps
