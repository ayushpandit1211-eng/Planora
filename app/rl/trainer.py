"""RL training / evaluation loop (Section 21 GENERALIZATION).

Reports training reward, evaluation reward, action distribution, recommendation validity and
unseen-pattern results. Stopping criteria: max_episodes reached, or the moving average of
episode reward changes by less than `tolerance` over `patience` consecutive windows.
Small demo-scale results do not establish scientific generalisation.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from app.rl.agent import QLearningAgent
from app.rl.environment import EpisodeContext, OptimizationEnv


@dataclass
class TrainerConfig:
    max_episodes: int = 600
    window: int = 50
    tolerance: float = 0.005
    patience: int = 4
    seed: int = 11
    # Final Upgrade P9 (used only when enabled; defaults keep the original behaviour)
    curriculum: bool = False
    prioritized_replay: bool = False
    replay_batch: int = 8


def run_episode(agent: QLearningAgent | None, ctx: EpisodeContext, explore: bool, learn: bool,
                chooser: Callable | None = None, replay=None) -> dict[str, Any]:
    """`chooser(state, valid) -> LowAction` replaces the agent (used to evaluate the fallback policy)."""
    env = OptimizationEnv(ctx)
    state = env.reset()
    total, actions, valid_choices = 0.0, [], 0
    done = False
    while not done:
        valid, _ = env.valid_actions()
        if chooser is not None:
            action = chooser(state, valid)
        else:
            action, _info = agent.select(state, valid, explore=explore)
        nxt, reward, done, info = env.step(action)
        if learn and agent is not None:
            next_valid, _ = env.valid_actions()
            td = agent.update(state, action, reward, nxt, list(next_valid), done)
            if replay is not None:
                replay.add((state, action, reward, nxt, list(next_valid), done), td)
                idx, batch, weights = replay.sample()
                tds = [agent.update(*t, weight=w) for t, w in zip(batch, weights)]
                replay.update(idx, tds)
        total += reward
        actions.append(f"{action.high}:{action.signature}")
        if action.high == "NO_CHANGE" or (info.get("cost_after") is not None and info.get("cost_before")
                                          and info["cost_after"] <= info["cost_before"]) or not info["evaluated"]:
            valid_choices += 1
        state = nxt
    return {"reward": total, "actions": actions, "steps": len(actions),
            "valid_fraction": valid_choices / max(len(actions), 1),
            "cost_reduction_pct": env.cumulative_reduction_pct(), "history": env.history}


class Trainer:
    def __init__(self, agent: QLearningAgent, cfg: TrainerConfig | None = None):
        self.agent = agent
        self.cfg = cfg or TrainerConfig()

    def train(self, contexts: list[EpisodeContext], log: Callable[[str], None] = print) -> dict[str, Any]:
        rng = random.Random(self.cfg.seed)
        rewards: list[float] = []
        moving: list[float] = []
        stale = 0
        t0 = time.time()
        stop_reason = "max_episodes"
        replay = None
        if self.cfg.prioritized_replay:
            from app.rl.replay import PrioritizedReplay, ReplayConfig
            replay = PrioritizedReplay(ReplayConfig(batch=self.cfg.replay_batch), seed=self.cfg.seed)
        levels = None
        if self.cfg.curriculum:
            from app.rl.curriculum import order_by_level
            levels = order_by_level(contexts) or [(1, c) for c in contexts]
            self.curriculum_levels = sorted({lv for lv, _ in levels})
        for ep in range(1, self.cfg.max_episodes + 1):
            if levels is not None:
                # unlock levels progressively: first fraction of training sees only the easiest level(s)
                frac = ep / self.cfg.max_episodes
                lv_list = self.curriculum_levels
                upto = lv_list[min(int(frac * len(lv_list)), len(lv_list) - 1)]
                pool = [c for lv, c in levels if lv <= upto]
                ctx = rng.choice(pool)
            else:
                ctx = rng.choice(contexts)
            res = run_episode(self.agent, ctx, explore=True, learn=True, replay=replay)
            rewards.append(res["reward"])
            self.agent.decay()
            if ep % self.cfg.window == 0:
                avg = sum(rewards[-self.cfg.window:]) / self.cfg.window
                if moving and abs(avg - moving[-1]) < self.cfg.tolerance and self.agent.epsilon <= self.agent.epsilon_min + 1e-9:
                    stale += 1
                else:
                    stale = 0
                moving.append(avg)
                log(f"[rl] episode {ep:4d} avg_reward={avg:.4f} epsilon={self.agent.epsilon:.3f}")
                if stale >= self.cfg.patience:
                    stop_reason = "converged (moving-average reward stable)"
                    break
        return {"episodes": len(rewards), "stop_reason": stop_reason,
                "curriculum_levels": getattr(self, "curriculum_levels", None),
                "replay_size": len(replay) if replay is not None else None,
                "training_reward_moving_avg": [round(m, 4) for m in moving],
                "final_epsilon": round(self.agent.epsilon, 4), "train_seconds": round(time.time() - t0, 1)}

    def evaluate(self, contexts: list[EpisodeContext], chooser: Callable | None = None) -> dict[str, Any]:
        results = [run_episode(self.agent, ctx, explore=False, learn=False, chooser=chooser)
                   | {"pattern": ctx.meta.get("pattern")} for ctx in contexts]
        dist = Counter(a.split(":")[0] for r in results for a in r["actions"])
        reductions = [r["cost_reduction_pct"] for r in results if r["cost_reduction_pct"] is not None]
        return {
            "episodes": len(results),
            "mean_reward": round(sum(r["reward"] for r in results) / max(len(results), 1), 4),
            "action_distribution": dict(dist),
            "recommendation_validity": round(sum(r["valid_fraction"] for r in results) / max(len(results), 1), 4),
            "mean_planner_cost_reduction_pct": round(sum(reductions) / len(reductions), 2) if reductions else None,
            "metric_note": "planner-estimated (HypoPG/EXPLAIN) cost reduction - not measured runtime",
            "per_pattern": [{"pattern": r["pattern"], "reward": round(r["reward"], 4), "actions": r["actions"],
                             "planner_cost_reduction_pct": r["cost_reduction_pct"]} for r in results],
        }
