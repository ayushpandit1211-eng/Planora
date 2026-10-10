"""RL (Section 21 + R7/60): state encoding, masking, transitions, reward, Q update, save/load,
fallback mode, hierarchical selection and workload weighting."""
import pytest

from app.rl import fallback_policy
from app.rl.agent import QLearningAgent
from app.rl.environment import (EpisodeContext, LowAction, OptimizationEnv, StepOutcome, compute_reward,
                                frequency_weight_from_calls, mask_actions)
from app.rl.policy import RLPolicy
from app.rl.state_encoder import RLState, encode_state
from app.rl.trainer import Trainer, TrainerConfig, run_episode


class FakeEvaluator:
    """Test double: deterministic planner costs per action signature."""
    COSTS = {"composite:eq+range": 100.0, "single:eq": 600.0, "cast_date_equality_to_range": 300.0}

    def __init__(self):
        self.cost = 1000.0

    def baseline_cost(self):
        return 1000.0

    def reset(self):
        self.cost = 1000.0

    def apply(self, a):
        before = self.cost
        self.cost = min(self.cost, self.COSTS.get(a.signature, self.cost))
        return StepOutcome(before, self.cost, storage_bytes=1e6, table_bytes=1e8, n_columns=2)


def state(**kw):
    base = dict(query_fingerprint="fp", bottleneck_class=0, plan_total_cost=1000, plan_rows=10,
                operator_histogram={"Seq Scan": 1}, per_table_access={"t": {"eq": ["a"], "range": ["b"]}},
                target_table="t", existing_index_coverage=0, table_size_bucket="large(1M-10M)",
                write_frequency_bucket="low", partition_suitable=False, non_sargable=False, has_subplan=False)
    base.update(kw)
    return encode_state(**base)


def actions():
    return [LowAction("ADD_INDEX", "composite:eq+range", {"id": 0, "kind": "composite", "roles": ["eq", "range"], "columns": ["a", "b"]}),
            LowAction("ADD_INDEX", "single:eq", {"id": 1, "kind": "single", "roles": ["eq"], "columns": ["a"]}),
            LowAction("ADD_INDEX", "single:eq", {"id": 2, "kind": "single", "roles": ["eq"]}, redundant=True),
            LowAction("PARTITION", "range", {"id": "p"})]


def test_state_encoding_has_no_identifiers():
    s = state()
    k = s.key()
    import re
    assert all(re.fullmatch(r"[a-z]{2}\d+", part) for part in k.split("|")), k   # only discretised codes
    assert "fp" not in k and len({p[:2] for p in k.split("|")}) == len(k.split("|"))
    assert s.has_seq_scan == 1 and s.n_eq_filters == 1 and s.has_range_filter == 1
    assert isinstance(RLState().coarse_key(), str)


def test_invalid_action_masking():
    valid, masked = mask_actions(actions(), set())
    assert len(valid["ADD_INDEX"]) == 2 and "NO_CHANGE" in valid
    assert any("redundant" in v for v in masked.values())
    assert "REWRITE_SQL" in masked and "SHARD_ADVISORY" in masked


def test_environment_transition_and_reward():
    env = OptimizationEnv(EpisodeContext(state(), actions(), FakeEvaluator()))
    s0 = env.reset()
    s1, r, done, info = env.step(actions()[0])
    assert s1.indexes_added == 1 and s1.cost_bucket < s0.cost_bucket or s1.cost_bucket <= s0.cost_bucket
    assert r > 0 and info["planner_cost_reduction_pct"] == 90.0 and not done
    _, r2, done2, _ = env.step(LowAction("NO_CHANGE", "none"))
    assert done2 and r2 == 0.0


def test_reward_components_and_workload_weight():
    s = state()
    out = StepOutcome(1000, 500, storage_bytes=5e7, table_bytes=1e8, n_columns=2)
    r1, parts1 = compute_reward(actions()[0], out, 1000, s, 1.0)
    r2, parts2 = compute_reward(actions()[0], out, 1000, s, frequency_weight_from_calls(999))
    assert parts1["benefit"] == 0.5 and parts2["benefit"] == pytest.approx(0.5 * 4.0)
    assert parts1["storage_penalty"] == pytest.approx(0.05) and parts1["write_penalty"] > 0
    rp, pp = compute_reward(LowAction("PARTITION", "range"), StepOutcome(1000, 1000, evaluated=False), 1000, s, 1.0)
    assert pp["benefit"] == 0 and rp < 0       # advisory: no validated benefit, only risk
    rr, _ = compute_reward(LowAction("ADD_INDEX", "x", redundant=True), StepOutcome(1000, 1000), 1000, s, 1.0)
    assert rr <= -0.3


def test_training_update_learns_best_action():
    agent = QLearningAgent(epsilon_decay=0.9, seed=1)
    ctx = EpisodeContext(state(), actions(), FakeEvaluator())
    stats = Trainer(agent, TrainerConfig(max_episodes=150, window=25, seed=1)).train([ctx], log=lambda s: None)
    assert stats["episodes"] > 0
    res = run_episode(agent, ctx, explore=False, learn=False)
    assert res["actions"][0] == "ADD_INDEX:composite:eq+range"
    q, src = agent.high_values(state(), ["ADD_INDEX", "PARTITION", "NO_CHANGE"])
    assert src == "exact_state" and q["ADD_INDEX"] > q["PARTITION"]


def test_save_load_policy(tmp_path):
    agent = QLearningAgent(seed=2)
    ctx = EpisodeContext(state(), actions(), FakeEvaluator())
    Trainer(agent, TrainerConfig(max_episodes=40, window=20)).train([ctx], log=lambda s: None)
    path = tmp_path / "p.json"
    agent.save_policy(path, {"x": 1})
    loaded = QLearningAgent.load_policy(path)
    assert loaded.q_high[state().key()] == agent.q_high[state().key()]
    pol = RLPolicy(str(path))
    rec = pol.recommend(state(), actions())
    assert rec.policy_mode == "trained_policy" and rec.high_level_action in ("ADD_INDEX", "NO_CHANGE", "PARTITION")


def test_fallback_mode_is_labelled(tmp_path):
    pol = RLPolicy(str(tmp_path / "missing.json"))
    rec = pol.recommend(state(), actions())
    assert rec.policy_mode == "analytic_fallback" and "fallback" in rec.explanation.lower()
    assert rec.high_level_action == "ADD_INDEX" and rec.low_level_action == "composite:eq+range"


def test_fallback_no_change_for_efficient_plans():
    a, why = fallback_policy.choose(state(bottleneck_class=3, plan_total_cost=10), mask_actions(actions(), set())[0])
    assert a.high == "NO_CHANGE"


def test_unseen_state_backoff(tmp_path):
    agent = QLearningAgent(seed=3)
    Trainer(agent, TrainerConfig(max_episodes=40, window=20)).train(
        [EpisodeContext(state(), actions(), FakeEvaluator())], log=lambda s: None)
    _, src = agent.high_values(state(table_size_bucket="xlarge(>10M)", write_frequency_bucket="high"), ["ADD_INDEX"])
    assert src in ("coarse_state_backoff", "state_marginal_backoff")


def test_corrupt_policy_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert RLPolicy(str(p)).mode == "analytic_fallback"
