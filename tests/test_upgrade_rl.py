"""P9 major tests: curriculum levels, prioritized replay (with TD-error updates in an episode), held-out split
by pattern, checkpoint compatibility. The full trainer loop with curriculum+replay is NOT RUN in tests: the
account-level ml-pipeline hook blocks test files that construct the trainer (see KNOWN_ISSUES K3)."""
import json

import pytest

from app.errors import QuerySentinelError
from app.rl.agent import QLearningAgent
from app.rl.curriculum import level_for, order_by_level, split_by_pattern
from app.rl.environment import EpisodeContext
from app.rl.replay import PrioritizedReplay, ReplayConfig
from app.rl import trainer as rl_trainer
from tests.test_rl import FakeEvaluator, actions, state


def test_curriculum_levels_from_parsed_sql():
    assert level_for("SELECT * FROM t WHERE a = $1") == 1
    assert level_for("SELECT * FROM t WHERE a = $1 AND b > $2") == 2
    assert level_for("SELECT * FROM t JOIN u ON u.id = t.uid WHERE t.a = $1") == 3
    assert level_for("SELECT * FROM t JOIN u ON u.id = t.uid JOIN v ON v.id = u.vid") == 4
    q5 = "SELECT u.k, count(*) FROM t JOIN u ON u.id = t.uid GROUP BY u.k ORDER BY 2 DESC"
    assert level_for(q5) == 5 and level_for(q5, drift=True) == 6
    assert level_for("not sql at all") is None


def test_prioritized_replay_priority_sampling_and_update():
    r = PrioritizedReplay(ReplayConfig(capacity=3, alpha=1.0, uniform_mix=0.0, eps=0.0), seed=1)
    for i, td in enumerate([0.1, 0.1, 10.0]):
        r.add(i, td)
    probs = r.probabilities()
    assert probs[2] > 0.95 and abs(sum(probs) - 1) < 1e-9
    picks = [r.sample(1)[1][0] for _ in range(200)]
    assert picks.count(2) > 180
    _, _, w = r.sample(3)
    assert max(w) == 1.0
    r.update([2], [0.0])
    assert r.prio[2] == 0.0
    r.add(99, 1.0)
    assert len(r) == 3 and 99 in r.items


def test_episode_with_replay_heldout_split_and_checkpoint(tmp_path):
    sqls = ["SELECT * FROM t WHERE a = $1", "SELECT * FROM t JOIN u ON u.id = t.uid WHERE t.a = $1",
            "SELECT u.k, count(*) FROM t JOIN u ON u.id = t.uid GROUP BY u.k ORDER BY 2"]
    ctxs = [EpisodeContext(state(), actions(), FakeEvaluator(), meta={"sql": s, "pattern": f"p{i}"})
            for i, s in enumerate(sqls)]
    assert [lv for lv, _ in order_by_level(ctxs)] == [1, 3, 5]
    train, held = split_by_pattern(ctxs, {"p2"})
    assert len(train) == 2 and len(held) == 1
    agent = QLearningAgent()
    rep = PrioritizedReplay(seed=2)
    res = rl_trainer.run_episode(agent, train[0], explore=True, learn=True, replay=rep)
    assert res["steps"] >= 1 and len(rep) >= 1 and all(p > 0 for p in rep.prio)
    p = tmp_path / "pol.json"
    agent.save_policy(p, {"action_schema_version": 1, "state_encoding_version": 1, "curriculum": True})
    assert QLearningAgent.load_policy(p).meta["curriculum"] is True
    d = json.loads(p.read_text())
    d["meta"]["action_schema_version"] = 2
    p.write_text(json.dumps(d))
    with pytest.raises(QuerySentinelError):
        QLearningAgent.load_policy(p)
