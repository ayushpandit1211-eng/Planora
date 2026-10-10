#!/usr/bin/env python3
"""Train the hierarchical Q-learning optimisation policy against REAL planner costs.

Each training episode is built from a workload template instance on the demo database; actions
are evaluated with HypoPG hypothetical indexes / EXPLAIN of rewritten SQL (planner estimates).
Held-out evaluation uses templates that never appear in training. The fallback (rule) policy is
evaluated on the same held-out episodes as a baseline.

    python scripts/train_rl.py                 # writes models/rl_policy.json + models/rl_metrics.json
    python scripts/train_rl.py --episodes 1200

Requires the demo database (scripts/seed_data.py) with HypoPG installed. Rewards are
planner-cost based: they do NOT measure real user value or runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg2  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.gnn.analyzer import GNNAnalyzer  # noqa: E402
from app.masking_vault import DataObfuscator, InMemoryVaultBackend  # noqa: E402
from app.postgres_instrumentation import PostgresInstrumentation  # noqa: E402
from app.rl import fallback_policy  # noqa: E402
from app.rl.agent import QLearningAgent  # noqa: E402
from app.rl.trainer import Trainer, TrainerConfig  # noqa: E402

# Workload templates. {t} tenant uuid, {d0}/{d1} dates, {sku}, {st} status, {em} email, {amt}
TRAIN_TEMPLATES = {
    "tenant_date_report": "SELECT date_trunc('week', o.order_date) AS wk, count(*), sum(o.total_amount) FROM orders o "
                          "WHERE o.tenant_id = '{t}' AND o.order_date >= '{d0}' AND o.order_date < '{d1}' GROUP BY 1 ORDER BY 1",
    "sku_join_tenant": "SELECT o.order_id, i.quantity FROM orders o JOIN order_items i ON i.order_id = o.order_id "
                       "WHERE o.tenant_id = '{t}' AND i.product_sku = '{sku}' ORDER BY o.order_date DESC LIMIT 50",
    "correlated_count": "SELECT o.order_id, (SELECT count(*) FROM order_items i WHERE i.order_id = o.order_id) AS n "
                        "FROM orders o WHERE o.status = '{st}' AND o.order_date >= '{d0}'",
    "email_lookup": "SELECT o.order_id, o.total_amount FROM orders o WHERE o.customer_email = '{em}'",
    "status_date_count": "SELECT count(*) FROM orders o WHERE o.status = '{st}' AND o.order_date >= '{d0}'",
    "cast_date_tenant": "SELECT o.order_id FROM orders o WHERE o.tenant_id = '{t}' AND o.order_date::date = '{d0}'",
    "sku_aggregate": "SELECT i.product_sku, sum(i.quantity) FROM order_items i WHERE i.product_sku = '{sku}' GROUP BY i.product_sku",
    "tier_join_count": "SELECT t.tier, count(*) FROM tenants t JOIN orders o ON o.tenant_id = t.tenant_id "
                       "WHERE t.tier = 'enterprise' AND o.order_date >= '{d0}' GROUP BY t.tier",
    "point_lookup": "SELECT o.status FROM orders o WHERE o.order_id = '{oid}'",
    "amount_range_top": "SELECT o.order_id, o.total_amount FROM orders o WHERE o.total_amount > {amt} "
                        "ORDER BY o.total_amount DESC LIMIT 20",
}
HELDOUT_TEMPLATES = {
    "correlated_sum_unseen": "SELECT o.order_id, (SELECT sum(i.quantity) FROM order_items i WHERE i.order_id = o.order_id) AS q "
                             "FROM orders o WHERE o.tenant_id = '{t}' AND o.order_date >= '{d0}'",
    "tenant_status_date_unseen": "SELECT count(*) FROM orders o WHERE o.tenant_id = '{t}' AND o.status = '{st}' "
                                 "AND o.order_date >= '{d0}' AND o.order_date < '{d1}'",
    "items_date_join_unseen": "SELECT i.product_sku, count(*) FROM order_items i JOIN orders o ON o.order_id = i.order_id "
                              "WHERE o.order_date >= '{d0}' AND o.order_date < '{d1}' AND i.product_sku = '{sku}' GROUP BY 1",
    "email_in_unseen": "SELECT o.order_id FROM orders o WHERE o.customer_email IN ('{em}', '{em2}')",
    "cast_date_status_unseen": "SELECT o.order_id, o.total_amount FROM orders o WHERE o.status = '{st}' AND o.order_date::date = '{d0}'",
}


def instances(templates: dict[str, str], tenants: list[str], order_ids: list[str], per: int, seed: int):
    rng = random.Random(seed)
    for name, tpl in templates.items():
        for _ in range(per):
            m = rng.randint(1, 8)
            d0 = f"2026-{m:02d}-{rng.randint(1, 28):02d}"
            d1 = f"2026-{min(m + rng.choice([1, 2, 3]), 9):02d}-{rng.randint(1, 28):02d}"
            if d1 <= d0:
                d1 = "2026-09-30"
            yield name, tpl.format(t=rng.choice(tenants[:5]) if rng.random() < 0.7 else rng.choice(tenants),
                                   d0=d0, d1=d1, sku=f"SKU-{rng.randint(1, 300):05d}",
                                   st=rng.choice(["pending", "paid", "shipped", "refunded"]),
                                   em=f"customer{rng.randint(1, 250000)}@example.com",
                                   em2=f"customer{rng.randint(1, 250000)}@example.com",
                                   oid=rng.choice(order_ids), amt=rng.choice([500, 1000, 5000]))


async def build_all(templates, tenants, order_ids, per, seed, sync_conn, instr, obf, gnn, prefix):
    from app.rl.episode_builder import build_episode
    ctxs = []
    for i, (name, sql) in enumerate(instances(templates, tenants, order_ids, per, seed)):
        ctxs.append(await build_episode(f"{prefix}-{i}", sql, instr, obf, gnn, sync_conn, name))
    return ctxs


async def main_async(args) -> int:
    s = get_settings()
    db = Database(s)
    if not await db.connect():
        print("[rl] database unavailable", file=sys.stderr)
        return 1
    instr = PostgresInstrumentation(db)
    sync_conn = psycopg2.connect(host=s.postgres_host, port=s.postgres_port, dbname=s.postgres_db,
                                 user=s.qs_app_user, password=s.qs_app_password.get_secret_value() or None)
    sync_conn.set_session(readonly=True, autocommit=True)
    with sync_conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
        cur.execute("SELECT tenant_id::text FROM tenants ORDER BY name")
        tenants = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT order_id::text FROM orders LIMIT 50")
        order_ids = [r[0] for r in cur.fetchall()]
    obf = DataObfuscator(InMemoryVaultBackend())
    gnn = GNNAnalyzer()
    print(f"[rl] building episodes (GNN mode: {gnn.mode})")
    train_ctx = await build_all(TRAIN_TEMPLATES, tenants, order_ids, args.per_template, args.seed, sync_conn, instr, obf, gnn, "rltrain")
    eval_seen = await build_all(TRAIN_TEMPLATES, tenants, order_ids, 2, args.seed + 1, sync_conn, instr, obf, gnn, "rlevals")
    eval_unseen = await build_all(HELDOUT_TEMPLATES, tenants, order_ids, 3, args.seed + 2, sync_conn, instr, obf, gnn, "rlevalu")
    print(f"[rl] episodes: train={len(train_ctx)} eval_seen={len(eval_seen)} eval_unseen={len(eval_unseen)}")
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=1.0, epsilon_min=0.05,
                           epsilon_decay=args.epsilon_decay, seed=args.seed)
    trainer = Trainer(agent, TrainerConfig(max_episodes=args.episodes, seed=args.seed))
    train_stats = trainer.train(train_ctx)
    fb = lambda st, valid: fallback_policy.choose(st, valid)[0]  # noqa: E731
    metrics = {
        "algorithm": "hierarchical tabular Q-learning (R7)",
        "hyperparameters": {"alpha": args.alpha, "gamma": args.gamma, "epsilon_decay": args.epsilon_decay,
                            "max_episodes": args.episodes, "max_steps_per_episode": 3},
        "reward": "w_freq*normalized_planner_cost_reduction - storage - write - redundancy - risk (app/rl/environment.py)",
        "training": train_stats,
        "evaluation_seen_templates": trainer.evaluate(eval_seen),
        "evaluation_unseen_templates": trainer.evaluate(eval_unseen),
        "baseline_fallback_policy_unseen": trainer.evaluate(eval_unseen, chooser=fb),
        "baseline_fallback_policy_seen": trainer.evaluate(eval_seen, chooser=fb),
        "states_visited": len(agent.visits),
        "note": "Planner-estimated (HypoPG/EXPLAIN) costs on the demo dataset; small-scale demo - "
                "not evidence of scientific generalisation.",
    }
    summary = {k: {"mean_reward": metrics[k]["mean_reward"],
                   "mean_planner_cost_reduction_pct": metrics[k]["mean_planner_cost_reduction_pct"],
                   "recommendation_validity": metrics[k]["recommendation_validity"]}
               for k in ("evaluation_seen_templates", "evaluation_unseen_templates",
                         "baseline_fallback_policy_unseen", "baseline_fallback_policy_seen")}
    out = Path(args.out) if args.out else s.resolve_path(s.rl_policy_path)
    agent.save_policy(out, {"evaluation_summary": summary, "train_episodes": train_stats["episodes"],
                            "templates_train": list(TRAIN_TEMPLATES), "templates_heldout": list(HELDOUT_TEMPLATES)})
    (out.parent / "rl_metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    print(f"[rl] policy -> {out}\n[rl] metrics -> {out.parent / 'rl_metrics.json'}")
    sync_conn.close()
    await db.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon-decay", type=float, default=0.993)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
