#!/usr/bin/env python3
"""Train the GNN bottleneck classifier on synthetic PostgreSQL-shaped plans.

    python scripts/train_gnn.py                    # default config, writes models/gnn_bottleneck.pt
    python scripts/train_gnn.py --epochs 80 --per-class 600
    python scripts/train_gnn.py --real-plans       # also report agreement on live demo-DB plans

Metrics are written to models/gnn_metrics.json. "Agreement on real plans" compares the GNN
with the deterministic analytic rules on EXPLAIN plans of examples/slow_queries.sql; it is an
agreement rate, not an accuracy against ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.gnn.training import TrainConfig, train  # noqa: E402


def real_plan_agreement() -> dict:
    import psycopg2
    from app.contracts import BOTTLENECK_CLASSES
    from app.gnn.analyzer import GNNAnalyzer
    from app.gnn import fallback_analytics as fb
    from app.plan_parser import parse_explain
    from app.security import SecurityGateway, EXPLAIN_PREFIX

    text = (ROOT / "examples" / "slow_queries.sql").read_text()
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("--"))
    queries = [q.strip() for q in text.split(";") if q.strip()]
    s = get_settings()
    conn = psycopg2.connect(host=s.postgres_host, port=s.postgres_port, dbname=s.postgres_db,
                            user=s.qs_app_user, password=s.qs_app_password.get_secret_value() or None)
    conn.set_session(readonly=True)
    analyzer = GNNAnalyzer()
    rows = []
    with conn.cursor() as cur:
        for i, q in enumerate(queries):
            v = SecurityGateway().validate(q)
            cur.execute(EXPLAIN_PREFIX + v.sql)
            plan = parse_explain(cur.fetchone()[0][0])
            diag = analyzer.analyze(plan)
            rule = fb.classify(plan)
            rows.append({"scenario": i, "gnn": diag.primary_bottleneck_type, "gnn_confidence": diag.confidence,
                         "rules": BOTTLENECK_CLASSES[rule.class_id], "agree": diag.bottleneck_class_id == rule.class_id})
    conn.close()
    return {"plans": len(rows), "agreement_rate": round(sum(r["agree"] for r in rows) / max(len(rows), 1), 3),
            "details": rows, "note": "agreement with analytic rules on live demo plans - not ground-truth accuracy"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real-plans", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else get_settings().resolve_path(get_settings().gnn_checkpoint_path)
    metrics = train(TrainConfig(n_per_class=args.per_class, epochs=args.epochs, seed=args.seed), str(out))
    if args.real_plans:
        try:
            metrics["real_plan_agreement"] = real_plan_agreement()
        except Exception as exc:  # noqa: BLE001
            metrics["real_plan_agreement"] = {"error": f"{type(exc).__name__}: database not reachable?"}
    metrics_path = out.parent / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("best_val_accuracy", "heldout_same_distribution",
                                              "heldout_unseen_template_variants") }, indent=1))
    if "real_plan_agreement" in metrics:
        print(json.dumps(metrics["real_plan_agreement"], indent=1))
    print(f"[gnn] checkpoint -> {out}\n[gnn] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
