#!/usr/bin/env python3
"""Reproducible benchmark + ablation workflow (Sections 36 / 61, Addendum R12).

    python scripts/run_benchmark.py                 # planner-estimate mode, all demo queries
    BENCHMARK_ENABLED=true BENCHMARK_ADMIN_DSN=postgresql://...disposable... \\
        python scripts/run_benchmark.py --measure   # adds real EXPLAIN ANALYZE timings

Writes benchmark_results/benchmark_<timestamp>.json. Numbers are produced by the environment
the script runs in; none are hard-coded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.benchmark import BenchmarkRunner, load_demo_queries  # noqa: E402
from app.db import Database  # noqa: E402
from app.gnn.analyzer import GNNAnalyzer  # noqa: E402
from app.masking_vault import DataObfuscator, InMemoryVaultBackend  # noqa: E402
from app.rl.policy import RLPolicy  # noqa: E402


async def main_async(args) -> int:
    db = Database()
    if not await db.connect():
        print("database unavailable", file=sys.stderr)
        return 1
    runner = BenchmarkRunner(db, DataObfuscator(InMemoryVaultBackend()), GNNAnalyzer(), RLPolicy())
    queries = load_demo_queries()
    if args.query is not None:
        queries = [queries[args.query]]
    out = []
    for i, q in enumerate(queries):
        res = await runner.run(q, f"bench-{i}", measure=args.measure)
        out.append(res)
        print(f"\nquery {i}  fingerprint {res['query_fingerprint']}  baseline planner cost {res['baseline_planner_cost']:.2f}")
        for r in res["results"]:
            print(f"  {r['variant']:<32} planner_cost={r['planner_cost']:>14.2f}  "
                  f"planner_reduction={str(r['planner_cost_reduction_pct']) + '%':>9}  idx={r['indexes']} rw={r['rewrite']}")
        print("  measurement:", json.dumps(res["measurement"])[:300])
    await db.close()
    path = ROOT / "benchmark_results"
    path.mkdir(exist_ok=True)
    f = path / f"benchmark_{time.strftime('%Y%m%dT%H%M%S')}.json"
    f.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nreport -> {f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="real timing (needs BENCHMARK_ENABLED + disposable DSN)")
    ap.add_argument("--query", type=int, default=None, help="index of a single demo query")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
