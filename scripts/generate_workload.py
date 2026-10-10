#!/usr/bin/env python3
"""Generates a realistic demo workload so pg_stat_statements has evidence to analyse, and
(optionally) exports anonymisable sample history files built from REAL plans of this database.

* read workload: the demonstration scenarios with varying tenants / date ranges
* write workload: INSERT orders / INSERT order_items / UPDATE orders SET status
  (executed inside transactions that are ROLLED BACK - pg_stat_statements still records
  them, the demo data stays unchanged)
* --export-examples: writes examples/sample_history.json and examples/sample_history.csv
  with EXPLAIN JSON plans captured live. Two time windows are produced; in the earlier window
  a temporary index exists (captured plan uses it), in the later window it does not, which
  gives the regression detector a genuine plan change to find. Timings come from real
  EXPLAIN ANALYZE runs in the environment where the script is executed (never invented).

Runs as the superuser from .env (demo database only).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from seed_data import connect  # noqa: E402

READS = {
    "tenant_weekly_report": (
        "SELECT date_trunc('week', o.order_date) AS week, o.status, count(*) AS orders, sum(o.total_amount) AS revenue "
        "FROM orders o WHERE o.tenant_id = %(tenant)s AND o.order_date >= %(d0)s AND o.order_date < %(d1)s "
        "GROUP BY 1, 2 ORDER BY 1, 2"),
    "sku_orders_for_tenant": (
        "SELECT o.order_id, o.order_date, o.total_amount, i.quantity, i.unit_price FROM orders o "
        "JOIN order_items i ON i.order_id = o.order_id WHERE o.tenant_id = %(tenant)s AND i.product_sku = %(sku)s "
        "ORDER BY o.order_date DESC LIMIT 50"),
    "status_dashboard": (
        "SELECT o.status, count(*) FROM orders o WHERE o.status = %(status)s AND o.order_date >= %(d0)s GROUP BY o.status"),
    "customer_lookup": (
        "SELECT o.order_id, o.total_amount FROM orders o WHERE o.customer_email = %(email)s ORDER BY o.order_date DESC LIMIT 20"),
}
WRITES = {
    "insert_order": ("INSERT INTO orders (order_id, tenant_id, customer_email, total_amount, status, order_date) "
                     "VALUES (%(oid)s, %(tenant)s, %(email)s, %(amount)s, 'pending', now())"),
    "insert_item": ("INSERT INTO order_items (item_id, order_id, product_sku, quantity, unit_price) "
                    "SELECT %(iid)s, order_id, %(sku)s, 1, 9.99 FROM orders WHERE tenant_id = %(tenant)s LIMIT 1"),
    "update_status": "UPDATE orders SET status = %(status)s WHERE order_id = (SELECT order_id FROM orders WHERE tenant_id = %(tenant)s LIMIT 1)",
}
WRITE_WEIGHTS = {"insert_order": 30, "insert_item": 60, "update_status": 25}


def params(rng: random.Random, tenants: list[str]) -> dict:
    d0 = datetime(2026, rng.randint(1, 8), rng.randint(1, 28))
    return {"tenant": rng.choice(tenants[:5]) if rng.random() < 0.8 else rng.choice(tenants),
            "d0": d0, "d1": d0 + timedelta(days=rng.choice([7, 30, 90])),
            "sku": f"SKU-{int(rng.paretovariate(1.2)) % 5000 + 1:05d}",
            "status": rng.choice(["pending", "paid", "shipped"]),
            "email": f"customer{rng.randint(1, 250000)}@example.com",
            "oid": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "iid": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "amount": round(rng.uniform(5, 500), 2)}


def already_generated(cur) -> bool:
    cur.execute("SELECT count(*) FROM pg_stat_statements WHERE query ILIKE 'INSERT INTO order_items%%'")
    return cur.fetchone()[0] > 0


def run_workload(conn, rng, tenants, n_reads: int, n_writes: int) -> None:
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '20s'")
    for k in range(n_reads):
        name = rng.choice(list(READS))
        cur.execute(READS[name], params(rng, tenants))
        cur.fetchall()
    conn.commit()
    names = [n for n, w in WRITE_WEIGHTS.items() for _ in range(w)]
    for k in range(n_writes):
        name = rng.choice(names)
        cur.execute(WRITES[name], params(rng, tenants))
        if k % 10 == 9:
            conn.rollback()   # keep the demo dataset unchanged
    conn.rollback()
    print(f"[workload] executed {n_reads} reads and {n_writes} writes (writes rolled back)")


def explain(cur, sql: str, prm: dict, analyze: bool) -> dict:
    opts = "ANALYZE TRUE, BUFFERS TRUE, FORMAT JSON" if analyze else "FORMAT JSON"
    cur.execute(f"EXPLAIN ({opts}) " + sql, prm)
    return cur.fetchone()[0][0]


def export_examples(conn, rng, tenants) -> None:
    """Capture real plans in two windows; earlier window has a temporary composite index."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '60s'")
    records = []
    base = datetime(2026, 9, 1, 8)
    queries = ["tenant_weekly_report", "sku_orders_for_tenant", "status_dashboard", "customer_lookup"]
    windows = [("historical", base, True), ("recent", base + timedelta(days=14), False)]
    for label, t0, with_temp_index in windows:
        if with_temp_index:
            cur.execute("CREATE INDEX qs_demo_tmp_tenant_date ON orders (tenant_id, order_date)")
            cur.execute("ANALYZE orders")
        for h in range(6):
            for name in queries:
                prm = params(rng, tenants)
                sql = READS[name]
                plan = explain(cur, sql, prm, analyze=True)
                literal_sql = cur.mogrify(sql, prm).decode()
                records.append({
                    "query": literal_sql,
                    "calls": rng.randint(20, 400) if name != "customer_lookup" else rng.randint(1, 20),
                    "mean_time_ms": round(plan.get("Execution Time", 0.0), 3),
                    "timestamp": (t0 + timedelta(hours=h)).isoformat(),
                    "plan": plan,
                    "window": label,
                    "timing_source": "EXPLAIN ANALYZE in the generating environment",
                })
        if with_temp_index:
            conn.rollback()     # temporary index removed (DDL rolled back)
            cur.execute("SET statement_timeout = '60s'")
    conn.rollback()
    out_json = ROOT / "examples" / "sample_history.json"
    out_json.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                                    "note": "Synthetic demo data; plans captured from the demo DB. "
                                            "Contains literal values on purpose so the ingestion "
                                            "pipeline can demonstrate anonymisation.",
                                    "records": records}, indent=1, default=str))
    out_csv = ROOT / "examples" / "sample_history.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "calls", "mean_time_ms", "timestamp", "plan_json"])
        for r in records:
            w.writerow([r["query"], r["calls"], r["mean_time_ms"], r["timestamp"], json.dumps(r["plan"])])
    print(f"[workload] exported {len(records)} records to {out_json.name} and {out_csv.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", type=int, default=120)
    ap.add_argument("--writes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--if-empty", action="store_true")
    ap.add_argument("--export-examples", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT tenant_id::text FROM tenants ORDER BY name")
    tenants = [r[0] for r in cur.fetchall()]
    if not tenants:
        print("[workload] no tenants - run seed_data.py first", file=sys.stderr)
        return 1
    if args.if_empty and already_generated(cur):
        print("[workload] workload already present in pg_stat_statements - skipping")
    else:
        t = time.time()
        run_workload(conn, rng, tenants, args.reads, args.writes)
        print(f"[workload] done in {time.time() - t:.1f}s")
    if args.export_examples:
        export_examples(conn, rng, tenants)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
