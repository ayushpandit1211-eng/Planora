#!/usr/bin/env python3
"""Deterministic synthetic data generator for the Planora demo database (Section 10).

* streams rows to PostgreSQL with COPY in fixed-size batches (never holds the
  full dataset in RAM)
* deterministic seed (SEED_RANDOM_SEED)
* configurable size via SEED_TENANTS / SEED_ORDERS / SEED_ORDER_ITEMS
* intentional workload skew: ~80% of orders belong to 5 highly active tenants

Lightweight mode (default):  SEED_ORDERS=100000 SEED_ORDER_ITEMS=300000
Full demonstration mode:     SEED_ORDERS=1000000 SEED_ORDER_ITEMS=3000000

All values are synthetic. Emails use the reserved example.com domain.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2 import sql as psql

ROOT = Path(__file__).resolve().parent.parent

STATUSES = [("delivered", 0.55), ("shipped", 0.15), ("paid", 0.12), ("pending", 0.08),
            ("cancelled", 0.06), ("refunded", 0.04)]
TIERS = [("free", 0.6), ("pro", 0.3), ("enterprise", 0.1)]
N_SKUS = 5000
HOT_TENANTS = 5
HOT_SHARE = 0.80
DATE_END = datetime(2026, 9, 30, 23, 59, 59)
DATE_SPAN_DAYS = 3 * 365
BATCH = 50_000


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def connect():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "querysentinel"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD") or None,
    )


def weighted(rng: random.Random, items):
    r = rng.random()
    acc = 0.0
    for value, w in items:
        acc += w
        if r <= acc:
            return value
    return items[-1][0]


def det_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class ZipfSampler:
    """Inverse-CDF Zipf(s=1.1) sampler over SKU ranks (realistic product popularity)."""

    def __init__(self, n: int, s: float = 1.1):
        weights = [1.0 / (k ** s) for k in range(1, n + 1)]
        total = sum(weights)
        self.cdf = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self.cdf.append(acc)

    def sample(self, rng: random.Random) -> int:
        import bisect
        return min(bisect.bisect_left(self.cdf, rng.random()), len(self.cdf) - 1) + 1


def ensure_schema(cur) -> None:
    for name in ("create_extensions.sql", "init_db.sql"):
        cur.execute((ROOT / "scripts" / name).read_text())


def set_app_role_password(cur) -> None:
    pw = os.environ.get("QS_APP_PASSWORD")
    if pw:
        cur.execute(psql.SQL("ALTER ROLE qs_app PASSWORD {}").format(psql.Literal(pw)))


def table_count(cur, table: str) -> int:
    cur.execute(psql.SQL("SELECT count(*) FROM {}").format(psql.Identifier(table)))
    return int(cur.fetchone()[0])


def copy_rows(cur, table: str, columns: list[str], buf: io.StringIO) -> None:
    buf.seek(0)
    cur.copy_expert(
        psql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT text)").format(
            psql.Identifier(table), psql.SQL(", ").join(map(psql.Identifier, columns))
        ).as_string(cur),
        buf,
    )


def seed(n_tenants: int, n_orders: int, n_items: int, seed_value: int, reset: bool, if_empty: bool) -> None:
    rng = random.Random(seed_value)
    zipf = ZipfSampler(N_SKUS)
    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()
    ensure_schema(cur)
    set_app_role_password(cur)
    conn.commit()

    existing = table_count(cur, "orders")
    if existing and if_empty and not reset:
        print(f"[seed] orders already contains {existing} rows - skipping (use --reset to rebuild)")
        conn.close()
        return
    if reset or existing:
        print("[seed] truncating demo tables")
        cur.execute("TRUNCATE order_items, orders, tenants")
        conn.commit()

    t0 = time.time()
    # ---------------- tenants ----------------
    tenant_ids = []
    buf = io.StringIO()
    for i in range(n_tenants):
        tid = det_uuid(rng)
        tenant_ids.append(tid)
        created = DATE_END - timedelta(days=DATE_SPAN_DAYS + rng.randint(0, 700))
        tier = "enterprise" if i < HOT_TENANTS else weighted(rng, TIERS)
        buf.write(f"{tid}\tTenant {i + 1:04d} Ltd\t{tier}\t{created:%Y-%m-%d %H:%M:%S}\n")
    copy_rows(cur, "tenants", ["tenant_id", "name", "tier", "created_at"], buf)
    conn.commit()
    print(f"[seed] tenants: {n_tenants}")

    hot = tenant_ids[:HOT_TENANTS]
    cold = tenant_ids[HOT_TENANTS:] or hot
    items_per_order = n_items / max(n_orders, 1)
    items_left = n_items
    orders_done = 0
    while orders_done < n_orders:
        batch = min(BATCH, n_orders - orders_done)
        obuf, ibuf = io.StringIO(), io.StringIO()
        for j in range(batch):
            oid = det_uuid(rng)
            tenant = rng.choice(hot) if rng.random() < HOT_SHARE else rng.choice(cold)
            # growth trend: more recent orders are more frequent (sqrt skew)
            age_days = DATE_SPAN_DAYS * (1.0 - math.sqrt(rng.random()))
            odate = DATE_END - timedelta(days=age_days, seconds=rng.randint(0, 86399))
            status = weighted(rng, STATUSES)
            if age_days < 3 and status == "delivered":
                status = "pending"
            amount = round(min(max(rng.lognormvariate(4.3, 0.9), 1.0), 99999.0), 2)
            email = f"customer{rng.randint(1, 250000)}@example.com" if rng.random() < 0.85 else "\\N"
            obuf.write(f"{oid}\t{tenant}\t{email}\t{amount}\t{status}\t{odate:%Y-%m-%d %H:%M:%S}\n")
            # items for this order
            remaining_orders = n_orders - (orders_done + j)
            if remaining_orders <= 1:
                k = items_left
            else:
                k = max(0, min(items_left, int(round(rng.gauss(items_per_order, 1.2)))))
                k = max(1, k) if items_left > remaining_orders else k
            for _ in range(k):
                sku = zipf.sample(rng)
                qty = 1 + int(rng.expovariate(0.8))
                price = round(5 + (sku % 500) * 0.37 + rng.random() * 3, 2)
                ibuf.write(f"{det_uuid(rng)}\t{oid}\tSKU-{sku:05d}\t{qty}\t{price}\n")
            items_left -= k
        copy_rows(cur, "orders", ["order_id", "tenant_id", "customer_email", "total_amount", "status", "order_date"], obuf)
        copy_rows(cur, "order_items", ["item_id", "order_id", "product_sku", "quantity", "unit_price"], ibuf)
        conn.commit()
        orders_done += batch
        print(f"[seed] orders {orders_done}/{n_orders}  items {n_items - items_left}/{n_items}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    conn.autocommit = True
    for t in ("tenants", "orders", "order_items"):
        cur.execute(psql.SQL("ANALYZE {}").format(psql.Identifier(t)))
    print(f"[seed] ANALYZE complete. total {time.time() - t0:.1f}s")
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenants", type=int, default=env_int("SEED_TENANTS", 100))
    ap.add_argument("--orders", type=int, default=env_int("SEED_ORDERS", 100_000))
    ap.add_argument("--items", type=int, default=env_int("SEED_ORDER_ITEMS", 300_000))
    ap.add_argument("--seed", type=int, default=env_int("SEED_RANDOM_SEED", 42))
    ap.add_argument("--reset", action="store_true", help="truncate and regenerate")
    ap.add_argument("--if-empty", action="store_true", help="only seed when orders is empty")
    ap.add_argument("--full", action="store_true", help="full demo size 100/1M/3M")
    args = ap.parse_args()
    if args.full:
        args.tenants, args.orders, args.items = 100, 1_000_000, 3_000_000
    if args.tenants < HOT_TENANTS:
        print("[seed] need at least 5 tenants", file=sys.stderr)
        return 2
    seed(args.tenants, args.orders, args.items, args.seed, args.reset, args.if_empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
