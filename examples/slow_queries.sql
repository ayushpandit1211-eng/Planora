-- Planora demonstration scenarios (Section 52).
-- The UUIDs below are produced by scripts/seed_data.py with SEED_RANDOM_SEED=42:
--   bdd640fb-0667-4ad1-9c80-317fa3b1799d = "Tenant 0001 Ltd" (one of the 5 hot tenants)
--   337ea2df-b09b-4a5c-badc-c32ac1590f53 = "Tenant 0042 Ltd" (a cold tenant)
-- Emails use the reserved example.com domain and are synthetic.
-- No performance numbers are hard-coded anywhere: run them through the platform and read
-- the planner estimates / benchmark output for your own environment.

-- ===========================================================================
-- Scenario A - Unindexed multi-table join
-- Expected evidence : Seq Scan (often Parallel) on order_items filtered by product_sku,
--                     join to orders via orders_pkey, filter on tenant_id without index
-- Possible indexes  : order_items(product_sku, order_id) ; orders(tenant_id, order_date)
-- Possible rewrite  : none required (join is already flat)
-- Sandbox method    : HypoPG virtual index + EXPLAIN cost comparison
-- Blast radius      : INSERTs into order_items (new orders) from pg_stat_statements
-- Deployment review : CREATE INDEX CONCURRENTLY on staging, check write latency
-- ===========================================================================
SELECT o.order_id, o.order_date, o.total_amount, i.quantity, i.unit_price
FROM orders o
JOIN order_items i ON i.order_id = o.order_id
WHERE o.tenant_id = '337ea2df-b09b-4a5c-badc-c32ac1590f53'
  AND i.product_sku = 'SKU-00042'
ORDER BY o.order_date DESC
LIMIT 50;

-- ===========================================================================
-- Scenario B - Date-range aggregation ("weekly sales reporting dashboard")
-- Expected evidence : Bitmap Heap Scan via idx_orders_order_date with a post-filter on
--                     tenant_id, then Sort + GroupAggregate
-- Possible indexes  : composite orders(tenant_id, order_date) (equality column first),
--                     optionally covering INCLUDE (status, total_amount)
-- Possible rewrite  : none; partitioning advisory (range on order_date) is evaluated
-- Sandbox method    : HypoPG composite index, EXPLAIN comparison
-- Blast radius      : INSERT INTO orders / UPDATE orders SET status statements
-- ===========================================================================
SELECT date_trunc('week', o.order_date) AS week, o.status,
       count(*) AS orders, sum(o.total_amount) AS revenue
FROM orders o
WHERE o.tenant_id = 'bdd640fb-0667-4ad1-9c80-317fa3b1799d'
  AND o.order_date >= '2026-06-01' AND o.order_date < '2026-09-01'
GROUP BY 1, 2
ORDER BY 1, 2;

-- ===========================================================================
-- Scenario C - Nested (correlated) subquery
-- Expected evidence : SubPlan executed per outer row with a Seq Scan on order_items
--                     (foreign key order_items.order_id has no supporting index)
-- Possible indexes  : order_items(order_id)  (FK index)
-- Possible rewrite  : correlated COUNT(*) subquery -> LEFT JOIN on a grouped derived
--                     table with COALESCE(...,0) (semantic diff checked)
-- Sandbox method    : HypoPG FK index + EXPLAIN of original vs rewritten SQL
-- Blast radius      : INSERT INTO order_items
-- ===========================================================================
SELECT o.order_id, o.total_amount,
       (SELECT count(*) FROM order_items i WHERE i.order_id = o.order_id) AS item_count
FROM orders o
WHERE o.status = 'pending'
  AND o.order_date >= '2026-09-01';

-- ===========================================================================
-- Scenario D - Non-sargable date predicate + PII literal
-- Expected evidence : cast on order_date prevents use of idx_orders_order_date
-- Possible rewrite  : order_date::date = X  ->  order_date >= X AND order_date < X + 1 day
-- ===========================================================================
SELECT o.order_id, o.total_amount
FROM orders o
WHERE o.customer_email = 'customer1234@example.com'
  AND o.order_date::date = '2026-08-15';

-- ===========================================================================
-- Scenario E - Efficient query (control case: should report "no major bottleneck")
-- ===========================================================================
SELECT o.order_id, o.status FROM orders o
WHERE o.order_id = '5a6f2a0e-8f8b-4b4e-9d5c-8a1c0b7e3f11';
