"""P7 major tests: index portfolio reasoning."""
from app.index_portfolio import IndexInfo, find_redundancy, plan_actions, workload_coverage

OPC = "{1978}"
COL = "{0}"


def ix(name, cols, **kw):
    return IndexInfo(name=name, table="orders", columns=cols, opclasses="{" + ",".join(["1978"] * len(cols)) + "}",
                     collations="{" + ",".join(["0"] * len(cols)) + "}", **kw)


def test_duplicate_prefix_and_protection_rules():
    idx = [ix("a_dup1", ["tenant_id"]), ix("a_dup2", ["tenant_id"]), ix("comp", ["tenant_id", "order_date"]),
           ix("pk", ["order_id"], is_primary=True, constraint_backed=True),
           ix("pk_like", ["order_id", "status"]),
           ix("partial", ["status"], predicate="(status = 'x')"), ix("st", ["status", "order_date"]),
           IndexInfo(name="lower_email", table="orders", columns=["customer_email"], opclasses="{3126}",
                     collations="{100}"), ix("email2", ["customer_email", "status"])]
    red = {r["index"]: r for r in find_redundancy(idx, {"orders": [["tenant_id"]]})}
    assert "a_dup2" in red and red["a_dup2"]["kind"] == "exact" and "a_dup1" in red   # dup1 is prefix of comp
    assert red["a_dup1"]["kind"] == "prefix" and red["a_dup1"]["supports_fk"]
    assert red["pk"]["protected"]                       # structure-redundant but constraint-backed
    assert "partial" not in red                         # predicate differs
    assert "lower_email" not in red                     # operator class / collation differ
    acts = {a["indexes_affected"][0]: a for a in plan_actions(idx, {"orders": [["tenant_id"]]}, [], 30, {})}
    assert acts["pk"]["action"] == "KEEP_INDEX" and "protected" in acts["pk"]["warning"]
    rm = acts["a_dup2"]
    assert rm["action"] == "REMOVE_REDUNDANT_INDEX"
    assert rm["review"]["migration"] == 'DROP INDEX CONCURRENTLY IF EXISTS public.a_dup2;'
    assert rm["review"]["rollback"].startswith('CREATE INDEX CONCURRENTLY IF NOT EXISTS a_dup2 ON public.orders')
    assert "HUMAN REVIEW REQUIRED" in rm["review"]["label"]


def test_pruning_needs_workload_evidence_and_consolidation_is_suggestion():
    unused = ix("unused_idx", ["status"], idx_scan=0)
    stm = [{"queryid": "1", "query": "SELECT * FROM orders WHERE tenant_id = $1", "calls": 500}]
    thin = plan_actions([unused], {}, stm, stats_age_days=1, table_writes={})
    assert thin[0]["action"] == "KEEP_INDEX" and "too thin" in " ".join(thin[0]["evidence"])
    enough = plan_actions([unused], {}, stm, stats_age_days=30, table_writes={})
    assert enough[0]["action"] == "REMOVE_REDUNDANT_INDEX"
    used = plan_actions([unused], {}, [{"queryid": "2", "query": "SELECT 1 FROM orders WHERE status = $1",
                                        "calls": 500}], 30, {})
    assert used[0]["action"] == "KEEP_INDEX"
    cov = workload_coverage(ix("t", ["tenant_id"]), stm + [{"query": "SELECT 1 FROM orders WHERE status = $1",
                                                             "calls": 500}])
    assert cov["coverage"] == 0.5
    acts = plan_actions([ix("t_single", ["tenant_id"])], {}, stm, 30, {},
                        candidates=['CREATE INDEX ON public.orders USING btree (tenant_id, order_date)'])
    rep = [a for a in acts if a["action"] == "REPLACE_INDEX_PORTFOLIO"][0]
    assert rep["indexes_affected"] == ["t_single"] and "SUGGESTION ONLY" in " ".join(rep["evidence"])
