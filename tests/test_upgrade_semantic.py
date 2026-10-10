"""P6 major tests: semantic-risk screening of rewrites."""
from app.agents.sql_rewriter import SQLRewriter
from app.sql_semantic_diff import risk_screen

BASE = "SELECT a.x, count(*) AS n FROM t a JOIN u b ON b.id = a.id WHERE a.y = 1 GROUP BY a.x ORDER BY n DESC LIMIT 10"


def test_risk_classes():
    assert risk_screen(BASE, BASE.replace("SELECT", "select"))["risk"] == "LOW"           # reformatting
    cases = {
        "output column": BASE.replace("a.x, count(*)", "a.x, a.z, count(*)").replace("GROUP BY a.x", "GROUP BY a.x, a.z"),
        "grouping": BASE.replace("GROUP BY a.x", "GROUP BY a.x, a.y"),
        "predicate": BASE.replace("WHERE a.y = 1", "WHERE a.y > 0"),
        "join type": BASE.replace("JOIN u b", "LEFT JOIN u b"),
        "limit": BASE.replace("LIMIT 10", "LIMIT 100"),
        "aggregate": BASE.replace("count(*)", "count(DISTINCT a.y)"),
    }
    for name, sql in cases.items():
        r = risk_screen(BASE, sql)
        assert r["risk"] == "HIGH" and r["blocked"], name
    assert "not a formal equivalence proof" in r["terminology"]


def test_existing_rewrite_flow_still_works_and_is_screened():
    sql = "SELECT tbl_a.col_b FROM tbl_a WHERE tbl_a.col_c = $1 OR tbl_a.col_c = $2 OR tbl_a.col_c = $3"
    prop = SQLRewriter().apply(sql)
    assert prop is not None and "or_equalities_to_in_list" in prop.rules_applied
    scr = prop.semantic_diff["risk_screening"]
    assert scr["risk"] in ("LOW", "MEDIUM") and not scr["blocked"]
