"""Security gateway (Section 11/60): statement rules, DDL/dangerous rejection, length limit."""
import pytest

from app.errors import ErrorCategory, SecurityRejection
from app.security import EXPLAIN_GENERIC_PREFIX, EXPLAIN_PREFIX, SecurityGateway


@pytest.fixture
def gw():
    return SecurityGateway(max_query_length=500)


def test_simple_select_passes(gw):
    v = gw.validate("SELECT o.order_id FROM orders o WHERE o.status = 'paid';")
    assert v.statement_type == "SelectStmt"
    assert v.referenced_relations == ["orders"]
    assert v.explain_sql.startswith(EXPLAIN_PREFIX)
    assert "ANALYZE" not in v.explain_sql.upper().replace("FORMAT JSON", "")


def test_multiple_statements_rejected(gw):
    with pytest.raises(SecurityRejection) as e:
        gw.validate("SELECT 1; DELETE FROM orders")
    assert e.value.details["rule"] == "single_statement"


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "CREATE INDEX ON orders(status)", "ALTER TABLE orders ADD COLUMN x int",
    "TRUNCATE orders", "CREATE TABLE t (a int)",
])
def test_ddl_rejected(gw, sql):
    with pytest.raises(SecurityRejection) as e:
        gw.validate(sql)
    assert e.value.category == ErrorCategory.UNSAFE_SQL


@pytest.mark.parametrize("sql", [
    "COPY orders TO '/tmp/x'", "SELECT pg_read_file('/etc/passwd')", "SELECT pg_sleep(10)",
    "SELECT * FROM pg_authid", "DELETE FROM orders", "UPDATE orders SET status = 'x'",
    "SELECT * INTO new_t FROM orders", "SELECT * FROM orders FOR UPDATE",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "SET statement_timeout = 0",
    "ALTER SYSTEM SET work_mem = '1GB'", "VACUUM orders", "EXPLAIN ANALYZE SELECT 1",
    "SELECT set_config('x','y',false)", "SELECT nextval('s')", "SELECT dblink('x','y')",
    "SELECT lo_import('/etc/passwd')", "SELECT pg_terminate_backend(1)",
])
def test_dangerous_rejected(gw, sql):
    with pytest.raises(SecurityRejection):
        gw.validate(sql)


def test_query_length_limit(gw):
    with pytest.raises(SecurityRejection) as e:
        gw.validate("SELECT 1 " + " " * 600)
    assert e.value.details["rule"] == "max_query_length"


def test_invalid_sql_does_not_echo_input(gw):
    with pytest.raises(SecurityRejection) as e:
        gw.validate("SELEC secret_value_123 FROM")
    assert e.value.category == ErrorCategory.INVALID_SQL
    assert "secret_value_123" not in e.value.message


def test_empty_and_nul(gw):
    with pytest.raises(SecurityRejection):
        gw.validate("   ")
    with pytest.raises(SecurityRejection):
        gw.validate("SELECT 1\x00")


def test_parameterised_query_uses_generic_plan(gw):
    v = gw.validate("SELECT * FROM orders WHERE status = $1")
    assert v.has_params
    assert v.explain_sql.startswith(EXPLAIN_GENERIC_PREFIX)


def test_cte_select_allowed(gw):
    v = gw.validate("WITH x AS (SELECT * FROM orders) SELECT count(*) FROM x")
    assert v.statement_type == "SelectStmt"
