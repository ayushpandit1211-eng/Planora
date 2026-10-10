"""Privacy vault, PII filter, bitmask and the zero-leak LLM guard (Sections 12-15, 60)."""
import pytest

from app.errors import ErrorCategory, PrivacyViolation
from app.llm_guard import OutgoingPayloadGuard
from app.masking_vault import DataObfuscator, InMemoryVaultBackend, PSEUDONYM_RE
from app.pii import PIIDetector, PrivacyFlag, decode_bitmask

SQL = ("SELECT o.order_id FROM orders o WHERE o.customer_email = 'amit@example.com' "
       "AND o.total_amount > 5000 AND o.order_date >= '2026-10-01' "
       "AND o.tenant_id = 'bc8960a9-23b8-41e9-b924-56de3eb13b90'")


def test_literals_parameterized(obfuscator):
    r = obfuscator.mask_sql(SQL, "r1")
    assert r["literal_count"] == 4
    for lit in ("amit@example.com", "5000", "2026-10-01", "bc8960a9"):
        assert lit not in r["masked_sql"]
    assert "$1" in r["masked_sql"] and "$4" in r["masked_sql"]
    # numbered in textual order
    assert r["masked_sql"].index("$1") < r["masked_sql"].index("$2") < r["masked_sql"].index("$3")


def test_identifier_pseudonymization_is_deterministic_hmac(obfuscator):
    a = obfuscator.mask_sql("SELECT tenant_id FROM orders", "r1")["masked_sql"]
    b = obfuscator.mask_sql("SELECT tenant_id FROM orders", "r2")["masked_sql"]
    assert a == b
    assert "orders" not in a and "tenant_id" not in a
    assert {m.group(1) for m in PSEUDONYM_RE.finditer(a)} == {"tbl", "col"}
    other = DataObfuscator(InMemoryVaultBackend(), hmac_key=b"other-key" * 4)
    assert other.mask_sql("SELECT tenant_id FROM orders", "x")["masked_sql"] != a


def test_reverse_translation(obfuscator):
    r = obfuscator.mask_sql(SQL, "r1")
    assert obfuscator.translate_llm_sql_to_real(r["masked_sql"], "r1").count("amit@example.com") == 1
    tbl = next(k for k in r["mapped_tables"])
    col = next(k for k in r["mapped_columns"])
    ddl = obfuscator.translate_llm_ddl_to_real(f"CREATE INDEX ON {tbl} ({col})", "r1")
    assert ddl.startswith("CREATE INDEX ON orders (")


def test_unknown_identifier_rejected(obfuscator):
    obfuscator.mask_sql(SQL, "r1")
    with pytest.raises(PrivacyViolation) as e:
        obfuscator.translate_llm_ddl_to_real("CREATE INDEX ON tbl_deadbe (col_abcdef)", "r1")
    assert e.value.category == ErrorCategory.UNKNOWN_IDENTIFIER


def test_translation_only_allows_create_index(obfuscator):
    r = obfuscator.mask_sql(SQL, "r1")
    tbl = next(k for k in r["mapped_tables"])
    with pytest.raises(PrivacyViolation):
        obfuscator.translate_llm_ddl_to_real(f"DROP TABLE {tbl}", "r1")


def test_request_isolation_and_clear(obfuscator):
    obfuscator.mask_sql(SQL, "r1")
    obfuscator.mask_sql("SELECT 1 FROM tenants WHERE name = 'zz top'", "r2")
    v1, v2 = obfuscator.get_vault("r1"), obfuscator.get_vault("r2")
    assert "amit@example.com" in v1.literal_values and "amit@example.com" not in v2.literal_values
    obfuscator.clear_vault("r1")
    assert obfuscator.get_vault("r1") is None
    with pytest.raises(PrivacyViolation):
        obfuscator.translate_llm_sql_to_real("SELECT 1", "r1")


def test_mask_result_never_contains_reverse_map(obfuscator):
    r = obfuscator.mask_sql(SQL, "r1")
    assert "orders" not in str(r["mapped_tables"]) and "customer_email" not in str(r["mapped_columns"])


def test_ordinals_kept(obfuscator):
    r = obfuscator.mask_sql("SELECT status, count(*) FROM orders GROUP BY 1 ORDER BY 2 DESC", "r1")
    assert "GROUP BY 1" in r["masked_sql"] and "ORDER BY 2" in r["masked_sql"]


def test_fallback_quarantines_unsafe_constructs(obfuscator):
    r = obfuscator.mask_sql("SELECT $q$secret$q$ FROM orders WHERE (", "r1")
    assert r["quarantined"] and not r["privacy_audit"]["safe_for_llm"]


def test_pii_detection_types():
    s = PIIDetector().scan("mail amit@example.com uuid bc8960a9-23b8-41e9-b924-56de3eb13b90 "
                           "phone +91 98765 43210 token eyJhbGciOiJI.eyJzdWIiOiIx.SflKxwRJSMeKKF2QT4 "
                           "password=hunter22 key AKIAIOSFODNN7EXAMPLE")
    assert {"email", "uuid", "phone"} <= set(s.pii_types)
    assert {"jwt", "password_assignment", "api_key"} <= set(s.secret_types)


def test_uuid_and_email_queries_not_rejected(obfuscator):
    r = obfuscator.mask_sql("SELECT * FROM orders WHERE customer_email = 'a@b.co'", "r1")
    assert r["privacy_audit"]["pii_detected"] and r["privacy_audit"]["safe_for_llm"] and not r["quarantined"]


def test_bitmask_flags(obfuscator):
    r = obfuscator.mask_sql("SELECT t.tier, count(*) FROM orders o JOIN tenants t ON t.tenant_id = o.tenant_id "
                            "WHERE o.customer_email = 'x@y.io' GROUP BY 1 ORDER BY 2 LIMIT 5", "r1")
    flags = decode_bitmask(r["privacy_bitmask"])
    for f in ("contains_email_literal", "contains_join", "contains_aggregation", "contains_sort", "contains_limit"):
        assert f in flags
    assert r["privacy_bitmask"] & PrivacyFlag.CONTAINS_JOIN


# --------------------------------------------------------------- outgoing LLM guard
def test_guard_allows_clean_payload(obfuscator):
    r = obfuscator.mask_sql(SQL, "r1")
    g = OutgoingPayloadGuard(obfuscator)
    assert g.check("r1", "template", {"sql": r["masked_sql"]}).allowed


@pytest.mark.parametrize("payload,violation", [
    ({"sql": "... 'amit@example.com' ..."}, "pii:email"),
    ({"note": "literal 2026-10-01"}, "raw_literal"),
    ({"hint": "table orders is large"}, "reverse_map_identifier"),
    ({"k": "sk-abcdefghijklmnopqrstuvwxyz123456"}, "secret:api_key"),
    ({"k": "password: hunter2!"}, "secret:password_assignment"),
])
def test_guard_blocks_and_audits(obfuscator, payload, violation):
    from app.audit import AuditLogger
    audit = AuditLogger()
    obfuscator.mask_sql(SQL, "r1")
    g = OutgoingPayloadGuard(obfuscator, audit)
    with pytest.raises(PrivacyViolation) as e:
        g.inspect("r1", "template", payload)
    assert violation in e.value.details["violations"]
    ev = audit.events_for("r1")
    assert ev and ev[-1].event_type == "PRIVACY_BLOCK" and ev[-1].severity == "security"


def test_guard_blocks_raw_result_values(obfuscator):
    obfuscator.mask_sql(SQL, "r1")
    g = OutgoingPayloadGuard(obfuscator)
    assert not g.check("r1", "", {"rows": ["Tenant 0001 Ltd"]}, extra_forbidden=["Tenant 0001 Ltd"]).allowed
