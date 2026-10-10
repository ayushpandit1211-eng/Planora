import pytest
from app.errors import PrivacyViolation
from app.llm_guard import OutgoingPayloadGuard
from app.pii import PIIDetector

# This is the "Weekly Sales Dashboard" query from our sample dataset
WEEKLY_SALES_SQL = """
SELECT date_trunc('week', o.order_date) AS week, o.status,
       count(*) AS orders, sum(o.total_amount) AS revenue
FROM orders o
WHERE o.tenant_id = 'bdd640fb-0667-4ad1-9c80-317fa3b1799d'
  AND o.order_date >= '2026-06-01' AND o.order_date < '2026-09-01'
GROUP BY 1, 2
ORDER BY 1, 2;
"""

def test_weekly_sales_dashboard_is_safely_obfuscated(obfuscator):
    """
    Tests that the specific weekly sales query from our dataset
    has its sensitive tenant UUID completely hidden before AI analysis.
    """
    result = obfuscator.mask_sql(WEEKLY_SALES_SQL, "demo_request_001")
    
    # 1. Ensure the PII Detector caught the UUID
    assert result["privacy_audit"]["pii_detected"] is True
    
    # 2. Ensure the real tenant ID is NOT in the SQL that the AI sees
    assert "bdd640fb-0667-4ad1-9c80-317fa3b1799d" not in result["masked_sql"]
    
    # 3. Ensure the dates were also parameterized (hidden)
    assert "2026-06-01" not in result["masked_sql"]
    
    # 4. Ensure it was replaced by a safe parameter (like $1)
    assert "$1" in result["masked_sql"]

def test_ai_prevented_from_seeing_raw_tenant_id(obfuscator):
    """
    Tests that if the AI attempts to receive a payload containing the raw
    tenant ID from the weekly sales query, the payload guard blocks it.
    """
    # Simulate masking the query
    obfuscator.mask_sql(WEEKLY_SALES_SQL, "demo_request_001")
    
    # Initialize the payload guard
    guard = OutgoingPayloadGuard(obfuscator)
    
    # Simulate an AI payload that accidentally includes the raw tenant ID
    unsafe_ai_payload = {
        "sql": "SELECT * FROM orders",
        "tenant_context": "bdd640fb-0667-4ad1-9c80-317fa3b1799d"
    }
    
    # Assert that the guard throws a PrivacyViolation exception, blocking the AI
    with pytest.raises(PrivacyViolation) as exc_info:
        guard.inspect("demo_request_001", "rl_agent", unsafe_ai_payload)
    
    # Ensure it was blocked specifically because it tried to leak a raw literal (the UUID)
    assert "raw_literal" in exc_info.value.details["violations"]
