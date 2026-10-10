"""Event bus / SSE (Section 38/60): creation, request isolation, serialisation, redaction."""
import json

import pytest

from app.events import PIPELINE_EVENTS, EventBus, to_sse
from dashboard.streaming_ui import parse_sse_lines


def test_event_creation_and_contract():
    bus = EventBus()
    ev = bus.publish("r1", "SECURITY_SCAN_COMPLETE", "security", "done")
    d = ev.model_dump()
    assert set(d) >= {"event", "timestamp", "request_id", "stage", "message"}
    with pytest.raises(ValueError):
        bus.publish("r1", "NOT_AN_EVENT", "x", "y")
    assert {"REQUEST_RECEIVED", "FINAL_REPORT_READY", "ERROR", "SANDBOX_STARTED"} <= set(PIPELINE_EVENTS)


def test_request_isolation():
    bus = EventBus()
    bus.publish("a", "REQUEST_RECEIVED", "api", "a")
    bus.publish("b", "REQUEST_RECEIVED", "api", "b")
    assert [e.message for e in bus.history("a")] == ["a"] and [e.message for e in bus.history("b")] == ["b"]


def test_sse_serialisation_roundtrip():
    bus = EventBus()
    ev = bus.publish("r1", "EXPLAIN_COMPLETE", "explain", "cost 12.5", {"total_cost": 12.5})
    wire = to_sse(ev)
    assert wire.startswith("event: EXPLAIN_COMPLETE\ndata: ") and wire.endswith("\n\n")
    parsed = list(parse_sse_lines(iter(wire.split("\n"))))
    assert parsed[0]["event"] == "EXPLAIN_COMPLETE" and parsed[0]["data"]["total_cost"] == 12.5


def test_sensitive_payloads_redacted():
    bus = EventBus()
    ev = bus.publish("r1", "ERROR", "x", "failed for amit@example.com with password=hunter2",
                     {"detail": "user 'secret literal' token eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4fw"})
    blob = json.dumps(ev.model_dump())
    assert "amit@example.com" not in blob and "hunter2" not in blob and "secret literal" not in blob
    assert "eyJhbGciOiJIUzI1" not in blob


async def test_subscribe_replays_and_terminates():
    bus = EventBus()
    bus.publish("r1", "REQUEST_RECEIVED", "api", "start")
    bus.publish("r1", "FINAL_REPORT_READY", "final", "done")
    got = [e.event async for e in bus.subscribe("r1", heartbeat_s=0.05)]
    assert got == ["REQUEST_RECEIVED", "FINAL_REPORT_READY"]
