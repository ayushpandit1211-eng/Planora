"""Backend client + SSE consumer for the Streamlit dashboard (Section 47).

The dashboard never talks to PostgreSQL directly and never executes SQL; it only calls the
FastAPI backend. SSE is consumed with httpx streaming and parsed per the text/event-stream
format (event: / data: lines separated by a blank line).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterator

import httpx

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")


class BackendClient:
    def __init__(self, base_url: str = DEFAULT_BACKEND, api_key: str | None = None, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.timeout = timeout

    def _req(self, method: str, path: str, **kw) -> tuple[int, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.request(method, self.base + path, headers=self.headers, **kw)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {"error": "non-JSON response"}
        except httpx.HTTPError as exc:
            return 0, {"error": "backend_unreachable", "message": type(exc).__name__}

    def get(self, path: str) -> tuple[int, Any]:
        return self._req("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, Any]:
        return self._req("POST", path, json=body)

    def stream(self, request_id: str, on_event: Callable[[dict], None], timeout_s: float = 300.0) -> list[dict]:
        events: list[dict] = []
        for ev in iter_sse(self.base + f"/api/v1/stream/{request_id}", self.headers, timeout_s):
            if ev.get("event") == "HEARTBEAT":
                continue
            events.append(ev)
            on_event(ev)
            if ev.get("event") in ("FINAL_REPORT_READY", "ERROR", "SECURITY_REJECTED"):
                break
        return events


def parse_sse_lines(lines: Iterator[str]) -> Iterator[dict]:
    """Parse text/event-stream lines into event dicts (data must be JSON)."""
    event_name, data_lines = None, []
    for line in lines:
        if line == "":
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = {"message": "\n".join(data_lines)}
                payload.setdefault("event", event_name)
                yield payload
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)


def iter_sse(url: str, headers: dict, timeout_s: float) -> Iterator[dict]:
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s, read=timeout_s)) as c:
            with c.stream("GET", url, headers=headers) as r:
                if r.status_code != 200:
                    yield {"event": "ERROR", "message": f"stream HTTP {r.status_code}"}
                    return
                yield from parse_sse_lines(r.iter_lines())
    except httpx.HTTPError as exc:
        yield {"event": "ERROR", "message": f"SSE failure: {type(exc).__name__}"}
