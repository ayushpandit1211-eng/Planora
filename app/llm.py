"""Environment-configurable LLM client (Sections 27-28).

Providers: none (default, deterministic agents only) | gemini | openai (any OpenAI-compatible
endpoint) | mock (tests). Every call:
  1. builds a prompt that carries the LLM privacy contract,
  2. passes the outgoing payload through OutgoingPayloadGuard,
  3. validates the JSON reply against LLMRecommendation (Pydantic) before use.
No API key is hardcoded; keys come from LLM_API_KEY and are sent in headers, never URLs.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.contracts import LLMRecommendation
from app.errors import ErrorCategory, QuerySentinelError
from app.llm_guard import OutgoingPayloadGuard

PRIVACY_CONTRACT = (
    "You receive anonymized SQL and anonymized metadata only.\n"
    "Do not request raw values.\n"
    "Do not request secrets.\n"
    "Do not request credentials.\n"
    "Do not invent schema.\n"
    "Do not claim measured performance without measurement.\n"
    "Separate observations, assumptions, recommendations, and risks.\n"
)

RESPONSE_SCHEMA_HINT = json.dumps({
    "rewritten_sql": "string or null (anonymized identifiers and $n parameters only)",
    "proposed_indexes": [{"ddl": "CREATE INDEX ON tbl_x (col_y)", "reason": "...", "columns": ["col_y"]}],
    "partitioning_recommendation": None, "sharding_recommendation": None,
    "technical_reasoning": "...", "assumptions": [], "risks": [], "confidence": None,
})


@dataclass
class LLMCallResult:
    provider: str
    model: str | None
    success: bool
    recommendation: LLMRecommendation | None = None
    error: str | None = None
    error_category: str | None = None
    attempts: int = 0
    latency_ms: float | None = None
    blocked_by_guard: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "success": self.success,
                "error": self.error, "error_category": self.error_category, "attempts": self.attempts,
                "latency_ms": self.latency_ms, "blocked_by_guard": self.blocked_by_guard}


def build_prompt(task: str, data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    template = (
        "ROLE: PostgreSQL performance assistant for a DBA decision-support tool.\n"
        + PRIVACY_CONTRACT
        + f"TASK: {task}\n"
        "Use only identifiers that appear in the provided anonymized metadata.\n"
        "Reply with a single JSON object exactly matching this schema:\n" + RESPONSE_SCHEMA_HINT + "\n"
        "INPUT (anonymized JSON):\n"
    )
    return template, data


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[start:end + 1])


class LLMClient:
    def __init__(self, settings: Settings | None = None, guard: OutgoingPayloadGuard | None = None,
                 mock_responder: Callable[[str, dict], str] | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings or get_settings()
        self.provider = (self.settings.llm_provider or "none").lower()
        self.model = self.settings.llm_model or {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"}.get(self.provider)
        self.guard = guard
        self.mock_responder = mock_responder
        self.transport = transport

    @property
    def enabled(self) -> bool:
        if self.provider == "mock":
            return self.mock_responder is not None
        if self.provider in ("gemini", "openai"):
            return bool(self.settings.llm_api_key.get_secret_value()) or bool(self.settings.llm_base_url)
        return False

    def status(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "enabled": self.enabled,
                "api_key_configured": bool(self.settings.llm_api_key.get_secret_value())}

    # -------------------------------------------------------------- transport
    def _send(self, template: str, data: dict[str, Any]) -> str:
        prompt = template + json.dumps(data, sort_keys=True)
        if self.provider == "mock":
            assert self.mock_responder is not None
            return self.mock_responder(template, data)
        key = self.settings.llm_api_key.get_secret_value()
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)
        with httpx.Client(timeout=timeout, transport=self.transport) as client:
            if self.provider == "gemini":
                base = self.settings.llm_base_url or "https://generativelanguage.googleapis.com/v1beta"
                resp = client.post(
                    f"{base}/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": key, "content-type": "application/json"},
                    json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}},
                )
                resp.raise_for_status()
                body = resp.json()
                return body["candidates"][0]["content"]["parts"][0]["text"]
            base = self.settings.llm_base_url or "https://api.openai.com/v1"
            headers = {"content-type": "application/json"}
            if key:
                headers["authorization"] = f"Bearer {key}"
            resp = client.post(f"{base.rstrip('/')}/chat/completions", headers=headers,
                               json={"model": self.model, "temperature": 0.1,
                                     "response_format": {"type": "json_object"},
                                     "messages": [{"role": "system", "content": template},
                                                  {"role": "user", "content": json.dumps(data, sort_keys=True)}]})
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # -------------------------------------------------------------- public
    def recommend(self, request_id: str, task: str, data: dict[str, Any],
                  max_retries: int | None = None) -> LLMCallResult:
        result = LLMCallResult(provider=self.provider, model=self.model, success=False)
        if not self.enabled:
            result.error = "LLM provider not configured (deterministic agents used)"
            result.error_category = ErrorCategory.LLM_UNAVAILABLE.value
            return result
        template, payload = build_prompt(task, data)
        if self.guard is not None:
            try:
                self.guard.inspect(request_id, template, payload)
            except QuerySentinelError as exc:
                result.blocked_by_guard = True
                result.error = exc.message
                result.error_category = exc.category.value
                return result
        retries = self.settings.llm_max_retries if max_retries is None else max_retries
        t0 = time.perf_counter()
        for attempt in range(1, retries + 2):
            result.attempts = attempt
            try:
                text = self._send(template, payload)
                result.recommendation = LLMRecommendation.model_validate(_extract_json(text))
                result.success = True
                result.error = None
                result.error_category = None
                break
            except httpx.TimeoutException:
                result.error, result.error_category = "LLM request timed out", ErrorCategory.LLM_TIMEOUT.value
            except httpx.HTTPStatusError as exc:
                result.error = f"LLM HTTP {exc.response.status_code}"
                result.error_category = ErrorCategory.LLM_UNAVAILABLE.value
            except httpx.HTTPError as exc:
                result.error, result.error_category = f"LLM transport error: {type(exc).__name__}", ErrorCategory.LLM_UNAVAILABLE.value
            except (ValueError, ValidationError, KeyError, IndexError, json.JSONDecodeError):
                result.error, result.error_category = "malformed LLM JSON", ErrorCategory.LLM_MALFORMED.value
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # -------------------------------------------------------------- Final Upgrade P4 (additive)
    def complete_json(self, template: str, data: dict[str, Any]) -> dict[str, Any]:
        """Raw JSON completion for the NL-to-SQL generator (its own privacy rules are applied by the caller:
        schema metadata only, no row values, PII-free question). Raises QuerySentinelError on failure."""
        if not self.enabled:
            raise QuerySentinelError(ErrorCategory.LLM_UNAVAILABLE, "LLM provider not configured", stage="llm")
        try:
            return _extract_json(self._send(template, data))
        except httpx.TimeoutException:
            raise QuerySentinelError(ErrorCategory.LLM_TIMEOUT, "LLM request timed out", stage="llm") from None
        except httpx.HTTPError as exc:
            raise QuerySentinelError(ErrorCategory.LLM_UNAVAILABLE, f"LLM unavailable ({type(exc).__name__})",
                                     stage="llm") from None
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            raise QuerySentinelError(ErrorCategory.LLM_MALFORMED, "malformed LLM JSON", stage="llm") from None
