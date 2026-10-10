"""Natural-language DBA assistant (Section 26) + similar-incident retrieval (Addendum R13).

Natural-language question -> intent extraction -> candidate query/workload selection ->
security boundary -> privacy boundary -> optimisation pipeline.

The assistant NEVER executes SQL derived from natural language and never gains database
access of its own: it only RANKS statements that already exist in pg_stat_statements (already
normalised to $n placeholders) using schema metadata, and the DBA chooses which candidate to
send through the normal gateway + masking + pipeline. Every answer cites the metadata evidence
it used. Returned SQL is anonymised.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any

from app.masking_vault import DataObfuscator
from app.security import SecurityGateway
from app.workload_store import WorkloadStore

INTENT_PATTERNS = {
    "timeout_or_slow": r"\b(slow|timing out|time[- ]?out|timeout|hang|lag|latency|takes? (too )?long|sluggish)\b",
    "regression": r"\b(suddenly|started|regress|got slower|after (the )?(deploy|release|upgrade))\b",
    "write_impact": r"\b(insert|update|write|ingest|load)s?\b.*\b(slow|lock|block)",
    "capacity": r"\b(disk|storage|space|bloat|size)\b",
}
TIME_HINTS = {"weekly": "week", "week": "week", "daily": "day", "day": "day", "monthly": "month",
              "month": "month", "hourly": "hour", "yearly": "year", "annual": "year"}
# Generic business vocabulary -> schema-name fragments. Matching is done against REAL catalog
# names on the trusted backend; only pseudonymised results leave it.
SYNONYMS = {
    "sales": ["amount", "total", "order", "revenue", "price"], "revenue": ["amount", "total", "price"],
    "report": ["date", "time", "created"], "reporting": ["date", "time", "created"], "dashboard": ["date", "status"],
    "customer": ["customer", "email", "tenant", "client"], "customers": ["customer", "email", "tenant"],
    "tenant": ["tenant"], "tenants": ["tenant"], "client": ["tenant", "customer"],
    "product": ["sku", "product", "item"], "products": ["sku", "product", "item"], "items": ["item", "sku"],
    "order": ["order"], "orders": ["order"], "status": ["status"], "email": ["email"],
    "week": ["date"], "weekly": ["date"], "daily": ["date"], "monthly": ["date"],
}
STOPWORDS = {"the", "a", "an", "is", "are", "why", "what", "my", "our", "of", "to", "for", "on", "in", "and",
             "it", "this", "that", "be", "so", "very", "how", "can", "i", "we", "do", "does", "with"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if t not in STOPWORDS]


class NLIntentAssistant:
    def __init__(self, instrumentation, obfuscator: DataObfuscator, store: WorkloadStore,
                 known_tables: set[str] | None = None):
        self.instr = instrumentation
        self.obf = obfuscator
        self.store = store
        self.known_tables = known_tables or set()
        self._candidates: dict[str, tuple[float, str]] = {}   # candidate_id -> (expiry, trusted query text)

    def extract_intent(self, question: str, schema_names: dict[str, list[str]]) -> dict[str, Any]:
        q = question.lower()
        intents = [name for name, pat in INTENT_PATTERNS.items() if re.search(pat, q)] or ["general_performance"]
        toks = _tokens(question)
        frags: set[str] = set()
        for t in toks:
            frags.update(SYNONYMS.get(t, []))
            frags.add(t.rstrip("s"))
        matched_tables, matched_columns = [], []
        for table, cols in schema_names.items():
            if any(f and f in table for f in frags):
                matched_tables.append(table)
            for c in cols:
                if any(f and len(f) >= 3 and f in c for f in frags):
                    matched_columns.append(f"{table}.{c}")
        time_hint = next((TIME_HINTS[t] for t in toks if t in TIME_HINTS), None)
        return {"intents": intents, "keywords": toks, "time_granularity": time_hint,
                "matched_tables": sorted(set(matched_tables)), "matched_columns": sorted(set(matched_columns))}

    async def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        rid = f"nl-{uuid.uuid4().hex[:10]}"
        cols = await self.instr.columns(None)
        schema_names: dict[str, list[str]] = {}
        for c in cols:
            schema_names.setdefault(c["table_name"], []).append(c["column_name"])
        intent = self.extract_intent(question, schema_names)
        try:
            stmts = await self.instr.statements(300)
            source = "pg_stat_statements"
        except Exception:  # noqa: BLE001
            stmts, source = [], "unavailable"
        max_time = max((s["total_exec_time"] or 0 for s in stmts), default=1) or 1
        scored = []
        for s in stmts:
            text = s["query"] or ""
            if SecurityGateway.classify_statement(text) != "SelectStmt":
                continue
            low = text.lower()
            if any(x in low for x in ("pg_catalog", "pg_stat", "hypopg", "information_schema", "pg_class",
                                      "pg_settings", "pg_attribute", "pg_index", "explain")):
                continue
            ev, score = [], 0.0
            for t in intent["matched_tables"]:
                if re.search(rf"\b{re.escape(t)}\b", low):
                    score += 2
                    ev.append(f"references table matched from question ({t})")
            for tc in intent["matched_columns"]:
                col = tc.split(".")[1]
                if re.search(rf"\b{re.escape(col)}\b", low):
                    score += 1
                    ev.append(f"uses column matched from question ({tc})")
            if intent["time_granularity"] and "date_trunc" in low:
                score += 3
                ev.append(f"time-bucketed aggregation (date_trunc) matches '{intent['time_granularity']}' reporting")
            elif intent["time_granularity"] and "group by" in low:
                score += 1
                ev.append("grouped aggregation matches a reporting question")
            if any(k in intent["keywords"] for k in ("sales", "revenue")) and re.search(r"\b(sum|count)\s*\(", low):
                score += 1.5
                ev.append("aggregates (sum/count) match a sales/revenue question")
            weight = (s["total_exec_time"] or 0) / max_time
            score += 2 * weight
            ev.append(f"share of total execution time rank weight {weight:.2f} (calls={s['calls']}, "
                      f"mean={s['mean_exec_time']:.2f} ms)")
            if score <= 2 * weight:      # nothing but runtime matched the question
                continue
            scored.append((score, s, ev))
        scored.sort(key=lambda x: -x[0])
        out = []
        now = time.time()
        self._candidates = {k: v for k, v in self._candidates.items() if v[0] > now}
        for score, s, ev in scored[:top_k]:
            masked = self.obf.mask_sql(s["query"], rid)
            if not masked["privacy_audit"]["safe_for_llm"]:
                continue
            cid = f"cand-{uuid.uuid4().hex[:10]}"
            self._candidates[cid] = (now + 900, s["query"])
            out.append({"candidate_id": cid, "anonymized_sql": masked["masked_sql"], "score": round(score, 3),
                        "calls": s["calls"], "mean_exec_time_ms": round(s["mean_exec_time"] or 0, 3),
                        "total_exec_time_ms": round(s["total_exec_time"] or 0, 1),
                        "evidence": ev, "evidence_kind": "observed (pg_stat_statements counters)"})
        self.obf.clear_vault(rid)
        incidents = self.similar_incidents(intent)
        answer = (f"Identified {len(out)} candidate workload statement(s) from {source} matching "
                  f"{', '.join(intent['matched_tables']) or 'no specific table'}"
                  + (f" with {intent['time_granularity']}ly time bucketing" if intent["time_granularity"] else "")
                  + ". Select a candidate to run the full privacy-preserving optimisation pipeline."
                  if out else "No existing workload statement matched the question; submit the slow SQL directly.")
        return {"question_intent": {k: v for k, v in intent.items() if k != "keywords"}, "candidates": out,
                "similar_incidents": incidents, "answer": answer, "source": source,
                "boundary": "No SQL is generated or executed from natural language; candidates are existing, "
                            "normalised workload statements that still pass the security gateway and masking."}

    def resolve_candidate(self, candidate_id: str) -> str | None:
        item = self._candidates.get(candidate_id)
        if not item or item[0] < time.time():
            return None
        return item[1]

    def similar_incidents(self, intent: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        """R13: anonymised historical records similar to the question (shared tables / shape)."""
        recs = self.store.all()
        if not recs:
            return []
        want_time = intent.get("time_granularity") is not None
        scored = []
        for r in recs:
            shape = (r.plan_shape or "")
            s = 0.0
            if want_time and ("Aggregate" in shape or "Sort" in shape):
                s += 1
            if r.mean_exec_time_ms:
                s += min(r.mean_exec_time_ms / 100.0, 2.0)
            if "Seq Scan" in shape:
                s += 0.5
            scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for s, r in scored:
            if r.query_fingerprint in seen:
                continue
            seen.add(r.query_fingerprint)
            out.append({"query_fingerprint": r.query_fingerprint, "plan_fingerprint": r.plan_fingerprint,
                        "plan_shape": r.plan_shape, "timestamp_bucket": r.timestamp_bucket,
                        "mean_exec_time_ms": r.mean_exec_time_ms, "anonymized_sql": r.anonymized_sql,
                        "similarity_score": round(s, 3)})
            if len(out) >= limit:
                break
        return out
