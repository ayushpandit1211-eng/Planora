"""Reusable Streamlit components: status bar, badges, agent tree, evidence chain, tables.

V4 "forest glass" presentation layer: one app-owned stylesheet (CSS), plus small HTML display helpers
(page header, summary strip, dark activity panel, capability list). Helpers only render data they are given;
they never call the backend. Every dynamic string is html-escaped.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

# status colours (text label always shown next to the colour)
STATUS_COLORS = {"QUEUED": "#6B6F76", "RUNNING": "#18243A", "COMPLETED": "#1D654D", "SKIPPED": "#6B6F76",
                 "REJECTED": "#805308", "ERROR": "#AC2F3E"}
KIND_COLORS = {"observed": "#1D654D", "inferred": "#6E4F7A", "simulated": "#18243A", "estimated": "#805308",
               "benchmark-measured": "#A64F20"}

# Local, decorative background and styles. No network calls or application state.
from pathlib import Path
import base64
from functools import lru_cache

@lru_cache(maxsize=1)
def _forest_css(style_version: float) -> str:
    assets = Path(__file__).resolve().parent / "assets"
    photo = base64.b64encode((assets / "forest-dusk.png").read_bytes()).decode("ascii")
    style = (assets / "forest.css").read_text(encoding="utf-8")
    return "<style>" + style.replace("__FOREST_IMAGE__", "data:image/png;base64," + photo) + "</style>"


# small inline SVG glyphs for the white icon holders (local, no network, decorative -> aria-hidden)
_ICON = {
    "cost": '<path d="M4 15a8 8 0 1 1 16 0" fill="none" stroke="#F1BC70" stroke-width="2" stroke-linecap="round"/>'
            '<path d="M12 15l4-5" stroke="#C0D9E4" stroke-width="2" stroke-linecap="round"/>',
    "write": '<circle cx="12" cy="12" r="8" fill="none" stroke="#F1BC70" stroke-width="2"/>'
             '<path d="M12 7v5l3 2" fill="none" stroke="#C0D9E4" stroke-width="2" stroke-linecap="round"/>',
    "storage": '<ellipse cx="12" cy="6.5" rx="7" ry="2.8" fill="none" stroke="#F1BC70" stroke-width="2"/>'
               '<path d="M5 6.5v11c0 1.6 3.1 2.8 7 2.8s7-1.2 7-2.8v-11" fill="none" stroke="#C0D9E4" stroke-width="2"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z" fill="none" stroke="#F1BC70" stroke-width="2"/>'
              '<path d="M9 12l2 2 4-4" fill="none" stroke="#C0D9E4" stroke-width="2" stroke-linecap="round"/>',
    "db": '<ellipse cx="12" cy="6.5" rx="7" ry="2.8" fill="none" stroke="#C0D9E4" stroke-width="2"/>'
          '<path d="M5 6.5v11c0 1.6 3.1 2.8 7 2.8s7-1.2 7-2.8v-11M5 12c0 1.6 3.1 2.8 7 2.8s7-1.2 7-2.8" fill="none" '
          'stroke="#F1BC70" stroke-width="2"/>',
}

AGENT_LABELS = {"supervisor": "Supervisor", "sql_rewriter": "SQL rewriter", "schema_expert": "Schema expert",
                "index_strategist": "Index strategist", "rl_optimizer": "RL optimizer",
                "recommendation_validator": "Recommendation validator", "hypopg_sandbox": "HypoPG simulation",
                "blast_radius": "Blast radius", "final_report": "Final report"}


def icon(name: str, size: int = 18) -> str:
    return (f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24">{_ICON.get(name, "")}'
            "</svg>")


def fmt_sql(sql: str | None) -> str | None:
    """Display-only line breaks before major clauses (does not change the SQL)."""
    if not sql:
        return sql
    import re
    out = re.sub(r"\s+(FROM|WHERE|LEFT JOIN|INNER JOIN|JOIN|GROUP BY|ORDER BY|LIMIT|HAVING)\s+", r"\n\1 ", sql)
    return re.sub(r"\s+(AND|OR)\s+", r"\n  \1 ", out)


def compact(v: float | None) -> str:
    if v is None:
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:.2f}"


def inject_css() -> None:
    st.markdown(_forest_css((Path(__file__).resolve().parent / "assets/forest.css").stat().st_mtime), unsafe_allow_html=True)


# ------------------------------------------------------------------ V3 layout helpers (display only)
def brand() -> None:
    st.markdown('<div class="qs-brand"><div class="mark" aria-hidden="true">P</div>'
                '<div class="name">Planora</div></div>'
                '<div class="qs-brand-sub">PostgreSQL query analysis. Suggests fixes; nothing changes until a '
                'person approves.</div>', unsafe_allow_html=True)


def group_label(text: str) -> None:
    st.markdown(f'<div class="qs-group">{html.escape(text)}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="qs-section">{html.escape(title)}</div>'
                + (f'<div class="qs-section-sub">{html.escape(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def page_header(source: dict | None) -> None:
    """Dashboard title + truthful current-source context (no user account, no avatar)."""
    src = source or {}
    kind = {"demo": "Demo database", "dynamic": "Uploaded dataset"}.get(src.get("kind", "demo"), src.get("kind", ""))
    st.markdown(
        '<div class="qs-topbar"><a class="qs-wordmark" href="#workspace"><span class="qs-gem">◆</span> Planora</a>'
        '<nav aria-label="Dashboard sections"><a href="#workspace">Workspace</a><a href="#your-data">Your data</a>'
        '<a href="#report-views">Reports</a></nav></div>'
        '<div class="qs-head"><div><h1>Your queries.<br><span>Clearly understood.</span></h1>'
        '<div class="lede">Inspect execution plans. Review privacy. Optimize with evidence.</div></div>'
        f'<div class="qs-source" title="active data source id: {html.escape(str(src.get("id", "demo")))}">'
        f'<div class="ic">{icon("db", 18)}</div><div><div class="t">{html.escape(str(src.get("name", "Demo Database")))}'
        f'</div><div class="s">{html.escape(kind)} · {html.escape(str(src.get("id", "demo")))}</div></div></div></div>',
        unsafe_allow_html=True)


def summary_strip(tiles: list[dict] | None) -> None:
    """Shared metric strip. tiles=None -> neutral pre-run description (no invented numbers)."""
    if not tiles:
        st.markdown('<div class="qs-strip" role="group" aria-label="Report summary"><div class="qs-strip-empty">'
                    'No report yet. Run the optimisation pipeline to see the planner cost change, extra write time, '
                    'index storage and raw-data exposure for your query.</div></div>', unsafe_allow_html=True)
        return
    cells = "".join(
        f'<div class="qs-tile" title="{html.escape(t.get("help") or "")}"><div class="qs-ico">{icon(t["icon"])}</div>'
        f'<div><div class="qs-tl">{html.escape(t["label"])}</div><div class="qs-tv">{html.escape(str(t["value"]))}</div>'
        f'<div class="qs-tn">{html.escape(t.get("note", ""))}</div></div></div>' for t in tiles)
    st.markdown(f'<div class="qs-strip" role="group" aria-label="Report summary">{cells}</div>',
                unsafe_allow_html=True)


def activity_html(statuses: dict[str, str], events: list[dict], request_id: str | None = None,
                  started: bool = True) -> str:
    """Dark pipeline-activity card from existing agent statuses and SSE events (escaped)."""
    rows = "".join(
        f'<li class="st-{html.escape(statuses.get(a, "QUEUED"))}"><span>{html.escape(lab)}</span>'
        f'<span class="lab">{html.escape(statuses.get(a, "QUEUED").title())}</span></li>'
        for a, lab in AGENT_LABELS.items())
    head = ('<h3>Pipeline activity</h3>'
            + (f'<div class="rid">run {html.escape(request_id)}</div>' if request_id else ""))
    if not started:
        return (f'<div class="qs-activity" role="region" aria-label="Pipeline activity">{head}'
                '<div class="empty">No run started. Stages appear here when you run the optimisation '
                f'pipeline.</div><ol>{rows}</ol></div>')
    ev = "".join(f'<div>[{html.escape((e.get("timestamp") or "")[11:19])}] {html.escape(str(e.get("message") or ""))}</div>'
                 for e in events[-6:])
    return (f'<div class="qs-activity" role="region" aria-label="Pipeline activity">{head}<ol>{rows}</ol>'
            + (f'<div class="ev" aria-label="latest events">{ev}</div>' if ev else "") + "</div>")


def capability_panel(status: dict | None, error: str | None = None) -> None:
    """Capability facts from the REAL /api/v1/status response (same states as status_bar)."""
    if status is None:
        st.markdown('<div class="qs-facts"><h3>Capabilities</h3><div class="qs-fact bad"><div><div class="k">Backend'
                    f'</div><div class="v">Unreachable</div></div><span class="st">{html.escape(error or "unavailable")}'
                    '</span></div></div>', unsafe_allow_html=True)
        return

    def ok(key):
        return bool((status.get(key) or {}).get("ok"))
    pg, gnn, rl, llm = (status.get(k, {}) or {} for k in ("postgresql", "gnn", "rl", "llm"))
    facts = [
        ("PostgreSQL", (pg.get("version", "") or "-").split(" (")[0], "ok" if ok("postgresql") else "bad",
         "Online" if ok("postgresql") else "Unavailable"),
        ("HypoPG", str((status.get("hypopg") or {}).get("version", "") or "-"), "ok" if ok("hypopg") else "bad",
         "Online" if ok("hypopg") else "Unavailable"),
        ("pg_stat_statements", "workload statistics", "ok" if ok("pg_stat_statements") else "bad",
         "Online" if ok("pg_stat_statements") else "Unavailable"),
        ("Privacy vault", (status.get("privacy_vault") or {}).get("backend", "?"), "ok" if ok("privacy_vault") else "bad",
         "Online" if ok("privacy_vault") else "Unavailable"),
        ("GNN", gnn.get("mode", "?"), "ok" if gnn.get("mode") == "trained" else "warn",
         "Trained" if gnn.get("mode") == "trained" else "Fallback"),
        ("RL", rl.get("mode", "?"), "ok" if rl.get("mode") == "trained_policy" else "warn",
         "Trained" if rl.get("mode") == "trained_policy" else "Fallback"),
        ("LLM", llm.get("provider", "none") + (f"/{llm.get('model')}" if llm.get("enabled") else ""),
         "ok" if llm.get("enabled") else "warn", "Enabled" if llm.get("enabled") else "Disabled"),
        ("LangGraph", "agent orchestration", "ok" if ok("langgraph") else "bad",
         "Online" if ok("langgraph") else "Unavailable"),
    ]
    rows = "".join(f'<div class="qs-fact {c}"><div class="k">{html.escape(k)}</div><div class="v">'
                   f'{html.escape(str(v))}</div><span class="st">{html.escape(s)}</span></div>'
                   for k, v, c, s in facts)
    st.markdown(f'<div class="qs-facts" role="region" aria-label="Capabilities"><h3>Capabilities</h3>'
                f'<div class="qs-factgrid">{rows}</div>'
                '<div class="qs-tn" style="margin-top:8px">From the backend status check on this page load. '
                'LLM disabled is normal: everything runs locally.</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ existing helpers (restyled)
def chip(label: str, state: str, detail: str = "") -> str:
    cls = {"ok": "qs-ok", "bad": "qs-bad", "warn": "qs-warn"}[state]
    tip = f' title="{html.escape(detail)}"' if detail else ""
    return f'<span class="qs-chip {cls}"{tip}>{html.escape(label)}</span>'


def status_bar(status: dict | None, error: str | None = None) -> None:
    """Top status bar from REAL backend responses (Section 42) - no fake indicators."""
    if status is None:
        st.markdown(chip(f"Backend unreachable: {error or ''}", "bad"), unsafe_allow_html=True)
        return
    def ok(key):
        return bool((status.get(key) or {}).get("ok"))
    pg = status.get("postgresql", {})
    gnn = status.get("gnn", {})
    rl = status.get("rl", {})
    llm = status.get("llm", {})
    chips = [
        chip(f"PostgreSQL {pg.get('version', '')}".strip(), "ok" if ok("postgresql") else "bad", pg.get("error", "")),
        chip("HypoPG " + str((status.get('hypopg') or {}).get('version', '')), "ok" if ok("hypopg") else "bad"),
        chip("pg_stat_statements", "ok" if ok("pg_stat_statements") else "bad"),
        chip(f"Privacy Vault ({(status.get('privacy_vault') or {}).get('backend', '?')})", "ok" if ok("privacy_vault") else "bad"),
        chip(f"LLM: {llm.get('provider', 'none')}" + (f"/{llm.get('model')}" if llm.get('enabled') else " (disabled)"),
             "ok" if llm.get("enabled") else "warn"),
        chip(f"GNN: {gnn.get('mode', '?')}", "ok" if gnn.get("mode") == "trained" else "warn", str(gnn.get("error") or "")),
        chip(f"RL: {rl.get('mode', '?')}", "ok" if rl.get("mode") == "trained_policy" else "warn", str(rl.get("error") or "")),
        chip("LangGraph", "ok" if ok("langgraph") else "bad"),
    ]
    st.markdown("".join(chips), unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="qs-badge" style="background:{color}">{html.escape(text)}</span>'


def agent_tree(statuses: dict[str, str]) -> None:
    def s(name: str) -> str:
        v = statuses.get(name, "QUEUED")
        return badge(v, STATUS_COLORS.get(v, "#6B6F76"))
    lines = [
        f"Supervisor {s('supervisor')}",
        "│",
        f"├── SQL Rewriter {s('sql_rewriter')}",
        f"├── Schema Expert {s('schema_expert')}",
        f"├── Index Strategist {s('index_strategist')}",
        "│",
        f"├── RL Optimizer {s('rl_optimizer')}",
        f"├── Recommendation Validator {s('recommendation_validator')}",
        "│",
        f"├── HypoPG {s('hypopg_sandbox')}",
        "│",
        f"└── Blast Radius {s('blast_radius')}",
        f"&nbsp;&nbsp;&nbsp;&nbsp;└── Final Report {s('final_report')}",
    ]
    st.markdown('<div class="qs-tree">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)


def labeled_code(label: str, code: str | None, css: str = "", language: str = "sql") -> None:
    st.markdown(f'<div class="qs-label {css}">{html.escape(label)}</div>', unsafe_allow_html=True)
    st.code((fmt_sql(code) if language == "sql" else code) or "(none)", language=language)


def evidence_chain(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("No evidence chain available.")
        return
    for it in items:
        k = it.get("kind", "inferred")
        st.markdown(f"{badge(it.get('step', ''), '#5C626B')} {badge(k, KIND_COLORS.get(k, '#6B6F76'))} "
                    f"<span style='color:#5C626B;font-size:12px'>source: {html.escape(str(it.get('source_type')))}</span><br>"
                    f"{html.escape(str(it.get('statement')))}", unsafe_allow_html=True)


def advisory(text: str) -> None:
    st.markdown(f'<div class="qs-advisory">{html.escape(text)}</div>', unsafe_allow_html=True)


def explain(text: str) -> None:
    """Plain-language 'what am I looking at' note at the top of a tab."""
    st.markdown(f'<div class="qs-explain">{text}</div>', unsafe_allow_html=True)


def answer_card(sentence: str, safe: bool | None) -> None:
    shield = '<div class="qs-shield">Zero raw data exposed to AI</div>' if safe else ""
    st.markdown(f'<div class="qs-answer"><div class="t">Answer</div>{html.escape(sentence)}{shield}</div>',
                unsafe_allow_html=True)
