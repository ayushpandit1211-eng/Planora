"""Planora - Streamlit DBA dashboard (Sections 41-50).

    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

Wide layout, V4 forest glass theme (.streamlit/config.toml + dashboard/components.py CSS). All status
indicators come from real backend responses. The dashboard never executes SQL and never executes migrations.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import components as ui  # noqa: E402
from dashboard import plots  # noqa: E402
from dashboard.streaming_ui import BackendClient  # noqa: E402
from dashboard import upgrade_ui as upg  # noqa: E402  (Final Upgrade P13)

st.set_page_config(page_title="Planora - Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

EVENT_TO_AGENT = {
    "SUPERVISOR_STARTED": ("supervisor", "RUNNING"),
    "SQL_AGENT_STARTED": ("sql_rewriter", "RUNNING"), "SQL_AGENT_COMPLETE": ("sql_rewriter", "COMPLETED"),
    "SCHEMA_AGENT_STARTED": ("schema_expert", "RUNNING"), "SCHEMA_AGENT_COMPLETE": ("schema_expert", "COMPLETED"),
    "INDEX_AGENT_STARTED": ("index_strategist", "RUNNING"), "INDEX_AGENT_COMPLETE": ("index_strategist", "COMPLETED"),
    "RL_ANALYSIS_STARTED": ("rl_optimizer", "RUNNING"), "RL_ANALYSIS_COMPLETE": ("rl_optimizer", "COMPLETED"),
    "VALIDATION_COMPLETE": ("recommendation_validator", "COMPLETED"),
    "SANDBOX_STARTED": ("hypopg_sandbox", "RUNNING"), "SANDBOX_COMPLETE": ("hypopg_sandbox", "COMPLETED"),
    "BLAST_RADIUS_STARTED": ("blast_radius", "RUNNING"), "BLAST_RADIUS_COMPLETE": ("blast_radius", "COMPLETED"),
    "FINAL_REPORT_READY": ("final_report", "COMPLETED"), "MIGRATION_GENERATED": ("final_report", "RUNNING"),
}
SKIP_HINTS = {"SQL rewriter": "sql_rewriter", "Index strategist": "index_strategist", "HypoPG": "hypopg_sandbox"}
AGENTS = ["supervisor", "sql_rewriter", "schema_expert", "index_strategist", "rl_optimizer",
          "recommendation_validator", "hypopg_sandbox", "blast_radius", "final_report"]

ss = st.session_state
ss.setdefault("report", None)
ss.setdefault("events", [])
ss.setdefault("raw_sql", "")
ss.setdefault("nl", None)
ss.setdefault("agent_status", {a: "QUEUED" for a in AGENTS})

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    ui.brand()
    # visual order: source, scenarios, display, connection (containers keep the original call order below)
    _sb_source, _sb_scen, _sb_display, _sb_conn = st.container(), st.container(), st.container(), st.container()
    with _sb_conn:
        ui.group_label("Connection")
        if os.environ.get("QS_DEPLOYMENT_MODE") == "production":
            backend = os.environ["BACKEND_URL"]
            api_key = os.environ["QS_API_KEY"]
            st.caption("Connection managed by the server administrator.")
        else:
            with st.expander("Connection settings"):
                backend = st.text_input("Backend URL", os.environ.get("BACKEND_URL", "http://localhost:8000"))
                api_key = st.text_input("API key (if configured)", type="password", value=os.environ.get("QS_API_KEY", ""))
    client = BackendClient(backend, api_key or None)
    _api = upg.UpgradeClient(backend, api_key or None, upg.session_id())
    with _sb_source:
        active_source = upg.sidebar_sources(_api)
        st.caption(f"session `{upg.session_id()}` (ownership, not authentication)")
    with _sb_scen:
        ui.group_label("Demo scenarios")
        code, ex = client.get("/api/v1/examples")
        examples = ex.get("examples", []) if code == 200 else []
        choice = st.selectbox("Demonstration scenario", ["-"] + [e["name"] for e in examples])
        if st.button("Load scenario", width="stretch") and choice != "-":
            ss.raw_sql = next(e["sql"] for e in examples if e["name"] == choice)
    with _sb_display:
        ui.group_label("Display")
        demo_mode = st.toggle("Demo mode (5 core tabs)", value=True,
                              help="ON: Privacy, Plan/GNN, Optimization, HypoPG, Deployment. OFF: all tabs.")
        ui.group_label("Quick start")
        st.caption("1. Pick a scenario and press **Load scenario**.\n\n"
                   "2. Press **Run optimisation pipeline**.\n\n"
                   "3. Read the **Answer** box, then open the detail views.")

# ------------------------------------------------------------------ header + status bar
code, status = client.get("/api/v1/status")
ui.page_header(active_source)
# Presentation-only placeholders: preserve widget/API evaluation order below.
st.markdown('<div id="workspace"></div>', unsafe_allow_html=True)
with st.container(key="qs_grid"):
    _col_main, _col_side = st.columns([67, 33], gap="medium")
with _col_main:
    _work = st.container(border=True, key="qs_glass_work")
with _col_side:
    activity_slot = st.empty()
activity_slot.markdown(ui.activity_html((ss.report or {}).get("agent_status") or ss.agent_status, ss.events,
                                        (ss.report or {}).get("request_id"), started=bool(ss.report or ss.events)),
                       unsafe_allow_html=True)
st.markdown('<div id="your-data"></div>', unsafe_allow_html=True)
_data = st.container(border=True, key="qs_glass_data")
with st.container(key="qs_capabilities"):
    ui.capability_panel(status if code == 200 else None,
                        None if code == 200 else str(status.get("message") or status.get("error")))
st.markdown('<div id="report-views"></div>', unsafe_allow_html=True)
_strip_slot = st.container()
_answer_slot = st.container()


# ------------------------------------------------------------------ run helper
def run_pipeline(start_path: str, body: dict) -> None:
    code, started = client.post(start_path, body)
    if code != 200 or "request_id" not in started:
        st.error(f"Could not start optimisation: {started}")
        return
    rid = started["request_id"]
    ss.events, ss.report, ss.sql_notes = [], None, None
    ss.agent_status = {a: "QUEUED" for a in AGENTS}
    log_box = st.empty()
    tree_box = activity_slot

    def on_event(ev: dict) -> None:
        ss.events.append(ev)
        e = ev.get("event")
        if e in EVENT_TO_AGENT:
            a, s = EVENT_TO_AGENT[e]
            ss.agent_status[a] = s
        if e == "AGENT_SKIPPED":
            for hint, a in SKIP_HINTS.items():
                if hint in (ev.get("message") or ""):
                    ss.agent_status[a] = "SKIPPED"
        lines = [f"[{(x.get('timestamp') or '')[11:19]}] {x.get('message')}" for x in ss.events[-14:]]
        log_box.markdown('<div class="qs-log">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        tree_box.markdown(ui.activity_html(ss.agent_status, ss.events, rid), unsafe_allow_html=True)

    with st.status(f"Running pipeline {rid} ...", expanded=True) as box:
        client.stream(rid, on_event)
        for _ in range(60):
            code, rep = client.get(f"/api/v1/report/{rid}")
            if code == 200:
                ss.report = rep
                break
            time.sleep(0.5)
        final = (ss.report or {}).get("status", "unknown")
        box.update(label=f"Pipeline {rid}: {final}", state="complete" if final == "completed" else "error",
                   expanded=False)
    if ss.report and ss.report.get("agent_status"):
        ss.agent_status = ss.report["agent_status"]
    log_box.empty()      # full log stays available in the LangGraph Agents tab
    tree_box.empty()     # final authoritative statuses are shown in the LangGraph Agents tab


# ------------------------------------------------------------------ Final Upgrade P13: data source + Ask Your Database
_MAX_MB = int(os.environ.get("INGEST_MAX_UPLOAD_MB", "100"))
_up_code, _up_status = _api.get("/api/v1/upgrade/status")
with _data:
    if ss.get("qs_mode") == "Upload Dataset":
        ui.section("Add a dataset", "Select a file, preview it (nothing is written), then commit the load.")
        upg.dataset_upload(_api, _MAX_MB)
        st.divider()
        upg.manage_datasets(_api)
    else:
        ui.section("Your data", "Active source, its tables, and questions answered from it.")
        upg.source_panel(_api, active_source)
        upg.delete_dataset_control(_api, active_source)
        upg.ask_your_database(_api, active_source)

# ------------------------------------------------------------------ inputs
with _work:
    ui.section("Query workspace", "Ask in plain English; Planora ranks existing workload statements and "
               "cites its evidence.")
    q_col, b_col = st.columns([2, 1], vertical_alignment="bottom")
    question = q_col.text_input("Natural-language performance question",
                                placeholder="Why is the weekly sales reporting dashboard timing out?",
                                label_visibility="visible")
    if b_col.button("Find workload", width="stretch", type="secondary") and question.strip():
        code, ans = client.post("/api/v1/nl/ask", {"question": question})
        ss.nl = ans if code == 200 else {"answer": f"error: {ans}", "candidates": []}
    if ss.nl:
        st.info(ss.nl.get("answer"))
        st.caption(ss.nl.get("boundary", ""))
        intent = ss.nl.get("question_intent") or {}
        if intent:
            st.caption(f"intent: {intent.get('intents')} · matched tables: {intent.get('matched_tables')} · "
                       f"time granularity: {intent.get('time_granularity')}")
        for c in ss.nl.get("candidates", []):
            with st.container(border=True):
                st.code(c["anonymized_sql"], language="sql")
                st.caption(f"score {c['score']} · calls {c['calls']} · mean {c['mean_exec_time_ms']} ms · evidence: "
                           + "; ".join(c["evidence"]))
                if st.button("Analyze this candidate", key=c["candidate_id"]):
                    ss.raw_sql = "(selected workload statement - text kept server-side)"
                    run_pipeline("/api/v1/nl/analyze", {"candidate_id": c["candidate_id"]})

    st.divider()
    ui.section("Inspect a slow query", "Paste one read-only SELECT, or load a demo scenario from the sidebar.")
    ss.raw_sql = st.text_area("Slow SQL query (single read-only SELECT)", value=ss.raw_sql, height=150,
                              label_visibility="visible")
    _target = upg.source_for_sql(_api, ss.raw_sql, active_source)
    if _target.get("kind", "demo") != "demo":
        st.caption(f"This query will run on your data source **{_target.get('name')}** (`{_target['id']}`): 10 safety "
                   "checks → full optimisation pipeline → results in the detail views below.")
    if st.button("Run optimisation pipeline", type="primary"):
        if not ss.raw_sql.strip():
            st.warning("Enter a query first.")
        elif _target.get("kind", "demo") == "demo":
            run_pipeline("/api/v1/optimize/async", {"sql": ss.raw_sql})
        else:
            # Final Upgrade: pasted SQL follows the data source that owns its tables; the EXISTING full pipeline runs
            # there and its complete report fills the same tabs as the demo flow.
            ss.sql_notes = None
            with st.spinner(f"Optimising on {_target.get('name')} ..."):
                _c, _o = _api.post("/api/v1/nlq/optimize", {"sql": ss.raw_sql, "data_source_id": _target["id"]})
                if _c == 200 and _o.get("request_id"):
                    _c2, _rep = _api.get(f"/api/v1/report/{_o['request_id']}")
                    ss.report = _rep if _c2 == 200 else {"status": "error", "error": _rep}
                    ss.report_source = _target
                    ss.sql_notes = (_o.get("sql_notes") or [], _o.get("corrected_sql"))
                else:
                    ss.report = {"status": "rejected", "error": {"message": _o.get("message"), "stage":
                                 (_o.get("details") or {}).get("stage_name") or _o.get("stage"),
                                 "error": _o.get("error"),
                                 "suggestions": (_o.get("details") or {}).get("suggestions") or []}}
            st.rerun()

rep = ss.report
with _answer_slot:
    if rep and rep.get("status") != "completed":
        with st.container(border=True, key="qs_glass_answer_err"):
            err = rep.get("error") or {}
            st.error(f"{rep.get('status', 'error').upper()}: {err.get('message')} (stage: {err.get('stage')}, "
                     f"category: {err.get('error')})")
            for _s in err.get("suggestions") or []:
                st.info(f"Suggestion: {_s}")
            if "42P01" in str(err.get("message")):
                st.info("That table does not exist in the **demo database**. If it belongs to a dataset you uploaded, "
                        "select that dataset under **Data source** in the sidebar (or check the table name shown in the "
                        "source panel above) and run again.")

_tiles = None
if rep and rep.get("status") == "completed":
    with _answer_slot:
        with st.container(border=True, key="qs_glass_answer"):
            _notes, _fixed = ss.get("sql_notes") or ([], None)
            for _n in _notes:
                st.info(f"Auto-corrected: {_n}.")
            if _notes and _fixed:
                st.code(_fixed, language="sql")
            ui.section("Report answer", "Returned by the backend for the last run; details and proof are below.")
            ans = rep.get("answer") or {}
            proof = rep.get("zero_exposure_proof") or {}
            ui.answer_card(ans.get("sentence_dba") or ans.get("sentence") or "", proof.get("zero_raw_data_exposed"))
            st.caption(ans.get("sentence_dba_label", "") + " · " + ans.get("metric_note", ""))
    pct = ans.get("planner_cost_reduction_pct")
    wl = ans.get("write_latency") or []
    _tiles = [
        {"icon": "cost", "label": "Query cost reduction", "note": "planner estimate, not runtime",
         "value": f"{pct:.2f}%" if isinstance(pct, (int, float)) else "none needed",
         "help": "PostgreSQL planner estimate with the fix applied virtually (HypoPG). Not a stopwatch time."},
        {"icon": "write", "label": "Extra write time (per 1,000 inserts)", "note": "measured on a temporary copy",
         "value": f"+{sum(w['added_ms_per_1000_rows'] for w in wl):.2f} ms" if wl else "0 ms",
         "help": "Measured by timing INSERTs on a temporary copy, before and after adding the index."},
        {"icon": "storage", "label": "Extra storage", "note": "HypoPG estimate",
         "value": f"{ans.get('storage_mb') or 0:.0f} MB", "help": "Size of the new index(es), estimated by HypoPG."},
        {"icon": "shield", "label": "Raw data shown to AI", "note": "privacy proof for this run",
         "value": "0 (100% masked)" if proof.get("zero_raw_data_exposed") else "BLOCKED",
         "help": proof.get("statement") or ""},
    ]
with _strip_slot:
    ui.summary_strip(_tiles)
activity_slot.markdown(ui.activity_html((rep or {}).get("agent_status") or ss.agent_status, ss.events,
                                        (rep or {}).get("request_id"), started=bool(rep or ss.events)),
                       unsafe_allow_html=True)

ALL_TABS = ["Privacy Audit", "Execution Plan / GNN", "LangGraph Agents", "Optimization", "HypoPG",
            "Deployment Review", "Evidence Chain", "Workload Intelligence", "Benchmark Lab"]
CORE_TABS = ["Privacy Audit", "Execution Plan / GNN", "Optimization", "HypoPG", "Deployment Review"]
_shown = CORE_TABS if demo_mode else ALL_TABS
TAB_LABELS = {"Privacy Audit": "Privacy Audit", "Execution Plan / GNN": "Execution Plan / GNN",
              "LangGraph Agents": "AI Agents", "Optimization": "Optimization", "HypoPG": "Simulation (HypoPG)",
              "Deployment Review": "Deployment Review", "Evidence Chain": "Evidence Chain",
              "Workload Intelligence": "Workload History", "Benchmark Lab": "Benchmark Lab"}
_details = st.container(border=True, key="qs_glass_details")
with _details:
    ui.section("Details", ("Core views. Turn off Demo mode in the sidebar for all nine views." if demo_mode
                           else "All nine views."))
    _tab_objs = dict(zip(_shown, st.tabs([TAB_LABELS[n] for n in _shown])))
_hidden_slot = st.empty()


def tab(name: str):
    """Container for a tab; hidden tabs render into a placeholder that is cleared at the end."""
    return _tab_objs.get(name) or _hidden_slot.container()


ok = bool(rep and rep.get("status") == "completed")

# ------------------------------------------------------------------ Privacy
with tab("Privacy Audit"):
    priv = (rep or {}).get("privacy_audit") or {}
    c1, c2 = st.columns(2)
    with c1:
        ui.labeled_code("Raw SQL: INTERNAL / SENSITIVE (this browser session only)", ss.raw_sql, "qs-sensitive")
    with c2:
        ui.labeled_code("Anonymized SQL: SAFE FOR MODEL INGESTION", priv.get("masked_sql"), "qs-safe")
    if priv:
        pa = priv.get("privacy_audit", {})
        m = st.columns(5)
        m[0].metric("PII detected", "yes" if pa.get("pii_detected") else "no", ", ".join(pa.get("pii_types", [])) or None)
        m[1].metric("Literals parameterized", priv.get("literal_count"))
        m[2].metric("Pseudonymized identifiers", len(priv.get("mapped_tables", {})) + len(priv.get("mapped_columns", {})))
        m[3].metric("Fallback used", "yes" if priv.get("fallback_used") else "no")
        m[4].metric("Outgoing LLM check", "SAFE" if pa.get("safe_for_llm") else "BLOCKED")
        st.caption(f"AST fingerprint: {priv.get('ast_signature')} · literal leak: {pa.get('literal_leak_detected')} · "
                   f"secret leak: {pa.get('secret_leak_detected')} · privacy bitmask: {priv.get('privacy_bitmask')}")
        flags = {k: v for k, v in (priv.get("privacy_flags") or {}).items() if isinstance(v, bool)}
        st.markdown(" ".join(ui.badge(k, "#1D654D" if v else "#6B6F76") for k, v in flags.items()), unsafe_allow_html=True)
        st.caption("The reverse mapping (pseudonym → real identifier) stays in the backend vault and is never displayed.")
    upg.privacy_upgrade_section(_up_status if _up_code == 200 else None)

# ------------------------------------------------------------------ Plan / GNN
with tab("Execution Plan / GNN"):
    ui.explain("""PostgreSQL plans a query as a <b>tree of steps</b> (scan a table, join, sort…). The GNN reads this tree and marks the <b>slowest step in red</b>. <i>Cost</i> is PostgreSQL's own estimate of work - lower is faster.""")
    if ok:
        plan, gnn = rep["plan_analysis"], rep["gnn_analysis"]
        hc = plan.get("highest_cost_node") or {}
        m = st.columns(4)
        m[0].metric("Planner total cost", ui.compact(plan["total_cost"]), help=f"{plan['total_cost']:,.2f}")
        m[1].metric("Planned rows", f"{plan['plan_rows']:,.0f}")
        m[2].metric("Operators", plan["node_count"])
        m[3].metric("GNN mode", gnn["model_mode"], str(gnn.get("confidence")))
        st.markdown(f"### Bottleneck: :red[{gnn['primary_bottleneck_type']}]")
        st.plotly_chart(plots.plan_dag(gnn["graph_nodes"], gnn["graph_edges"], hc.get("node_id")), width="stretch")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(plots.operator_costs(gnn["graph_nodes"], hc.get("node_id")), width="stretch")
        fig = plots.class_probabilities(gnn.get("class_probabilities"))
        if fig:
            c2.plotly_chart(fig, width="stretch")
        else:
            c2.info(f"Analytic fallback: rule-confidence label `{gnn.get('confidence')}` (not a probability). "
                    f"Reason: {gnn.get('fallback_reason')}")
        st.markdown(f"**Explanation:** {gnn['natural_language_summary']}")
        upg.attention_section(gnn, plots)
        st.markdown("**Evidence fields**")
        for e in gnn.get("evidence", []):
            st.markdown(f"- {e}")
        st.markdown("**Execution plan tree (sanitised)**")
        st.code(plan.get("tree_text", ""), language="text")
        if rep.get("workload_context", {}).get("regression"):
            st.warning(f"Workload history: {rep['workload_context']['regression']['event']} - "
                       + "; ".join(rep["workload_context"]["regression"]["evidence"]))
    else:
        st.info("Run the pipeline to see the execution plan.")

# ------------------------------------------------------------------ Agents
with tab("LangGraph Agents"):
    ui.explain("""Several small AI helpers work as a team. Each line shows what a helper did: COMPLETED, SKIPPED (not needed) or ERROR.""")
    ui.agent_tree(ss.agent_status)
    if ok:
        st.markdown("**Agent log**")
        st.dataframe(rep.get("agent_logs") or [], hide_index=True, width="stretch")
        llm = ((rep.get("final_report") or {}).get("model_modes") or {}).get("llm", {})
        st.caption(f"Model paths - GNN: {rep['gnn_analysis']['model_mode']} · RL: {rep['rl_recommendation']['policy_mode']} · "
                   f"LLM: {llm.get('provider')} ({'enabled' if llm.get('enabled') else 'disabled'})")
    if ss.events:
        with st.expander(f"Live event stream ({len(ss.events)} events)"):
            st.markdown('<div class="qs-log">' + "<br>".join(
                f"[{(e.get('timestamp') or '')[11:19]}] <b>{e.get('event')}</b> {e.get('message')}"
                for e in ss.events) + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Optimization
with tab("Optimization"):
    ui.explain("""The suggested fixes. An <b>index</b> is like a book's index: the database jumps straight to the rows it needs instead of reading the whole table. A <b>rewrite</b> is the same question written in a faster way.""")
    if ok:
        rec, rl = rep["agent_recommendation"], rep["rl_recommendation"]
        orig_masked = rep["privacy_audit"]["masked_sql"]
        c1, c2 = st.columns(2)
        with c1:
            ui.labeled_code("Original SQL (anonymised)", orig_masked)
        with c2:
            ui.labeled_code("AI rewritten SQL (anonymised)", rec.get("rewritten_sql"))
        if rec.get("rewritten_sql"):
            diff = "\n".join(difflib.unified_diff(orig_masked.replace(" AND ", "\n  AND ").splitlines(),
                                                  rec["rewritten_sql"].replace(" AND ", "\n  AND ").splitlines(),
                                                  "original", "rewritten", lineterm=""))
            st.markdown("**SQL diff**")
            st.code(diff, language="diff")
            rw = rec.get("rewrite") or {}
            st.caption(f"Rules: {rw.get('rules_applied')} · semantics verified by AST diff: {rw.get('semantics_verified')}"
                       f" · problems: {(rw.get('semantic_diff') or {}).get('verification_problems')}")
        st.markdown("**Proposed indexes (anonymised)**")
        st.dataframe([{"ddl": i["ddl"], "kind": i["kind"], "reason": i["reason"], "source": i["source"],
                       "evidence": "; ".join(i.get("evidence", []))} for i in rec.get("proposed_indexes", [])],
                     hide_index=True, width="stretch")
        p, sh = rec.get("partitioning_recommendation"), rec.get("sharding_recommendation")
        if p:
            st.markdown(f"**Partitioning suggestion:** {p['strategy_type']} on `{p['candidate_key']}` ({p['table']})")
            ui.advisory(p["label"])
            st.caption(p["rationale"] + " · Expected benefit: " + p["expected_benefit"])
            st.caption("Operational risks: " + "; ".join(p["operational_risks"]))
        if sh:
            st.markdown(f"**Sharding advisory:** distribute by `{sh['candidate_key']}`")
            ui.advisory(sh["label"])
            st.caption(sh["rationale"])
        st.markdown(f"**RL action ({rl['policy_mode']}):** `{rl['high_level_action']}`"
                    + (f" / `{rl['low_level_action']}`" if rl.get("low_level_action") else ""))
        st.caption(rl.get("explanation", ""))
        fig = plots.q_values(rl.get("q_values") or {})
        if fig:
            st.plotly_chart(fig, width="stretch")
        if rl.get("masked_actions"):
            with st.expander("Invalid-action masking"):
                st.json(rl["masked_actions"])
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Reasoning**")
        for r in rec.get("technical_reasoning", []):
            c1.markdown(f"- {r}")
        c2.markdown("**Assumptions**")
        for a in rec.get("assumptions", []):
            c2.markdown(f"- {a}")
        c3.markdown("**Risks**")
        for r in rec.get("risks", []) or ["none identified"]:
            c3.markdown(f"- {r}")
        st.markdown("**Uncertainty / limitations**")
        for lim in rep.get("limitations", []):
            st.markdown(f"- {lim}")
    else:
        st.info("Run the pipeline to see recommendations.")

# ------------------------------------------------------------------ HypoPG
with tab("HypoPG"):
    ui.explain("""Here the fixes are <b>tested without touching real tables</b>. Indexes are added as 'pretend' indexes (HypoPG). Write time and partitioning are measured on <b>temporary copies</b> that are thrown away right after.""")
    sb = (rep or {}).get("sandbox_simulation") if ok else None
    if sb and sb.get("status") == "completed":
        m = st.columns(4)
        m[0].metric("Baseline Planner Cost", ui.compact(sb["before_total_cost"]), help=f"{sb['before_total_cost']:,.2f}")
        m[1].metric("Optimized Planner Cost", ui.compact(sb["after_total_cost"]), help=f"{sb['after_total_cost']:,.2f}")
        m[2].metric("Planner-Estimated Cost Reduction", f"{sb['planner_cost_reduction_pct']}%")
        m[3].metric("Estimated Storage Impact", f"{sb.get('estimated_storage_impact_mb')} MB")
        st.caption(sb.get("metric_label"))
        st.plotly_chart(plots.cost_before_after(sb["before_total_cost"], sb["after_total_cost"],
                                                sb.get("rewrite_after_cost")), width="stretch")
        if sb.get("combination_note"):
            st.warning(sb["combination_note"])
        fig = plots.per_candidate(sb.get("per_candidate", []))
        if fig:
            st.plotly_chart(fig, width="stretch")
        st.markdown("**Virtual indexes used by the planner**")
        for v in sb.get("virtual_indexes_used", []):
            st.code(v, language="sql")
        st.markdown("#### Write latency (measured on a temporary copy)")
        for w in sb.get("write_latency_simulation") or []:
            if w.get("status") == "completed":
                c = st.columns(4)
                c[0].metric("Table", w["target_table"])
                c[1].metric("Before (per 1,000 inserts)", f"{w['ms_per_1000_rows_before']} ms")
                c[2].metric("After (per 1,000 inserts)", f"{w['ms_per_1000_rows_after']} ms")
                c[3].metric("Change", f"+{w['added_ms_per_1000_rows']} ms", f"{w['write_latency_change_pct']:+}%",
                            delta_color="inverse")
                st.caption(f"How: {w['method']}. {w['caveat']}")
            else:
                st.caption(f"Write-latency simulation {w.get('status')}: {w.get('reason')}")
        st.markdown("**Write risk / blast radius** (which write queries would also have to update the new index)")
        br = rep.get("blast_radius") or {}
        st.metric("Risk level", (br.get("overall_risk") or "n/a").upper())
        for b in br.get("results", []):
            with st.expander(f"{b['candidate']} - risk {b['risk_level']}"):
                st.json({"write_frequency_summary": b["write_frequency_summary"],
                         "estimated_write_impact": b["estimated_write_impact"],
                         "write_queries_analyzed": b["write_queries_analyzed"]})
                st.caption(b["methodology"])
    elif sb:
        st.warning(f"Sandbox {sb.get('status')}: {sb.get('error')}")
    else:
        st.info("No index simulation (no valid candidate or pipeline not run).")
    ps = ((rep or {}).get("structural_simulation") or {}).get("partitioning") if ok else None
    if ps:
        st.markdown("#### Partitioning simulation (temporary sample copy)")
        if ps.get("status") == "completed":
            st.caption("Partitioning splits a big table into smaller pieces (for example one per month). "
                       "A query for one month then reads only that piece.")
            c = st.columns(4)
            c[0].metric("Split by", "month" if ps["strategy"] == "range" else "hash", help=f"partition key: {ps['key']}")
            c[1].metric("Pieces read", f"{ps['partitions_scanned']} of {ps['partitions_total']}")
            c[2].metric("Cost: one big table", ui.compact(ps["cost_unpartitioned"]))
            c[3].metric("Cost: partitioned", ui.compact(ps["cost_partitioned"]),
                        f"-{ps['planner_cost_reduction_pct']}%", delta_color="inverse")
            st.caption(f"Query tested: {ps.get('query_shape')}. How: {ps['method']}. {ps['caveat']}")
        else:
            st.caption(f"Partitioning simulation {ps.get('status')}: {ps.get('reason')}")

# ------------------------------------------------------------------ Deployment
with tab("Deployment Review"):
    ui.explain("""Ready-to-review scripts for a DBA: one to <b>apply</b> the fix and one to <b>undo</b> it. Planora never runs them. A person must approve first.""")
    if ok:
        dep = rep["deployment"]
        st.error("HUMAN REVIEW REQUIRED - Planora never executes migrations. Apply to STAGING only after review.")
        st.markdown(f"**Migration ID:** `{dep.get('migration_id') or '-'}` · **Query fingerprint:** "
                    f"`{dep.get('query_fingerprint') or '-'}` · **Generated:** `{(dep.get('generated_at') or '')[:19]}` · "
                    f"**Approval status:** `{dep.get('approval_status')}`")
        _mc, _rc = st.columns(2)
        with _mc:
            ui.labeled_code("Recommended Staging Migration", dep.get("migration_sql"))
        with _rc:
            ui.labeled_code("Emergency Rollback", dep.get("rollback_sql"))
        st.markdown(f"**Rationale:** {dep.get('rationale')}")
        for n in dep.get("advisory_notes", []):
            ui.advisory(n)
        if dep.get("rewritten_sql_for_review"):
            ui.labeled_code(dep.get("rewritten_sql_label") or "Rewritten SQL for review", dep["rewritten_sql_for_review"],
                            "qs-sensitive")
        c1, c2 = st.columns(2)
        c1.download_button("Download migration.sql", dep.get("migration_sql") or "", f"{dep.get('migration_id')}_migration.sql")
        c2.download_button("Download rollback.sql", dep.get("rollback_sql") or "", f"{dep.get('migration_id')}_rollback.sql")
        st.download_button("Download full report (JSON)", json.dumps(rep, indent=1, default=str), f"{rep['request_id']}.json")
    else:
        st.info("Run the pipeline to generate a staging migration.")

# ------------------------------------------------------------------ Evidence
with tab("Evidence Chain"):
    ui.explain("""Every claim with its proof. Each line says whether it was <b>observed</b> (read from the database), <b>simulated</b> (tested in the sandbox) or <b>estimated</b>.""")
    if ok:
        ex = rep["final_report"].get("explanation", {})
        if ex:
            st.markdown(f"**Observation:** {ex.get('observation')}")
            st.json(ex.get("evidence"))
            st.markdown(f"**Recommendation:** `{ex.get('recommendation')}`")
            st.markdown(f"**Validation:** {ex.get('validation')}")
            st.markdown(f"**Operational impact:** {'; '.join(ex.get('operational_impact', []))}")
            st.markdown(f"**Uncertainty:** {ex.get('uncertainty')}")
        st.divider()
        ui.evidence_chain(rep["final_report"].get("evidence_chain", []))
        st.caption("Budget: " + (rep["final_report"].get("budget") or {}).get("statement", ""))
    else:
        st.info("Run the pipeline to build the evidence chain.")

# ------------------------------------------------------------------ Workload
with tab("Workload Intelligence"):
    ui.explain("""Upload old slow-query logs or read live statistics to spot queries that got slower over time.""")
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Ingest slow-query log / historical plans (JSON or CSV)", type=["json", "csv"])
    if up is not None and c1.button("Ingest file"):
        fmt = "csv" if up.name.endswith(".csv") else "json"
        code, res = client.post("/api/v1/ingest/plans", {"format": fmt, "content": up.getvalue().decode("utf-8")})
        c1.json({k: res.get(k) for k in ("accepted", "quarantined", "records_with_pii_removed", "plans_parsed",
                                         "persisted_to_optimizer_meta", "error")})
    if c2.button("Collect candidates from pg_stat_statements"):
        code, res = client.post("/api/v1/ingest/pg_stat_statements", {"limit": 50})
        c2.json({k: res.get(k) for k in ("accepted", "quarantined", "error")})
    code, wl = client.get("/api/v1/workload")
    if code == 200:
        st.json(wl["summary"])
        if wl["regressions"]:
            st.markdown("**Plan regression detection**")
            st.dataframe([{"event": r["event"], "fingerprint": r["query_fingerprint"], "at": r["changed_at_bucket"],
                           "cost_change_%": r["planner_cost_change_pct"], "time_change_%": r["observed_time_change_pct"],
                           "lost_index": r["lost_index_access"]} for r in wl["regressions"]], hide_index=True, width="stretch")
        d = wl["drift"]
        st.markdown(f"**Workload drift:** {d['status']} · drift detected: {d.get('drift_detected')}")
        fig = plots.drift_scores(d.get("scores") or {}, d.get("threshold", 0.2))
        if fig:
            st.plotly_chart(fig, width="stretch")
    code, hy = client.get("/api/v1/hygiene")
    if code == 200:
        with st.expander("Index hygiene (report-only)"):
            st.json(hy)

# ------------------------------------------------------------------ Benchmark
with tab("Benchmark Lab"):
    ui.explain("""Compares the full system with versions that switch off one part at a time, to show each part actually helps.""")
    st.caption("Planner-estimate mode always runs. Measured runtime requires BENCHMARK_ENABLED=true and a disposable "
               "database (BENCHMARK_ADMIN_DSN). The two metrics are never mixed.")
    bq = st.selectbox("Query", [e["name"] for e in examples] or ["-"], key="bench_q")
    measure = st.checkbox("Also measure real execution time (opt-in, disposable DB only)")
    if st.button("Run benchmark + ablations") and examples:
        idx = next(e["index"] for e in examples if e["name"] == bq)
        with st.spinner("Running HypoPG experiments ..."):
            code, br = client.post("/api/v1/benchmark", {"demo_query_index": idx, "measure": measure})
        if code == 200:
            ss.bench = br
        else:
            st.error(br)
    if ss.get("bench"):
        br = ss.bench
        st.plotly_chart(plots.benchmark_variants(br["results"]), width="stretch")
        st.dataframe(br["results"], hide_index=True, width="stretch")
        st.markdown("**Measurement**")
        st.json(br["measurement"])
        st.markdown("**Environment**")
        st.json(br["environment"])
        st.caption(br["caveat"])

_hidden_slot.empty()   # discard output of tabs hidden by demo mode
