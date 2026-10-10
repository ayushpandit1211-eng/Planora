"""Final Upgrade dashboard pieces (P13): data-source selector, dataset upload, Ask Your Database.

V3 light glass presentation: dense tables, restrained colour, provenance badges beside every metric. Every call goes through the backend API (never directly to PostgreSQL).
"""
from __future__ import annotations

import html
import time
import uuid
from typing import Any

import httpx
import streamlit as st

PROV_COLORS = {
    "POSTGRES OBSERVED": "#2563EB", "PLANNER ESTIMATE": "#6B7280", "HYPOPG SIMULATION": "#7C3AED",
    "MEASURED SANDBOX": "#059669", "ANALYTIC FALLBACK": "#B45309", "TRAINED GNN": "#0E7490",
    "RL POLICY": "#0F766E", "LLM RECOMMENDATION": "#BE185D", "HEURISTIC ESTIMATE": "#92400E",
    "SYNTHETIC DATA": "#9333EA", "DP-PROTECTED": "#047857",
}
KIND_LABEL = {"demo": "Demo Database", "dynamic": "Uploaded dataset"}


def prov(tags: list[str] | None) -> str:
    """Provenance badges (HTML)."""
    return " ".join(f'<span class="qs-badge" style="background:{PROV_COLORS.get(t, "#374151")}">'
                    f'{html.escape(t)}</span>' for t in (tags or []))


def metric_with_prov(col, label: str, value: Any, tags: list[str] | None, help_: str | None = None) -> None:
    col.metric(label, value, help=help_)
    col.markdown(prov(tags), unsafe_allow_html=True)


class UpgradeClient:
    def __init__(self, base: str, api_key: str | None, session_id: str, timeout: float = 180.0):
        self.base = base.rstrip("/")
        self.h = {"X-Session-Id": session_id}
        if api_key:
            self.h["x-api-key"] = api_key
        self.timeout = timeout

    def _do(self, method: str, path: str, headers: dict | None = None, **kw) -> tuple[int, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.request(method, self.base + path, headers={**self.h, **(headers or {})}, **kw)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {"error": "non-JSON response"}
        except httpx.HTTPError as exc:
            return 0, {"error": "backend_unreachable", "message": type(exc).__name__}

    def get(self, path: str):
        return self._do("GET", path)

    def post(self, path: str, body: dict | None = None):
        return self._do("POST", path, json=body or {})

    def upload(self, path: str, data: bytes):
        return self._do("POST", path, content=data, headers={"content-type": "application/octet-stream"})

    def delete(self, path: str):
        return self._do("DELETE", path)


def session_id() -> str:
    """Session UUID kept in session state and the URL (?sid=...). Ownership, not authentication."""
    ss = st.session_state
    if "qs_sid" not in ss:
        sid = st.query_params.get("sid")
        ss.qs_sid = sid if sid and 8 <= len(sid) <= 64 else f"sess-{uuid.uuid4().hex[:20]}"
        st.query_params["sid"] = ss.qs_sid
    return ss.qs_sid


def show_error(out: dict) -> None:
    code = out.get("error", "error")
    det = out.get("details") or {}
    st.error(f"`{code}` - {out.get('message', '')}" + (f" (stage {det.get('stage')}: {det.get('stage_name')})"
                                                       if det.get("stage") else ""))
    for s in det.get("suggestions", []) or []:
        st.caption(f"suggestion: {s}")
    if det.get("conflicts"):
        st.dataframe(det["conflicts"], width="stretch", hide_index=True)


# ------------------------------------------------------------------ sidebar: data source selector
def sidebar_sources(api: UpgradeClient) -> dict:
    ss = st.session_state
    ss.setdefault("qs_source", "demo")
    st.markdown('<div class="qs-group">Data source</div>', unsafe_allow_html=True)
    code, out = api.get("/api/v1/sources")
    sources = out.get("sources", []) if code == 200 else [{"id": "demo", "kind": "demo", "name": "Demo Database"}]
    ids = [s["id"] for s in sources]
    if ss.qs_source not in ids:
        ss.qs_source = "demo"
    labels = {s["id"]: f"{KIND_LABEL.get(s['kind'], s['kind'])} · {s.get('name', s['id'])}" for s in sources}
    ss.qs_source = st.selectbox("Active source", ids, index=ids.index(ss.qs_source),
                                format_func=lambda i: labels.get(i, i), label_visibility="collapsed")
    ss.setdefault("qs_mode", "Demo Database")
    modes = ["Demo Database", "Upload Dataset (CSV, JSON)"]
    if ss.qs_mode not in modes:
        ss.qs_mode = "Demo Database"
    ss.qs_mode = st.radio("Add data", modes, index=modes.index(ss.qs_mode), label_visibility="collapsed")
    return next((s for s in sources if s["id"] == ss.qs_source), sources[0])


def source_panel(api: UpgradeClient, src: dict) -> None:
    """Active source id, kind, status, tables/columns/rows, schema version."""
    sid = src["id"]
    code, sch = api.get(f"/api/v1/sources/{sid}/schema")
    if code != 200:
        show_error(sch)
        return
    ds = sch["data_source"]
    st.markdown(f'<div class="qs-srcname">{html.escape(str(ds["name"]))}</div>'
                f'<div class="qs-srcmeta">kind <b>{html.escape(str(ds["kind"]))}</b> · status '
                f'<b>{html.escape(str(ds["status"]))}</b> · schema v<b>{html.escape(str(ds["schema_version"]))}</b> · '
                f'id <code>{html.escape(str(ds["id"]))}</code> {prov(sch.get("provenance"))}</div>',
                unsafe_allow_html=True)
    rows = [{"table": t["name"], "rows (estimated)": t["estimated_rows"]["value"], "columns": len(t["columns"]),
             "column list": ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])[:160]}
            for t in sch["tables"]]
    st.dataframe(rows, width="stretch", hide_index=True)
    if sch["tables"] and len(sch["tables"]) <= 5:
        t0 = sch["tables"][0]
        cols = ", ".join(c["name"] for c in t0["columns"][:4]) or "*"
        st.caption("Use these exact table names in your SQL. Example:")
        st.code(f"SELECT {cols}\nFROM {t0['name']}\nLIMIT 100", language="sql")


def _delete_sources(api: UpgradeClient, ids: list[str], labels: dict[str, str]) -> None:
    ss = st.session_state
    done, failed = [], []
    for sid in ids:
        code, out = api.delete(f"/api/v1/data/datasets/{sid}")
        (done if code == 200 else failed).append((sid, out))
    for sid, out in done:
        st.toast(f"Deleted {labels.get(sid, sid)} ({', '.join(out.get('tables_dropped') or []) or 'no tables'})")
    for _sid, out in failed:
        show_error(out)
    if any(sid == ss.get("qs_source") for sid, _ in done):
        ss.qs_source = "demo"
    if done and not failed:
        time.sleep(0.4)
        st.rerun()


def delete_dataset_control(api: UpgradeClient, src: dict) -> None:
    """Delete the active uploaded dataset (tables + registry). Never shown for the demo database."""
    if src.get("kind") != "dynamic":
        return
    with st.expander(f"Delete this dataset ({src.get('name', src['id'])})"):
        st.caption("Removes its table(s) and all stored information about it. This cannot be undone. "
                   "The demo database is never affected.")
        sure = st.checkbox("Yes, delete this dataset permanently", key=f"del_ok_{src['id']}")
        if st.button("Delete dataset", key=f"del_{src['id']}", disabled=not sure, type="primary"):
            _delete_sources(api, [src["id"]], {src["id"]: src.get("name", src["id"])})


def manage_datasets(api: UpgradeClient) -> None:
    """List every uploaded dataset and delete the selected ones."""
    code, out = api.get("/api/v1/data/datasets")
    if code != 200:
        show_error(out)
        return
    by_src: dict[str, dict] = {}
    for r in out.get("datasets", []):
        e = by_src.setdefault(r["data_source_id"], {"dataset": r["name"], "tables": [], "rows": 0})
        if r.get("table_name"):
            e["tables"].append(r["table_name"])
            e["rows"] += int(r.get("row_count") or 0)
    st.markdown('<div class="qs-section">Your uploaded datasets</div>', unsafe_allow_html=True)
    if not by_src:
        st.caption("No uploaded datasets yet.")
        return
    st.dataframe([{"dataset": v["dataset"], "table(s) to use in SQL": ", ".join(v["tables"]), "rows": v["rows"],
                   "id": k} for k, v in by_src.items()], width="stretch", hide_index=True)
    labels = {k: f"{v['dataset']} ({', '.join(v['tables']) or 'no tables'})" for k, v in by_src.items()}
    with st.container(border=True, key="qs_danger"):
        st.markdown('<div class="qs-danger-t">Delete datasets</div><div class="qs-tn">Drops the selected tables and '
                    'their stored information. This cannot be undone. The demo database is never affected.</div>',
                    unsafe_allow_html=True)
        pick = st.multiselect("Select datasets to delete", list(by_src), format_func=lambda i: labels.get(i, i),
                              key="qs_del_pick")
        sure = st.checkbox("Yes, delete the selected datasets permanently", key="qs_del_sure")
        if st.button(f"Delete selected ({len(pick)})", disabled=not (pick and sure), key="qs_del_btn"):
            _delete_sources(api, pick, labels)


def source_for_sql(api: UpgradeClient, sql: str, active: dict) -> dict:
    """Which data source should run this SQL? The active non-demo source, or - when the demo source is active - an
    uploaded dataset of this session whose table names appear in the SQL text."""
    if active.get("kind", "demo") != "demo" or not (sql or "").strip():
        return active
    import re
    words = {w.lower() for w in re.findall(r'"([^"]+)"|([A-Za-z_][A-Za-z0-9_$]*)', sql) for w in w if w}
    code, out = api.get("/api/v1/sources")
    for src in (out.get("sources", []) if code == 200 else []):
        if src.get("kind") == "demo":
            continue
        if (src.get("name") or "").lower() in words:
            return src
        c2, sch = api.get(f"/api/v1/sources/{src['id']}/schema")
        if c2 == 200 and any(t["name"].lower() in words for t in sch.get("tables", [])):
            return src
    return active


# ------------------------------------------------------------------ dataset upload (preview -> commit)
def dataset_upload(api: UpgradeClient, max_mb: int) -> None:
    ss = st.session_state
    st.caption(f"Accepted: CSV (comma, semicolon, tab, pipe; UTF-8/BOM/Latin-1) or JSON (array, JSON Lines, object "
               f"with one array) · up to {max_mb} MB. Types are inferred; tables only grow (new nullable columns) - "
               "nothing is ever dropped.")
    st.markdown('<div class="qs-steps-line"><b>1</b> Select file <span>→</span> <b>2</b> Preview <span>→</span> '
                '<b>3</b> Commit load</div>', unsafe_allow_html=True)
    f = st.file_uploader("Dataset file", type=["csv", "tsv", "txt", "json", "jsonl", "ndjson"],
                         label_visibility="visible")
    name = st.text_input("Dataset name - this becomes the TABLE NAME you use in SQL (e.g. survey -> FROM survey). "
                         "Leave empty to use the file name.", value="")
    if f is not None and st.button("Preview (no writes)"):
        if f.size > max_mb * 1024 * 1024:
            st.error(f"`upload_rejected` - file larger than {max_mb} MB")
            return
        q = f"/api/v1/data/preview?filename={f.name}" + (f"&dataset_name={name}" if name else "")
        code, pv = api.upload(q, f.getvalue())
        ss.qs_preview = pv if code == 200 else None
        if code != 200:
            show_error(pv)
    pv = ss.get("qs_preview")
    if pv:
        st.markdown(f"**Preview** · format `{pv['detected_format']}` · encoding `{pv['encoding']}` · delimiter "
                    f"`{pv.get('delimiter')}` · rows read **{pv['rows_read']:,}** · target `{pv['target_table']}` · "
                    f"plan **{pv['plan'].get('action')}**")
        st.dataframe(pv["columns"], width="stretch", hide_index=True)
        for w in pv["warnings"] + pv["plan"].get("suggestions", []):
            st.warning(w)
        if pv["plan"].get("conflicts"):
            st.error("Incompatible with the existing table - the load will be rejected:")
            st.dataframe(pv["plan"]["conflicts"], width="stretch", hide_index=True)
        c = st.columns(3)
        dup = c[0].checkbox("Load even if this exact file was loaded before")
        wid = c[1].checkbox("Allow safe type widening")
        if c[2].button("Commit load", type="primary"):
            code, res = api.post("/api/v1/data/commit", {"preview_id": pv["preview_id"], "allow_duplicate": dup,
                                                         "allow_widening": wid})
            if code == 200:
                st.success(f"Loaded {res['rows_loaded']:,} rows into `{res['table']}` (schema v{res['schema_version']},"
                           f" rejected {res['rows_rejected']}).")
                ss.qs_source, ss.qs_preview, ss.qs_mode = res["data_source_id"], None, "Demo Database"
                time.sleep(0.5)
                st.rerun()
            else:
                show_error(res)


# ------------------------------------------------------------------ Ask Your Database (10 sections)
def ask_your_database(api: UpgradeClient, src: dict) -> None:
    ss = st.session_state
    st.markdown('<div class="qs-section" style="margin-top:8px">Ask your database</div><div class="qs-section-sub">'
                f'Answers come from <b>{html.escape(str(src.get("name", src["id"])))}</b> through 10 safety checks; '
                'the optimizer can analyse the generated SQL.</div>', unsafe_allow_html=True)
    with st.expander(f"Run your own SQL on this data source ({src.get('name', src['id'])})"):
        st.caption("Write a single SELECT (CTEs, joins, window functions allowed). It passes the same 10 safety "
                   "checks, then runs read-only. Table names are listed in the table above.")
        sql = st.text_area("SQL", key="qs_user_sql", height=140, label_visibility="collapsed",
                           placeholder="SELECT state, avg(profit) FROM <table> GROUP BY state")
        cc = st.columns([1, 1, 4])
        o2 = cc[0].toggle("Optimize", value=True, key="qs_sql_opt")
        if cc[1].button("Run SQL", type="primary") and sql.strip():
            with st.spinner("10-stage validation → optimizer → execution ..."):
                code, out = api.post("/api/v1/nlq/run", {"sql": sql, "data_source_id": src["id"], "optimize": o2})
            ss.qs_nlq = out if code == 200 else {"_error": out}
    c = st.columns([2.4, 1.6, 0.9], vertical_alignment="bottom")
    q = c[0].text_input("Ask your database", placeholder="e.g. total revenue by region in 2025",
                        label_visibility="visible", key="qs_q")
    opt = c[1].toggle("Optimize", value=True, help="Also run the Planora optimizer on the generated SQL.")
    if c[2].button("Ask", type="primary", width="stretch") and q.strip():
        with st.spinner("schema → SQL → 10-stage validation → optimizer → execution ..."):
            code, out = api.post("/api/v1/nlq/ask", {"question": q, "data_source_id": src["id"], "optimize": opt})
        ss.qs_nlq = out if code == 200 else {"_error": out}
    out = ss.get("qs_nlq")
    if not out:
        return
    if out.get("_error"):
        show_error(out["_error"])
        det = out["_error"].get("details") or {}
        if det.get("schema"):
            st.caption("available schema: " + "; ".join(f"{t}({', '.join(cs)})" for t, cs in det["schema"].items()))
        return
    if out.get("status") == "clarification":
        st.info(out["clarification"])
        return
    for w in out.get("warnings", []):
        st.warning(w)
    if out.get("anchored_rerun") and st.button(f"Re-run anchored to {out['anchored_rerun']['anchor_date']}"):
        code, o2 = api.post("/api/v1/nlq/ask", {"question": ss.qs_q, "data_source_id": src["id"],
                                                "anchor_date": out["anchored_rerun"]["anchor_date"]})
        ss.qs_nlq = o2 if code == 200 else {"_error": o2}
        st.rerun()
    ans = out.get("answer") or {}
    st.markdown(f'<div class="qs-answer">{html.escape(ans.get("text", ""))} {prov(ans.get("provenance"))}</div>',
                unsafe_allow_html=True)
    st.markdown("**1 · Understanding**")
    st.write(out.get("understanding"))
    for a in out.get("assumptions", []):
        st.caption(f"assumption: {a}")
    st.markdown("**2 · Detected tables & columns**")
    st.write(", ".join(out.get("detected_tables", [])) + "  ·  " + ", ".join(out.get("detected_columns", [])))
    st.markdown("**3 · Generated SQL**")
    st.code(out["generated_sql"], language="sql")
    if out.get("optimization", {}).get("status") == "completed":
        o = out["optimization"]
        h = o["hypopg_comparison"]
        st.markdown("**4 · Optimization**")
        pct = h["planner_cost_reduction_pct"]["value"]
        if pct is not None and pct > 0:
            m = st.columns(3)
            metric_with_prov(m[0], "Planner cost before", f"{h['before_total_cost']['value'] or 0:,.0f}",
                             h["before_total_cost"]["provenance"])
            metric_with_prov(m[1], "Planner cost after (HypoPG)", f"{h['after_total_cost']['value'] or 0:,.0f}",
                             h["after_total_cost"]["provenance"])
            metric_with_prov(m[2], "Planner cost reduction", f"{pct:.1f}%",
                             h["planner_cost_reduction_pct"]["provenance"])
            st.caption(h["statement"])
        else:
            st.info(h["statement"])
        recs = o["recommendations"]
        for d in recs["indexes"]:
            st.code(d + ";", language="sql")
        if recs["indexes"]:
            st.caption("HUMAN REVIEW REQUIRED - review artifact only, never executed on this database.")
            if src["kind"] == "dynamic" and st.button("Apply first index here and measure (uploaded dataset only)"):
                ddl = recs["indexes"][0].replace("CREATE INDEX CONCURRENTLY IF NOT EXISTS", "CREATE INDEX")
                ddl = "CREATE INDEX ON " + ddl.split(" ON ", 1)[1]
                code, m2 = api.post("/api/v1/nlq/apply-index", {"data_source_id": src["id"],
                                                                "sql": out["generated_sql"], "ddl": ddl})
                if code == 200:
                    st.markdown(f"median **{m2['before_ms']['median']} ms → {m2['after_ms']['median']} ms** "
                                f"({m2['runs']} runs each) {prov(m2['provenance'])}", unsafe_allow_html=True)
                    st.caption(m2["note"])
                else:
                    show_error(m2)
        if o.get("events"):
            st.warning("SEARCH_BUDGET_EXHAUSTED - exploration was bounded")
    ex = out.get("execution_result")
    if ex:
        st.markdown(f"**5 · Execution result** ({ex['row_count']} rows{', truncated' if ex['truncated'] else ''}) "
                    f"{prov(ex.get('provenance'))}", unsafe_allow_html=True)
        st.dataframe([dict(zip(ex["columns"], r)) for r in ex["rows"][:200]], width="stretch", hide_index=True)


def attention_section(gnn: dict, plots) -> None:
    att = (gnn or {}).get("attention")
    if not att:
        return
    st.markdown(f"**GATv2 · attention-based structural importance signal (interpretability aid, not causal "
                f"proof)** {prov(att.get('provenance'))}", unsafe_allow_html=True)
    if att.get("model_mode") != "trained":
        st.caption(f"fallback: {att.get('reason')}")
    rows = att.get("node_importance") or []
    if rows and att.get("edge_attention"):
        st.plotly_chart(plots.attention_dag(gnn["graph_nodes"], att["edge_attention"], rows), width="stretch")
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"{att.get('architecture')} · feature schema v{att.get('feature_schema_version')} · checkpoint "
               f"v{att.get('checkpoint_version')} · dataset {att.get('training_dataset')} · "
               "attention importance and planner cost share are separate fields.")


def privacy_upgrade_section(status: dict | None) -> None:
    if not status:
        return
    dp = status.get("differential_privacy") or {}
    m = st.columns(4)
    m[0].metric("Raw data exposed to AI", status.get("raw_data_exposed_to_ai", 0))
    m[1].metric("Differential privacy", "on" if dp.get("enabled") else "off")
    m[2].metric("ε spent", dp.get("epsilon_spent"))
    m[3].metric("ε remaining", dp.get("epsilon_remaining"))
    if dp.get("enabled"):
        st.markdown(prov(["DP-PROTECTED"]) + f" <span class='qs-sub'>{html.escape(dp.get('claim', ''))}; "
                    f"{html.escape(dp.get('composition', ''))}</span>", unsafe_allow_html=True)
    else:
        st.caption("Differential privacy is not active (optional; enable with QS_FEATURE_DP=true). Row counts and "
                   "other statistics sent to the AI are still pseudonymised, but not noised.")
