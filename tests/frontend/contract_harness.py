"""Frontend contract harness (V3 redesign). Not a pytest module - run directly:

    BACKEND_URL=http://localhost:8000 python tests/frontend/contract_harness.py out.json

Drives dashboard/app.py with Streamlit AppTest through fixed scenarios and records every HTTP request the
dashboard makes (method, path, JSON payload keys/values, body size). Comparing the output before and after a
presentation change proves that endpoints, payloads and call counts are unchanged. Only read-only or
optimise-only actions are exercised; destructive actions (commit, delete, apply-index) are never clicked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
CALLS: list[dict] = []
_orig_request = httpx.Client.request
_orig_stream = httpx.Client.stream


def _rec(method, url, kw):
    path = str(url).split("://", 1)[-1]
    path = path[path.find("/"):] if "/" in path else path
    body = kw.get("json")
    CALLS.append({"m": method, "path": path.split("?sid=")[0],
                  "json": body if body is None or len(json.dumps(body)) < 400 else sorted(body),
                  "content_bytes": len(kw["content"]) if kw.get("content") is not None else None,
                  "headers": sorted(k.lower() for k in (kw.get("headers") or {}))})


def request(self, method, url, **kw):
    _rec(method, url, kw)
    return _orig_request(self, method, url, **kw)


def stream(self, method, url, **kw):
    _rec(method, url, kw)
    return _orig_stream(self, method, url, **kw)


httpx.Client.request = request
httpx.Client.stream = stream


def run(name, fn, out):
    CALLS.clear()
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=600)
    at.query_params["sid"] = "sess-contract-0001"
    at.run()
    fn(at)
    out[name] = {"calls": list(CALLS), "exceptions": [str(e.value)[:200] for e in at.exception]}


def by_label(widgets, label):
    for w in widgets:
        if getattr(w, "label", None) == label:
            return w
    raise KeyError(label)


def s_initial(at):
    pass


def s_demo_run(at):
    by_label(at.sidebar.selectbox, "Demonstration scenario").select_index(1).run()
    by_label(at.sidebar.button, "Load scenario").click().run()
    by_label(at.button, "Run optimisation pipeline").click().run()


def s_all_tabs(at):
    by_label(at.sidebar.toggle, "Demo mode (5 core tabs)").set_value(False).run()


def s_upload_mode(at):
    at.sidebar.radio[0].set_value("Upload Dataset (CSV, JSON)").run()


def s_dataset_source(at):
    import os
    with httpx.Client() as c:   # not recorded: harness setup only
        srcs = _orig_request(c, "GET", os.environ.get("BACKEND_URL", "http://localhost:8000") + "/api/v1/sources",
                             headers={"X-Session-Id": "sess-contract-0001"}).json()["sources"]
    dyn = sorted(s["id"] for s in srcs if s["kind"] == "dynamic" and s["name"] == "enterprise_survey2")
    if not dyn:
        return
    by_label(at.sidebar.selectbox, "Active source").set_value(dyn[0]).run()


def s_find_workload(at):
    by_label(at.text_input, "Natural-language performance question").input("weekly sales dashboard slow").run()
    by_label(at.button, "Find workload").click().run()


def s_ask_db(at):
    s_dataset_source(at)
    by_label(at.text_input, "Ask your database").input("count by variable").run()
    by_label(at.button, "Ask").click().run()


def s_run_sql(at):
    s_dataset_source(at)
    by_label(at.text_area, "SQL").input("SELECT variable, count(*) FROM enterprise_survey2 GROUP BY 1").run()
    by_label(at.button, "Run SQL").click().run()


SCENARIOS = {"initial": s_initial, "find_workload": s_find_workload, "all_tabs": s_all_tabs,
             "upload_mode": s_upload_mode, "dataset_source": s_dataset_source, "demo_run": s_demo_run,
             "ask_db": s_ask_db, "run_sql": s_run_sql}

if __name__ == "__main__":
    out: dict = {}
    for n, f in SCENARIOS.items():
        run(n, f, out)
        print(n, len(out[n]["calls"]), "calls", "exceptions:", out[n]["exceptions"])
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
