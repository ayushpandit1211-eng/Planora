"""LangGraph state (Section 22). Everything in the state is anonymised / sanitised; trusted-only
data (raw SQL, real schema, real DDL) is held in `Services.trusted[request_id]`, never in state."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class QSState(TypedDict, total=False):
    # --- Section 22 required keys
    request_id: str
    masked_sql: str
    gnn_diagnosis: dict
    schema_stats: dict
    proposed_sql: str | None
    proposed_indexes: list
    partitioning_recommendation: dict | None
    rl_recommendation: dict | None
    sandbox_results: dict | None
    blast_radius: dict | None
    agent_logs: Annotated[list, operator.add]
    final_report: dict | None
    # --- supporting keys
    agent_status: Annotated[dict, merge_dicts]
    privacy: dict
    plan_analysis: dict
    graph_summary: dict
    access_pattern: dict
    workload_context: dict
    regression: dict | None
    route: list
    sql_agent: dict | None
    schema_agent: dict | None
    index_agent: dict | None
    agent_recommendation: dict | None
    validation: dict | None
    deployment: dict | None
    error: dict | None
    llm_calls: Annotated[list, operator.add]


AGENT_NODES = ["supervisor", "sql_rewriter", "schema_expert", "index_strategist", "rl_optimizer",
               "recommendation_validator", "hypopg_sandbox", "blast_radius", "final_report"]
STATUSES = ("QUEUED", "RUNNING", "COMPLETED", "SKIPPED", "REJECTED", "ERROR")


def initial_state(request_id: str) -> QSState:
    return QSState(request_id=request_id, agent_logs=[], agent_status={n: "QUEUED" for n in AGENT_NODES},
                   proposed_indexes=[], proposed_sql=None, partitioning_recommendation=None,
                   rl_recommendation=None, sandbox_results=None, blast_radius=None, final_report=None,
                   error=None, llm_calls=[])
