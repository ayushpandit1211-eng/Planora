"""Plotly figures for the dashboard (V4 forest glass styling; data, sorting and scales unchanged)."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
BG = "rgba(0,0,0,0)"
PLOT_BG = "#0C2633"          # light ivory reading interior
INK = "#F3F7F9"
GRID = "#294653"
ACCENT = "#F1BC70"           # warm orange: ordinary data
HOT = "#F0808F"              # red: bottleneck (always paired with a text marker)
MUTED = "#81ACBE"            # muted mauve: secondary data
GOOD = "#79D6AE"
LINE = "#0C2633"             # marker outline on the light interior


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=dict(text=title, font=dict(size=15, color=INK)), height=height,
                      paper_bgcolor=BG, plot_bgcolor=PLOT_BG, margin=dict(l=12, r=12, t=46, b=12),
                      font=dict(size=12, color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
                      hoverlabel=dict(bgcolor="#102F40", bordercolor="#385663", font=dict(color=INK, size=12)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#385663", tickfont=dict(color="#B6CDD7"))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#385663", tickfont=dict(color="#B6CDD7"))
    return fig


def plan_dag(nodes: list[dict[str, Any]], edges: list[list[int]], highlight_id: int | None) -> go.Figure:
    """Tree layout: y = depth, x = order among nodes of the same depth. Bottleneck highlighted."""
    by_depth: dict[int, list[dict]] = {}
    for n in nodes:
        by_depth.setdefault(n["depth"], []).append(n)
    pos = {}
    for d, ns in by_depth.items():
        for i, n in enumerate(ns):
            pos[n["id"]] = ((i + 1) / (len(ns) + 1), -d)
    fig = go.Figure()
    for p, c in edges:
        (x0, y0), (x1, y1) = pos[p], pos[c]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=MUTED, width=1.5),
                                 hoverinfo="skip", showlegend=False))
    maxc = max((n.get("effective_cost") or 0 for n in nodes), default=1) or 1
    xs = [pos[n["id"]][0] for n in nodes]
    ys = [pos[n["id"]][1] for n in nodes]
    colors = [HOT if n["id"] == highlight_id else ACCENT for n in nodes]
    sizes = [18 + 30 * ((n.get("effective_cost") or 0) / maxc) for n in nodes]
    text = [("BOTTLENECK: " if n["id"] == highlight_id else "") + f"{n['operator']}"
            + (f"<br>{n['relation']}" if n.get("relation") else "") for n in nodes]
    hover = [f"<b>{n['operator']}</b><br>relation: {n.get('relation')}<br>index: {n.get('index')}<br>"
             f"total cost: {n['total_cost']:.2f}<br>self cost: {n['exclusive_cost']:.2f}<br>"
             f"loops~{n.get('estimated_loops', 1):.0f}<br>effective cost: {n.get('effective_cost', 0):.2f}<br>"
             f"plan rows: {n['plan_rows']:.0f}" for n in nodes]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=text, textposition="bottom center",
                             marker=dict(size=sizes, color=colors, line=dict(color=LINE, width=2)),
                             hovertext=hover, hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "Execution plan graph (red = bottleneck, size = effective cost)", 120 + 110 * len(by_depth))


def operator_costs(nodes: list[dict[str, Any]], highlight_id: int | None) -> go.Figure:
    ns = sorted(nodes, key=lambda n: n.get("effective_cost") or 0, reverse=True)[:12]
    fig = go.Figure(go.Bar(x=[n.get("effective_cost") or 0 for n in ns],
                           y=[f"#{n['id']} {n['operator']}" + (" (bottleneck)" if n["id"] == highlight_id else "")
                              for n in ns], orientation="h",
                           marker_color=[HOT if n["id"] == highlight_id else ACCENT for n in ns]))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="log", title="effective planner cost (log)")
    return _layout(fig, "Operators by effective planner cost")


def class_probabilities(probs: dict[str, float] | None) -> go.Figure | None:
    if not probs:
        return None
    fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation="h", marker_color=ACCENT))
    fig.update_xaxes(range=[0, 1], title="softmax probability")
    return _layout(fig, "GNN class probabilities (trained model)", 260)


def cost_before_after(before: float | None, after: float | None, rewrite_after: float | None = None) -> go.Figure:
    labels, vals, colors = ["Baseline"], [before or 0], [MUTED]
    if after is not None:
        labels.append("With virtual indexes")
        vals.append(after)
        colors.append(GOOD)
    if rewrite_after is not None:
        labels.append("Rewritten SQL")
        vals.append(rewrite_after)
        colors.append(ACCENT)
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, text=[f"{v:,.0f}" for v in vals], textposition="outside"))
    fig.update_yaxes(type="log", title="planner cost (log)")
    return _layout(fig, "Planner-estimated cost (HypoPG / EXPLAIN) - not runtime", 340)


def per_candidate(rows: list[dict[str, Any]]) -> go.Figure | None:
    rows = [r for r in rows if r.get("planner_cost_reduction_pct") is not None]
    if not rows:
        return None
    fig = go.Figure(go.Bar(x=[r["planner_cost_reduction_pct"] for r in rows],
                           y=[r["candidate"][:70] for r in rows], orientation="h",
                           marker_color=[GOOD if r.get("used_by_planner") else MUTED for r in rows]))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="planner-estimated cost reduction % (alone)")
    return _layout(fig, "Per-candidate HypoPG experiments (green = used by planner)", 120 + 40 * len(rows))


def q_values(q: dict[str, float]) -> go.Figure | None:
    if not q:
        return None
    items = sorted(q.items(), key=lambda kv: kv[1], reverse=True)[:14]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=[GOOD if v >= 0 else HOT for _, v in items]))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "RL Q-values (learned planner-cost returns)", 140 + 26 * len(items))


def drift_scores(scores: dict[str, float], threshold: float) -> go.Figure | None:
    if not scores:
        return None
    fig = go.Figure(go.Bar(x=list(scores.keys()), y=list(scores.values()),
                           marker_color=[HOT if v >= threshold else ACCENT for v in scores.values()]))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#805308", annotation_text="threshold")
    fig.update_yaxes(range=[0, 1], title="Jensen-Shannon divergence")
    return _layout(fig, "Workload drift (recent vs historical)", 320)


def benchmark_variants(results: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[r["variant"] for r in results],
                           y=[r["planner_cost_reduction_pct"] or 0 for r in results],
                           marker_color=[GOOD if r["variant"] == "full" else ACCENT if r["variant"] in
                                         ("index_only", "rewrite_only", "gnn_assisted", "rl_assisted") else MUTED
                                         for r in results]))
    fig.update_yaxes(title="planner-estimated cost reduction %")
    return _layout(fig, "Benchmark modes and ablations (planner estimates)", 380)


def attention_dag(nodes: list[dict[str, Any]], edge_attention: list[dict[str, Any]],
                  importance: list[dict[str, Any]]) -> go.Figure:
    """Final Upgrade P8: plan DAG with GATv2 attention overlay (edge width = attention, node size = incoming
    attention mass). An attention-based structural importance signal - interpretability aid, not causal proof."""
    by_depth: dict[int, list[dict]] = {}
    for n in nodes:
        by_depth.setdefault(n["depth"], []).append(n)
    pos = {}
    for d, ns in by_depth.items():
        for i, n in enumerate(ns):
            pos[n["id"]] = ((i + 1) / (len(ns) + 1), -d)
    fig = go.Figure()
    seen = {}
    for e in edge_attention:                          # both directions -> combine into one undirected weight
        k = tuple(sorted((e["source"], e["target"])))
        seen[k] = max(seen.get(k, 0.0), e["weight"])
    for (a, b), w in seen.items():
        if a in pos and b in pos:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=ACCENT, width=1 + 7 * w),
                                     hovertext=f"attention {w:.3f}", hoverinfo="text", showlegend=False))
    imp = {r["id"]: r.get("attention_importance") or 0.0 for r in importance}
    mx = max(imp.values(), default=1.0) or 1.0
    xs = [pos[n["id"]][0] for n in nodes]
    ys = [pos[n["id"]][1] for n in nodes]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[n["operator"] for n in nodes],
                             textposition="bottom center",
                             marker=dict(size=[14 + 30 * imp.get(n["id"], 0) / mx for n in nodes], color=HOT,
                                         opacity=0.8, line=dict(color=LINE, width=2)),
                             hovertext=[f"{n['operator']}: attention importance {imp.get(n['id'], 0):.3f}"
                                        for n in nodes], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "GATv2 attention overlay (interpretability aid, not causal proof)", 120 + 110 * len(by_depth))
