"""GATv2 plan-graph model with exposed attention (Final Upgrade P8, behind QS_FEATURE_GATV2).

Uses the installed PyTorch Geometric GATv2Conv (signature read from torch_geometric 2.8:
GATv2Conv(in, out, heads, concat, negative_slope, dropout, add_self_loops, edge_dim, fill_value, bias,
share_weights, residual); forward(x, edge_index, edge_attr, return_attention_weights)).

Decisions
- add_self_loops=False (and residual=True so a node keeps its own features): the attention tensor returned by
  the layer then maps 1:1 onto the real plan edges, no self-loop entries to filter out.
- Edge features (edge_dim=8) are safe structural facts only: flow direction, Outer/Inner relationship, whether the
  parent is a join, and the parent's join type. The model never receives data values.
- Attention is an "attention-based structural importance signal (interpretability aid, not causal proof)".
- A checkpoint is loaded only if architecture + feature/edge schema versions match; a GCN checkpoint is never
  loaded into this architecture. Without a compatible checkpoint the analyzer stays in analytic fallback.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool

from app.contracts import BOTTLENECK_CLASSES, PlanAnalysis
from app.errors import ErrorCategory, QuerySentinelError
from app.gnn.mapper import NUM_FEATURES, node_features

ARCHITECTURE = "PlanGATv2"
FEATURE_SCHEMA_VERSION = 1           # 32 node features of app/gnn/mapper.py
EDGE_FEATURE_VERSION = 1
CHECKPOINT_VERSION = 1
EDGE_DIM = 8
HEADS = 2
WORDING = "attention-based structural importance signal (interpretability aid, not causal proof)"
_JOIN_NODES = ("Nested Loop", "Hash Join", "Merge Join")


def edge_features(parent, child, direction_down: bool) -> list[float]:
    rel = (child.parent_relationship or "").lower()
    pj = parent.node_type in _JOIN_NODES
    jt = (parent.join_type or "").lower()
    return [1.0 if direction_down else 0.0, 0.0 if direction_down else 1.0,
            1.0 if rel == "outer" else 0.0, 1.0 if rel == "inner" else 0.0, 1.0 if pj else 0.0,
            1.0 if pj and jt == "inner" else 0.0, 1.0 if pj and jt in ("left", "right", "full") else 0.0,
            1.0 if pj and jt in ("semi", "anti", "right semi", "right anti") else 0.0]


def plan_to_gat_graph(analysis: PlanAnalysis, label: int | None = None) -> Data:
    if not analysis.nodes:
        raise ValueError("plan has no nodes")
    by_id = {n.node_id: n for n in analysis.nodes}
    x = torch.tensor(node_features(analysis), dtype=torch.float32)
    src, dst, attrs = [], [], []
    for p, c in analysis.edges:
        src.append(p); dst.append(c); attrs.append(edge_features(by_id[p], by_id[c], True))     # noqa: E702
        src.append(c); dst.append(p); attrs.append(edge_features(by_id[p], by_id[c], False))    # noqa: E702
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    ea = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.empty((0, EDGE_DIM), dtype=torch.float32)
    data = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=x.size(0))
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    validate_edge_features(data)
    return data


def validate_edge_features(data: Data) -> None:
    if data.edge_attr is None or data.edge_attr.dim() != 2 or data.edge_attr.size(1) != EDGE_DIM:
        raise ValueError(f"edge_attr must be [E,{EDGE_DIM}]")
    if data.edge_attr.size(0) != data.edge_index.size(1):
        raise ValueError("edge_attr rows must match edge_index columns")
    if not torch.isfinite(data.edge_attr).all():
        raise ValueError("non-finite edge features")


class PlanGATv2(nn.Module):
    def __init__(self, input_features: int = NUM_FEATURES, num_classes: int = len(BOTTLENECK_CLASSES)):
        super().__init__()
        self.conv1 = GATv2Conv(input_features, 16, heads=HEADS, edge_dim=EDGE_DIM, add_self_loops=False,
                               residual=True)
        self.conv2 = GATv2Conv(16 * HEADS, 16, heads=HEADS, concat=False, edge_dim=EDGE_DIM, add_self_loops=False,
                               residual=True)
        self.head = nn.Linear(16, num_classes)
        self.act = nn.ELU()

    def forward(self, x, edge_index, edge_attr, batch=None, return_attention: bool = False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h, (ei1, a1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        h = self.act(h)
        h, (ei2, a2) = self.conv2(h, edge_index, edge_attr, return_attention_weights=True)
        h = self.act(h)
        out = self.head(global_max_pool(h, batch))
        if return_attention:
            return out, [(ei1, a1), (ei2, a2)]
        return out


def extract_attention(analysis: PlanAnalysis, data: Data, att: list) -> dict[str, Any]:
    """Map attention back to real plan edges; aggregate heads (mean) and layers (mean)."""
    real = {(int(s), int(d)) for s, d in data.edge_index.t().tolist()}
    raw, per_edge = [], {}
    for layer, (ei, alpha) in enumerate(att, 1):
        if ei.size(1) != data.edge_index.size(1):
            raise ValueError("attention edge count differs from plan edges (unexpected self-loops)")
        for k, (s, d) in enumerate(ei.t().tolist()):
            if (s, d) not in real:
                raise ValueError("attention references an edge that is not in the plan")
            for h in range(alpha.size(1)):
                raw.append({"source": s, "target": d, "layer": layer, "head": h, "weight": round(float(alpha[k, h]), 5)})
            per_edge.setdefault((s, d), []).append(float(alpha[k].mean()))
    edge_attention = [{"source": s, "target": d, "weight": round(sum(v) / len(v), 5)} for (s, d), v in
                      sorted(per_edge.items())]
    importance = {n.node_id: 0.0 for n in analysis.nodes}
    for e in edge_attention:
        importance[e["target"]] += e["weight"]               # incoming attention mass
    total_cost = max(sum(max(n.exclusive_cost, 0.0) for n in analysis.nodes), 1e-9)
    nodes = [{"id": n.node_id, "operator": n.node_type, "attention_importance": round(importance[n.node_id], 4),
              "planner_cost_share": round(max(n.exclusive_cost, 0.0) / total_cost, 4)} for n in analysis.nodes]
    bn = max(nodes, key=lambda x: (x["attention_importance"], x["planner_cost_share"])) if nodes else None
    return {"edge_attention": edge_attention, "node_importance": nodes, "raw_attention": raw,
            "bottleneck_node": bn, "aggregation": "mean over heads, then mean over layers; node importance = "
                                                  "incoming attention mass", "wording": WORDING,
            "note": "attention importance and planner cost share are separate fields; they are not combined"}


# ------------------------------------------------------------------ checkpoints
def checkpoint_meta(dataset_id: str, metrics: dict | None = None) -> dict[str, Any]:
    return {"architecture": ARCHITECTURE, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "edge_feature_version": EDGE_FEATURE_VERSION, "checkpoint_version": CHECKPOINT_VERSION,
            "feature_dim": NUM_FEATURES, "edge_dim": EDGE_DIM, "num_classes": len(BOTTLENECK_CLASSES),
            "training_dataset": dataset_id, "metrics": metrics or {},
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save_gat_checkpoint(model: PlanGATv2, path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"meta": meta, "state_dict": model.state_dict()}, tmp)
    tmp.replace(path)


def load_gat_checkpoint(path: str | Path) -> tuple[PlanGATv2, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise QuerySentinelError(ErrorCategory.GNN_UNAVAILABLE, "GATv2 checkpoint not found", stage="gnn")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, f"GATv2 checkpoint unreadable "
                                 f"({type(exc).__name__})", stage="gnn") from None
    meta = payload.get("meta") if isinstance(payload, dict) else None
    want = {"architecture": ARCHITECTURE, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "edge_feature_version": EDGE_FEATURE_VERSION, "checkpoint_version": CHECKPOINT_VERSION}
    if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in want.items()):
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT,
                                 "incompatible checkpoint for the GATv2 architecture (old GCN checkpoints are never "
                                 "loaded into GATv2)", stage="gnn")
    model = PlanGATv2()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, f"GATv2 state_dict invalid "
                                 f"({type(exc).__name__})", stage="gnn") from None
    model.eval()
    return model, meta


# ------------------------------------------------------------------ training (labelled synthetic plans)
def train_gat(path: str | Path | None = None, n_per_class: int = 300, epochs: int = 40, seed: int = 1234,
              log=print) -> dict[str, Any]:
    """Trains on the SAME labelled synthetic plan generator as the GCN (app/gnn/training.py); real evaluation
    numbers on held-out and unseen-variant splits are returned and stored in the checkpoint metadata."""
    import random
    from collections import Counter
    from torch_geometric.loader import DataLoader
    from app.gnn.training import GENERATORS, _distract
    from app.plan_parser import parse_explain

    def dataset(n, s, unseen=False):
        rng = random.Random(s)
        out = []
        for label, (fn, seen_v, unseen_v) in GENERATORS.items():
            vs = unseen_v if unseen else seen_v
            for i in range(n):
                plan = fn(rng, vs[i % len(vs)])
                if label != 3 and rng.random() < 0.6:
                    plan = _distract(rng, plan)
                out.append(plan_to_gat_graph(parse_explain({"Plan": plan}), label))
        rng.shuffle(out)
        return out

    def score(model, data):
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for b in DataLoader(data, batch_size=128):
                p = model(b.x, b.edge_index, b.edge_attr, b.batch).argmax(1)
                ok += int((p == b.y).sum())
                tot += b.num_graphs
        return {"accuracy": round(ok / max(tot, 1), 4), "n": tot}

    torch.manual_seed(seed)
    tr, va = dataset(n_per_class, seed), dataset(max(n_per_class // 4, 20), seed + 1)
    ho, un = dataset(max(n_per_class // 4, 20), seed + 2), dataset(max(n_per_class // 4, 20), seed + 3, True)
    model = PlanGATv2()
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for b in DataLoader(tr, batch_size=64, shuffle=True):
            opt.zero_grad()
            lossf(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).backward()
            opt.step()
        v = score(model, va)["accuracy"]
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0:
            log(f"[gatv2] epoch {ep} val_acc={v:.3f}")
    model.load_state_dict(best_state)
    metrics = {"train_size": len(tr), "best_val_accuracy": best, "heldout_same_distribution": score(model, ho),
               "heldout_unseen_template_variants": score(model, un), "epochs": epochs, "seed": seed,
               "class_balance_train": dict(Counter(int(d.y) for d in tr)),
               "label_source": "synthetic scenario generator (app/gnn/training.py) [SYNTHETIC DATA]"}
    if path:
        save_gat_checkpoint(model, path, checkpoint_meta("synthetic_plan_generator_v1", metrics))
    return metrics
