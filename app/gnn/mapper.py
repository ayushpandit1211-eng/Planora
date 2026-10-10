"""Execution plan -> PyTorch Geometric graph (Section 19).

Nodes = execution operators, edges = parent/child execution relationships (added in both
directions so GCN message passing flows up and down the tree). Node features use log1p
for numerical stability.
"""
from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from app.contracts import PlanAnalysis

OPERATOR_MAP: dict[str, int] = {
    "Seq Scan": 0,
    "Index Scan": 1,
    "Index Only Scan": 2,
    "Bitmap Heap Scan": 3,
    "Bitmap Index Scan": 4,
    "Nested Loop": 5,
    "Hash Join": 6,
    "Merge Join": 7,
    "Aggregate": 8,
    "Sort": 9,
    "Limit": 10,
    "Materialize": 11,
}
# Closely related operators share an ID (documented in docs/architecture.md); everything else
# (Gather, Hash, Result, Append, BitmapAnd, ...) is OTHER.
OPERATOR_ALIASES = {"Incremental Sort": 9, "Memoize": 11, "Parallel Seq Scan": 0, "WindowAgg": 8,
                    "Group": 8, "Unique": 8, "SetOp": 8}
OTHER_OPERATOR_ID = 12
NUM_OPERATORS = 13

# Operator FAMILY one-hot lets the model generalise to operators of a known family that it has
# not seen during training (e.g. Merge Join is a join even if only Hash/Nested Loop were seen).
FAMILIES = {
    "scan": {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan",
             "Tid Scan", "Tid Range Scan", "CTE Scan", "Subquery Scan", "Function Scan", "Values Scan",
             "Foreign Scan", "Sample Scan", "WorkTable Scan"},
    "join": {"Nested Loop", "Hash Join", "Merge Join"},
    "sort_agg": {"Sort", "Incremental Sort", "Aggregate", "Hash", "WindowAgg", "Group", "Unique", "SetOp"},
}
FAMILY_NAMES = ["scan", "join", "sort_agg", "other"]


def operator_family(node_type: str) -> int:
    for i, fam in enumerate(FAMILY_NAMES[:3]):
        if node_type in FAMILIES[fam]:
            return i
    return 3

FEATURE_NAMES = (
    [f"op_onehot_{i}" for i in range(NUM_OPERATORS)]
    + [f"family_{f}" for f in FAMILY_NAMES]
    + ["operator_id_norm", "log_startup_cost", "log_total_cost", "log_plan_rows", "log_plan_width",
       "log_exclusive_cost", "log_estimated_loops", "effective_cost_share", "depth_norm",
       "has_filter", "has_index_cond", "is_parallel", "is_subplan",
       "is_top_effective_cost", "share_gap_to_max", "log_plan_total_cost"]
)
NUM_FEATURES = len(FEATURE_NAMES)


def operator_id(node_type: str) -> int:
    return OPERATOR_MAP.get(node_type, OPERATOR_ALIASES.get(node_type, OTHER_OPERATOR_ID))


def node_features(analysis: PlanAnalysis) -> list[list[float]]:
    root_cost = max(analysis.total_cost, 1e-9)
    shares = [min(n.effective_cost / root_cost, 1.0) for n in analysis.nodes]
    max_share = max(shares) if shares else 0.0
    top_id = max(range(len(shares)), key=lambda i: shares[i]) if shares else -1
    feats = []
    for idx, n in enumerate(analysis.nodes):
        op = operator_id(n.node_type)
        onehot = [0.0] * NUM_OPERATORS
        onehot[op] = 1.0
        family = [0.0] * len(FAMILY_NAMES)
        family[operator_family(n.node_type)] = 1.0
        share = min(n.effective_cost / root_cost, 1.0) if root_cost > 0 else 0.0
        feats.append(onehot + family + [
            op / (NUM_OPERATORS - 1),
            math.log1p(max(n.startup_cost, 0.0)),
            math.log1p(max(n.total_cost, 0.0)),
            math.log1p(max(n.plan_rows, 0.0)),
            math.log1p(max(n.plan_width, 0)),
            math.log1p(max(n.exclusive_cost, 0.0)),
            math.log1p(max(n.estimated_loops, 1.0) - 1.0),
            share,
            min(n.depth / 10.0, 1.0),
            1.0 if (n.filter or n.join_filter) else 0.0,
            1.0 if (n.index_cond or n.recheck_cond) else 0.0,
            1.0 if n.parallel_aware else 0.0,
            1.0 if n.parent_relationship in ("SubPlan", "InitPlan") else 0.0,
            1.0 if idx == top_id else 0.0,
            share - max_share,
            math.log1p(max(analysis.total_cost, 0.0)),
        ])
    return feats


def plan_to_graph(analysis: PlanAnalysis, label: int | None = None) -> Data:
    if not analysis.nodes:
        raise ValueError("plan has no nodes")
    x = torch.tensor(node_features(analysis), dtype=torch.float32)
    src, dst = [], []
    for p, c in analysis.edges:
        src += [p, c]
        dst += [c, p]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    validate_graph(data)
    return data


def validate_graph(data: Data) -> None:
    if data.x.dim() != 2 or data.x.size(1) != NUM_FEATURES:
        raise ValueError(f"x must be [N,{NUM_FEATURES}]")
    if not torch.isfinite(data.x).all():
        raise ValueError("non-finite node features")
    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2,E]")
    if data.edge_index.numel() and (data.edge_index.min() < 0 or data.edge_index.max() >= data.x.size(0)):
        raise ValueError("edge_index references a missing node")


def graph_json(analysis: PlanAnalysis) -> tuple[list[dict], list[list[int]]]:
    """Dashboard-friendly (sanitised) node/edge lists."""
    nodes = [{"id": n.node_id, "operator": n.node_type, "operator_id": operator_id(n.node_type),
              "relation": n.relation_name, "index": n.index_name, "total_cost": n.total_cost,
              "exclusive_cost": round(n.exclusive_cost, 2), "effective_cost": round(n.effective_cost, 2),
              "estimated_loops": n.estimated_loops, "plan_rows": n.plan_rows, "depth": n.depth,
              "parent_relationship": n.parent_relationship}
             for n in analysis.nodes]
    return nodes, [[p, c] for p, c in analysis.edges]
