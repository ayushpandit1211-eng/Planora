"""GNN bottleneck classifier (Section 20).

GCNConv(F,32) -> ReLU -> GCNConv(32,16) -> ReLU -> global_max_pool -> Linear(16,4)
(graph readout added so a whole plan gets one class; see DECISIONS D8).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool

from app.contracts import BOTTLENECK_CLASSES
from app.errors import ErrorCategory, QuerySentinelError
from app.gnn.mapper import NUM_FEATURES

CHECKPOINT_VERSION = 1
NUM_CLASSES = len(BOTTLENECK_CLASSES)


class PlanGCN(nn.Module):
    def __init__(self, input_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = GCNConv(input_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.head = nn.Linear(16, num_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.act(self.conv1(x, edge_index))
        h = self.act(self.conv2(h, edge_index))
        g = global_max_pool(h, batch)
        return self.head(g)


def save_checkpoint(model: PlanGCN, path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CHECKPOINT_VERSION, "feature_dim": NUM_FEATURES, "num_classes": NUM_CLASSES,
               "classes": BOTTLENECK_CLASSES, "state_dict": model.state_dict(), "meta": meta}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> tuple[PlanGCN, dict[str, Any]]:
    """Load and VALIDATE a checkpoint. Raises structured errors for missing/corrupt files."""
    path = Path(path)
    if not path.exists():
        raise QuerySentinelError(ErrorCategory.GNN_UNAVAILABLE, "GNN checkpoint not found", stage="gnn")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT,
                                 f"GNN checkpoint unreadable ({type(exc).__name__})", stage="gnn") from None
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION \
            or payload.get("feature_dim") != NUM_FEATURES or payload.get("num_classes") != NUM_CLASSES:
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT,
                                 "GNN checkpoint metadata incompatible with this code version", stage="gnn")
    model = PlanGCN()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT,
                                 f"GNN state_dict invalid ({type(exc).__name__})", stage="gnn") from None
    for t in model.state_dict().values():
        if not torch.isfinite(t).all():
            raise QuerySentinelError(ErrorCategory.CORRUPT_CHECKPOINT, "GNN weights contain NaN/Inf", stage="gnn")
    model.eval()
    return model, dict(payload.get("meta") or {})
