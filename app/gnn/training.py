"""GNN training: synthetic PostgreSQL-shaped plan generation, training, evaluation, saving.

Labels come from the generating SCENARIO (which operator is made to dominate), not from the
analytic fallback rules. Plans are emitted in real EXPLAIN-JSON structure and go through the
same parser/mapper used at inference time. Held-out evaluation uses (a) a different seed and
(b) "unseen" template variants that never appear in training (see UNSEEN_VARIANTS).
This is a prototype-scale experiment; results do not prove generalisation to arbitrary
production workloads (README limitations).
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from app.contracts import BOTTLENECK_CLASSES
from app.gnn.mapper import plan_to_graph
from app.gnn.model import PlanGCN, save_checkpoint
from app.plan_parser import parse_explain

TABLES = ["t_events", "t_orders", "t_items", "t_users", "t_logs", "t_payments", "t_sessions"]


def _node(node_type: str, startup: float, total: float, rows: float, width: int = 32, **extra) -> dict:
    d = {"Node Type": node_type, "Startup Cost": round(startup, 2), "Total Cost": round(total, 2),
         "Plan Rows": max(int(rows), 1), "Plan Width": width}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


def _scan_seq(rng, rows_table, selectivity, filt=True, rel=None):
    cost = rows_table * rng.uniform(0.012, 0.03)
    out = max(rows_table * selectivity, 1)
    return _node("Seq Scan", 0.0, cost, out, Filter="(c = ?)" if filt else None,
                 **{"Relation Name": rel or rng.choice(TABLES), "Parallel Aware": rng.random() < 0.3})


def _scan_index(rng, rows, rel=None, only=False):
    cost = 0.42 + rows * rng.uniform(0.5, 4) + rng.uniform(4, 8)
    return _node("Index Only Scan" if only else "Index Scan", 0.29 + rng.random() * 0.2, cost, rows,
                 **{"Relation Name": rel or rng.choice(TABLES), "Index Name": "idx_x", "Index Cond": "(c = ?)"})


def _outer(d: dict, rel: str = "Outer") -> dict:
    return {**d, "Parent Relationship": rel}


def _wrap(rng, child: dict, kinds=("Limit", "Gather", "Result", "none")) -> dict:
    k = rng.choice(kinds)
    if k == "none":
        return child
    tc = child["Total Cost"]
    if k == "Gather":
        return _node("Gather", 1000.0, tc + 1000 + rng.uniform(0, 500), child["Plan Rows"] * 2, Plans=[_outer(child)])
    if k == "Limit":
        return _node("Limit", child["Startup Cost"], tc * rng.uniform(0.6, 1.0),
                     min(child["Plan Rows"], rng.choice([10, 50, 100])), Plans=[_outer(child)])
    return _node("Result", 0.0, tc + 0.01, child["Plan Rows"], Plans=[_outer(child)])


def _distract(rng, plan: dict, max_share: float = 0.12) -> dict:
    """Add a CHEAP operator of another family (<= max_share of total cost) so the model must
    learn which operator dominates the cost rather than which operators merely appear."""
    tc = plan["Total Cost"]
    extra = tc * rng.uniform(0.01, max_share)
    kind = rng.choice(["sort", "agg", "small_join", "gather"])
    if kind == "sort":
        return _node("Sort", tc + extra * 0.9, tc + extra, plan["Plan Rows"], **{"Sort Key": ["s"], "Plans": [_outer(plan)]})
    if kind == "agg":
        return _node("Aggregate", tc + extra * 0.9, tc + extra, max(plan["Plan Rows"] * 0.1, 1),
                     **{"Strategy": rng.choice(["Hashed", "Sorted", "Plain"]), "Group Key": ["g"], "Plans": [_outer(plan)]})
    if kind == "small_join":
        small = _scan_seq(rng, rng.uniform(20, 300), 0.5)
        small["Total Cost"] = extra * 0.5
        h = _node("Hash", small["Total Cost"], small["Total Cost"], small["Plan Rows"], Plans=[_outer(small)])
        return _node("Hash Join", h["Total Cost"], tc + extra, plan["Plan Rows"],
                     **{"Join Type": "Inner", "Hash Cond": "(a.k = b.k)",
                        "Plans": [_outer(plan), {**h, "Parent Relationship": "Inner"}]})
    return _node("Gather", 1000.0, tc + extra, plan["Plan Rows"], Plans=[_outer(plan)])


# --------------------------------------------------------------------------- scenario templates
def gen_seq_scan(rng, variant: str) -> dict:
    """class 0 - unindexed sequential scan dominates."""
    n = 10 ** rng.uniform(5, 7.3)
    if variant == "plain":
        return _wrap(rng, _scan_seq(rng, n, 10 ** rng.uniform(-5, -1)))
    if variant == "bitmap_filter":
        idx = _node("Bitmap Index Scan", 0, n * 0.004, n * 0.15, **{"Index Name": "idx_d", "Index Cond": "(d >= ?)"})
        heap = _node("Bitmap Heap Scan", n * 0.004, n * rng.uniform(0.02, 0.05), n * 0.15 * rng.uniform(0.02, 0.3),
                     **{"Relation Name": rng.choice(TABLES), "Recheck Cond": "(d >= ?)", "Filter": "(t = ?)",
                        "Plans": [_outer(idx)]})
        return _wrap(rng, heap, ("none", "Limit"))
    if variant == "subplan":  # correlated subquery scanning a table per outer row
        outer = _scan_index(rng, rng.uniform(500, 20000))
        inner = _scan_seq(rng, 10 ** rng.uniform(5, 6.7), 1e-6)
        agg = _node("Aggregate", inner["Total Cost"], inner["Total Cost"] + 0.02, 1, Plans=[_outer(inner)])
        root = dict(outer)
        root["Total Cost"] = outer["Total Cost"] + agg["Total Cost"] * outer["Plan Rows"]
        root["Plans"] = [{**agg, "Parent Relationship": "SubPlan", "Subplan Name": "SubPlan 1"}]
        return root
    if variant == "seq_under_agg":   # UNSEEN variant: seq scan dominating below a cheap aggregate
        scan = _scan_seq(rng, n, 10 ** rng.uniform(-4, -2))
        return _node("Aggregate", scan["Total Cost"], scan["Total Cost"] * 1.01, 1, Plans=[_outer(scan)])
    raise ValueError(variant)


def gen_join(rng, variant: str) -> dict:
    """class 1 - high-cost nested loop / join dominates."""
    if variant == "nl_repeated_inner":
        outer_rows = 10 ** rng.uniform(4, 5.5)
        outer = _scan_index(rng, outer_rows)
        outer["Total Cost"] = outer_rows * 0.05
        inner = _node("Materialize", 0, rng.uniform(50, 400), rng.uniform(200, 2000),
                      Plans=[_outer(_scan_seq(rng, rng.uniform(300, 3000), 0.5))])
        total = outer["Total Cost"] + inner["Total Cost"] * outer_rows * rng.uniform(0.3, 0.9)
        nl = _node("Nested Loop", 0, total, outer_rows * rng.uniform(1, 20),
                   **{"Join Type": "Inner", "Join Filter": "(a.x = b.y)",
                      "Plans": [_outer(outer), {**inner, "Parent Relationship": "Inner"}]})
        return _wrap(rng, nl, ("none", "Limit"))
    if variant == "hash_big":
        a = _scan_index(rng, 10 ** rng.uniform(3, 4))
        b = _scan_index(rng, 10 ** rng.uniform(3, 4))
        rows = a["Plan Rows"] * b["Plan Rows"] * rng.uniform(0.5, 3)
        h = _node("Hash", b["Total Cost"], b["Total Cost"], b["Plan Rows"], Plans=[_outer(b)])
        total = a["Total Cost"] + h["Total Cost"] + rows * rng.uniform(0.02, 0.05)
        hj = _node("Hash Join", h["Total Cost"], total, rows,
                   **{"Join Type": "Inner", "Hash Cond": "(a.k = b.k)",
                      "Plans": [_outer(a), {**h, "Parent Relationship": "Inner"}]})
        return _wrap(rng, hj)
    if variant == "merge_big":       # UNSEEN variant
        a = _scan_index(rng, 10 ** rng.uniform(3, 4))
        b = _scan_index(rng, 10 ** rng.uniform(3, 4))
        rows = a["Plan Rows"] * b["Plan Rows"] * rng.uniform(0.5, 3)
        total = a["Total Cost"] + b["Total Cost"] + rows * rng.uniform(0.02, 0.05)
        return _node("Merge Join", a["Total Cost"], total, rows,
                     **{"Join Type": "Inner", "Merge Cond": "(a.k = b.k)",
                        "Plans": [_outer(a), {**b, "Parent Relationship": "Inner"}]})
    raise ValueError(variant)


def gen_sort_agg(rng, variant: str) -> dict:
    """class 2 - heavy sort / aggregation dominates."""
    rows = 10 ** rng.uniform(5, 7)
    src = _scan_index(rng, rows, only=True)
    src["Total Cost"] = rows * rng.uniform(0.004, 0.01)
    if variant == "sort":
        sort_cost = rows * math.log2(rows) * rng.uniform(0.01, 0.03)
        s = _node("Sort", src["Total Cost"] + sort_cost, src["Total Cost"] + sort_cost + rows * 0.01, rows,
                  **{"Sort Key": ["c"], "Plans": [_outer(src)]})
        return _wrap(rng, s, ("none", "Limit"))
    if variant == "hashagg":
        groups = rows * rng.uniform(0.1, 0.9)
        cost = src["Total Cost"] + rows * rng.uniform(0.1, 0.4)
        agg = _node("Aggregate", cost, cost + groups * 0.02, groups,
                    **{"Strategy": "Hashed", "Group Key": ["g"], "Plans": [_outer(src)]})
        return _wrap(rng, agg, ("none", "Result"))
    if variant == "sort_then_group":   # UNSEEN variant
        sort_cost = rows * math.log2(rows) * rng.uniform(0.01, 0.03)
        s = _node("Sort", src["Total Cost"] + sort_cost, src["Total Cost"] + sort_cost + 1, rows,
                  **{"Sort Key": ["g"], "Plans": [_outer(src)]})
        return _node("Aggregate", s["Total Cost"], s["Total Cost"] + rows * 0.02, rows * 0.3,
                     **{"Strategy": "Sorted", "Group Key": ["g"], "Plans": [_outer(s)]})
    raise ValueError(variant)


def gen_efficient(rng, variant: str) -> dict:
    """class 3 - efficient / no major bottleneck."""
    if variant == "point_lookup":
        return _wrap(rng, _scan_index(rng, rng.uniform(1, 30), only=rng.random() < 0.3), ("none", "Limit"))
    if variant == "small_table":
        return _wrap(rng, _scan_seq(rng, rng.uniform(10, 3000), 0.2), ("none", "Limit"))
    if variant == "small_nl":
        a = _scan_index(rng, rng.uniform(1, 20))
        b = _scan_index(rng, 1)
        total = a["Total Cost"] + b["Total Cost"] * a["Plan Rows"]
        return _node("Nested Loop", 0.7, total, a["Plan Rows"],
                     **{"Join Type": "Inner", "Plans": [_outer(a), {**b, "Parent Relationship": "Inner"}]})
    if variant == "limit_index_order":   # UNSEEN variant
        idx = _scan_index(rng, 10 ** rng.uniform(4, 6))
        return _node("Limit", 0.43, rng.uniform(2, 60), rng.choice([10, 20, 50]), Plans=[_outer(idx)])
    raise ValueError(variant)


GENERATORS: dict[int, tuple[Callable, list[str], list[str]]] = {
    0: (gen_seq_scan, ["plain", "bitmap_filter", "subplan"], ["seq_under_agg"]),
    1: (gen_join, ["nl_repeated_inner", "hash_big"], ["merge_big"]),
    2: (gen_sort_agg, ["sort", "hashagg"], ["sort_then_group"]),
    3: (gen_efficient, ["point_lookup", "small_table", "small_nl"], ["limit_index_order"]),
}
UNSEEN_VARIANTS = {c: v[2] for c, v in GENERATORS.items()}


def synth_dataset(n_per_class: int, seed: int, unseen: bool = False) -> list:
    rng = random.Random(seed)
    data = []
    for label, (fn, seen_variants, unseen_variants) in GENERATORS.items():
        variants = unseen_variants if unseen else seen_variants
        for i in range(n_per_class):
            plan = fn(rng, variants[i % len(variants)])
            if label != 3 and rng.random() < 0.6:
                plan = _distract(rng, plan)
            doc = {"Plan": plan}
            data.append(plan_to_graph(parse_explain(doc), label))
    rng.shuffle(data)
    return data


@dataclass
class TrainConfig:
    n_per_class: int = 400
    epochs: int = 60
    lr: float = 0.01
    weight_decay: float = 5e-4
    batch_size: int = 64
    seed: int = 1234


def score_split(model: PlanGCN, data: list) -> dict[str, Any]:
    """Accuracy + confusion matrix on a held-out split."""
    model.eval()
    correct, total = 0, 0
    confusion = [[0] * len(BOTTLENECK_CLASSES) for _ in BOTTLENECK_CLASSES]
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=128):
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            for p, y in zip(pred.tolist(), batch.y.tolist()):
                confusion[y][p] += 1
                correct += int(p == y)
                total += 1
    return {"accuracy": round(correct / max(total, 1), 4), "n": total, "confusion": confusion}


def train(cfg: TrainConfig, checkpoint_path: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    train_set = synth_dataset(cfg.n_per_class, cfg.seed)
    val_set = synth_dataset(max(cfg.n_per_class // 4, 20), cfg.seed + 1)
    holdout_set = synth_dataset(max(cfg.n_per_class // 4, 20), cfg.seed + 2)
    unseen_set = synth_dataset(max(cfg.n_per_class // 4, 20), cfg.seed + 3, unseen=True)
    model = PlanGCN()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val, history = None, -1.0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            opt.step()
            total_loss += float(loss) * batch.num_graphs
        val = score_split(model, val_set)["accuracy"]
        history.append({"epoch": epoch, "loss": round(total_loss / len(train_set), 4), "val_accuracy": val})
        if val > best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            log(f"[gnn] epoch {epoch:3d} loss={history[-1]['loss']:.4f} val_acc={val:.3f}")
    model.load_state_dict(best_state)
    metrics = {
        "train_size": len(train_set), "val_size": len(val_set),
        "best_val_accuracy": best_val,
        "heldout_same_distribution": score_split(model, holdout_set),
        "heldout_unseen_template_variants": score_split(model, unseen_set),
        "unseen_variants": UNSEEN_VARIANTS,
        "class_balance_train": dict(Counter(int(d.y) for d in train_set)),
        "epochs": cfg.epochs, "lr": cfg.lr, "seed": cfg.seed, "train_seconds": round(time.time() - t0, 1),
        "history_tail": history[-5:],
        "label_source": "synthetic scenario generator (see app/gnn/training.py)",
    }
    if checkpoint_path:
        save_checkpoint(model, checkpoint_path, {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                 "metrics": {k: v for k, v in metrics.items() if k != "history_tail"}})
    return metrics
