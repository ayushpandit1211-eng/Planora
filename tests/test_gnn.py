"""GNN (Sections 19-20/60): graph nodes/edges, tensor shapes, forward pass, fallback mode,
checkpoint validation."""
import torch

from app.contracts import BOTTLENECK_CLASSES
from app.gnn import fallback_analytics as fb
from app.gnn.analyzer import GNNAnalyzer
from app.gnn.mapper import NUM_FEATURES, OPERATOR_MAP, graph_json, operator_id, plan_to_graph
from app.gnn.model import PlanGCN, load_checkpoint, save_checkpoint
from app.gnn.training import TrainConfig, synth_dataset, train
from app.plan_parser import parse_explain

from tests.conftest import PLANS


def test_operator_mapping_spec():
    assert OPERATOR_MAP["Seq Scan"] == 0 and OPERATOR_MAP["Materialize"] == 11
    assert operator_id("Gather") == 12


def test_graph_nodes_edges_shapes():
    pa = parse_explain(PLANS["scenario_2"])
    d = plan_to_graph(pa)
    assert d.x.shape == (pa.node_count, NUM_FEATURES)
    assert d.edge_index.shape == (2, 2 * len(pa.edges))          # both directions
    assert torch.isfinite(d.x).all()
    nodes, edges = graph_json(pa)
    assert len(nodes) == pa.node_count and len(edges) == len(pa.edges)


def test_model_forward_pass():
    m = PlanGCN()
    d = plan_to_graph(parse_explain(PLANS["scenario_0"]))
    out = m(d.x, d.edge_index)
    assert out.shape == (1, 4)


def test_fallback_mode_when_no_checkpoint(tmp_path):
    a = GNNAnalyzer(str(tmp_path / "missing.pt"))
    assert a.mode == "analytic_fallback"
    diag = a.analyze(parse_explain(PLANS["scenario_2"]))
    assert diag.model_mode == "analytic_fallback"
    assert isinstance(diag.confidence, str) and diag.confidence.startswith("rule:")
    assert diag.class_probabilities is None
    assert diag.primary_bottleneck_type == BOTTLENECK_CLASSES[0]


def test_corrupted_checkpoint_falls_back(tmp_path):
    p = tmp_path / "bad.pt"
    p.write_bytes(b"not a checkpoint")
    a = GNNAnalyzer(str(p))
    assert a.mode == "analytic_fallback" and "corrupted" in a.load_error


def test_fallback_rules_on_real_plans():
    assert fb.classify(parse_explain(PLANS["scenario_4"])).class_id == 3       # point lookup efficient
    assert fb.classify(parse_explain(PLANS["scenario_0"])).class_id == 0       # seq scan dominates
    assert fb.classify(parse_explain(PLANS["scenario_3"])).class_id == 0


def test_training_save_load_roundtrip(tmp_path):
    metrics = train(TrainConfig(n_per_class=20, epochs=3, seed=1), str(tmp_path / "g.pt"), log=lambda s: None)
    assert "heldout_unseen_template_variants" in metrics
    model, meta = load_checkpoint(tmp_path / "g.pt")
    assert "metrics" in meta
    a = GNNAnalyzer(str(tmp_path / "g.pt"))
    diag = a.analyze(parse_explain(PLANS["scenario_1"]))
    assert diag.model_mode == "trained" and 0 <= diag.confidence <= 1
    assert abs(sum(diag.class_probabilities.values()) - 1) < 1e-3


def test_synthetic_dataset_balanced():
    ds = synth_dataset(10, seed=3)
    assert len(ds) == 40 and {int(d.y) for d in ds} == {0, 1, 2, 3}


def test_shipped_checkpoint_valid():
    from app.config import get_settings
    s = get_settings()
    model, meta = load_checkpoint(s.resolve_path(s.gnn_checkpoint_path))
    assert meta["metrics"]["heldout_same_distribution"]["accuracy"] > 0.5
