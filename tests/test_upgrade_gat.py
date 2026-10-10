"""P8 major tests: GATv2 attention, edge features, checkpoint compatibility, fallback, flag."""
import os

import pytest
import torch

from app.config import get_settings, reset_settings_cache
from app.gnn.analyzer import GNNAnalyzer
from app.gnn.gat_model import (EDGE_DIM, PlanGATv2, extract_attention, load_gat_checkpoint, plan_to_gat_graph,
                               validate_edge_features)
from app.errors import QuerySentinelError
from app.plan_parser import parse_explain
from tests.conftest import PLANS


@pytest.fixture
def gat_on():
    os.environ["QS_FEATURE_GATV2"] = "true"
    reset_settings_cache()
    yield
    os.environ.pop("QS_FEATURE_GATV2")
    reset_settings_cache()


def test_forward_attention_shapes_and_edge_mapping():
    plan = parse_explain(PLANS["scenario_0"])
    data = plan_to_gat_graph(plan)
    assert data.edge_attr.shape == (data.edge_index.size(1), EDGE_DIM)
    out, att = PlanGATv2()(data.x, data.edge_index, data.edge_attr, return_attention=True)
    assert out.shape == (1, 4) and len(att) == 2
    for ei, a in att:
        assert ei.size(1) == data.edge_index.size(1) and a.shape == (data.edge_index.size(1), 2)  # no self loops
    res = extract_attention(plan, data, att)
    assert len(res["edge_attention"]) == data.edge_index.size(1)
    assert {"source", "target", "layer", "head", "weight"} <= set(res["raw_attention"][0])
    assert "not causal proof" in res["wording"]
    bad = data.clone()
    bad.edge_attr = bad.edge_attr[:-1]
    with pytest.raises(ValueError):
        validate_edge_features(bad)


def test_incompatible_checkpoint_refused_and_fallback(gat_on, tmp_path):
    gcn = get_settings().resolve_path(get_settings().gnn_checkpoint_path)
    with pytest.raises(QuerySentinelError):
        load_gat_checkpoint(gcn)                          # old GCN checkpoint never loaded into GATv2
    an = GNNAnalyzer()
    an._load_gat(str(gcn))
    d = an._analyze_gat(parse_explain(PLANS["scenario_0"]))
    assert d.attention["model_mode"] == "analytic_fallback" and "ANALYTIC FALLBACK" in d.attention["provenance"]
    assert "incompatible" in d.attention["reason"]


def test_trained_gat_mode_and_old_gcn_flag_path(gat_on):
    an = GNNAnalyzer()
    d = an.analyze(parse_explain(PLANS["scenario_0"]))
    assert d.model_mode == "trained" and d.attention["provenance"] == ["TRAINED GNN"]
    assert d.attention["architecture"] == "PlanGATv2" and d.attention["training_dataset"]
    assert an.status()["gatv2"]["mode"] == "trained"
    os.environ["QS_FEATURE_GATV2"] = "false"
    reset_settings_cache()
    d2 = GNNAnalyzer().analyze(parse_explain(PLANS["scenario_0"]))
    assert d2.attention is None and d2.model_mode == "trained"       # old GCN path unchanged
