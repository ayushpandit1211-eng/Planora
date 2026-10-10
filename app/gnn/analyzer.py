"""GNN inference with explicit model-mode transparency (Sections 20 and 69).

MODE 1 trained: valid checkpoint -> GCN inference; confidence = softmax probability.
MODE 2 analytic_fallback: checkpoint missing/invalid -> deterministic rules; confidence is a
rule label. Untrained/random weights are never presented as a trained model.
"""
from __future__ import annotations

from typing import Any

import torch

from app.config import get_settings
from app.contracts import BOTTLENECK_CLASSES, GNNDiagnosis, PlanAnalysis
from app.errors import QuerySentinelError
from app.gnn import fallback_analytics as fb
from app.gnn.mapper import graph_json, plan_to_graph
from app.gnn.model import PlanGCN, load_checkpoint


class GNNAnalyzer:
    def __init__(self, checkpoint_path: str | None = None):
        path = checkpoint_path or str(get_settings().resolve_path(get_settings().gnn_checkpoint_path))
        self.checkpoint_path = path
        self.model: PlanGCN | None = None
        self.meta: dict[str, Any] = {}
        self.load_error: str | None = None
        try:
            self.model, self.meta = load_checkpoint(path)
        except QuerySentinelError as exc:
            self.load_error = f"{exc.category.value}: {exc.message}"
        # Final Upgrade P8: GATv2 path, loaded lazily and only used when QS_FEATURE_GATV2 is on
        self.gat_model = None
        self.gat_meta: dict[str, Any] = {}
        self.gat_error: str | None = None
        self._gat_loaded = False

    def _load_gat(self, path: str | None = None) -> None:
        from app.gnn.gat_model import load_gat_checkpoint
        self._gat_loaded = True
        p = path or str(get_settings().resolve_path(get_settings().gnn_gat_checkpoint_path))
        try:
            self.gat_model, self.gat_meta = load_gat_checkpoint(p)
            self.gat_error = None
        except QuerySentinelError as exc:
            self.gat_model, self.gat_error = None, f"{exc.category.value}: {exc.message}"

    @property
    def mode(self) -> str:
        return "trained" if self.model is not None else "analytic_fallback"

    def status(self) -> dict[str, Any]:
        m = self.meta.get("metrics", {}) if self.meta else {}
        from app import feature_flags
        gat = None
        if feature_flags.enabled("gatv2"):
            if not self._gat_loaded:
                self._load_gat()
            gm = self.gat_meta.get("metrics", {})
            gat = {"mode": "trained" if self.gat_model is not None else "analytic_fallback",
                   "error": self.gat_error, "architecture": self.gat_meta.get("architecture"),
                   "feature_schema_version": self.gat_meta.get("feature_schema_version"),
                   "checkpoint_version": self.gat_meta.get("checkpoint_version"),
                   "training_dataset": self.gat_meta.get("training_dataset"),
                   "held_out_accuracy": (gm.get("heldout_same_distribution") or {}).get("accuracy"),
                   "unseen_variant_accuracy": (gm.get("heldout_unseen_template_variants") or {}).get("accuracy")}
        return {"gatv2": gat,"mode": self.mode, "checkpoint": self.checkpoint_path, "error": self.load_error,
                "trained_at": self.meta.get("trained_at"),
                "held_out_accuracy": (m.get("heldout_same_distribution") or {}).get("accuracy"),
                "unseen_variant_accuracy": (m.get("heldout_unseen_template_variants") or {}).get("accuracy")}

    def analyze(self, plan: PlanAnalysis) -> GNNDiagnosis:
        """`plan` must already be sanitised (pseudonymised) - output may reach the LLM/dashboard."""
        from app import feature_flags
        if feature_flags.enabled("gatv2"):
            return self._analyze_gat(plan)
        return self._analyze_gcn(plan)

    def _analyze_gat(self, plan: PlanAnalysis) -> GNNDiagnosis:
        from app.gnn.gat_model import WORDING, extract_attention, plan_to_gat_graph
        if not self._gat_loaded:
            self._load_gat()
        if self.gat_model is None:
            diag = self._analyze_gcn(plan)
            total = max(sum(max(n.exclusive_cost, 0.0) for n in plan.nodes), 1e-9)
            diag.attention = {"model_mode": "analytic_fallback", "provenance": ["ANALYTIC FALLBACK"],
                              "reason": self.gat_error or "no compatible GATv2 checkpoint",
                              "node_importance": [{"id": n.node_id, "operator": n.node_type,
                                                   "attention_importance": None,
                                                   "planner_cost_share": round(max(n.exclusive_cost, 0) / total, 4)}
                                                  for n in plan.nodes],
                              "edge_attention": [], "wording": "no attention available (fallback mode)"}
            return diag
        try:
            data = plan_to_gat_graph(plan)
            with torch.no_grad():
                logits, att = self.gat_model(data.x, data.edge_index, data.edge_attr, return_attention=True)
                probs = torch.softmax(logits, dim=1)[0]
            attn = extract_attention(plan, data, att)
        except (ValueError, RuntimeError) as exc:
            diag = self._analyze_gcn(plan)
            diag.attention = {"model_mode": "analytic_fallback", "provenance": ["ANALYTIC FALLBACK"],
                              "reason": f"GATv2 inference failed ({type(exc).__name__})"}
            return diag
        rule = fb.classify(plan)
        cls = int(probs.argmax())
        nodes, edges = graph_json(plan)
        agree = cls == rule.class_id
        evidence = list(rule.evidence) + [f"analytic rule {rule.rule} -> {BOTTLENECK_CLASSES[rule.class_id]} "
                                          f"({'agrees' if agree else 'DISAGREES'} with GATv2)"]
        bn = attn.get("bottleneck_node") or {}
        summary = fb.summarize(fb.RuleResult(cls, rule.rule, rule.confidence_label, rule.focus_node, rule.evidence),
                               plan, f"GATv2 trained mode, p={float(probs[cls]):.2f}")
        if bn:
            summary += (f" The {WORDING} is highest at node {bn['id']} ({bn['operator']}).")
        if not agree:
            summary += (f" Analytic rules suggest '{BOTTLENECK_CLASSES[rule.class_id]}' instead; treat the "
                        "classification as uncertain.")
        attn.update({"model_mode": "trained", "provenance": ["TRAINED GNN"],
                     "architecture": self.gat_meta.get("architecture"),
                     "feature_schema_version": self.gat_meta.get("feature_schema_version"),
                     "checkpoint_version": self.gat_meta.get("checkpoint_version"),
                     "training_dataset": self.gat_meta.get("training_dataset")})
        attn.pop("raw_attention", None) if len(attn.get("raw_attention", [])) > 400 else None
        return GNNDiagnosis(primary_bottleneck_type=BOTTLENECK_CLASSES[cls], bottleneck_class_id=cls,
                            confidence=round(float(probs[cls]), 4),
                            class_probabilities={BOTTLENECK_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)},
                            highest_cost_node=plan.highest_cost_node or {}, natural_language_summary=summary,
                            graph_nodes=nodes, graph_edges=edges, model_mode="trained", evidence=evidence,
                            attention=attn)

    def _analyze_gcn(self, plan: PlanAnalysis) -> GNNDiagnosis:
        rule = fb.classify(plan)
        nodes, edges = graph_json(plan)
        hc = plan.highest_cost_node or {}
        evidence = list(rule.evidence)
        if self.model is not None:
            try:
                data = plan_to_graph(plan)
                with torch.no_grad():
                    probs = torch.softmax(self.model(data.x, data.edge_index), dim=1)[0]
                cls = int(probs.argmax())
                prob_map = {BOTTLENECK_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}
                agree = cls == rule.class_id
                evidence.append(f"analytic rule {rule.rule} -> {BOTTLENECK_CLASSES[rule.class_id]} "
                                f"({'agrees' if agree else 'DISAGREES'} with GNN)")
                note = f"GNN trained mode, p={float(probs[cls]):.2f}"
                summary = fb.summarize(fb.RuleResult(cls, rule.rule, rule.confidence_label, rule.focus_node,
                                                     rule.evidence), plan, note)
                if not agree:
                    summary += (f" Analytic rules suggest '{BOTTLENECK_CLASSES[rule.class_id]}' instead; "
                                "treat the classification as uncertain.")
                return GNNDiagnosis(primary_bottleneck_type=BOTTLENECK_CLASSES[cls], bottleneck_class_id=cls,
                                    confidence=round(float(probs[cls]), 4), class_probabilities=prob_map,
                                    highest_cost_node=hc, natural_language_summary=summary,
                                    graph_nodes=nodes, graph_edges=edges, model_mode="trained",
                                    evidence=evidence)
            except (ValueError, RuntimeError) as exc:
                reason = f"GNN inference failed ({type(exc).__name__}); analytic fallback used"
            else:  # pragma: no cover
                reason = None
        else:
            reason = self.load_error or "no checkpoint"
        return GNNDiagnosis(
            primary_bottleneck_type=BOTTLENECK_CLASSES[rule.class_id], bottleneck_class_id=rule.class_id,
            confidence=f"rule:{rule.confidence_label}", highest_cost_node=hc,
            natural_language_summary=fb.summarize(rule, plan, "analytic fallback - deterministic rules, not a neural network"),
            graph_nodes=nodes, graph_edges=edges, model_mode="analytic_fallback", evidence=evidence,
            fallback_reason=reason)
