from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn.functional as F

from .config import Config
from .memory_bank import MemoryBank, Pathway


@dataclass
class RouteInfo:
    source: str
    fast_confidence: float
    pathway_confidence: float
    pathway_id: Optional[int]
    pathway_score: float
    pathway_pred: Optional[int]


class SimilarityRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _score_pathway(
        self,
        memory: MemoryBank,
        pathway: Pathway,
        activations: Dict[int, torch.Tensor],
        sample_idx: int,
    ) -> float:
        sims: List[float] = []
        for node_id in pathway.node_ids:
            node = memory.nodes[node_id]
            if node.stage_idx not in activations:
                continue
            sample_act = F.normalize(activations[node.stage_idx][sample_idx], dim=0)
            centroid = F.normalize(node.get_centroid(use_quantized=memory.quantize), dim=0)
            sim = F.cosine_similarity(sample_act.unsqueeze(0), centroid.unsqueeze(0)).item()
            sims.append(sim)
        if not sims:
            return -1.0
        return float(sum(sims) / len(sims))

    def best_pathway(
        self,
        memory: MemoryBank,
        activations: Dict[int, torch.Tensor],
        sample_idx: int,
        allowed_pathways: Optional[Iterable[int]] = None,
    ) -> Tuple[Optional[int], float]:
        pathway_ids = list(memory.pathways.keys()) if allowed_pathways is None else list(allowed_pathways)
        if not pathway_ids:
            return None, -1.0
        best_id = None
        best_score = -1.0
        for pid in pathway_ids:
            score = self._score_pathway(memory, memory.pathways[pid], activations, sample_idx)
            if score > best_score:
                best_score = score
                best_id = pid
        if best_score < self.cfg.tau_route:
            return None, best_score
        return best_id, best_score

    def pathway_pred_label(
        self,
        memory: MemoryBank,
        pathway: Pathway,
        activations: Dict[int, torch.Tensor],
        sample_idx: int,
    ) -> Tuple[int, float]:
        per_label: Dict[int, List[float]] = defaultdict(list)
        for node_id in pathway.node_ids:
            node = memory.nodes[node_id]
            if node.stage_idx not in activations:
                continue
            sample_act = F.normalize(activations[node.stage_idx][sample_idx], dim=0)
            centroid = F.normalize(node.get_centroid(use_quantized=memory.quantize), dim=0)
            sim = F.cosine_similarity(sample_act.unsqueeze(0), centroid.unsqueeze(0)).item()
            per_label[node.class_label].append(sim)

        if not per_label:
            return 0, -1.0

        best_label = -1
        best_score = -1.0
        for label, sims in per_label.items():
            score = float(sum(sims) / len(sims))
            if score > best_score:
                best_score = score
                best_label = label
        return int(best_label), best_score

    def decide(
        self,
        fast_probs: torch.Tensor,
        pathway_pred: Optional[int],
        pathway_confidence: float,
        sample_idx: int,
    ) -> Tuple[int, RouteInfo]:
        fast_conf, fast_pred = torch.max(fast_probs[sample_idx], dim=0)
        fast_conf_value = float(fast_conf.item())
        fast_pred_value = int(fast_pred.item())

        if pathway_pred is None:
            return fast_pred_value, RouteInfo(
                source="fast_no_pathway",
                fast_confidence=fast_conf_value,
                pathway_confidence=pathway_confidence,
                pathway_id=None,
                pathway_score=pathway_confidence,
                pathway_pred=None,
            )

        rho_f = fast_conf_value
        rho_p = pathway_confidence

        if not self.cfg.use_confidence_logic:
            return int(pathway_pred), RouteInfo(
                source="pathway_threshold_only",
                fast_confidence=rho_f,
                pathway_confidence=rho_p,
                pathway_id=None,
                pathway_score=pathway_confidence,
                pathway_pred=int(pathway_pred),
            )

        if rho_f > self.cfg.gamma_high and rho_p < self.cfg.gamma_low:
            return fast_pred_value, RouteInfo(
                source="fast_high_conf",
                fast_confidence=rho_f,
                pathway_confidence=rho_p,
                pathway_id=None,
                pathway_score=pathway_confidence,
                pathway_pred=int(pathway_pred),
            )
        if rho_p > self.cfg.gamma_high:
            return int(pathway_pred), RouteInfo(
                source="pathway_high_conf",
                fast_confidence=rho_f,
                pathway_confidence=rho_p,
                pathway_id=None,
                pathway_score=pathway_confidence,
                pathway_pred=int(pathway_pred),
            )
        if rho_p - rho_f > self.cfg.delta_conf:
            return int(pathway_pred), RouteInfo(
                source="pathway_conf_gap",
                fast_confidence=rho_f,
                pathway_confidence=rho_p,
                pathway_id=None,
                pathway_score=pathway_confidence,
                pathway_pred=int(pathway_pred),
            )
        return fast_pred_value, RouteInfo(
            source="fast_default",
            fast_confidence=rho_f,
            pathway_confidence=rho_p,
            pathway_id=None,
            pathway_score=pathway_confidence,
            pathway_pred=int(pathway_pred),
        )
