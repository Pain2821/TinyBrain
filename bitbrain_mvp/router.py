from __future__ import annotations

from typing import Dict, Iterable, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F

from .memory_bank import MemoryBank, Pathway


class SimilarityRouter:
    def __init__(
        self,
        route_threshold: float = 0.60,
        high_conf: float = 0.85,
        low_conf: float = 0.60,
        conf_gap: float = 0.15,
    ):
        self.route_threshold = route_threshold
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.conf_gap = conf_gap

    def pathway_score(
        self,
        memory: MemoryBank,
        pathway: Pathway,
        stage_acts: Dict[int, torch.Tensor],
        sample_idx: int,
    ) -> float:
        sims = []
        for node_id in pathway.node_ids:
            node = memory.nodes[node_id]
            if node.stage not in stage_acts:
                continue
            act = stage_acts[node.stage][sample_idx]
            act = F.normalize(act, dim=0)
            sim = F.cosine_similarity(node.centroid_fp32.unsqueeze(0), act.unsqueeze(0)).item()
            sims.append(sim)
        if not sims:
            return -1.0
        return float(sum(sims) / len(sims))

    def pathway_pred_label(
        self,
        memory: MemoryBank,
        pathway: Pathway,
        stage_acts: Dict[int, torch.Tensor],
        sample_idx: int,
    ) -> Tuple[int, float]:
        per_label = defaultdict(list)
        for node_id in pathway.node_ids:
            node = memory.nodes[node_id]
            if node.stage not in stage_acts:
                continue
            act = stage_acts[node.stage][sample_idx]
            act = F.normalize(act, dim=0)
            sim = F.cosine_similarity(node.centroid_fp32.unsqueeze(0), act.unsqueeze(0)).item()
            per_label[node.label].append(sim)
        if not per_label:
            return 0, -1.0
        best_label, best_score = -1, -1.0
        for label, sims in per_label.items():
            score = float(sum(sims) / len(sims))
            if score > best_score:
                best_score = score
                best_label = label
        return int(best_label), best_score

    def route(
        self,
        memory: MemoryBank,
        stage_acts: Dict[int, torch.Tensor],
        fast_probs: torch.Tensor,
        sample_idx: int,
        allowed_pathways: Iterable[int] | None = None,
        force_pathway: bool = False,
    ) -> tuple[int, dict]:
        pathway_ids = list(memory.pathways.keys()) if allowed_pathways is None else list(allowed_pathways)
        if len(pathway_ids) == 0:
            pred = int(fast_probs[sample_idx].argmax().item())
            return pred, {"source": "fast_only", "route_conf": -1.0}

        best_pid, best_score = -1, -1.0
        for pid in pathway_ids:
            score = self.pathway_score(memory, memory.pathways[pid], stage_acts, sample_idx)
            if score > best_score:
                best_score = score
                best_pid = pid

        fast_conf, fast_pred = torch.max(fast_probs[sample_idx], dim=0)
        pathway_pred, pathway_conf = self.pathway_pred_label(memory, memory.pathways[best_pid], stage_acts, sample_idx)

        if force_pathway:
            return pathway_pred, {"source": "pathway_forced", "route_conf": pathway_conf, "pathway_id": best_pid}

        if best_score < self.route_threshold:
            return int(fast_pred.item()), {"source": "fast_threshold", "route_conf": pathway_conf, "pathway_id": best_pid}

        if fast_conf.item() > self.high_conf and pathway_conf < self.low_conf:
            return int(fast_pred.item()), {"source": "fast_confident", "route_conf": pathway_conf, "pathway_id": best_pid}
        if pathway_conf > self.high_conf:
            return pathway_pred, {"source": "pathway_confident", "route_conf": pathway_conf, "pathway_id": best_pid}
        if pathway_conf - fast_conf.item() > self.conf_gap:
            return pathway_pred, {"source": "pathway_gap", "route_conf": pathway_conf, "pathway_id": best_pid}
        return int(fast_pred.item()), {"source": "fast_default", "route_conf": pathway_conf, "pathway_id": best_pid}

