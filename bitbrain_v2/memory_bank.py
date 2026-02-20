from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.nn.functional as F


@dataclass
class ConceptNode:
    node_id: int
    stage_idx: int
    class_label: int
    task_name: str
    centroid: torch.Tensor
    timestamp: float
    quality_score: float
    centroid_int8: Optional[torch.Tensor] = None
    scale: Optional[float] = None

    def get_centroid(self, use_quantized: bool = True) -> torch.Tensor:
        if use_quantized and self.centroid_int8 is not None and self.scale is not None:
            return self.centroid_int8.float() * float(self.scale)
        return self.centroid

    def storage_bytes(self) -> int:
        fp32_bytes = self.centroid.numel() * 4
        q_bytes = 0
        if self.centroid_int8 is not None:
            q_bytes += self.centroid_int8.numel()
            q_bytes += 4  # scale
        return fp32_bytes + q_bytes


@dataclass
class Pathway:
    pathway_id: int
    task_name: str
    class_label: int
    node_ids: List[int]


class MemoryBank:
    def __init__(self, quantize: bool = True):
        self.quantize = quantize
        self.nodes: Dict[int, ConceptNode] = {}
        self.pathways: Dict[int, Pathway] = {}
        self._next_node_id = 0
        self._next_pathway_id = 0

    @staticmethod
    def _normalize(vec: torch.Tensor) -> torch.Tensor:
        return F.normalize(vec.float(), dim=0).cpu()

    @staticmethod
    def quantize_centroid(centroid: torch.Tensor) -> Tuple[torch.Tensor, float]:
        max_abs = centroid.abs().max().item()
        if max_abs == 0.0:
            return torch.zeros_like(centroid, dtype=torch.int8), 1.0
        scale = max_abs / 127.0
        q = torch.round((centroid / scale).clamp(-127, 127)).to(torch.int8)
        return q, float(scale)

    def find_reusable_node(self, stage_idx: int, centroid: torch.Tensor, tau_reuse: float) -> Optional[int]:
        centroid = self._normalize(centroid)
        best_id = None
        best_sim = -1.0
        for node_id, node in self.nodes.items():
            if node.stage_idx != stage_idx:
                continue
            sim = F.cosine_similarity(node.centroid.unsqueeze(0), centroid.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_id = node_id
        if best_id is not None and best_sim >= tau_reuse:
            return best_id
        return None

    def add_node(
        self,
        stage_idx: int,
        class_label: int,
        task_name: str,
        centroid: torch.Tensor,
        quality_score: float,
    ) -> int:
        centroid = self._normalize(centroid)
        centroid_int8 = None
        scale = None
        if self.quantize:
            centroid_int8, scale = self.quantize_centroid(centroid)
        node = ConceptNode(
            node_id=self._next_node_id,
            stage_idx=int(stage_idx),
            class_label=int(class_label),
            task_name=task_name,
            centroid=centroid,
            timestamp=time.time(),
            quality_score=float(quality_score),
            centroid_int8=centroid_int8,
            scale=scale,
        )
        self.nodes[node.node_id] = node
        self._next_node_id += 1
        return node.node_id

    def add_pathway(self, task_name: str, class_label: int, node_ids: List[int]) -> int:
        ordered = sorted(node_ids, key=lambda nid: self.nodes[nid].stage_idx)
        pathway = Pathway(
            pathway_id=self._next_pathway_id,
            task_name=task_name,
            class_label=int(class_label),
            node_ids=ordered,
        )
        self.pathways[pathway.pathway_id] = pathway
        self._next_pathway_id += 1
        return pathway.pathway_id

    def memory_overhead_bytes(self) -> int:
        return sum(node.storage_bytes() for node in self.nodes.values())
