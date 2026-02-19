from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import time
import torch
import torch.nn.functional as F


@dataclass
class ConceptNode:
    node_id: int
    stage: int
    label: int
    task_name: str
    centroid_fp32: torch.Tensor
    centroid_q: torch.Tensor
    scale: float
    created_at: float
    quality: float

    def dequantized(self) -> torch.Tensor:
        return self.centroid_q.float() * self.scale

    def to_export(self) -> dict:
        payload = asdict(self)
        payload["centroid_fp32"] = self.centroid_fp32.tolist()
        payload["centroid_q"] = self.centroid_q.tolist()
        return payload


@dataclass
class Pathway:
    pathway_id: int
    task_name: str
    node_ids: List[int]


class MemoryBank:
    def __init__(self):
        self.nodes: Dict[int, ConceptNode] = {}
        self.pathways: Dict[int, Pathway] = {}
        self._next_node_id = 0
        self._next_pathway_id = 0

    @staticmethod
    def quantize_centroid(centroid: torch.Tensor) -> tuple[torch.Tensor, float]:
        max_abs = centroid.abs().max().item()
        if max_abs == 0.0:
            return torch.zeros_like(centroid, dtype=torch.int8), 1.0
        scale = max_abs / 127.0
        q = torch.round((centroid / scale).clamp(-127, 127)).to(torch.int8)
        return q, scale

    def add_node(
        self,
        stage: int,
        label: int,
        task_name: str,
        centroid: torch.Tensor,
        quality: float,
    ) -> int:
        centroid = F.normalize(centroid.float(), dim=0)
        centroid_q, scale = self.quantize_centroid(centroid)
        node = ConceptNode(
            node_id=self._next_node_id,
            stage=stage,
            label=label,
            task_name=task_name,
            centroid_fp32=centroid.cpu(),
            centroid_q=centroid_q.cpu(),
            scale=float(scale),
            created_at=time.time(),
            quality=float(quality),
        )
        self.nodes[node.node_id] = node
        self._next_node_id += 1
        return node.node_id

    def add_pathway(self, task_name: str, node_ids: List[int]) -> int:
        pathway = Pathway(
            pathway_id=self._next_pathway_id,
            task_name=task_name,
            node_ids=sorted(node_ids, key=lambda nid: self.nodes[nid].stage),
        )
        self.pathways[pathway.pathway_id] = pathway
        self._next_pathway_id += 1
        return pathway.pathway_id

    def find_reusable_node(self, stage: int, centroid: torch.Tensor, threshold: float) -> int | None:
        centroid = F.normalize(centroid.float(), dim=0).cpu()
        best_id = None
        best_sim = -1.0
        for node_id, node in self.nodes.items():
            if node.stage != stage:
                continue
            sim = F.cosine_similarity(node.centroid_fp32.unsqueeze(0), centroid.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_id = node_id
        if best_id is not None and best_sim >= threshold:
            return best_id
        return None

    def to_export(self) -> dict:
        return {
            "nodes": [self.nodes[nid].to_export() for nid in sorted(self.nodes.keys())],
            "pathways": [
                {
                    "pathway_id": p.pathway_id,
                    "task_name": p.task_name,
                    "node_ids": list(p.node_ids),
                }
                for _, p in sorted(self.pathways.items(), key=lambda kv: kv[0])
            ],
            "next_node_id": self._next_node_id,
            "next_pathway_id": self._next_pathway_id,
        }

    @classmethod
    def from_export(cls, payload: dict) -> "MemoryBank":
        mb = cls()
        for node_payload in payload.get("nodes", []):
            node = ConceptNode(
                node_id=int(node_payload["node_id"]),
                stage=int(node_payload["stage"]),
                label=int(node_payload["label"]),
                task_name=str(node_payload["task_name"]),
                centroid_fp32=torch.tensor(node_payload["centroid_fp32"], dtype=torch.float32),
                centroid_q=torch.tensor(node_payload["centroid_q"], dtype=torch.int8),
                scale=float(node_payload["scale"]),
                created_at=float(node_payload["created_at"]),
                quality=float(node_payload.get("quality", 1.0)),
            )
            mb.nodes[node.node_id] = node

        for p_payload in payload.get("pathways", []):
            pathway = Pathway(
                pathway_id=int(p_payload["pathway_id"]),
                task_name=str(p_payload["task_name"]),
                node_ids=[int(nid) for nid in p_payload["node_ids"]],
            )
            mb.pathways[pathway.pathway_id] = pathway

        mb._next_node_id = int(payload.get("next_node_id", len(mb.nodes)))
        mb._next_pathway_id = int(payload.get("next_pathway_id", len(mb.pathways)))
        return mb
