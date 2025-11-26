# core/memory_bank.py
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.utils_logging import log, vlog

class ConceptNode:
    def __init__(self, node_id: str, centroid: np.ndarray, int8_blob_path: str,
                 stage_idx: int, class_label: str, class_idx: int, tag: str = None, meta: dict = None): # type: ignore
        self.id = node_id
        centroid = centroid.astype(np.float32)
        norm = np.linalg.norm(centroid) + 1e-9
        self.centroid = (centroid / norm).astype(np.float32)
        self.int8_blob_path = int8_blob_path
        self.stage_idx = stage_idx
        self.class_label = class_label
        self.class_idx = class_idx
        self.tag = tag or node_id
        self.meta = meta or {}
        self.frozen = True

    def __repr__(self):
        return f"Node({self.id}, stage={self.stage_idx}, class={self.class_label})"

class MemoryBank:
    def __init__(self, root_dir: str = "memory_bank"):
        self.root = Path(root_dir)
        self.nodes_dir = self.root / "nodes"
        self.pathways_fp = self.root / "pathways.json"
        self.meta_fp = self.root / "meta.json"

        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        if not self.pathways_fp.exists():
            self.pathways_fp.write_text(json.dumps({}))
        if not self.meta_fp.exists():
            self.meta_fp.write_text(json.dumps({"version": "1.0", "created": time.time()}))

        self.nodes: Dict[str, ConceptNode] = {}
        self.pathways: Dict[str, List[str]] = json.loads(self.pathways_fp.read_text())
        self._load_nodes()
        log(f"Memory Bank initialized: {len(self.nodes)} nodes, {len(self.pathways)} pathways")

    def _load_nodes(self):
        for meta_file in self.nodes_dir.glob("*.json"):
            try:
                j = json.loads(meta_file.read_text())
                nid = j["id"]
                centroid = np.array(j["centroid"], dtype=np.float32)
                blob = str(self.nodes_dir / j["blob"])
                stage = j.get("stage_idx", -1)
                cls_label = j.get("class_label", "unknown")
                cls_idx = j.get("class_idx", -1)
                tag = j.get("tag", nid)
                meta = j.get("meta", {})
                node = ConceptNode(nid, centroid, blob, stage, cls_label, cls_idx, tag, meta)
                self.nodes[nid] = node
            except Exception as e:
                vlog(True, f"Warning: Failed to load node {meta_file}: {e}")

    def add_node(self, node: ConceptNode):
        self.nodes[node.id] = node
        meta_fp = self.nodes_dir / f"{node.id}.json"
        meta = {
            "id": node.id,
            "centroid": node.centroid.tolist(),
            "blob": os.path.basename(node.int8_blob_path),
            "stage_idx": node.stage_idx,
            "class_label": node.class_label,
            "class_idx": node.class_idx,
            "tag": node.tag,
            "meta": node.meta
        }
        meta_fp.write_text(json.dumps(meta, indent=2))

    def atomic_update(self, new_nodes: List[ConceptNode], new_pathways: Dict[str, List[str]]):
        for node in new_nodes:
            dest = self.nodes_dir / os.path.basename(node.int8_blob_path)
            shutil.copy(node.int8_blob_path, dest)
            node.int8_blob_path = str(dest)
            self.add_node(node)
        self.pathways.update(new_pathways)
        self.pathways_fp.write_text(json.dumps(self.pathways, indent=2))
        log(f"✅ Memory bank updated: +{len(new_nodes)} nodes, +{len(new_pathways)} pathways")

    def find_similar_node(self, centroid: np.ndarray, stage_idx: int, class_idx: Optional[int] = None) -> Tuple[Optional[ConceptNode], float]:
        if len(self.nodes) == 0:
            return None, 0.0
        c = centroid.astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-9)
        best_node = None
        best_sim = -1.0
        for nid, node in self.nodes.items():
            if node.stage_idx != stage_idx:
                continue
            if class_idx is not None and node.class_idx != class_idx:
                continue
            sim = float(np.dot(c, node.centroid))
            if sim > best_sim:
                best_sim = sim
                best_node = node
        return best_node, best_sim

    def get_pathway_info(self, pathway_tag: str) -> dict:
        if pathway_tag not in self.pathways:
            return {}
        node_ids = self.pathways[pathway_tag]
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        if not nodes:
            return {"exists": False}
        class_labels = [n.class_label for n in nodes]
        majority_class = max(set(class_labels), key=class_labels.count)
        return {
            "exists": True,
            "num_nodes": len(nodes),
            "majority_class": majority_class,
            "stages": [n.stage_idx for n in nodes],
            "node_ids": node_ids
        }
