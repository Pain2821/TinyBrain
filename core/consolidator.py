# core/consolidator.py
"""
Consolidation engine: extracts multi-stage activations, clusters them,
creates/reuses nodes, writes int8 blobs and registers pathways.
"""
from typing import List, Tuple, Dict
import os
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core.config import (
    TORCH_DEVICE, MIN_EXAMPLES_PER_CLASS, REUSE_THRESHOLD,
    STAGING_DIR, CLUSTER_MIN_SIZE, MAX_STAGE_CLUSTERS
)
from core.memory_bank import MemoryBank, ConceptNode
from core.utils_quant import quantize_centroid_int8
from core.utils_logging import log, vlog

class Consolidator:
    def __init__(self, fast_learner, memory_bank: MemoryBank, router, min_examples_per_class: int = MIN_EXAMPLES_PER_CLASS, reuse_threshold: float = REUSE_THRESHOLD, staging_dir: str = STAGING_DIR): # type: ignore
        self.fast_learner = fast_learner
        self.mb = memory_bank
        self.router = router
        self.min_examples = min_examples_per_class
        self.reuse_threshold = reuse_threshold
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.class_buffers: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        vlog(True, f"Consolidator initialized: min_examples={self.min_examples}, reuse_threshold={self.reuse_threshold}")

    def add_to_buffer(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        for i in range(x_batch.shape[0]):
            class_idx = int(y_batch[i])
            self.class_buffers.setdefault(class_idx, []).append((x_batch[i].cpu(), y_batch[i].cpu()))
            if len(self.class_buffers[class_idx]) > 100:
                self.class_buffers[class_idx] = self.class_buffers[class_idx][-100:]

    def should_consolidate(self) -> bool:
        for examples in self.class_buffers.values():
            if len(examples) >= self.min_examples:
                return True
        return False

    def run_consolidation(self, class_names: List[str]):
        log("\n🌙 Sleep consolidation started...")
        total_new_nodes = 0
        total_new_pathways = 0
        for class_idx, examples in list(self.class_buffers.items()):
            if len(examples) < self.min_examples:
                continue
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            log(f"\n  📦 Processing class '{class_name}' ({len(examples)} examples)...")
            new_nodes, new_pathway = self._consolidate_class(class_idx, class_name, examples)
            if new_nodes and new_pathway:
                self.mb.atomic_update(new_nodes, {new_pathway["tag"]: new_pathway["node_ids"]})
                total_new_nodes += len(new_nodes)
                total_new_pathways += 1
                for node in new_nodes:
                    try:
                        os.remove(node.int8_blob_path)
                    except Exception:
                        pass
        self.class_buffers = {}
        log(f"\n✨ Consolidation complete: +{total_new_nodes} nodes, +{total_new_pathways} pathways")
        log(f"   Total: {len(self.mb.nodes)} nodes, {len(self.mb.pathways)} pathways\n")

    def _consolidate_class(self, class_idx: int, class_name: str, examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        xs = torch.stack([e[0] for e in examples])
        self.fast_learner.eval()
        stage_data = {}
        with torch.no_grad():
            for i in range(xs.shape[0]):
                x = xs[i:i+1].to(TORCH_DEVICE)
                stage_acts, _ = self.fast_learner.forward_with_cache(x)
                for stage_idx, act in stage_acts:
                    if stage_idx not in stage_data:
                        stage_data[stage_idx] = []
                    stage_data[stage_idx].append(act[0].cpu().numpy())

        pathway_node_ids = []
        new_nodes = []
        reuse_count = 0
        for stage_idx in sorted(stage_data.keys()):
            acts = np.stack(stage_data[stage_idx], axis=0)
            optimal_k = self._find_optimal_clusters(acts, max_k=MAX_STAGE_CLUSTERS)
            if optimal_k < 1:
                continue
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(acts)
            labels = kmeans.labels_
            for cluster_id in range(optimal_k):
                mask = labels == cluster_id
                cluster_size = int(mask.sum())
                if cluster_size < CLUSTER_MIN_SIZE:
                    continue
                centroid = kmeans.cluster_centers_[cluster_id]
                similar_node, similarity = self.mb.find_similar_node(centroid, stage_idx, class_idx=class_idx)
                if similar_node and similarity >= self.reuse_threshold:
                    pathway_node_ids.append(similar_node.id)
                    reuse_count += 1
                    log(f"    ♻️  Stage {stage_idx}: Reused {similar_node.id} (sim={similarity:.3f})")
                else:
                    node = self._create_node(centroid, stage_idx, class_name, class_idx, cluster_id)
                    new_nodes.append(node)
                    pathway_node_ids.append(node.id)
                    log(f"    ✨ Stage {stage_idx}: Created {node.id}")

        if not pathway_node_ids:
            return [], None
        pathway_tag = f"pathway_{class_name}_{int(time.time())}"
        pathway = {"tag": pathway_tag, "node_ids": pathway_node_ids, "class_name": class_name, "class_idx": class_idx}
        log(f"    🛤️  Pathway '{pathway_tag}': {len(pathway_node_ids)} nodes ({len(new_nodes)} new, {reuse_count} reused)")
        return new_nodes, pathway

    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 5) -> int:
        if len(data) < 10:
            return 1
        max_k = min(max_k, len(data) // 5)
        if max_k < 2:
            return 1
        best_k = 1
        best_score = -1.0
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k

    def _create_node(self, centroid: np.ndarray, stage_idx: int, class_name: str, class_idx: int, cluster_id: int):
        timestamp = int(time.time() * 1000)
        node_id = f"node_s{stage_idx}_c{class_idx}_{timestamp}_{cluster_id}"
        blob_name = f"{node_id}.int8"
        blob_path = str(self.staging_dir / blob_name)
        centroid = centroid.astype(np.float32)
        scale, quantized = quantize_centroid_int8(centroid)
        with open(blob_path, "wb") as f:
            f.write(np.array([scale], dtype=np.float32).tobytes())
            f.write(np.array([centroid.shape[0]], dtype=np.int32).tobytes())
            f.write(quantized.tobytes())
        node = ConceptNode(node_id=node_id, centroid=centroid, int8_blob_path=blob_path,
                           stage_idx=stage_idx, class_label=class_name, class_idx=class_idx,
                           tag=f"concept_{node_id}", meta={"cluster_id": cluster_id, "created_at": timestamp})
        return node
