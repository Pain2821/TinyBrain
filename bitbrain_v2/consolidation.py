from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .config import Config
from .fast_learner import FastLearner
from .memory_bank import MemoryBank


@dataclass
class ConsolidationResult:
    task_name: str
    pathway_ids: List[int]
    nodes_created: int
    nodes_reused: int


def _choose_k(data: np.ndarray, max_k: int, seed: int) -> int:
    n = len(data)
    if n < 4:
        return 1
    upper = min(max_k, max(2, int(np.sqrt(n))))
    best_k = 2 if upper >= 2 else 1
    best_score = -1.0
    for k in range(2, upper + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=min(256, n))
        labels = km.fit_predict(data)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


class ConsolidationModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def collect_activations(
        self,
        model: FastLearner,
        loader,
    ) -> Dict[Tuple[int, int], List[torch.Tensor]]:
        model.eval()
        buckets: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        counts = defaultdict(int)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                _, acts = model(x)
                for i in range(x.size(0)):
                    label = int(y[i].item())
                    if counts[label] >= self.cfg.max_samples_per_class:
                        continue
                    counts[label] += 1
                    for stage_idx, stage_act in acts.items():
                        buckets[(label, stage_idx)].append(stage_act[i].detach().cpu())
        return buckets

    def consolidate_task(
        self,
        memory: MemoryBank,
        model: FastLearner,
        train_loader,
        task_name: str,
    ) -> ConsolidationResult:
        buckets = self.collect_activations(model=model, loader=train_loader)
        class_stage_node_ids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        created = 0
        reused = 0

        for (label, stage_idx), vectors in buckets.items():
            if len(vectors) == 0:
                continue
            data = torch.stack(vectors).numpy()
            if len(vectors) < self.cfg.min_samples_for_cluster:
                centroids = [torch.tensor(data.mean(axis=0), dtype=torch.float32)]
            else:
                k = _choose_k(data, self.cfg.max_clusters, self.cfg.seed)
                km = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=self.cfg.seed,
                    n_init=5,
                    batch_size=min(256, len(vectors)),
                )
                labels_k = km.fit_predict(data)
                centroids = [torch.tensor(c, dtype=torch.float32) for c in km.cluster_centers_]
                supports = np.bincount(labels_k, minlength=k).tolist()

            if len(vectors) < self.cfg.min_samples_for_cluster:
                supports = [len(vectors)]

            for centroid, support in zip(centroids, supports):
                reuse_id = memory.find_reusable_node(stage_idx=stage_idx, centroid=centroid, tau_reuse=self.cfg.tau_reuse)
                if reuse_id is not None:
                    class_stage_node_ids[(label, stage_idx)].append(reuse_id)
                    reused += 1
                    continue
                quality = float(support / max(len(vectors), 1))
                node_id = memory.add_node(
                    stage_idx=stage_idx,
                    class_label=label,
                    task_name=task_name,
                    centroid=centroid,
                    quality_score=quality,
                )
                class_stage_node_ids[(label, stage_idx)].append(node_id)
                created += 1

        pathway_ids: List[int] = []
        class_labels = sorted({key[0] for key in class_stage_node_ids.keys()})
        for class_label in class_labels:
            node_ids: List[int] = []
            for stage_idx in sorted(self.cfg.stage_indices):
                node_ids.extend(class_stage_node_ids.get((class_label, stage_idx), []))
            if not node_ids:
                continue
            pid = memory.add_pathway(task_name=task_name, class_label=class_label, node_ids=node_ids)
            pathway_ids.append(pid)

        return ConsolidationResult(
            task_name=task_name,
            pathway_ids=pathway_ids,
            nodes_created=created,
            nodes_reused=reused,
        )
