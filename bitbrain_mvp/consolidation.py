from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .memory_bank import MemoryBank
from .model import FastLearner
from .config import BitBrainMVPConfig


def _choose_k(data: np.ndarray, k_max: int, random_state: int) -> int:
    n = len(data)
    if n < 4:
        return 1
    upper = min(k_max, max(2, int(np.sqrt(n))))
    if upper <= 2:
        return 2
    best_k = 2
    best_score = -1.0
    for k in range(2, upper + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5, batch_size=min(256, n))
        labels = km.fit_predict(data)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def collect_activations(
    model: FastLearner,
    loader,
    device: str,
    max_samples_per_class: int,
) -> Dict[Tuple[int, int], List[torch.Tensor]]:
    model.eval()
    buckets: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
    class_counts: Dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            acts = model.extract_stage_activations(x)
            for i in range(x.size(0)):
                label = int(y[i].item())
                if class_counts[label] >= max_samples_per_class:
                    continue
                class_counts[label] += 1
                for stage, tensor in acts.items():
                    buckets[(label, stage)].append(tensor[i].detach().cpu())
    return buckets


def consolidate_task_mvp(
    memory: MemoryBank,
    model: FastLearner,
    train_loader,
    task_name: str,
    cfg: BitBrainMVPConfig,
) -> int:
    buckets = collect_activations(
        model=model,
        loader=train_loader,
        device=cfg.device,
        max_samples_per_class=cfg.max_samples_per_class,
    )
    task_node_ids: List[int] = []
    for (label, stage), vectors in buckets.items():
        if len(vectors) == 0:
            continue
        data = torch.stack(vectors).numpy()
        if len(vectors) < 8:
            centroids = [torch.tensor(data.mean(axis=0), dtype=torch.float32)]
        else:
            k = _choose_k(data, cfg.k_max, cfg.random_state)
            km = MiniBatchKMeans(n_clusters=k, random_state=cfg.random_state, n_init=5, batch_size=min(256, len(vectors)))
            km.fit(data)
            centroids = [torch.tensor(c, dtype=torch.float32) for c in km.cluster_centers_]

        for c in centroids:
            reusable = memory.find_reusable_node(stage=stage, centroid=c, threshold=cfg.reuse_threshold)
            if reusable is not None:
                task_node_ids.append(reusable)
                continue
            quality = 1.0
            nid = memory.add_node(
                stage=stage,
                label=label,
                task_name=task_name,
                centroid=c,
                quality=quality,
            )
            task_node_ids.append(nid)

    unique_ids = sorted(set(task_node_ids))
    return memory.add_pathway(task_name=task_name, node_ids=unique_ids)
