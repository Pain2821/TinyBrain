from __future__ import annotations

from typing import Dict, List
import numpy as np


def average_accuracy(final_task_accuracies: Dict[str, float]) -> float:
    if not final_task_accuracies:
        return 0.0
    return float(np.mean(list(final_task_accuracies.values())))


def forgetting(task_accuracy_history: Dict[str, List[float]]) -> Dict[str, float]:
    per_task = {}
    for task_name, history in task_accuracy_history.items():
        if len(history) < 2:
            per_task[task_name] = 0.0
            continue
        max_prev = max(history[:-1])
        per_task[task_name] = float(max_prev - history[-1])
    return per_task


def forward_transfer(pre_train_accuracies: Dict[str, float], random_baseline: Dict[str, float], task_order: List[str]) -> float:
    vals = []
    for task_name in task_order[1:]:
        if task_name not in pre_train_accuracies:
            continue
        rand_acc = random_baseline.get(task_name, 0.0)
        vals.append(pre_train_accuracies[task_name] - rand_acc)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def routing_accuracy(correct_when_pathway_used: int, routed_samples: int) -> float:
    if routed_samples == 0:
        return 0.0
    return float(correct_when_pathway_used / routed_samples)


def memory_overhead(memory_bytes: int, baseline_model_bytes: int) -> dict:
    if baseline_model_bytes <= 0:
        return {"memory_overhead_ratio": 0.0, "memory_overhead_bytes": memory_bytes}
    return {
        "memory_overhead_ratio": float(memory_bytes / baseline_model_bytes),
        "memory_overhead_bytes": int(memory_bytes),
    }


def inference_time_ratio(routed_time: float, baseline_time: float) -> float:
    if baseline_time <= 0:
        return 0.0
    return float(routed_time / baseline_time)
