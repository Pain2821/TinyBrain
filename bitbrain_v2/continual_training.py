from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.data import get_task_dataloaders
from .config import Config
from .consolidation import ConsolidationModule
from .fast_learner import FastLearner
from .memory_bank import MemoryBank
from .metrics import (
    average_accuracy,
    forgetting,
    forward_transfer,
    inference_time_ratio,
    memory_overhead,
    routing_accuracy,
)
from .router import SimilarityRouter, RouteInfo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tasks(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Tasks file must contain a non-empty list")
    return tasks


def model_size_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def train_fast_learner(
    model: FastLearner,
    train_loader,
    val_loader,
    task_name: str,
    cfg: Config,
    epochs: int,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if i % cfg.log_every == 0:
                pass
        scheduler.step()
        val_acc = evaluate_fast_only(model, val_loader, cfg.device)
        best_val = max(best_val, val_acc)
        print(f"[BitBrain-v2] {task_name} epoch {epoch+1}/{epochs} train_acc={correct/max(total,1):.4f} val_acc={val_acc:.4f}")
    return float(best_val)


def evaluate_fast_only(model: FastLearner, loader, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct / max(total, 1))


def predict_with_router(
    model: FastLearner,
    memory: MemoryBank,
    router: SimilarityRouter,
    x: torch.Tensor,
    allowed_pathways: Optional[List[int]],
    cfg: Config,
) -> Tuple[torch.Tensor, List[RouteInfo]]:
    model.eval()
    x = x.to(cfg.device)
    with torch.no_grad():
        logits, acts = model(x)
        probs = torch.softmax(logits, dim=1)
        preds: List[int] = []
        infos: List[RouteInfo] = []
        for i in range(x.size(0)):
            best_pid, best_score = router.best_pathway(
                memory=memory,
                activations=acts,
                sample_idx=i,
                allowed_pathways=allowed_pathways,
            )
            pathway_pred = None
            pathway_conf = best_score
            if best_pid is not None:
                pathway_pred, pathway_conf = router.pathway_pred_label(
                    memory=memory,
                    pathway=memory.pathways[best_pid],
                    activations=acts,
                    sample_idx=i,
                )
            pred, info = router.decide(
                fast_probs=probs,
                pathway_pred=pathway_pred,
                pathway_confidence=pathway_conf,
                sample_idx=i,
            )
            info.pathway_id = best_pid
            info.pathway_score = best_score
            preds.append(pred)
            infos.append(info)
    return torch.tensor(preds, device=cfg.device, dtype=torch.long), infos


def evaluate_with_router(
    model: FastLearner,
    memory: MemoryBank,
    router: SimilarityRouter,
    loader,
    cfg: Config,
    allowed_pathways: Optional[List[int]],
) -> dict:
    total = 0
    correct = 0
    pathway_selected_samples = 0
    pathway_selected_correct = 0
    pathway_used_samples = 0
    pathway_used_correct = 0
    for x, y in loader:
        y = y.to(cfg.device)
        preds, infos = predict_with_router(model, memory, router, x, allowed_pathways=allowed_pathways, cfg=cfg)
        batch_correct = (preds == y)
        correct += batch_correct.sum().item()
        total += y.size(0)
        for idx, info in enumerate(infos):
            if info.pathway_id is not None:
                pathway_selected_samples += 1
                if info.pathway_pred is not None and int(info.pathway_pred) == int(y[idx].item()):
                    pathway_selected_correct += 1
            if info.source.startswith("pathway"):
                pathway_used_samples += 1
                pathway_used_correct += int(batch_correct[idx].item())
    return {
        "accuracy": float(correct / max(total, 1)),
        "pathway_selected_samples": pathway_selected_samples,
        "pathway_used_samples": pathway_used_samples,
        "pathway_selection_accuracy": routing_accuracy(pathway_selected_correct, pathway_selected_samples),
        "routing_accuracy": routing_accuracy(pathway_used_correct, pathway_used_samples),
        "pathway_selection_rate": float(pathway_selected_samples / max(total, 1)),
        "route_rate": float(pathway_used_samples / max(total, 1)),
        "total": total,
    }


def _timed_fast_inference(model: FastLearner, loader, device: str, max_batches: int) -> float:
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            _ = model(x)
    return float(time.perf_counter() - start)


def _timed_routed_inference(
    model: FastLearner,
    memory: MemoryBank,
    router: SimilarityRouter,
    loader,
    cfg: Config,
    allowed_pathways: Optional[List[int]],
    max_batches: int,
) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            _preds, _infos = predict_with_router(model, memory, router, x, allowed_pathways, cfg)
    return float(time.perf_counter() - start)


def run_experiment(args) -> dict:
    cfg = Config()
    cfg.seed = args.seed
    cfg.device = args.device or cfg.device
    cfg.epochs_per_task = args.epochs if args.epochs > 0 else cfg.epochs_per_task
    cfg.batch_size = args.batch_size if args.batch_size > 0 else cfg.batch_size
    cfg.quantize = not args.no_quantize
    cfg.task_aware_routing = not args.global_routing
    cfg.use_confidence_logic = not args.disable_confidence_logic
    set_seed(cfg.seed)

    tasks = load_tasks(args.tasks_file)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    model = FastLearner(
        num_classes=int(tasks[0]["num_classes"]),
        stage_indices=cfg.stage_indices,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        device=cfg.device,
    )
    memory = MemoryBank(quantize=cfg.quantize)
    consolidator = ConsolidationModule(cfg)
    router = SimilarityRouter(cfg)

    task_data = []
    for task in tasks:
        train_loader, val_loader = get_task_dataloaders(
            data_dir=task["data_dir"],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        task_data.append({**task, "train_loader": train_loader, "val_loader": val_loader})

    task_pathways: Dict[str, List[int]] = {}
    task_accuracy_history: Dict[str, List[float]] = {}
    pre_train_accuracies: Dict[str, float] = {}
    random_baseline: Dict[str, float] = {}
    evaluation_matrix: List[dict] = []
    phase_logs: List[dict] = []

    for idx, task in enumerate(task_data):
        task_name = str(task["name"])
        num_classes = int(task["num_classes"])
        random_baseline[task_name] = 1.0 / max(num_classes, 1)

        if model.num_classes != num_classes:
            model.set_num_classes(num_classes)
            model.to(cfg.device)

        pre_acc = evaluate_fast_only(model, task["val_loader"], cfg.device)
        pre_train_accuracies[task_name] = pre_acc

        epochs = int(task.get("epochs", cfg.epochs_per_task))
        if args.epochs > 0:
            epochs = args.epochs
        best_val = train_fast_learner(
            model=model,
            train_loader=task["train_loader"],
            val_loader=task["val_loader"],
            task_name=task_name,
            cfg=cfg,
            epochs=epochs,
        )

        consolidation_result = consolidator.consolidate_task(
            memory=memory,
            model=model,
            train_loader=task["train_loader"],
            task_name=task_name,
        )
        task_pathways[task_name] = consolidation_result.pathway_ids

        after_task_eval = {}
        routing_eval = {}
        for seen in task_data[: idx + 1]:
            seen_name = str(seen["name"])
            allowed = task_pathways.get(seen_name) if cfg.task_aware_routing else None
            eval_res = evaluate_with_router(
                model=model,
                memory=memory,
                router=router,
                loader=seen["val_loader"],
                cfg=cfg,
                allowed_pathways=allowed,
            )
            after_task_eval[seen_name] = eval_res["accuracy"]
            routing_eval[seen_name] = {
                "routing_accuracy": eval_res["routing_accuracy"],
                "pathway_selection_accuracy": eval_res["pathway_selection_accuracy"],
                "pathway_selection_rate": eval_res["pathway_selection_rate"],
                "route_rate": eval_res["route_rate"],
            }
            task_accuracy_history.setdefault(seen_name, []).append(eval_res["accuracy"])

        evaluation_matrix.append({"after_task": task_name, "accuracies": after_task_eval})
        phase_logs.append(
            {
                "task": task_name,
                "best_val_accuracy": best_val,
                "pre_train_accuracy": pre_acc,
                "consolidation": {
                    "pathway_ids": consolidation_result.pathway_ids,
                    "nodes_created": consolidation_result.nodes_created,
                    "nodes_reused": consolidation_result.nodes_reused,
                },
                "routing": routing_eval,
                "memory": {
                    "num_nodes": len(memory.nodes),
                    "num_pathways": len(memory.pathways),
                    "memory_bank_bytes": memory.memory_overhead_bytes(),
                },
            }
        )

    final_accuracies = evaluation_matrix[-1]["accuracies"] if evaluation_matrix else {}
    forgetting_per_task = forgetting(task_accuracy_history)
    avg_forgetting = float(np.mean(list(forgetting_per_task.values()))) if forgetting_per_task else 0.0
    avg_acc = average_accuracy(final_accuracies)
    fwt = forward_transfer(pre_train_accuracies, random_baseline, [str(t["name"]) for t in task_data])

    baseline_bytes = model_size_bytes(model)
    memory_stats = memory_overhead(memory.memory_overhead_bytes(), baseline_bytes)

    ref_loader = task_data[0]["val_loader"]
    first_task_name = str(task_data[0]["name"])
    allowed = task_pathways.get(first_task_name) if cfg.task_aware_routing else None
    t_fast = _timed_fast_inference(model, ref_loader, cfg.device, cfg.eval_time_batches)
    t_routed = _timed_routed_inference(model, memory, router, ref_loader, cfg, allowed, cfg.eval_time_batches)
    time_ratio = inference_time_ratio(t_routed, t_fast)

    route_rates = []
    pathway_selection_rates = []
    route_accs = []
    pathway_selection_accs = []
    for row in phase_logs:
        for v in row["routing"].values():
            route_rates.append(v["route_rate"])
            pathway_selection_rates.append(v["pathway_selection_rate"])
            route_accs.append(v["routing_accuracy"])
            pathway_selection_accs.append(v["pathway_selection_accuracy"])

    results = {
        "config": cfg.__dict__,
        "tasks": [{"name": t["name"], "data_dir": t["data_dir"], "num_classes": t["num_classes"]} for t in task_data],
        "phase_logs": phase_logs,
        "evaluation_matrix": evaluation_matrix,
        "task_accuracy_history": task_accuracy_history,
        "metrics": {
            "average_accuracy": avg_acc,
            "forgetting_per_task": forgetting_per_task,
            "average_forgetting": avg_forgetting,
            "forward_transfer": fwt,
            "routing_accuracy": float(np.mean(route_accs)) if route_accs else 0.0,
            "pathway_selection_accuracy": float(np.mean(pathway_selection_accs)) if pathway_selection_accs else 0.0,
            "average_pathway_selection_rate": float(np.mean(pathway_selection_rates)) if pathway_selection_rates else 0.0,
            "average_route_rate": float(np.mean(route_rates)) if route_rates else 0.0,
            "memory_overhead": memory_stats,
            "inference_time_ratio": time_ratio,
            "timing_seconds": {"baseline_fast": t_fast, "routed": t_routed},
        },
        "final_state": {
            "num_nodes": len(memory.nodes),
            "num_pathways": len(memory.pathways),
            "task_pathways": task_pathways,
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="BitBrain v2 continual learning runner")
    parser.add_argument("--tasks-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--global-routing", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--disable-confidence-logic", action="store_true")
    parser.add_argument("--output", type=str, default="logs_internal/bitbrain_v2_results.json")
    args = parser.parse_args()

    results = run_experiment(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[BitBrain-v2] Results saved to: {out}")
    print(f"[BitBrain-v2] Avg accuracy: {results['metrics']['average_accuracy']:.4f}")
    print(f"[BitBrain-v2] Avg forgetting: {results['metrics']['average_forgetting']:.4f}")


if __name__ == "__main__":
    main()
