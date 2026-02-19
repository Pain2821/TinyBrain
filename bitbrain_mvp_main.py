import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch

from bitbrain_mvp.config import BitBrainMVPConfig
from bitbrain_mvp.system import BitBrainMVP
from utils.data import get_task_dataloaders


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("tasks config must be a non-empty list")
    return tasks


def compute_forgetting(task_accuracies: dict, task_order: list[str]) -> dict:
    forgetting = {}
    for i, task_name in enumerate(task_order[:-1]):
        vals = task_accuracies.get(task_name, [])
        if len(vals) == 0:
            continue
        initial = vals[0]
        final = vals[-1]
        f = initial - final
        forgetting[task_name] = {
            "initial_acc": initial,
            "final_acc": final,
            "forgetting": f,
            "retention_rate": final / initial if initial > 0 else 0.0,
        }
    if len(forgetting) == 0:
        return {"per_task": {}, "avg_forgetting": 0.0, "avg_retention_rate": 0.0}
    return {
        "per_task": forgetting,
        "avg_forgetting": float(np.mean([v["forgetting"] for v in forgetting.values()])),
        "avg_retention_rate": float(np.mean([v["retention_rate"] for v in forgetting.values()])),
    }


def run(args):
    set_seed(args.seed)
    tasks = load_tasks(args.tasks_file)
    cfg = BitBrainMVPConfig()
    cfg.epochs_per_task = args.epochs if args.epochs > 0 else cfg.epochs_per_task
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    first_num_classes = tasks[0]["num_classes"]
    model = BitBrainMVP(num_classes=first_num_classes, cfg=cfg)

    task_data = []
    for t in tasks:
        train_loader, val_loader = get_task_dataloaders(
            t["data_dir"],
            batch_size=args.batch_size or cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        task_data.append({**t, "train_loader": train_loader, "val_loader": val_loader})

    results = {"tasks": [], "evaluation_matrix": [], "task_accuracies": {}, "memory_history": []}
    for i, task in enumerate(task_data):
        name = task["name"]
        num_classes = task["num_classes"]
        model.reset_fast_learner(num_classes)
        best = model.train_task(
            train_loader=task["train_loader"],
            val_loader=task["val_loader"],
            task_name=name,
            epochs=cfg.epochs_per_task,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        pathway_id = model.consolidate_task(task["train_loader"], name)
        results["tasks"].append({"name": name, "task_idx": i, "best_val_acc": best, "pathway_id": pathway_id})
        results["memory_history"].append(
            {
                "num_pathways": len(model.memory.pathways),
                "num_nodes": len(model.memory.nodes),
            }
        )

        phase = {}
        for prev in task_data[: i + 1]:
            prev_name = prev["name"]
            eval_res = model.evaluate_loader(prev["val_loader"], task_name=prev_name, use_task_pathway_only=True)
            phase[prev_name] = float(eval_res["accuracy"])
            results["task_accuracies"].setdefault(prev_name, [])
            results["task_accuracies"][prev_name].append(float(eval_res["accuracy"]))
        results["evaluation_matrix"].append({"after_task": name, "accuracies": phase})

    task_order = [t["name"] for t in task_data]
    results["forgetting_summary"] = compute_forgetting(results["task_accuracies"], task_order)
    out = Path("logs_internal/bitbrain_mvp_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[BitBrain-MVP] Saved results to {out}")
    print(f"[BitBrain-MVP] Avg forgetting: {results['forgetting_summary']['avg_forgetting']:.4f}")
    print(f"[BitBrain-MVP] Avg retention: {results['forgetting_summary']['avg_retention_rate']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BitBrain MVP runner")
    parser.add_argument("--tasks-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=0)
    run(parser.parse_args())
