from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import torch
import torch.nn as nn

from .config import BitBrainMVPConfig
from .memory_bank import MemoryBank
from .model import FastLearner
from .router import SimilarityRouter
from .consolidation import consolidate_task_mvp


class BitBrainMVP:
    def __init__(self, num_classes: int, cfg: Optional[BitBrainMVPConfig] = None):
        self.cfg = cfg or BitBrainMVPConfig()
        self.memory = MemoryBank()
        self.fast = FastLearner(num_classes=num_classes, stage_indices=self.cfg.stage_indices, device=self.cfg.device)
        self.router = SimilarityRouter(
            route_threshold=self.cfg.route_threshold,
            high_conf=self.cfg.high_conf,
            low_conf=self.cfg.low_conf,
            conf_gap=self.cfg.conf_gap,
        )
        self.current_num_classes = num_classes
        self.task_pathways: Dict[str, int] = {}

    def save(self, path: str) -> None:
        payload = {
            "cfg": self.cfg.__dict__,
            "current_num_classes": self.current_num_classes,
            "task_pathways": self.task_pathways,
            "fast_state": self.fast.state_dict(),
            "memory": self.memory.to_export(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "BitBrainMVP":
        payload = torch.load(path, map_location=map_location or "cpu")
        cfg = BitBrainMVPConfig(**payload["cfg"])
        model = cls(num_classes=int(payload["current_num_classes"]), cfg=cfg)
        model.fast.load_state_dict(payload["fast_state"])
        model.memory = MemoryBank.from_export(payload["memory"])
        model.task_pathways = {str(k): int(v) for k, v in payload.get("task_pathways", {}).items()}
        return model

    def reset_fast_learner(self, num_classes: int):
        if num_classes == self.current_num_classes:
            return
        self.fast.set_num_classes(num_classes)
        self.current_num_classes = num_classes

    def train_task(self, train_loader, val_loader, task_name: str, epochs: int, lr: float, weight_decay: float) -> float:
        self.fast.train()
        opt = torch.optim.AdamW(self.fast.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        best = 0.0

        for ep in range(epochs):
            total, correct, loss_sum = 0, 0, 0.0
            self.fast.train()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                out = self.fast(x)
                loss = crit(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                if i % self.cfg.log_every == 0:
                    pass
            sched.step()
            val = self.evaluate_loader(val_loader, task_name=task_name, use_task_pathway_only=False)["accuracy"]
            if val > best:
                best = val
            print(f"[BitBrain-MVP] {task_name} epoch {ep+1}/{epochs} train_acc={correct/max(total,1):.4f} val_acc={val:.4f}")
        return float(best)

    def consolidate_task(self, train_loader, task_name: str) -> int:
        pid = consolidate_task_mvp(self.memory, self.fast, train_loader, task_name, self.cfg)
        self.task_pathways[task_name] = pid
        return pid

    def predict_batch(
        self,
        x: torch.Tensor,
        task_name: Optional[str] = None,
        use_task_pathway_only: bool = False,
    ) -> torch.Tensor:
        self.fast.eval()
        x = x.to(self.cfg.device)
        with torch.no_grad():
            fast_logits = self.fast(x)
            fast_probs = torch.softmax(fast_logits, dim=1)
            acts = self.fast.extract_stage_activations(x)
            preds = []
            allowed = None
            force_pathway = False
            if task_name is not None and task_name in self.task_pathways:
                allowed = [self.task_pathways[task_name]]
                force_pathway = use_task_pathway_only
            for i in range(x.size(0)):
                pred, _ = self.router.route(
                    memory=self.memory,
                    stage_acts=acts,
                    fast_probs=fast_probs,
                    sample_idx=i,
                    allowed_pathways=allowed,
                    force_pathway=force_pathway,
                )
                preds.append(pred)
            return torch.tensor(preds, device=self.cfg.device, dtype=torch.long)

    def evaluate_loader(self, loader, task_name: str, use_task_pathway_only: bool = True) -> dict:
        total, correct = 0, 0
        for x, y in loader:
            y = y.to(self.cfg.device)
            pred = self.predict_batch(x, task_name=task_name, use_task_pathway_only=use_task_pathway_only)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return {"task_name": task_name, "accuracy": (correct / max(total, 1)), "correct": correct, "total": total}

