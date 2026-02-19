from __future__ import annotations

from typing import Dict, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FastLearner(nn.Module):
    def __init__(self, num_classes: int, stage_indices: Iterable[int], device: str):
        super().__init__()
        self.stage_indices = tuple(stage_indices)
        self.device = device
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.net = models.mobilenet_v2(weights=weights)
        self.set_num_classes(num_classes)
        self.to(device)

    def set_num_classes(self, num_classes: int):
        self.num_classes = num_classes
        self.net.classifier = nn.Linear(1280, num_classes)
        nn.init.xavier_uniform_(self.net.classifier.weight)
        nn.init.zeros_(self.net.classifier.bias)
        self.net.classifier.to(self.device)

    def extract_stage_activations(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        acts: Dict[int, torch.Tensor] = {}
        h = x
        for idx, block in enumerate(self.net.features):
            h = block(h)
            if idx in self.stage_indices:
                pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
                acts[idx] = pooled
        return acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net.features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.net.classifier(pooled)

