from __future__ import annotations

from typing import Dict, Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FastLearner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        stage_indices: Iterable[int],
        backbone: str = "mobilenet_v2",
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.stage_indices = tuple(stage_indices)
        self.device = device
        self._build_backbone(pretrained=pretrained)
        self.set_num_classes(num_classes)
        self.to(device)

    def _build_backbone(self, pretrained: bool) -> None:
        if self.backbone_name != "mobilenet_v2":
            raise ValueError(f"Unsupported backbone: {self.backbone_name}. Supported: mobilenet_v2")
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        self.features = net.features
        self.feature_dim = 1280
        self.classifier = nn.Linear(self.feature_dim, 2)

    def set_num_classes(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.classifier.to(self.device)

    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        activations: Dict[int, torch.Tensor] = {}
        h = x
        for idx, block in enumerate(self.features):
            h = block(h)
            if idx in self.stage_indices:
                # Flatten pooled activation per sample.
                activations[idx] = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return h, activations

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        feats, activations = self._extract_features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        logits = self.classifier(pooled)
        return logits, activations
