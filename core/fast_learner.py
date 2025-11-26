# core/fast_learner.py
"""
MobileNetV2 wrapper that captures multi-stage activations.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models
from core.config import TORCH_DEVICE

class MobileNetFastLearner(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, capture_stages=5):
        super().__init__()
        try:
            # new torchvision API
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.mobilenet_v2(weights=weights)
        except Exception:
            base = models.mobilenet_v2(pretrained=pretrained)

        self.features = base.features
        last_channel = getattr(base, "last_channel", 1280)
        self.head = nn.Linear(last_channel, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        total_blocks = len(self.features)
        if capture_stages >= total_blocks:
            self.capture_indices = list(range(total_blocks))
        else:
            step = total_blocks / float(capture_stages)
            self.capture_indices = [min(total_blocks - 1, int(i * step)) for i in range(capture_stages)]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out).reshape(out.size(0), -1)
        return self.head(out)

    def forward_with_cache(self, x: torch.Tensor) -> Tuple[List[Tuple[int, torch.Tensor]], torch.Tensor]:
        B = x.shape[0]
        out = x
        stage_activations = []
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx in self.capture_indices:
                pooled = self.avgpool(out).reshape(B, -1).detach().cpu()
                stage_activations.append((idx, pooled))
        final_vec = self.avgpool(out).reshape(B, -1)
        logits = self.head(final_vec.to(TORCH_DEVICE))
        return stage_activations, logits
