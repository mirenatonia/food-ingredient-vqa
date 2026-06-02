"""EfficientNet-based multi-label ingredient recognition model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class IngredientClassifier(nn.Module):
    """
    Predict one logit per ingredient class.

    We do not apply sigmoid here because BCEWithLogitsLoss expects raw logits.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
