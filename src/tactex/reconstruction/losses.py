import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.bce_weight: float = bce_weight
        self.eps: float = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss: torch.Tensor = self.bce(logits, targets)
        probabilities: torch.Tensor = torch.sigmoid(logits)
        soft_intersection: torch.Tensor = (probabilities * targets).sum(dim=(1, 2, 3))
        soft_union: torch.Tensor = probabilities.sum(dim=(1, 2, 3)) + targets.sum(
            dim=(1, 2, 3)
        )
        dice_loss: torch.Tensor = (
            1.0
            - ((2.0 * soft_intersection + self.eps) / (soft_union + self.eps)).mean()
        )
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
