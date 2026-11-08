import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricTracker


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    metric_tracker: MetricTracker,
    device: torch.device,
):
    model.eval()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = (
            x_batch.to(device, torch.float32),
            y_batch.to(device, torch.float32),
        )
        logit_batch: torch.Tensor = model(x_batch)
        metric_tracker.update(logit_batch.detach(), y_batch.int())
