import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection, MetricTracker


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    metrics: Metric | MetricCollection | MetricTracker,
    device: torch.device,
):
    """Run the model over a dataset and feed all predictions into ``metrics``.

    Only calls ``metrics.update`` once per batch. It does not reset, increment,
    or compute the metrics, so the caller controls what counts as one
    evaluation: call ``increment()`` (tracker) or ``reset()`` (metric or
    collection) before, and ``compute()`` after.

    Runs without gradient tracking and puts the model in eval mode (BatchNorm
    uses its running statistics, dropout is disabled). The model is left in
    eval mode; call ``model.train()`` before continuing training.

    Args:
        model: Model producing logits of the same shape as the targets.
            Must already be on ``device``.
        dataloader: Yields ``(x, y)`` batches.
        metrics: Receives ``(logits, targets)`` per batch, with integer targets.
            Must already be on ``device``.
        device: Device the batches are moved to.

    Returns:
        None. Results are read from ``metrics`` afterwards.
    """
    model.eval()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = (
            x_batch.to(device, torch.float32),
            y_batch.to(device, torch.float32),
        )
        logit_batch: torch.Tensor = model(x_batch)
        metrics.update(logit_batch.detach(), y_batch.int())
