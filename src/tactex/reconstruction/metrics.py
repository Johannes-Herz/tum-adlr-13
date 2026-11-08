from collections.abc import Callable

import torch
from torchmetrics import Metric, MetricCollection, MetricTracker
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)


class LossMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        batch_size: int = predictions.shape[0]
        loss = self.loss_fn(predictions, target.float()).detach()
        self.loss_sum += loss * batch_size
        self.n_samples += batch_size

    def compute(self) -> torch.Tensor:
        return self.loss_sum / self.n_samples


def build_metrics(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], threshold: float
) -> MetricCollection:
    return MetricCollection(
        {
            "loss": LossMetric(loss_fn=loss_fn),
            "iou_occ": BinaryJaccardIndex(threshold=threshold),
            "accuracy": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
        }
    )


def build_trackers(
    metrics: MetricCollection, device: torch.device
) -> tuple[MetricTracker, MetricTracker]:
    maximize: list[bool] = [m.higher_is_better for m in metrics.values()]
    train_tracker: MetricTracker = MetricTracker(
        metrics.clone(prefix="train/"), maximize=maximize
    )
    train_tracker.to(device)
    val_tracker: MetricTracker = MetricTracker(
        metrics.clone(prefix="val/"), maximize=maximize
    )
    val_tracker.to(device)
    return train_tracker, val_tracker
