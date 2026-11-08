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
    """Sample-weighted mean of a loss over all batches since the last reset.

    Lets the training loss live in a ``MetricCollection`` next to the other
    metrics, with the same ``update(preds, target)`` signature.

    Assumes ``loss_fn`` uses mean reduction and that all samples have the same
    number of elements (e.g. fixed-size grids), so that batch mean times batch
    size equals the batch sum.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            loss_fn: Loss on ``(logits, targets)`` with mean reduction, e.g.
                ``nn.BCEWithLogitsLoss()``.
        """
        super().__init__()
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = loss_fn
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        """Add the loss of one batch.

        Args:
            predictions: Logits of shape ``(B, ...)``.
            target: Targets of the same shape; converted to float for the loss.
        """
        batch_size: int = predictions.shape[0]
        loss = self.loss_fn(predictions, target.float()).detach()
        self.loss_sum += loss * batch_size
        self.n_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Return the mean loss over all samples seen since the last reset."""
        return self.loss_sum / self.n_samples


def build_metrics(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], threshold: float
) -> MetricCollection:
    """Build the metric collection for binary occupancy reconstruction.

    All metrics take ``(logits, long_targets)``. The classification metrics
    refer to the occupied class, with logits turned into occupied/free by
    ``threshold`` on the sigmoid probability.

    Args:
        loss_fn: Loss tracked as ``"loss"``; should be the training loss.
        threshold: Probability above which a cell counts as occupied.

    Returns:
        Collection with keys ``loss``, ``iou_occ``, ``accuracy``, ``precision``,
        ``recall`` and ``f1``. Every metric defines ``higher_is_better``.
    """
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
    """Build one tracker for training and one for validation.

    Each tracker wraps a copy of ``metrics`` with the prefix ``"train/"`` or
    ``"val/"`` and records one result per epoch. Whether higher or lower is
    better is taken from each metric's ``higher_is_better``, which
    ``best_metric()`` relies on.

    Create new trackers for every new model or run; reusing them continues the
    previous history.

    Args:
        metrics: Template collection, e.g. from ``build_metrics``. Not modified.
        device: Device the trackers are moved to.

    Returns:
        ``(train_tracker, val_tracker)``.
    """
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
