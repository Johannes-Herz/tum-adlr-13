from collections.abc import Protocol

from torch import nn, optim

from tactex.configs import FileConfig


class TrainingConfig(FileConfig):
    pass


class TrainingCallback(Protocol):
    """Base class — subclasses only override the hooks they need."""

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None: ...
