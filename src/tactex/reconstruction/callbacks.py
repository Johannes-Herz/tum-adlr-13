import os
from pathlib import Path
from typing import Literal, Protocol, Self

import torch
from torch import nn, optim

import wandb
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
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None: ...

    def on_training_end(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None: ...


class WandBRun(TrainingCallback):
    """Callback for logging metrics to Weights & Biases."""

    def __init__(
        self,
        entity: str | None = None,
        project: str | None = None,
        id: str | None = None,
        name: str | None = None,
        dir: Path | None = None,
        config: dict | None = None,
    ) -> None:
        if os.environ.get("WANDB_API_KEY") is None:
            raise ValueError(
                "WANDB_API_KEY is not set. Please set it to your Weights & Bias account API key to use WandBRun."
            )
        if os.environ.get("WANDB_MODE", "online") == "offline":
            print("Warning: WANDB_MODE=offline, so WandBRun will not log metrics.")
        self.run: wandb.Run = wandb.run or wandb.init(
            entity=entity,
            project=project,
            id=id,
            name=name,
            dir=dir,
            resume="allow",
            config=config,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.run.finish()

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        self.run.log(metrics, step=epoch)

    def on_training_end(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        pass


class Checkpointer(TrainingCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_dir: Path,
        every_n_epochs: int,
        metric: str,
        mode: str,
        log_to_wandb: bool,
    ) -> None:
        self.save_dir: Path = save_dir
        self.every_n_epochs: int = every_n_epochs
        self.metric: str = metric
        self.mode: Literal["min", "max"] = mode
        self.log_to_wandb: bool = log_to_wandb
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float | None = None

    def _save_wandb_artifact(self, path: Path, alias: str) -> None:
        artifact = wandb.Artifact(name=f"checkpoint-{wandb.run.id}", type="model")
        artifact.add_file(str(path), name=path.name)
        wandb.run.log_artifact(artifact, aliases=[alias])

    def _save(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict()
                if lr_scheduler is not None
                else None,
                "metrics": metrics,
            },
            path,
        )

    def _is_better(self, new_metric: float, best_metric: float) -> bool:
        if self.mode == "min":
            return new_metric < best_metric
        elif self.mode == "max":
            return new_metric > best_metric

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        if (epoch + 1) % self.every_n_epochs == 0:
            self._save(
                self.save_dir / "latest.pt",
                epoch,
                metrics,
                model,
                optimizer,
                lr_scheduler,
            )
        if self.metric in metrics:
            new_metric: float = metrics[self.metric]
            if self.best_metric is None or self._is_better(
                new_metric, self.best_metric
            ):
                self.best_metric = new_metric
                self._save(
                    self.save_dir / "best.pt",
                    epoch,
                    metrics,
                    model,
                    optimizer,
                    lr_scheduler,
                )

    def on_training_end(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        if self.log_to_wandb and wandb.run is not None:
            if (self.save_dir / "best.pt").exists():
                self._save_wandb_artifact(self.save_dir / "best.pt", alias="best")
            if (self.save_dir / "latest.pt").exists():
                self._save_wandb_artifact(self.save_dir / "latest.pt", alias="latest")
