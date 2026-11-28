from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import Field, field_serializer, field_validator, model_validator

from tactex.configs import FileConfig


class DataConfig(FileConfig):
    """Configuration for data loading and preprocessing."""

    name: str = Field(required=True)
    resolution: int = Field(default=64, gt=0)
    version: int = Field(default=1, gt=0)
    limits: dict[Literal["train", "val", "test"], int | None] = Field(
        default_factory=lambda: {"train": None, "val": None, "test": None}
    )
    max_rays: int = Field(default=100, gt=0)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    in_memory: bool = Field(default=True)


class ModelConfig(FileConfig):
    """Configuration for the model architecture."""

    type: Literal["unet"] = Field(default="unet")
    load_checkpoint: bool = Field(default=False)
    checkpoint_path: Path | None = Field(default=None)
    in_channels: int = Field(default=3, gt=0)
    out_channels: int = Field(default=1, gt=0)
    widths: list[int] | None = Field(default=[64, 128], min_length=1)
    depth: int | None = Field(default=3, gt=0)
    base_width: int | None = Field(default=32, gt=0)
    max_width: int | None = Field(default=None)
    kernel_size: int = Field(default=3, gt=0)


class OptimizerConfig(FileConfig):
    """Configuration for optimization and training."""

    type: Literal["adam", "sgd", "adamw"] = Field(default="adam")
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    warmup_steps: int = Field(default=0, ge=0)
    grad_clip: float = Field(default=0.5, gt=0)


class LRSchedulerConfig(FileConfig):
    """Configuration for learning rate scheduler."""

    type: Literal["step", "reduce_on_plateau"] = Field(default="reduce_on_plateau")
    step_size: int = Field(default=10, gt=0)
    gamma: float = Field(default=0.1, gt=0)
    factor: float = Field(default=0.1, gt=0)
    patience: int = Field(default=10, ge=0)
    mode: Literal["min", "max"] = Field(default="min")


class LossConfig(FileConfig):
    """Configuration for the loss function."""

    type: Literal["bce", "bcedice"] = Field(default="bce")
    bce_weight: float = Field(default=0.5, gt=0)
    eps: float = Field(default=1e-8, gt=0)


class WandBConfig(FileConfig):
    """Configuration for WandB logging."""

    entity: str = Field(required=True)
    project: str = Field(required=True)
    run_id: str | None = Field(default=None)
    run_name: str | None = Field(default=None)


class SweepConfig(FileConfig):
    """Configuration for hyperparameter sweeps."""

    method: Literal["random", "bayes"] = Field(default="random")
    parameters: dict[str, Any] = Field(default_factory=dict)
    early_terminate: bool = Field(default=False)
    metric: str = Field(default="val/iou_occ")
    goal: Literal["minimize", "maximize"] = Field(default="maximize")
    num_runs: int = Field(default=5, gt=0)
    min_iter: int = Field(default=1, gt=0)

    def to_wandb_sweep(self) -> dict[str, Any]:
        """Convert to a WandB sweep configuration dictionary."""
        sweep_config: dict[str, Any] = {
            "method": self.method,
            "parameters": self.parameters,
        }
        if self.early_terminate:
            sweep_config["early_terminate"] = {
                "type": "hyperband",
                "min_iter": self.min_iter,
            }
        if self.metric:
            sweep_config["metric"] = {"name": self.metric, "goal": self.goal}

        return sweep_config


class TrainConfig(FileConfig):
    """Configuration for the training process."""

    name: str = Field(required=True)
    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=10, gt=0)
    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_every: int = Field(default=1, gt=0)
    ckpt_path: Path | None = Field(default=None)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    wandb: WandBConfig | None = Field(default=None)
    sweep: SweepConfig | None = Field(default=None)

    @field_validator("device", mode="before")
    def _validate_device(cls, v):
        if isinstance(v, str):
            return torch.device(v)
        return v

    @field_serializer("device")
    def _serialize_device(self, v: torch.device) -> str:
        return str(v)

    @model_validator(mode="after")
    def _validate_model_resolution(self) -> Self:
        depth: int = (
            len(self.model.widths)
            if self.model.widths is not None
            else self.model.depth
        )
        divisor: int = 2 ** (depth - 1)
        if self.data.resolution % divisor != 0:
            raise ValueError(
                f"data.resolution ({self.data.resolution}) must be divisible by 2**(depth - 1) = {divisor} for depth={depth}"
            )
        return self

    @model_validator(mode="after")
    def _default_name(self) -> Self:
        if not self.name:
            mode: Literal["sweep", "train"] = (
                "sweep" if self.sweep is not None else "train"
            )
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")
            self.name: str = f"{self.model.type}_{self.data.name}_res{self.data.resolution}_{mode}_{timestamp}"
        return self
