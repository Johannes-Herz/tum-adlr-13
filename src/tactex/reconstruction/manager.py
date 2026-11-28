from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
from torchmetrics import MetricCollection, MetricTracker

import wandb
from tactex.configs import FileConfig
from tactex.reconstruction.callbacks import Checkpointer, WandBRun
from tactex.reconstruction.config import (
    DataConfig,
    LossConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from tactex.reconstruction.data import OccupancyGridDataset, build_dataloader
from tactex.reconstruction.losses import BCEDiceLoss
from tactex.reconstruction.metrics import build_metrics, build_trackers
from tactex.reconstruction.training import train
from tactex.reconstruction.unet import UNet
from tactex.settings import get_settings


class ModuleManager[TConfig: FileConfig](ABC):
    """Base class for managing modules configured by a FileConfig."""

    def __init__(self, config: TConfig):
        self.config: TConfig = config

    def _resolve_config(self, overrides: dict | None = None) -> TConfig:
        if not overrides:
            return self.config
        valid_keys = type(self.config).model_fields.keys()
        filtered = {k: v for k, v in overrides.items() if k in valid_keys}
        return type(self.config).model_validate(self.config.model_dump() | filtered)

    @abstractmethod
    def build(self, *args: Any, config: dict | None = None, **kwargs: Any) -> Any:
        """Build the managed component, optionally overriding config values."""
        ...


class ModelManager(ModuleManager[ModelConfig]):
    def build(self, config: dict | None = None) -> torch.nn.Module:
        cfg: ModelConfig = self._resolve_config(config)
        checkpoint: dict | None = None
        if cfg.load_checkpoint and cfg.checkpoint_path is not None:
            checkpoint: dict = torch.load(f=cfg.checkpoint_path, map_location="cpu")
            cfg = self._resolve_config(checkpoint["model_config"])
        if cfg.type == "unet":
            if cfg.widths is not None:
                model: UNet = UNet(
                    in_channels=cfg.in_channels,
                    out_channels=cfg.out_channels,
                    widths=cfg.widths,
                    kernel_size=cfg.kernel_size,
                )
            elif cfg.depth is not None and cfg.base_width is not None:
                model: UNet = UNet.from_depth(
                    in_channels=cfg.in_channels,
                    out_channels=cfg.out_channels,
                    depth=cfg.depth,
                    base_width=cfg.base_width,
                    max_width=cfg.max_width,
                    kernel_size=cfg.kernel_size,
                )
            else:
                raise ValueError(
                    "Either 'widths' or both 'depth' and 'base_width' must be specified in the model configuration to construct a U-Net."
                )
            if checkpoint is not None:
                model.load_state_dict(checkpoint["model_state_dict"])
            return model
        raise ValueError(f"Unsupported model type: {cfg.type}")


class OptimizerManager(ModuleManager[OptimizerConfig]):
    def build(
        self, model: torch.nn.Module, config: dict | None = None
    ) -> torch.optim.Optimizer:
        cfg: OptimizerConfig = self._resolve_config(config)
        if cfg.type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        if cfg.type == "adam":
            return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        if cfg.type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {cfg.type}")


class LRSchedulerManager(ModuleManager[LRSchedulerConfig]):
    def build(
        self, optimizer: torch.optim.Optimizer, config: dict | None = None
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        cfg: LRSchedulerConfig = self._resolve_config(config)
        if cfg.type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        if cfg.type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
            )
        return None


class DataManager(ModuleManager[DataConfig]):
    @classmethod
    def _build_dataloader(
        cls,
        dataset: OccupancyGridDataset,
        cfg: DataConfig,
        split: Literal["train", "val", "test"] = "train",
    ) -> torch.utils.data.DataLoader:
        return build_dataloader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            resolution=cfg.resolution,
            max_rays=cfg.max_rays,
            name=f"{split}.dataloader",
            seed=None,  # TODO: Add a fixed seed for val, test splits to ensure reproducibility over different model trainings
            num_workers=cfg.num_workers,
            optimized=True,
        )

    @classmethod
    def _build_datasets(
        cls, cfg: DataConfig
    ) -> tuple[OccupancyGridDataset, OccupancyGridDataset, OccupancyGridDataset]:
        dataset_dir: Path = (
            get_settings().data_root
            / "datasets"
            / cfg.name
            / f"res{cfg.resolution:03d}"
            / f"v{cfg.version:03d}"
        )
        return (
            OccupancyGridDataset(
                root=dataset_dir,
                split="train",
                in_memory=cfg.in_memory,
                limit=cfg.limits.get("train", None),
            ),
            OccupancyGridDataset(
                root=dataset_dir,
                split="val",
                in_memory=cfg.in_memory,
                limit=cfg.limits.get("val", None),
            ),
            OccupancyGridDataset(
                root=dataset_dir,
                split="test",
                in_memory=cfg.in_memory,
                limit=cfg.limits.get("test", None),
            ),
        )

    def build(
        self, config: dict | None = None
    ) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        cfg: DataConfig = self._resolve_config(config)
        train_dataset, val_dataset, test_dataset = self._build_datasets(cfg=cfg)
        return (
            self._build_dataloader(
                train_dataset,
                cfg=cfg,
                split="train",
            ),
            self._build_dataloader(val_dataset, cfg=cfg, split="val"),
            self._build_dataloader(
                test_dataset,
                cfg=cfg,
                split="test",
            ),
        )


class LossManager(ModuleManager[LossConfig]):
    def build(self, config: dict | None = None) -> torch.nn.Module:
        cfg: LossConfig = self._resolve_config(config)
        if cfg.type == "bce":
            return torch.nn.BCEWithLogitsLoss()
        if cfg.type == "bcedice":
            return BCEDiceLoss(
                bce_weight=cfg.bce_weight,
                eps=cfg.eps,
            )
        raise ValueError(f"Unsupported loss function: {cfg.type}")


class TrainManager:
    def __init__(self, config: TrainConfig):
        self.config: TrainConfig = config
        self.model_manager: ModelManager = ModelManager(config=self.config.model)
        self.optimizer_manager: OptimizerManager = OptimizerManager(
            config=self.config.optimizer
        )
        self.lr_scheduler_manager: LRSchedulerManager = LRSchedulerManager(
            config=self.config.lr_scheduler
        )
        self.data_manager: DataManager = DataManager(config=self.config.data)
        self.loss_manager: LossManager = LossManager(config=self.config.loss)

    def _get_callbacks(self, run_id: str | None = None) -> list:
        callbacks: list = []
        if self.config.ckpt_every > 0:
            base_dir: Path = self.config.ckpt_path or (
                get_settings().data_root / "models"
            )
            checkpointer_save_dir: Path = (
                base_dir / self.config.data.name / (run_id or self.config.name)
            )
            callbacks.append(
                Checkpointer(
                    save_dir=checkpointer_save_dir,
                    every_n_epochs=self.config.ckpt_every,
                    metric="val/iou_occ",
                    mode="max",
                    save_to_wandb=self.config.wandb is not None,
                )
            )
        return callbacks

    def _build_trackers(
        self, loss_fn: torch.nn.Module
    ) -> tuple[MetricTracker, MetricTracker]:
        metrics: MetricCollection = build_metrics(
            loss_fn=loss_fn, threshold=0.5
        )  # NOTE: The threshold is hardcoded here, but it could and should be made configurable if needed.
        return build_trackers(metrics=metrics, device=self.config.device)

    @classmethod
    def extract_namespace(cls, config: dict | None, namespace: str) -> dict | None:
        if not config:
            return None
        prefix: str = f"{namespace}."
        return {
            key.removeprefix(prefix): value
            for key, value in config.items()
            if key.startswith(prefix)
        }

    def _build_modules(self, config: dict | None = None) -> tuple:
        model: torch.nn.Module = self.model_manager.build(
            config=self.extract_namespace(config, "model")
        )
        train_dataloader, val_dataloader, _ = self.data_manager.build(
            config=self.extract_namespace(config, "data")
        )
        loss_fn: torch.nn.Module = self.loss_manager.build(
            config=self.extract_namespace(config, "loss")
        )
        optimizer: torch.optim.Optimizer = self.optimizer_manager.build(
            model=model, config=self.extract_namespace(config, "optimizer")
        )

        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            self.lr_scheduler_manager.build(
                optimizer=optimizer,
                config=self.extract_namespace(config, "lr_scheduler"),
            )
        )
        train_tracker, val_tracker = self._build_trackers(loss_fn=loss_fn)
        return (
            model,
            train_dataloader,
            val_dataloader,
            loss_fn,
            optimizer,
            lr_scheduler,
            train_tracker,
            val_tracker,
        )

    def _with_wandb(self) -> WandBRun:
        wandb_run_name: str = self.config.wandb.run_name or self.config.name
        return WandBRun(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            id=self.config.wandb.run_id,
            name=wandb_run_name,
            config=self.config.model_dump(mode="json"),
        )

    def train(self):
        ctx: Any = (
            nullcontext(None) if self.config.wandb is None else self._with_wandb()
        )
        with ctx as wandb_run:
            run_id: str = (
                wandb_run.run.id if wandb_run is not None else self.config.name
            )
            callbacks: list = self._get_callbacks(run_id=run_id)
            if wandb_run is not None:
                callbacks.append(wandb_run)

            (
                model,
                train_dataloader,
                val_dataloader,
                loss_fn,
                optimizer,
                lr_scheduler,
                train_tracker,
                val_tracker,
            ) = self._build_modules()
            train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_tracker=train_tracker,
                val_tracker=val_tracker,
                epochs=self.config.epochs,
                device=self.config.device,
                lr_scheduler=lr_scheduler,
                callbacks=callbacks,
            )

    def _sweep(self):
        with WandBRun() as wandb_run:
            run_config: dict = dict(wandb_run.run.config)
            wandb_run.run.config.update(
                self.config.model_dump(mode="json"),
                allow_val_change=False,
            )
            callbacks: list = self._get_callbacks(run_id=wandb_run.run.id) + [wandb_run]

            try:
                (
                    model,
                    train_dataloader,
                    val_dataloader,
                    loss_fn,
                    optimizer,
                    lr_scheduler,
                    train_tracker,
                    val_tracker,
                ) = self._build_modules(config=run_config)
            except ValueError:
                print(f"Skipping run due to invalid configuration: {run_config}")
                return

            train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_tracker=train_tracker,
                val_tracker=val_tracker,
                epochs=self.config.epochs,
                device=self.config.device,
                lr_scheduler=lr_scheduler,
                callbacks=callbacks,
            )

    def sweep(self):
        if self.config.sweep is None:
            raise ValueError(
                "Sweep config is not set. Please provide a sweep config to run a hyperparameter sweep."
            )
        if self.config.wandb is None:
            raise ValueError(
                "WandB config is not set. Please provide a WandB config to run a hyperparameter sweep."
            )
        sweep: dict = self.config.sweep.to_wandb_sweep()
        sweep["name"] = self.config.name
        sweep["project"] = self.config.wandb.project
        sweep["entity"] = self.config.wandb.entity

        sweep_id: str = wandb.sweep(
            sweep=sweep,  # TODO: Make sure this method converts the SweepConfig to a valid WandB sweep config dictionary.
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
        )
        wandb.agent(
            sweep_id=sweep_id,
            function=self._sweep,
            count=self.config.sweep.num_runs,
        )


# class InferenceManager:
#     def __init__(self, model_config: ModelConfig, device: torch.device):
#         self.model_manager: ModelManager = ModelManager(
#             model_config=model_config, device=device
#         )

#     def load_model(self, checkpoint_path: Path) -> torch.nn.Module:
#         """Load a model from a checkpoint."""
#         checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
#         model_config: dict = checkpoint["model_config"]
#         model: torch.nn.Module = self.model_manager.from_dict(model_config)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         return model
