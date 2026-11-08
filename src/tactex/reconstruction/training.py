from collections.abc import Callable
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MetricTracker
from tqdm.auto import tqdm

from tactex.reconstruction.evaluation import evaluate


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    train_tracker: MetricTracker,
    val_tracker: MetricTracker,
    epochs: int,
    device: torch.device,
    lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    model.to(device)
    train_tracker.to(device)
    val_tracker.to(device)

    epoch_tqdm: tqdm = tqdm(range(epochs), leave=True, desc="Training", unit="epoch")
    for epoch in epoch_tqdm:
        # --- Training ---
        model.train()
        train_tracker.increment()

        batch_tqdm: tqdm = tqdm(
            train_dataloader, desc=f"Epoch {epoch}", unit="batch", leave=False
        )
        for x_batch, y_batch in batch_tqdm:
            x_batch, y_batch = (
                x_batch.to(device, torch.float32),
                y_batch.to(device, torch.float32),
            )

            logit_batch: torch.Tensor = model(x_batch)
            loss = loss_fn(logit_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_tracker.update(logit_batch.detach(), y_batch.int())
            batch_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        # --- Evaluation ---
        val_tracker.increment()
        evaluate(model, val_dataloader, val_tracker, device)

        # ---- Report ----
        results: dict[str, Any] = {
            k: v.item() for k, v in train_tracker.compute().items()
        } | {k: v.item() for k, v in val_tracker.compute().items()}
        epoch_tqdm.set_postfix(
            {
                k: f"{v:.4f}"
                for k, v in results.items()
                if k in ("train/loss", "val/loss", "val/iou_occ")
            }
        )
        if log_fn is not None:
            log_fn(results)
