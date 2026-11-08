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
    log_fn: Callable[[dict[str, Any, int]], None] | None = None,
) -> None:
    """Train a model and track training and validation metrics per epoch.

    Each epoch runs one pass over ``train_dataloader`` (forward, loss, backward,
    optimizer step), then evaluates on ``val_dataloader``. Metrics are recorded
    in the given trackers, which are modified in place. Calling ``train`` again
    with the same trackers continues the history: epoch numbering starts at the
    trackers' current step count.

    Args:
        model: Model to train. Moved to ``device``; its parameters must be the
            ones registered in ``optimizer``.
        train_dataloader: Yields ``(x, y)`` batches for training.
        val_dataloader: Yields ``(x, y)`` batches for validation.
        loss_fn: Loss on ``(logits, targets)``, used for backpropagation.
            Targets are passed as float tensors.
        optimizer: Optimizer over ``model.parameters()``.
        train_tracker: Tracks metrics on the training batches. Receives detached
            logits and integer targets; one step per epoch.
        val_tracker: Tracks metrics on the validation set; one step per epoch.
        epochs: Number of epochs to train in this call.
        device: Device for the model, the trackers, and all batches.
        lr_scheduler: Optional scheduler, stepped once per epoch.
        log_fn: Optional callback called once per epoch as
            ``log_fn(results, epoch)``, where ``results`` maps metric names
            (e.g. ``"train/loss"``, ``"val/iou_occ"``) to floats. Use it to
            forward results to a logger such as wandb.

    Returns:
        None. Results are available through ``train_tracker`` and ``val_tracker``
        (e.g. ``compute_all()``, ``best_metric()``).
    """
    model.to(device)
    train_tracker.to(device)
    val_tracker.to(device)

    start: int = train_tracker.n_steps
    epoch_tqdm: tqdm = tqdm(
        range(start, start + epochs), leave=True, desc="Training", unit="epoch"
    )
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
            log_fn(results, epoch)
