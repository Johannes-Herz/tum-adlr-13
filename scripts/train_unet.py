import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim
from torchmetrics import MetricCollection
from wandb.sdk.lib.runid import generate_id

import wandb
from tactex.paths import category_dir, res_dir, versioned_dir
from tactex.reconstruction.callbacks import Checkpointer, WandBRun
from tactex.reconstruction.data import OccupancyGridDatset, build_dataloader
from tactex.reconstruction.losses import BCEDiceLoss
from tactex.reconstruction.metrics import build_metrics, build_trackers
from tactex.reconstruction.split import save_split_indices, split_indices
from tactex.reconstruction.training import train
from tactex.reconstruction.unet import UNet
from tactex.settings import get_settings


def input_dataset(data_config: dict) -> tuple[Path, dict]:
    input_dir = category_dir(
        data_config["source"], data_config["category"], full_grid=True, extracted=False
    )
    input_dir = res_dir(input_dir, data_config["resolution"])
    input_dir = versioned_dir(input_dir, data_config["version"])

    dataset_path = input_dir / "occupancy_grid.npy"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_path} does not exist")

    metadata_path = input_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata for dataset is missing. {metadata_path} does not exist"
        )
    with (metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)

    return dataset_path, metadata


def model_output_dir(
    config: dict, models_root: Path, run_name: str, run_id: str
) -> Path:
    """
    Save trained models in a file structure analogues to the input dataset
    """
    output_dir = models_root / config["source"] / config["category"]
    output_dir = res_dir(output_dir, config["resolution"])
    output_dir = versioned_dir(output_dir, config["version"])
    # output_dir = new_versioned_dir(output_dir / "unet")
    output_dir = output_dir / f"{run_name}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "train_unet.yaml",
    )
    parser.add_argument(
        "--resume", type=Path, help="Pfad zu latest checkpoint (.pt file)."
    )
    ### change this so path does not have to be provided, find correct path later on its on,
    ### based on config and where the model will be saved
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config ust be yaml mapping.")

    settings = get_settings()

    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    device = torch.device(device_name)
    print(f"Run on device: {device}")

    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)

    wandb.login(key=settings.wandb_key.get_secret_value())

    data_root = settings.data_root
    models_root = settings.models_root
    dataset_path, metadata = input_dataset(config["dataset"])

    if checkpoint is None:
        wandb_run_id = generate_id()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + wandb_run_id
        output_dir = model_output_dir(
            config=config["dataset"],
            models_root=models_root,
            run_name=config["run_name"],
            run_id=run_id,
        )
    else:
        wandb_run_id = checkpoint["wandb_run_id"]
        run_id = checkpoint["run_id"]
        output_dir = args.resume.resolve().parents[1]  ### Check correct path again

    checkpoint_dir = output_dir / f"checkpoints_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        splits = split_indices(
            grid_count=metadata["number_of_samples"],
            samples_per_object=metadata["samples_per_object"],
            num_objects=metadata["num_objects"],
            val_fraction=config["val_fraction"],
            test_fraction=config["test_fraction"],
            seed=config["seed"],
        )
        save_split_indices(output_dir, splits)
    else:
        with np.load(output_dir / "split_indices.npz") as saved_splits:
            splits = {name: saved_splits[name] for name in ("train", "val", "test")}

    train_dataset = OccupancyGridDatset(
        root=dataset_path.parent,
        split="train",
        in_memory=False,
        limit=None,
        filename="occupancy_grid",
        indices=splits["train"],
    )
    val_dataset = OccupancyGridDatset(
        root=dataset_path.parent,
        split="val",
        in_memory=False,
        limit=None,
        filename="occupancy_grid",
        indices=splits["val"],
    )

    ray_count_upper_bound = config["max_rays"] + 1
    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        resolution=config["dataset"]["resolution"],
        max_rays=ray_count_upper_bound,
        name="train",
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    val_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        resolution=config["dataset"]["resolution"],
        max_rays=ray_count_upper_bound,
        name="validation",
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    unet: UNet = UNet(
        in_channels=3,
        out_channels=1,
        widths=config["widths"],
        kernel_size=config["kernel_size"],
    ).to(device)

    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        BCEDiceLoss()
    )  # nn.BCEWithLogitsLoss()

    optimizer: optim.Adam = optim.Adam(
        params=unet.parameters(), lr=config["learning_rate"]
    )
    metrics: MetricCollection = build_metrics(loss_fn=loss_fn, threshold=0.5)
    train_tracker, val_tracker = build_trackers(metrics=metrics, device=device)
    lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
        )
    )

    epochs = config["epochs"]
    print("Start training")

    completed_epochs = 0
    if checkpoint is not None:
        unet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        completed_epochs = checkpoint["epoch"]

    remaining_epochs = config["epochs"] - completed_epochs

    checkpoint_callback: Checkpointer = Checkpointer(
        save_dir=checkpoint_dir,
        every_n_epochs=1,
        metric="val/iou_occ",
        mode="max",
        save_to_wandb=True,
        run_metadata={
            "wandb_run_id": wandb_run_id,
            "run_id": run_id,
            "config": config,
        },
    )

    if checkpoint is not None:
        checkpoint_callback.best_metric = checkpoint.get("best_metric")

    with WandBRun(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        name=config["run_name"],
        id=wandb_run_id,
        resume="must" if checkpoint is not None else None,
        config=config if checkpoint is None else None,
        dir=str(output_dir),
    ) as wandb_run:
        val_batches = list(val_dataloader)
        train(
            model=unet,
            train_dataloader=train_dataloader,
            val_dataloader=val_batches,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_tracker=train_tracker,
            val_tracker=val_tracker,
            epochs=remaining_epochs,
            device=device,
            lr_scheduler=lr_scheduler,
            callbacks=[checkpoint_callback, wandb_run],
            start_epoch=completed_epochs,
        )
        torch.save(unet.state_dict(), output_dir / "model.pt")
    print(f"Model saved under {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
