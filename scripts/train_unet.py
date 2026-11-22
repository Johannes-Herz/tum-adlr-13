import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torchmetrics import MetricCollection
from wandb.sdk.lib.runid import generate_id

import wandb
from tactex.paths import category_dir, res_dir, versioned_dir
from tactex.reconstruction.data import OccupancyGridDatset, build_dataloader
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
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config ust be yaml mapping.")

    settings = get_settings()

    wandb.login(key=settings.wandb_key.get_secret_value())

    wandb_run_id = generate_id()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + wandb_run_id

    data_root = settings.data_root
    models_root = settings.models_root
    dataset_path, metadata = input_dataset(config["dataset"])
    output_dir = model_output_dir(
        config=config["dataset"],
        models_root=models_root,
        run_name=config["run_name"],
        run_id=run_id,
    )
    checkpoint_dir = output_dir / "checkpoints_" + f"{run_id}"
    checkpoint_dir.mkdir(parents=True, exists_ok=True)

    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    device = torch.device(device_name)

    splits = split_indices(
        grid_count=metadata["number_of_samples"],
        samples_per_object=metadata["samples_per_object"],
        num_objects=metadata["num_objects"],
        val_fraction=config["val_fraction"],
        test_fraction=config["test_fraction"],
        seed=config["seed"],
    )
    save_split_indices(output_dir, splits)

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
        nn.BCEWithLogitsLoss()
    )
    optimizer: optim.Adam = optim.Adam(
        params=unet.parameters(), lr=config["learning_rate"]
    )
    metrics: MetricCollection = build_metrics(loss_fn=loss_fn, threshold=0.5)
    train_tracker, val_tracker = build_trackers(metrics=metrics, device=device)
    learning_rate_scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
        )
    )

    epochs = config["epochs"]

    with wandb.init(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        name=config["run_name"],
        id=wandb_run_id,
        resume="allow",
        config=config,
        dir=str(output_dir),
    ) as run:
        wandb_callback = lambda results, _: run.log(results)
        train(
            model=unet,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_tracker=train_tracker,
            val_tracker=val_tracker,
            epochs=config["epochs"],
            device=device,
            callback=wandb_callback,
        )
        torch.save(unet.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    main()
