import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from wandb.sdk.lib.runid import generate_id

import wandb
from tactex.paths import category_dir, res_dir, versioned_dir, new_versioned_dir
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

def model_output_dir(config: dict, models_root: Path, run_name: str, run_id: str) -> Path:
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
        default=Path(__file__).resolve().parents[1] / "config" / "train_unet.yaml",
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
    models_root = settings.models_root‚
    dataset_path, metadata = input_dataset(config["dataset"])
    output_dir = model_output_dir(config=config, models_root=models_root, run_name=config["run_name"], run_id=run_id)


if __name__ == "__main__":
    main()
