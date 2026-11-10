import argparse
from pathlib import Path

import numpy as np
import yaml

from tactex.settings import get_settings
from tactex.shapes.mesh.grid_dataset import generate_grid_dataset


def main() -> None:
    """Load generation settings from a YAML file and generate occupancy grids."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "grid_dataset.yaml",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(
            "Config must contain a YAML mapping of parameter names to values."
        )

    data_root = config.get("data_root")
    extracted_dir = config.get("extracted_dir")
    rng_config = config["rng"]
    output_path = generate_grid_dataset(
        data_root=Path(data_root)
        if data_root is not None
        else get_settings().data_root,
        category=config["category"],
        extracted_dir=Path(extracted_dir) if extracted_dir is not None else None,
        specific_object_id=config.get("specific_object_id"),
        max_objects=config["max_objects"],
        all_objects=config["all_objects"],
        grid_size=config["grid_size"],
        samples_per_object=config["samples_per_object"],
        rng=np.random.default_rng(rng_config["seed"])
        if rng_config is not None
        else None,
        scale_range=tuple(config["scale_range"]),
        padding=config["padding"],
    )
    print(f"Saved occupancy grids to {output_path}")


if __name__ == "__main__":
    main()
