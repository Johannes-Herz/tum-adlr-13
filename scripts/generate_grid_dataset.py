import argparse
from pathlib import Path

import yaml

from tactex.shapes.mesh.config import DatasetShapenetConfig
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

    output_path = generate_grid_dataset(config=DatasetShapenetConfig(**config))
    print(f"Saved occupancy grids to {output_path}")


if __name__ == "__main__":
    main()
