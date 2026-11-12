import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tactex.paths import category_dir, new_versioned_dir, res_dir

TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def _to_json(value: Any):
    """Convert NumPy arrays and scalars to JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_shapenet(
    masks: np.ndarray,
    config: dict,
) -> Path:
    category_path = category_dir(
        source="shapenet",
        category=config["category"],
        full_grid=True,
        extracted=False,
    )
    res_path = res_dir(base_path=category_path, resolution=config["grid_size"])
    output_path = new_versioned_dir(base_path=res_path)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "occupancy_grid.npz", masks=masks)

    generation_params = config.copy()
    generation_params.pop("data_root", None)
    generation_params.pop("extracted_dir", None)
    generation_params.pop("max_objects", None)
    if config["specific_object_id"] != None:
        generation_params["all_objects"] = False

    metadata = {
        **generation_params,
        "version": output_path.name,
        "num_objects": len(masks) / config["samples_per_object"],
        "number_of_samples": len(masks),
        "created": datetime.now(tz=UTC).strftime(TIMESTAMP_FORMAT),
    }

    with (output_path / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(obj=metadata, fp=file, indent=2, default=_to_json)
        file.write("\n")
    return output_path
