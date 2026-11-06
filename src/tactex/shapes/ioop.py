import json
from pathlib import Path
from typing import Literal

import numpy as np

from tactex.settings import get_settings

TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def _to_json(value):
    """Convert NumPy arrays and scalars to JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(
    shape: str,
    split: Literal["train", "eval", "test"],
    occupancy_grids: np.ndarray,
    sdf_grids: np.ndarray,
    resolution: int,
    seed: int | None,
    **kwargs,
) -> Path:
    """Save occupancy grids, SDF grids and metadata for one shape and split.

    Files are written to
    `<data_root>/occupancy_grids/<R>x<R>/<split>/`:

    - `<shape>_occupancy.npy`: bool array of shape (n, R, R),
      True = occupied, False = free.
    - `<shape>_sdf.npy`: float32 array of shape (n, R, R) with the
      signed distance per cell.
    - `<shape>_meta.json`: resolution, seed, number of samples and all
      extra keyword arguments.

    Arrays are stored as uncompressed .npy so they can be memory-mapped
    with `np.load(..., mmap_mode="r")`. Missing directories are created,
    existing files are overwritten.

    Args:
        shape: Name of the shape type, used in the file names (e.g. "circle").
        split: Dataset split the grids belong to.
        occupancy_grids: Occupancy grids of shape (n, R, R).
        sdf_grids: Signed distance grids of shape (n, R, R).
        resolution: Number of cells per axis (R).
        seed: Random seed used to generate the grids, or None.
        **kwargs: Shape parameters stored in the metadata file, e.g.
            `radii=...` and `centers=...` for circles. NumPy arrays and
            scalars are converted to plain Python types.

    Returns:
        Path to the directory containing the saved files.
    """
    base_path: Path = (
        get_settings().data_root
        / "occupancy_grids"
        / f"{resolution}x{resolution}"
        / split
    )
    occupancy_grid_filename: str = f"{shape}_occupancy.npy"
    sdf_grid_filename: str = f"{shape}_sdf.npy"
    metadata_filename: str = f"{shape}_metadata.json"

    base_path.mkdir(parents=True, exist_ok=True)
    np.save(
        file=base_path / occupancy_grid_filename, arr=occupancy_grids.astype(np.bool)
    )
    np.save(file=base_path / sdf_grid_filename, arr=sdf_grids.astype(np.float32))

    metadata = {
        "resolution": resolution,
        "seed": seed,
        "number_of_samples": int(occupancy_grids.shape[0]),
        **{key: _to_json(value) for key, value in kwargs.items()},
    }
    with open(file=base_path / metadata_filename, mode="w") as file:
        json.dump(obj=metadata, fp=file, indent=2)
    return base_path
