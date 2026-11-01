from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tactex.settings import get_settings

TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def save(
    shape: str,
    occupancy_grids: np.ndarray,
    sdf_grids: np.ndarray,
    resolution: int,
    **kwargs,
) -> Path:
    """Save occupancy grids and their parameters to a timestamped .npz file.

    The file is written to
    `<data_root>/processed/occupancy_grid/<shape>/<R>x<R>/<timestamp>.npz`,
    where the timestamp is the current UTC time. Missing directories are
    created. Grids are stored under the key `arr` as int8 (1 = occupied,
    0 = free).

    Args:
        occupancy_grids: Occupancy grids of shape (n, resolution, resolution).
        shape: Name of the shape type, used as folder name (e.g. "circle").
        resolution: Number of cells per axis.
        **kwargs: Shape parameters to store alongside the grids, e.g.
            `radii=...` and `centers=...` for circles. Each is saved under
            its keyword as key.

    Returns:
        Path to the saved file.
    """
    filename: str = f"{datetime.now(tz=UTC).strftime(format=TIMESTAMP_FORMAT)}.npz"
    filepath: Path = (
        get_settings().data_root
        / "processed"
        / "occupancy_grid"
        / shape
        / f"{resolution}x{resolution}"
        / filename
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=filepath,
        occupancy_grids=occupancy_grids.astype(np.int8),
        sdf_grids=sdf_grids,
        resolution=resolution,
        **kwargs,
    )
    return filepath
