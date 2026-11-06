from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import circle as circle_sdf


def generate(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate n random circles inside [-1, 1]^2 as SDF and occupancy grids.

    Radii are drawn uniformly from [min_radius, max_radius]. Each center is
    drawn uniformly such that the whole circle lies inside [-1, 1]^2.

    Args:
        n: Number of circles to generate.
        resolution: Number of cells per axis.
        rng: Random number generator used for all sampling.
        min_radius: Smallest allowed radius. Should be well above the cell
            size (2 / resolution) so no circle rasterizes to an empty grid.
        max_radius: Largest allowed radius, at most 1.

    Returns:
        A tuple (occupancy_grids, sdf_grids, params):
        - occupancy_grids: bool array of shape (n, resolution, resolution).
        - sdf_grids: float array of shape (n, resolution, resolution).
        - params: dict with "radii" of shape (n,) and "centers" of shape (n, 2).
    """
    radii: np.ndarray = rng.uniform(low=0, high=1, size=n)
    centers: np.ndarray = rng.uniform(
        low=-1 + radii[:, None], high=1 - radii[:, None], size=(n, 2)
    )
    sdf: Callable[[np.ndarray], np.ndarray] = circle_sdf(radii=radii, centers=centers)
    sdf_grids: np.ndarray = rasterize(sdf=sdf, resolution=resolution)
    occupancy_grids: np.ndarray = sdf_to_occupancy(sdf_grids=sdf_grids)
    params: dict[str, Any] = {"radii": radii, "centers": centers}
    return occupancy_grids, sdf_grids, params
