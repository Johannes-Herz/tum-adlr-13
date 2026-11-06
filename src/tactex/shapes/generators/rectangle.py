from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import rectangle as rectangle_sdf


def generate(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate n randomly sized, rotated and placed rectangles as SDF and occupancy grids.

    Widths and heights are sampled uniformly in [0, 2], angles uniformly in
    [0, pi/2). Rectangles whose rotated bounding box does not fit inside
    [-1, 1]^2 are resampled, for at most 25 rounds. Each center is then
    sampled uniformly within the range that keeps its rectangle fully inside
    the grid.

    Args:
        n: Number of rectangles to generate.
        resolution: Number of cells per axis.
        rng: Random number generator used for all sampling.

    Returns:
        A tuple (occupancy_grids, sdf_grids, params):
        - occupancy_grids: array of shape (n, resolution, resolution).
        - sdf_grids: array of shape (n, resolution, resolution).
        - params: dict with "sizes" of shape (n, 2) (full width and height),
          "centers" of shape (n, 2) and "angles" of shape (n,) in radians,
          counterclockwise.
    """
    sizes: np.ndarray = rng.uniform(low=0, high=2, size=(n, 2))
    angles: np.ndarray = rng.uniform(low=0, high=np.pi / 2, size=n)
    MAX_RETRIES: int = 25
    for _ in range(MAX_RETRIES):
        half_sizes: np.ndarray = sizes / 2
        c, s = np.abs(np.cos(angles)), np.abs(np.sin(angles))
        ex, ey = (
            c * half_sizes[:, 0] + s * half_sizes[:, 1],
            s * half_sizes[:, 0] + c * half_sizes[:, 1],
        )
        bad: np.ndarray = (ex > 1) | (ey > 1)
        if not bad.any():
            break
        k: int = bad.sum()
        print(f"Resampling for {k} bad combinations")
        sizes[bad] = rng.uniform(low=0, high=2, size=(k, 2))
        angles[bad] = rng.uniform(low=0, high=np.pi / 2, size=(k))

    cx, cy = rng.uniform(-1 + ex, 1 - ex), rng.uniform(-1 + ey, 1 - ey)
    centers: np.ndarray = np.stack([cx, cy], axis=-1)

    sdf: Callable[[np.ndarray], np.ndarray] = rectangle_sdf(
        sizes=sizes, centers=centers, angles=angles
    )
    sdf_grids: np.ndarray = rasterize(sdf=sdf, resolution=resolution)
    occupancy_grids: np.ndarray = sdf_to_occupancy(sdf_grids=sdf_grids)
    params: dict[str, Any] = {"sizes": sizes, "centers": centers, "angles": angles}
    return (occupancy_grids, sdf_grids, params)
