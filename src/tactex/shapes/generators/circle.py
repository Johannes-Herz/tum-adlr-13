from collections.abc import Callable
from pathlib import Path

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import circle as circle_sdf


def generate(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate occupancy grids of n circles centered at the origin.

    Radii are evenly spaced in (0, 1], i.e. 1/n, 2/n, ..., 1, so every circle
    fits inside [-1, 1]^2.

    Args:
        n: Number of circles to generate.
        resolution: Number of cells per axis.

    Returns:
        A tuple (occupancy_grids, sdf_grids, radii, centers) with shapes
        (n, resolution, resolution), (n, resolution, resolution), (n,)
        and (n, 2).
    """
    radii: np.ndarray = rng.uniform(low=0, high=1, size=n)
    centers: np.ndarray = rng.uniform(
        low=-1 + radii[:, None], high=1 - radii[:, None], size=(n, 2)
    )
    sdf: Callable[[np.ndarray], np.ndarray] = circle_sdf(radii=radii, centers=centers)
    sdf_grids: np.ndarray = rasterize(sdf=sdf, resolution=resolution)
    return sdf_to_occupancy(sdf_grids=sdf_grids), sdf_grids, radii, centers


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--number-of-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args: Namespace = parser.parse_args()

    occupancy_grids, sdf_grids, radii, centers = generate(
        n=args.number_of_samples,
        resolution=args.resolution,
        rng=np.random.default_rng(seed=args.seed),
    )
    filepath: Path = save(
        shape="circle",
        occupancy_grids=occupancy_grids,
        sdf_grids=sdf_grids,
        resolution=args.resolution,
        seed=args.seed,
        radii=radii,
        centers=centers,
    )
    print(filepath)
