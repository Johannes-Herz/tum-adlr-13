from collections.abc import Callable
from pathlib import Path

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import rectangle as rectangle_sdf


def generate(
    n: int, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    sizes: np.ndarray = np.random.uniform(low=0, high=2, size=(n, 2))
    centers: np.ndarray = np.random.uniform(low=0, high=2, size=(n, 2))
    angles: np.ndarray = np.random.uniform(low=0, high=np.pi / 2, size=n)
    sdf: Callable[[np.ndarray], np.ndarray] = rectangle_sdf(
        sizes=sizes, centers=centers, angles=angles
    )
    sdf_grids: np.ndarray = rasterize(sdf=sdf, resolution=resolution)
    return (
        sdf_to_occupancy(sdf_grids=sdf_grids),
        sdf_grids,
        sizes,
        centers,
        angles,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--number-of-samples", type=int, required=True)

    args: Namespace = parser.parse_args()

    (
        occupancy_grids,
        sdf_grids,
        sizes,
        centers,
        angles,
    ) = generate(n=args.number_of_samples, resolution=args.resolution)
    filepath: Path = save(
        shape="rectangle",
        occupancy_grids=occupancy_grids,
        sdf_grids=sdf_grids,
        resolution=args.resolution,
        sizes=sizes,
        centers=centers,
        angles=angles,
    )
    print(filepath)
