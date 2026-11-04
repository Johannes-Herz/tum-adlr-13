from collections.abc import Callable
from pathlib import Path

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import rectangle as rectangle_sdf


def generate(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate occupancy grids of n randomly sized, rotated and placed rectangles.

    Widths and heights are sampled uniformly in [0, 2], angles uniformly in
    [0, pi/2). Rectangles whose rotated bounding box does not fit inside
    [-1, 1]^2 are resampled. Each center is then sampled uniformly within the
    range that keeps its rectangle fully inside the grid.

    Args:
        n: Number of rectangles to generate.
        resolution: Number of cells per axis.
        rng: Random number generator used for all sampling.

    Returns:
        A tuple (occupancy_grids, sdf_grids, sizes, centers, angles) with shapes
        (n, resolution, resolution), (n, resolution, resolution), (n, 2),
        (n, 2) and (n,). Sizes are full width and height, angles are in
        radians, counterclockwise.
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
    parser.add_argument("--seed", type=int, default=42)

    args: Namespace = parser.parse_args()

    (
        occupancy_grids,
        sdf_grids,
        sizes,
        centers,
        angles,
    ) = generate(
        n=args.number_of_samples,
        resolution=args.resolution,
        rng=np.random.default_rng(seed=args.seed),
    )
    filepath: Path = save(
        shape="rectangle",
        occupancy_grids=occupancy_grids,
        sdf_grids=sdf_grids,
        resolution=args.resolution,
        seed=args.seed,
        sizes=sizes,
        centers=centers,
        angles=angles,
    )
    print(filepath)
