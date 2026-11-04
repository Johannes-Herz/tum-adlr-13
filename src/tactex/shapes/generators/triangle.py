from collections.abc import Callable
from pathlib import Path

import numpy as np

from tactex.shapes.grid import rasterize, sdf_to_occupancy
from tactex.shapes.ioop import save
from tactex.shapes.sdf import triangle as triangle_sdf


def generate(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate occupancy grids of n random triangles.

    Each vertex is sampled uniformly in [-1, 1]^2, so every triangle lies
    fully inside the grid. Degenerate triangles are resampled: those with an
    area below 10 grid cells, or a smallest height (distance from a vertex to
    its opposite edge) below 3 grid cells, which catches thin slivers.

    Args:
        n: Number of triangles to generate.
        resolution: Number of cells per axis.
        rng: Random number generator used for all sampling.

    Returns:
        A tuple (occupancy_grids, sdf_grids, vertices) with shapes
        (n, resolution, resolution), (n, resolution, resolution) and
        (n, 3, 2).
    """
    vertices: np.ndarray = rng.uniform(low=-1, high=1, size=(n, 3, 2))
    MAX_RETRIES: int = 25
    MIN_AREA: float = 10 * ((2 / resolution) * (2 / resolution))
    MIN_HEIGHT: float = 3 * (2 / resolution)
    for _ in range(MAX_RETRIES):
        ab, ac = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        areas: np.ndarray = (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2.0
        edges: np.ndarray = np.roll(vertices, -1, axis=1) - vertices
        edge_lengths: np.ndarray = np.linalg.norm(edges, axis=-1)
        heights = 2 * np.abs(areas)[:, None] / edge_lengths
        min_heights = heights.min(axis=1)

        bad: np.ndarray = (np.abs(areas) < MIN_AREA) | (min_heights < MIN_HEIGHT)
        if not bad.any():
            break
        k: int = bad.sum()
        print(f"Resampling for {k} bad combinations")
        vertices[bad] = rng.uniform(low=-1, high=1, size=(k, 3, 2))

    sdf: Callable[[np.ndarray], np.ndarray] = triangle_sdf(vertices=vertices)
    sdf_grids: np.ndarray = rasterize(sdf=sdf, resolution=resolution)
    return (sdf_to_occupancy(sdf_grids=sdf_grids), sdf_grids, vertices)


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--number-of-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args: Namespace = parser.parse_args()

    (occupancy_grids, sdf_grids, vertices) = generate(
        n=args.number_of_samples,
        resolution=args.resolution,
        rng=np.random.default_rng(seed=args.seed),
    )
    filepath: Path = save(
        shape="triangle",
        occupancy_grids=occupancy_grids,
        sdf_grids=sdf_grids,
        resolution=args.resolution,
        vertices=vertices,
    )
    print(filepath)
