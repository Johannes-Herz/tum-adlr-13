from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tactex.settings import get_settings

TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def circle(
    radii: np.ndarray, centers: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batched signed distance function for n circles.

    Args:
        radii: Shape (n,), one radius per circle.
        centers: Shape (n, 2), one center per circle.

    Returns:
        A function mapping points of shape (p, 2) to signed distances of
        shape (n, p): negative inside, zero on the boundary, positive outside.
    """

    def sdf(points: np.ndarray) -> np.ndarray:
        offsets: np.ndarray = points[None, :, :] - centers[:, None, :]
        distances: np.ndarray = np.linalg.norm(offsets, axis=-1)
        return distances - radii[:, None]

    return sdf


def make_coords(resolution: int) -> np.ndarray:
    """Generate the cell-center coordinates of a regular 2D grid over [-1, 1]^2.

    The grid is split into `resolution` equal cells per axis, and one point is
    placed at the center of each cell. Points are ordered with `indexing="ij"`,
    so a reshaped grid has x along axis 0 and y along axis 1.

    Args:
        resolution: Number of cells per axis.

    Returns:
        Points of shape (resolution**2, 2).
    """
    linspace: np.ndarray = np.linspace(-1, 1, resolution + 1)
    steps: np.ndarray = (linspace[1:] + linspace[:-1]) / 2
    grid: np.ndarray = np.meshgrid(*([steps] * 2), indexing="ij")
    points: np.ndarray = np.stack(grid, axis=-1).reshape(-1, 2)
    return points


def rasterize(sdf: Callable[[np.ndarray], np.ndarray], resolution: int) -> np.ndarray:
    """Sample a signed distance function on a regular 2D grid over [-1, 1]^2.

    Evaluates `sdf` at every cell center and reshapes the result into a grid.
    Any leading batch axes returned by `sdf` are kept.

    Args:
        sdf: Signed distance function mapping points of shape (p, 2) to
            distances of shape (..., p).
        resolution: Number of cells per axis.

    Returns:
        Signed distances of shape (..., resolution, resolution), indexed as
        [x, y]. Transpose the last two axes for image-style [row, col] display.
    """
    points: np.ndarray = make_coords(resolution=resolution)
    sdf_values: np.ndarray = sdf(points)
    return sdf_values.reshape(*sdf_values.shape[:-1], *([resolution] * 2))


def sdf_to_occupancy(
    sdf_grids: np.ndarray,
) -> np.ndarray:
    """Convert a signed distance grid into a binary occupancy grid.

    Uses the center rule: a cell is occupied if the signed distance at its
    center is less than or equal to zero.

    Args:
        sdf_grids: Signed distances of any shape.

    Returns:
        Occupancy as int8 of the same shape as `sdf_grids`: 1 = occupied,
        0 = free.
    """
    return (sdf_grids <= 0).astype(np.int8)


def generate(
    n: int, resolution: int
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
    radii: np.ndarray = np.linspace(0, 1, n + 1)[1:]
    centers: np.ndarray = np.zeros(shape=(n, 2))
    circle_sdf: Callable[[np.ndarray], np.ndarray] = circle(
        radii=radii, centers=centers
    )
    sdf_grids: np.ndarray = rasterize(sdf=circle_sdf, resolution=resolution)
    return sdf_to_occupancy(sdf_grids=sdf_grids), sdf_grids, radii, centers


def save(
    occupancy_grids: np.ndarray, resolution: int, radii: np.ndarray, centers: np.ndarray
) -> Path:
    """Save occupancy grids and their parameters to a timestamped .npz file.

    The file is written to
    `<data_root>/processed/occupancy_grid/simple/circle/<R>x<R>/<timestamp>.npz`,
    where the timestamp is the current UTC time. Missing directories are
    created. Grids are stored as int8 (1 = occupied, 0 = free).

    Args:
        occupancy_grids: Occupancy grids of shape (n, resolution, resolution).
        resolution: Number of cells per axis.
        radii: Circle radii of shape (n,).
        centers: Circle centers of shape (n, 2).

    Returns:
        Path to the saved file.
    """
    filename: str = f"{datetime.now(tz=UTC).strftime(format=TIMESTAMP_FORMAT)}.npz"
    filepath: Path = (
        get_settings().data_root
        / "processed"
        / "occupancy_grid"
        / "simple"
        / "circle"
        / f"{resolution}x{resolution}"
        / filename
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=filepath,
        arr=occupancy_grids.astype(np.int8),
        resolution=resolution,
        radii=radii,
        centers=centers,
    )
    return filepath


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--number-of-samples", type=int, required=True)

    args: Namespace = parser.parse_args()

    occupancy_grids, _, radii, centers = generate(
        n=args.number_of_samples, resolution=args.resolution
    )
    filepath: Path = save(
        occupancy_grids=occupancy_grids,
        resolution=args.resolution,
        radii=radii,
        centers=centers,
    )
    print(filepath)
