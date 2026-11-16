import numpy as np

from tactex.shapes.primitives.config import GridConfig
from tactex.typing import OccupancyGrid2D, Point2D, SDFGrid2D, SignedDistanceFunction


def _make_coords(grid_config: GridConfig) -> Point2D:
    """Generate the cell-center coordinates of a regular 2D grid over [-1, 1]^2.

    The grid is split into `resolution` equal cells per axis, and one point is
    placed at the center of each cell. Points are ordered with `indexing="ij"`,
    so a reshaped grid has x along axis 0 and y along axis 1.

    Args:
        resolution: Number of cells per axis.

    Returns:
        Points of shape (resolution**2, 2).
    """
    linspace: np.ndarray = np.linspace(
        grid_config.min_coord, grid_config.max_coord, grid_config.resolution + 1
    )
    steps: np.ndarray = (linspace[1:] + linspace[:-1]) / 2
    grid: np.ndarray = np.meshgrid(*([steps] * 2), indexing="ij")
    points: Point2D = np.stack(grid, axis=-1).reshape(-1, 2)
    return points


def rasterize(sdf: SignedDistanceFunction, grid_config: GridConfig) -> SDFGrid2D:
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
    points: Point2D = _make_coords(grid_config=grid_config)
    sdf_values: np.ndarray = sdf(points)
    return sdf_values.reshape(*sdf_values.shape[:-1], *([grid_config.resolution] * 2))


def sdf_to_occupancy(
    sdf_grids: SDFGrid2D,
) -> OccupancyGrid2D:
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
