import numpy as np
from numpy.typing import NDArray


def ray_caster(
    ray_origin: np.ndarray, ray_target: np.ndarray, grid: np.ndarray
) -> tuple[tuple[int, int] | None, NDArray[np.bool_]]:
    """Find the first occupied pixel along a ray through a 2D grid.

    The ray starts at the center of the origin pixel and points toward the
    center of the target pixel. It continues beyond the target until it hits an
    occupied pixel or leaves the grid. Coordinates use (row, column), i.e. (y, x), order.
    Ray has no radius (radius of 0).

    Args:
        ray_origin: Integer array of shape (2,) giving the starting pixel.
            Must be within the grid.
        ray_target: Integer array of shape (2,) giving a target pixel
            that defines the ray direction. Must be within the grid.
        grid: Occupancy array of shape (height, width). A value of 1 indicates
            an occupied pixel; all other values are treated as free.

    Returns:
        A tuple containing:
            hit: Coordinates (y, x) of the first occupied pixel, or None if
                the ray leaves the grid without a hit.
            miss_grid: Boolean array with the same shape as grid. True marks
                free pixels traversed before the hit or grid boundary.

    Raises:
        ValueError: If the origin and target are equal or either is outside
            the grid.
    """

    if np.array_equal(ray_origin, ray_target):
        raise ValueError("Ray origin and target have to be different.")

    height, width = grid.shape

    for name, pixel in (("origin", ray_origin), ("target", ray_target)):
        pixel_y, pixel_x = pixel
        if not (0 <= pixel_y < height and 0 <= pixel_x < width):
            raise ValueError(f"Ray {name} must be within the grid.")

    miss_grid = np.zeros(grid.shape, dtype=bool)

    pixel_y, pixel_x = ray_origin

    # Start and aim at pixel centers.
    origin = np.asarray(ray_origin, dtype=float) + 0.5
    target = np.asarray(ray_target, dtype=float) + 0.5

    current_pos_y, current_pos_x = origin

    direction_vector = target - origin
    dy, dx = direction_vector
    direction_x = int(np.sign(dx))
    direction_y = int(np.sign(dy))

    while 0 <= pixel_x < width and 0 <= pixel_y < height:
        if grid[pixel_y, pixel_x] == 1:
            hit = (pixel_y, pixel_x)
            return hit, miss_grid

        miss_grid[pixel_y, pixel_x] = True

        # Compute ray parameters to the next vertical and horizontal boundaries.
        if direction_x > 0:
            t_to_boundary_x = (pixel_x + 1 - current_pos_x) / dx
        elif direction_x < 0:
            t_to_boundary_x = (pixel_x - current_pos_x) / dx
        else:
            t_to_boundary_x = np.inf

        if direction_y > 0:
            t_to_boundary_y = (pixel_y + 1 - current_pos_y) / dy
        elif direction_y < 0:
            t_to_boundary_y = (pixel_y - current_pos_y) / dy
        else:
            t_to_boundary_y = np.inf

        # At a corner, advance both coordinates to the diagonal pixel.
        if t_to_boundary_x > t_to_boundary_y:
            pixel_y += direction_y
            delta_t = t_to_boundary_y
        elif t_to_boundary_x < t_to_boundary_y:
            pixel_x += direction_x
            delta_t = t_to_boundary_x
        else:
            pixel_x += direction_x
            pixel_y += direction_y
            delta_t = t_to_boundary_x

        current_pos_y += delta_t * dy
        current_pos_x += delta_t * dx

    return None, miss_grid
