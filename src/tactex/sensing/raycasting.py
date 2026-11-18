import numpy as np

from tactex.timings import timed
from tactex.typing import OccupancyGrid2D, Ray2D, Traversal2D, TraversalGrid2D


@timed()
def sample(resolution: int, b: int, n: int, rng: np.random.Generator) -> Ray2D:
    """Sample random 2D rays between points on the grid's circumcircle.

    For each of the `b` batch elements and `n` rays, two angles are sampled
    independently and uniformly from `[0, 2*pi)` on a circle of radius
    `(resolution / 2) * sqrt(2)` centered at `(resolution / 2, resolution / 2)`,
    which fully encloses the `[0, resolution]^2` grid.

    Args:
        resolution: Number of grid cells per spatial axis.
        b: Batch size (number of grids).
        n: Number of rays to sample per grid.
        rng: NumPy random number generator.

    Returns:
        Continuous ray endpoints of shape `(b, n, 2, 2)`, where the third axis
        holds `[start, end]` and the last axis holds `(x, y)` coordinates.
    """
    angles: np.ndarray = rng.uniform(low=0, high=1, size=(b, n, 2)) * 2 * np.pi
    radius: int = float(resolution / 2 * np.sqrt(2))
    center: np.ndarray = np.array([resolution / 2, resolution / 2])
    start: np.ndarray = (
        np.stack(
            [
                radius * np.cos(angles[..., 0]),
                radius * np.sin(angles[..., 0]),
            ],
            axis=-1,
        )
        + center
    )
    end: np.ndarray = (
        np.stack(
            [
                radius * np.cos(angles[..., 1]),
                radius * np.sin(angles[..., 1]),
            ],
            axis=-1,
        )
        + center
    )
    rays: Ray2D = np.stack([start, end], axis=-2)
    return rays


@timed()
def traverse(rays: Ray2D, resolution: int) -> Traversal2D:
    """Rasterize continuous 2D rays into ordered integer grid cell indices.

    Uniformly samples `3 * resolution` continuous points along each ray segment
    and floors them to integer cell coordinates `(ix, iy)`. To prevent diagonal
    corner-skipping between consecutive steps, an orthogonal bridge cell
    (`x`-first or `y`-first) is selected per step based on which integer grid
    boundary the continuous ray crosses first.

    Args:
        rays: Continuous ray endpoints of shape `(B, N, 2, 2)`.
        resolution: Number of grid cells per spatial axis.

    Returns:
        Integer grid cell coordinates of shape `(B, N, M, 2)` traversed by
        each ray.
    """
    num_steps: int = 3 * resolution
    ray_points: np.ndarray = np.linspace(
        rays[..., 0, :], rays[..., 1, :], num=num_steps, axis=-2
    )

    # Compute box indices from continous ray_points
    indices: np.ndarray = np.floor(ray_points).astype(np.int32)

    # Compute bridge elements for filling unwanted diagonal skips
    bridge_x: np.ndarray = np.stack(
        [indices[..., 1:, 0], indices[..., :-1, 1]], axis=-1
    )
    bridge_y: np.ndarray = np.stack(
        [indices[..., :-1, 0], indices[..., 1:, 1]], axis=-1
    )

    # Figure out which bridge direction we need for which consecutive point pair
    boundary: np.ndarray = np.maximum(indices[..., 1:, :], indices[..., :-1, :])
    distances: np.ndarray = np.abs(boundary - ray_points[..., :-1, :])
    deltas: np.ndarray = np.abs(ray_points[..., 1:, :] - ray_points[..., :-1, :])
    # use_x_first: np.ndarray = (distances[:, :, 0] < distances[:, :, 1]) / np.abs(deltas)[..., None] # Issue with np.abs being 0
    use_x_first: np.ndarray = (
        distances[..., 0] * deltas[..., 1] < distances[..., 1] * deltas[..., 0]
    )  # Fix via cross multiplication

    # Pick bridges
    bridges: np.ndarray = np.where(use_x_first[..., None], bridge_x, bridge_y)

    # Merge initial ray_points and bridges
    # points: np.ndarray = np.unique(np.concat([indices, bridges], axis=-2), axis=-2) # we dont need to dedup!!!
    points: np.ndarray = np.concat([indices, bridges], axis=-2)
    return points


@timed()
def cast(
    occupancy_grids: OccupancyGrid2D, traversals: Traversal2D, num_rays: np.ndarray
) -> TraversalGrid2D:
    """Cast rasterized ray traversals onto binary occupancy grids.

    Evaluates ground-truth occupancy along each active ray's traversal path and
    extracts line-of-sight visibility using a cumulative sum along the ray:
    all in-bounds cells before the first occupied cell are marked as `free`,
    the first occupied cell along each ray is marked as `occupied`, and all
    remaining cells (occluded behind a hit or unvisited by any ray) are marked
    as `unknown`.

    Args:
        occupancy_grids: Ground-truth binary occupancy grids of shape
            `(B, H, W)` with `0` (free) and `1` (occupied).
        traversals: Rasterized ray cell coordinates of shape `(B, N, M, 2)`.
        num_rays: Array of shape `(B, 1)` or `(B,)` specifying how many of the
            `N` rays are active for each batch element.

    Returns:
        Boolean traversal grids of shape `(B, 3, H, W)` with channels
        `[occupied, free, unknown]`.
    """
    B, H, W = occupancy_grids.shape
    _, N, M, _ = traversals.shape

    active_rays: np.ndarray = np.arange(N) < num_rays  # (B, N)
    in_bound_mask: np.ndarray = ((traversals >= 0) & (traversals <= H - 1)).all(
        axis=-1
    )  # (B, N, M)
    bounded_traversals: Traversal2D = traversals.clip(0, H - 1)

    batch_idx: np.ndarray = np.broadcast_to(np.arange(B)[:, None, None], (B, N, M))

    # Compute occupancy values along each ray
    occupancies: np.ndarray = (
        (
            occupancy_grids[
                batch_idx, bounded_traversals[..., 0], bounded_traversals[..., 1]
            ]
        ).astype(np.bool)
        & in_bound_mask
    )  # (B, N, M)

    # Define point indices on the line that are free
    free_indices_mask: np.ndarray = (
        (np.cumsum(occupancies, axis=-1) == 0) & in_bound_mask & active_rays[..., None]
    )  # (B, N, M)

    # Define point indices on the line that are occupied
    occupied_indices_mask: np.ndarray = (
        (np.cumsum(occupancies, axis=-1) == 1)
        & in_bound_mask
        & active_rays[..., None]
        & occupancies
    )  # (B, N, M)

    occupancy_mask: np.ndarray = np.zeros(shape=(B, H, W), dtype=np.bool)
    occupancy_mask[
        batch_idx[occupied_indices_mask],
        bounded_traversals[..., 0][occupied_indices_mask],
        bounded_traversals[..., 1][occupied_indices_mask],
    ] = True
    free_mask: np.ndarray = np.zeros(shape=(B, H, W), dtype=np.bool)
    free_mask[
        batch_idx[free_indices_mask],
        bounded_traversals[..., 0][free_indices_mask],
        bounded_traversals[..., 1][free_indices_mask],
    ] = True
    unknown_mask: np.ndarray = ~(occupancy_mask | free_mask)
    traversal_grids: TraversalGrid2D = np.stack(
        [occupancy_mask, free_mask, unknown_mask], axis=1
    )
    return traversal_grids
