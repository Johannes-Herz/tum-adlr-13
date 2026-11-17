from collections.abc import Callable

import numpy as np

type OccupancyGrid2D = np.ndarray[tuple[int, int, int], np.dtype[np.int8 | np.bool]]
"""Binary occupancy grids of shape `(B, H, W)` with values `0` (free) and `1` (occupied)."""
type SDFGrid2D = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
"""Signed distance grids of shape `(B, H, W)` with negative values inside, zero on the boundary, and positive outside."""
type Point2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
"""2D point coordinates of shape `(N, 2)` with `(x, y)` pairs."""
type Distance = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
"""Signed or unsigned distances of shape `(N,)` or `(B, N)`."""
type SignedDistanceFunction = Callable[[Point2D], Distance]
"""Callable mapping 2D query points of shape `(P, 2)` to signed distances of shape `(B, P)`."""
type SignedDistanceFunctionBuilder = Callable[..., SignedDistanceFunction]
"""Factory function taking primitive parameters and returning a `SignedDistanceFunction`."""
type TraversalGrid2D = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.int8 | np.bool]
]
"""Traversal grids of shape `(B, 3, H, W)` with channels `[occupied, free, unknown]`."""
type Ray2D = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.float32]
]  # Have (B, N, 2, 2) or (N, 2, 2) ???
"""Continuous 2D rays of shape `(B, N, 2, 2)`, where `B` is the batch size, `N` is the number of rays per grid, the third axis holds `[start, end]` points, and the last axis holds `(x, y)` coordinates."""
type Traversal2D = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.int32]
]  # Have (B, N, M, 2) or (N, M, 2) ???
"""Rasterized ray traversals of shape `(B, N, M, 2)`, where `B` is the batch size, `N` is the number of rays per grid, `M` is the number of ordered step indices along each ray, and the last axis holds `(ix, iy)` grid cell coordinates."""
