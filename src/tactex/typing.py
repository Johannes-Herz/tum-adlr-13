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
