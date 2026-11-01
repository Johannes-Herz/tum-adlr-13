from collections.abc import Callable

import numpy as np


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


def rectangle(
    measurements: np.ndarray, centers: np.ndarray, rotations: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batched signed distance function for n rotated rectangles.

    Each point is moved into the rectangle's local frame (shifted by the
    center, rotated by the negative angle) and then evaluated against an
    axis-aligned rectangle. Since rotation preserves distances, the result
    is an exact signed distance.

    Args:
        measurements: Shape (n, 2), full width and height per rectangle.
        centers: Shape (n, 2), one center per rectangle.
        rotations: Shape (n,), one angle per rectangle in radians,
            counterclockwise. An angle of 0 means axis-aligned.

    Returns:
        A function mapping points of shape (p, 2) to signed distances of
        shape (n, p): negative inside, zero on the boundary, positive outside.
    """

    def sdf(points: np.ndarray) -> np.ndarray:
        _cos, _sin = np.cos(-rotations), np.sin(-rotations)

        rotation_matrices: np.ndarray = np.stack(
            [np.stack([_cos, -_sin], axis=-1), np.stack([_sin, _cos], axis=-1)],
            axis=-2,
        )

        offsets: np.ndarray = points[None, :, :] - centers[:, None, :]
        rotated: np.ndarray = np.einsum("nij,npj->npi", rotation_matrices, offsets)
        absolute: np.ndarray = np.abs(rotated)
        relative: np.ndarray = absolute - (measurements[:, None, :] / 2)

        inside_mask: np.ndarray = (relative < 0).all(axis=-1)

        distances: np.ndarray = np.where(
            inside_mask,
            np.max(relative, axis=-1),
            np.linalg.norm(np.maximum(relative, 0), axis=-1),
        )
        return distances

    return sdf
