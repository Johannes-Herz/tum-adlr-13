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
        signed_distances = distances - radii[:, None]
        return signed_distances

    return sdf


def rectangle(
    sizes: np.ndarray, centers: np.ndarray, angles: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batched signed distance function for n rotated rectangles.

    Each point is moved into the rectangle's local frame (shifted by the
    center, rotated by the negative angle) and then evaluated against an
    axis-aligned rectangle. Since rotation preserves distances, the result
    is an exact signed distance.

    Args:
        sizes: Shape (n, 2), full width and height per rectangle.
        centers: Shape (n, 2), one center per rectangle.
        angles: Shape (n,), one rotation angle per rectangle in radians,
            counterclockwise. An angle of 0 means axis-aligned.

    Returns:
        A function mapping points of shape (p, 2) to signed distances of
        shape (n, p): negative inside, zero on the boundary, positive outside.
    """

    def sdf(points: np.ndarray) -> np.ndarray:
        _cos, _sin = np.cos(-angles), np.sin(-angles)

        rotation_matrices: np.ndarray = np.stack(
            [np.stack([_cos, -_sin], axis=-1), np.stack([_sin, _cos], axis=-1)],
            axis=-2,
        )

        offsets: np.ndarray = points[None, :, :] - centers[:, None, :]
        rotated: np.ndarray = np.einsum("nij,npj->npi", rotation_matrices, offsets)
        absolute: np.ndarray = np.abs(rotated)
        relative: np.ndarray = absolute - (sizes[:, None, :] / 2)

        inside_mask: np.ndarray = (relative < 0).all(axis=-1)

        signed_distances: np.ndarray = np.where(
            inside_mask,
            np.max(relative, axis=-1),
            np.linalg.norm(np.maximum(relative, 0), axis=-1),
        )
        return signed_distances

    return sdf


def triangle(vertices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batched signed distance function for n triangles.

    The distance is the minimum distance from a point to the three edge
    segments, found by projecting the point onto each edge and clamping the
    projection to the segment. The sign comes from 2D cross products: a point
    is inside if it lies on the same side of all three edges. This works for
    both clockwise and counterclockwise vertex order.

    Args:
        vertices: Shape (n, 3, 2), the three corner points of each triangle.
            Triangles must be non-degenerate (no repeated vertices).

    Returns:
        A function mapping points of shape (p, 2) to signed distances of
        shape (n, p): negative inside, zero on the boundary, positive outside.
    """
    vertices = vertices[:, :, None, :]

    def sdf(points: np.ndarray) -> np.ndarray:
        points = points[None, None, :, :]
        edges: np.ndarray = np.roll(vertices, -1, axis=1) - vertices
        vertices_to_points: np.ndarray = points - vertices
        cross_products: np.ndarray = (
            edges[:, :, :, 0] * vertices_to_points[:, :, :, 1]
            - edges[:, :, :, 1] * vertices_to_points[:, :, :, 0]
        )
        dot_products: np.ndarray = np.sum(edges * vertices_to_points, axis=-1)
        squared_edge_lengths: np.ndarray = np.sum(edges * edges, axis=-1)
        edge_fractions: np.ndarray = np.clip(dot_products / squared_edge_lengths, 0, 1)
        projections: np.ndarray = edge_fractions[:, :, :, None] * edges
        distances: np.ndarray = np.linalg.norm(
            vertices_to_points - projections, axis=-1
        ).min(axis=1)
        inside_mask: np.ndarray = (cross_products > 0).all(axis=1) | (
            cross_products < 0
        ).all(axis=1)
        signed_distances: np.ndarray = np.where(inside_mask, -distances, distances)
        return signed_distances

    return sdf
