from collections.abc import Callable

import numpy as np

from tactex.shapes.primitives.config import (
    CircleConfig,
    GridConfig,
    PrimitiveConfig,
    RectangleConfig,
    TriangleConfig,
)

type SampleParams = dict[str, np.ndarray]
"""Dictionary mapping parameter names (e.g. `"centers"`, `"radii"`) to batched arrays of shape `(B, ...)`."""
type SampleFunction = Callable[
    [PrimitiveConfig, GridConfig, np.random.Generator], SampleParams
]
"""Callable that samples a batch of primitive parameters given primitive and grid configurations."""


def circle(
    primitive_config: CircleConfig, grid_config: GridConfig, rng: np.random.Generator
) -> SampleParams:
    min_radius: float = max(primitive_config.min_radius, 0.0)
    max_radius: float = min(primitive_config.max_radius, grid_config.span / 2)

    def _radii(k: int) -> np.ndarray:
        return rng.uniform(
            low=min_radius,
            high=max_radius,
            size=(k,),
        )

    def _centers(k: int, radii: np.ndarray) -> np.ndarray:
        return rng.uniform(
            low=grid_config.min_coord + radii[:, None],
            high=grid_config.max_coord - radii[:, None],
            size=(k, 2),
        )

    radii: np.ndarray = _radii(k=primitive_config.num_samples)
    centers: np.ndarray = _centers(k=primitive_config.num_samples, radii=radii)
    return {"centers": centers, "radii": radii}


def rectangle(
    primitive_config: RectangleConfig, grid_config: GridConfig, rng: np.random.Generator
) -> SampleParams:
    min_size: float = max(primitive_config.min_size, 0.0)
    max_size: float = min(primitive_config.max_size, grid_config.span)
    min_angle: float = max(primitive_config.min_angle, 0.0)
    max_angle: float = min(primitive_config.max_angle, np.pi / 2)

    def _sizes(k: int) -> np.ndarray:
        return rng.uniform(
            low=min_size,
            high=max_size,
            size=(k, 2),
        )

    def _angles(k: int) -> np.ndarray:
        return rng.uniform(
            low=min_angle,
            high=max_angle,
            size=(k,),
        )

    sizes: np.ndarray = _sizes(k=primitive_config.num_samples)
    angles: np.ndarray = _angles(k=primitive_config.num_samples)
    half_span: float = grid_config.span / 2
    for _ in range(primitive_config.max_retries):
        half_sizes: np.ndarray = sizes / 2
        c, s = np.abs(np.cos(angles)), np.abs(np.sin(angles))
        ex, ey = (
            c * half_sizes[:, 0] + s * half_sizes[:, 1],
            s * half_sizes[:, 0] + c * half_sizes[:, 1],
        )
        bad: np.ndarray = (ex > half_span) | (ey > half_span)
        if not bad.any():
            break
        k: int = bad.sum()
        sizes[bad] = _sizes(k=k)
        angles[bad] = _angles(k=k)
    cx, cy = (
        rng.uniform(grid_config.min_coord + ex, grid_config.max_coord - ex),
        rng.uniform(grid_config.min_coord + ey, grid_config.max_coord - ey),
    )
    centers: np.ndarray = np.stack([cx, cy], axis=-1)
    return {"sizes": sizes, "centers": centers, "angles": angles}


def triangle(
    primitive_config: TriangleConfig, grid_config: GridConfig, rng: np.random.Generator
) -> SampleParams:
    min_area: float = max(primitive_config.min_area, 0.0)
    max_area: float = min(primitive_config.max_area, (grid_config.span**2) / 2)
    min_height: float = max(primitive_config.min_height, 0.0)
    max_height: float = min(primitive_config.max_height, grid_config.span)

    def _vertices(k: int) -> np.ndarray:
        return rng.uniform(
            low=grid_config.min_coord,
            high=grid_config.max_coord,
            size=(k, 3, 2),
        )

    vertices: np.ndarray = _vertices(primitive_config.num_samples)
    for _ in range(primitive_config.max_retries):
        ab, ac = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        areas: np.ndarray = (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2.0
        edges: np.ndarray = np.roll(vertices, -1, axis=1) - vertices
        edge_lengths: np.ndarray = np.linalg.norm(edges, axis=-1)
        heights: np.ndarray = 2 * np.abs(areas)[:, None] / edge_lengths
        min_heights: np.ndarray = heights.min(axis=1)

        bad: np.ndarray = (
            (np.abs(areas) < min_area)
            | (np.abs(areas) > max_area)
            | (min_heights < min_height)
            | (min_heights > max_height)
        )
        if not bad.any():
            break
        k: int = bad.sum()
        vertices[bad] = _vertices(k=k)

    return {"vertices": vertices}


REGISTRY: dict[str, SampleFunction] = {
    "circle": circle,
    "rectangle": rectangle,
    "triangle": triangle,
}
