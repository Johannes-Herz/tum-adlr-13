from pathlib import Path

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def load_shape_from_obj(obj_path: Path) -> trimesh.Trimesh:
    """Load an .obj file as a single mesh.

    Args:
        obj_path: Path to the .obj file.

    Returns:
        mesh: The loaded mesh.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded mesh has no vertices or faces.
    """
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    mesh = trimesh.load_mesh(obj_path)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"OBJ file is empty or does not have mesh faces: {obj_path}")
    return mesh


def mesh_to_grid(
    mesh: trimesh.Trimesh,
    grid_size: int = 64,
    scale_range: tuple[float, float] = (0.6, 1.0),
    padding: int = 2,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Rasterize randomly rotated 3D mesh onto a 2D occupancy grid.

    Project the rotated vertices onto the XY plane, center the vertices, and scale them to 60%–100%
    of the available grid span with padding. Fill projected triangles to form a silhouette.
    The image y-axis points downward.

    Args:
        mesh: 3D input mesh.
        grid_size: Grid width and height in pixels. Must be at least 2 * padding + 2.
        scale_range: Min and max scaling factors of the available grid span. Defautls to 60%-100%.
        padding: Padding defaults to 2.
        rng: Generator used for both rotation and scaling. If None, a new generator
            is created without a fixed seed.

    Returns:
        mask: Array of shape (grid_size, grid_size) and dtype uint8. Coordinates are (y, x) and y points downwards to
            match image coordinates. Occupied pixels are 1; free pixels are 0.

    Raises:
        ValueError: If grid_size is too small, the mesh is empty, extended projection is smaller equals zero,
            scaling factors are invalid.
    """

    if grid_size <= 2 * padding + 1:
        raise ValueError(f"Grid size must be at least {2 * padding + 2} for padding.")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("Mesh must contain vertices and triangle faces.")
    if scale_range[0] <= 0 or scale_range[1] > 1 or scale_range[0] > scale_range[1]:
        raise ValueError(
            f"Scaling factors must be greater 0 and small equal 1. Min must be smaller then max. Scale_range: {scale_range}"
        )

    if rng is None:
        rng = np.random.default_rng()

    rotation = Rotation.random(rng=rng)

    # Project onto XY plane
    vertices_2d = rotation.apply(mesh.vertices)[:, :2].copy()

    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)

    center = (min_coords + max_coords) / 2
    extent = (max_coords - min_coords).max()
    if extent <= 0:
        raise ValueError(f"Invalid extend value: {extent}.")

    max_scale = (grid_size - 1 - 2 * padding) / extent
    scale = max_scale * rng.uniform(scale_range[0], scale_range[1])
    pixels = (vertices_2d - center) * scale + (grid_size - 1) / 2
    pixels[:, 1] = grid_size - 1 - pixels[:, 1]  # Flip y-axis to match image coordinates
    pixels = np.round(pixels).astype(np.int32)

    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for face in mesh.faces:
        cv2.fillConvexPoly(mask, pixels[face], 1)
    return mask
