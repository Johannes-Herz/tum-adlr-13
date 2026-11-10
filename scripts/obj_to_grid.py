from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from tactex.settings import get_settings


def generate_dataset_as_grid(
    data_root: Path,
    extracted_dir: Path,
    category: str,
    all_objects: bool = False,
    specific_object: str = None,
    size: int = 64,
    num_samples_per_obj: int = 1,
) -> None:
    """
    Generate dataset categorywise as an occupancy grid from obj files. Dataset will be saved as .npz file.

    Args:
        data_root (Path): Root directory for the dataset.
        extracted_dir (Path): Path to the extracted directory containing category folders.
        category (str): Name of category (e.g., "02773838").
        all_objects (bool, optional): Whether to process all objects in the category.
            Defaults to False and processes only the first 5 objects.
        specific_object (str, optional): Specific object ID. If provided, only this object is processed. Defaults to None.
        size (int, optional): Size of the occupancy grid. Defaults to 64.
        num_samples_per_obj (int, optional): Number of samples to generate per object. Defaults to 1.

    Returns:
        None
    """

    category_dir = extracted_dir / category

    masks = []

    if specific_object:
        obj_path = category_dir / specific_object / "models" / "model_normalized.obj"
        mesh = load_shape_from_obj(obj_path)
        for _ in range(num_samples_per_obj):
            mask = obj_to_grid(mesh, size=size)
            masks.append(mask)

    elif all_objects:
        for folder in sorted(category_dir.iterdir()):
            if folder.is_dir():
                obj_path = (
                    category_dir / folder.name / "models" / "model_normalized.obj"
                )
                mesh = load_shape_from_obj(obj_path)
                for _ in range(num_samples_per_obj):
                    mask = obj_to_grid(mesh, size=size)
                    masks.append(mask)

    # Default case: process only the first 5 objects in the category
    else:
        for folder in sorted(category_dir.iterdir())[: 15 + 1]:
            if folder.is_dir():
                obj_path = (
                    category_dir / folder.name / "models" / "model_normalized.obj"
                )
                mesh = load_shape_from_obj(obj_path)
                for _ in range(num_samples_per_obj):
                    mask = obj_to_grid(mesh, size=size)
                    masks.append(mask)

    masks_array = np.stack(masks, axis=0).astype(np.uint8)
    output_dir = data_root / "full_grid" / "shapenet"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{category}_occupancy_grid.npz",
        masks=masks_array,
    )
    print(
        f"Saved occupancy grid dataset for category {category} at {output_dir / f'{category}_occupancy_grid.npz'}. npz has shpe of {masks_array.shape}"
    )


def obj_to_grid(mesh: trimesh.Trimesh, size: int = 64):
    """
    Convert a 3D mesh to a 2D occupancy grid.
    Mesh is randomly rotated and projected onto the XY plane. Resulting 2D points are
    randomly scaled to a size between 70% and 100% of the max grid size, rasterized to a grid, and the
    triangles (defined by the mesh faces and rotated vertices) are filled in the grid.

    Args:
        mesh (trimesh.Trimesh): The input 3D mesh.
        size (int, optional): The size of the output occupancy grid. Defaults to 64

    Returns:
        np.ndarray: A binary 2D occupancy grid of shape (size, size).
    """

    rotation = Rotation.random()
    rotated_vertices = rotation.apply(mesh.vertices)

    vertices_2d = rotated_vertices[:, :2].copy()

    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)

    center = (min_coords + max_coords) / 2
    extent = max(max_coords - min_coords)

    if extent == 0:
        raise ValueError(f"Invalid extent value: {extent}")

    padding = 2
    max_scale = (size - 1 - 2 * padding) / extent

    # Random scaling between 70% and 100% of max_scale
    scale = max_scale * np.random.uniform(0.6, 1.0)

    pixels = (vertices_2d - center) * scale  # center vertices around origin
    pixels += (size - 1) / 2  # shift to center of grid

    # Flip y-axis to match image coordinates
    pixels[:, 1] = size - 1 - pixels[:, 1]

    pixels = np.round(pixels).astype(int)

    mask = np.zeros((size, size), dtype=np.uint8)

    for face in mesh.faces:
        triangle = pixels[face]
        cv2.fillConvexPoly(mask, triangle, 1)

    return mask


def load_shape_from_obj(obj_path: Path) -> trimesh.Trimesh:
    """
    Load a 3D shape from an OBJ file and return it as a Trimesh object.

    Args:
        obj_path (Path): Path to the OBJ file.

    Returns:
        trimesh.Trimesh: Loaded 3D shape as trimesh object.
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    mesh = trimesh.load_mesh(obj_path)
    return mesh


def main():
    settings = get_settings()
    data_root = settings.data_root
    extracted_dir = data_root / "extracted" / "shapenet"
    category = "03797390"
    all_objects = False
    specific_object = "1b9ef45fefefa35ed13f430b2941481"
    num_samples_per_obj = 5

    generate_dataset_as_grid(
        data_root,
        extracted_dir,
        category,
        all_objects,
        specific_object=None,
        size=64,
        num_samples_per_obj=num_samples_per_obj,
    )


if __name__ == "__main__":
    main()
