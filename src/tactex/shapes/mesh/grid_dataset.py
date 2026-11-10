import warnings
from pathlib import Path

import numpy as np

from tactex.shapes.mesh.mesh_to_grid import load_shape_from_obj, mesh_to_grid


def generate_grid_dataset(
    data_root: Path,
    category: str,
    extracted_dir: Path | None = None,
    specific_object_id: str | None = None,
    max_objects: int = 15,
    all_objects: bool = False,
    grid_size: int = 64,
    samples_per_object: int = 5,
    rng: np.random.Generator | None = None,
    scale_range: tuple[float, float] = (0.6, 1.0),
    padding: int = 2,
) -> Path:
    """Generate and save random projections of .obj files, from ShapeNet a category, as 2D occupancy grids.

    Args:
        data_root: Dataset root. Results are written to full_grid/shapenet.
        category: ShapeNet category identifier, e.g. "03797390".
        extracted_dir: Directory containing category folders. Defaults to
            data_root / "extracted" / "shapenet".
        specific_object_id: Select one object instead of scanning the category. Defaults to None.
            When not None, max_objects and all_objects do not apply.
        max_objects: Maximum number of object directories to process, in
            sorted order. Defaults to 15.
            Ignored if either all_objects is True and/or specific_object_id is not None.
        all_objects: Process every object in the category.
            Ignored if specific_object_id is not None.
        grid_size: Grid width and height in pixels. Defaults to 64. Must be at least 32.
        samples_per_object: Positive number of projections per object, defaults to 5.
        rng: Generator shared by all samples for rotation and scaling. If None, a new generator
            is created without a fixed seed.
        scale_range: Min and max scaling factors of the available grid span. Defautls to 60%-100%.
        padding: Padding defaults to 2.

    Returns:
        output_path: Path to the saved .npz file. Its "masks" entry has shape
            (num_objects * samples_per_object, grid_size, grid_size) and dtype uint8.
            An existing file for this category is overwritten.

    Raises:
        ValueError: If an argument is invalid, no objects are selected,
            or a mesh cannot be projected.
        FileNotFoundError: If a required directory or OBJ file is missing.
    """
    if not category:
        raise ValueError("A ShapeNet category is required.")
    if specific_object_id is not None and all_objects:
        warnings.warn(
            "all_objects is ignored because specific_object_id is provided.",
            UserWarning,
            stacklevel=2,
        )
    if max_objects <= 0:
        raise ValueError("max_objects must be positive.")
    if samples_per_object <= 0:
        raise ValueError("samples_per_object must be positive.")
    if grid_size < 32:
        raise ValueError("Grid size must be at least 32.")

    if extracted_dir is None:
        extracted_dir = data_root / "extracted" / "shapenet"
    category_dir = extracted_dir / category

    if specific_object_id is not None:
        object_dir = category_dir / specific_object_id
        if not object_dir.is_dir():
            raise ValueError(
                f"Specific object ID directory does not exist: {object_dir}"
            )
        # List, to use same for loop
        object_dirs = [object_dir]
    else:
        object_dirs = sorted(path for path in category_dir.iterdir() if path.is_dir())
        if not all_objects:
            object_dirs = object_dirs[:max_objects]

    if not object_dirs:
        raise ValueError(f"No object directories found in {category_dir}")
    if rng is None:
        rng = np.random.default_rng()

    masks = []
    for object_dir in object_dirs:
        obj_path = object_dir / "models" / "model_normalized.obj"
        mesh = load_shape_from_obj(obj_path)
        for _ in range(samples_per_object):
            masks.append(
                mesh_to_grid(
                    mesh, grid_size=grid_size, scale_range=scale_range, padding=padding, rng=rng
                )
            )

    output_path = (
        data_root / "full_grid" / "shapenet" / f"{category}_occupancy_grid_TEST.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, masks=np.stack(masks, axis=0))
    return output_path
