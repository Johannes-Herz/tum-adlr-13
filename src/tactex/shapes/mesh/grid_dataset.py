import warnings
from pathlib import Path

import numpy as np

from tactex.shapes.ioop import save_shapenet
from tactex.shapes.mesh.config import DatasetShapenetConfig
from tactex.shapes.mesh.mesh_to_grid import load_shape_from_obj, mesh_to_grid


def generate_grid_dataset(config: DatasetShapenetConfig) -> Path:
    """Generate and save random 2D projections of ShapeNet OBJ meshes.

    Args:
        config: Generation settings, documented in DatasetShapenetConfig.

    Returns:
        output_path: Version directory containing occupancy_grid.npz and metadata.json.
            The archive's "masks" entry has shape
            (num_objects * samples_per_object, grid_size, grid_size) and dtype uint8.
            Each run creates a new version folder, preserving previous runs.
            metadata.json saves settings

    Raises:
        ValueError: If an argument is invalid, no objects are selected,
            or a mesh cannot be projected.
        FileNotFoundError: If a required directory or OBJ file is missing.
    """

    if config.specific_object_id is not None and config.all_objects:
        warnings.warn(
            "all_objects is ignored because specific_object_id is provided.",
            UserWarning,
            stacklevel=2,
        )

    extracted_dir = config.extracted_dir
    if extracted_dir is None:
        extracted_dir = config.resolved_data_root / "extracted" / "shapenet"
    category_dir = extracted_dir / config.category

    if config.specific_object_id is not None:
        object_dir = category_dir / config.specific_object_id
        if not object_dir.is_dir():
            raise ValueError(
                f"Specific object ID directory does not exist: {object_dir}"
            )
        # List, to use same for loop
        object_dirs = [object_dir]
    else:
        object_dirs = sorted(path for path in category_dir.iterdir() if path.is_dir())
        if not config.all_objects:
            object_dirs = object_dirs[: config.max_objects]

    if not object_dirs:
        raise ValueError(f"No object directories found in {category_dir}")
    rng = np.random.default_rng(config.seed)

    masks = []
    for object_dir in object_dirs:
        obj_path = object_dir / "models" / "model_normalized.obj"
        mesh = load_shape_from_obj(obj_path)
        for _ in range(config.samples_per_object):
            masks.append(
                mesh_to_grid(
                    mesh,
                    grid_size=config.grid_size,
                    scale_range=config.scale_range,
                    padding=config.padding,
                    rng=rng,
                )
            )

    masks_array = np.stack(masks, axis=0)

    output_path = save_shapenet(masks=masks_array, config=config)

    return output_path
