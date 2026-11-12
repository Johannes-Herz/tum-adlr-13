import numpy as np

from tactex.paths import category_dir, res_dir


def load_grid(
    category: str | None = None,
    shapenet: bool = False,
    resolution: int | None = None,
    version: int | None = None,
) -> np.ndarray:
    """Load a category's binary occupancy grids from a .npz file.
    Currently only supports shapenet files.

    Args:
        category: ShapeNet category identifier, e.g. "03797390".
        shapenet: Whether to load ShapeNet data. Currently, only True is
            supported.

    Returns:
        Array stored under the "masks" key in the .npz file, with shape
        (num_grids, height, width). Values of 0 -> free pixels,
        values of 1 -> occupied pixels.

    Raises:
        ValueError: If ShapeNet loading is not selected or category is missing.
        FileNotFoundError: If the category archive does not exist.
        KeyError: If the archive does not contain a "masks" entry.
    """
    if not shapenet:
        raise ValueError("Currently only ShapeNet is supported for loading grids.")
    if not category:
        raise ValueError("A category is required to load ShapeNet grids.")
    if resolution is None or resolution <= 0:
        raise ValueError("A resolution is required to load ShapeNet grids.")

    category_path = category_dir(
        source="shapenet", category=category, full_grid=True, extracted=False
    )
    res_path = res_dir(base_path=category_path, resolution=resolution)
    input_path = res_path / f"v{version:03d}" / "occupancy_grid.npz"

    if not input_path.is_file():
        raise ValueError(f"File not found: {input_path}")
    else:
        print(input_path)

    with np.load(input_path) as data:
        return data["masks"]
