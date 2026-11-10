import numpy as np

from tactex.settings import get_settings


def load_grid(category: str | None = None, shapenet: bool = False) -> np.ndarray:
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

    data_root = get_settings().data_root
    category_path = (
        data_root / "full_grid" / "shapenet" / f"{category}_occupancy_grid.npz"
    )

    with np.load(category_path) as data:
        return data["masks"]
