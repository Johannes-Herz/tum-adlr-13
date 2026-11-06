import numpy as np

from tactex.settings import get_settings


def load_grid(category: str = None, shapenet: bool = False):
    """
    Load binary occupancy grids from .npz files.

    Args:
        None

    Returns:


    """
    settings = get_settings()
    data_root = settings.data_root
    if shapenet:
        category_path = (
            data_root / "full_grid" / "shapenet" / f"{category}_occupancy_grid.npz"
        )
    else:
        raise ValueError("Currently only shapenet is supported for loading grids.")
        return None

    with np.load(category_path) as data:
        grids = data["masks"]

    return grids
