from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def _infer_grid_type(grids: np.ndarray) -> Literal["occupancy", "probability", "sdf"]:
    if grids.dtype in (np.bool_, np.int8):
        return "occupancy"
    if (
        np.issubdtype(grids.dtype, np.floating)
        and grids.min() >= 0.0
        and grids.max() <= 1.0
    ):
        return "probability"
    return "sdf"


def show(grids: np.ndarray, k: int, rng: np.random.Generator, cols: int = 3):
    """Plot k randomly chosen grids as occupancy, probability, or SDF images.

    Grids are sampled without replacement and shown on [-1, 1]^2 with the
    origin at the lower left. The boundary is drawn as a gold contour: the
    0-level set for SDFs and the 0.5-level set for occupancy and probability
    grids.

    The plot type is inferred from the sampled grids: bool or int8 grids are
    shown as occupancy (binary colormap), float grids within [0, 1] as
    probabilities (gray colormap on the fixed range [0, 1]: white free, black
    occupied, gray uncertain), and all others as SDFs (RdBu colormap on a
    symmetric range [-m, m], where m is the largest absolute value among the
    sampled grids). Probability and SDF plots get one shared colorbar.

    Args:
        grids: Array of shape (N, H, W) holding occupancy, probability, or
            SDF grids.
        k: Number of grids to show. Clipped to N if larger.
        rng: Random number generator used to choose which grids to show.
        cols: Number of subplot columns. Rows are ceil(k / cols), and
            unused axes are hidden.

    Returns:
        None. The figure is displayed with plt.show().
    """
    k: int = min(k, len(grids))
    indices: np.ndarray = rng.choice(len(grids), size=k, replace=False)
    samples: np.ndarray = grids[indices]

    rows: int = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    style = {"origin": "lower", "extent": (-1, 1, -1, 1), "interpolation": "nearest"}

    grid_type: Literal["occupancy", "probability", "sdf"] = _infer_grid_type(
        grids=grids
    )
    if grid_type == "occupancy":
        image_style = {"cmap": "binary", "vmin": 0, "vmax": 1}
        level, colorbar_label = 0.5, None
    elif grid_type == "probability":
        image_style = {"cmap": "Greys", "vmin": 0.0, "vmax": 1.0}
        level, colorbar_label = 0.5, "occupancy probability"
    elif grid_type == "sdf":
        m: float = float(np.abs(samples).max())
        image_style = {"cmap": "RdBu", "vmin": -m, "vmax": m}
        level, colorbar_label = 0.0, "signed distance"

    for ax, grid in zip(axs.flatten(), grids):
        im = ax.imshow(grid.astype(float), **image_style, **style)
        ax.contour(
            grid.astype(float),
            levels=[level],
            colors="gold",
            linewidths=1.5,
            origin="lower",
            extent=(-1, 1, -1, 1),
        )
        ax.set_aspect("equal")

    if colorbar_label is not None:
        fig.colorbar(im, ax=axs, shrink=0.8, label=colorbar_label)

    for ax in axs.flatten()[k:]:
        ax.axis("off")
    plt.show()
