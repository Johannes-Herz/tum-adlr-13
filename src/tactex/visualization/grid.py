from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def show(grids: np.ndarray, k: int, rng: np.random.Generator, cols: int = 3):
    """Plot k randomly chosen grids as occupancy images or SDF heatmaps.

    Grids are sampled without replacement and shown on [-1, 1]^2 with the
    origin at the lower left. The 0-level set is drawn as a gold contour.
    The plot type is inferred from the dtype: bool or int8 grids are shown
    as occupancy (binary colormap), all others as SDFs (RdBu colormap with
    a symmetric range [-m, m], where m is the largest absolute value among
    the sampled grids, plus one shared colorbar).

    Args:
        grids: Array of shape (N, H, W) holding occupancy or SDF grids.
        k: Number of grids to show. Clipped to N if larger.
        rng: Random number generator used to choose which grids to show.
        cols: Number of subplot columns. Rows are ceil(k / cols), and
            unused axes are hidden.

    Returns:
        None. The figure is displayed with plt.show().
    """
    k: int = min(k, len(grids))
    indices: np.ndarray = rng.choice(len(grids), size=k, replace=False)

    rows: int = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    m = np.abs(grids[indices]).max()
    style = {"origin": "lower", "extent": (-1, 1, -1, 1), "interpolation": "nearest"}

    _type: Literal["occupancy", "sdf"] = (
        "occupancy" if grids.dtype in (np.bool_, np.int8) else "sdf"
    )

    for ax, i in zip(axs.flatten(), indices):
        grid: np.ndarray = grids[i]
        if _type == "occupancy":
            ax.imshow(grid, cmap="binary", **style)
        elif _type == "sdf":
            im = ax.imshow(grid, cmap="RdBu", vmin=-m, vmax=m, **style)
        ax.contour(
            grid,
            levels=[0],
            colors="gold",
            linewidths=1.5,
            origin="lower",
            extent=(-1, 1, -1, 1),
        )
        ax.set_aspect("equal")

    if _type == "sdf":
        fig.colorbar(im, ax=axs, shrink=0.8, label="signed distance")

    for ax in axs.flatten()[k:]:
        ax.axis("off")
    plt.show()
