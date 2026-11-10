import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap

from tactex.shapes.load_grid import load_grid
from tactex.shapes.ray_caster import ray_caster


def main():
    """
    Create and visualize ray_caster grid of the FIRST.
    Grid: -1: unknown, 0: observed free, 1: observed occupied.
    To determin origin of ray, only free pixels of ground truth grid are used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "create_contact_shape.yaml",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(
            "Config must contain a YAML mapping of parameter names to values."
        )

    grids = load_grid(category=config["category"], shapenet=True)
    grid = grids[0, :, :]
    height, width = grid.shape
    rng = np.random.default_rng(config["seed"])

    contact_grid = np.full(grid.shape, -1, dtype=np.int8)
    free_pixels = np.argwhere(grid == 0)
    if len(free_pixels) == 0:
        raise ValueError("The grid has no free pixels for ray origins.")
    if grid.size < 2:
        raise ValueError("At least two pixels are needed for a ray.")

    for _ in range(config["num_rays"]):
        # The ray caster takes and returns coordinates in (row, column) order.
        origin = free_pixels[rng.integers(len(free_pixels))]
        # Use grid indices to make sure origin and goal of ray are note the same, with max one random draw
        origin_index = int(origin[0] * width + origin[1])
        goal_index = int(rng.integers(grid.size - 1))
        if goal_index >= origin_index:
            goal_index += 1  # skip origin index
        goal = np.array(np.unravel_index(goal_index, (height, width)))

        print(f"origin={origin}, goal={goal}, direction={goal - origin}")
        hit, miss_grid = ray_caster(origin, goal, grid)
        contact_grid[miss_grid & (contact_grid != 1)] = 0
        if hit is not None:
            contact_grid[hit] = 1

    cmap = ListedColormap(["lightgray", "white", "red"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots()
    image = ax.imshow(
        contact_grid,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["Unknown", "Observed empty", "Observed occupied"])

    ax.set_title(f"Contact Shape after {config['num_rays']} Rays")
    ax.set_xlabel("x / Columns")
    ax.set_ylabel("y / Rows")

    fig.tight_layout()
    output_path = "contact_shape.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
