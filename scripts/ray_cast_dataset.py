import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from tactex.shapes.ioop import save_partial_shapenet
from tactex.shapes.load_grid import load_grid
from tactex.shapes.ray_caster import ray_caster


def generate_partial_grids(
    grids: np.ndarray,
    partials_per_grid: int = 10,
    min_rays: int = 1,
    max_rays: int = 15,
    origin_pixel_edge: bool = True,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return partial masks and ray counts, grouped in input-grid order.

    Output indices i * partials_per_grid through (i + 1) * partials_per_grid
    belong to input grid i. Each partial starts unknown (-1); traversed free
    pixels become 0 and first occupied hits become 1. Ray counts are sampled
    uniformly from the inclusive interval [min_rays, max_rays].

    Args:
        grids: Binary occupancy grids of shape (N, H, W), with N > 0 and
            H, W >= 2. Values are 0 (free) and 1 (occupied).
        partials_per_grid: Positive number of independent partial grids to
            generate per input grid. Defaults to 10.
        min_rays: Inclusive minimum number of rays per partial grid. Must be
            at least 1. Defaults to 1.
        max_rays: Inclusive maximum number of rays per partial grid. Must be
            at least min_rays. Defaults to 15.
        origin_pixel_edge: If True, sample ray origins only from the grid
            border, which must be entirely free. Otherwise, sample from all
            free pixels. Defaults to True.
        seed: Random seed for ray counts, origins, and targets. None uses
            a nonfixed seed. Defaults to 42.

    Returns:
        A tuple (partial_grids, ray_counts):
            partial_grids: int8 array of shape (N * partials_per_grid, H, W).
                Values are -1 (unknown), 0 (observed free), and 1 (observed
                occupied). The slice [i * partials_per_grid:
                (i + 1) * partials_per_grid] contains the partials for input i.
            ray_counts: int64 array of shape (N * partials_per_grid,) containing
                the number of cast rays for each corresponding partial grid,
                including rays that do not hit an occupied pixel.
    """
    if grids.ndim != 3 or len(grids) == 0 or min(grids.shape[1:]) < 2:
        raise ValueError("Expected nonempty grids with shape (N, H, W), H and W >= 2.")
    if not np.all((grids == 0) | (grids == 1)):
        raise ValueError("Input grids must contain only 0 (free) and 1 (occupied).")
    if partials_per_grid <= 0 or not 1 <= min_rays <= max_rays:
        raise ValueError("Require partials_per_grid > 0 and 1 <= min_rays <= max_rays.")

    rng = np.random.default_rng(seed)
    height, width = grids.shape[1:]
    num_partials = len(grids) * partials_per_grid
    partial_grids = np.full((num_partials, height, width), -1, dtype=np.int8)
    ray_counts = np.empty(num_partials, dtype=np.int64)

    for grid_index, grid in enumerate(tqdm(grids, desc="Ray casting input grids")):
        if origin_pixel_edge:
            if np.any(grid[[0, -1], :] != 0) or np.any(grid[:, [0, -1]] != 0):
                raise ValueError(
                    f"Input grid {grid_index} needs a free border for edge origins."
                )
            edge_mask = np.zeros(grid.shape, dtype=bool)
            edge_mask[[0, -1], :] = True
            edge_mask[:, [0, -1]] = True
            available_origins = np.argwhere(edge_mask)
        else:
            available_origins = np.argwhere(grid == 0)
        if len(available_origins) == 0:
            raise ValueError(
                f"Input grid {grid_index} has no free pixels for ray origins."
            )

        for partial_index in range(partials_per_grid):
            output_index = grid_index * partials_per_grid + partial_index
            contact_grid = partial_grids[output_index]
            num_rays = int(rng.integers(min_rays, max_rays + 1))
            ray_counts[output_index] = num_rays
            for _ in range(num_rays):
                origin = available_origins[rng.integers(len(available_origins))]
                origin_index = int(origin[0] * width + origin[1])
                goal_index = int(rng.integers(grid.size - 1))
                if goal_index >= origin_index:
                    goal_index += 1
                goal = np.array(np.unravel_index(goal_index, (height, width)))

                hit, miss_grid = ray_caster(origin, goal, grid)
                contact_grid[miss_grid & (contact_grid != 1)] = 0
                if hit is not None:
                    contact_grid[hit] = 1

    return partial_grids, ray_counts


def main() -> None:
    """
    Load full ShapeNet occupancy grids, and save dataset of partil ray casted grids.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "ray_cast_dataset.yaml",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(
            "Config must contain a YAML mapping of parameter names to values."
        )

    grids = load_grid(
        category=config["category"],
        shapenet=True,
        resolution=config["dataset_resolution"],
        version=config["dataset_version"],
    )
    # grids = grids[:15]
    partial_grids, ray_counts = generate_partial_grids(
        grids,
        partials_per_grid=config["partials_per_grid"],
        min_rays=config["min_rays"],
        max_rays=config["max_rays"],
        origin_pixel_edge=config["origin_pixel_edge"],
        seed=config["seed"],
    )
    output_path = save_partial_shapenet(
        masks=partial_grids,
        ray_counts=ray_counts,
        config=config,
        num_input_grids=len(grids),
    )
    print(
        f"Saved {len(partial_grids)} partial grids to {output_path / 'partial_grid.npz'}"
    )


if __name__ == "__main__":
    main()
