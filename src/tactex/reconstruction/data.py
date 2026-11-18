import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tactex.sensing.raycasting import cast, sample, traverse
from tactex.typing import OccupancyGrid2D, Ray2D, Traversal2D, TraversalGrid2D


class TactileCollate:
    """Turns a batch of full occupancy grids into (observation, target)."""

    def __init__(self, resolution: int, max_rays: int, seed: int | None = None):
        self.resolution: int = resolution
        self.max_rays: int = max_rays
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

    def __call__(self, batch):
        if isinstance(batch[0], (tuple, list)):
            batch = [b[0] for b in batch]
        batch: OccupancyGrid2D = np.stack(batch, axis=0)
        B, _, _ = batch.shape
        rays: Ray2D = sample(
            resolution=self.resolution, b=B, n=self.max_rays, rng=self.rng
        )
        traversals: Traversal2D = traverse(rays=rays, resolution=self.resolution)
        _, N, _, _ = traversals.shape
        num_rays: np.ndarray = self.rng.integers(low=1, high=N, size=(B, 1))
        traversal_grids: TraversalGrid2D = cast(
            occupancy_grids=batch, traversals=traversals, num_rays=num_rays
        )
        x = torch.from_numpy(traversal_grids).float()  # (B, 3, H, W)
        y = torch.from_numpy(batch[:, None]).float()  # (B, 1, H, W)
        return x, y


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    resolution: int,
    max_rays: int,
    seed: int | None,
    num_workers: int | None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TactileCollate(resolution=resolution, max_rays=max_rays, seed=seed),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
