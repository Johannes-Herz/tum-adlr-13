from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tactex import timings
from tactex.sensing.raycasting import cast, sample, traverse
from tactex.timings import TimedDataLoader, timed
from tactex.typing import OccupancyGrid2D, Ray2D, Traversal2D, TraversalGrid2D


class OccupancyGridDatset(Dataset):
    """Loads occupancy grids from disk.
    Expected layout in `root`:
        grids.npy   (N, D, H, W)  uint8/bool
    """

    def __init__(
        self,
        root: Path,
        split: Literal["train", "val", "test"],
        in_memory: bool = False,
        limit: int | None = None,
    ):
        self.root: Path = root
        self.split: Literal["train", "val", "test"] = split
        self.in_memory: bool = in_memory
        self.mmap_mode: Literal["r"] | None = None if in_memory else "r"
        self.grids: np.ndarray | np.memmap = None
        self.limit: int = limit

    def _lazy_load(self) -> np.ndarray | np.memmap:
        if self.grids is None:
            self.grids = np.load(
                self.root / f"{self.split}.npy", mmap_mode=self.mmap_mode
            )
        return self.grids

    def __getitem__(self, index: int):
        if self.grids is None:
            self._lazy_load()
        return np.array(self.grids[index]).astype(np.bool)

    def __len__(self):
        if self.grids is None:
            self._lazy_load()
        return (
            min(self.limit, self.grids.shape[0])
            if self.limit is not None
            else self.grids.shape[0]
        )


class TactileCollate:
    """Turns a batch of full occupancy grids into (observation, target)."""

    def __init__(self, resolution: int, max_rays: int, seed: int | None = None):
        self.resolution: int = resolution
        self.max_rays: int = max_rays
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

    @timed()
    def __call__(self, batch):
        if isinstance(batch[0], (tuple, list)):
            batch = [b[0] for b in batch]
        batch: OccupancyGrid2D = np.stack(batch, axis=0)
        B, _, _ = batch.shape
        num_rays: np.ndarray = self.rng.integers(low=1, high=self.max_rays, size=(B, 1))
        max_rays_for_batch: int = int(np.max(num_rays))
        rays: Ray2D = sample(
            resolution=self.resolution,
            b=B,
            n=max_rays_for_batch,  # Sample no. of rays per grid before and limit ray generation here to minimize memory
            rng=self.rng,
        )
        traversals: Traversal2D = traverse(rays=rays, resolution=self.resolution)
        traversal_grids: TraversalGrid2D = cast(
            occupancy_grids=batch, traversals=traversals, num_rays=num_rays
        )
        x = torch.from_numpy(traversal_grids).bool()  # (B, 3, H, W)
        y = torch.from_numpy(batch[:, None]).bool()  # (B, 1, H, W)
        return x, y


def build_dataloader(
    dataset: OccupancyGridDatset,
    batch_size: int,
    shuffle: bool,
    resolution: int,
    max_rays: int,
    name: str,
    seed: int | None,
    num_workers: int = 0,
) -> DataLoader:
    collate_fn = TactileCollate(resolution=resolution, max_rays=max_rays, seed=seed)
    common: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    if timings.is_enabled():
        return TimedDataLoader(
            dataset,
            name=name,
            collate_fn=collate_fn,
            **common,
        )
    return DataLoader(dataset, collate_fn=collate_fn, **common)
