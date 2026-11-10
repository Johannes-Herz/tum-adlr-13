import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tactex.paths import generator_dir, list_versions, new_versioned_dir, versioned_dir

TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def _to_json(value: Any):
    """Convert NumPy arrays and scalars to JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _does_seed_exist(base_path: Path, seed: int) -> bool:
    """Return True if any complete version of a generator used this seed."""
    versions: list[str] = list_versions(base_path=base_path)
    for version in versions:
        metadata_filepath: Path = (
            versioned_dir(base_path=base_path, version=version) / "metadata.json"
        )
        if not metadata_filepath.is_file():
            continue
        with open(
            file=metadata_filepath,
            mode="r",
        ) as file:
            metadata: dict[str, Any] = json.load(fp=file)
        if metadata.get("seed") == seed:
            return True
    return False


def save(
    generator: str,
    occupancy_grids: np.ndarray,
    sdf_grids: np.ndarray,
    resolution: int,
    seed: int,
    generator_params: dict[str, Any],
    sample_params: dict[str, Any],
) -> Path:
    """Save one generator run as a new, immutable version.

    Writes to `<data_root>/generated/<generator>/res<R>/<version>/`:
    - `occupancy.npy`: bool (n, R, R), True = occupied
    - `sdf.npy`: float32 (n, R, R), signed distance per cell
    - `sample_params.npz`: per-sample parameters, each array of length n
    - `metadata.json`: generator config, seed, format info; written last

    Never overwrites: each call creates a new version folder.
    """

    N: int = occupancy_grids.shape[0]
    SHAPE: tuple[int, int, int] = (N, resolution, resolution)
    if occupancy_grids.shape != SHAPE:
        raise ValueError(
            f"occupancy_grids has shape {occupancy_grids.shape}. Expected {SHAPE}."
        )
    if sdf_grids.shape != occupancy_grids.shape:
        raise ValueError(
            f"sdf_grids.shape {sdf_grids.shape}!= occupancy_grids.shape {occupancy_grids.shape}."
        )
    for key, value in sample_params.items():
        if len(value) != N:
            raise ValueError(
                f"sample_params[{key!r}] has length {len(value)}. Expected {N}."
            )
    base_path: Path = generator_dir(generator=generator, resolution=resolution)
    if _does_seed_exist(base_path=base_path, seed=seed):
        raise ValueError(
            f"Seed {seed} was already used for generator {generator!r} at resolution "
            f"{resolution}. Use a new seed, or pass allow_seed_reuse=True if the "
            f"generator settings changed on purpose."
        )

    versioned_path: Path = new_versioned_dir(base_path=base_path)

    np.save(versioned_path / "occupancy.npy", occupancy_grids.astype(bool))
    np.save(versioned_path / "sdf.npy", sdf_grids.astype(np.float32))
    np.savez(versioned_path / "sample_params.npz", **sample_params)

    metadata: dict[str, Any] = {
        "format_version": 1,
        "generator": generator,
        "version": versioned_path.name,
        "resolution": resolution,
        "seed": seed,
        "number_of_samples": int(occupancy_grids.shape[0]),
        "generator_params": generator_params,
        "arrays": {
            "occupancy": {"dtype": "bool", "shape": SHAPE},
            "sdf": {"dtype": "float32", "shape": SHAPE},
        },
        "created": datetime.now(tz=UTC).strftime(TIMESTAMP_FORMAT),
    }
    with open(versioned_path / "metadata.json", "w") as file:
        json.dump(obj=metadata, fp=file, indent=2, default=_to_json)
    return versioned_path
