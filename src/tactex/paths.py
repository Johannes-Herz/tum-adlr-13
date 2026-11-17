import re
from pathlib import Path

from tactex.settings import get_settings

_VERSION_RE = re.compile(r"^v(\d{3})$")


def raw_dir(source: str) -> Path:
    """e.g. raw_dir("shapenet) -> <data_root>/raw/shapenet"""
    return get_settings().data_root / "raw" / source


def category_dir(
    source: str, category: str, full_grid: bool = False, extracted: bool = False
) -> Path:
    """e.g. full_grid_dir("shapenet) -> <data_root>/full_grid/shapenet
    source: is source of data, e.g. shapenet"""
    if full_grid == extracted:
        raise ValueError("'full_grid' and 'extracted' must not both be True or False.")
    if full_grid:
        return get_settings().data_root / "full_grid" / source / category
    else:
        return get_settings().data_root / "extracted" / source / category


def res_dir(base_path: Path, resolution: int) -> Path:
    """e.g. resr_dir("base_path", 64) -> base_path/res064"""
    return base_path / f"res{resolution:03d}"


def generator_dir(generator: str, resolution: int) -> Path:
    """e.g. generator_dir("circles", 64) -> <data_root>/generated/circles/res064"""
    return get_settings().data_root / "generated" / generator / f"res{resolution:03d}"


def dataset_dir(dataset: str, resolution: int) -> Path:
    """e.g. generator_dir("basic_shapes", 64) -> <data_root>/datasets/basic_shapes/res064"""
    return get_settings().data_root / "datasets" / dataset / f"res{resolution:03d}"


def versioned_dir(base_path: Path, version: str) -> Path:
    if not _VERSION_RE.match(version):
        raise ValueError(f"Version must look like 'v001', got {version!r}")
    return base_path / version


def list_versions(base_path: Path) -> list[str]:
    if not base_path.is_dir():
        return []
    return sorted(
        folder.name
        for folder in base_path.iterdir()
        if folder.is_dir() and _VERSION_RE.match(folder.name)
    )


def latest_version(base_path: Path) -> str:
    versions: list[str] = list_versions(base_path=base_path)
    if not versions:
        raise FileNotFoundError(f"No versions in {base_path}")
    return versions[-1]


def new_versioned_dir(base_path: Path) -> Path:
    """Create the next free version folder (v001, v002, ...) and return it.

    Never returns an existing folder, so generated data can't be overwritten.
    """
    versions: list[str] = list_versions(base_path=base_path)
    next_version: int = int(versions[-1][1:]) + 1 if versions else 1
    path: Path = versioned_dir(base_path=base_path, version=f"v{next_version:03d}")
    path.mkdir(parents=True)
    return path
