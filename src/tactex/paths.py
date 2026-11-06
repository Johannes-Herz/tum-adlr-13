import re
from pathlib import Path

from tactex.settings import get_settings

_VERSION_RE = re.compile(r"^v(\d{3})$")


def raw_dir(source: str) -> Path:
    """e.g. raw_dir("shapenet) -> <data_root>/raw/shapenet"""
    return get_settings().data_root / "raw" / source


def generator_dir(generator: str, resolution: int) -> Path:
    """e.g. generator_dir("circles", 64) -> <data_root>/generated/circles/res064"""
    return get_settings().data_root / "generated" / generator / f"res{resolution:03d}"


def versioned_dir(generator: str, resolution: int, version: str) -> Path:
    """Folder of one existing version, e.g. .../generated/circles/res064/v002"""
    if not _VERSION_RE.match(version):
        raise ValueError(f"Version must look like 'v001', got {version!r}")
    return generator_dir(generator=generator, resolution=resolution) / version


def list_versions(generator: str, resolution: str) -> list[str]:
    base_path: Path = generator_dir(generator=generator, resolution=resolution)
    if not base_path.is_dir():
        return []
    return sorted(
        folder.name
        for folder in base_path.iterdir()
        if folder.is_dir() and _VERSION_RE.match(folder.name)
    )


def latest_version(generator: str, resolution: str) -> str:
    versions: list[str] = list_versions(generator=generator, resolution=resolution)
    if not versions:
        raise FileNotFoundError(
            f"No versions in {generator_dir(generator=generator, resolution=resolution)}"
        )
    return versions[-1]


def new_versioned_dir(generator: str, resolution: int) -> Path:
    """Create the next free version folder (v001, v002, ...) and return it.

    Never returns an existing folder, so generated data can't be overwritten.
    """
    versions: list[str] = list_versions(generator=generator, resolution=resolution)
    next_version: int = int(versions[-1][1:]) + 1 if versions else 1
    path: Path = versioned_dir(
        generator=generator, resolution=resolution, version=f"v{next_version:03d}"
    )
    path.mkdir(parents=True)
    return path
