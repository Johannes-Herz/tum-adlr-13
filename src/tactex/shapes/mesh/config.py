from dataclasses import dataclass
from pathlib import Path

from tactex.settings import get_settings


@dataclass(frozen=True)
class DatasetShapenetConfig:
    """Settings for generating ShapeNet occupancy grids.

    Attributes:
        category: ShapeNet category identifier, e.g. "03797390". Required.
        data_root: Dataset root. None uses the application settings.
        extracted_dir: Directory containing category folders. None uses
            <data_root>/extracted/shapenet.
        specific_object_id: Select one object. When set, all_objects and
            max_objects are ignored.
        max_objects: Maximum number of objects, selected in sorted order.
            Ignored if either all_objects is True and/or specific_object_id is not None.
        all_objects: Process every object in the category.
            Ignored if specific_object_id is not None.
        grid_size: Grid width and height in pixels. Must be at least 32.
        samples_per_object: Positive number of projections per object.
        seed: Random seed shared by rotation and scaling. None uses a nonfixed seed.
        scale_range: Minimum and maximum scale of the available grid span.
        padding: Free border around the projected object, in pixels.
    """

    category: str
    data_root: Path | None = None
    extracted_dir: Path | None = None
    specific_object_id: str | None = None
    max_objects: int = 15
    all_objects: bool = False
    grid_size: int = 64
    samples_per_object: int = 5
    seed: int | None = 42
    scale_range: tuple[float, float] = (0.6, 1.0)
    padding: int = 2

    def __post_init__(self) -> None:
        """Check generation settings immediately after construction."""
        if not self.category:
            raise ValueError("A ShapeNet category is required.")
        if self.max_objects <= 0:
            raise ValueError("max_objects must be positive.")
        if self.samples_per_object <= 0:
            raise ValueError("samples_per_object must be positive.")
        if self.grid_size < 32:
            raise ValueError("Grid size must be at least 32.")

    @property
    def resolved_data_root(self) -> Path:
        """Dataset root used consistently for reading and saving grids."""
        return (
            self.data_root if self.data_root is not None else get_settings().data_root
        )
