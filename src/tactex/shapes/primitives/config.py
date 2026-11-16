from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, Field


class YamlConfig(BaseModel):
    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with open(file=path, mode="r") as file:
            contents: Any = yaml.safe_load(file)
        return cls.model_validate(contents)

    def __str__(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False).strip()


class GridConfig(YamlConfig):
    resolution: int = 64
    min_coord: float = -1.0
    max_coord: float = 1.0

    @property
    def span(self) -> float:
        return self.max_coord - self.min_coord


class BasePrimitiveConfig(YamlConfig):
    primitive: str
    num_samples: int
    max_retries: int = 25


class CircleConfig(BasePrimitiveConfig):
    primitive: Literal["circle"] = "circle"
    min_radius: float = 0.0
    max_radius: float = 1.0


class RectangleConfig(BasePrimitiveConfig):
    primitive: Literal["rectangle"] = "rectangle"
    min_size: float = 0.0
    max_size: float = 2.0
    min_angle: float = 0.0
    max_angle: float = float(np.pi / 2)


class TriangleConfig(BasePrimitiveConfig):
    primitive: Literal["triangle"] = "triangle"
    min_area: float = 0.0
    max_area: float = 2.0
    min_height: float = 0.0
    max_height: float = 2.0


type PrimitiveConfig = Annotated[
    CircleConfig | RectangleConfig | TriangleConfig,
    Field(discriminator="primitive"),
]

PRIMITIVE_CONFIGS_REGISTRY: dict[str, type[PrimitiveConfig]] = {
    "circle": CircleConfig,
    "rectangle": RectangleConfig,
    "triangle": TriangleConfig,
}


class GenerationConfig(YamlConfig):
    seed: int = 42
    grid_config: GridConfig = GridConfig()
    primitive_configs: list[PrimitiveConfig]
