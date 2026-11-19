import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel

type Loader = Callable[[str], Any]

_LOADERS: dict[str, Loader] = {
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.load,
}


class FileConfig(BaseModel):
    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a configuration from a file."""
        extension: str = path.suffix.lower()
        loader: Loader | None = _LOADERS.get(extension)
        if loader is None:
            supported_extensions: list[str] = ", ".join(_LOADERS.keys())
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported extensions: {supported_extensions}"
            )
        contents: Any = loader(path.read_text(encoding="utf-8"))
        return cls.model_validate(contents)

    def __str__(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False).strip()
