import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel

type Loader = Callable[[str], Any]
type Dumper = Callable[[dict[str, Any]], str]

_LOADERS: dict[str, Loader] = {
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}

_DUMPERS: dict[str, Dumper] = {
    ".yaml": lambda data: yaml.safe_dump(data, sort_keys=False),
    ".yml": lambda data: yaml.safe_dump(data, sort_keys=False),
    ".json": lambda data: json.dumps(data, indent=4),
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

    def to_file(self, path: Path) -> Path:
        """Save the configuration to a file."""
        extension: str = path.suffix.lower()
        dumper: Dumper | None = _DUMPERS.get(extension)
        if dumper is None:
            supported_extensions: list[str] = ", ".join(_DUMPERS.keys())
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported extensions: {supported_extensions}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        contents: str = dumper(self.model_dump(mode="json"))
        path.write_text(contents, encoding="utf-8")
        return path

    def __str__(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False).strip()
