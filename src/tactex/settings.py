from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Machine-specific settings, read from environment variables and `.env`.

    Every field can be set via an environment variable with the prefix `TACTEX_`, e.g. `TACTEX_DATA_ROOT`.

    Attributes:
        data_root: Root directory for all datasets and processed data. Either an absolute path, or a path relative to the project root. Defaults to `<project_root>/data`.
    """

    model_config = SettingsConfigDict(
        env_prefix="TACTEX_", env_file=_PROJECT_ROOT / ".env", extra="ignore"
    )

    data_root: Path = Path("data")
    hf_token: SecretStr | None = None

    @property
    def project_root(self) -> Path:
        """Root of the repository (not configurable)."""
        return _PROJECT_ROOT

    @model_validator(mode="after")
    def _resolve_paths(self) -> Self:
        if not (self.project_root / "pyproject.toml").is_file():
            raise ValueError(f"Project root has no pyproject.toml: {self.project_root}")
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
