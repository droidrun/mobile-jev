"""Typed agent configuration over the shared loader and migration pipeline."""

from typing import TYPE_CHECKING

from mobilerun.config.loader import ConfigLoader as BaseConfigLoader
from mobilerun.config.loader import OutdatedConfigError

from mobilerun.agents.config.user_config import MobileConfig


class ConfigLoader(BaseConfigLoader):
    config_type = MobileConfig

    if TYPE_CHECKING:

        @classmethod
        def load(cls, config_path: str | None = None) -> MobileConfig:
            """The inherited loader constructs ``cls.config_type``."""
            ...


__all__ = ["ConfigLoader", "OutdatedConfigError"]
