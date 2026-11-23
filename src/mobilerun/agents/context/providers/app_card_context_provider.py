"""Context provider backed by local and/or remote app cards."""

import logging

from mobilerun.agents.config.settings import DEFAULT_APP_CARDS_DIR
from mobilerun.agents.config.user_config import AppCardConfig
from mobilerun.agents.context.context_provider import (
    ContextProvider,
    DisabledContextProvider,
    StepContext,
)
from mobilerun.agents.context.providers.composite_provider import CompositeCardBackend
from mobilerun.agents.context.providers.local_provider import LocalCardBackend
from mobilerun.agents.context.providers.server_provider import ServerCardBackend


class AppCardContextProvider(ContextProvider):
    def __init__(
        self,
        mode: str = "local",
        app_cards_dir: str = DEFAULT_APP_CARDS_DIR,
        server_url: str | None = None,
        server_timeout: float = 2.0,
        server_max_retries: int = 2,
    ) -> None:
        mode = mode.lower()
        if mode in ("server", "composite") and not server_url:
            logging.getLogger("mobilerun").warning(
                "%s mode but no server_url, falling back to local", mode
            )
            mode = "local"
        self.backend: ServerCardBackend | CompositeCardBackend | LocalCardBackend
        if mode == "server" and server_url is not None:
            self.backend = ServerCardBackend(
                server_url, timeout=server_timeout, max_retries=server_max_retries
            )
        elif mode == "composite" and server_url is not None:
            self.backend = CompositeCardBackend(
                server_url, app_cards_dir, server_timeout, server_max_retries
            )
        else:
            if mode != "local":
                logging.getLogger("mobilerun").warning(
                    "Unknown app_card mode '%s', falling back to local", mode
                )
            self.backend = LocalCardBackend(app_cards_dir)

    async def get_context(self, context: StepContext) -> str:
        platform = (context.platform or "").lower()
        if platform not in ("android", "ios"):
            platform = None
        return await self.backend.fetch_card(context.package_name, context.instruction, platform)


def context_provider_from_config(config: AppCardConfig) -> ContextProvider:
    if not config.enabled:
        return DisabledContextProvider()
    return AppCardContextProvider(
        mode=config.mode,
        app_cards_dir=config.app_cards_dir,
        server_url=config.server_url,
        server_timeout=config.server_timeout,
        server_max_retries=config.server_max_retries,
    )
