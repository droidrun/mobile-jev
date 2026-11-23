"""
Composite app card provider.

Tries server first, falls back to local if server fails or returns empty.
"""

import logging
from typing import TypedDict

from mobilerun.agents.config.settings import DEFAULT_APP_CARDS_DIR
from mobilerun.agents.context.providers.local_provider import LocalCardBackend
from mobilerun.agents.context.providers.server_provider import ServerCardBackend

logger = logging.getLogger("mobilerun")


class CacheStats(TypedDict):
    server: dict[str, int]
    local: dict[str, int]


class CompositeCardBackend:
    """
    Load app cards from server with local fallback.

    Strategy:
    1. Try server first
    2. If server fails or returns empty, try local
    3. Return first non-empty result, or empty if both fail
    """

    def __init__(
        self,
        server_url: str,
        app_cards_dir: str = DEFAULT_APP_CARDS_DIR,
        server_timeout: float = 2.0,
        server_max_retries: int = 2,
    ) -> None:
        """
        Initialize composite provider.

        Args:
            server_url: Base URL of the app card server
            app_cards_dir: Directory containing the android/ and ios/ card folders
            server_timeout: Server request timeout in seconds
            server_max_retries: Number of server retry attempts
        """
        self.server_provider = ServerCardBackend(
            server_url=server_url,
            timeout=server_timeout,
            max_retries=server_max_retries,
        )
        self.local_provider = LocalCardBackend(app_cards_dir=app_cards_dir)

    async def fetch_card(
        self, package_name: str, instruction: str = "", platform: str | None = None
    ) -> str:
        """
        Load app card with server-first, local-fallback strategy.

        Args:
            package_name: Android package name or iOS bundle id
            instruction: User instruction/goal
            platform: "android" or "ios", when known

        Returns:
            App card content from server or local, or empty string if both fail
        """
        if not package_name:
            return ""

        # Try server first
        server_result = await self.server_provider.fetch_card(package_name, instruction, platform)

        if server_result:
            return server_result

        # Server failed or returned empty, try local
        logger.debug(f"Composite provider: falling back to local for {package_name}")
        local_result = await self.local_provider.fetch_card(package_name, instruction, platform)

        if local_result:
            logger.debug(f"Composite provider: using local fallback for {package_name}")
        else:
            logger.debug(f"Composite provider: no app card found for {package_name}")

        return local_result

    def clear_cache(self) -> None:
        """Clear caches in both providers."""
        self.server_provider.clear_cache()
        self.local_provider.clear_cache()
        logger.debug("Composite app card cache cleared")

    def get_cache_stats(self) -> CacheStats:
        """
        Get cache statistics from both providers.

        Returns:
            Dict with cache stats from server and local providers
        """
        return {
            "server": self.server_provider.get_cache_stats(),
            "local": self.local_provider.get_cache_stats(),
        }
