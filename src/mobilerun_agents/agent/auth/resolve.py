import os
from dataclasses import dataclass, field

from mobilerun.errors import CredentialNotFoundError

from . import store


class CredentialMissing(CredentialNotFoundError):
    """No usable agent credential was supplied or saved."""


@dataclass(frozen=True)
class Credential:
    provider: str
    api_key: str = field(repr=False)


async def resolve(provider: str, api_key: str | None = None) -> Credential:
    provider = store.provider_name(provider)
    env = "XAI_API_KEY" if provider == "grok" else f"{provider.upper()}_API_KEY"
    if key := api_key or os.environ.get(env):
        return Credential(provider, api_key=key)
    saved = store.load(provider)
    if saved:
        return Credential(provider, saved.api_key)
    raise CredentialMissing(
        f"No API key for {provider}: pass api_key or set {env}. "
        "You can also save a key with `mobilerun agent providers add`."
    )
