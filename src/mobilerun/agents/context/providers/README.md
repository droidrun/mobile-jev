# agents/context/providers/

Builds agent context from local and remote app cards.

Keep app-card backends and context assembly here; generic context interfaces live in `context/context_provider.py`.

## Quick example

```python
import asyncio
from mobilerun.agents.context.providers import LocalCardBackend

async def main():
    cards = LocalCardBackend()
    print(await cards.fetch_card("com.google.android.gm", "Summarize email", "android"))

asyncio.run(main())
```

The bundled card is selected by platform and package. See [app cards](../../assets/app_cards/README.md) for mappings and custom directories.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `LocalCardBackend`, `ServerCardBackend`, `CompositeCardBackend`. |
| `app_card_context_provider.py` | `AppCardContextProvider`, `context_provider_from_config` — context provider backed by local and/or remote app cards. |
| `composite_provider.py` | `CacheStats`, `CompositeCardBackend` — composite app card provider. |
| `local_provider.py` | `LocalCardBackend` — local file-based app card provider. |
| `server_provider.py` | `ServerCardBackend` — server-based app card provider. |


See [context providers](../README.md).
