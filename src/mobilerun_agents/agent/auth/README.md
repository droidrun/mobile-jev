# Provider keys

**Purpose:** Resolve a provider credential without exposing it in status or logs.

**What belongs here / what doesn't:** Save, remove and resolve API keys here. There are no subscription login flows or model clients.

**Contents:**

- `__init__.py` — public credential and resolution API.
- `resolve.py` — choose explicit key, environment key, then saved key.
- `store.py` — manage per-provider records in the user config directory.
- `oauth.py` — exchange a state-checked loopback callback for an OpenRouter API key using [OpenRouter PKCE](https://openrouter.ai/docs/use-cases/oauth-pkce). No subscription credentials or registered first-party client IDs.

**Examples:** `from mobilerun_agents.agent.auth import resolve`; `credential = await resolve("typesafe")`. Jev reads `TYPESAFE_API_KEY` before the key saved by `mobilerun agent setup`. Run `mobilerun agent logout typesafe` to remove the saved key and profile without changing the environment.

**Depends on / used by:** The LLM factory, TypeSafe client and setup and logout commands use this package. It does not import workflows.
