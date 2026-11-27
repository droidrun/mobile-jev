# assets/app_cards/

**Purpose:** Bundle platform-specific app guidance and mappings.

**What belongs here / what doesn't:** App card data belongs here; card loading belongs in agent memory.

**Contents:**

- `README.md` — Module overview.
- `android/` — Android app cards and mapping.
- `ios/` — iOS app cards and mapping.

**Examples:**

```python
from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.config.paths import resolve_prompt_path

block = AppCardBlock(platform="android", cards_dir=resolve_prompt_path("assets/app_cards"))
```

**Depends on / used by:** `AppCardBlock` looks up local cards by platform and package name.
