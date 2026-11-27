# errors/

**Purpose:** Define errors for agent runs.

**What belongs here / what doesn't:** Agent-specific exceptions belong here; logging setup and provider retry policy do not.

**Contents:**

- `README.md` — Module overview.
- `__init__.py` — Agent errors with actionable messages, including `UnknownModel` for local adapter rejection.

**Examples:**

```python
from mobilerun_agents.errors import SelectorCannotServe

raise SelectorCannotServe("No matching action")
```

**Depends on / used by:** Extends the base agent error; used by model adapters and selectors.
