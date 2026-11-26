# errors/

**Purpose:** Produce short, log-safe descriptions of provider failures.

**What belongs here / what doesn't:** Error formatting belongs here; logging setup and provider retry policy do not.

**Contents:**

- `README.md` — Module overview.
- `__init__.py` — Agent errors with actionable messages, including `UnknownModel` for local adapter rejection.
- `descriptions.py` — Safe error summaries and HTML filter.

**Examples:**

```python
from mobilerun_agents.errors.descriptions import describe_error

message = describe_error(ValueError("invalid input"))
```

**Depends on / used by:** Uses size limits from `config/settings.py`. Callers can summarize exceptions directly or install the HTML error filter on a logger.
