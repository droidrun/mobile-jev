# Shared tool helpers

**Purpose:** Share UI element and coordinate helpers across the agent module.

**What belongs here / what doesn't:** Helpers may use run state and core contracts, but must not depend on the registry or atomic expansion.

**Contents:** `__init__.py` — shared exports; `elements.py` — element targeting and labels; `coordinates.py` — coordinate fields, dimensions, and bounds; `usage.py` — schema-derived coordinate and UI-index tool names.

**Examples:** `check_point(x, y, run)` checks a point against the pinned observation.

**Depends on / used by:** Tools and selectors use these helpers.
