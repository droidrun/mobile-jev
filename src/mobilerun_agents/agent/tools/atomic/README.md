# Atomic tools

**Purpose:** Turn executable tools into fixed choices for Jev.

**What belongs here / what doesn't:** Expand finite inputs; do not dispatch actions.

**Contents:** `__init__.py` — expansion entry point; `schema.py` — finite schemas and limits; `exceptions.py` — indexed taps, swipes, and app launches.

**Examples:** `await registry.atomic(run)` produces choices such as `press_button back`.

**Depends on / used by:** The registry calls this package, which uses shared helpers and tool metadata.
