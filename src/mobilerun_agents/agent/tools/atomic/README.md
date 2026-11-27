# Atomic tools

**Purpose:** Turn callable tools into finite choices for decision selectors.

**What belongs here / what doesn't:** Expand finite schemas and define special cases. Dispatch belongs to the registry.

**Contents:** `__init__.py` — expansion entry point; `schema.py` — finite schema conversion; `exceptions.py` — special-case expansions.

**Examples:** `await registry.atomic_tools()` creates one `tap_element` choice per element, four `swipe` directions, one `start_app` per installed app, and completion choices with free-form messages.

With `mobilerun --debug agent run …`, `tools:` reports exceptions, finite schema expansions, parameterless pass-throughs, and drops, in that order. Unchanged counts aren't repeated on later turns.

**Depends on / used by:** The registry calls this package; it uses tool metadata and shared helpers.
