# Observation context blocks

**Purpose:** Render specific sources of context for the current step.

**What belongs here / what doesn't:** Each block contributes information, not a selection or device action. Token budgeting belongs to the memory builder.

**Contents:**

- `__init__.py` — `ObservationBlock` reads live context rather than archived messages.
- `app_cards.py` — local app-specific guidance for the current platform.
- `device_context.py` — context published by the device and its extensions.
- `repeat.py` — prior action on the current screen and repeat guidance.
- `notes.py` — render the goal and notes without storing them in chat.

**Examples:** `await memory.aget(snapshot=state.current, state=state)` renders context for the current screen.

**Depends on / used by:** Blocks use the snapshot and memory block contract; the operator or memory builder composes them for a run.
