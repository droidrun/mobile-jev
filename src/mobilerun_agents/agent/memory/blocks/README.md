# Observation context blocks

**Purpose:** Render specific sources of context for the current step.

**What belongs here / what doesn't:** Each block contributes information, not a selection or device action. Token budgeting belongs to the memory builder.

**Contents:**

- `__init__.py` — package entry.
- `app_cards.py` — app-specific guidance.
- `device_context.py` — context published by the device and its extensions.
- `retriever.py` — relevant material from a LlamaIndex retriever.
- `repeat.py` — prior action on the current screen and repeat guidance.
- `run_notes.py` — render the goal and notes without storing them in chat.

**Examples:** `from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock`; pass `DeviceContextBlock()` in the block list to `build_memory`.

**Depends on / used by:** Blocks use the snapshot and memory block contract; the operator or memory builder composes them for a run.
