# Run-state memory blocks

**Purpose:** Render selected parts of the persisted executor or planner state.

**What belongs here / what doesn't:** Blocks render stored run context, not chat history or the current observation prompt. Native LlamaIndex memory applies the token budget.

**Contents:**

- `__init__.py` — `StateBlock`, the base for blocks that render workflow state.
- `app_cards.py` — local app-specific guidance for the current platform.
- `notes.py` — opt-in notes for either role, with a `note` tool to save facts between turns.
- `goal.py` — the run goal and any saved notes, unless a separate `notes` block renders them.
- `history.py` — older executor steps and progress warnings.
- `last_step.py` — the previous executor step, including its screen transition and optional before-screenshot.
- `subgoals.py` — the latest planner subgoal result and older subgoal history.

**Examples:** `await read_memory(memory, ctx)` renders the configured blocks from the latest workflow state.

**Depends on / used by:** Blocks use workflow state and the LlamaIndex memory contract. Executor and planner builders compose them for each run.
