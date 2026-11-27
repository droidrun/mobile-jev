# Run-state memory blocks

**Purpose:** Render selected parts of the persisted executor or planner state.

**What belongs here / what doesn't:** Blocks render stored run context, not chat history or the current observation prompt. Native LlamaIndex memory applies the token budget.

**Contents:**

- `__init__.py` — `StateBlock`, the base for blocks that render workflow state.
- `app_cards.py` — local app-specific guidance for the current platform.
- `findings.py` — planner findings and notes.
- `goal.py` — the run goal, plus notes for the executor only.
- `history.py` — older executor steps and progress warnings.
- `last_step.py` — the previous executor step, including its screen transition and optional before-screenshot.
- `subgoals.py` — the latest planner subgoal result and older subgoal history.

**Examples:** `await read_memory(memory, ctx, redact)` renders the configured blocks from the latest workflow state using the run's redactor.

**Depends on / used by:** Blocks use workflow state and the LlamaIndex memory contract. Executor and planner builders compose them for each run.
