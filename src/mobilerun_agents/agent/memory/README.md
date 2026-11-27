# Run memory

**Purpose:** Carry context between steps with native LlamaIndex
`Memory.from_defaults(memory_blocks=…, token_limit=…, insert_method=…)`.

**What belongs here / what doesn't:** Memory has no chat history:
blocks render history from persisted workflow state. Per-step
instructions, the current screen and screenshot, device context, and repair hints
belong in the caller's prompt, outside memory.

**Contents:**

- `__init__.py` — exports `read_memory`.
- `memory.py` — reads memory and builds executor and planner memory with a shared template and engine setup.
- `blocks/` — run-state block base and context blocks.

**Examples:** Read workflow memory with `await read_memory(memory, ctx, redact)`. It loads the latest
`AgentState` or `PlanningState` with `await ctx.store.get_state()` and passes it to
`memory.aget(state=state, redact=redact)`, using the run's redactor to mask protected UI labels. Do not pass snapshots or read from a selector's local
copy of state. Setup is a native `StaticMemoryBlock` named `setup` at priority 0:
the caller renders its text once and passes that text to `build_setup_block(text)`.

Create one planner memory per planning run and a fresh executor memory for each
child. Both are closed when their workflows exit. Each memory owns its SQLite
engine unless the caller supplies one; `memory.aclose()` leaves supplied engines open.

**Depends on / used by:** Uses LlamaIndex memory, typed config, and workflow state.
Operators supply memory to workflows; selectors read the resulting messages.

## Add a block

Subclass `StateBlock[YourState, str]` or `StateBlock[YourState, list[ContentBlock]]`.
Give the block a `name` and a `priority`, and implement `async def render(self, state, redact)`
to return text or native content blocks. `StateBlock` ignores short-term memory,
does nothing on put, and raises `TypeError` if read without `state=` or `redact=`. Override
`atruncate` only when a block can shrink itself; the native default drops its content.

## Native behavior pinned by tests

- Priority 0 is never truncated, even when it exceeds the budget on its own.
- Under pressure, LlamaIndex sorts positive priorities in ascending order and
  calls `atruncate` on priority 1 before priority 2, even though its documentation
  calls 1 the highest priority. Retained blocks render in their configured order.
- `InsertMethod.USER` prepends block content to the latest user message. If there
  is no user message, it appends a new one.
- A block can return a `TextBlock` and an `ImageBlock` together. The default template
  preserves URL-backed images (the tested case) and handles paths, but omits
  inline-byte images that have neither a URL nor a path.

## Executor

`memory.executor.blocks` selects blocks in order. By default it lists `setup`,
`goal`, `last_step`, `history`, and `app_cards`. App cards are available on
Android and iOS for apps with a bundled card. To disable them, omit `app_cards`
from `memory.executor.blocks`. Delete the removed top-level `memory.blocks` key.
`last_step` includes every call from the previous step and its screen transition.
It masks call arguments and the executor's thought, but keeps result summaries
unmasked so the executor can use a just-fetched value such as a verification code.
The saved before-screenshot is optional (`memory.executor.last_step_screenshot`,
default `true`); only vision-enabled LLM selectors send it. `history` shows the
previous `memory.executor.history` steps (default `5`), excluding `last_step`,
plus any repeat and cycle warnings. Each older action includes masked arguments
and a short result. Under token pressure app cards (priority 1) are dropped
before executor history (priority 2); history loses its oldest lines first.
