# Run memory

**Purpose:** Carry context between steps with native LlamaIndex
`Memory.from_defaults(memory_blocks=…, token_limit=…, insert_method=…)`.

**What belongs here / what doesn't:** Memory has no chat history:
blocks render history from persisted workflow state. Per-step
instructions, the current screen and screenshot, device context, and repair hints
belong in the caller's prompt, outside memory.

**Contents:**

- `__init__.py` — exports `StateBlock`, `BlockContext`, `BlockFactory`, and `read_memory`.
- `memory.py` — reads memory and builds executor and planner memory with a shared template and engine setup.
- `registry.py` — builds named blocks from built-in, installed, and operator-supplied factories.
- `blocks/` — run-state block base and context blocks.

**Examples:** Read workflow memory with `await read_memory(memory, ctx)`. It loads the latest
`AgentState` or `PlanningState` with `await ctx.store.get_state()` and passes it to
`memory.aget(state=state)`. Do not pass snapshots or read from a selector's local
copy of state. The `setup` factory renders its prompt once into a native
`StaticMemoryBlock` at priority 0.

Create one planner memory per planning run and a fresh executor memory for each
child. Both are closed when their workflows exit. Each memory owns its SQLite
engine unless the caller supplies one; `memory.aclose()` leaves supplied engines open.

**Depends on / used by:** Uses LlamaIndex memory, typed config, and workflow state.
Operators supply memory to workflows; selectors read the resulting messages.

## Write your own block

Subclass `StateBlock[YourState, str]` or `StateBlock[YourState, list[ContentBlock]]`.
Give the block a `name` and a `priority`, and implement `async def render(self, state)`
to return text or native content blocks. `StateBlock` ignores short-term memory,
does nothing on put, and raises `TypeError` if read without `state=`. Override
`atruncate` only when a block can shrink itself; the native default drops its content.

```python
# my_pkg/blocks.py
from mobilerun_agents import AgentState, BlockContext, StateBlock


class PriceWatch(StateBlock[AgentState, str]):
    name: str = "price_watch"
    priority: int = 3
    currency: str

    async def render(self, state: AgentState) -> str:
        return f"Report prices in {self.currency} for: {state.goal}"


def price_watch(context: BlockContext) -> PriceWatch:
    if context.agent != "executor":
        raise ValueError("price_watch requires the executor")
    currency = context.options.get("currency", "USD")
    if type(currency) is not str or currency not in {"USD", "EUR"}:
        raise ValueError("currency must be USD or EUR")
    return PriceWatch(currency=currency)
```

The factory receives the role (`executor` or `planner`), device platform, raw
options for this block, and the read-only run config. Planner factories also
receive the initial or restored `PlanningState` as `context.state`, so setup
can use `state.executor` before any turns run. Use `render`'s `state` argument for
live content instead of retaining build-time state. The factory should return a
fresh block whose `name` matches its registration.

Register the factory in your package's `pyproject.toml` and install the package:

```toml
[project.entry-points."mobilerun.memory_blocks"]
price_watch = "my_pkg.blocks:price_watch"
```

Enable it at the desired position in `agents.yaml`. The `blocks` list replaces
the defaults for that role:

```yaml
memory:
  executor:
    blocks: [setup, goal, price_watch, last_step, history, app_cards]
    options:
      price_watch: {currency: EUR}
```

For code-only registration, pass `memory_blocks={"price_watch": price_watch}`
to `AgentOperator` or `PlanningOperator`, and enable it using the same config.
The mapping applies only to that operator's runs, including planning children.
Names cannot shadow built-ins or installed entry points.

To contribute model-callable tools, give the block a `tools()` method that
returns a sequence of `Tool` declarations (for example, functions decorated
with `@tool`). Only configured blocks contribute tools. Use the injected
workflow `Context` to read or edit live state.

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
It includes call arguments, the executor's thought, and result summaries so the
executor can use a just-fetched value such as a verification code.
The saved before-screenshot is optional (`memory.executor.options.last_step.screenshot`,
default `true`); only vision-enabled LLM selectors send it. `history` shows the
previous `memory.executor.options.history.steps` steps (default `5`), excluding `last_step`,
plus any repeat and cycle warnings. Each older action includes its arguments
and a short result. Under token pressure app cards (priority 1) are dropped
before executor history (priority 2); history loses its oldest lines first.

## Notes

Add `notes` to either role's `blocks` list to render saved facts separately and
give that role a `note(text)` tool. The tool trims whitespace, masks active
secrets, and skips blank and duplicate notes. Neither role enables the block
by default. The `goal` block still shows saved notes when no separate `notes`
block is configured; this includes the planner's `set_plan` findings.
