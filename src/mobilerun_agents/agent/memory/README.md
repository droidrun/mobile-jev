# Run memory

**Purpose:** Carry useful action history and current device context into model calls within one run using LlamaIndex's native `Memory` resource.

**What belongs here / what doesn't:** Budget chat history and render observation context here. Persisted run records and replay artifacts belong to checkpoints. Current UI state and screenshot references are held by `StepSnapshot`, not copied into chat history.

**Contents:**

- `__init__.py` — package entry.
- `memory.py` — `build_run_memory` and its private SQLite engine. The owner must call `await memory.aclose()` once the workflow has stopped using it.
- `messages.py` — `remember(memory, call, record)` appends a native action turn using the model-visible call; `text_messages` projects history for text-only models.
- `blocks/` — `RepeatBlock`, `DeviceContextBlock`, run notes and optional context sources rendered for the model.

**Examples:** Selectors use `await memory.aget(snapshot=state.current, state=state)` to include explicit notes, live repeat feedback and device context. Add the bundled app cards with `memory: {blocks: [app_cards]}` in the agent config. No extra blocks are enabled by default.

Each invocation, including each planning child, creates fresh `Memory`. A tool call and result are appended together with a user boundary marker for turn eviction. Whether the screen changed is only known at the next observation, so selectors give that feedback with the current screen instead of storing it with the tool result. `RepeatBlock` reads it directly from the action records. The chat queue gets 80% of `memory.token_limit` (24,000 of 30,000 tokens by default), counting tool arguments. Overflow targets a 2,500-token flush at the default budget and evicts complete turns. Archived messages stay in the private in-memory SQLite store for the invocation only. Full action records remain in workflow state. The SQL engine never sends chat rows or bound arguments to host debug handlers.

**Depends on / used by:** Uses LlamaIndex memory, typed config and action records. The operator supplies memory to workflows; selectors see the assembled context.
