# Run memory

**Purpose:** Carry useful action history and current device context into model calls within one run using LlamaIndex's native `Memory` resource.

**What belongs here / what doesn't:** Budget chat history and render observation context here. Persisted run records and replay artifacts belong to checkpoints. Current UI state and screenshot references are held by `StepSnapshot`, not copied into chat history.

**Contents:**

- `__init__.py` — package entry.
- `memory.py` — `build_memory` and the observation-aware block base.
- `messages.py` — native action turns and their text projection.
- `blocks/` — `RepeatBlock`, `DeviceContextBlock`, run notes and optional context sources rendered for the model.

**Examples:** The operator includes the goal, explicit notes, live guard feedback, and device context. Add the bundled app cards with `memory: {blocks: [app_cards]}` in the agent config. No extra blocks are enabled by default. A custom Python memory factory can still supply a `RetrieverBlock` with its own retriever.

Each invocation, including each planning child, creates fresh `Memory`. Steps are stored as an assistant tool call plus its tool result, with a user boundary marker for turn eviction. No separate history field is part of the memory contract. The current screen and screenshot are not copied into chat history; the snapshot supplies current context through `RepeatBlock`, `DeviceContextBlock` and notes. Optional configured blocks such as `app_cards` extend the context. The chat queue gets 80% of `memory.token_limit` (24,000 of 30,000 tokens by default); each overflow flush targets one twelfth of the total budget and extends to a complete turn. Flushed messages remain in the private in-memory SQLite store for the invocation only. Full action records also live in workflow state. The engine hides SQL parameters from host logs. Runs are not resumed after context restore.

**Depends on / used by:** Uses LlamaIndex memory, typed config and action records. The operator supplies memory to workflows; selectors see the assembled context.
