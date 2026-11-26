# Run memory

**Purpose:** Carry useful history and current device context into model calls within one run.

**What belongs here / what doesn't:** Budget chat history and render observations here. Persisted evidence and replay artifacts belong to checkpoints.

**Contents:**

- `__init__.py` — package entry.
- `memory.py` — `build_memory` and the observation-aware block base.
- `snapshot.py` — typed per-step `StepSnapshot`.
- `blocks/` — context sources rendered for the model.

**Examples:** `build_memory(config.memory, [DeviceContextBlock()])` builds a token-budgeted LlamaIndex memory for a run.

**Depends on / used by:** Uses LlamaIndex memory, typed config and action records. The operator supplies memory to workflows; selectors see the assembled context.
