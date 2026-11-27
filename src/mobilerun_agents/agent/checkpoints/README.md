# Run checkpoints and macro export

**Purpose:** Preserve the latest workflow context and export a guarded base macro.

**What belongs here / what doesn't:** The context holds the run record. This package neither dispatches actions nor writes separate action or UI sidecars.

**Contents:**
- `__init__.py` — package entry.
- `store.py` — atomic writes in the native LlamaIndex context format, without masking.
- `macro_export.py` — converts successful compatible calls to base macro actions and derives guards from their preceding UI snapshots.

**Examples:** With checkpoints enabled, `<runs>/<run_id>/context.json` is updated after each action and at completion. Read it with `Context.from_dict(workflow, payload)`. Planning saves child actions in its parent context with continuous step numbers. Screenshots remain separate in `<images>/<run_id>/`. Macros go in the base macro directory; guards are included when the recorded screen contains evidence.

**Depends on / used by:** Operated by the stream consumer. The base macro player consumes `macro.json`. Restrict access to run, image and macro folders because they can contain sensitive values.

Checkpoints from before this version's planning-state change (failure summaries are now structured instead of strings) cannot be restored; older `context.json` files can still be read as plain JSON.
