# Agent actions

**Purpose:** Expose device actions and run-local tools through one registry.

**What belongs here / what doesn't:** Declare, project and dispatch actions here. Selectors choose; workflows control run progression.

**Contents:**

- `__init__.py` — public tool exports.
- `base.py` — tool declaration, results and completion.
- `registry.py` — run-local action registry and projections.
- `atomic/` — expand finite inputs into fixed Jev choices.
- `helpers/` — shared element, coordinate and schema helpers.

**Examples:** `registry.tools(exclude={"wait"})` returns the model-facing schemas. `await registry.atomic_tools(exclude={"wait"}, ui=snapshot.ui, goal=state.goal)` expands finite arguments. Neither changes the registry. `registry.validate(call)` returns a selection error or `None` without dispatching. `await registry.execute(call, ctx)` dispatches a selected action. `registry.resolve(call)` gives the action record its canonical action name and arguments without changing the call ID. `@tool` hides parameters annotated with `Context` or `Device` from the schema and injects them on execution.

**Depends on / used by:** Uses base device actions, validation and run state. Workflows dispatch through the same registry offered to selectors.

Successful actions describe their target from the pinned UI, for example `tapped [26] TextView "Enter a US zip code"` or `tapped (12, 12) Button "Open ZIP sheet"`. Focus is included only when the action already obtained a newer UI state. Void device actions omit the `→ None` suffix.

The base's `secret_scope` masks sensitive values in log records during sensitive calls. The registry does not mask arguments or results stored in run data.

**Callable tools:** Layer 1 exposes callable actions as `device.x`: native methods and extensions, minus a denylist, gated by device support. Selectors may add run-local tools such as `complete` and `wait`. The registry does not know about vision; selectors construct exclusions for `vision="off"`, `"on"` or `"only"`.

**Atomic choices:** Literal and boolean values expand. Unsupported or unconvertible tools are omitted. Exceptions expand `tap_element` once per element, `swipe` into four directions, `start_app` once per installed app, and `complete` as success/failure with its message left open.
