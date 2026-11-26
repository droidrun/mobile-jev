# Agent actions

**Purpose:** Expose device actions and run-local tools through one registry.

**What belongs here / what doesn't:** Declare, project and dispatch actions here. Selectors choose; workflows control run progression.

**Contents:**

- `__init__.py` — public tool exports.
- `base.py` — tool declaration, results, completion and delegation.
- `registry.py` — run-local action registry and projections.
- `atomic/` — expand finite inputs into fixed Jev choices.
- `helpers/` — shared element, coordinate and schema helpers.

**Examples:** `registry.tools(exclude={"wait"}, include=(extra,))` returns the callable projection. `await registry.atomic_tools()` expands finite arguments. Both accept `exclude=()` and `include=()` and do not mutate the registry.

**Depends on / used by:** Uses base device actions, validation and run state. Workflows dispatch through the same registry offered to selectors.

**Callable tools:** Layer 1 exposes callable actions as `device.x`: native methods and extensions, minus a denylist, gated by device support. Selectors may add run-local tools such as `complete` and `wait`. The registry does not know about vision; selectors construct exclusions for `vision="off"`, `"on"` or `"only"`.

**Atomic choices:** Literal and boolean values expand. Unsupported or unconvertible tools are dropped and reported in `--debug`. Exceptions expand `tap_element` once per element, `swipe` into four directions, `start_app` once per installed app, and `complete` as success/failure with its message left open.
