# Agent actions

**Purpose:** Expose the operator’s available actions through one run-local registry.

**What belongs here / what doesn't:** Expose device actions through one run-local registry, derive atomic choices for Jev, and resolve UI indices against the observed tree. Selectors choose actions; workflows decide when the run ends.

**Contents:**

```text
agent/
  run.py                   Run state and pinned observation
  tools/
    __init__.py            Public tool exports
    base.py                @tool, action results, completion and delegation
    registry.py            Discover, validate and dispatch actions; expose .atomic(run)
    atomic/
      __init__.py          Expand registry members into fixed choices
      schema.py            Finite schema values, limits and drop diagnostics
      exceptions.py        Indexed taps, directional swipes and app launches
    helpers/
      __init__.py          Shared helper exports for tools and selectors
      elements.py          Element visibility, targeting and labels
      coordinates.py       Device dimensions and point bounds
```

Module entry points: `__init__.py`, `base.py`, and `registry.py`.

**Examples:** `AgentOperator(tools=[inventory_label])` adds a documented `@tool` async function to the defaults. Its first argument, `run: Run`, is injected and excluded from the input schema. `@tool(sensitive_parameters=frozenset({"token"}))` hides that argument in action history. Name collisions fail before the selector runs. `registry.tools` holds executable LlamaIndex tools; `await registry.by_name["inventory_label"].acall()` returns a `ToolOutput`. `registry.extend(extra, exclude={Source.DEVICE, "wait", "extension:bench"})` creates a new registry. Nothing is excluded by default.

**Depends on / used by:** Uses base device actions, schema validation, config and templates. The workflow dispatches through the same registry it shows the selector.

**Three layers:** (1) The LLM sees the callable base actions, generically wrapped: `tap_element`, `input_text`, `press_button`, `start_app`, and `swipe` (plus run tools such as `complete` and `wait`). Coordinate actions `tap`, `swipe`, and `drag` are offered with vision. (2) `registry.atomic(run)` converts finite schema values. It drops free values and unsupported shapes, logging drops in `--debug`. It expands `tap_element` per visible element, `swipe` into four directions, and `start_app` per installed app. (3) Jev turns these atomic tools into choices; `tap(x, y)` is available only with `vision="only"`.

Example: `DecisionSelector(model, vision="only", exclude={"press_button"})` keeps system-button actions out of Jev's choices. Jev's `swipe down` option moves the finger up to reveal content below; `[tools].swipe_paths` and `[tools].swipe_duration` configure it.
