# Choosing an action

**Purpose:** Turn a workflow observation and the available actions into a selection, without touching the device.

**What belongs here / what doesn't:** Compose selection and handoff policies here. Executing an action or retrying a tap belongs to neither selector.

**Contents:**

- `__init__.py` — selector exports.
- `base.py` — the `select(run)` contract, candidates and returned selections.
- `delegating.py` — hand off explicit delegation or low confidence.
- `fallback.py` — try another selector after an unusable selection.
- `loader.py` — resolve configured profiles into a selector.
- `llm/`, `decision_model/` — chat-based and choice-based implementations.

**Examples:** `DelegatingSelector(DecisionSelector(TypeSafeDecisionModel(), vision="off", exclude={"press_button"}), LlamaIndexLLMSelector(llm, vision="on"))` pairs tree-only Jev with an LLM that requests a screenshot only when called, while excluding system-button choices. Pass it as `AgentOperator(selector=selector)`.

Each selector sets its own `vision` mode: `off` sends the tree without direct coordinate tools; `on` sends tree and image; `only` sends the image and omits indexed tools. Indexed tools are only offered when there are elements. Jev accepts `exclude` tool names to omit atomic choices, for example `DecisionSelector(model, vision="on", exclude={"press_button"})`. Jev uses a located `tap(x, y)` choice only in `only` mode and cannot be the planning selector.

**Depends on / used by:** Workflows call the selector service. It uses the registry, authentication and config; it does not drive the workflow.
