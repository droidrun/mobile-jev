# Choosing an action

**Purpose:** Turn a workflow observation and available actions into a selection, without executing actions.

**What belongs here / what doesn't:** Compose selection and handoff policies here. Executing an action or retrying a tap belongs to neither selector.

**Contents:**

- `__init__.py` — selector exports.
- `base.py` — the `select(ctx, device, memory, registry)` contract and native tool selections.
- `delegating.py` — hand off an explicit request or typed completion to the LLM and route retries for that step.
- `loader.py` — resolve a setup into a selector or planner and executor, with optional Jev delegation.
- `llm/`, `decision_model/` — chat-based and choice-based implementations.

**Examples:** `DelegatingSelector(DecisionSelector(TypeSafeDecisionModel(), vision="off", exclude={"press_button"}), LlamaIndexLLMSelector(llm, vision="on"))` pairs tree-only Jev with an LLM that consumes the observation's screenshot when called, while excluding system-button choices. Pass it as `AgentOperator(selector=selector)`. The workflow passes the run's registry into every selection call.

Each selector sets its own `vision` mode: `off` sends the tree without direct coordinate tools; `on` sends tree and image; `only` sends the image and omits indexed tools. Indexed tools are only offered when there are elements. The registry has no vision policy: selectors build their own exclusions for `vision="off"`, `"on"` or `"only"`. A located tap is offered only with `vision="only"`.

Set Jev's confidence cutoff with `DecisionSelector(model, vision="on", exclude=(), min_confidence=0.7)`. When Jev chooses Unfinished or a pick scores below the cutoff, the agent run ends unfinished without retrying. In a planning run, the planner gets the subgoal back and can choose a new approach. `DelegatingSelector(primary, delegate)` does not call the LLM for an Unfinished outcome. It calls the LLM when Jev selects `delegate` and meets the cutoff, or chooses successful completion in a run that requires typed output. Repairs for that step stay with the LLM. The `delegate` option is never registered as an executable tool.

**Depends on / used by:** Workflows call the selector service. It uses the registry, authentication and config; it does not drive the workflow.
