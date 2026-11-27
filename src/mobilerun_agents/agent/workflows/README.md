# Workflow steps

**Purpose:** Observe the current device, choose and execute actions, then return a result.

**What belongs here / what doesn't:** Coordinate observations, selection repair, action dispatch and screenshot capture here. Never retry a device action after it may have executed. Operators own checkpoint and macro export.

**Contents:**

- `__init__.py` — package entry.
- `agent.py` — observe, select, act and finish.
- `guards.py` — derive stop reasons from state without device or model access.
- `planning.py` — observe, plan the next subgoal or finish, and execute a child `AgentWorkflow`. The event stream reports planner observations and each subgoal as it starts and finishes.
- `resources.py` — nonserializable run resources.
- `core/` — shared state and result models.

**Examples:** `await AgentOperator().execute(device, goal)` runs the observation/action loop.

An Unfinished outcome ends the agent run without retrying selection or executing a tool. It keeps the internal `reason="abstained"`. Invalid responses still use selection repair. In a planning run, the executor hands the unfinished subgoal back to the planner, which can re-plan within the remaining step budget.

Installed plugins implement the base `Operator` and are loaded by [`ExternalOperator`](../operators/README.md), outside the workflow layer.

The operator resolves the step budget from `execute(max_steps=...)` or `config.agent.max_steps`. `AgentState.max_steps` travels with the checkpoint.

`PlanningOperator` starts `PlanningWorkflow` with `config.agent.planning_max_steps` as the total step budget, unless `execute(max_steps=...)` overrides it. Each planner turn and executor action costs one step. The planner observes at the start and after each subgoal without spending a step. A `Plan` replaces the remaining plan, then starts a child `AgentWorkflow` for its first subgoal. The child's budget is capped by `config.agent.planning.subgoal_steps` and the remaining total minus one. The last step is reserved for the planner to call `complete` and explain what blocked it if the goal is unfinished. Only a planner's `Finish` can complete the overall goal with a typed result.

`ctx.store` keeps the raw goal, notes and action records. `DeviceResource`, `MemoryResource` and `ServicesResource` inject the phone, LlamaIndex memory, and the registry and image store. Selectors take the registry as an argument and read the current snapshot from context. Both the event stream and operator outputs can contain raw run data.

`StepSnapshot` keeps the observed UI and screenshot together. `ActionRecord` keeps the canonical base action and its `before` snapshot for macro export, and the model-visible call for memory blocks to render. With an empty UI tree, observation captures a screenshot for the screen fingerprint. With a tree, only a selector using vision captures one, on demand. Action execution saves any captured image only after the step finishes; checkpoints include paths, not image bytes. The next observation sets `screen_changed` for the previous step.

Selection repair and abstention never save a screenshot. When checkpoints are disabled, the operator deletes the run's image folder when it closes.

**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for checkpoints, logs and analytics.

The workflow stops on the step limit, on five consecutive calls with the same tool, target and other arguments (`stuck_fail_after`), or on five consecutive unchanged steps. Repeated targets count even when the screen changes. Unknown comparisons break the unchanged streak. The cycle guard counts returns to the same screen, not consecutive observations of it; discovering a new screen resets its streak. `RepeatBlock` warns at three matching calls (`repeat_warn_after`) without changing which tools can be selected.

The planning workflow reads each child's final state and result directly, then refreshes the full UI and device context. It retains only the current planner screenshot. After a step limit, stuck guard, or cycle, the next planner request gets the reason, recent thoughts and action results, screen visits, and observed blockers.
