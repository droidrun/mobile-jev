# Workflow steps

**Purpose:** Observe the current device, choose and execute actions, then return a result.

**What belongs here / what doesn't:** Coordinate observations, selection repair, action dispatch and screenshot safety here. Never retry a device action after it may have executed. Operators own checkpoint and macro export.

**Contents:**

- `__init__.py` — package entry.
- `agent.py` — observe, select, act and finish.
- `guards.py` — derive stop reasons from state without device or model access.
- `planning.py` — two steps: plan the next subgoal or finish, and execute a child `AgentWorkflow`. The event stream reports each subgoal as it starts and finishes.
- `resources.py` — nonserializable run resources.
- `core/` — shared state and result models.

**Examples:** `await AgentOperator().execute(device, goal)` runs the observation/action loop.

Installed plugins implement the base `Operator` and are loaded by [`ExternalOperator`](../operators/README.md), outside the workflow layer.

The operator resolves the step budget from `execute(max_steps=...)` or `config.agent.max_steps`. `AgentState.max_steps` travels with the checkpoint.

`PlanningOperator` starts `PlanningWorkflow` with `config.agent.planning_max_steps` as the total step budget. A `Plan` replaces the remaining plan, then starts a child `AgentWorkflow` for its first subgoal. Its budget is the smaller of `config.agent.planning.subgoal_steps` and the remaining total. After every subgoal, the planner is asked to review progress and choose what to run next. `config.agent.planning.subgoal_steps` bounds each subgoal and `config.agent.planning.max_planner_turns` bounds planner turns. Only a planner's `Finish` can complete the overall goal with a typed result.

`ctx.store` keeps the raw goal, notes and action records. `DeviceResource`, `MemoryResource` and `ServicesResource` inject the phone, LlamaIndex memory, and the registry and image store. Selectors take the registry as an argument and read the current snapshot from context. `handler.stream_events()` on the handler returned by `workflow.run()` carries raw values like `ctx.store`; operator outputs (console, `context.json`, macro, traces, `ExecutionResult`) are masked.

`StepSnapshot` keeps the observed UI and screenshot together. `ActionRecord` keeps the canonical base action and its `before` snapshot for macro export; memory keeps the model-visible call. With an empty UI tree, observation captures a screenshot for the screen fingerprint. With a tree, only a selector using vision captures one, on demand. Action execution saves any captured image only after the step finishes; checkpoints include paths, not image bytes. The next observation sets `screen_changed` for the previous step.

Once any value is protected in a run, no more screenshots are saved. At action commit or selection repair, late identification at step N removes saved images and their references from the first step whose raw record contains the value through step N. If no raw record contains the value, the fallback covers N−1 and N. Selection repair never saves a screenshot. A value visible only in pixels in an earlier step can survive in a saved image.


**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for checkpoints, logs and analytics.

The workflow stops on the step limit, on five consecutive calls with the same tool, target and other arguments (`stuck_fail_after`), or on five consecutive unchanged steps. Repeated targets count even when the screen changes. Unknown comparisons break the unchanged streak. The cycle guard counts returns to the same screen, not consecutive observations of it; discovering a new screen resets its streak. `RepeatBlock` warns at three matching calls (`repeat_warn_after`) without changing which tools can be selected.

The planning workflow reads each child's final state and result directly. After a step limit, stuck guard, or cycle, the next planner request gets the reason, recent thoughts and action results, screen visits, and observed blockers. Plan and result data stay raw for the planner; public diagnostics are masked.
