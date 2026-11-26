# Workflow steps

**Purpose:** Observe the current device, choose and execute actions, then return a result.

**What belongs here / what doesn't:** Coordinate steps, cancellation, validation and retries here. Never retry a device action after it may have executed. Persistence belongs to the operator’s consumers.

**Contents:**

- `__init__.py` — package entry.
- `agent.py` — observe, select, act and finish.
- `planning.py` — ask a `Planner` for the next subgoal or final result, and run each subgoal as an independent `AgentWorkflow`.
- `external.py` — load and run installed workflow plugins by name from the `mobilerun.agents` entry-point group.
- `events.py` — typed workflow and stream events.
- `resources.py` — nonserializable run resources.
- `core/` — shared state and result models.

**Examples:** `await AgentOperator().execute(device, goal)` runs the observation/action loop.

The operator resolves the step budget from `execute(max_steps=...)` or `config.agent.max_steps`. `AgentState.max_steps` travels with the checkpoint.

`PlanningOperator` starts `PlanningWorkflow` with `config.agent.planning_max_steps` as the total executor budget. A `Plan` replaces the remaining plan, then starts a child `AgentWorkflow` for its first subgoal. Its budget is the smaller of `config.agent.planning.subgoal_steps` and the remaining total. After each child result, the planner reviews the remaining plan and explicitly chooses what to run next. `config.agent.planning.max_planner_turns` bounds planner turns. Only a planner's `Finish` can complete the overall goal with a typed result.

`run.ui` pins the base UI state. `await run.screenshot()` fetches the full-resolution image once per turn, shared across delegates and repair attempts. Checkpoints and tracing only receive an image if a selector requested it. Immediately before selection the workflow copies its state into `run.state`; changes to that copy are discarded. Steps resolve one `Run` resource. Its factory uses invocation-local storage and `cache=False` because the resource manager is shared by all invocations of the workflow object. The live device, model, memory and tools never enter the serialized context.


**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for checkpoints, logs and analytics.
