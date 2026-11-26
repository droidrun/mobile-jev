# Workflow steps

**Purpose:** Observe the current device, choose and execute actions, then return a result.

**What belongs here / what doesn't:** Coordinate steps, cancellation, validation and retries here. Never retry a device action after it may have executed. Persistence belongs to the operator’s consumers.

**Contents:**

- `__init__.py` — package entry.
- `agent.py` — observe, select, act and finish.
- `planning.py` — ask a `Planner` for the next subgoal or final result, and run each subgoal as an independent `AgentWorkflow`.
- `events.py` — typed workflow and stream events.
- `resources.py` — nonserializable run resources.
- `core/` — shared state and result models.

**Examples:** `await AgentOperator().execute(device, goal)` runs the observation/action loop.

Installed plugins implement the base `Operator` and are loaded by [`ExternalOperator`](../operators/README.md), outside the workflow layer.

The operator resolves the step budget from `execute(max_steps=...)` or `config.agent.max_steps`. `AgentState.max_steps` travels with the checkpoint.

`PlanningOperator` starts `PlanningWorkflow` with `config.agent.planning_max_steps` as the total step budget. A `Plan` replaces the remaining plan, then starts a child `AgentWorkflow` for its first subgoal. Its budget is the smaller of `config.agent.planning.subgoal_steps` and the remaining total. After every subgoal, the planner is asked to review progress and choose what to run next. `config.agent.planning.subgoal_steps` bounds each subgoal and `config.agent.planning.max_planner_turns` bounds planner turns. Only a planner's `Finish` can complete the overall goal with a typed result.

`run.ui` pins the base UI state. `await run.screenshot()` fetches the full-resolution image once per turn, shared across delegates and repair attempts. Checkpoints and tracing only receive an image if a selector requested it. Immediately before selection the workflow copies its state into `run.state`; changes to that copy are discarded. Steps resolve one `Run` resource. Its factory uses invocation-local storage and `cache=False` because the resource manager is shared by all invocations of the workflow object. The live device, model, memory and tools never enter the serialized context.


**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for checkpoints, logs and analytics.

The stuck guard excludes the same action on an unchanged screen after `stuck_exclude_after` (2) repeats and fails the run/subgoal at `stuck_fail_after` (4).

The cycle guard counts returns to the same screen, not consecutive observations of it. It adds a model note on visit `cycle_note_after` (3) and ends the run or subgoal with reason `cycle` on visit `cycle_fail_after` (4).

When a child reaches its step limit, gets stuck, or cycles, the next planner turn gets a summary of the reason, recent thoughts and action results, screen visits, and observed blockers. The serialized summary masks tracked sensitive values before extracting blocker sentences or truncating text.
