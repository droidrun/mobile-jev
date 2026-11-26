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

`ctx.store` keeps the raw run record: goal, notes, actions, and guard state. Treat `ctx.to_dict()` as private. Stream events and on-disk artifacts are separate, masked output projections. Only the device and LlamaIndex memory are injected resources, cached on each workflow invocation. The selector, registry, and observation cache are live invocation objects, not serialized data.

`StepSnapshot` is the sole home for observed UI and screenshot reference: it carries `at`, `ui`, `screenshot` (`ImageBlock` with a saved path), `device_context` and one fingerprint. `ActionRecord.before` points to the corresponding snapshot. Screenshot bytes are kept in memory only for the current turn; checkpoints store the path reference. Once a sensitive value is known, checkpoint and trace outputs suppress the image. Selectors read a detached copy through `run.state`.


**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for checkpoints, logs and analytics.

The same-action guard excludes a repeated action on an unchanged screen after `stuck_exclude_after` (2) executions and ends the run as `stuck` at `stuck_fail_after` (4). The no-progress guard counts consecutive unchanged screens across different actions too. It adds a model note after `no_progress_note_after` (3) and stops as `stuck` at `no_progress_fail_after` (5). An unknown comparison breaks the streak.

The cycle guard counts returns to the same screen, not consecutive observations of it. It adds a model note on visit `cycle_note_after` (3) and ends the run or subgoal with reason `cycle` on visit `cycle_fail_after` (4).

The planning workflow reads each child's final state and result directly. After a step limit, stuck guard, or cycle, the next planner request gets the reason, recent thoughts and action results, screen visits, and observed blockers. Plan and result data stay raw for the planner; public diagnostics are masked.
