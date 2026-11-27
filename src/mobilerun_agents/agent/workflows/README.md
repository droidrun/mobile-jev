# Workflow steps

**Purpose:** Observe the current device, choose and execute actions, then return a result.

**What belongs here / what doesn't:** Coordinate observations, selection repair, action dispatch and screenshot capture here. Never retry a device action after it may have executed. Core owns device-action recording; operators own persisted reasoning and macro export.

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

The operator resolves the step budget from `execute(max_steps=...)` or `config.agent.max_steps`. `AgentState.max_steps` holds the live budget.

`PlanningOperator` starts `PlanningWorkflow` with `config.agent.planning_max_steps` as the total step budget, unless `execute(max_steps=...)` overrides it. Each planner turn and executor action costs one step. The planner observes at the start and after each subgoal without spending a step. A `Plan` replaces the remaining plan, then starts a child `AgentWorkflow` for its first subgoal. The child's budget is capped by `config.agent.planning.subgoal_steps` and the remaining total minus one. The last step is reserved for the planner to call `complete` and explain what blocked it if the goal is unfinished. Only a planner's `Finish` can complete the overall goal with a typed result.

`ctx.store` keeps the live goal, notes and action records. `DeviceResource`, `MemoryResource` and `ServicesResource` inject the phone, LlamaIndex memory and tool registry. Selectors take the registry as an argument and read the current snapshot from context. Declared sensitive arguments are masked in action records and selection events, not in the call sent to the device.

`StepSnapshot` adapts the core `DeviceSnapshot` for live prompts. `ActionRecord` holds the model-visible call and its `before` snapshot only for live memory and guards. `Acted.step_ids` links each tool to the core records produced by that call, including inherited child tasks but not unrelated concurrent tasks. A single call can link to zero or more rows, and planner offsets never change those links. Macros are built from the core action journal, not workflow state.

Observations and bookkeeping reads do not enter the journal. SMS and mailbox verification results are shown to the next selector only; they are not stored in action history. Typing a known code records a masked literal, not a replayable secret.

With an empty UI tree, observation captures a screenshot for live vision. With a tree, a vision selector can capture one on demand. Workflow screenshots are unredacted and kept in memory only, in the current observation and the latest action batch's before-screen. Only the core recorder can write screen files. The next observation sets `screen_changed` for the previous step using the core structural fingerprint, not pixels or visible text.

**Depends on / used by:** Workflows use services and core state. `AgentOperator` consumes the ordered event stream for reasoning evidence, logs and analytics.

The workflow stops on the step limit, on five consecutive calls with the same tool, target and other arguments (`stuck_fail_after`), or on five consecutive unchanged steps. Repeated targets count even when the screen changes. Unknown comparisons break the unchanged streak. The cycle guard counts returns to the same screen, not consecutive observations of it; discovering a new screen resets its streak. `RepeatBlock` warns at three matching calls (`repeat_warn_after`) without changing which tools can be selected.

The planning workflow reads each child's final state and result directly, then refreshes the full UI and device context. It replaces the previous planner screenshot with the current one. After a step limit, stuck guard, or cycle, the next planner request gets the reason, recent thoughts and action results, screen visits, and observed blockers.
