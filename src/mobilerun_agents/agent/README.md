# Running an agent

**Purpose:** Combine device actions and run-local tools for one instruction, run a workflow and return an `ExecutionResult`.

**What belongs here / what doesn't:** The operator creates run resources and owns the stream consumers. The workflow, not the operator, decides when to observe or act.

**Contents:**

- `__init__.py` — package marker; import the public API from `mobilerun_agents`.
- `events.py` — shared raw events written by workflows and selectors and consumed at the output boundary.
- `operators/` — agent and external plugin operators.
- `workflows/` — observation, action and completion steps.
- `action_selector/`, `tools/` — choose an action and dispatch it through the run registry.
- `planner/` — the `Planner` contract and `LlamaIndexPlanner` implementation.
- `memory/` — live prompt context.
- `analytics/`, `logs/`, `auth/` — telemetry, console output and API keys.

**Examples:** `await AgentOperator().execute(device, "Read the battery level", max_steps=10)`.

The agent only writes `agent.json` alongside an active core recording; prompt screenshots stay in memory. A plain device is wrapped for actions-only capture by default. Pass an explicit full recorder for before/after evidence, including unredacted screenshots. See [recording options](../../../README.md#saved-data-and-diagnostics).

**Depends on / used by:** The operator composes workflows and services atop the base device. Python callers, CLI and macro recovery use it.
