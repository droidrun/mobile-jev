# Operators

**Purpose:** Choose a workflow and execute a device instruction.

**What belongs here / what doesn't:** Operators load configuration and consume events; workflow steps and tool dispatch stay in `workflows/`.

**Contents:**
- `__init__.py` — public operator exports.
- `agent.py` — `AgentOperator` for direct action selection.
- `planning.py` — `PlanningOperator` starts `PlanningWorkflow` with a `Planner` and a separate executor `ActionSelector`. The default planner is `LlamaIndexPlanner` using the configured default LLM.
- `external.py` — `ExternalOperator` for an installed plugin.
- `base.py` — execution and result mapping; never selects a workflow.
- `_run.py` — immutable run inputs shared by the operators.
- `_consumers.py` — routes the event stream to the console, checkpoints, analytics and tracing.

**Examples:** `await ExternalOperator("inventory").execute(device, "Count items")`.

**Depends on / used by:** Built on the base `Operator` contract and agent workflows; used by Python callers, macro recovery and the CLI.
