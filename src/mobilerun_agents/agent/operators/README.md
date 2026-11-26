# Operators

**Purpose:** Execute a device instruction through a built-in workflow or an installed operator.

**What belongs here / what doesn't:** Built-in operators load configuration and consume events; workflow steps and tool dispatch stay in `workflows/`. External operators own their execution and don't use the built-in consumers.

**Contents:**
- `__init__.py` — public operator exports.
- `agent.py` — `AgentOperator` for direct action selection.
- `planning.py` — `PlanningOperator` starts `PlanningWorkflow` with a `Planner` and a separate executor `ActionSelector`. The default planner is `LlamaIndexPlanner` using the configured default LLM.
- `external.py` — `ExternalOperator` lazily loads exactly one named `mobilerun.agents` entry point exporting a base `Operator` instance or no-argument subclass. It forwards `execute` and returns the plugin result unchanged.
- `base.py` — built-in workflow execution and result mapping; never selects a workflow.
- `_run.py` — immutable run inputs shared by built-in operators.
- `_consumers.py` — routes the event stream to the console, checkpoints, analytics and tracing.

**Examples:** `await ExternalOperator("inventory").execute(device, "Count items")`.

**Depends on / used by:** Built on the base `Operator` contract and agent workflows; used by Python callers, macro recovery and the CLI.
