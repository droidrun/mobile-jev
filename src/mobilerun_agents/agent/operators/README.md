# Operators

**Purpose:** Execute a device instruction through a built-in workflow or an installed operator.

**What belongs here / what doesn't:** Built-in operators load configuration and consume events; workflow steps and tool dispatch stay in `workflows/`. External operators own their execution and don't use the built-in consumers.

The handler returned by `workflow.run()` streams raw internal data, like `ctx.store`; built-in operator consumers mask console output, `context.json`, macro exports, traces, and `ExecutionResult`. If a masked value no longer fits the requested output schema, `ExecutionResult.output` is `None`.

**Contents:**
- `__init__.py` — public operator exports.
- `agent.py` — `AgentOperator` for direct action selection.
- `planning.py` — `PlanningOperator` starts `PlanningWorkflow` with a planner and executor, or uses the default planning setup when either is not supplied.
- `external.py` — `ExternalOperator` lazily loads exactly one named `mobilerun.agents` entry point exporting a base `Operator` instance or no-argument subclass. It forwards `execute` and returns the plugin result unchanged.
- `consumers.py` — builds per-run resources, consumes one event stream, and maps the workflow result to `ExecutionResult`. Closes the image store on success, failure, and cancellation.

**Examples:** `await ExternalOperator("inventory").execute(device, "Count items")`.

**Depends on / used by:** Built on the base `Operator` contract and agent workflows; used by Python callers, macro recovery and the CLI.
