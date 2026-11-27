# Operators

**Purpose:** Execute a device instruction through a built-in workflow or an installed operator.

**What belongs here / what doesn't:** Built-in operators load configuration and consume events; workflow steps and tool dispatch stay in `workflows/`. External operators own their execution and don't use the built-in consumers.

The workflow streams live data; the operator writes only `agent.json` into the recorded run. It stores sanitized thoughts, plans, memory, and tool choices linked to committed action IDs in the core `steps.jsonl`. Workflow contexts and prompt screenshots are not saved by the agent. If a successful workflow's output does not match the requested schema, `ExecutionResult.output` is `None`.

Direct and planning operators take `record: bool | None = None`. An explicit value overrides `MOBILERUN_AGENT_RECORDING_ENABLED`, then `agents.yaml` (`recording.enabled`, default `true`). Stealth and a parent recording veto take priority. A plain device is auto-wrapped with `capture="actions"`, which records actions without extra state or screenshot calls. The operator honors a caller's existing recorder and capture level without finalizing it or publishing a macro. It finalizes a run it created, publishing a versioned macro only after successful completion with replayable actions. The phone stays open for its caller.

**Contents:**
- `__init__.py` — public operator exports.
- `agent.py` — `AgentOperator` for direct action selection.
- `planning.py` — `PlanningOperator` starts `PlanningWorkflow` with a planner and executor, or uses the default planning setup when either is not supplied.
- `external.py` — `ExternalOperator` lazily loads exactly one named `mobilerun.agents` entry point exporting a base `Operator` instance or no-argument subclass. It forwards `execute` and returns the plugin result unchanged.
- `consumers.py` — builds per-run resources, consumes one event stream, and maps the workflow result to `ExecutionResult`.
- `recording.py` — resolves opt-in policy and recorder ownership.
- `evidence.py` — writes sanitized entries to `agent.json` atomically.

**Examples:** `await ExternalOperator("inventory").execute(device, "Count items")`.

**Depends on / used by:** Built on the base `Operator` contract and agent workflows; used by Python callers, macro recovery and the CLI.
