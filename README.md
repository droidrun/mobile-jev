# Mobilerun Agents

Run model-guided tasks and guarded macro replay on Mobilerun devices.

## Quick start

```sh
mobilerun agent providers add openai
mobilerun agent profiles add luna --provider openai --model gpt-6-luna
mobilerun agent configure fast --selector luna
mobilerun agent run "Open Settings and read the Android version" DEVICE --setup fast
```

Set the provider API key in the environment or save it with `agent providers add`. Profiles name reusable models, including Jev (TypeSafe). Setups choose a selector or a planner and executor, with optional LLM delegation for Jev. See the [agent CLI guide](src/mobilerun_agents/cli/agent/README.md). Agent runs may send screen contents to a model and incur provider charges; cloud-device charges follow the base CLI's `MOBILERUN_BILLING` setting without an additional run prompt.

## Operators

```python
from mobilerun_agents import AgentOperator, PlanningOperator, MacroAgentOperator

# Direct selection and execution
result = await AgentOperator().execute(device, "Read the battery level")

# Planner selects subgoals; executor handles each one (requires a default planning setup)
result = await PlanningOperator().execute(device, "Find and report the app version")

# Replay a macro, handing off to an agent if the screen diverges
result = await MacroAgentOperator("macros/my-task/v1.jsonl").execute(device, "Finish the task")
```

### External operators

To plug in your own operator, register it in your package's `pyproject.toml`:

```toml
[project.entry-points."mobilerun.agents"]
inventory = "my_package.plugin:InventoryOperator"
```

Export a no-argument subclass of `mobilerun.core.operator.Operator`, or an already configured instance. After installing the package, call `await ExternalOperator("inventory").execute(device, "Count items")` (import `ExternalOperator` from `mobilerun_agents`). It loads the named operator once per wrapper, forwards `output`, `max_steps` and `timeout`, and returns the operator's `ExecutionResult` unchanged. The plugin owns execution and any logging or tracing it needs.

To add context without replacing the operator, [write a memory block](src/mobilerun_agents/agent/memory/README.md#write-your-own-block). Register its factory in `mobilerun.memory_blocks`, or pass it to an operator's `memory_blocks` mapping, then enable it by name in `agents.yaml`.

## Defaults

| Setting | Default |
|---|---|
| Agent step budget | 25 (`agent.max_steps`) |
| Planning total budget | 40 steps shared by planner turns and executor actions (`agent.planning_max_steps`) |
| Planning subgoal budget | 10 (`agent.planning.subgoal_steps`) |
| Progress guards | Note after 3 identical calls on the same target in a row and stop after 5; note after 3 unchanged-screen steps and stop after 5; note after 3 returns to an already seen screen and stop after 4 |
| Vision | Profile/provider-dependent; override with `--vision off|on|only` |
| Screen settling | Pause 100 ms before observing after an action (`agent.settle_delay: 0.1`, in seconds; `0` disables it) |

## Saved data and diagnostics

The agent uses four workflow steps: observe → select → act → finish. Planning uses three: observe → plan → execute a child agent, with one planner memory per run and a fresh executor memory for each child. Blocks render from workflow state, not chat history. Screenshots used in prompts stay in memory; the agent never writes them to disk.

Each planner turn and executor action costs one step. Override the total with `PlanningOperator.execute(max_steps=...)`. The run stops when the budget runs out. If the planner takes the last step, it must call `complete` and report what blocked it if the goal is unfinished.

The device decides what is recorded. The agent reuses an existing recorder or wraps a plain device in `RecordingDevice(capture="actions")`. Set `record=False`, use `--no-record`, set `recording.enabled: false`, or set `MOBILERUN_AGENT_RECORDING_ENABLED=false` to opt out of agent recording (explicit operator settings take precedence).

| Recording | What core saves |
|---|---|
| Explicit `RecordingDevice(capture="full")` (recorder default) | Action, arguments, outcome and timing, plus before/after fingerprints, redacted UI and context JSON, and **unredacted** screenshots when available. |
| Agent auto-wraps a plain device with `capture="actions"` | Action, arguments (secrets and known OTPs masked), outcome and timing. No extra observations for recording. |
| `StealthDevice` anywhere in the chain or recording opt-out (e.g. `--no-record`) | Nothing during the agent run, even with an existing recorder. |

Core writes `run.json`, `steps.jsonl` and, for full recording, `screens/` under `<MOBILERUN_RUNS_DIR>/<run_id>/`. The agent adds only `agent.json` with reasoning linked to step IDs. Macros are built from `steps.jsonl` into `<MOBILERUN_MACRO_DIRECTORY>/<name>/vN.jsonl`. Actions-only macros replay without screen guards, but repair needs a recorded after fingerprint. Treat run records, screenshots and macros as private device data.

Use `mobilerun --debug agent run …` for selector choices, model-call diagnostics, run details and tracebacks. The agent uses standard logging; Mobilerun's `secret_scope` masks sensitive values in log records during sensitive calls. Persisted agent evidence is sanitized; run data and macros remain private device data.

See [demo instructions](tests/demo/README.md), [contributing](CONTRIBUTING.md), and [MIT license](LICENSE).
