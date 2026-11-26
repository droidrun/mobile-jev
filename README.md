# Mobilerun Agents

Run model-guided tasks and guarded macro replay on Mobilerun devices.

## Quick start

```sh
mobilerun agent providers add openai
mobilerun agent profiles add luna --provider openai --model gpt-6-luna
mobilerun agent configure fast --selector luna
mobilerun agent run "Open Settings and read the Android version" DEVICE --setup fast
```

Set the provider API key in the environment or save it with `agent providers add`. Profiles name reusable models, including Jev (TypeSafe). Setups choose a selector or a planner and executor, with optional LLM delegation for Jev. See the [agent CLI guide](src/mobilerun_agents/cli/agent/README.md). Agent runs may send screen contents to a model and incur provider charges.

## Operators

```python
from mobilerun_agents import AgentOperator, PlanningOperator, MacroAgentOperator

# Direct selection and execution
result = await AgentOperator().execute(device, "Read the battery level")

# Planner selects subgoals; executor handles each one (requires a default planning setup)
result = await PlanningOperator().execute(device, "Find and report the app version")

# Replay a macro, handing off to an agent if the screen diverges
result = await MacroAgentOperator("macro.json").execute(device, "Finish the task")
```

### External operators

To plug in your own operator, register it in your package's `pyproject.toml`:

```toml
[project.entry-points."mobilerun.agents"]
inventory = "my_package.plugin:InventoryOperator"
```

Export a no-argument subclass of `mobilerun.core.operator.Operator`, or an already configured instance. After installing the package, call `await ExternalOperator("inventory").execute(device, "Count items")` (import `ExternalOperator` from `mobilerun_agents`). It loads the named operator once per wrapper, forwards `output`, `max_steps` and `timeout`, and returns the operator's `ExecutionResult` unchanged. The plugin owns execution and any logging or tracing it needs.

## Defaults

| Setting | Default |
|---|---|
| Agent step budget | 25 (`agent.max_steps`) |
| Planning total budget | 40 (`agent.planning_max_steps`) |
| Planning subgoal budget | 10 (`agent.planning.subgoal_steps`) |
| Planner turns | 10 (`agent.planning.max_planner_turns`) |
| Stuck guard | Exclude an unchanged-screen repeated action after 2 repeats; fail after 4 (`stuck_exclude_after`, `stuck_fail_after`) |
| Vision | Profile/provider-dependent; override with `--vision off|on|only` |

## Saved data and diagnostics

Run evidence is stored at `<MOBILERUN_USER_CONFIG_DIR>/runs/<id>/`; macros are stored in the base Mobilerun macro directory. Treat screenshots, UI state and macros as private device data.

Use `mobilerun agent run --debug` (or the root CLI `--debug`) for selector choices, model-call diagnostics, safe run details and tracebacks. Jev decision sidecars are saved with checkpoints; debug output points to their relative paths. Secret values are masked in outputs, not sent masked to the model. Typed text is not secret by default; vault credentials and explicitly sensitive parameters are protected.

See [demo instructions](tests/demo/README.md), [contributing](CONTRIBUTING.md), and [MIT license](LICENSE).
