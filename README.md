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
| Planning total budget | 40 steps shared by planner turns and executor actions (`agent.planning_max_steps`) |
| Planning subgoal budget | 10 (`agent.planning.subgoal_steps`) |
| Progress guards | Note after 3 identical calls on the same target in a row and stop after 5; note after 3 unchanged-screen steps and stop after 5; note after 3 returns to an already seen screen and stop after 4 |
| Vision | Profile/provider-dependent; override with `--vision off|on|only` |
| Screen settling | Pause 100 ms before observing after an action (`agent.settle_delay: 0.1`, in seconds; `0` disables it) |

## Saved data and diagnostics

The agent uses four workflow steps: observe → select → act → finish. Planning uses three: observe → plan → execute a child agent, with one planner memory per run and a fresh executor memory for each child. Blocks render from workflow state, not chat history. The device, LlamaIndex memory, tool registry and image store are live resources, not checkpoint data.

Each planner turn and executor action costs one step. Override the total with `PlanningOperator.execute(max_steps=...)`. The run stops when the budget runs out. If the planner takes the last step, it must call `complete` and report what blocked it if the goal is unfinished.

Run data is organized by run ID. Empty directory overrides use the defaults.

| Data | Environment override | Default | Layout |
|---|---|---|---|
| Native workflow checkpoint | `MOBILERUN_AGENT_RUNS_DIR` | `<MOBILERUN_USER_CONFIG_DIR>/runs` | `<runs>/<run_id>/context.json` |
| Screenshots | `MOBILERUN_AGENT_IMAGES_DIR` | `<MOBILERUN_USER_CONFIG_DIR>/images` | `<images>/<run_id>/NNNN.png` |

With checkpoints enabled, the operator saves the native LlamaIndex context after each step and at completion. It exports a macro from the saved action history into the base Mobilerun macro directory. Step snapshots point to image files; checkpoints don't embed screenshot bytes. When checkpoints are disabled, the image directory is deleted at the end. Treat checkpoints, screenshots and macros as private device data.

Use `mobilerun --debug agent run …` for selector choices, model-call diagnostics, safe run details and tracebacks. Secret values are masked in outputs, not in model inputs. Typed text is not secret by default; vault credentials and explicitly sensitive parameters are protected.

See [demo instructions](tests/demo/README.md), [contributing](CONTRIBUTING.md), and [MIT license](LICENSE).
