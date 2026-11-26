# Mobilerun Agents

Give a [Mobilerun](https://github.com/droidrun/mobilerun) device a goal instead of a sequence of taps. `AgentOperator` observes the screen, asks a selector for the next action and executes it on the device. `MacroAgentOperator` replays a recording and can hand control to an agent if the screen diverges.

## Install and configure

Requires Python 3.11–3.13. Until the add-on is published, use the [installed-wheel setup](CONTRIBUTING.md). After publication: `pip install mobilerun-agents`. Provider adapters ship with the package; `langfuse` adds optional tracing.

```sh
mobilerun agent setup                  # choose provider and model; save a key if needed
mobilerun agent run "Open Settings and read the Android version"
mobilerun agent task --help            # status, stop and trajectory for cloud tasks
mobilerun macro-agent replay --help    # replay a run ID, macro folder, or macro.json
mobilerun agent logout openai          # remove the saved key, not the environment variable
```

Authentication is by API key, not subscription login. You can set the provider environment variable (such as `OPENAI_API_KEY`) instead of saving a key. MCP tools are temporarily unavailable. Live runs can incur provider and device charges.

## Agent defaults

| Setting | Default |
|---|---|
| Model | The profile selected in `mobilerun agent setup`. Without a profile, use the first provider with an environment or saved key: OpenAI (`gpt-5-mini`), Anthropic (`claude-sonnet-5`), Gemini (`gemini-3.8-flash`), then OpenRouter (`openai/gpt-5-mini`). No key: stop and ask for setup. |
| Reasoning effort | Planner: high; executor: low; agent: medium. A profile can override `reasoning_effort`. Applied only when the installed adapter identifies support offline; otherwise omitted. |
| Vision | On for OpenAI, Anthropic and Gemini; off for other providers unless configured. Unknown models may need `setup --provider … --vision` or `--no-vision`. Setup does not fetch model capabilities. |
| Step budget | Agent: 25 (`agent.max_steps`); planning: 40 (`agent.planning_max_steps`). |
| Timeout | 120 seconds per LLM call (`agent.decision.timeout`); no whole-run deadline unless `agent.workflow_timeout` or `execute(timeout=…)` is set. |

The run registry always includes `complete` and `wait()`; `wait()` uses the configured duration. Other callable device actions are exposed under their base names, such as `tap_element`, `input_text`, `press_button` and `start_app`. Coordinate tools (`tap`, `swipe`, `drag`) are available with vision. Jev gets finite choices: one `tap_element` per visible element, four directional `swipes`, and one `start_app` per installed app.

## How it plugs into mobilerun

Mobilerun discovers this add-on through Python entry points. `mobilerun.cli` registers the `agent` and `macro-agent` commands; `mobilerun.operators` registers `agent_macro` for replay handoffs. Run `mobilerun plugins list` in the environment where you installed `mobilerun-agents` to check discovery.

External workflow plugins register in the `mobilerun.agents` entry-point group. This is a metadata identifier, not the Python import name (`mobilerun_agents`).

## First run

With a connected Android device and a configured model:

```python
import asyncio

from mobilerun import AndroidDevice
from mobilerun_agents import AgentOperator

async def main():
    async with AndroidDevice("emulator-5554") as device:
        operator = AgentOperator()
        result = await operator.execute(device, "Read the Android version", max_steps=12)
        print(result.success, result.answer)

asyncio.run(main())
```

`AgentOperator(selector=...)` runs the observation/action loop. `PlanningOperator(planner=..., executor=...)` uses a text-only `Planner` and runs a separate `AgentWorkflow` for each subgoal using the executor selector. Omit either argument to use its configured default. On the CLI, use `mobilerun agent run --plan`. `ExternalOperator("plugin-name")` runs an installed workflow plugin. For typed results, pass a Pydantic model as `output=BatteryInfo` to `execute` and read `result.output`.

## Choose actions and add tools

Let Jev pick UI actions and an LLM handle explicit handoffs or low-confidence choices. Inside an async function:

```python
from mobilerun_agents import AgentOperator, DecisionSelector, DelegatingSelector, LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.decision_model import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.factory import build_llm

llm = await build_llm("openai", "gpt-4.1-mini")
selector = DelegatingSelector(DecisionSelector(TypeSafeDecisionModel()), LlamaIndexLLMSelector(llm))
operator = AgentOperator(selector=selector)
```

A custom selector reads the current observation and returns an action; it does not execute the action or write workflow state:

```python
from mobilerun_agents import ActionSelector, Run
from mobilerun_agents.agent.action_selector import ActionSelection
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall

class ReportScreen(ActionSelector):
    async def select(self, run: Run) -> ActionSelection:
        message = f"{run.state.goal}: {len(run.ui.elements)} visible elements"
        return ActionSelection(actions=[ToolCall("complete", {"success": True, "message": message})])
```

Add a run-local tool. The model sees `inventory_label` with no arguments; the operator supplies `run`.

```python
from mobilerun_agents import AgentOperator, Run, tool

@tool
async def inventory_label(run: Run) -> str:
    """Read the inventory label for this bench."""
    return "QA bench 3"

operator = AgentOperator(tools=[inventory_label])
```

`tools` adds to the UI actions and `complete`. A delegating selector also gets `delegate`. Every selector receives the pinned `run.ui` and the same base registry. It can fetch the full-resolution image with `await run.screenshot()`. Each model sizes its own image and returns coordinates in device pixels; tools must not convert them a second time. `run.state` is a detached snapshot. For capabilities shared outside the agent, bind a base `Extension` to the device yourself. For Jev, `DecisionSelector(model, vision="only", exclude={"press_button"})` excludes system-button choices and enables its located `tap(x, y)` choice. See the [demos](tests/demo/README.md).

## Record and replay

### Where things are saved

- **Run data:** `<Mobilerun config dir>/runs/<run_id>/` holds `checkpoints/`, `screenshots/`, and `ui_states/`. The config directory is platform-specific (on macOS, `~/Library/Application Support/droidrun`) and can be changed with `MOBILERUN_USER_CONFIG_DIR`. Set `checkpoints.folder` in `agents.yaml` to override just the run-data directory.
- **Macros:** `<Mobilerun config dir>/macros/<run_id>/macro.json` records executed device actions. Set `MOBILERUN_MACRO_DIRECTORY` to override the macro directory. `mobilerun macro list` shows these recordings.

The CLI prints the run folder on the final line and also shows the macro path with `--debug`. In Python, `result.trajectory_path` is the run folder, not the macro folder. Treat both as private device data. Set `checkpoints.enabled: false` to skip both recordings.

Replay with `mobilerun macro-agent replay <run_id>`, or pass the macro folder or its `macro.json` path.

```python
from mobilerun_agents import MacroAgentOperator

# Within your open device session, starting on the recorded screen:
result = await MacroAgentOperator(macro_file="macro.json").execute(device, "Read the Android version")
```

See the [standalone demos](tests/demo/README.md) for complete examples. [MIT license](LICENSE).
