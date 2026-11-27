# config/

**Purpose:** Define agent configuration, fresh environment snapshots and packaged asset paths.

**What belongs here / what doesn't:** Configuration data and prompt resolution belong here. Provider construction, workflow execution and authentication protocols do not.

**Contents:**

- `__init__.py` — Package marker.
- `README.md` — Module overview.
- `user_config.py` — Typed agent, model profile, tool, memory, analytics, tracing and recording settings.
- `loader.py` — Read `agents.yaml` in the Mobilerun config directory, or an explicit path. Saves store only non-default choices. `ConfigLoader.update(edit)` holds a file lock from load through atomic replacement and leaves the base device config alone.
- `settings.py` — Runtime defaults, environment-backed settings and masked Langfuse keys.
- `paths.py` — Packaged asset and caller-relative path resolution.
- `prompts.py` — Resolves templates and renders them in an immutable sandbox. Missing variables raise an error.

## Agent defaults

| Setting | Default |
|---|---|
| Model | Use `default_setup` when set. Otherwise use a straight agent with the first LLM profile; if there are no LLM profiles, use the first provider with an environment or saved key: OpenAI (`gpt-5-mini`), Anthropic (`claude-sonnet-5`), Gemini (`gemini-3.8-flash`), then OpenRouter (`openai/gpt-5-mini`). |
| Reasoning effort | Planner: high; executor: low; agent: medium. An LLM profile can override `reasoning` for supported OpenAI models. |
| Vision | On for OpenAI, Anthropic, Gemini and TypeSafe; off for other providers unless configured. Use `vision: off`, `on`, or `only` on a profile. |
| Step budget | Agent: 25 (`agent.max_steps`); planning: 40 shared by planner turns and executor actions (`agent.planning_max_steps`). |
| Timeout | 120 seconds per LLM call (`agent.decision.timeout`); no whole-run deadline unless `agent.workflow_timeout` or `execute(timeout=…)` is set. |
| Screen settling | Pause 100 ms (`agent.settle_delay: 0.1`, in seconds) before observing after an action. Set the delay to `0` to disable it. |

**Examples:**

```python
from mobilerun_agents.config.user_config import MobileConfig

config = MobileConfig()
config.agent.max_steps = 10
config.memory.token_limit = 8000
```

`agents.yaml` stores choices, not credentials. Use `agent providers add` for keys. Profile base URLs cannot include userinfo, a query string, or a fragment. There is no free-form profile options field. A setup names either a selector or both a planner and executor. Only a decision model selector or executor can have a delegate. The planner and delegate always name LLM profiles.

```yaml
profiles:
  luna: {provider: openai, model: gpt-6-luna, reasoning: low}
  sol: {provider: openai, model: gpt-6.1-sol, reasoning: high}
  jev-on: {provider: typesafe, vision: on, min_confidence: 0.7}
setups:
  fast: {selector: luna}
  deep: {planner: sol, executor: jev-on, delegate: luna}
default_setup: fast
```

### Memory

Use `memory.executor.blocks` and `memory.planner.blocks` to select and order the
blocks for each role. Put block-specific settings under `options.<block_name>`:

```yaml
memory:
  executor:
    options:
      last_step: {screenshot: true}
      history: {steps: 5}
  planner:
    options:
      subgoal_history: {subgoals: 5}
```

These values are the defaults. Move `memory.executor.last_step_screenshot`,
`memory.executor.history`, and `memory.planner.history` to the corresponding
paths above; the old keys are no longer accepted. For other removed keys,
recreate profiles and setups with the new commands. There is no automatic migration.

### Recording

```yaml
recording:
  enabled: true
```

Agent recording is on by default. `MOBILERUN_AGENT_RECORDING_ENABLED` overrides the YAML value; the operator's `record=` argument (or `agent run --no-record`) takes precedence over both. Stealth vetoes recording regardless of settings.

Recorded runs are saved under `<MOBILERUN_RUNS_DIR>/<run_id>/`, defaulting to `<MOBILERUN_USER_CONFIG_DIR>/runs`. `ExecutionResult.trajectory_path` locates the recording. The agent auto-wraps plain devices for actions-only capture and adds only `agent.json`; its prompt screenshots stay in memory. An explicit full recorder also saves screen evidence, including unredacted screenshots. Macros use `MOBILERUN_MACRO_DIRECTORY`.

**Depends on / used by:** Depends on shared base configuration and Pydantic. Relative prompt paths resolve only inside the add-on package; set a prompt field to an absolute path to override a bundled prompt. Providers, services, workflows and entry points consume these models.
