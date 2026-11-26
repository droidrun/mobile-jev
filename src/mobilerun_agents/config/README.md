# config/

**Purpose:** Define agent configuration, fresh environment snapshots and packaged asset paths.

**What belongs here / what doesn't:** Configuration data and prompt resolution belong here. Provider construction, workflow execution and authentication protocols do not.

**Contents:**

- `__init__.py` — Package marker.
- `README.md` — Module overview.
- `user_config.py` — Typed agent, model profile, tool, memory, analytics, tracing and checkpoint settings.
- `loader.py` — Read `agents.yaml` in the Mobilerun config directory, or an explicit path. Setup saves only non-default choices, including edits inside nested sections. Unset options follow the installed version's defaults. Writes are atomic and leave the base device config alone.
- `settings.py` — Runtime defaults, environment-backed settings and masked Langfuse keys.
- `paths.py` — Packaged asset and caller-relative path resolution.
- `prompts.py` — Resolves templates and renders them in an immutable sandbox. Missing variables raise an error.

## Agent defaults

| Setting | Default |
|---|---|
| Model | Use `default_setup`, or the first setup if no default was chosen. With no setups, use the first LLM profile. With no LLM profiles, use the first provider with an environment or saved key: OpenAI (`gpt-5-mini`), Anthropic (`claude-sonnet-5`), Gemini (`gemini-3.8-flash`), then OpenRouter (`openai/gpt-5-mini`). |
| Reasoning effort | Planner: high; executor: low; agent: medium. An LLM profile can override `reasoning` for supported OpenAI models. |
| Vision | On for OpenAI, Anthropic, Gemini and TypeSafe; off for other providers unless configured. Use `vision: off`, `on`, or `only` on a profile. |
| Step budget | Agent: 25 (`agent.max_steps`); planning: 40 (`agent.planning_max_steps`). |
| Timeout | 120 seconds per LLM call (`agent.decision.timeout`); no whole-run deadline unless `agent.workflow_timeout` or `execute(timeout=…)` is set. |

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

If `run` reports removed keys, set the old file aside and recreate profiles and setups with the new commands. There is no automatic migration.

`checkpoints.folder` overrides the default `<Mobilerun config dir>/runs` parent directory. `ExecutionResult.trajectory_path` locates the run. Macros are saved separately as `<macro directory>/<run_id>/macro.json`, using Mobilerun's macro directory (`MOBILERUN_MACRO_DIRECTORY`, otherwise `<Mobilerun config dir>/macros`). Set `checkpoints.enabled: false` to skip recording both.

**Depends on / used by:** Depends on shared base configuration and Pydantic. Asset lookup checks the working directory, then the add-on package. Providers, services, workflows and entry points consume these models.
