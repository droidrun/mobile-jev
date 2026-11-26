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
| Model | The profile selected in `mobilerun agent setup`. Without a profile, use the first provider with an environment or saved key: OpenAI (`gpt-5-mini`), Anthropic (`claude-sonnet-5`), Gemini (`gemini-3.8-flash`), then OpenRouter (`openai/gpt-5-mini`). No key: stop and ask for setup. |
| Reasoning effort | Planner: high; executor: low; agent: medium. A profile can override `reasoning_effort`. Applied only when the installed adapter identifies support offline; otherwise omitted. |
| Vision | On for OpenAI, Anthropic and Gemini; off for other providers unless configured. Unknown models may need `setup --provider … --vision` or `--no-vision`. Setup does not fetch model capabilities. |
| Step budget | Agent: 25 (`agent.max_steps`); planning: 40 (`agent.planning_max_steps`). |
| Timeout | 120 seconds per LLM call (`agent.decision.timeout`); no whole-run deadline unless `agent.workflow_timeout` or `execute(timeout=…)` is set. |

**Examples:**

```python
from mobilerun_agents.config.user_config import MobileConfig

config = MobileConfig()
config.agent.max_steps = 10
config.memory.token_limit = 8000
```

`agent.selector` names the active profile. `mobilerun agent setup` saves one LLM profile per configured provider. `setup --default PROVIDER` points `default_llm` and the active LLM role at that profile; a Jev primary stays active and can pin a different delegate. Keys stay out of `agents.yaml`. Old manager/executor configuration and subscription authentication are not accepted. MCP is unavailable in v3.

`checkpoints.folder` overrides the default `<Mobilerun config dir>/runs` parent directory. `ExecutionResult.trajectory_path` locates the run. Macros are saved separately as `<macro directory>/<run_id>/macro.json`, using Mobilerun's macro directory (`MOBILERUN_MACRO_DIRECTORY`, otherwise `<Mobilerun config dir>/macros`). Set `checkpoints.enabled: false` to skip recording both.

`setup --provider typesafe --default` activates the native Jev decision profile and points `agent.delegation.delegate` at the default LLM. Its confidence threshold follows `agent.delegation.min_confidence` (0.7 unless configured). `--jev-only` saves `jev_only: true` on the decision profile so reselecting it does not enable delegation. The planner always uses the default LLM when Jev is active.

**Depends on / used by:** Depends on shared base configuration and Pydantic. Asset lookup checks the working directory, then the add-on package. Providers, services, workflows and entry points consume these models.
