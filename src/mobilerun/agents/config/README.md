# mobilerun/agents/config

Loads agent configuration and defines typed settings for agent workflows, LLM profiles and integrations.

## Quick example

```python
from mobilerun.agents import AgentOperator
from mobilerun.agents.config.user_config import MobileConfig

config = MobileConfig()
config.agent.reasoning = True
config.agent.max_steps = 10
operator = AgentOperator(config=config)
```

Pass the operator to a device as in the [agent quick start](../README.md). Running an LLM task may incur charges. Use `mobilerun agent llm configure` to set up provider profiles.

`get_agent_settings()` and `get_agent_secrets()` return fresh environment snapshots. Secret fields are masked. OAuth credential paths can be overridden with `OPENAI_OAUTH_CREDENTIAL_PATH`, `ANTHROPIC_OAUTH_CREDENTIAL_PATH`, `GEMINI_OAUTH_CREDENTIAL_PATH`, or `GROK_OAUTH_CREDENTIAL_PATH`. See [tracing](../telemetry/README.md) for Phoenix and Langfuse settings.

| File | Purpose |
| --- | --- |
| `user_config.py` | Agent, profile, tool and credential configuration |
| `loader.py` | Load configuration |
| `settings.py`, `env.py` | Runtime settings and masked agent secrets |
