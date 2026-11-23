# mobilerun/agents

The optional advanced-agents add-on for Mobilerun runs LLM-backed device tasks and replays macros with an agent fallback. `AgentOperator` uses a fast single-LLM workflow by default. `DelegationOperator` uses a planning manager and action-selecting executor. `AgentMacroOperator` replays a macro with agent fallback.

## Quick example

```python
from mobilerun.agents import AgentMacroOperator, AgentOperator, DelegationOperator, MobileConfig

config = MobileConfig()
config.agent.reasoning = True
operator = AgentOperator(config=config)
planner = DelegationOperator()
macro = AgentMacroOperator()
```

Reasoning is configured on `config.agent.reasoning`. `DelegationOperator` enables reasoning without mutating the caller's configuration. Pass an operator to a Mobilerun device to run a task. Configure an LLM with `mobilerun agent llm configure`; inspect commands with `mobilerun agent --help`. Agent runs can incur provider and device or cloud charges.

The add-on is not yet published. See [installation and follow-ups](../../../README.md) for wheel installation and editable-install limitations.

## Contents

| Path | Purpose |
| --- | --- |
| [operators/](operators/README.md) | Agent, delegation and macro operators |
| [workflows/](workflows/README.md) | Agent execution workflows |
| [llm/](llm/README.md) | Provider configuration, model selection and inference |
| [tools/](tools/README.md) | Agent actions, context and tool registry |
| [trajectory/](trajectory/README.md) | Execution trajectories |
| [context/](context/README.md) | Context providers |
| [assets/](assets/README.md) | Prompts and app cards |
| [mcp/](mcp/README.md) | MCP integration |
| [telemetry/](telemetry/README.md) | Tracing and telemetry |
| [config/](config/README.md) | Agent configuration |
| [cli/](cli/README.md) | Agent CLI commands |
| [clients/](clients/README.md) | OAuth and Portal clients |
| [errors/](errors/README.md) | Agent errors |
| [logs/](logs/README.md) | Logging |
| [utils/](utils/README.md) | Shared agent utilities |
| `runner.py` | CLI and programmatic orchestration |

## Legacy import paths

Older paths, including `mobilerun.agents.operator`, `mobilerun.agents.agent`, and `mobilerun.agents.agent_macro`, remain available as compatibility shims. Prefer the root exports or the canonical modules above for new code.
