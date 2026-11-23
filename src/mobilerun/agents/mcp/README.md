# mcp/

**Purpose:** Connect agent tools to MCP servers.

**What belongs here / what doesn't:** MCP configuration and adaptation belong here; shared device code must not depend on them.

**Contents:**

- `__init__.py` — Package exports.
- `adapter.py` — MCP tool adapter.
- `client.py` — MCP client.
- `config.py` — MCP configuration models.

## Quick example

Configure a local MCP server before starting the agent. Only run servers you trust: their tools execute with your permissions. The server executable must already be installed.

```python
from mobilerun.agents.mcp.config import MCPConfig, MCPServerConfig
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents import AgentOperator

config = MobileConfig()
config.mcp = MCPConfig(enabled=True, servers={
    "example": MCPServerConfig(command="my-mcp-server"),
})
operator = AgentOperator(config=config)
```

Pass `operator` to a device as in the [agent example](../README.md). `args`, `env`, `prefix`, `include_tools`, and `exclude_tools` can be set on each server. An agent task may incur LLM charges.
