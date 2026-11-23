# mobilerun/agents/workflows

Implements the built-in mobile agent loop and its supporting workflows. Most callers should use `AgentOperator` through a device instead of constructing workflow internals.

## Quick example

Select the direct fast agent (default) or the reasoning manager and executor. This configures an operator without starting a task; running one may incur LLM charges.

```python
from mobilerun.agents import AgentOperator
from mobilerun.agents.config.user_config import MobileConfig

config = MobileConfig()
config.agent.reasoning = True  # False selects the fast agent
operator = AgentOperator(config=config)
```

Pass `operator` to a device as shown in the [agent quick start](../README.md). The CLI equivalents are `--reasoning` and `--no-reasoning`.

| Path | Purpose |
| --- | --- |
| [droid/](droid/README.md) | `MobileAgent` coordination and state |
| [fast_agent/](fast_agent/README.md) | Direct tool-calling workflow |
| [manager/](manager/README.md), [executor/](executor/README.md) | Planning and action execution |
| [external/](external/README.md) | External agent discovery and contract |
| [oneflows/](oneflows/README.md) | Focused app and output workflows |
| [../llm/](../llm/README.md), [../utils/](../utils/README.md) | LLM metadata, loading and helpers |
| [../trajectory/](../trajectory/README.md), [../tools/](../tools/README.md) | Recording and tool registration |
| `options.py` | Shared workflow keyword options |
| [common/](common/README.md) | Shared workflow events |
