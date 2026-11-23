# Operators

Device operators provide LLM-backed operation, reasoning-enabled delegation, and
macro replay with agent fallback. Use these only with an appropriately configured
provider; executions may consume LLM tokens or device resources.

```python
from mobilerun.agents.operators import DelegationOperator

operator = DelegationOperator()  # Configuration loads on execution.
# await operator.execute(device, "Open Settings")
```

| Module | Responsibility |
| --- | --- |
| `agent.py` | Existing configurable agent execution. |
| `delegation.py` | Enable reasoning without changing caller configuration. |
| `agent_macro.py` | Replay a macro, falling back to an agent. |

See [workflows](../workflows/README.md) and [configuration](../config/README.md).
