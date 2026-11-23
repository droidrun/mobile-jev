# mobilerun/agents/workflows/external

Discovers external Android agents in this directory. Each agent owns its LLM client and action loop and receives an already-connected raw ADB device.

## Quick example

Place this contract in `my_agent.py` in this directory, then implement the task. It is a template, not a working agent:

```python
async def run(device, instruction: str, config: dict, max_steps: int) -> dict:
    # Use the connected device and your own agent loop here.
    return {"success": False, "reason": "Not implemented", "steps": 0}
```

Set `external_agents.my_agent` in the config if the agent needs settings, then run `mobilerun agent run "Open Settings" DEVICE --agent my_agent`. The optional module-level `DEFAULT_CONFIG` supplies defaults. An external agent can incur provider charges. A custom context provider is supplied once at startup; there is no per-step hook.

| File | Purpose |
| --- | --- |
| `loader.py` | Discover, import and invoke external agents |
| `__init__.py` | Exports discovery and loading helpers |

See [agent workflows](../README.md) and [context providers](../../context/README.md).
