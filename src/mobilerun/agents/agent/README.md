# mobilerun/agents/agent (legacy paths)

Compatibility shims only. Each module here aliases its canonical module, so old imports keep working and return the same objects.

```python
from mobilerun.agents.agent.droid import MobileAgent      # legacy
from mobilerun.agents.workflows.droid import MobileAgent  # canonical, same class
```

New code should import the canonical paths:

| Legacy | Canonical |
| --- | --- |
| `agent.droid`, `agent.fast_agent`, `agent.manager`, `agent.executor`, `agent.oneflows`, `agent.external`, `agent.common` | `workflows.*` |
| `agent.workflow_options` | `workflows.options` |
| `agent.providers`, `agent.usage`, `agent.utils.{inference,llm_loader,llm_picker}`, `agent.utils.oauth` | `llm.*` |
| `agent.tool_registry`, `agent.action_context`, `agent.action_result`, `agent.utils.{actions,signatures}` | `tools.*` |
| `agent.trajectory` | `trajectory` |
| `agent.utils.trajectory` | `trajectory.trajectory` |
| `agent.utils.errors` | `logs.error_descriptions` |
| `agent.utils.tracing_setup` | `telemetry.setup` |

The full map is in `scripts/module_map.json`.
