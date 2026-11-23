# Context providers

Provide guidance to all agent types.

## Quick example

```python
from mobilerun.agents import AgentOperator, ContextProvider, StepContext
from mobilerun.core.devices import AndroidDevice


class SettingsTips(ContextProvider):
    async def get_context(self, context: StepContext) -> str:
        if context.package_name == "com.android.settings":
            return "Tip: the search bar is at the top of Settings."
        return ""


device = AndroidDevice("emulator-5554", operator=AgentOperator(context_provider=SettingsTips()))
```

The built-in reasoning manager calls the provider once per step after reading
the screen, and shares its result with the executor. A custom provider takes
precedence over `agent.app_cards` settings, including `enabled: false`. Fast
mode calls a custom provider after reading each screen. External agents receive
custom context once at startup with step 0 and no screen; their interface has
no per-step hook. Without a custom provider, the reasoning manager's existing
app-card configuration selects the built-in provider.

| Path | Purpose |
| --- | --- |
| `context_provider.py` | Context interface and per-step data |
| [providers/](providers/README.md) | Local, server and composite app-card backends |

See [app cards](../assets/app_cards/README.md) and the [agent quick start](../README.md).
