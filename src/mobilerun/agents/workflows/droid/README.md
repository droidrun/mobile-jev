# agents/workflows/droid/

Coordinates the mobile agent, its execution state and workflow events.

Keep agent orchestration and its events here; planner and executor implementations live in sibling packages.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `MobileAgent`, `MobileAgentState`. |
| `droid_agent.py` | `MobileAgent` — mobileagent - a wrapper class that coordinates the planning and execution of tasks. |
| `events.py` | `FastAgentExecuteEvent`, `FastAgentResultEvent`, `ManagerInputEvent`, `ManagerPlanEvent`, `ExecutorInputEvent`, `ExecutorResultEvent`, `ExternalUserMessageAppliedEvent`, `ExternalUserMessageDroppedEvent`, `FinalizeEvent`, `ResultEvent` — mobileagent coordination events. |
| `state.py` | `QueuedUserMessage`, `MobileAgentState`. |


See [agent overview](../README.md).
