# agents/workflows/executor/

Executes planned device actions and emits executor events.

Keep executor workflow, prompts and events here; planning belongs in `workflows/manager/`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `ExecutorAgent`, `ExecutorInputEvent`, `ExecutorResultEvent`, `ExecutorContextEvent`, `ExecutorResponseEvent`, `ExecutorActionEvent`, `ExecutorActionResultEvent`. |
| `events.py` | `ExecutorContextEvent`, `ExecutorResponseEvent`, `ExecutorActionEvent`, `ExecutorActionResultEvent` — events for the executoragent workflow. |
| `executor_agent.py` | `ExecutorAgent` — executoragent - action execution workflow. |
| `prompts.py` | `parse_executor_response` — prompts for the executoragent. |


See [agent overview](../README.md).
