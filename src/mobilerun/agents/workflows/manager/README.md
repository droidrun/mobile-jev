# agents/workflows/manager/

Plans mobile tasks and validates manager responses.

Keep planner workflows, prompts and events here; device action execution belongs in `workflows/executor/`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `ManagerAgent`, `StatelessManagerAgent`, `ManagerInputEvent`, `ManagerPlanEvent`, `ManagerContextEvent`, `ManagerResponseEvent`, `ManagerPlanDetailsEvent`, `parse_manager_response`. |
| `events.py` | `ManagerContextEvent`, `ManagerResponseEvent`, `ManagerPlanDetailsEvent` — events for the manageragent workflow. |
| `manager_agent.py` | `ManagerAgent` — manageragent - planning and reasoning workflow. |
| `prompts.py` | `ManagerResponseValidation`, `ManagerResponseValidationError`, `parse_manager_response`, `validate_manager_response` — prompts for the manageragent. |
| `stateless_manager_agent.py` | `StatelessManagerAgent` — statelessmanageragent - stateless planning agent that rebuilds context each turn. |


See [agent overview](../README.md).
