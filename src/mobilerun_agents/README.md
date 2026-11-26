# Mobilerun Agents

**Purpose:** Run model-guided tasks and guarded macro replay on a Mobilerun device.

**What belongs here / what doesn't:** Public composition APIs and agent configuration live here. Device connections, extensions and the macro player belong to base.

**Contents:**

- `__init__.py` — public operator, workflow, selector and tool imports.
- [agent/](agent/README.md) — select and execute actions for a goal.
- [macro_agent/](macro_agent/README.md) — replay a macro with agent recovery.
- `cli/` — agent and macro-agent command groups.
- `config/`, `errors/`, `assets/` — settings, exceptions, prompts and app cards.

**Examples:** `from mobilerun_agents import AgentOperator`; pass `AgentOperator()` to your device code. See the [quickstart](../../README.md).

**Depends on / used by:** The Python API and CLI compose the operators. `macro_agent` depends on `agent`, never the reverse.
