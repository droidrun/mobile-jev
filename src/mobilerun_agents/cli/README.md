# cli/

**Purpose:** Register agent commands in the shared Mobilerun CLI.

**What belongs here / what doesn't:** Agent command adapters belong here; device operations and shared CLI rendering belong in base Mobilerun.

**Contents:**

- `__init__.py` — Package marker.
- `agent/` — Model setup, saved-key logout, agent runs and cloud tasks.
- `macro_agent/` — Macro replay with agent handoff.

**Examples:**

```sh
mobilerun agent setup
mobilerun agent run "Open Settings" emulator-5554 --steps 10
mobilerun macro list
mobilerun macro-agent replay <run_id> emulator-5554
```

Agent runs may incur LLM charges. Cloud tasks and devices cost money.

**Depends on / used by:** Uses agent workflows through the runner and base CLI primitives. The shared CLI registers these commands.
