# agents/cli/agent/

Groups commands for running agents, replaying macros and configuring models.

Keep agent-specific command entry points here; core command registration belongs in `cli/app.py`.

## Running an agent

Local agent runs may incur LLM charges. Cloud devices and tasks cost money.

```sh
mobilerun agent run "Open settings" DEVICE --llm-base-url https://llm.example/v1
```

Use `--reasoning` to plan with the manager and executor, or `--no-reasoning` for the direct fast agent. Direct mode is the default. Use `--steps` to limit agent steps.

`--llm-base-url` sets the LLM provider's `base_url`; `--llm-api-base` sets its
`api_base`. The legacy `--base_url` (also `-u`) and `--api_base` flags remain
aliases. These LLM settings override the corresponding configured values. If
both settings are supplied, the LLM provider determines how they interact.
The separate `--base-url` flag selects the cloud API endpoint, not the LLM.

## Files

- `__init__.py` — Exports `agent`.
- `group.py` — `agent` — agent commands.
- `replay.py` — `replay` — replay an agent macro, resuming with the agent on divergence.
- `run.py` — `run` — run an agent from the command line.

Subpackages: `llm/`.

**Depends on / used by:** Uses agent runner and configuration, device selection and `macro.cli.replay`; registered by the CLI app.

## Cloud tasks

Use the task ID printed by `agent run` to reconnect after losing a terminal:

```sh
mobilerun agent task status TASK_ID --json
mobilerun agent task trajectory TASK_ID --out ./screenshots
mobilerun agent task stop TASK_ID --yes
```

Status and trajectory output contain the fields returned by the cloud API; the
status response may not include the original goal, device, or timestamps.
Trajectory screenshots are downloaded only when `--out` is supplied. Stopping
requires confirmation (or `--yes` in automation). Without a task ID, an
interactive terminal prompts for one. The cloud client has no task-list API.
