# mobilerun/agents/cli

Agent commands are registered in the shared Mobilerun CLI. Agent runs may incur LLM charges; cloud tasks and cloud devices also cost money.

## Quick example

```sh
mobilerun agent llm configure
mobilerun agent run "Open Settings" emulator-5554 --steps 10
mobilerun agent replay login.json emulator-5554 --dry-run
```

Use `mobilerun agent run --help` for device selection and agent options. See [agent commands](agent/README.md) and [LLM setup](agent/llm/README.md).

| Path | Purpose |
| --- | --- |
| `agent/` | Run, replay, task and LLM commands |
| `base/` | OAuth terminal helpers |
