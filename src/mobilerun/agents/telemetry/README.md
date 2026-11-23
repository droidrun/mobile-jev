# mobilerun/agents/telemetry

Agent telemetry and optional tracing integrations. Traces can contain sensitive task and screen information; use a trusted backend.

## Quick example

Enable Phoenix tracing for a local agent run (the run may incur LLM charges):

```sh
export PHOENIX_URL="http://localhost:6006"
export PHOENIX_PROJECT_NAME="mobilerun"
mobilerun agent run "Open Settings" emulator-5554 --tracing
```

To use Langfuse instead, set `tracing.enabled: true` and `tracing.provider: langfuse` in the agent config, and supply `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` through the environment. Langfuse requires its optional integration dependencies. `tracing.langfuse_screenshots` controls screenshot capture. Phoenix and Langfuse are alternative tracing providers; telemetry is a separate setting.

| File | Purpose |
| --- | --- |
| `phoenix.py` | Phoenix instrumentation |
| `langfuse_processor.py` | Langfuse span processing |
| `events.py`, `tracker.py` | Telemetry events and tracking |

See [configuration](../config/README.md).
