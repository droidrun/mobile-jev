# agents/workflows/oneflows/

Runs focused agent workflows for opening apps and producing structured output.

Keep single-purpose workflows here; full mobile-agent orchestration belongs in `workflows/droid/`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker; no public names defined. |
| `app_starter_workflow.py` | `AppStarter`, `main` — simple workflow to open an app based on a description. |
| `structured_output_agent.py` | `StructuredOutputAgent` — structuredoutputagent - extract structured data from final answers. |


See [agent overview](../README.md).
