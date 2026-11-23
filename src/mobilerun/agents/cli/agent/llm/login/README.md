# agents/cli/agent/llm/login/

Exposes provider-specific OAuth login commands.

Keep login prompts and output here; OAuth protocol and storage live in `agents/clients/oauth/`.

## Quick example

```sh
mobilerun agent llm login anthropic
```

This starts an interactive provider login; it does not run an agent task. Use `mobilerun agent llm login --help` for other providers.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `login`. |
| `anthropic.py` | `configure_anthropic`. |
| `gemini.py` | `configure_gemini`. |
| `group.py` | `login`. |
| `openai.py` | `configure_openai`. |
| `xai.py` | `configure_xai`. |


See [agent commands](../README.md).
